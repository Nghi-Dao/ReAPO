import argparse
import gc
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
import warnings
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import ray
import torch
import wandb
from hydra import compose, initialize_config_module
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import ResourcePoolManager
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.utils import Role, need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.device import auto_set_device
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    fsdp2_load_full_state_dict,
    fsdp_version,
    offload_fsdp_model_to_cpu,
)
from verl.workers.engine_workers import ActorRolloutRefWorker


PROJECT_ROOT = os.path.abspath(os.getcwd())
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
src_dir = os.path.join(PROJECT_ROOT, "src")
if os.path.isdir(src_dir) and src_dir not in sys.path:
    sys.path.insert(0, src_dir)


DEFAULT_DATASET_PATHS = [
    "~/data/minerva/test.parquet",
    "~/data/aime24/test.parquet",
    "~/data/aime25/test.parquet",
    "~/data/beyondaime/test.parquet",
    "~/data/hmmt25/test.parquet",
    "~/data/gpqa-diamond/test.parquet",
    "~/data/math-500/test.parquet",
]

DATASET_CONFIGS = {
    "aime24": {"step_samples": 256, "main_samples": 4096},
    "aime25": {"step_samples": 256, "main_samples": 4096},
    "beyondaime": {"step_samples": 256, "main_samples": 4096},
    "hmmt25": {"step_samples": 256, "main_samples": 4096},
    "gpqa-diamond": {"step_samples": 64, "main_samples": 1024},
    "math-500": {"step_samples": 64, "main_samples": 1024},
    "minerva": {"step_samples": 64, "main_samples": 1024},
}


# Match the validation decoding configuration used by commands/qwen1.7b.sh.
# Phase 1 gets its attempt count from --max_attempts. The deep pass@k phase is
# intentionally pinned to one attempt so its samples are independent rollouts.
EVAL_TEMPERATURE = 0.6
EVAL_TOP_P = 0.95
EVAL_TOP_K = -1
EVAL_DO_SAMPLE = True
PASS_AT_K_MAX_ATTEMPTS = 1
EVAL_RETRY_PROMPT = "Your answer is wrong. Reflect on your previous answer and try something else."
DEFAULT_TARGET_STEP = 1040
MAX_VALIDATION_RESPONSES_PER_BATCH = 8192
# This controls the small logger table only. Every generation is separately
# written as JSONL beneath eval_outputs via trainer.validation_data_dir.
LOG_VAL_GENERATIONS = 128
RESUME_MANIFEST_NAME = "_eval_complete.json"
RESUME_SCHEMA_VERSION = 1


def _validate_max_attempts(max_attempts: int) -> int:
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")
    return max_attempts


def _jsonable(value):
    """Convert NumPy values and nested containers into JSON-safe objects."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _atomic_write_json(path: str, payload: dict) -> None:
    """Write a JSON file atomically so a crash cannot create a valid-looking partial manifest."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = f"{path}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        with open(temporary_path, "w", encoding="utf-8") as output_file:
            json.dump(_jsonable(payload), output_file, indent=2, sort_keys=True)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _file_fingerprint(path: str) -> dict:
    resolved_path = os.path.abspath(path)
    stat_result = os.stat(resolved_path)
    return {
        "path": resolved_path,
        "size": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
    }


def _checkpoint_fingerprint(checkpoint_dir: str) -> dict:
    """Cheaply detect a checkpoint being replaced while retaining the same step number."""
    resolved_dir = os.path.abspath(checkpoint_dir)
    files = []
    for filename in sorted(os.listdir(resolved_dir)):
        path = os.path.join(resolved_dir, filename)
        if not os.path.isfile(path):
            continue
        stat_result = os.stat(path)
        files.append(
            {
                "name": filename,
                "size": stat_result.st_size,
                "mtime_ns": stat_result.st_mtime_ns,
            }
        )
    return {"path": resolved_dir, "files": files}


def _evaluation_signature(
    phase: str,
    step: int,
    checkpoint_dir: str,
    dataset_names: List[str],
    validation_paths: List[str],
    n_samples: int,
    max_attempts: int,
) -> dict:
    return {
        "schema_version": RESUME_SCHEMA_VERSION,
        "phase": phase,
        "step": int(step),
        "checkpoint": _checkpoint_fingerprint(checkpoint_dir),
        "datasets": [
            {"name": name, **_file_fingerprint(path)}
            for name, path in zip(dataset_names, validation_paths, strict=True)
        ],
        "n_samples": int(n_samples),
        "max_attempts": int(max_attempts),
        "temperature": EVAL_TEMPERATURE,
        "top_p": EVAL_TOP_P,
        "top_k": EVAL_TOP_K,
        "do_sample": EVAL_DO_SAMPLE,
        "retry_prompt": EVAL_RETRY_PROMPT,
        "max_prompt_length": 1024,
        "max_response_length": 4096,
        "reward_function": "src/decay_reward.py:compute_decay_reward",
    }


def _saved_generation_files(output_dir: str) -> List[dict]:
    saved_files = []
    for path in sorted(glob.glob(os.path.join(output_dir, "*.jsonl"))):
        if os.path.isfile(path):
            saved_files.append(
                {
                    "name": os.path.basename(path),
                    "size": os.path.getsize(path),
                }
            )
    return saved_files


def _load_completed_eval(output_dir: str, expected_signature: dict) -> Optional[dict]:
    """Return saved metrics only when the manifest and generation dump are complete and compatible."""
    manifest_path = os.path.join(output_dir, RESUME_MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        return None

    try:
        with open(manifest_path, encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[!] Ignoring unreadable resume manifest {manifest_path}: {exc}")
        return None

    if manifest.get("status") != "complete":
        return None
    if manifest.get("signature") != expected_signature:
        print(f"[!] Saved evaluation configuration changed; rerunning {output_dir}")
        return None

    metrics = manifest.get("metrics")
    saved_files = manifest.get("generation_files")
    if not isinstance(metrics, dict) or not isinstance(saved_files, list) or not saved_files:
        print(f"[!] Resume manifest is incomplete; rerunning {output_dir}")
        return None

    for saved_file in saved_files:
        path = os.path.join(output_dir, saved_file.get("name", ""))
        expected_size = saved_file.get("size")
        if (
            not os.path.isfile(path)
            or not isinstance(expected_size, int)
            or expected_size <= 0
            or os.path.getsize(path) != expected_size
        ):
            print(f"[!] Saved generation dump is missing or changed; rerunning {output_dir}")
            return None

    return metrics


def _wait_for_generation_dumps(trainer: RayPPOTrainer) -> None:
    """Wait for VERL's asynchronous JSONL writer before marking an evaluation complete."""
    futures = list(getattr(trainer, "_dump_futures", []))
    for future in futures:
        future.result()
    if hasattr(trainer, "_dump_futures"):
        trainer._dump_futures = [future for future in trainer._dump_futures if not future.done()]


def _save_completed_eval(
    trainer: RayPPOTrainer,
    output_dir: str,
    signature: dict,
    metrics: dict,
) -> None:
    _wait_for_generation_dumps(trainer)
    generation_files = _saved_generation_files(output_dir)
    if not generation_files or any(item["size"] <= 0 for item in generation_files):
        raise RuntimeError(
            f"Validation returned metrics but no complete JSONL dump was saved in {output_dir}"
        )

    _atomic_write_json(
        os.path.join(output_dir, RESUME_MANIFEST_NAME),
        {
            "status": "complete",
            "signature": signature,
            "metrics": metrics,
            "generation_files": generation_files,
            "completed_unix_time": time.time(),
        },
    )


def _execute_or_resume_eval_pass(
    trainer: RayPPOTrainer,
    val_file_list: List[str],
    dataset_names: List[str],
    n_samples: int,
    output_dir: str,
    phase: str,
    checkpoint: dict,
    max_attempts: int,
    resume: bool,
) -> Tuple[dict, bool]:
    signature = _evaluation_signature(
        phase=phase,
        step=checkpoint["step"],
        checkpoint_dir=checkpoint["merged_dir"],
        dataset_names=dataset_names,
        validation_paths=val_file_list,
        n_samples=n_samples,
        max_attempts=max_attempts,
    )

    if resume:
        saved_metrics = _load_completed_eval(output_dir, signature)
        if saved_metrics is not None:
            print(
                f"[RESUME] Reusing completed {phase} evaluation at step "
                f"{checkpoint['step']} for {dataset_names} (n={n_samples})"
            )
            return saved_metrics, True

    metrics = _execute_eval_pass(trainer, val_file_list, n_samples, output_dir)
    _save_completed_eval(trainer, output_dir, signature, metrics)
    print(
        f"[+] Saved resumable {phase} result at step {checkpoint['step']} "
        f"for {dataset_names} (n={n_samples})"
    )
    return metrics, False


def _set_eval_agent_environment(max_attempts: int) -> None:
    """Set agent-loop behavior before creating a trainer or Ray worker."""
    os.environ["MAX_ATTEMPTS"] = str(_validate_max_attempts(max_attempts))
    os.environ["VERL_RETRY_PROMPT"] = EVAL_RETRY_PROMPT


class EvalHotSwapActorRolloutRefWorker(ActorRolloutRefWorker):
    """Actor/rollout worker with an eval-only merged-HF hot-load endpoint.

    Keeping this subclass in the evaluator avoids modifying the installed VERL
    package. Ray creates these workers instead of the stock class.
    """

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    @torch.no_grad()
    def load_hf_checkpoint(self, local_path: str) -> None:
        assert "actor" in self.role, "load_hf_checkpoint only supports the actor role"
        engine = self.actor.engine

        if engine.engine_config.strategy != "fsdp2" or fsdp_version(engine.module) != 2:
            raise RuntimeError("Merged Hugging Face hot-loading requires actor.strategy=fsdp2.")
        if engine._is_lora:
            raise NotImplementedError(
                "Merged Hugging Face hot-loading currently supports full-model checkpoints only, not PEFT/LoRA."
            )

        rank = torch.distributed.get_rank()
        source_model = None
        full_state = {}
        load_error = None

        if rank == 0:
            try:
                if not os.path.isdir(local_path):
                    raise FileNotFoundError(f"Merged Hugging Face checkpoint not found: {local_path}")

                from verl.utils.model import get_hf_auto_model_class
                from verl.utils.torch_dtypes import PrecisionType

                torch_dtype = engine.engine_config.model_dtype
                if torch_dtype is None:
                    torch_dtype = torch.float32 if not engine.engine_config.forward_only else torch.bfloat16
                torch_dtype = PrecisionType.to_dtype(torch_dtype)

                auto_class = get_hf_auto_model_class(hf_config=engine.model_config.hf_config)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    source_model = auto_class.from_pretrained(
                        pretrained_model_name_or_path=local_path,
                        torch_dtype=torch_dtype,
                        config=engine.model_config.hf_config,
                        trust_remote_code=engine.model_config.trust_remote_code,
                        low_cpu_mem_usage=True,
                    )

                for attribute in getattr(source_model, "_verl_strip_modules", []):
                    if hasattr(source_model, attribute):
                        delattr(source_model, attribute)

                source_model.to(dtype=torch_dtype)
                full_state = source_model.state_dict()
            except Exception as exc:
                load_error = f"{type(exc).__name__}: {exc}"

        # If rank 0 cannot read the checkpoint, notify every rank before any
        # state-dict collective starts. This avoids leaving other ranks hung.
        status = [load_error]
        torch.distributed.broadcast_object_list(status, src=0)
        if status[0] is not None:
            raise RuntimeError(f"Failed to load merged checkpoint {local_path}: {status[0]}")

        cpu_offload_policy = None
        if getattr(engine, "_uses_fsdp2_cpu_offload_policy", False):
            cpu_offload_policy = CPUOffloadPolicy(pin_memory=True)

        # VERL broadcasts rank 0's full state and reshards it according to the
        # live four-GPU device mesh. The old eight-GPU shard count is irrelevant.
        fsdp2_load_full_state_dict(
            engine.module,
            full_state,
            engine.device_mesh,
            cpu_offload_policy,
        )
        torch.distributed.barrier()

        del full_state
        if source_model is not None:
            del source_model
        gc.collect()

        if engine._is_offload_param:
            offload_fsdp_model_to_cpu(engine.module)

        self.base_sync_done = False


def parse_dataset_info(path_str: str) -> Tuple[str, str]:
    resolved_path = os.path.abspath(os.path.expanduser(path_str))

    # Accept either a parquet file or a dataset directory. This lets the
    # default "~/data/beyondaime" entry resolve to its test split without
    # requiring the caller to know the exact filename.
    if os.path.isdir(resolved_path):
        dataset_name = os.path.basename(os.path.normpath(resolved_path))
        preferred_files = [
            os.path.join(resolved_path, filename)
            for filename in ("test.parquet", "validation.parquet", "val.parquet")
        ]
        selected_file = next((path for path in preferred_files if os.path.isfile(path)), None)

        if selected_file is None:
            parquet_files = sorted(glob.glob(os.path.join(resolved_path, "*.parquet")))
            if len(parquet_files) == 1:
                selected_file = parquet_files[0]
            elif len(parquet_files) > 1:
                raise ValueError(
                    f"Dataset directory {resolved_path} contains multiple parquet files and no "
                    "test.parquet/validation.parquet/val.parquet. Pass the intended file explicitly."
                )
            else:
                raise FileNotFoundError(
                    f"Dataset directory {resolved_path} does not contain a parquet file."
                )

        if selected_file is not None:
            print(f"[+] Resolved dataset directory {resolved_path} -> {selected_file}")
            return dataset_name, selected_file

    dir_name = os.path.basename(os.path.dirname(resolved_path))
    file_name = os.path.splitext(os.path.basename(resolved_path))[0]
    dataset_name = dir_name if dir_name and dir_name not in {"data", "test", "val"} else file_name
    return dataset_name, resolved_path


def get_sorted_checkpoints(base_dir: str) -> List[Tuple[int, str]]:
    checkpoints = []
    for checkpoint_dir in glob.glob(os.path.join(base_dir, "global_step_*")):
        match = re.search(r"global_step_(\d+)$", checkpoint_dir)
        if match:
            checkpoints.append((int(match.group(1)), checkpoint_dir))
    return sorted(checkpoints, key=lambda item: item[0])


def _find_embedding_shape(model_dir: str):
    """Read the embedding shape from safetensors metadata without loading weights."""
    try:
        from safetensors import safe_open
    except ImportError:
        return None, None

    index_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors.index.json")))
    if index_files:
        with open(index_files[0], encoding="utf-8") as index_file:
            weight_map = json.load(index_file).get("weight_map", {})
        keys = list(weight_map)
        preferred = [
            key
            for key in keys
            if key == "model.embed_tokens.weight" or key.endswith(".embed_tokens.weight")
        ]
        if not preferred:
            preferred = [key for key in keys if key.endswith(".wte.weight")]
        if not preferred:
            return None, None
        key = preferred[0]
        tensor_file = os.path.join(model_dir, weight_map[key])
        with safe_open(tensor_file, framework="pt", device="cpu") as checkpoint:
            return key, tuple(checkpoint.get_slice(key).get_shape())

    for tensor_file in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(tensor_file, framework="pt", device="cpu") as checkpoint:
            keys = list(checkpoint.keys())
            preferred = [
                key
                for key in keys
                if key == "model.embed_tokens.weight" or key.endswith(".embed_tokens.weight")
            ]
            if not preferred:
                preferred = [key for key in keys if key.endswith(".wte.weight")]
            if preferred:
                key = preferred[0]
                return key, tuple(checkpoint.get_slice(key).get_shape())
    return None, None


def _read_model_dimensions(model_dir: str, fallback_dir: str):
    for config_dir in (model_dir, fallback_dir):
        config_path = os.path.join(config_dir, "config.json")
        if not os.path.isfile(config_path):
            continue
        with open(config_path, encoding="utf-8") as config_file:
            config = json.load(config_file)
        text_config = config.get("text_config", config)
        vocab_size = text_config.get("vocab_size", config.get("vocab_size"))
        hidden_size = text_config.get("hidden_size", config.get("hidden_size"))
        if vocab_size is not None and hidden_size is not None:
            return int(vocab_size), int(hidden_size)
    return None, None


def _merged_checkpoint_status(model_dir: str, actor_dir: str) -> dict:
    files = os.listdir(model_dir) if os.path.isdir(model_dir) else []
    has_weights = any(filename.endswith((".safetensors", ".bin")) for filename in files)
    has_tokenizer = any(
        filename.startswith(("tokenizer", "vocab", "merges"))
        or filename.endswith((".model", ".tiktoken"))
        for filename in files
    )
    has_config = os.path.isfile(os.path.join(model_dir, "config.json"))
    shape_error = None

    if has_weights and any(filename.endswith(".safetensors") for filename in files):
        key, actual_shape = _find_embedding_shape(model_dir)
        vocab_size, hidden_size = _read_model_dimensions(model_dir, actor_dir)
        if vocab_size is not None and hidden_size is not None:
            expected_shape = (vocab_size, hidden_size)
            if key is None:
                shape_error = "the safetensors checkpoint has no token-embedding tensor"
            elif actual_shape != expected_shape:
                shape_error = (
                    f"{key} has shape {actual_shape}, expected {expected_shape}; "
                    "the cached files are still FSDP shards, not a merged Hugging Face checkpoint"
                )

    return {
        "has_weights": has_weights,
        "has_tokenizer": has_tokenizer,
        "has_config": has_config,
        "shape_error": shape_error,
        "valid": has_weights and has_tokenizer and has_config and shape_error is None,
    }


def _select_clean_merge_target(actor_dir: str, requested_dir: str):
    """Reuse a valid cache or choose a non-destructive repair directory."""
    status = _merged_checkpoint_status(requested_dir, actor_dir)
    if status["valid"]:
        print(f"[+] Verified merged checkpoint at {requested_dir}. Skipping merge.")
        return requested_dir, status

    if status["shape_error"] is None:
        return requested_dir, status

    print(f"[!] Invalid merged-checkpoint cache at {requested_dir}: {status['shape_error']}")
    print("[+] Keeping the invalid cache untouched and rebuilding from the native checkpoint.")

    suffix = 1
    while True:
        suffix_text = "_full" if suffix == 1 else f"_full_{suffix}"
        candidate = requested_dir + suffix_text
        candidate_status = _merged_checkpoint_status(candidate, actor_dir)
        if candidate_status["valid"]:
            print(f"[+] Reusing verified repaired checkpoint at {candidate}")
            return candidate, candidate_status
        if not candidate_status["has_weights"]:
            return candidate, candidate_status
        if candidate_status["shape_error"] is None:
            # Its weights look complete; only metadata may need to be copied.
            return candidate, candidate_status
        suffix += 1


def merge_verl_checkpoint(actor_dir: str, requested_dir: str) -> str:
    target_dir, status = _select_clean_merge_target(actor_dir, requested_dir)
    has_weights = status["has_weights"]

    if has_weights:
        print(f"[+] Complete weights found; patching metadata in {target_dir} if needed.")
    else:
        print(f"[+] Merging FSDP checkpoint: {actor_dir} -> {target_dir}")
    os.makedirs(target_dir, exist_ok=True)

    if not has_weights:
        patch_script = """
import numpy as np

from verl.model_merger.fsdp_model_merger import FSDPModelMerger

def _extract_mesh_without_process_group(self, state_dict, world_size):
    # Old checkpoints can contain a pickled DeviceMesh whose process groups do
    # not exist in this standalone merger process. Accessing weight.device_mesh.mesh
    # therefore fails. The tensor placements survive unpickling, and this run used
    # a one-dimensional FSDP mesh, so reconstruct only the shape information that
    # FSDPModelMerger needs to decide how many rank files to load.
    for weight in state_dict.values():
        if hasattr(weight, '_local_tensor') and hasattr(weight, 'placements'):
            placements = tuple(weight.placements)
            if len(placements) != 1:
                raise RuntimeError(
                    f'Expected a one-dimensional FSDP checkpoint, got placements={placements!r}'
                )
            return np.arange(world_size, dtype=np.int64), ('fsdp',)

    # A non-DTensor checkpoint already contains full tensors on rank zero.
    return np.array([world_size], dtype=np.int64), ('fsdp',)

FSDPModelMerger._extract_device_mesh_info = _extract_mesh_without_process_group

from verl.model_merger.__main__ import main
main()
"""
        command = [
            sys.executable,
            "-c",
            patch_script,
            "merge",
            "--backend",
            "fsdp",
            "--local_dir",
            actor_dir,
            "--target_dir",
            target_dir,
        ]
        subprocess.run(command, check=True)
        print(f"[+] Successfully merged weights into {target_dir}")

    copied = 0
    for filename in os.listdir(actor_dir):
        if filename.endswith((".json", ".txt", ".model", ".tiktoken")):
            source_path = os.path.join(actor_dir, filename)
            destination_path = os.path.join(target_dir, filename)
            if not os.path.exists(destination_path):
                shutil.copy2(source_path, destination_path)
                copied += 1
    if copied:
        print(f"[+] Copied {copied} tokenizer/config files to merged directory.")

    final_status = _merged_checkpoint_status(target_dir, actor_dir)
    if final_status["shape_error"] is not None:
        raise RuntimeError(
            f"Model merger produced an invalid checkpoint at {target_dir}: "
            f"{final_status['shape_error']}"
        )
    if not final_status["has_weights"] or not final_status["has_config"]:
        raise RuntimeError(f"Merged checkpoint is incomplete: {target_dir}")
    if not final_status["has_tokenizer"]:
        raise RuntimeError(f"Merged checkpoint has no tokenizer files: {target_dir}")

    print(f"[+] Verified full Hugging Face checkpoint: {target_dir}")
    return target_dir


class _EvalTaskSession:
    """Own the placement groups created by one evaluator trainer boot."""

    def __init__(self) -> None:
        self.resource_pool_manager = None

    def cleanup(self, timeout_seconds: int = 120) -> None:
        if self.resource_pool_manager is None:
            return

        from ray.util.placement_group import placement_group_table, remove_placement_group

        placement_groups = []
        for resource_pool in self.resource_pool_manager.resource_pool_dict.values():
            placement_groups.extend(resource_pool.pgs or [])
        if not placement_groups:
            return

        print(f"[+] Releasing {len(placement_groups)} Ray placement group(s)...")
        for placement_group in placement_groups:
            remove_placement_group(placement_group)

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            states = []
            for placement_group in placement_groups:
                try:
                    states.append(placement_group_table(placement_group).get("state", "REMOVED"))
                except Exception:
                    states.append("REMOVED")
            if all(state == "REMOVED" for state in states):
                print("[+] Ray placement groups released")
                return
            time.sleep(1)
        print("[!] Timed out waiting for Ray placement-group removal")


def _build_single_trainer(pipeline_spec: dict, session: _EvalTaskSession) -> RayPPOTrainer:
    num_gpus = pipeline_spec["num_gpus"]
    initial_model_dir = pipeline_spec["checkpoint_specs"][0]["merged_dir"]
    sample_val_files = pipeline_spec["sample_val_files"]
    max_attempts = _validate_max_attempts(int(pipeline_spec["max_attempts"]))

    _set_eval_agent_environment(max_attempts)
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"

    home_dir = os.path.expanduser("~")
    overrides = [
        "algorithm.adv_estimator=multi_attempt_grpo",
        f"data.train_files={home_dir}/data/polaris/train.parquet",
        f"data.val_files={sample_val_files}",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        "data.val_batch_size=256",
        "data.max_prompt_length=1024",
        "data.max_response_length=4096",
        f"reward.custom_reward_function.path={os.path.abspath('./src/decay_reward.py')}",
        "reward.custom_reward_function.name=compute_decay_reward",
        "reward.reward_manager.name=naive",
        "reward.num_workers=32",
        f"actor_rollout_ref.model.path={initial_model_dir}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.strategy=fsdp2",
        "actor_rollout_ref.ref.strategy=fsdp2",
        "actor_rollout_ref.actor.ppo_mini_batch_size=256",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.6",
        f"actor_rollout_ref.rollout.val_kwargs.temperature={EVAL_TEMPERATURE}",
        f"actor_rollout_ref.rollout.val_kwargs.top_p={EVAL_TOP_P}",
        f"actor_rollout_ref.rollout.val_kwargs.top_k={EVAL_TOP_K}",
        f"actor_rollout_ref.rollout.val_kwargs.do_sample={str(EVAL_DO_SAMPLE).lower()}",
        "++actor_rollout_ref.max_model_len=5120",
        "actor_rollout_ref.rollout.multi_turn.enable=True",
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path={os.path.abspath('./src/custom_agents.yaml')}",
        "++actor_rollout_ref.rollout.agent.default_agent_loop=multi_attempt",
        "trainer.val_only=True",
        f"trainer.n_gpus_per_node={num_gpus}",
        "trainer.nnodes=1",
        f"trainer.log_val_generations={LOG_VAL_GENERATIONS}",
        "trainer.logger=[console]",
    ]

    try:
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    except Exception:
        pass

    with initialize_config_module(config_module="verl.trainer.config", version_base=None):
        config = compose(config_name="ppo_trainer", overrides=overrides)

    expected_reward_path = os.path.abspath("./src/decay_reward.py")
    resolved_reward_path = os.path.abspath(str(config.reward.custom_reward_function.path))
    resolved_reward_name = str(config.reward.custom_reward_function.name)
    if resolved_reward_path != expected_reward_path or resolved_reward_name != "compute_decay_reward":
        raise RuntimeError(
            "The async reward loop is not configured with the custom scorer: "
            f"resolved_path={resolved_reward_path!r}, resolved_name={resolved_reward_name!r}"
        )

    val_kwargs = config.actor_rollout_ref.rollout.val_kwargs
    resolved_sampling = {
        "temperature": float(val_kwargs.temperature),
        "top_p": float(val_kwargs.top_p),
        "top_k": int(val_kwargs.top_k),
        "do_sample": bool(val_kwargs.do_sample),
    }
    expected_sampling = {
        "temperature": EVAL_TEMPERATURE,
        "top_p": EVAL_TOP_P,
        "top_k": EVAL_TOP_K,
        "do_sample": EVAL_DO_SAMPLE,
    }
    if resolved_sampling != expected_sampling:
        raise RuntimeError(
            "Resolved validation sampling does not match the requested settings: "
            f"resolved={resolved_sampling}, expected={expected_sampling}"
        )
    if os.environ.get("MAX_ATTEMPTS") != str(max_attempts):
        raise RuntimeError(
            "MAX_ATTEMPTS was not set correctly before trainer initialization: "
            f"expected {max_attempts}, got {os.environ.get('MAX_ATTEMPTS')!r}"
        )

    print(
        "[+] Resolved validation settings: "
        f"temperature={resolved_sampling['temperature']}, "
        f"top_p={resolved_sampling['top_p']}, "
        f"top_k={resolved_sampling['top_k']}, "
        f"do_sample={resolved_sampling['do_sample']}, "
        f"max_attempts={max_attempts}, "
        f"reward_function={resolved_reward_name}"
    )

    auto_set_device(config)
    use_critic = need_critic(config)
    use_ref = need_reference_policy(config)
    validate_config(config=config, use_reference_policy=use_ref, use_critic=use_critic)

    tokenizer = AutoTokenizer.from_pretrained(initial_model_dir, trust_remote_code=True)
    role_worker_mapping = {Role.ActorRollout: ray.remote(EvalHotSwapActorRolloutRefWorker)}

    pool_id = f"single_trainer_eval_{uuid.uuid4().hex[:8]}"
    resource_pool_spec = {pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
    mapping = {Role.ActorRollout: pool_id}

    if use_critic:
        try:
            from verl.workers.engine_workers import CriticWorker
        except ImportError:
            from verl.workers.fsdp_workers import CriticWorker
        role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
        mapping[Role.Critic] = pool_id

    if use_ref:
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
        mapping[Role.RefPolicy] = pool_id

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    session.resource_pool_manager = resource_pool_manager

    print(f"\n[+] Booting one trainer on {num_gpus} GPUs from {initial_model_dir}")
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
    )
    trainer.init_workers()
    # Track which weights are actually resident in the actor separately from
    # the checkpoint label used by the evaluation loop. This distinction is
    # essential when resume skips the first several checkpoints: init_workers()
    # still constructs the actor from checkpoint_specs[0], not from the first
    # unfinished checkpoint.
    trainer._eval_loaded_checkpoint_step = pipeline_spec["checkpoint_specs"][0]["step"]
    trainer._eval_rollout_synced = False
    print("[+] Trainer and vLLM servers initialized; they will be reused for every checkpoint")
    return trainer


def _activate_checkpoint(trainer: RayPPOTrainer, checkpoint: dict, current_step: Optional[int]) -> int:
    step = checkpoint["step"]
    merged_dir = checkpoint["merged_dir"]
    loaded_step = getattr(trainer, "_eval_loaded_checkpoint_step", None)
    rollout_synced = bool(getattr(trainer, "_eval_rollout_synced", False))

    # `current_step` is only the loop's view of state. `loaded_step` records the
    # weights actually loaded in the actor and is authoritative after resume.
    if current_step == step and loaded_step == step and rollout_synced:
        return step

    started_at = time.monotonic()
    if loaded_step == step:
        # The actor was constructed from this checkpoint. init_workers() leaves
        # the initial rollout asleep, so only actor -> vLLM synchronization is
        # required.
        print(f"[+] Activating initial merged checkpoint at step {step}")
    else:
        print(f"\n[+] Hot-loading merged checkpoint step {step}: {merged_dir}")
        if rollout_synced:
            trainer.checkpoint_manager.sleep_replicas()
        trainer.actor_rollout_wg.load_hf_checkpoint(merged_dir)

    trainer.global_steps = step
    trainer.checkpoint_manager.update_weights(step)
    trainer._eval_loaded_checkpoint_step = step
    trainer._eval_rollout_synced = True
    elapsed = time.monotonic() - started_at
    print(f"[+] Step {step} is active in actor + vLLM ({elapsed:.1f}s; trainer was not restarted)")
    return step


def _execute_eval_pass(
    trainer: RayPPOTrainer,
    val_file_list: List[str],
    n_samples: int,
    output_dir: str,
) -> dict:
    val_kwargs = trainer.config.actor_rollout_ref.rollout.val_kwargs
    val_kwargs.n = n_samples
    val_kwargs.do_sample = EVAL_DO_SAMPLE
    val_kwargs.temperature = EVAL_TEMPERATURE
    val_kwargs.top_p = EVAL_TOP_P
    val_kwargs.top_k = EVAL_TOP_K
    trainer.config.trainer.validation_data_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(
        "[+] Starting validation pass: "
        f"datasets={val_file_list}, n={n_samples}, "
        f"temperature={val_kwargs.temperature}, top_p={val_kwargs.top_p}, "
        f"top_k={val_kwargs.top_k}, do_sample={val_kwargs.do_sample}, "
        f"max_attempts={os.environ.get('MAX_ATTEMPTS')}"
    )

    validation_dataset = _build_validation_dataset(trainer, val_file_list)
    configured_prompt_batch_size = getattr(trainer.config.data, "val_batch_size", None) or 256
    validation_batch_size = min(
        configured_prompt_batch_size,
        max(1, MAX_VALIDATION_RESPONSES_PER_BATCH // n_samples),
    )
    print(
        "[+] Validation batching: "
        f"prompt_batch_size={validation_batch_size}, "
        f"responses_per_full_batch={validation_batch_size * n_samples}"
    )
    trainer.val_dataloader = DataLoader(
        validation_dataset,
        batch_size=validation_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )
    return trainer._validate()


def _build_validation_dataset(
    trainer: RayPPOTrainer,
    val_file_list: List[str],
) -> RLHFDataset:
    dataset_config = trainer.config.data
    dataset_config.prompt_key = trainer.config.data.prompt_key
    dataset_config.max_prompt_length = trainer.config.data.max_prompt_length
    dataset_config.filter_overlong_prompts = trainer.config.data.filter_overlong_prompts
    dataset_config.truncation = trainer.config.data.truncation
    return RLHFDataset(
        data_files=val_file_list,
        tokenizer=trainer.tokenizer,
        config=dataset_config,
    )


def _recover_metrics_from_existing_jsonl(
    trainer: RayPPOTrainer,
    val_file_list: List[str],
    n_samples: int,
    output_dir: str,
    step: int,
) -> Optional[dict]:
    """Recover VERL metrics from a complete pre-manifest generation dump.

    This supports evaluations produced by an older copy of this script. A dump
    is trusted only if every row parses and its row count exactly matches the
    current filtered validation dataset times eval.n. Otherwise the pass is
    rerun normally.
    """
    jsonl_files = sorted(glob.glob(os.path.join(output_dir, "*.jsonl")))
    expected_path = os.path.join(output_dir, f"{step}.jsonl")
    if os.path.isfile(expected_path):
        jsonl_files = [expected_path]
    if len(jsonl_files) != 1 or os.path.getsize(jsonl_files[0]) <= 0:
        return None

    validation_dataset = _build_validation_dataset(trainer, val_file_list)
    expected_rows = len(validation_dataset) * n_samples
    data_sources = []
    sample_uids = []
    for prompt_index in range(len(validation_dataset)):
        item = validation_dataset[prompt_index]
        data_source = item.get("data_source", "unknown")
        uid = item.get("uid", f"resume-prompt-{prompt_index}")
        if isinstance(data_source, np.generic):
            data_source = data_source.item()
        if isinstance(uid, np.generic):
            uid = uid.item()
        data_sources.extend([data_source] * n_samples)
        sample_uids.extend([uid] * n_samples)

    base_keys = {"input", "output", "gts", "score", "step"}
    reward_extra_infos: Dict[str, List] = {}
    row_count = 0
    try:
        with open(jsonl_files[0], encoding="utf-8") as generation_file:
            for line in generation_file:
                if not line.strip():
                    continue
                record = json.loads(line)
                if int(record.get("step", step)) != step:
                    print(
                        f"[!] Existing dump has the wrong checkpoint step; "
                        f"rerunning {output_dir}"
                    )
                    return None

                if row_count == 0:
                    extra_keys = sorted(set(record) - base_keys)
                    reward_extra_infos = {key: [] for key in extra_keys}
                    if "reward" not in reward_extra_infos:
                        reward_extra_infos["reward"] = []

                for key in reward_extra_infos:
                    if key == "reward" and key not in record:
                        reward_extra_infos[key].append(record.get("score"))
                    elif key in record:
                        reward_extra_infos[key].append(record[key])
                    else:
                        print(
                            f"[!] Existing dump has inconsistent fields; rerunning {output_dir}"
                        )
                        return None
                row_count += 1
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"[!] Existing generation dump is incomplete or invalid; rerunning {output_dir}: {exc}")
        return None

    if row_count != expected_rows:
        print(
            f"[!] Existing generation dump has {row_count} rows, expected "
            f"{expected_rows}; rerunning {output_dir}"
        )
        return None
    if len(data_sources) != row_count or len(sample_uids) != row_count:
        return None

    try:
        metrics = trainer._val_metrics_update(
            np.asarray(data_sources, dtype=object),
            sample_uids,
            reward_extra_infos,
            [],
        )
    except Exception as exc:
        print(
            f"[!] Could not reconstruct metrics from {jsonl_files[0]}; "
            f"rerunning evaluation: {type(exc).__name__}: {exc}"
        )
        return None

    print(
        f"[RESUME] Reconstructed metrics from existing complete dump "
        f"{jsonl_files[0]} ({row_count} rows)"
    )
    return metrics


def _select_target_step(phase1_results: List[Tuple[int, dict]], requested_step: Optional[int]) -> int:
    if requested_step is not None:
        return requested_step

    step_scores = {}
    for step, metrics in phase1_results:
        accuracy_values = [
            value
            for key, value in metrics.items()
            if any(term in key for term in ("acc", "score", "reward"))
            and isinstance(value, (int, float, np.number))
        ]
        if accuracy_values:
            step_scores[step] = float(np.mean(accuracy_values))
    return max(step_scores, key=step_scores.get) if step_scores else phase1_results[-1][0]


@ray.remote(num_cpus=0)
class EvalEventReporter:
    """Small mailbox used to stream completed metrics back to the W&B driver."""

    def __init__(self) -> None:
        self.events = []

    def publish(self, event: dict) -> None:
        self.events.append(event)

    def drain(self) -> List[dict]:
        events = self.events
        self.events = []
        return events


def _publish_eval_event(reporter, event: dict) -> None:
    if reporter is not None:
        # Wait only for the tiny mailbox write. This guarantees that the driver
        # can log the event even if a later checkpoint crashes.
        ray.get(reporter.publish.remote(event))


def _run_phase1(
    trainer: RayPPOTrainer,
    pipeline_spec: dict,
) -> Tuple[List[Tuple[int, dict]], int, Optional[int]]:
    checkpoints = pipeline_spec["checkpoint_specs"]
    datasets = pipeline_spec["datasets"]
    default_step_samples = pipeline_spec["default_step_samples"]
    eval_outputs_folder = pipeline_spec["eval_outputs_folder"]
    event_reporter = pipeline_spec.get("event_reporter")
    resume = bool(pipeline_spec.get("resume", True))
    max_attempts = int(pipeline_spec["max_attempts"])

    current_step = None
    phase1_results = []
    print(
        "\n=== PHASE 1: Merged Checkpoint Evaluation "
        f"(max_attempts={pipeline_spec['max_attempts']}) ==="
    )

    for checkpoint_index, checkpoint in enumerate(checkpoints, start=1):
        step = checkpoint["step"]
        checkpoint_started_at = time.monotonic()
        _publish_eval_event(
            event_reporter,
            {
                "type": "phase1_started",
                "step": step,
                "checkpoint_index": checkpoint_index,
                "checkpoint_total": len(checkpoints),
            },
        )
        step_metrics = {"global_step": step}

        groups = {}
        for dataset_name, validation_path in datasets.items():
            if os.path.exists(validation_path):
                samples = DATASET_CONFIGS.get(dataset_name, {}).get(
                    "step_samples", default_step_samples
                )
                groups.setdefault(samples, []).append((dataset_name, validation_path))

        pending_groups = []
        for samples, dataset_group in groups.items():
            dataset_names = [item[0] for item in dataset_group]
            validation_paths = [item[1] for item in dataset_group]
            output_dir = os.path.join(
                eval_outputs_folder,
                "phase1",
                f"step_{step}",
                f"n_{samples}_{'_'.join(dataset_names)}",
            )
            signature = _evaluation_signature(
                phase="phase1",
                step=step,
                checkpoint_dir=checkpoint["merged_dir"],
                dataset_names=dataset_names,
                validation_paths=validation_paths,
                n_samples=samples,
                max_attempts=max_attempts,
            )
            manifest_exists = os.path.isfile(os.path.join(output_dir, RESUME_MANIFEST_NAME))
            saved_metrics = _load_completed_eval(output_dir, signature) if resume else None
            # Only use raw-JSONL recovery for outputs made before manifests
            # existed. If a manifest exists but is incompatible, rerun instead
            # of bypassing its configuration/checkpoint safety checks.
            if saved_metrics is None and resume and not manifest_exists:
                saved_metrics = _recover_metrics_from_existing_jsonl(
                    trainer=trainer,
                    val_file_list=validation_paths,
                    n_samples=samples,
                    output_dir=output_dir,
                    step=step,
                )
                if saved_metrics is not None:
                    _save_completed_eval(trainer, output_dir, signature, saved_metrics)
            if saved_metrics is not None:
                print(
                    f"[RESUME] Reusing completed phase1 evaluation at step {step} "
                    f"for {dataset_names} (n={samples})"
                )
                step_metrics.update(saved_metrics)
            else:
                pending_groups.append((samples, dataset_names, validation_paths, output_dir))

        if pending_groups:
            current_step = _activate_checkpoint(trainer, checkpoint, current_step)
            for samples, dataset_names, validation_paths, output_dir in pending_groups:
                metrics, _ = _execute_or_resume_eval_pass(
                    trainer=trainer,
                    val_file_list=validation_paths,
                    dataset_names=dataset_names,
                    n_samples=samples,
                    output_dir=output_dir,
                    phase="phase1",
                    checkpoint=checkpoint,
                    max_attempts=max_attempts,
                    resume=False,
                )
                step_metrics.update(metrics)
        else:
            print(f"[RESUME] Checkpoint step {step} is fully complete; no checkpoint load needed")

        phase1_results.append((step, step_metrics))
        _publish_eval_event(
            event_reporter,
            {
                "type": "phase1_completed",
                "step": step,
                "checkpoint_index": checkpoint_index,
                "checkpoint_total": len(checkpoints),
                "elapsed_seconds": time.monotonic() - checkpoint_started_at,
                "metrics": step_metrics,
            },
        )

    target_step = _select_target_step(phase1_results, pipeline_spec["target_step"])
    _publish_eval_event(
        event_reporter,
        {
            "type": "target_selected",
            "step": target_step,
        },
    )
    return phase1_results, target_step, current_step


def _run_phase2(
    trainer: RayPPOTrainer,
    pipeline_spec: dict,
    target_step: int,
    current_step: Optional[int],
) -> List[dict]:
    checkpoints = pipeline_spec["checkpoint_specs"]
    datasets = pipeline_spec["datasets"]
    default_main_samples = pipeline_spec["default_main_samples"]
    eval_outputs_folder = pipeline_spec["eval_outputs_folder"]
    event_reporter = pipeline_spec.get("event_reporter")
    resume = bool(pipeline_spec.get("resume", True))

    target_checkpoint = next(checkpoint for checkpoint in checkpoints if checkpoint["step"] == target_step)
    print(
        f"\n=== PHASE 2: Deep Pass@k Evaluation for Target Step {target_step} "
        f"(max_attempts={PASS_AT_K_MAX_ATTEMPTS}) ==="
    )
    phase2_groups_by_samples = {}
    for dataset_name, validation_path in datasets.items():
        if not os.path.exists(validation_path):
            continue
        samples = DATASET_CONFIGS.get(dataset_name, {}).get("main_samples", default_main_samples)
        phase2_groups_by_samples.setdefault(samples, []).append((dataset_name, validation_path))

    phase2_results = []
    phase2_groups = list(phase2_groups_by_samples.items())
    target_is_active = current_step == target_step
    for group_index, (samples, dataset_group) in enumerate(phase2_groups, start=1):
        dataset_names = [item[0] for item in dataset_group]
        validation_paths = [item[1] for item in dataset_group]
        output_dir = os.path.join(
            eval_outputs_folder,
            "phase2",
            f"step_{target_step}",
            f"n_{samples}_{'_'.join(dataset_names)}",
        )
        group_started_at = time.monotonic()
        _publish_eval_event(
            event_reporter,
            {
                "type": "phase2_group_started",
                "step": target_step,
                "datasets": dataset_names,
                "samples": samples,
                "group_index": group_index,
                "group_total": len(phase2_groups),
            },
        )
        signature = _evaluation_signature(
            phase="phase2",
            step=target_step,
            checkpoint_dir=target_checkpoint["merged_dir"],
            dataset_names=dataset_names,
            validation_paths=validation_paths,
            n_samples=samples,
            max_attempts=PASS_AT_K_MAX_ATTEMPTS,
        )
        manifest_exists = os.path.isfile(os.path.join(output_dir, RESUME_MANIFEST_NAME))
        saved_metrics = _load_completed_eval(output_dir, signature) if resume else None
        if saved_metrics is None and resume and not manifest_exists:
            saved_metrics = _recover_metrics_from_existing_jsonl(
                trainer=trainer,
                val_file_list=validation_paths,
                n_samples=samples,
                output_dir=output_dir,
                step=target_step,
            )
            if saved_metrics is not None:
                _save_completed_eval(trainer, output_dir, signature, saved_metrics)
        if saved_metrics is not None:
            print(
                f"[RESUME] Reusing completed phase2 evaluation at step {target_step} "
                f"for {dataset_names} (n={samples})"
            )
            metrics = saved_metrics
        else:
            if not target_is_active:
                _activate_checkpoint(trainer, target_checkpoint, current_step)
                current_step = target_step
                target_is_active = True
            metrics, _ = _execute_or_resume_eval_pass(
                trainer=trainer,
                val_file_list=validation_paths,
                dataset_names=dataset_names,
                n_samples=samples,
                output_dir=output_dir,
                phase="phase2",
                checkpoint=target_checkpoint,
                max_attempts=PASS_AT_K_MAX_ATTEMPTS,
                resume=False,
            )
        phase2_results.append(
            {
                "datasets": dataset_names,
                "samples": samples,
                "output_dir": output_dir,
                "metrics": metrics,
            }
        )
        _publish_eval_event(
            event_reporter,
            {
                "type": "phase2_group_completed",
                "step": target_step,
                "datasets": dataset_names,
                "samples": samples,
                "group_index": group_index,
                "group_total": len(phase2_groups),
                "elapsed_seconds": time.monotonic() - group_started_at,
                "metrics": metrics,
            },
        )
    return phase2_results


@ray.remote(max_calls=1)
def run_single_trainer_eval_task(pipeline_spec: dict) -> dict:
    """Run one or both phases with one trainer and one fixed worker environment."""
    phase_mode = pipeline_spec.get("phase_mode", "both")
    if phase_mode not in {"both", "phase1", "phase2"}:
        raise ValueError(f"Unknown phase_mode: {phase_mode!r}")

    max_attempts = _validate_max_attempts(int(pipeline_spec["max_attempts"]))
    if phase_mode in {"both", "phase2"} and max_attempts != PASS_AT_K_MAX_ATTEMPTS:
        raise ValueError(
            f"{phase_mode} must run with max_attempts={PASS_AT_K_MAX_ATTEMPTS}, "
            f"got {max_attempts}"
        )
    _set_eval_agent_environment(max_attempts)

    task_spec = pipeline_spec
    if phase_mode == "phase2":
        target_step = pipeline_spec["target_step"]
        if target_step is None:
            raise ValueError("phase2 requires an explicit target_step")
        target_checkpoint = next(
            checkpoint
            for checkpoint in pipeline_spec["checkpoint_specs"]
            if checkpoint["step"] == target_step
        )
        # Construct the restarted Phase 2 trainer directly from the target HF
        # checkpoint, avoiding an unnecessary hot-load of the first checkpoint.
        task_spec = dict(pipeline_spec)
        task_spec["checkpoint_specs"] = [target_checkpoint]

    session = _EvalTaskSession()
    try:
        trainer = _build_single_trainer(task_spec, session)
        phase1_results = []
        phase2_results = []
        current_step = None

        if phase_mode in {"both", "phase1"}:
            phase1_results, target_step, current_step = _run_phase1(trainer, pipeline_spec)
        else:
            target_step = int(pipeline_spec["target_step"])

        if phase_mode in {"both", "phase2"}:
            phase2_results = _run_phase2(
                trainer,
                pipeline_spec,
                target_step,
                current_step,
            )

        return {
            "phase1_results": phase1_results,
            "selected_target_step": target_step,
            "phase2_results": phase2_results,
        }
    finally:
        session.cleanup()


_VERL_BEST_AT_K_RE = re.compile(
    r"^val-(?:core|aux)/(?P<data_source>.+)/(?P<variable>acc|reward)/best@(?P<k>\d+)/mean$"
)
_VERL_MEAN_AT_N_RE = re.compile(
    r"^val-(?:core|aux)/(?P<data_source>.+)/(?P<variable>acc|reward)/mean@(?P<n>\d+)$"
)


def extract_pass_at_k_from_verl_metrics(metrics: dict, eval_n: int) -> Dict[str, Dict[int, float]]:
    """Extract one pass@k curve per VERL data_source from a single n-rollout eval.

    VERL already groups samples by data_source and reports best@2/mean,
    best@4/mean, ..., best@n/mean. Its mean@n is the k=1 baseline. Prefer
    the `acc` variable when available and otherwise use `reward`.
    """
    by_source_and_variable: Dict[str, Dict[str, Dict[int, float]]] = {}

    for metric_name, metric_value in metrics.items():
        if not isinstance(metric_value, (int, float, np.number)):
            continue

        best_match = _VERL_BEST_AT_K_RE.match(metric_name)
        if best_match:
            source = best_match.group("data_source")
            variable = best_match.group("variable")
            k = int(best_match.group("k"))
            if k <= eval_n:
                by_source_and_variable.setdefault(source, {}).setdefault(variable, {})[k] = float(metric_value)
            continue

        mean_match = _VERL_MEAN_AT_N_RE.match(metric_name)
        if mean_match and int(mean_match.group("n")) == eval_n:
            source = mean_match.group("data_source")
            variable = mean_match.group("variable")
            by_source_and_variable.setdefault(source, {}).setdefault(variable, {})[1] = float(metric_value)

    pass_by_source = {}
    for source, variables in by_source_and_variable.items():
        selected_variable = "acc" if "acc" in variables else "reward"
        pass_by_source[source] = dict(sorted(variables[selected_variable].items()))
    return pass_by_source


def log_pass_at_k_history_to_wandb(
    mean_pass_by_dataset: Dict,
    target_step: int,
) -> None:
    """Log pass@k as ordinary W&B history metrics with k as the x-axis."""
    if not mean_pass_by_dataset:
        return

    # These are regular scalar metrics, not custom charts. Defining k as their
    # step metric makes W&B use k as the default x-axis and keeps each dataset
    # under the val-main section for cross-run comparison. Never override
    # W&B's run-wide Step here; that would corrupt checkpoint-step charts.
    wandb.define_metric("k")
    for dataset_name in mean_pass_by_dataset:
        wandb.define_metric(f"val-main/{dataset_name}/pass@k", step_metric="k")

    k_values = sorted(
        {int(k) for pass_values in mean_pass_by_dataset.values() for k in pass_values}
    )
    for k in k_values:
        payload = {
            "k": k,
            "selected_target_step": target_step,
        }
        for dataset_name, pass_values in mean_pass_by_dataset.items():
            if k in pass_values:
                payload[f"val-main/{dataset_name}/pass@k"] = float(pass_values[k])

        # Log every dataset for this k in one committed row. Each curve uses k
        # as its independent x-axis while W&B's ordinary Step advances naturally.
        _safe_wandb_log(payload)

    print(f"[W&B] Logged native VERL pass@k history with k={k_values}")


def _safe_wandb_log(payload: dict) -> None:
    """Keep a transient W&B problem from killing a long evaluation job."""
    try:
        wandb.log(payload, commit=True)
    except Exception as exc:
        print(f"[!] W&B logging failed, but evaluation will continue: {type(exc).__name__}: {exc}")


def _handle_eval_event(event: dict, state: dict) -> None:
    event_type = event["type"]
    step = event.get("step")

    if event_type == "phase1_started":
        state["current_step"] = step
        state["current_started_at"] = time.monotonic()
        state["checkpoint_index"] = event["checkpoint_index"]
        state["checkpoint_total"] = event["checkpoint_total"]
        state["phase"] = 1
        _safe_wandb_log(
            {
                "eval/runner_alive": 1,
                "eval/phase": 1,
                "eval/current_checkpoint": step,
                "eval/checkpoints_completed": event["checkpoint_index"] - 1,
                "eval/checkpoints_total": event["checkpoint_total"],
            }
        )
        if wandb.run is not None:
            wandb.run.summary["eval_current_checkpoint"] = step
            wandb.run.summary["eval_status"] = "running_phase1"
        print(
            f"[W&B] Checkpoint {event['checkpoint_index']}/{event['checkpoint_total']} "
            f"(step {step}) started"
        )
        return

    if event_type == "phase1_completed":
        # global_step is written only on completed Phase 1 checkpoints, so it
        # remains strictly increasing and valid as a W&B x-axis.
        previously_logged_step = -1
        if wandb.run is not None:
            previously_logged_step = int(
                wandb.run.summary.get("latest_completed_eval_step", -1)
            )

        if step <= previously_logged_step:
            state["logged_phase1_steps"].add(step)
            print(
                f"[W&B RESUME] Step {step} already logged "
                f"(latest={previously_logged_step}); skipping duplicate"
            )
            return


        for metric_name, metric_value in event["metrics"].items():
            if (
                metric_name != "global_step"
                and metric_name not in state["defined_checkpoint_metrics"]
                and isinstance(metric_value, (int, float, np.number))
            ):
                wandb.define_metric(metric_name, step_metric="global_step")
                state["defined_checkpoint_metrics"].add(metric_name)

        payload = dict(event["metrics"])
        payload.update(
            {
                "global_step": step,
                "eval/runner_alive": 1,
                "eval/phase": 1,
                "eval/current_checkpoint": step,
                "eval/checkpoints_completed": event["checkpoint_index"],
                "eval/checkpoints_total": event["checkpoint_total"],
                "eval/checkpoint_minutes": event["elapsed_seconds"] / 60.0,
            }
        )
        _safe_wandb_log(payload)
        state["logged_phase1_steps"].add(step)
        if wandb.run is not None:
            wandb.run.summary["latest_completed_eval_step"] = step
            wandb.run.summary["phase1_checkpoints_completed"] = event["checkpoint_index"]
        print(f"[W&B] Logged completed validation metrics for checkpoint step {step}")
        return

    if event_type == "target_selected":
        state["phase"] = 2
        if wandb.run is not None:
            wandb.run.summary["selected_target_step"] = step
            wandb.run.summary["eval_status"] = "running_phase2"
        _safe_wandb_log(
            {
                "selected_target_step": step,
                "eval/phase": 2,
                "eval/selected_target_step": step,
            }
        )
        print(f"[W&B] Selected step {step} for Phase 2")
        return

    if event_type == "phase2_group_started":
        dataset_names = event["datasets"]
        group_label = ", ".join(dataset_names)
        state["current_step"] = step
        state["current_started_at"] = time.monotonic()
        state["current_dataset"] = group_label
        state["phase"] = 2
        _safe_wandb_log(
            {
                "selected_target_step": step,
                "eval/runner_alive": 1,
                "eval/phase": 2,
                "eval/phase2_group_started": 1,
                "eval/phase2_eval_n": event["samples"],
                "eval/phase2_groups_completed": event["group_index"] - 1,
                "eval/phase2_groups_total": event["group_total"],
            }
        )
        if wandb.run is not None:
            wandb.run.summary["eval_current_datasets"] = group_label
        print(
            f"[W&B] Phase 2 group {event['group_index']}/{event['group_total']} "
            f"started with eval.n={event['samples']}: {group_label}"
        )
        return

    if event_type == "phase2_group_completed":
        dataset_names = event["datasets"]
        group_label = ", ".join(dataset_names)
        # Keep the raw deep-eval metrics separate from Phase 1 checkpoint
        # curves. The returned metrics still retain VERL's data_source names.
        payload = {
            f"eval-phase2/{metric_name}": metric_value
            for metric_name, metric_value in event["metrics"].items()
            if metric_name != "global_step"
        }
        payload.update(
            {
                "selected_target_step": step,
                "eval/runner_alive": 1,
                "eval/phase": 2,
                "eval/phase2_group_completed": 1,
                "eval/phase2_eval_n": event["samples"],
                "eval/phase2_groups_completed": event["group_index"],
                "eval/phase2_groups_total": event["group_total"],
                "eval/phase2_group_minutes": event["elapsed_seconds"] / 60.0,
            }
        )
        _safe_wandb_log(payload)
        if wandb.run is not None:
            wandb.run.summary["latest_completed_phase2_datasets"] = group_label
        print(
            f"[W&B] Logged Phase 2 VERL metrics for eval.n={event['samples']}: "
            f"{group_label}"
        )


def _wait_for_eval_with_live_wandb(task_ref, reporter):
    """Poll the Ray task while streaming its metric events to the driver."""
    state = {
        "logged_phase1_steps": set(),
        "current_step": None,
        "current_dataset": None,
        "current_started_at": None,
        "checkpoint_index": 0,
        "checkpoint_total": 0,
        "phase": 0,
        "defined_checkpoint_metrics": set(),
    }
    runner_started_at = time.monotonic()
    last_heartbeat = runner_started_at

    while True:
        ready, _ = ray.wait([task_ref], num_returns=1, timeout=5.0)

        for event in ray.get(reporter.drain.remote()):
            _handle_eval_event(event, state)

        now = time.monotonic()
        if now - last_heartbeat >= 60:
            heartbeat = {
                "eval/runner_alive": 1,
                "eval/total_elapsed_minutes": (now - runner_started_at) / 60.0,
            }
            if state["current_step"] is not None:
                heartbeat["eval/current_checkpoint"] = state["current_step"]
                if state["phase"] == 2:
                    heartbeat["selected_target_step"] = state["current_step"]
            if state["current_started_at"] is not None:
                heartbeat["eval/current_work_minutes"] = (now - state["current_started_at"]) / 60.0
            _safe_wandb_log(heartbeat)
            last_heartbeat = now

        if ready:
            # publish() calls are synchronous, but drain once more to handle an
            # event that arrived between the previous drain and task completion.
            for event in ray.get(reporter.drain.remote()):
                _handle_eval_event(event, state)
            return ray.get(task_ref), state


def evaluate_pipeline(
    checkpoint_folder: str,
    eval_base_dir: str,
    verl_dir: str,
    run_name: str,
    dataset_paths: List[str],
    default_step_samples: int = 256,
    default_main_samples: int = 4096,
    wandb_project: str = "r-grpo_eval",
    num_gpus: int = 1,
    target_step: Optional[int] = DEFAULT_TARGET_STEP,
    max_attempts: int = 1,
    resume: bool = True,
) -> None:
    phase1_max_attempts = _validate_max_attempts(max_attempts)
    phase2_max_attempts = PASS_AT_K_MAX_ATTEMPTS
    restart_for_phase2 = phase1_max_attempts != phase2_max_attempts
    _set_eval_agent_environment(phase1_max_attempts)

    absolute_verl_dir = os.path.abspath(verl_dir)
    if absolute_verl_dir not in sys.path:
        sys.path.insert(0, absolute_verl_dir)

    run_eval_dir = os.path.abspath(os.path.join(eval_base_dir, run_name))
    merged_output_folder = os.path.join(run_eval_dir, "merged_checkpoints")
    eval_outputs_folder = os.path.join(run_eval_dir, "eval_outputs")
    os.makedirs(merged_output_folder, exist_ok=True)
    os.makedirs(eval_outputs_folder, exist_ok=True)

    checkpoints = get_sorted_checkpoints(checkpoint_folder)
    if not checkpoints:
        raise FileNotFoundError(f"No global_step_* folders found inside {checkpoint_folder}")

    datasets = dict(parse_dataset_info(path) for path in dataset_paths)
    existing_datasets = [path for path in datasets.values() if os.path.exists(path)]
    if not existing_datasets:
        raise FileNotFoundError(f"None of the validation datasets exist: {list(datasets.values())}")

    checkpoint_specs = []
    for step, checkpoint_dir in checkpoints:
        actor_dir = os.path.join(checkpoint_dir, "actor")
        requested_merged_dir = os.path.join(merged_output_folder, f"merged_step_{step}")
        merged_dir = merge_verl_checkpoint(actor_dir, requested_merged_dir)
        checkpoint_specs.append({"step": step, "merged_dir": merged_dir})

    available_steps = [checkpoint["step"] for checkpoint in checkpoint_specs]
    if target_step is not None and target_step not in available_steps:
        raise ValueError(f"Requested target step {target_step} was not found. Available steps: {available_steps}")

    if not ray.is_initialized():
        ray_runtime_env = get_ppo_ray_runtime_env()
        ray_runtime_env.setdefault("env_vars", {}).update(
            {
                "MAX_ATTEMPTS": str(phase1_max_attempts),
                "VERL_RETRY_PROMPT": EVAL_RETRY_PROMPT,
            }
        )
        ray.init(address="auto", ignore_reinit_error=True, runtime_env=ray_runtime_env)

    wandb.init(
        project=wandb_project,
        name=run_name,
        config={
            "run_name": run_name,
            "checkpoint_folder": checkpoint_folder,
            "target_step": target_step,
            "default_step_samples": default_step_samples,
            "default_main_samples": default_main_samples,
            "datasets": list(datasets),
            "gpus": num_gpus,
            "single_trainer": not restart_for_phase2,
            "trainer_boot_count": 2 if restart_for_phase2 else 1,
            "restart_trainer_for_phase2": restart_for_phase2,
            "gpu_memory_utilization": 0.3,
            "validation_temperature": EVAL_TEMPERATURE,
            "validation_top_p": EVAL_TOP_P,
            "validation_top_k": EVAL_TOP_K,
            "validation_do_sample": EVAL_DO_SAMPLE,
            "max_attempts": phase1_max_attempts,
            "phase1_max_attempts": phase1_max_attempts,
            "phase2_max_attempts": phase2_max_attempts,
            "max_validation_responses_per_batch": MAX_VALIDATION_RESPONSES_PER_BATCH,
            "trainer_log_val_generations": LOG_VAL_GENERATIONS,
            "resume_completed_evaluations": resume,
        },
    )
    wandb.define_metric("global_step")

    event_reporter = None
    try:
        event_reporter = EvalEventReporter.remote()
        base_pipeline_spec = {
            "num_gpus": num_gpus,
            "sample_val_files": repr([existing_datasets[0]]),
            "datasets": datasets,
            "checkpoint_specs": checkpoint_specs,
            "default_step_samples": default_step_samples,
            "default_main_samples": default_main_samples,
            "eval_outputs_folder": eval_outputs_folder,
            "target_step": target_step,
            "event_reporter": event_reporter,
            "resume": resume,
        }

        def launch_eval_task(phase_mode: str, task_max_attempts: int, selected_step=None):
            task_spec = dict(base_pipeline_spec)
            task_spec.update(
                {
                    "phase_mode": phase_mode,
                    "max_attempts": task_max_attempts,
                }
            )
            if selected_step is not None:
                task_spec["target_step"] = selected_step
            return run_single_trainer_eval_task.options(
                runtime_env={
                    "env_vars": {
                        "MAX_ATTEMPTS": str(task_max_attempts),
                        "VERL_RETRY_PROMPT": EVAL_RETRY_PROMPT,
                    }
                }
            ).remote(task_spec)

        if restart_for_phase2:
            print(
                "[+] Phase 1 will use "
                f"max_attempts={phase1_max_attempts}. Phase 2 requires "
                f"max_attempts={phase2_max_attempts}, so the trainer will restart once."
            )
            phase1_ref = launch_eval_task("phase1", phase1_max_attempts)
            phase1_result, live_state = _wait_for_eval_with_live_wandb(
                phase1_ref,
                event_reporter,
            )
            selected_target_step = phase1_result["selected_target_step"]

            print(
                "\n[+] Phase 1 complete. Booting a fresh Phase 2 trainer from "
                f"merged step {selected_target_step} with max_attempts={phase2_max_attempts}."
            )
            phase2_ref = launch_eval_task(
                "phase2",
                phase2_max_attempts,
                selected_step=selected_target_step,
            )
            phase2_result, _ = _wait_for_eval_with_live_wandb(
                phase2_ref,
                event_reporter,
            )
            result = {
                "phase1_results": phase1_result["phase1_results"],
                "selected_target_step": selected_target_step,
                "phase2_results": phase2_result["phase2_results"],
            }
        else:
            print("[+] Both phases use max_attempts=1; reusing one trainer for the entire run.")
            task_ref = launch_eval_task("both", phase1_max_attempts)
            result, live_state = _wait_for_eval_with_live_wandb(task_ref, event_reporter)

        # A late replay would append an old global_step after Phase 2 and make
        # the checkpoint x-axis non-monotonic. Reporter writes are synchronous,
        # so missing events are reported rather than replayed out of order.
        missing_streamed_steps = [
            step
            for step, _ in result["phase1_results"]
            if step not in live_state["logged_phase1_steps"]
        ]
        if missing_streamed_steps:
            print(f"[!] Missing streamed Phase 1 W&B steps: {missing_streamed_steps}")

        selected_target_step = result["selected_target_step"]
        wandb.config.update({"selected_target_step": selected_target_step}, allow_val_change=True)

        mean_pass_by_dataset = {}
        for phase2_result in result["phase2_results"]:
            extracted = extract_pass_at_k_from_verl_metrics(
                phase2_result["metrics"],
                eval_n=phase2_result["samples"],
            )
            if not extracted:
                print(
                    "[!] VERL returned no acc/reward best@k metrics for "
                    f"eval.n={phase2_result['samples']} datasets={phase2_result['datasets']}"
                )
                continue

            for data_source, pass_values in extracted.items():
                mean_pass_by_dataset.setdefault(data_source, {}).update(pass_values)

        if mean_pass_by_dataset:
            log_pass_at_k_history_to_wandb(
                mean_pass_by_dataset,
                target_step=selected_target_step,
            )
    finally:
        if event_reporter is not None:
            try:
                ray.kill(event_reporter, no_restart=True)
            except Exception:
                pass
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VERL merged-checkpoint evaluator with conditional Phase 2 restart"
    )
    parser.add_argument("--checkpoint_dir", required=True, help="Path containing global_step_* checkpoints")
    parser.add_argument("--eval_dir", default="./checkpoints/r-grpo_eval", help="Base evaluation output directory")
    parser.add_argument("--verl_dir", default="./verl")
    parser.add_argument("--wandb_project", default="r-grpo_eval")
    parser.add_argument(
        "--target_step",
        type=int,
        default=DEFAULT_TARGET_STEP,
        help=f"Explicit checkpoint for deep evaluation (default: {DEFAULT_TARGET_STEP})",
    )
    parser.add_argument("--val_files", nargs="+", default=DEFAULT_DATASET_PATHS)
    parser.add_argument("--default_step_samples", type=int, default=256)
    parser.add_argument("--default_main_samples", type=int, default=4096)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Ignore saved completion manifests and rerun every evaluation pass.",
    )
    parser.add_argument(
        "--max_attempts",
        "--maxattempts",
        dest="max_attempts",
        type=int,
        default=1,
        help=(
            "Maximum attempts for checkpoint-vs-step evaluation (default: 1). "
            "The target-step pass@k phase always uses 1."
        ),
    )
    args = parser.parse_args()

    evaluate_pipeline(
        checkpoint_folder=os.path.abspath(args.checkpoint_dir),
        eval_base_dir=os.path.abspath(args.eval_dir),
        verl_dir=os.path.abspath(args.verl_dir),
        run_name=os.path.basename(os.path.normpath(args.checkpoint_dir)),
        dataset_paths=args.val_files,
        default_step_samples=args.default_step_samples,
        default_main_samples=args.default_main_samples,
        wandb_project=args.wandb_project,
        num_gpus=args.gpus,
        target_step=args.target_step,
        max_attempts=args.max_attempts,
        resume=not args.no_resume,
    )
