# ReAPO

<!-- TODO: full method name, paper title, author list, arXiv link, and an image -->

Code for **TODO: paper title** ([arXiv:TODO](https://arxiv.org/abs/TODO)).

ReAPO is a RLVR sampling and advantage calculation method in which a single rollout can contain several attempts at the same problem. When an attempt is wrong, the model is prompted to try again in the same context, up to a fixed number of attempts. Advantages are then computed after pooling together all attempts rather than individual rollouts, so the attempt that finally solves the problem is credited separately from the failed attempts before it.

The implementation is built on [verl](https://github.com/verl-project/verl). A copy of verl with a small number of changes is included in `verl/` (see [Changes to verl](#changes-to-verl)).

With `MAX_ATTEMPTS=1` the estimator reduces to GRPO with a binary reward, which is how the baseline scripts use it.

## Repository layout

```
.
├── src/
│   ├── multi_attempt_agent.py        # multi-attempt agent loop (synchronous training)
│   ├── custom_agents.yaml            # registers the two agent loops with verl
│   └── eval_checkpoints.py           # checkpoint merging + evaluation pipeline
├── data_prep/                        # scripts that write the parquet datasets to ~/data
├── commands/                         # training and evaluation launch scripts
└── verl/                             # vendored verl (see "Changes to verl")
```

## Installation

<!-- TODO: state the exact Python / CUDA / torch / vLLM / transformers versions used for the paper (e.g. paste `pip freeze`). -->

```bash
git clone https://github.com/Nghi-Dao/ReAPO.git
cd ReAPO
python -m venv venv && source venv/bin/activate

cd verl
bash scripts/install_vllm_sglang_mcore.sh
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
cd ..

pip install -r requirements.txt     
pip install flash-attn --no-build-isolation
wandb login                         
```

## Data

Each script downloads a dataset from the Hugging Face Hub and writes verl-format parquet files under `~/data/<name>/` (override with `--local_dir`).

```bash
python data_prep/polaris.py        # training set for the Qwen3 runs
for d in aime24 aime25 hmmt25 beyondaime math-500 minerva gpqa-diamond; do
    python data_prep/$d.py
done

# Qwen3-0.6B-Base runs
python data_prep/gsm8k.py          # ~/data/gsm8k/train.parquet
python data_prep/platinum.py       # ~/data/gsm8k-platinum/test.parquet
```

| Script | Source on the Hub | Used as |
|---|---|---|
| `polaris.py` | `POLARIS-Project/Polaris-Dataset-53K` | training |
| `gsm8k.py` | `openai/gsm8k` | training |
| `platinum.py` | `madrylab/gsm8k-platinum` | evaluation |
| `aime24.py` | `math-ai/aime24` | evaluation |
| `aime25.py` | `math-ai/aime25` | evaluation |
| `hmmt25.py` | `MathArena/hmmt_feb_2025` | evaluation |
| `beyondaime.py` | `ByteDance-Seed/BeyondAIME` | evaluation |
| `math-500.py` | `HuggingFaceH4/MATH-500` | evaluation |
| `minerva.py` | `math-ai/minervamath` | evaluation |
| `gpqa-diamond.py` | `fingertap/GPQA-Diamond` | evaluation |

Every problem is posed as a single user message ending in *"Please reason step by step, and put your final answer within \boxed{}."* (GSM8K: *"Let's think step by step and output the final answer within \boxed{}."*). For the evaluation sets only `test.parquet` is used.

## Training

```bash
bash commands/qwen1.7b.sh
```

Each script sets a few environment variables at the top and then launches `verl.trainer.main_ppo` with Hydra overrides:

| Variable | Meaning |
|---|---|
| `MAX_ATTEMPTS` | attempts per rollout (1 = standard single-attempt RL) |
| `VERL_RETRY_PROMPT` | feedback message appended after a wrong attempt |
| `ROLLOUT_NUM` | rollouts per prompt |
| `RESPONSE_LEN` / `PROMPT_LEN` | max response / prompt tokens; the context limit is their sum and is shared by all attempts |
| `SEED` | data shuffling seed |

The agent loop reads `MAX_ATTEMPTS` and `VERL_RETRY_PROMPT` from the environment. If they are unset it falls back to 4 attempts and *"Please check your answer and try again."*. The training scripts use *"Your answer is wrong. Reflect on your previous answer and try something else."* (`qwen1.7b_difprompt.sh`: *"This response is not correct, lets start over."*).


## Evaluation

`src/eval_checkpoints.py` evaluates every saved checkpoint of a run and then runs a large-sample pass@k evaluation on one selected checkpoint. It needs a running Ray cluster:

```bash
ray start --head --num-gpus=4
python -m src.eval_checkpoints \
    --checkpoint_dir checkpoints/r-grpo_train/<experiment_name> \
    --gpus 4 \
    --target_step 1040 \
    --max_attempts 1
```

`commands/eval.sh` wraps the same call; edit `--checkpoint_dir` before running it.


## Changes to verl

`verl/` is [verl](https://github.com/verl-project/verl) at commit [`16bdafa`](https://github.com/verl-project/verl/commit/16bdafaada14bccf317bfae13332df08fc2be857) (2026-05-21, version 0.8.0.dev) with these changes:

- `verl/verl/trainer/ppo/core_algos.py`: adds the `multi_attempt_grpo` and `maxrl` advantage estimators.
- `verl/verl/trainer/ppo/ray_trainer.py`: passes `norm_adv_by_std_in_grpo` to those two estimators, and passes the non-tensor batch (which carries `attempt_mask`) to `multi_attempt_grpo`.
- `verl/verl/experimental/reward_loop/reward_manager/dapo.py`: forwards the agent loop's extra fields and `num_turns` into `extra_info` for the reward function, as the naive reward manager already does.
- The `recipe` submodule and `.vscode/` folder are removed.

## Citation

<!-- TODO: replace with the arXiv BibTeX once the preprint is live. -->

```bibtex
@misc{reapo2026,
  title         = {TODO},
  author        = {},
  year          = {2026},
  eprint        = {TODO},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```
