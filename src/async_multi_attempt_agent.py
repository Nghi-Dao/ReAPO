import os
import uuid
import torch
import logging
import asyncio
from typing import Dict, Any

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from src.reward import compute_score

logger = logging.getLogger(__name__)

class FullyAsyncMultiAttemptGRPO(AgentLoopBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.retry_prompt = os.getenv('VERL_RETRY_PROMPT', "Please check your answer and try again.")
        self.max_attempts = int(os.getenv('MAX_ATTEMPTS', '4'))
        self.retry_ids = self._get_retry_ids()

    def _get_retry_ids(self):
        """Helper to safely calculate retry token IDs using robust extraction."""
        if not self.tokenizer:
            return []

        test_msg = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "a"}]
        full_msg = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "a"},
                    {"role": "user", "content": self.retry_prompt}]

        def robust_extract(obj):
            # Check for dictionary-like access first
            if hasattr(obj, 'get') and obj.get('input_ids') is not None:
                return obj['input_ids']
            # Fallback for BatchEncoding which might not pass isinstance(dict)
            if hasattr(obj, 'input_ids'):
                return obj.input_ids
            return obj

        prefix = robust_extract(self.tokenizer.apply_chat_template(test_msg, tokenize=True, add_generation_prompt=False))
        full = robust_extract(self.tokenizer.apply_chat_template(full_msg, tokenize=True, add_generation_prompt=True))

        # Convert to list to handle potential tensors before slicing
        prefix_list = prefix.flatten().tolist() if isinstance(prefix, torch.Tensor) else list(prefix)
        full_list = full.flatten().tolist() if isinstance(full, torch.Tensor) else list(full)

        return full_list[len(prefix_list)-1:]

    async def run(self, sampling_params: Dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = kwargs.get('prompt') or kwargs.get('raw_prompt')
        ground_truth = kwargs.get('reward_model', {}).get('ground_truth', "")

        rollout_cfg = self.config.actor_rollout_ref
        max_model_len = rollout_cfg.get('max_model_len', 4096)

        # 1. Offload CPU-bound tokenization to a background thread
        prompt_output = await asyncio.to_thread(
            self.tokenizer.apply_chat_template,
            messages, tokenize=True, add_generation_prompt=True
        )
        prompt_ids = prompt_output

        current_context_ids = list(prompt_ids)
        accumulated_response_ids = []
        accumulated_response_mask = []
        accumulated_log_probs = []  # Track log probs
        attempt_mask = []

        request_id = str(uuid.uuid4())
        attempt = 0

        # Track model versions (global_steps) across potentially multiple attempts
        min_global_steps = None
        max_global_steps = None
        global_steps = None

        metrics = {}
        for n in range(1, self.max_attempts + 1):
            metrics[f'trial_{n}_acc'] = 0.0
            metrics[f'cumulative_trial_{n}_acc'] = 0.0

        for i in range(1, self.max_attempts + 1):
            current_len = len(current_context_ids)
            if max_model_len - current_len <= 1:
                break

            attempt = i

            # 2. Asynchronous generation request (non-blocking)
            out = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=current_context_ids,
                sampling_params=sampling_params
            )

            # Safely extract gen_ids and extra_fields handling potential list wrappers
            out_obj = out[0] if isinstance(out, list) and len(out) > 0 and not isinstance(out[0], int) else out
            gen_ids = getattr(out_obj, 'input_ids', out_obj)

            current_extra_fields = getattr(out_obj, 'extra_fields', {})
            current_global_steps = current_extra_fields.get('global_steps', None)

            # Update step tracking
            if current_global_steps is not None:
                global_steps = current_global_steps
                if min_global_steps is None:
                    min_global_steps = current_global_steps
                max_global_steps = current_global_steps

            # Extract log_probs (default to 0.0 if missing, though the server should provide them)
            current_log_probs = getattr(gen_ids, 'log_probs', None)
            if not current_log_probs:
                current_log_probs = [0.0] * len(gen_ids.token_ids)

            accumulated_response_ids.extend(gen_ids.token_ids)
            accumulated_log_probs.extend(current_log_probs)  # Store log probs
            current_context_ids.extend(gen_ids.token_ids)
            accumulated_response_mask.extend([1] * len(gen_ids.token_ids))
            attempt_mask.extend([i] * len(gen_ids.token_ids))

            # 3. Offload CPU-bound decoding and reward computation to background threads
            response_str = await asyncio.to_thread(
                self.tokenizer.decode,
                gen_ids.token_ids, skip_special_tokens=True
            )
            score = await asyncio.to_thread(
                compute_score,
                None, response_str, ground_truth
            )

            if score > 0:
                metrics[f'trial_{i}_acc'] = 1.0
                for n in range(i, self.max_attempts + 1):
                    metrics[f'cumulative_trial_{n}_acc'] = 1.0
                break
            else:
                metrics[f'trial_{i}_acc'] = 0.0
                metrics[f'cumulative_trial_{i}_acc'] = 0.0

            if attempt == self.max_attempts:
                break

            # 4. Append Retry Prompt
            current_context_ids.extend(self.retry_ids)
            accumulated_response_ids.extend(self.retry_ids)
            accumulated_log_probs.extend([0.0] * len(self.retry_ids))  # Pad log probs for retry tokens (0.0 because they are masked out)
            accumulated_response_mask.extend([0] * len(self.retry_ids))
            attempt_mask.extend([0] * len(self.retry_ids))

        # Truncate to max_response_length
        max_response_length = self.config.data.max_response_length
        final_ids = accumulated_response_ids[:max_response_length]
        final_mask = accumulated_response_mask[:max_response_length]
        final_log_probs = accumulated_log_probs[:max_response_length]  # Truncate log probs

        attempt_mask_tensor = torch.tensor(attempt_mask[:max_response_length])

        # Final decode offloaded
        final_response_str = await asyncio.to_thread(
            self.tokenizer.decode, final_ids, skip_special_tokens=True
        )

        # 5. Pack everything EXCEPT log probs into extra_fields
        extra_fields = {
            "trial_reward": metrics,
            "response": final_response_str,
            "attempt_mask": attempt_mask_tensor,
            "global_steps": global_steps,
            "min_global_steps": min_global_steps,
            "max_global_steps": max_global_steps,
        }

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=final_ids,
            response_mask=final_mask,
            response_logprobs=final_log_probs,  # <--- PASS DIRECTLY HERE AS A LIST OF FLOATS
            num_turns=attempt,
            metrics={},
            extra_fields=extra_fields
        )
