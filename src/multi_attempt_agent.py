import os
import uuid
import torch
import logging
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from src.reward import compute_score 

class MultiAttemptGRPO(AgentLoopBase):
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
    
    async def run(self, sampling_params, **kwargs) -> AgentLoopOutput:
        # 1. Correct extraction based on your printed kwargs
        messages = kwargs.get('prompt') or kwargs.get('raw_prompt')
        ground_truth = kwargs.get('reward_model', {}).get('ground_truth', "")

        rollout_cfg = self.config.actor_rollout_ref
        max_model_len = rollout_cfg.get('max_model_len', 4096)
        
        # 2. Tokenize prompt
        prompt_output = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        prompt_ids = prompt_output#['input_ids']

        current_context_ids = list(prompt_ids)
        accumulated_response_ids = []
        accumulated_response_mask = []
        attempt_mask = []

        request_id = str(uuid.uuid4())
        attempt = 0

        
        metrics = {}
        for n in range(1, self.max_attempts + 1):
            metrics[f'trial_{n}_acc'] = 0.0
            metrics[f'cumulative_trial_{n}_acc'] = 0.0

        for i in range(1, self.max_attempts + 1):
            current_len = len(current_context_ids)
            if max_model_len - current_len <= 1:
                break

            attempt = i
            # 3. Generate
            out = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=current_context_ids,
                sampling_params=sampling_params
            )

            gen_ids = getattr(out, 'input_ids', out)
            if isinstance(gen_ids, list) and len(gen_ids) > 0 and not isinstance(gen_ids[0], int):
                gen_ids = getattr(gen_ids[0], 'input_ids', gen_ids[0])

            accumulated_response_ids.extend(gen_ids.token_ids)
            current_context_ids.extend(gen_ids.token_ids)
            accumulated_response_mask.extend([1] * len(gen_ids.token_ids))
            attempt_mask.extend([i] * len(gen_ids.token_ids))

            # 4. Score
            response_str = self.tokenizer.decode(gen_ids.token_ids, skip_special_tokens=True)
            score = compute_score(None, response_str, ground_truth)
            

            if score > 0:
                #accumulated_response_mask = ([0] * len(accumulated_response_mask)) + ([1] * len(gen_ids.token_ids))
                metrics[f'trial_{i}_acc'] = 1.0
                for n in range(i, self.max_attempts + 1):
                    metrics[f'cumulative_trial_{n}_acc'] = 1.0
                break
            else:
                #accumulated_response_mask.extend([1] * len(gen_ids.token_ids))
                metrics[f'trial_{i}_acc'] = 0.0
                metrics[f'cumulative_trial_{i}_acc'] = 0.0

            if attempt == self.max_attempts:
                break
            
            # 5. Append Retry Prompt
            current_context_ids.extend(self.retry_ids)
            accumulated_response_ids.extend(self.retry_ids)
            accumulated_response_mask.extend([0] * len(self.retry_ids))
            attempt_mask.extend([0] * len(self.retry_ids))

        final_ids = accumulated_response_ids[:self.config.data.max_response_length]
        final_mask = accumulated_response_mask[:self.config.data.max_response_length]
        attempt_mask = torch.tensor(attempt_mask[:self.config.data.max_response_length])
        
        extra_fields = {
            "trial_reward": metrics,
            "response": self.tokenizer.decode(final_ids, skip_special_tokens=True),
            "attempt_mask": attempt_mask,
            "max_attempts": self.max_attempts
        }
        
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=final_ids,
            response_mask=final_mask,
            num_turns=attempt,
            metrics={},
            extra_fields=extra_fields
        )
