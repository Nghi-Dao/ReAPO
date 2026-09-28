import re
from src.reward import compute_score as base_compute_score
from collections import Counter

def compute_decay_reward(data_source, solution_str, ground_truth, extra_info):
    
    trial_reward = extra_info.get('trial_reward', {})
    
    is_correct = base_compute_score(None, solution_str, ground_truth)
    
    output = {}

    if is_correct > 0:
        output["score"] = 1.0 #/ extra_info['num_turns']
    else:
        output["score"] = 0.0

    for k in trial_reward:
        output[k] = trial_reward[k]
    
    attempt_mask = [v.item() for i, v in enumerate(extra_info.get('attempt_mask', {}))]
    counts = dict(Counter(attempt_mask))

    for i in range(1, extra_info.get('max_attempts', 1)+1):
        if i in counts:
            output[f'trial_{i}_length'] = counts[i]
        else:
            output[f'trial_{i}_length'] = 0
    
    return output
