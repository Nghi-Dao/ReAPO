from verl.utils.reward_score.math_verify import compute_score as base_compute_score

#def compute_score(data_source, solution_str, ground_truth, extra_info=None):
#    return base_compute_score(solution_str, ground_truth)

# Import VERL's native, safely-picklable math_verify
from verl.utils.reward_score import math_verify as verl_math_verify

def extract_boxed_answer(text):
    # Find the last occurrence of \boxed{
    start_idx = text.rfind("\\boxed{")
    if start_idx == -1:
        return None

    content_start = start_idx + 7
    brace_count = 1

    # Count braces until we find the matching closing brace
    for i in range(content_start, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1

        if brace_count == 0:
            return text[content_start:i].strip()

    return None

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    try:
        # 1. Run your custom extraction locally (it's fast string manipulation, so it won't hang)
        extracted_answer = solution_str # extract_boxed_answer(solution_str)
        

        #if extracted_answer is None:
        #    return 0.0

        # 2. Hand off the heavy, timeout-prone parsing to VERL's built-in pool.
        # verl_math_verify safely handles the subprocess 'spawn' because it lives on the standard python path.
        score = verl_math_verify.compute_score(
            model_output=extracted_answer, 
            ground_truth=str(ground_truth)
        )
        
        return score

    except Exception as e:
        print(f"REWARD CRASH: {e}")
        return 0.0
