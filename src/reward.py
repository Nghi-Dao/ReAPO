from verl.utils.reward_score.math_verify import compute_score as base_compute_score

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    try:
        extracted_answer = solution_str 
        
        # verl_math_verify safely handles the subprocess 'spawn' because it lives on the standard python path.
        score = verl_math_verify.compute_score(
            model_output=extracted_answer, 
            ground_truth=str(ground_truth)
        )
        
        return score

    except Exception as e:
        print(f"REWARD CRASH: {e}")
        return 0.0
