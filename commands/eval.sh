export PROJECT_ROOT=$(pwd)
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

source venv/bin/activate

export MAX_ATTEMPTS=1
export VERL_RETRY_PROMPT="Your answer is wrong. Reflect on your previous answer and try something else."


ray stop --force
pkill -f "ray"
ray start --head --num-gpus=4

unset ROCR_VISIBLE_DEVICES

export RAY_EXPERIMENTAL_NO_CUDA_VISIBLE_DEVICES=0
PYTHONUNBUFFERED=1 python -m src.eval_checkpoints \
    --checkpoint_dir ./checkpoints/r-grpo_train/qwen3-1.7b_attempt2_rollout16_response4096_seed42 \
    --eval_dir ./checkpoints/r-grpo_eval \
    --verl_dir ./verl \
    --target_step 1040 \
    --wandb_project r-grpo_eval \
    --gpus 4 \
    --response_length 4096 \
    --max_attempts 2
