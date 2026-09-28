export PROJECT_ROOT=$(pwd)
export PYTHONPATH=$(pwd)/src:$PYTHONPATH


export VENV_LIB="${PROJECT_ROOT}/beta_venv/lib/python3.12/site-packages"
export CUDA_HOME="$VENV_LIB/nvidia/cu13"
export PATH="${PROJECT_ROOT}/beta_venv/bin:$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/include:$CPATH"
export LIBRARY_PATH="$CUDA_HOME/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_LIB/torch/lib:$VENV_LIB/nvidia/cuda_runtime/lib:$CUDA_HOME/lib:$LD_LIBRARY_PATH"

source beta_venv/bin/activate

export MAX_ATTEMPTS=1
export VERL_RETRY_PROMPT="Your answer is wrong. Reflect on your previous answer and try something else."


# --- vLLM V1 & Ray Compatibility Fixes ---
export VLLM_USE_V1="1"                         # Enable V1 explicitly for verl
export VLLM_DISABLE_CUMEM="1"                  # Disables cumem_allocator.cpp to fix CUDA invalid argument errors
export VLLM_WORKER_MULTIPROC_METHOD="spawn"    # Prevents memory pointer corruption across Ray process boundaries
export VLLM_ATTENTION_BACKEND="FLASH_ATTN"     # Prevents Triton memory access exceptions
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

export VLLM_DISTRIBUTED_EXECUTOR_BACKEND="ray"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

# Ensure CUDA device order is consistent
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

ray stop --force
pkill -f "ray"
ray start --head --num-gpus=4

unset ROCR_VISIBLE_DEVICES

export RAY_EXPERIMENTAL_NO_CUDA_VISIBLE_DEVICES=0
PYTHONUNBUFFERED=1 python -m src.eval_checkpoints \
    --checkpoint_dir ./checkpoints/r-grpo_train/qwen3-1.7b_baseline_rollout16_response4096_seed2 \
    --eval_dir ./checkpoints/r-grpo_eval \
    --verl_dir ./verl \
    --target_step 1040 \
    --wandb_project r-grpo_eval \
    --gpus 4 \
    --max_attempts 1
