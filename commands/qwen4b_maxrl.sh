#!/bin/bash
#SBATCH --job-name=r-grpo_base
#SBATCH --partition=beta
#SBATCH --nodes=1
#SBATCH --qos=long
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=144
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err



#export WANDB_RUN_ID="5opv9xg2"
#export WANDB_RESUME="must"


export PROJECT_ROOT=$(pwd)
export PYTHONPATH=$(pwd)/src:$PYTHONPATH


export VENV_LIB="${PROJECT_ROOT}/beta_venv/lib/python3.12/site-packages"
export CUDA_HOME="$VENV_LIB/nvidia/cu13"
export PATH="${PROJECT_ROOT}/beta_venv/bin:$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/include:$CPATH"
export LD_LIBRARY_PATH="$VENV_LIB/torch/lib:$VENV_LIB/nvidia/cuda_runtime/lib:$CUDA_HOME/lib:$LD_LIBRARY_PATH"




export MODEL_PATH="Qwen/Qwen3-4B-Base"
export MAX_ATTEMPTS=1
export VERL_RETRY_PROMPT="Your answer is wrong. Reflect on your previous answer and try something else."

export RESPONSE_LEN=4096
export PROMPT_LEN=1024
export ROLLOUT_NUM=16

export SEED=42

# 1. Activate the virtual environment FIRST
source beta_venv/bin/activate


# 3. Now use Ray (it will use the healthy version inside beta_venv)
ray stop --force
pkill -f "ray"
ray start --head --num-gpus=4

unset ROCR_VISIBLE_DEVICES
 
PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=maxrl \
    data.train_files=$HOME/data/polaris/train.parquet \
    data.val_files=$"['$HOME/data/aime24/test.parquet','$HOME/data/aime25/test.parquet','$HOME/data/math-500/test.parquet']" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.train_batch_size=256 \
    data.max_prompt_length=$PROMPT_LEN \
    data.max_response_length=$RESPONSE_LEN \
    data.seed=$SEED \
    custom_reward_function.path=$(pwd)/src/decay_reward.py \
    custom_reward_function.name=compute_decay_reward \
    reward.reward_manager.name=naive \
    reward.num_workers=8 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.grad_clip=0.3 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    ++actor_rollout_ref.max_model_len=$((PROMPT_LEN + RESPONSE_LEN)) \
    ++actor_rollout_ref.rollout.free_cache_engine=True \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.enable_sleep_mode=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=$ROLLOUT_NUM \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.model.use_liger=True \
    algorithm.use_kl_in_reward=False \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.log_val_generations=128 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=r-grpo_train \
    trainer.experiment_name=qwen3-4b_maxrl_rollout${ROLLOUT_NUM}_response${RESPONSE_LEN}_seed${SEED} \
    trainer.total_epochs=5 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=$(pwd)/src/custom_agents.yaml \
    ++actor_rollout_ref.rollout.agent.default_agent_loop=multi_attempt \
    2>&1 | tee verl_demo.log
