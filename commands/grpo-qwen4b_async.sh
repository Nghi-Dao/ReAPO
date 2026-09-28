#!/bin/bash

export PROJECT_ROOT=$(pwd)
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

export MODEL_PATH="Qwen/Qwen3-4B-Base"
export MAX_ATTEMPTS=2
export VERL_RETRY_PROMPT="Your answer is wrong. Reflect on your previous answer and try something else."

export RESPONSE_LEN=8096
export PROMPT_LEN=1024
export ROLLOUT_NUM=8
export SEED=42

# --- Fully Async Resource & Configuration Tuning ---
export NNODES_TRAIN=1
export NGPUS_TRAIN=2

export NNODES_ROLLOUT=1
export NGPUS_ROLLOUT=6

# Mathematically aligned with 1040 global steps and 50 test freq
export STALENESS_THRESHOLD=0.5     
export TRIGGER_SYNC_STEP=1         # 256 / (1 * 256) = 1
export REQUIRE_BATCHES=1           
export PARTIAL_ROLLOUT="True"      
export TOTAL_ROLLOUT_STEPS=$((240 * 1100)) # train_batch_size * 1040 steps
export TEST_FREQ=50                # Validate every 50 syncs (which equals 50 global steps)
# ----------------------------------------------------

ray stop --force
pkill -f "ray"

unset ROCR_VISIBLE_DEVICES

PYTHONUNBUFFERED=1 python3.10 -m verl.experimental.fully_async_policy.fully_async_main \
    algorithm.adv_estimator=multi_attempt_grpo \
    data.train_files=$HOME/data/polaris/train.parquet \
    data.val_files=$"['$HOME/data/beyondaime/test.parquet','$HOME/data/math-500/test.parquet']" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.train_batch_size=0 \
    data.gen_batch_size=1 \
    data.return_raw_chat=True \
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
    actor_rollout_ref.actor.ppo_mini_batch_size=240 \
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
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.hybrid_engine=False \
    ++actor_rollout_ref.max_model_len=$((PROMPT_LEN + RESPONSE_LEN)) \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=$ROLLOUT_NUM \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.model.use_liger=True \
    algorithm.use_kl_in_reward=False \
    trainer.val_before_train=False \
    trainer.nnodes="${NNODES_TRAIN}" \
    trainer.n_gpus_per_node="${NGPUS_TRAIN}" \
    rollout.nnodes="${NNODES_ROLLOUT}" \
    rollout.n_gpus_per_node="${NGPUS_ROLLOUT}" \
    rollout.total_rollout_steps="${TOTAL_ROLLOUT_STEPS}" \
    ++rollout.test_freq="${TEST_FREQ}" \
    async_training.staleness_threshold="${STALENESS_THRESHOLD}" \
    async_training.trigger_parameter_sync_step="${TRIGGER_SYNC_STEP}" \
    async_training.require_batches="${REQUIRE_BATCHES}" \
    async_training.partial_rollout="${PARTIAL_ROLLOUT}" \
    trainer.save_freq=5 \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.log_val_generations=128 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=r-grpo \
    trainer.experiment_name=qwen3-4b_async_attempt${MAX_ATTEMPTS}_rollout${ROLLOUT_NUM}_response${RESPONSE_LEN}_seed${SEED} \
    trainer.total_epochs=5 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=$(pwd)/src/custom_agents.yaml \
    ++actor_rollout_ref.rollout.agent.default_agent_loop=async_multi_attempt \
    2>&1 | tee verl_demo.log
