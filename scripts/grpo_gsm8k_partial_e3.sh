#!/bin/bash
# GRPO training configuration with partial e3 components for GSM8K.
# "Partial e3" = negative gradients + loose clipping, but NO curriculum.
# This replicates Chung's original setup with controlled data (M rows).
#
# Differs from grpo_gsm8k_a100.sh in two parameters:
#   - clip_ratio_high: 0.28 -> 0.5 (loose upper clipping)
#   - only_train_on_positive: True -> False (negative gradients enabled)
#
# Required env vars:
#   TRAIN_PARQUET    absolute path to training parquet
#   VAL_PARQUET      absolute path to clean GSM8K test parquet
#   BASE_MODEL       HF id or local path to base model
#   CKPT_DIR         absolute path under Modal Volume for checkpoints
#   EXPERIMENT_NAME  wandb experiment name
#   TOTAL_STEPS      total training steps (default 400)
#
# Optional:
#   WANDB_PROJECT    defaults to "cs224r-trivia-gsm8k"
#   SAVE_FREQ        defaults to 100
#   TEST_FREQ        defaults to 25

set -euo pipefail

: "${TRAIN_PARQUET:?TRAIN_PARQUET is required}"
: "${VAL_PARQUET:?VAL_PARQUET is required}"
: "${BASE_MODEL:?BASE_MODEL is required}"
: "${CKPT_DIR:?CKPT_DIR is required}"
: "${EXPERIMENT_NAME:?EXPERIMENT_NAME is required}"
: "${TOTAL_STEPS:?TOTAL_STEPS is required}"

WANDB_PROJECT="${WANDB_PROJECT:-cs224r-trivia-gsm8k}"
SAVE_FREQ="${SAVE_FREQ:-100}"
TEST_FREQ="${TEST_FREQ:-25}"

echo "[grpo_gsm8k_partial_e3] TRAIN_PARQUET=$TRAIN_PARQUET"
echo "[grpo_gsm8k_partial_e3] VAL_PARQUET=$VAL_PARQUET"
echo "[grpo_gsm8k_partial_e3] BASE_MODEL=$BASE_MODEL"
echo "[grpo_gsm8k_partial_e3] CKPT_DIR=$CKPT_DIR"
echo "[grpo_gsm8k_partial_e3] EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "[grpo_gsm8k_partial_e3] TOTAL_STEPS=$TOTAL_STEPS"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_PARQUET" \
    data.val_files="$VAL_PARQUET" \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    +actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.5 \
    actor_rollout_ref.actor.only_train_on_positive=False \
    actor_rollout_ref.actor.remove_truncated=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=verl/utils/reward_score/gsm8k_custom.py \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.total_training_steps="$TOTAL_STEPS" \
    trainer.total_epochs=20 \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.resume_mode=auto \
    "${@}"
