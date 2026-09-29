#!/usr/bin/env bash
# M0 baseline: verl GRPO + LoRA + 8bit optimizer, Qwen2.5-1.5B-Instruct, single RTX 3060 (6GB).
#
# 必须经 GPU 锁运行:
#   ~/tools/bin/with-lock gpu 1800 -- bash bench/m0/run_grpo.sh
# 每 5 步存一次 checkpoint 到 ~/tools/rheo-checkpoints/$EXP（防断电；
# verl 的 resume_mode=auto 会自动从最新 checkpoint 续跑，重跑同一命令即可）
#
# 显存策略(6GB 卡 + Windows 桌面占用约 1-2.5GB):
#   - 生成期 vLLM 独占 GPU(util≤0.72, 动态; KV ~1.2GB), util 太小时 KV 不足,
#     ~0.5GB, 生成陷入抢占-重算循环(实测 35min 跑不完一步)
#   - 训练期 vLLM sleep level 2 全量释放, FSDP 装载 3.1GB 主干
#   - lora.merge=true: 每步把 LoRA 合并进基础权重同步给 vLLM → sleep level 2
#     (默认 lora_as_adapter 模式只 sleep level 1, vLLM 保留 3.1GB 权重,
#      6GB 卡上 FSDP 加载训练必然爆显存 —— WDDM 报 "device not ready")
#   - actor: FSDP param/optimizer 全 offload, LoRA 冻结主干, bnb AdamW8bit
#   - 不加载 reference policy(GRPO 无 KL), 省一份 3GB 权重
#   - WSL2 不支持 CUDA IPC, 权重传输走宿主共享内存(VERL_DISABLE_CUDA_IPC=1,
#     需 verl/utils/device.py 本地补丁, 见 bench/results/m0-baseline/env.md)
set -euo pipefail

PYTHON=${PYTHON:-$HOME/tools/venvs/rheo/bin/python}
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL=${MODEL:-$HOME/models/Qwen2.5-1.5B-Instruct}
DATA_DIR=${DATA_DIR:-$HOME/datasets/rheo/gsm8k}
RESULTS="$REPO_DIR/bench/results/m0-baseline"
CKPT_DIR=${CKPT_DIR:-$HOME/tools/rheo-checkpoints}
STEPS=${STEPS:-60}
BATCH=${BATCH:-16}
MAX_RESP=${MAX_RESP:-512}
TEST_FREQ=${TEST_FREQ:-10}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-true}
EXP=${EXP:-grpo-lora-qwen25-1.5b}

# verl 约束：train_batch_size >= ppo_mini_batch_size(8)，且单卡下应为 8 的倍数
if [ "$BATCH" -lt 8 ] || [ $((BATCH % 8)) -ne 0 ]; then
  echo "[run_grpo] BATCH=$BATCH 无效：必须 >= 8 且为 8 的倍数（ppo_mini_batch_size=8）" >&2
  exit 64
fi

mkdir -p "$RESULTS/logs" "$CKPT_DIR"

# ray/vllm 在 WSL2 下的稳定性开关
export RAY_memory_monitor_refresh_ms=0
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL=WARNING
export HYDRA_FULL_ERROR=1
export TENSORBOARD_DIR="$RESULTS/tb/$EXP"
# WSL2 不支持 CUDA IPC：verl 权重传输走宿主共享内存
# （需要 ~/tools/venvs/rheo 中 verl/utils/device.py 的本地补丁，见 env.md）
export VERL_DISABLE_CUDA_IPC=1
# WSL2/WDDM：vLLM 进程 sleep 释放数 GB 会破坏训练进程的经典缓存分配器
# （CUDACachingAllocator INTERNAL ASSERT），改用 VMM expandable segments 并
# 禁止 verl 运行时切换回经典池（device.py 本地补丁）
export PYTORCH_ALLOC_CONF=expandable_segments:True
export RHEO_KEEP_EXPANDABLE=1
# RheoTrace 插桩（TASK-A step 3）：置 1 时每个 ray worker 进程经 sitecustomize
# 加载 bench/m0/rheo_trace_hooks.py，事件落 $RHEO_TRACE_DIR/spill-<pid>.jsonl，
# 训练结束后用 bench/m0/merge_trace.py 合并 + validate
# spill 目录生命周期：全新跑之前清空它；断点续跑时保留（merge 需要完整历史）
export RHEO_TRACE=${RHEO_TRACE:-0}
export RHEO_TRACE_HOOKS="$REPO_DIR/bench/m0/rheo_trace_hooks.py"
export RHEO_TRACE_DIR=${RHEO_TRACE_DIR:-$RESULTS/traces-spill}

# rollout util 按启动时实际空闲显存动态算（Windows 桌面占用会波动，vLLM 0.12
# 启动时检查 free < util*total 直接拒绝）。边距 0.6GB；若 vLLM 仍报 Free memory
# 不足则降 0.06 重试，最低 0.55（KV 会小、生成会慢，但能跑）。
UTIL=$(RHEO_TRACE=0 "$PYTHON" -c "import torch; f,t=torch.cuda.mem_get_info(0); f/=2**30; t/=2**30; print(f'{min(0.72, max(0.55, (f-0.6)/t)):.2f}')")
FREE=$(RHEO_TRACE=0 "$PYTHON" -c "import torch; print(f'{torch.cuda.mem_get_info(0)[0]/2**30:.2f}')")

run_training() {
  "$PYTHON" -m verl.trainer.main_ppo "$@"
}
# calculate_log_probs=true: vLLM 返回 token logprobs（trace 带 token_logprob，W03 消失）；
# 默认 decoupled 模式下 actor 仍重算 old_log_probs，训练语义不变

while true; do
  echo "[run_grpo] attempt util=$UTIL (free_gb=$FREE)"
  ATTEMPT_LOG="$RESULTS/logs/attempt-$(date +%H%M%S)-$$.log"
  rc=0
  run_training \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/test.parquet" \
  data.train_batch_size="$BATCH" \
  data.max_prompt_length=256 \
  data.max_response_length="$MAX_RESP" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.lora_rank=16 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.target_modules=all-linear \
  ++actor_rollout_ref.model.lora.merge=true \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.optim.lr=1e-4 \
  actor_rollout_ref.actor.optim.optimizer_impl=bitsandbytes.optim \
  actor_rollout_ref.actor.optim.optimizer=AdamW8bit \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.calculate_log_probs=true \
  actor_rollout_ref.rollout.gpu_memory_utilization="$UTIL" \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.max_model_len=1024 \
  actor_rollout_ref.rollout.max_num_seqs=48 \
  custom_reward_function.path="$REPO_DIR/bench/m0/gsm8k_reward.py" \
  custom_reward_function.name=compute_score \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=1 \
  trainer.total_training_steps="$STEPS" \
  trainer.total_epochs=10 \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.save_freq="${SAVE_FREQ:-5}" \
  trainer.logger='[console,tensorboard]' \
  trainer.project_name=rheo-m0 \
  trainer.experiment_name="$EXP" \
  trainer.default_local_dir="$CKPT_DIR/$EXP" > "$ATTEMPT_LOG" 2>&1 || rc=$?
  tail -40 "$ATTEMPT_LOG"
  if [ "$rc" -eq 0 ]; then
    echo "[run_grpo] training finished ok"
    break
  fi
  echo "[run_grpo] attempt failed (rc=$rc)" >&2
  # 只对 vLLM 显存检查失败降 util 重试；其他错误（配置/代码）直接失败
  if ! grep -q "Free memory on device" "$ATTEMPT_LOG"; then
    echo "[run_grpo] not a free-memory failure; giving up (full log: $ATTEMPT_LOG)" >&2
    exit 1
  fi
  if [ "$UTIL" = "0.55" ]; then
    echo "[run_grpo] util already at floor 0.55 and still failing; giving up" >&2
    exit 1
  fi
  UTIL=$(RHEO_TRACE=0 "$PYTHON" -c "print(f'{max(0.55, $UTIL - 0.06):.2f}')")
  # 清掉残留 ray 集群，避免下次 attempt 挂到死 worker 上
  "$PYTHON" -m ray stop --force >/dev/null 2>&1 || true
  sleep 10
done

