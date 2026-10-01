#!/usr/bin/env bash
# M0 baseline: verl GRPO + LoRA + 8bit optimizer, Qwen2.5-1.5B-Instruct, single GPU.
#
# 必须经 GPU 锁运行（本地 WSL 示例）:
#   ~/tools/bin/with-lock gpu 1800 -- bash bench/m0/run_grpo.sh
# 云端（AutoDL 等）直接跑 supervise_run.sh，见 bench/m0/CLOUD_GUIDE.md
#
# 可移植性（TASK-A2 B6/B7/F20）: 无私人绝对路径——REPO_DIR 从脚本位置推导，
# PYTHON 默认取 PATH 上的 python（建议先激活 venv），sitecustomize 经
# PYTHONPATH 零拷贝加载（不再需要 cp 进 site-packages）。
#
# 失败分类（TASK-A2 E15/C11）:
#   - 导入/配置/路径类异常 → 立即终止并高亮根因，不烧重试
#   - OOM/引擎启动类       → 降档重试，梯子 = util↓ → micro 减半 → max_num_seqs 减半
#   - WDDM/驱动瞬态        → 同档重试
set -euo pipefail

PYTHON=${PYTHON:-python}
# PATH 上的 python 没有 torch（venv 未激活）时，回退到已知 venv 路径，
# 免得烧一次失败 attempt 才被 E15 拦下
if ! "$PYTHON" -c "import torch" >/dev/null 2>&1 && [ -x "$HOME/tools/venvs/rheo/bin/python" ]; then
  PYTHON="$HOME/tools/venvs/rheo/bin/python"
  echo "[run_grpo] PATH python 无 torch，回退到 $PYTHON"
fi
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL=${MODEL:-$HOME/models/Qwen2.5-1.5B-Instruct}
DATA_DIR=${DATA_DIR:-$HOME/datasets/rheo/gsm8k}

# scratch 盘统一变量（TASK-A2 D13）：checkpoint / 日志 / trace spill 都走这里
SCRATCH=${SCRATCH:-$HOME/rheo-scratch}
CKPT_DIR=${CKPT_DIR:-$SCRATCH/checkpoints}
LOG_DIR=${LOG_DIR:-$SCRATCH/logs}
SPILL_ROOT=${SPILL_ROOT:-$SCRATCH/spill}

STEPS=${STEPS:-40}
BATCH=${BATCH:-16}
MAX_RESP=${MAX_RESP:-512}
ROLLOUT_N=${ROLLOUT_N:-8}
MICRO=${MICRO:-2}          # TASK-A2 重跑规格：micro=2 安全档跑全程（G4）
UTIL=${UTIL:-0.55}         # TASK-A2 重跑规格：util=0.55 安全档
TEST_FREQ=${TEST_FREQ:-10}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-false}
EXP=${EXP:-grpo-lora-qwen25-1.5b}
SEQS=${SEQS:-48}

# verl 约束：train_batch_size >= ppo_mini_batch_size(8)，且单卡下应为 8 的倍数
if [ "$BATCH" -lt 8 ] || [ $((BATCH % 8)) -ne 0 ]; then
  echo "[run_grpo] BATCH=$BATCH 无效：必须 >= 8 且为 8 的倍数（ppo_mini_batch_size=8）" >&2
  exit 64
fi

mkdir -p "$CKPT_DIR" "$LOG_DIR"

export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL=WARNING
export HYDRA_FULL_ERROR=1
export TENSORBOARD_DIR="$SCRATCH/tb/$EXP"

# ---- WSL2 专属稳定性开关（原生 Linux 自动跳过）----
if grep -qi microsoft /proc/version 2>/dev/null; then
  export RAY_memory_monitor_refresh_ms=0
  # WSL2 不支持 CUDA IPC：verl 权重传输走宿主共享内存
  # （需要 venv 中 verl/utils/device.py 的本地补丁，见 bench/results/m0-baseline/env.md）
  export VERL_DISABLE_CUDA_IPC=1
  # WSL2/WDDM：vLLM sleep 释放数 GB 会破坏经典缓存分配器 → VMM expandable segments
  export PYTORCH_ALLOC_CONF=expandable_segments:True
  export RHEO_KEEP_EXPANDABLE=1
  echo "[run_grpo] WSL2 detected: WDDM workarounds enabled"
fi

# RheoTrace 插桩（TASK-A2 G1）：每个 run 一个 spill 子目录，续跑绝不清理
# sitecustomize 经 PYTHONPATH 零拷贝加载（bench/m0/sitecustomize.py）
export RHEO_TRACE=${RHEO_TRACE:-0}
export RHEO_TRACE_HOOKS="$REPO_DIR/bench/m0/rheo_trace_hooks.py"
export RHEO_TRACE_DIR="$SPILL_ROOT"
RUN_ID=${RHEO_RUN_ID:-r$(date +%Y%m%d_%H%M%S)}
export RHEO_RUN_ID="$RUN_ID"
export PYTHONPATH="$REPO_DIR/bench/m0${PYTHONPATH:+:$PYTHONPATH}"

run_training() {
  "$PYTHON" -m verl.trainer.main_ppo "$@"
}

ATTEMPT=0
LAST_SIG=""
while true; do
  ATTEMPT=$((ATTEMPT + 1))
  ATTEMPT_LOG="$LOG_DIR/attempt-${RUN_ID}-$(printf %03d "$ATTEMPT").log"
  # TASK-A2 F21：attempt 日志轮转，只留最近 10 份
  # 轮转：只留最近 10 份（|| true 防 set -e 在首次无匹配时杀死脚本）
  ls -t "$LOG_DIR"/attempt-*.log 2>/dev/null | tail -n +11 | xargs -r rm -f || true

  {
    # TASK-A2 G7：最终生效参数回显，永远在日志首行
    echo "[run_grpo] RUN_ID=$RUN_ID attempt=$ATTEMPT util=$UTIL micro=$MICRO seqs=$SEQS batch=$BATCH steps=$STEPS model=$MODEL ckpt=$CKPT_DIR/$EXP spill=$SPILL_ROOT/$RUN_ID"
  } > "$ATTEMPT_LOG"
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
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO" \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$MICRO" \
  actor_rollout_ref.rollout.calculate_log_probs=true \
  actor_rollout_ref.rollout.gpu_memory_utilization="$UTIL" \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.max_model_len=1024 \
  actor_rollout_ref.rollout.max_num_seqs="$SEQS" \
  custom_reward_function.path="$REPO_DIR/bench/m0/gsm8k_reward.py" \
  custom_reward_function.name=compute_score \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=1 \
  trainer.total_training_steps="$STEPS" \
  trainer.total_epochs=10 \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.save_freq="${SAVE_FREQ:-5}" \
  trainer.max_ckpt_to_keep="${MAX_CKPT_TO_KEEP:-2}" \
  trainer.logger='[console,tensorboard]' \
  trainer.project_name=rheo-m0 \
  trainer.experiment_name="$EXP" \
  trainer.default_local_dir="$CKPT_DIR/$EXP" >> "$ATTEMPT_LOG" 2>&1 || rc=$?
  tail -40 "$ATTEMPT_LOG"
  if [ "$rc" -eq 0 ]; then
    echo "[run_grpo] training finished ok (RUN_ID=$RUN_ID)"
    break
  fi
  echo "[run_grpo] attempt $ATTEMPT failed (rc=$rc)" >&2

  # ---- TASK-A2 E15/C11：失败分类 ----
  if grep -qE "ModuleNotFoundError|ImportError|ConfigCompositionException|not in struct|FileNotFoundError" "$ATTEMPT_LOG"; then
    echo "[run_grpo] FATAL: 导入/配置/路径类错误，重试无意义。根因：" >&2
    grep -m2 -E "ModuleNotFoundError|ImportError|ConfigCompositionException|not in struct|FileNotFoundError" "$ATTEMPT_LOG" >&2
    echo "[run_grpo] 完整日志: $ATTEMPT_LOG" >&2
    exit 1
  fi

  if grep -qE "Free memory on device|CUDA out of memory|OutOfMemoryError" "$ATTEMPT_LOG"; then
    # 降档梯子（TASK-A2 顺序）：util ↓ → micro 减半 → seqs 减半
    # 同签名（连续 OOM）快速跳档：一次降两档
    if [ "$UTIL" != "0.45" ]; then
      UTIL=$(RHEO_TRACE=0 "$PYTHON" -c "print(f'{max(0.45, $UTIL - 0.06):.2f}')")
    elif [ "$MICRO" -gt 1 ]; then
      MICRO=$((MICRO / 2))
    elif [ "$SEQS" -gt 12 ]; then
      SEQS=$((SEQS / 2))
    else
      echo "[run_grpo] OOM 梯子到底仍失败；放弃（完整日志: $ATTEMPT_LOG）" >&2
      exit 1
    fi
    if [ "$LAST_SIG" = "oom" ]; then
      if [ "$UTIL" != "0.45" ]; then UTIL=$(RHEO_TRACE=0 "$PYTHON" -c "print(f'{max(0.45, $UTIL - 0.06):.2f}')"); elif [ "$MICRO" -gt 1 ]; then MICRO=$((MICRO / 2)); elif [ "$SEQS" -gt 12 ]; then SEQS=$((SEQS / 2)); fi
    fi
    echo "[run_grpo] OOM → 降档至 util=$UTIL micro=$MICRO seqs=$SEQS" >&2
    LAST_SIG="oom"
  elif grep -qE "device not ready|INTERNAL ASSERT|invalid resource handle|Watchdog|ActorDiedError|CUDA error" "$ATTEMPT_LOG"; then
    echo "[run_grpo] 瞬态 CUDA/驱动/NCCL 失败；同档重试" >&2
    LAST_SIG=""
  else
    echo "[run_grpo] 未识别的失败，放弃（完整日志: $ATTEMPT_LOG）" >&2
    exit 1
  fi

  # 清掉残留 ray 集群，避免下次 attempt 挂到死 worker 上
  "$PYTHON" -m ray stop --force >/dev/null 2>&1 || true
  sleep 15
done
