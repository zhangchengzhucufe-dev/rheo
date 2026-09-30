#!/usr/bin/env bash
# 无人值守监督循环（TASK-A2 E18/G6/D13）：
#   - run_grpo.sh 内部有失败分类 + 降档梯子；本脚本在其放弃后整批重启
#   - PID 文件管理：防双开，停止用 bench/m0/stop_supervisor.sh（kill $(cat pidfile)）
#   - 每批开始前 df 预检 scratch 盘，低于阈值显式报 ENOSPC 并停（不烧钱）
#   - 成功出口可挂自动关机（G6）：AUTO_SHUTDOWN=1 时 rc=0 后 shutdown（AutoDL 实例用）
# 用法: nohup 经 ZCode 后台任务 / 云端 nohup 运行；断电后重跑同命令。
set -u

MAX_BATCHES=${MAX_BATCHES:-20}
SCRATCH=${SCRATCH:-$HOME/rheo-scratch}
MIN_FREE_GB=${MIN_FREE_GB:-20}
PID_FILE=${PID_FILE:-$SCRATCH/supervisor.pid}
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

mkdir -p "$SCRATCH"

# ---- E18: PID 文件防双开 ----
if [ -f "$PID_FILE" ] && ps -p "$(cat "$PID_FILE")" >/dev/null 2>&1; then
  echo "[supervisor] 已有实例在跑 (pid $(cat "$PID_FILE"))，退出。停止请用: kill \$(cat $PID_FILE)" >&2
  exit 1
fi
echo $$ > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT

i=0
while [ "$i" -lt "$MAX_BATCHES" ]; do
  i=$((i + 1))

  # ---- D13: scratch 盘 ENOSPC 预检 ----
  FREE_KB=$(df -Pk "$SCRATCH" | awk 'NR==2 {print $4}')
  FREE_GB=$((FREE_KB / 1048576))
  if [ "$FREE_GB" -lt "$MIN_FREE_GB" ]; then
    echo "[supervisor] ENOSPC: 训练需要约 ${MIN_FREE_GB}GB scratch 空间，当前仅 ${FREE_GB}GB 可用（$SCRATCH）。请清理后重跑。" >&2
    exit 1
  fi

  echo "[supervisor] batch $i/$MAX_BATCHES start $(date '+%F %T') (scratch free ${FREE_GB}GB)"

  # 本地 WSL：清理上批可能泄漏的 gpu 锁（持有者已死时）——with-lock 自动接管要等 2 小时
  lf=/mnt/c/Users/15985/tools/.locks/gpu.lock
  if [ -d /mnt/c ] && [ -f "$lf" ] && ! ps -p "$(head -1 "$lf" | awk '{print $1}')" >/dev/null 2>&1; then
    rm -f "$lf"
    echo "[supervisor] removed stale gpu.lock"
  fi

  # with-lock 只在本地 WSL 存在；云端原生 Linux 直接跑
  WITH_LOCK_BIN=""
  if [ -n "${WITH_LOCK:-}" ]; then WITH_LOCK_BIN="$WITH_LOCK"
  elif [ -x "$HOME/tools/bin/with-lock" ]; then WITH_LOCK_BIN="$HOME/tools/bin/with-lock"
  elif command -v with-lock >/dev/null 2>&1; then WITH_LOCK_BIN="$(command -v with-lock)"
  fi

  if [ -n "$WITH_LOCK_BIN" ]; then
    RHEO_TRACE=1 EXP=grpo-lora-qwen25-1.5b STEPS=40 TEST_FREQ=10 VAL_BEFORE_TRAIN=false \
      "$WITH_LOCK_BIN" gpu 1800 -- bash "$REPO_DIR/bench/m0/run_grpo.sh"
  else
    RHEO_TRACE=1 EXP=grpo-lora-qwen25-1.5b STEPS=40 TEST_FREQ=10 VAL_BEFORE_TRAIN=false \
      bash "$REPO_DIR/bench/m0/run_grpo.sh"
  fi
  rc=$?
  echo "[supervisor] batch $i exit rc=$rc $(date '+%F %T')"

  if [ "$rc" -eq 0 ]; then
    echo "[supervisor] training finished successfully (pid file removed)"
    # ---- G6: 成功出口自动关机（AutoDL 实例置 AUTO_SHUTDOWN=1 生效）----
    if [ "${AUTO_SHUTDOWN:-0}" = "1" ]; then
      echo "[supervisor] AUTO_SHUTDOWN=1 → shutdown in 60s"
      sleep 60
      shutdown now
    fi
    exit 0
  fi
  # 失败后等驱动/VRAM 状态恢复；checkpoint 已存的进度不会丢
  sleep 120
done
echo "[supervisor] batch budget exhausted"
exit 1
