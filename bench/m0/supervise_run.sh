#!/usr/bin/env bash
# 无人值守监督循环：run_grpo.sh 内部有 6 次 attempt 重试；
# 本脚本在其 6 次用尽仍失败时整批重启（每批之间等 VRAM/驱动状态恢复），
# 直到训练成功（rc=0）或达到 MAX_BATCHES。
# 用法: nohup 经 ZCode 后台任务运行；断电后重跑同命令。
set -u

MAX_BATCHES=${MAX_BATCHES:-20}
i=0
while [ "$i" -lt "$MAX_BATCHES" ]; do
  i=$((i + 1))
  echo "[supervisor] batch $i/$MAX_BATCHES start $(date '+%F %T')"
  # 清理上批可能泄漏的锁（若持有者已死）—— with-lock 自动接管要等 2 小时
  lf=/mnt/c/Users/15985/tools/.locks/gpu.lock
  if [ -f "$lf" ] && ! ps -p "$(head -1 "$lf" | awk '{print $1}')" >/dev/null 2>&1; then
    rm -f "$lf"
    echo "[supervisor] removed stale gpu.lock"
  fi
  RHEO_TRACE=1 EXP=grpo-lora-qwen25-1.5b STEPS=40 TEST_FREQ=10 VAL_BEFORE_TRAIN=false \
    "$HOME/tools/bin/with-lock" gpu 1800 -- bash "$HOME/rheo-a/bench/m0/run_grpo.sh"
  rc=$?
  echo "[supervisor] batch $i exit rc=$rc $(date '+%F %T')"
  if [ "$rc" -eq 0 ]; then
    echo "[supervisor] training finished successfully"
    exit 0
  fi
  # 失败后等驱动/VRAM 状态恢复；checkpoint 已存的进度不会丢
  sleep 120
done
echo "[supervisor] batch budget exhausted"
exit 1
