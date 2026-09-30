#!/usr/bin/env bash
# 停止监督循环 + 训练（TASK-A2 E18）：kill $(cat pidfile) 方式，不用 pkill 模式匹配。
set -u
SCRATCH=${SCRATCH:-$HOME/rheo-scratch}
PID_FILE=${PID_FILE:-$SCRATCH/supervisor.pid}

if [ -f "$PID_FILE" ]; then
  PID=$(cat "$PID_FILE")
  if kill "$PID" >/dev/null 2>&1; then
    echo "[stop] supervisor (pid $PID) terminated"
  else
    echo "[stop] pid $PID not running; removing stale pidfile"
  fi
  rm -f "$PID_FILE"
else
  echo "[stop] no pidfile at $PID_FILE"
fi

# 训练与 ray 残留（精确匹配命令名，不用宽泛模式）
pkill -f "verl.trainer.main_ppo" 2>/dev/null && echo "[stop] trainer killed" || true
sleep 2
pkill -9 -f "verl.trainer.main_ppo" 2>/dev/null || true
ray stop --force >/dev/null 2>&1 || python -m ray stop --force >/dev/null 2>&1 || true
echo "[stop] done"
