#!/usr/bin/env bash
# 预检脚本（TASK-A2 A2/A3/F19）：import 冒烟 + 版本矩阵断言 + 路径/磁盘检查。
# 训练前跑一次，90 秒内发现 90% 的环境问题；全绿再启动训练。
set -u
FAIL=0

# 找对 python：优先已激活的 venv（PATH 上的 python 含 torch 则用它），
# 否则回退到已知 venv 路径
if python -c "import torch" >/dev/null 2>&1; then
  PYTHON=python
elif [ -x "$HOME/tools/venvs/rheo/bin/python" ]; then
  PYTHON="$HOME/tools/venvs/rheo/bin/python"
else
  PYTHON=python
fi

ok() { echo "  ✓ $1"; }
bad() { echo "  ✗ $1"; FAIL=1; }

echo "[doctor] python: $($PYTHON -V 2>&1)"
case "$($PYTHON -V 2>&1)" in
  *3.12*|*3.11*) ok "python 版本在支持矩阵内" ;;
  *) bad "python 版本不在支持矩阵（需 3.11/3.12）" ;;
esac

echo "[doctor] import 冒烟"
for mod in torch verl vllm transformers bitsandbytes ray; do
  "$PYTHON" -c "import $mod" 2>/dev/null && ok "import $mod" || bad "import $mod 失败"
done

echo "[doctor] 版本矩阵"
"$PYTHON" - <<'EOF' || FAIL=1
import importlib.metadata as md
expect = {"torch": "2.9.0", "vllm": "0.12.0", "verl": "0.8.0"}
for pkg, want in expect.items():
    got = md.version(pkg)
    print(f"  {'✓' if got == want else '✗'} {pkg} {got} (期望 {want})")
    assert got == want, f"{pkg} 版本漂移: {got} != {want}"
hub = md.version("huggingface-hub")
print(f"  {'✓' if hub.startswith('0.') else '✗'} huggingface-hub {hub} (须 0.x，1.x 与 transformers<5 不兼容)")
assert hub.startswith("0.")
EOF

echo "[doctor] GPU"
"$PYTHON" - <<'EOF' || FAIL=1
import torch
assert torch.cuda.is_available(), "CUDA 不可用"
print(f"  ✓ {torch.cuda.get_device_name(0)}")
free, total = torch.cuda.mem_get_info(0)
print(f"  ✓ 显存空闲 {free/2**30:.1f}/{total/2**30:.1f} GiB")
EOF

echo "[doctor] 插桩引导（sitecustomize 经 PYTHONPATH）"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# 注意：检查必须走脚本文件而非 python -c——sitecustomize 的门控会正确地
# 跳过 -c 一次性进程（防止污染 run_grpo 的命令替换），生产路径不受影响
CHECK="$(mktemp /tmp/rheo-doctor-XXXX.py)"
cat > "$CHECK" <<'PY'
import sys
assert "rheo_trace_hooks" in sys.modules, "钩子未被 sitecustomize 加载"
print("ok")
PY
RHEO_TRACE=1 RHEO_TRACE_HOOKS="$REPO_DIR/bench/m0/rheo_trace_hooks.py"   PYTHONPATH="$REPO_DIR/bench/m0${PYTHONPATH:+:$PYTHONPATH}"   "$PYTHON" "$CHECK" | grep -q "^ok$" && ok "sitecustomize → rheo_trace_hooks 加载正常" || bad "sitecustomize 未加载钩子（检查 PYTHONPATH/RHEO_TRACE_HOOKS）"
rm -f "$CHECK"

echo "[doctor] 数据/模型路径"
for p in "${MODEL:-$HOME/models/Qwen2.5-1.5B-Instruct}/config.json" \
         "${DATA_DIR:-$HOME/datasets/rheo/gsm8k}/train.parquet" \
         "${DATA_DIR:-$HOME/datasets/rheo/gsm8k}/test.parquet"; do
  [ -f "$p" ] && ok "$p" || bad "缺文件: $p（MODEL/DATA_DIR 环境变量可覆盖路径）"
done

echo "[doctor] 磁盘"
SCRATCH_DIR="${SCRATCH:-$HOME/rheo-scratch}"
mkdir -p "$SCRATCH_DIR"
FREE_GB=$(df -Pk "$SCRATCH_DIR" 2>/dev/null | awk 'NR==2 {print int($4/1048576)}')
[ "${FREE_GB:-0}" -ge 20 ] && ok "scratch 盘空闲 ${FREE_GB}GB" || bad "scratch 盘仅 ${FREE_GB:-0}GB（需 ≥20GB）"

if [ "$FAIL" -eq 0 ]; then echo "[doctor] ALL GREEN —— 可以启动训练"; else echo "[doctor]存在问题，先修复上列 ✗ 项"; exit 1; fi
