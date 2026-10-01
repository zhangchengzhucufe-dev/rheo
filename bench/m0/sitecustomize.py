"""[rheo M0 local patch] Opt-in instrumentation bootstrap for Ray worker processes.

Ray actors don't re-execute the user's shell profile, so a plain env-var hook in
run_grpo.sh would only reach the driver. sitecustomize runs in EVERY python
process started from this venv (including Ray actors spawned with the spawn
method), which is exactly what the RheoTrace instrumentation needs.
No-op unless RHEO_TRACE=1 and RHEO_TRACE_HOOKS points at the hooks module.
"""

import os
import sys

if os.environ.get("RHEO_TRACE") == "1" and sys.argv[0] != "-c":
    # -c 一次性进程（如 run_grpo.sh 里算显存的子命令）不装钩子，
    # 否则钩子的 stdout 会污染 $() 命令替换
    _p = os.environ.get("RHEO_TRACE_HOOKS", "")
    if _p and os.path.exists(_p):
        try:
            import importlib.util
            import sys

            _spec = importlib.util.spec_from_file_location("rheo_trace_hooks", _p)
            _mod = importlib.util.module_from_spec(_spec)
            # 先注册进 sys.modules：钩子内部经 sys.modules 取 ContextVar，
            # 保证 ray 序列化 actor 类时闭包不携带不可 pickle 的对象
            sys.modules["rheo_trace_hooks"] = _mod
            _spec.loader.exec_module(_mod)
        except Exception:
            import traceback

            traceback.print_exc()
