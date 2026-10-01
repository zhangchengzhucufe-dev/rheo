"""离散事件仿真器（TASK-S2 #1）：读 RheoTrace 事件流重演，调度策略离线验证。

公开入口：
- ``load_workload(path)``        — trace → 按 step 分组的段作业
- ``run_step_barrier(wl, cfg)``  — step 屏障回放，产出墙钟与停顿分解
- ``calibrate(trace_path)``      — 用真实 trace 拟合集群参数并量化误差
- ``ClusterConfig``              — 集群参数化（n_gpu / 吞吐 / 同步停顿 / seed）
"""

from .calibrate import Calibration, calibrate, fit
from .cluster import ClusterConfig
from .engine import SimResult, StepRecord, assign_fifo_greedy, run_step_barrier
from .workload import SegmentJob, Step, Workload, load_workload

__all__ = [
    "Calibration",
    "ClusterConfig",
    "SegmentJob",
    "SimResult",
    "Step",
    "StepRecord",
    "Workload",
    "assign_fifo_greedy",
    "calibrate",
    "fit",
    "load_workload",
    "run_step_barrier",
]
