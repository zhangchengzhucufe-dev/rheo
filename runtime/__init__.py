"""L1 Runtime 层。

WeightManager（双缓冲热切换 + token 边界安全点，M3）、KVManager（版本感知选择性失效，M4）、
Trajectory 状态机、Scheduler（continue | shadow | ε-stale | re-prefill 成本模型）。

当前进度：`scheduler.py` 已落地 v1（组感知调度 + DAPO 零方差组 abort + token 边界暂停 stub），
接口冻结见 `docs/scheduler-design-v0.md`；WeightManager/KVManager 为 M3/M4 占位（PLAN §1 L1）。
"""
