"""L1 Runtime 层。

WeightManager（双缓冲热切换 + token 边界安全点）、KVManager（版本感知选择性失效）、
Trajectory 状态机、Scheduler（continue | shadow | ε-stale | re-prefill 成本模型）。
M2–M4 分阶段实现，当前为骨架占位（见 PLAN.md §1 L1）。
"""
