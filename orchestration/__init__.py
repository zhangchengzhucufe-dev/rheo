"""L2 编排层。

Rollout API（submit / sync_weights / collect / abort）、Env Gateway（工具调用拦截 +
KV 分级留存）、Staleness 控制器、Drift 估计器、遥测（落 RheoTrace）。见 PLAN.md §1 L2。
"""
