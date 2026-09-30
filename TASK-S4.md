# TASK-S4：Env Gateway 与 env-wait KV 分级（纯 CPU）

> 你的 worktree：`~/rheo-s4`（分支 `feat/env-gateway-v0`）
> 开工先读：`PLAN.md` §1 L2（Env Gateway）、§2 迭代5、§4 M2 + 本文件。跨板块问题记 `docs/issues.md`。

## 阶段 1 留给你的资产

- 轨迹状态机与事件语义：`docs/rheotrace-spec-v0.md`（env-wait 事件、segment 生命周期已定义）
- 合成工具延迟负载：S3 的 agent-tool 工作负载（D1 规格定稿后可用其延迟分布参数当测试夹具）
- 指标口径：`docs/metrics-v0.md`（env-wait 停顿的归类方式）

## 目标

多轮工具调用时 KV 不能全留 HBM（爆）、丢弃则重算贵——把"分级留存 + 等待预测"做成有 mock 单测的骨架，为 M2 后半的真机集成备好料。

## 任务（按序）

1. **D1：`docs/env-gateway-design-v0.md` 定稿发 PR**：拦截点选择（工具调用异步等待处）、KV 分级判据（HBM → host pinned → 4bit 压缩 → 磁盘 checkpoint）、等待预测接口、与 S1 轨迹状态机的 pause/resume 对齐（S1 设计文档同日定稿，互审）。
2. **`orchestration/env_gateway.py` 骨架**：分级策略 + 可插拔成本模型（成本输入：等待时长估计、KV 段大小、各级迁移带宽）；mock KV block 单测——给定（等待估计, 显存压力）→ 分层决策，覆盖全部迁移路径与边界（显存满、等待超时、轨迹 abort）。
3. **等待时长预测 v0**：按工具 EWMA；用 S3 的合成延迟数据验证收敛性并配测试。
4. （时间允许）与 S2 联调：env-wait 事件的分级决策在仿真器里可回放。

## 验收

- [ ] 设计文档 D1 定稿（与 S1 状态机对齐无矛盾）
- [ ] mock 单测绿：分级决策覆盖全部迁移路径 + 边界条件
- [ ] 预测器在合成延迟数据上收敛且有测试
- [ ] `ruff check .` + `pytest` 过

## 不做

不接真机引擎（集成排 M2 后半）；不改冻结 spec——需要新字段（比如 KV 段分级标记）就记 `docs/issues.md`，由 S2 走 spec §8 增量。
提前完工 → 预研"组内跨 env 调用连续批处理"（迭代 5 后半）的设计文档。
