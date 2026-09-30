# TASK-S2：离散事件仿真器（纯 CPU）

> 你的 worktree：`~/rheo-s2`（分支 `feat/sim-v0`）
> 开工先读：`PLAN.md` §1 L1、§2 迭代6、§4 M2 + 本文件。跨板块问题记 `docs/issues.md`。

## 阶段 1 留给你的资产

- `rheotrace` 包（read/iread，gz 透明解压）——回放的输入层就是它
- 真实 trace：`bench/traces/m0-baseline.rheotrace.jsonl`（A 的 PR 合入后）；合成 trace 3 份已在 `bench/traces/synthetic/`
- 生成器可造任意负载：`python -m rheotrace gen --help`（grpo / bimodal / agent 预置）

## 目标

把"调度策略离线验证"变成生产力：真机实验前，策略先在仿真器里筛一遍。**你是仿真结论的第一责任人，同时接手 rheotrace spec §8 的增量维护。**

## 任务（按序）

1. **回放核心**（不依赖 S1，立即开工）：`sim/` 离散事件驱动，读 RheoTrace 事件流重演；集群参数化（n_gpu / 显存 / PCIe 带宽）；确定性可复现（seed）。用真实 m0 trace 校准，墙钟误差与原因写进文档。
2. **策略 harness**（等 S1 设计文档 D1 合入后）：按 S1 冻结的 `policy(observation) → decision` 接口实现三个策略——FIFO 基线、组感知、错峰批（staggered batch）。
3. **实验 + 报告** `docs/sim-study-v0.md`：同一批 trace（真实 + 合成长尾 + agent env-wait）上对比策略：吞吐、停顿分解、长尾掉队份额。至少产出一个"仿真先证明、待真机复核"的调度结论给 S1。
4. **接手 rheotrace §8 增量**：S4 等会话的字段需求从 `docs/issues.md` 领，走 spec §8 向后兼容增量流程（改 spec + 改 validator/writer/reader + 测试，一个 PR 内完成）。

## 验收

- [ ] 回放真实 trace 复现墙钟（校准误差量化记录）
- [ ] ≥3 策略在同一 trace 集上可比，出对比表
- [ ] ≥1 个有数据支撑的调度结论（标注"待真机复核"）
- [ ] CI 绿（纯 CPU）

## 不做

不碰 GPU；不发明新事件语义（一律走 §8 增量）；不实现调度器本体（那是 S1 的，你只做 harness 和策略实现）。
提前完工 → 给 sim 加"云规模外推"模式（8×A800 配置下仿真阶段②末的爆发实验，给真机租卡决策提供依据）。
