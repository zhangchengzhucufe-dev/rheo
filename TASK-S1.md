# TASK-S1：调度引擎 v1（GPU 唯一持有者，关键路径枢纽）

> 你的 worktree：`~/rheo-s1`（分支 `feat/scheduler-v1`）
> 开工先读：`PLAN.md` §1 L1/L2、§2 迭代6、§4 M2、**§9 硬件事实** + 本文件。
> 跨板块问题记 `docs/issues.md`，不顺手实现。

## 阶段 1 留给你的资产

- A 的 verl 集成经验：`bench/m0/rheo_trace_hooks.py`（hooks 模式）、`bench/results/m0-baseline/env.md`（3 个 verl 补丁的 diff 说明）、`bench/results/m0-baseline/RESUME.md`（运行手册：with-lock、断点续跑、坑清单）——**动手集成前必读**
- 真实 trace：`bench/traces/m0-baseline.rheotrace.jsonl`（A 的 PR 合入后出现在 main）
- 指标口径已冻结：`docs/metrics-v0.md`；trace 格式已冻结：`docs/rheotrace-spec-v0.md`（§8 只允许向后兼容增量）

## 目标

M2 前半：组感知调度 + DAPO 动态采样引擎化，设计上为 M3/M4 留好接口。**你是 S2（仿真）和 S4（网关）的接口供应商，第 1 天的设计文档最优先。**

## 任务（按序）

1. **D1：`docs/scheduler-design-v0.md` 定稿发 PR**，冻结三件事：
   - **策略接口**：`policy(observation) → decision`，observation 含在途轨迹段/组状态/显存水位，decision ∈ {continue, pause, re-prefill, abort(group, reason)}——S2 仿真器实现的就是这个接口
   - **轨迹状态机扩展**：token 边界 pause/resume 原语的函数签名（M3 才实现本体，v1 只留接口位）
   - **成本模型**：continue | shadow | ε-stale | re-prefill 四分支的判定输入表——**v1 只激活 continue 与 re-prefill（含 DAPO abort）**，shadow/ε-stale 留空位不实现
   - 附 6GB 显存预算表（0.5B 默认档；引用 PLAN §9）
2. **实现 `runtime/scheduler.py`**：组感知调度（同组 prompt 共批）、DAPO 零方差组 abort（带 reason 落 trace，废 token 可对账）、token 边界暂停原语 stub。纯逻辑配 mock 单测。
3. **等 `feat/m0-baseline` 合入 main 后 rebase**，把 scheduler 接进 verl rollout 路径（沿用 A 的 hooks 模式，不改 verl 本体）。
4. **真机 A/B**（0.5B 默认档，`~/tools/bin/with-lock gpu 1800 --` 包锁）：用 S3 的负载配置跑 scheduler on/off 对比；trace 落 `bench/traces/`，结果落 `bench/results/scheduler-v1/`，用 `python -m bench.analysis` 出对比报告。

## 验收

- [ ] 设计文档 D1 定稿 merge（S2/S4 被你解锁）
- [ ] scheduler 核心单测绿（四分支判定输入有覆盖）
- [ ] DAPO abort 真机生效：零方差组被引擎侧中止，废 token 率能在 trace 里对账
- [ ] A/B 初步数据落盘（方向性即可；+30–50% 是 M2 全期验收，不压本阶段）
- [ ] `ruff check .` + `pytest` 过

## 不做

WeightManager/KVT/kernels 实现（M3/M4）；shadow/ε-stale 实现（只留接口）；改 rheotrace spec（需求走 issues.md）。
提前完工 → 把 A/B 实验扩到 1.5B 档，或帮 S4 做 mock 联调。
