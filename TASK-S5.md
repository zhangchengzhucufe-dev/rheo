# TASK-S5：M1 测量研究 + 博客 #1（纯 CPU / 写作）

> 你的 worktree：`~/rheo-s5`（分支 `feat/study-m1`）
> 开工先读：`PLAN.md` §0（四大问题）、§4 M1、§5 + 本文件。跨板块问题记 `docs/issues.md`。

## 阶段 1 留给你的资产

- 分析流水线（已可用）：`python -m bench.analysis <trace>` 一条命令出报告（MFU 分解、停顿瀑布、长尾分布、tokens/s/GPU）
- 指标口径已冻结：`docs/metrics-v0.md`——**你引用的每个数字必须标注口径章节**
- trace：`bench/traces/synthetic/` 3 份合成 trace 已在 main；真实 trace `bench/traces/m0-baseline.rheotrace.jsonl` 随 A 的 PR 进 main（先拿 A 分支 `feat/m0-baseline` 的版本起步，正式跑合入后刷新数字）

## 目标

完成 M1 验收："回答 rollout GPU 周期去哪了"——这是整个项目的第一份公开研究产出，也是博客 #1 的底稿。

## 任务（按序）

1. **立即开跑**：对现有 trace（真实 + 合成）跑分析，搭出研究文档骨架。
2. **`docs/studies/m1-anatomy-v0.md`**：核心问题逐个给数据支撑的答案——
   - rollout MFU 分解（M1×duty）、停顿瀑布（weight-sync / env-wait / schedule 各占比）
   - 长尾掉队证据（S1 掉队份额、长度分布）——这是 partial rollout 立项依据的核心数据
   - tokens/s/GPU 基线；每个数字可溯源（trace 文件 + 口径引用）
3. **A 的正式跑 trace 合入 main 后刷新全部数字**（只许改适配层，不改指标定义）。
4. **博客草稿 `docs/blog/01-rollout-gpu-anatomy.md`**（中文）：面向工程读者讲清"为什么 rollout 是 RL 训练的真瓶颈"；图表落 `bench/results/study-m1/`。**草稿完成后停在人审，不自行发布。**

## 验收

- [ ] 研究 v0 文档：核心问题有初步数据答案，全部数字可溯源
- [ ] 正式 trace 合入后数字刷新一轮
- [ ] 博客草稿可交人审
- [ ] `ruff check .` + `pytest` 过（图表脚本部分）

## 不做

不改指标定义（口径缺口记 `docs/issues.md`）；不实现 M2 功能；博客不自行发布（署名与发布由人决定）。
提前完工 → 把研究结论翻译成 verl issue 的素材草稿（供 S6 的 PR 引用），或给 S1 的 A/B 实验设计提建议。
