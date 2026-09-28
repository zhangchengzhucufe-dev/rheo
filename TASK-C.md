# TASK-C：分析流水线 v0（纯 CPU，不碰 GPU）

> 你的 worktree：`~/rheo-c`（分支 `feat/analysis`）
> 开工先读：`PLAN.md` §1 L2 遥测 / §4 里程碑 M1 / §5 RolloutBench + 本文件。
> 跨板块问题记 `docs/issues.md`，不顺手实现。

## 已由主会话代做（勿重做）

仓库脚手架已合入 main，含空 `bench/analysis/` 包——**直接开工，不用等任何人**。

## 目标

交付"rollout GPU 周期去哪了"的分析工具，A 的真实 trace 一到就能出报告。

## 任务（按序）

1. **第 1 天先出指标定义** `docs/metrics-v0.md` 并当天发 PR：
   - rollout MFU 分解口径、停顿分类口径（权重同步 / env 等待 / 调度间隙 / 长尾掉队）、
     tokens/s/GPU、轨迹长度分布与长尾统计
   - 与 B 的 `docs/rheotrace-spec-v0.md` 互相校对：字段不够就记 `docs/issues.md` 提给 B 改 spec
2. **`bench/analysis/`**：输入一份 trace 文件 → 输出报告（Markdown + 图）：
   MFU 分解、停顿瀑布图、轨迹长度分布（双峰/长尾检验）、tokens/s/GPU
3. 用 B 的合成 trace（`bench/traces/synthetic/`；未合入前用 B 的生成器自产，
   生成器用法读 TASK-B）全流程跑通，先试"双峰长尾"负载验证长尾统计方法

## 验收（全部达成才 merge 回 main）

- [ ] 指标定义文档 merge
- [ ] `python -m bench.analysis <trace>` 一条命令出完整报告
- [ ] 在合成 trace 上全部指标跑通
- [ ] A 的真实 trace 到位后：**只许改适配层、不许改指标定义**（口径先行的验收）
- [ ] `ruff check .` + `pytest` 过

## 不做

不写 sim/ 仿真器（M2）、不设计调度器、不写博客成稿（等真实数据一起写）。
提前完工 → 预研 `sim/` 的事件回放接口设计（**只写文档不实现**）。

## 纪律（对所有会话生效）

- 分支生命周期 ≤ 1 周，验收达标即 merge 回 main；勤 rebase main
- 不启动 M2 的东西：scheduler、WeightManager、sim/ 实现
- 别人板块的坑 → `docs/issues.md`，不顺手改
