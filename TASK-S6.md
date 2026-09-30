# TASK-S6：verl 上游 PR 打包（纯 CPU，主要工作在 verl fork 外部仓库）

> 你的 worktree：`~/rheo-s6`（分支 `feat/verl-pr-pack`，rheo 侧只放文档）
> 开工先读：`PLAN.md` §2 迭代9、§7 分工 + 本文件。跨板块问题记 `docs/issues.md`。

## 前置检查（开工第一步）

确认 verl 已 fork 到用户 GitHub 账号（zhangchengzhucufe-dev）下：
`gh repo view zhangchengzhucufe-dev/verl` 或网页检查。**没有 fork 就停下来报告用户**（fork 需要用户网页点一下）。

## 背景与目标

木马式贡献路径（迭代 9）：先把 RheoTrace 遥测以普通 PR 贡进 verl（他们本就需要 rollout 可观测性），格式成为生态通用语之后，引擎就是"说这个标准的东西"。你的任务：把阶段 1 的插桩成果提炼成**最小可合入**的上游贡献。

## 素材（从 rheo 仓提炼，不带入调度逻辑）

- `feat/m0-baseline` 分支：`bench/m0/rheo_trace_hooks.py`（hooks）、`bench/results/m0-baseline/env.md`（3 个 verl 补丁的完整 diff 说明）
- `rheotrace` 包（纯 Python 零重依赖，作为可选依赖引入 verl）

## 任务（按序）

1. **提炼最小补丁集**：只保留"遥测"——RheoTrace writer + rollout 路径 hooks + `pip install verl[rheotrace]` 可选依赖；**剔除**一切 rheo 调度器专属逻辑。
2. **干净 checkout 重打**：clone 用户 fork，建分支 `feat/rheotrace-telemetry`，补丁重打 + 按 verl 风格补测试与文档章节。
3. **验证**：verl 自身测试套件通过；跑一个官方 example 训练产出过 `rheotrace.validate` 的 trace。
4. **PR 材料成稿** `docs/verl-pr-notes.md`（rheo 仓内）：动机（rollout 可观测性缺口）、基准数字（引用 S5 的研究）、兼容性与回退方案、PR 描述全文。

## 验收

- [ ] 干净 verl checkout + 分支上 verl 测试通过
- [ ] example 训练产出有效 trace（validate ok=True）
- [ ] PR 描述成稿，分支已推到用户 fork
- [ ] rheo 仓 CI 绿（文档变更）

## 不做

**不提交 PR**——社区沟通必须用户本人出面（PLAN §7），你把材料备到"确认即可发送"的程度；不等上游回复（社区节奏以周/月计，不阻塞任何人）；不把调度器塞进这个 PR（保持最小可合入，调度器等 M2 后半第二个 PR）。
