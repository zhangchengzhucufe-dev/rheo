# TASKS.md — 并行开发总调度（2026-09-29）

> 各会话任务书已拆分：**A → [TASK-A.md](TASK-A.md)** ｜ **B → [TASK-B.md](TASK-B.md)** ｜ **C → [TASK-C.md](TASK-C.md)**
> 每个会话只读 `PLAN.md` + 自己的 TASK-X.md。本文件是给人看的总览与节拍表。

## 状态

- [x] 仓库脚手架（原 A 的第 1 步，主会话已代做并合入 main：目录骨架 / pyproject / ruff+pytest / CI / README）
- [ ] A：venv 环境 → 基线训练 → verl 插桩出真实 trace
- [ ] B：RheoTrace spec 定稿 → rheotrace 包 + validator + 合成 trace
- [ ] C：metrics 口径定稿 → `bench.analysis` 报告流水线

## worktree 布局

| 会话 | 目录 | 分支 | 任务书 |
|---|---|---|---|
| A 基建 + GPU 关键路径 | `~/rheo-a` | `feat/m0-baseline` | TASK-A.md |
| B RheoTrace v0（纯 CPU） | `~/rheo-b` | `feat/rheotrace` | TASK-B.md |
| C 分析流水线 v0（纯 CPU） | `~/rheo-c` | `feat/analysis` | TASK-C.md |

## 第 1 周节拍

| 时间 | A | B | C |
|---|---|---|---|
| D1 | 建 venv / 查清单 / 拉权重 | spec 文档定稿 PR | metrics 文档定稿 PR |
| D2–4 | 基线训练（GPU 持锁） | rheotrace 包 + validator merge | 分析代码对合成 trace 跑通 |
| D5–7 | rheotrace 插桩 → 真实 trace | 支援 A 插桩 / 修 validator | 真实 trace 出第一版报告 |

依赖：B 的 spec 被 A（插桩）和 C（分析）引用，所以 B 的 spec PR 当天定稿最优先。

## 全局纪律（三个任务书里都有，此处汇总）

- **GPU 只有 A 碰**，一律 `~/tools/bin/with-lock gpu 1800 -- <命令>`；B、C 纯 CPU 工作
- 模型 → `~/models`、数据集 → `~/datasets`、venv → `~/tools/venvs`，严禁进工作目录；
  下载前查两端 `~/tools/.download-manifest.txt`
- 分支 ≤ 1 周，验收达标即 merge，PR 过 CI（ruff + CPU pytest）
- **不启动 M2**：kernels/ 实现、scheduler、WeightManager、sim/
- 跨板块问题 → `docs/issues.md`，不顺手实现
- 提前完工的去向：A 帮 B/C 写测试；B 预研 protocol/ proto 草稿；C 预研 sim/ 回放接口——都只写文档
