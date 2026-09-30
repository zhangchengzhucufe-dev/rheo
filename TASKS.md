# TASKS.md — 总调度（阶段 2：M1 收尾 + M2 前半）

> 阶段 1 已完成归档（任务书 [TASK-A.md](TASK-A.md) / [TASK-B.md](TASK-B.md) / [TASK-C.md](TASK-C.md)）。
> 阶段 2 各会话只读 `PLAN.md` + 自己的 TASK-SX.md + 本文件全局纪律。

## 阶段 1 状态（归档，2026-09-29 ~ 09-30）

- [x] M0：GRPO+LoRA 基线跑通、真实 trace 产出（feat/m0-baseline 最终验证中，PR 后合入）
- [x] RheoTrace v0.1 规格冻结 + `rheotrace` 包（98 测试绿，已合入 main）
- [x] metrics-v0 口径定稿 + `bench.analysis` 报告流水线（已合入 main）
- [x] 附加产出：protocol v0 预研、verl 补丁记录、Apache 2.0 开源门面
- [x] 硬件事实修正：6GB 实测（见 PLAN §9），默认实验规模改 0.5B

## 阶段 2 worktree 布局

| 会话 | 目录 | 分支 | 任务书 | 资源 |
|---|---|---|---|---|
| S1 调度引擎（关键路径） | `~/rheo-s1` | `feat/scheduler-v1` | [TASK-S1.md](TASK-S1.md) | **GPU 唯一持有者**（with-lock） |
| S2 仿真器 | `~/rheo-s2` | `feat/sim-v0` | [TASK-S2.md](TASK-S2.md) | 纯 CPU |
| S3 工作负载 | `~/rheo-s3` | `feat/workloads-v0` | [TASK-S3.md](TASK-S3.md) | 纯 CPU |
| S4 Env Gateway | `~/rheo-s4` | `feat/env-gateway-v0` | [TASK-S4.md](TASK-S4.md) | 纯 CPU |
| S5 测量研究+博客 | `~/rheo-s5` | `feat/study-m1` | [TASK-S5.md](TASK-S5.md) | 纯 CPU |
| S6 verl PR 打包 | `~/rheo-s6` | `feat/verl-pr-pack` | [TASK-S6.md](TASK-S6.md) | 纯 CPU + verl fork |

## 节拍（2 周）

| 时间 | 内容 |
|---|---|
| D1 接口日 | S1 设计文档定稿（解锁 S2/S4）；S3 负载规格、S4 分级设计、S5 研究大纲同日发 PR |
| D2–4 实现日 | 各自实现+单测；S5 用现有 trace 出研究 v0；S6 提炼补丁 |
| D5–7 集成日 | S1 接 S3 负载真机冒烟；S2 接 S1 策略 harness；S4 mock 联调 |
| D8–10 实验日 | S1 真机 A/B（0.5B 默认档）；S2 仿真策略对比实验 |
| D11–14 成稿日 | S1 数据成文、S2 仿真报告、S5 博客交人审；S6 PR 材料就绪 |

## 依赖关系

```
S1 设计文档 ──→ S2 策略 harness、S4 分级设计
S3 负载规格 ──→ S1 真机实验、S2 回放供料
S1 实验数据 ──→ S5 研究结论
S6 独立；rheotrace §8 增量由 S2 维护（需求一律走 docs/issues.md）
```

## 全局纪律

- **GPU 只有 S1 碰**，一律 `~/tools/bin/with-lock gpu 1800 -- <命令>`；其余会话纯 CPU
- 模型 → `~/models`、数据集 → `~/datasets`、venv → `~/tools/venvs`，严禁进工作目录；下载前查两端 `~/tools/.download-manifest.txt`
- 分支 ≤ 2 周，验收达标即 merge，PR 过 CI（ruff + CPU pytest）；接口类 PR 当天合
- **不启动 M3+**：WeightManager、KVT/probe、kernels 实现（scheduler 只留接口位）
- 默认实验规模 **0.5B**（PLAN §9）；绝对吞吐数字一律标注模型规模
- 跨板块问题 → `docs/issues.md`，不顺手实现
- 提前完工去向：见各任务书末尾

## 阶段 3 预告（本阶段不启动）

M3 WeightManager：本地 0.5B 双缓冲验证（fp16 双份 ≈2GB）；1.5B/7B 热切换上云单卡 A100/A800。
M2 末云爆发：8×A800 NVLink 整机 6–12h 聚焦实验（≈¥400–1200，PLAN §9），验证调度规模化——届时给你预算清单拍板。
