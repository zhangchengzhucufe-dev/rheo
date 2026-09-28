# rheo

> Weights flow. Trajectories follow. — 原生为 RL 后训练设计的 rollout 引擎：版本化权重流、版本感知 KV cache、轨迹调度。

完整设计与实施蓝图见 [PLAN.md](PLAN.md)。当前阶段：**M0**（3060 上 verl + Qwen2.5-1.5B GRPO 基线跑通）。

## 并行开发

三个会话在独立 worktree 并行工作，各自只读 `PLAN.md` + 自己的任务书：

| 会话 | 任务书 | 分支 | worktree |
|---|---|---|---|
| A 基建 + GPU 关键路径 | [TASK-A.md](TASK-A.md) | `feat/m0-baseline` | `~/rheo-a` |
| B RheoTrace v0 | [TASK-B.md](TASK-B.md) | `feat/rheotrace` | `~/rheo-b` |
| C 分析流水线 v0 | [TASK-C.md](TASK-C.md) | `feat/analysis` | `~/rheo-c` |

总调度与节拍表见 [TASKS.md](TASKS.md)。

## 开发

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

跨会话发现的问题记入 [docs/issues.md](docs/issues.md)，不顺手实现。
