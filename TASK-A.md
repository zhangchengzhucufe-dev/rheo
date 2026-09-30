# TASK-A：基建 + GPU 关键路径（唯一持 GPU 的会话）

> 你的 worktree：`~/rheo-a`（分支 `feat/m0-baseline`）
> 开工先读：`PLAN.md` 全文 + 本文件。跨板块问题记 `docs/issues.md`，不顺手实现。

## 已由主会话代做（勿重做）

仓库脚手架已合入 main：目录骨架（kernels/runtime/orchestration/adapters/sim/protocol/bench）、
`pyproject.toml`（包名 `rheorl`）、ruff + pytest 配置、GitHub Actions CI、README。
`git log` 可查。**你的任务从下面第 1 步开始。**

## 目标

完成 PLAN M0 验收：3060 上 verl + Qwen2.5-1.5B GRPO 跑通，并产出第一份真实 RheoTrace。

## 任务（按序）

1. **环境**：在 `~/tools/venvs/rheo` 建 Python 3.11+ venv，装 PyTorch(CUDA) / verl / SGLang，
   再 `pip install -e ~/rheo-a`。
   - 任何下载前先查两端清单 `~/tools/.download-manifest.txt`（WSL：`~/tools/...`；
     Windows：`/mnt/c/Users/15985/tools/...`），避免重复下载
   - Qwen2.5-1.5B 权重拉到 `~/models`，**严禁进工作目录**
   - 关键包版本表记到 `bench/results/m0-baseline/env.md`
2. **基线训练**：GRPO + LoRA + 8bit 优化器，3060 上完整跑通训练循环
   （step 数以跑得完为准，先小后大）。
   - **所有 GPU 命令一律包锁**：`~/tools/bin/with-lock gpu 1800 -- <命令>`；
     抢不到锁就等，等待是正确行为，不要绕过
   - 产出：reward 曲线图 + 训练配置 + 日志 → `bench/results/m0-baseline/`
3. **打桩**：等 `rheotrace` 包合入 main（会话 B 交付）后 rebase，在 verl rollout 路径插桩
   （submit / 生成 / token 边界 / 权重同步 / env 等待各阶段计时），重跑一次，
   产出通过 `rheotrace.validate` 的 trace → `bench/traces/`
   （trace 本身被 .gitignore；校验结果与统计摘要写进 `bench/results/m0-baseline/`）

## 验收（全部达成才 merge 回 main）

- [ ] venv 可复现（`env.md` 有版本表）
- [ ] 完整训练循环出 reward 曲线（M0 验收上半）
- [ ] 基线遥测落 RheoTrace 且过 validator（M0 验收下半）
- [ ] `ruff check .` + `pytest` 过，PR 过 CI

## 不做

任何性能优化、任何调度逻辑、kernels——基线越朴素越好。
提前完工 → 帮 B/C 写测试（读他们的 TASK 看缺什么）。

## 纪律（对所有会话生效）

- 分支生命周期 ≤ 1 周，验收达标即 merge 回 main；勤 rebase main
- 不启动 M2 的东西：kernels/ 实现、scheduler、WeightManager、sim/
- 别人板块的坑 → `docs/issues.md`，不顺手改

---

## 第二阶段交接（2026-09-30 加，本文件其余部分为阶段 1 存档）

正式跑收尾（RESUME 步骤 5–7：plot_reward → merge_trace（必须 validate ok=True）→ summary.md）完成后：

1. push 分支开 PR 合入 main；有冲突就 rebase main（main 已前进：B/C 的工作都在里面）
2. 合入后通读 **TASK-S1.md**——调度器的 verl 集成需要你积累的插桩知识（hooks 模式、3 个补丁、RESUME.md）。若 S1 由新会话接任，把 `bench/m0/RESUME.md` 和 `env.md` 留给对方作为必读
3. 你的会话即可功成身退，或转去帮 S1 做 verl 集成（读 TASK-S1.md 任务 3）
