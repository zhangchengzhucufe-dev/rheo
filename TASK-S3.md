# TASK-S3：RolloutBench 工作负载（纯 CPU）

> 你的 worktree：`~/rheo-s3`（分支 `feat/workloads-v0`）
> 开工先读：`PLAN.md` §4 M2、§5 RolloutBench、§9 硬件事实 + 本文件。跨板块问题记 `docs/issues.md`。

## 阶段 1 留给你的资产

- M0 的 GRPO 数学管线：`bench/m0/`（prepare_gsm8k.py、run_grpo.sh、plot_reward.py），`bench/results/m0-baseline/env.md` 有环境与补丁说明
- 合成 trace 生成器（`python -m rheotrace gen`）可参考负载形态，但你要造的是**真机负载配置**，不是 trace

## 目标

把 PLAN §5 的三套工作负载做成可一键执行的配置，交到 S1（真机）和 S2（仿真回放供料）手里。

## 任务（按序）

1. **D1：`bench/workloads/README.md` 负载规格定稿发 PR**：三套负载的构造方法、参数表（模型规模 0.5B 默认 / 1.5B 可选、group 数、G、max_len）、每套的验收口径。
2. **grpo-math**：参数化复用 M0 的 GSM8K 配方，暴露组数 / 每组 G / 长度分布旋钮。
3. **agent-tool**：合成工具服务——每工具可配置延迟分布（快查询 / 慢爬取 / 超长计算三类），多轮对话制造真实 env-wait。这同时是 S4（KV 分级）和 S2（仿真）的供料。
4. **bimodal-longtail**：双峰长度分布构造（短平快 + 超长尾混合 + max_len 约束），专打 partial rollout 弱点。
5. **runner**：`bench/workloads/run.py` 生成 verl 可用配置 + 输出交给 S1 的命令清单（含 with-lock 用法）；CPU 侧干跑校验（配置合法性、数据管道走通，不占 GPU）。

## 验收

- [ ] 规格文档 D1 merge
- [ ] 三套负载配置齐，干跑通过（全程不占 GPU）
- [ ] S1 拿到可直接复制执行的跑分 runbook
- [ ] `ruff check .` + `pytest` 过

## 不做

不跑真机（统一由 S1 执行，避免 GPU 争抢）；不改指标口径（`docs/metrics-v0.md` 冻结）；不做调度逻辑。
提前完工 → 给每套负载写"预期 trace 形态"描述（S2 回放校验用），或帮 S5 准备博客图表素材。
