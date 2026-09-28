# TASK-B：RheoTrace v0（纯 CPU，不碰 GPU）

> 你的 worktree：`~/rheo-b`（分支 `feat/rheotrace`）
> 开工先读：`PLAN.md` §1 L2 遥测 / §2 迭代2 / §4 里程碑 M1 + 本文件。
> 跨板块问题记 `docs/issues.md`，不顺手实现。

## 已由主会话代做（勿重做）

仓库脚手架已合入 main，含空 `rheotrace/` 占位包——**直接开工，不用等任何人**。

## 目标

冻结 M1 的核心接缝——trace 格式，交付 `rheotrace` Python 包。A（插桩）和 C（分析）都在等这个接口。

## 任务（按序）

1. **第 1 天先出规格** `docs/rheotrace-spec-v0.md` 并当天发 PR 定稿：
   - 事件模型：轨迹段生命周期（running / paused / env-wait / finished / aborted）、
     权重版本戳（birth_version）、token 级 logprob 记录、
     阶段计时区间（prefill / decode / weight-sync / env-wait / schedule）
   - 字段表 + 存储格式选型结论（JSONL 事件流 vs Arrow 列存，给出取舍理由）
   - 版本语义：何时 bump version、版本号与 trainer step 的对应关系
2. **`rheotrace` 包**：替换占位包，实现 `write / read / validate` 三个核心 API
   + 合成 trace 生成器（参数：轨迹数、长度分布可设双峰、各阶段延迟可注入）
3. 用生成器产出 2–3 份样例 → `bench/traces/synthetic/`
   （小文件允许 `git add -f` 强制进 git，供 C 直接使用）

## 验收（全部达成才 merge 回 main）

- [ ] spec 文档 merge，作为冻结接口被 A、C 引用
- [ ] 合成 trace 过 write→read round-trip 无损（测试覆盖）
- [ ] validator 能拒绝坏文件：缺字段 / 乱序事件 / 版本回退，各至少一例测试
- [ ] 纯 CPU 可测，`ruff check .` + `pytest` 过

## 不做

不碰 verl（插桩是 A 的事）、不做 Grafana 面板、不定义 gRPC 协议（M6）。
提前完工 → 预研 `protocol/` 的 proto 草稿（**只写文档不实现**）。

## 纪律（对所有会话生效）

- 分支生命周期 ≤ 1 周，验收达标即 merge 回 main；接口类 PR 优先合入，别人在等
- 不启动 M2 的东西：scheduler、WeightManager、sim/
- 别人板块的坑 → `docs/issues.md`，不顺手改
