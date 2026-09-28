# 合成 trace 样例（synthetic samples）

由 `rheotrace.gen` 生成，格式见 `docs/rheotrace-spec-v0.md`。
正常情况下 trace 不进 git（`.gitignore: bench/traces/*`）；这里三份小样例经 `git add -f`
强制入库，供会话 C 在真实 trace 到位前开发与自测分析流水线。

| 文件 | 预置 | 负载形态 | 再生成命令 |
|---|---|---|---|
| `synthetic-grpo-small.jsonl` | `grpo` (seed 11) | 单峰长度、无 env、2 次 weight_sync | `python -m rheotrace gen --preset grpo --seed 11 --out <path>` |
| `synthetic-bimodal-longtail.jsonl` | `bimodal` (seed 7) | 双峰长尾（短峰 64 / 长峰 1024）+ 换权重暂停续跑 + 早停 | `python -m rheotrace gen --preset bimodal --seed 7 --out <path>` |
| `synthetic-agent-envwait.jsonl` | `agent` (seed 3) | 多轮工具调用 env_wait、跨版本续跑 | `python -m rheotrace gen --preset agent --seed 3 --out <path>` |

- 同 seed 字节级可复现；参数记录在每个文件 `run_start.meta.preset/params`。
- 校验：`python -m rheotrace validate bench/traces/synthetic/*.jsonl`（当前 0 error 0 warning）。
