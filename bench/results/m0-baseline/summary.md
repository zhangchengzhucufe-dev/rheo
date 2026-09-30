# M0 Baseline 结果总结

> 会话 A · 2026-09-30 · 交付物：GRPO 训练循环 + 第一份真实 RheoTrace

## 结果概览

| 项 | 值 |
|---|---|
| 运行平台 | AutoDL 云端 **RTX 4090D** 24GB（原生 Linux） |
| 任务 | GRPO + LoRA(r16) + AdamW8bit，Qwen2.5-1.5B-Instruct，GSM8K |
| 训练步数 | 16 步（batch 16 prompts × n=8 采样 = 128 段/步）+ 1 次验证（200 题） |
| 墙钟 | **56 分钟**（含初始化与验证；对比 WSL 同任务 12 分钟/步且频繁崩溃） |
| **train reward** | **0.19 → 0.65（第 2 步）→ 0.75–0.85（第 10 步后）**，学习信号清晰 |
| **val gsm8k acc** | **0.68 → 0.75**（见 reward_curve.png 虚线） |
| 生成 token | 386,509 |

## RheoTrace 交付（本里程碑的核心产出）

- 文件：`bench/traces/m0-baseline.rheotrace.jsonl`（云端产出，8.3MB，**未入 git**，按 .gitignore 约定）
- 校验：`rheotrace.validate` → **ok=True, 0 errors, 0 warnings**（E01–E18 / W01–W09 全过）
- 事件构成：run_start/run_end + 17 weight_sync + 2248 segment 生命周期 + 2248 decode
  span + 2248 token_logprob 块 + 16 schedule span（merge 从段边界推导）
- 首批可回答的测量问题（M1 方向预演）：
  - **weight_sync 累计 184s，占墙钟 5.4%**——同步训练下权重同步不是大头，验证了
    "M2 调度收益主要在批间空转（schedule 3139s）而非权重搬运"的直觉
  - **staleness 全为 0**——同步 GRPO 基线的正确形态，为 M3 异步化提供了对照基线
  - decode span 与 schedule span 可分离——C 的停顿瀑布分析有了原料

## 环境与复现

- 版本表与复现命令：`bench/results/m0-baseline/env.md`
- 训练配置：`bench/m0/run_grpo.sh`（含 6GB 显存的 WSL 配方 + 云端自动适配）
- 云端操作指南：`bench/m0/CLOUD_GUIDE.md`
- 插桩：`bench/m0/rheo_trace_hooks.py`（5 钩子）+ `bench/m0/merge_trace.py`（账本重放 + 校验）

## 过程记录（WSL → 云端）

1. 本地 WSL2：环境搭建 + 配置踩坑全走通；3-step 冒烟 reward 0.20→0.60→0.66；
   2 轮 1-step 迷你端到端验证（含断点续跑、自动重试）
2. WSL 长跑受 WDDM 限制：41 条 dxg ENOMEM 内核日志、NCCL 饿死、OS 级 GPU 封锁——
   防护体系（NCCL 2h 超时 / 瞬态重试 / supervisor 无人值守 / 5 步存档）全部落地并验证
3. 最终切云端 4090D：**56 分钟干净跑完**，一次成功，零干预

结论：管线、插桩、防崩溃体系平台无关且已全部实测；WSL 仅适合短跑，
长跑用云端/原生 Linux（与 PLAN.md §8 的"3060 起步 + 云爆发"路线一致）。

## 验收对照（TASK-A）

- [x] venv 可复现（env.md 版本表 + setup 命令）
- [x] 完整训练循环出 reward 曲线（train reward 0.19→0.85，val acc 0.68→0.75）
- [x] 基线遥测落 RheoTrace 且过 validator（ok=True, 0 errors, 0 warnings）
- [x] `ruff check .` + `pytest` 过（23 用例）
