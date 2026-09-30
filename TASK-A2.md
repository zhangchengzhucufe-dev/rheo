# TASK-A2：M0 收尾冲刺——复盘加固 + 完整重跑（A 的最后一棒）

> 你的 worktree：`~/rheo-a`（分支 `feat/m0-baseline`，先 `git pull` 拿到本文件）
> 背景：云端 40 步训练完成，但复盘（用户转发的《M0 云端 40 步训练复盘》）发现两处交付硬伤。
> 本文件是阶段 1 的最终任务，做完即可功成身退。

## 复盘判定（为什么还要干一天）

- 训练本体达标：40 步 rc=0，GSM8K 68.5%→75.5%→72.0%，成本 ¥10-11
- **硬伤 1（G1）**：最终 trace 只覆盖 step 30-40——崩溃续跑前 spill 目录被清，1-29 步轨迹事件丢失
- **硬伤 2（G3/E16）**：计划 4 个验证点只有 3 个（step 30 随崩溃丢失）
- 结论：管线验证通过，但 S5 的公开测量研究需要完整 40 步 trace → **加固后完整重跑一次（¥4-6，1.5-2h）**

## 第 0 步：云端实例处置（先做，涉及钱）

- **实例还活着** → 立即把配好的环境**存成 AutoDL 镜像**（复盘 G6，本次 1.5h 环境试错的教训），再继续
- **实例已关** → 不必为镜像重开；按 CLOUD_GUIDE + 复盘避坑清单新开实例
- 任何情况下：supervisor 成功出口挂**自动关机钩子**（G6），不许无人值守烧卡

## 必修清单（重跑前，全部对应复盘编号）

按"会再次丢数据/烧钱"优先：

1. **D14/G1 — spill 生命周期**：每次启动生成 run_id，spill 目录按 run_id 分子目录；`merge_trace.py` 按 run_id 合并、拒绝混入历史 run；续跑时**绝不**清 spill（verl 会续训，事件必须续写）。merge 输出头部加 `covered_steps:` 注记（对照训练日志的 step 集合，缺步至少 WARNING）
2. **D12/D13 — checkpoint 策略**：默认只存 LoRA 适配器（~30MB，不存 6.8GB 合并模型）；`max_ckpt_to_keep=2`；CKPT_DIR/日志/trace spill 统一走一个 scratch 盘变量；supervisor 每批开始前 `df` 预检，低于阈值显式报 `ENOSPC: 需要 X，剩 Y` 并停
3. **E15/C11 — 失败分类 + 降档梯子**：导入/配置/路径类异常 → 立即终止并高亮根因（不烧重试）；OOM/引擎启动类 → 降档重试，梯子 = util → micro 减半 → max_num_seqs 减半，同一 OOM 签名快速跳档
4. **B6/B7/F20 — 可移植性**：脚本禁止 `$HOME/rheo-a` 私人路径（从脚本位置推导）；`PYTHON` 默认 `${PYTHON:-python}`；sitecustomize 改走 `PYTHONPATH` 零拷贝
5. **G7 — 参数对账**：每次 attempt 把最终生效参数（micro/util/steps/model_path/ckpt_dir）echo 到日志首行

应修（有时间就做）：

6. A2/A3/F19 — `doctor` 预检脚本（import 冒烟 + 版本矩阵断言）+ `huggingface-hub` 钉版
7. E16 — resume 后立即补跑一次验证
8. E18 — supervisor 改 PID 文件管理，停止走 `kill $(cat pidfile)`
9. F21 — attempt 日志轮转（保留 N 份）+ 同走 scratch 盘

## 完整重跑规格

- **单一配置跑全程**（micro=2, util=0.55——第一轮被验证过的安全档），消灭 G4 的三段拼接问题
- 1.5B 配置不变（与首轮可比）；删旧 checkpoint 从头跑 40 步；`test_freq=10` 拿满 4 个验证点
- 旧 trace 保留改名 `bench/traces/m0-baseline-resumed.rheotrace.jsonl`（标注覆盖 30-40），新 trace 为 `m0-baseline.rheotrace.jsonl`
- 顺手审计（G5）：summary.md 里加一张"指标 × 语义"对照表，查 `val-aux/gsm8k/reward/mean@1≡2.0` 的串位问题，结论一句话记录（不动 verl 源码）
- 产物：完整 trace（validate ok=True 且 covered_steps=1-40）+ 4 点曲线 + summary.md + 更新后的 CLOUD_GUIDE（并入复盘避坑部分）

## 验收

- [ ] 必修清单 5 项合入分支
- [ ] 完整 40 步 trace 过 validate 且覆盖率注记齐全
- [ ] 4 个验证点的 reward 曲线（单次连续运行，无拼接）
- [ ] AutoDL 镜像已存（或新实例指南已含镜像步骤）+ 自动关机钩子生效
- [ ] push + 开 PR 合 main；然后读 TASK-A.md 末尾交接说明（帮 S1 做集成或退场）

## 不做

不改 verl 源码（C10 流式化、A4 flash_attn 建议留给 S6 的 PR 素材，本阶段用 micro=2 安全档绕过）；不改 rheotrace spec/validator（覆盖率断言是 S2 的第一单 §8 增量）；不启动任何 M2 功能。
