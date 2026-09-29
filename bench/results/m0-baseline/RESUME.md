# M0 续跑备忘（断电前快照 · 2026-09-29）

> 关机前状态。重启后按"下一步"节继续。代码已推送 `origin/feat/m0-baseline`（commit 38c4fa8+）。

## 已达成（全部落盘）

1. **环境完整**：`~/tools/venvs/rheo`（py3.12 + torch 2.9.0 + verl 0.8.0 + vllm 0.12.0 + bnb +
   flash-attn 2.8.1 预编译轮 + rheorl -e），`~/tools/venvs/rheo-sglang`（sglang 0.5.8，独立）。
   3 个 WSL 补丁已打进 site-packages（重启不丢，env.md 有补丁表 + 重打方法）。
   **sitecustomize.py 引导已在 venv 内**（RHEO_TRACE=1 时自动加载插桩）。
2. **数据/模型**：`~/datasets/rheo/gsm8k/{train,test}.parquet` ✓；
   `~/models/Qwen2.5-1.5B-Instruct` ✓（HF 直连不通，用 `HF_ENDPOINT=https://hf-mirror.com`）。
3. **冒烟训练已验证**（3 步冒烟，断电时跑到第 3 步，checkpoint 未存——但目的已达成）：
   - 完整循环走通：gen → vllm sleep → old_log_prob → actor update(LoRA+AdamW8bit) →
     lora.merge 全量权重同步(shm) → 下一轮 gen
   - reward mean：step1=0.203 → step2=0.602（GSM8K，格式学习曲线正常）
   - GPU 共享（与其他 AI 会话）下 ~26-30 min/step；无人争用时估计 5-10 min
   - 日志：`bench/results/m0-baseline/logs/smoke-1125.log`
   - 无 TB 事件文件（未 flush 即断电）；不影响结论
4. **RheoTrace 插桩已就绪**（未在真实训练中启用过）：
   - `bench/m0/rheo_trace_hooks.py`（monkey patch，env 开关）
   - `bench/m0/merge_trace.py`（spill 合并 + 账本重放 + validate；有单测覆盖）
   - 合成 spill→merge→validate 全流程已测通（tests/test_merge_trace.py 3 个用例过）

## 当前进行中（2026-09-29 17:15 起）

**正式跑 r1 已启动**：`RHEO_TRACE=1 EXP=grpo-lora-qwen25-1.5b STEPS=40 TEST_FREQ=5`，
日志 `logs/train-r1.log` + `logs/attempt-171340.log`，spill → `traces-spill/`，
checkpoint 每 5 步 → `~/tools/rheo-checkpoints/grpo-lora-qwen25-1.5b/`。
中断后：清理 ray 残留 + 删 gpu.lock，**重跑同一命令**即从 checkpoint 续跑。
注意：pkill 的模式别写进启动命令里（会匹配自杀）；清理与启动分两条命令跑。

## 历史修复记录（迷你端到端暴露的 4 个 bug，均已修 + 推送）

1. hooks 的 `install()` 定义了但从未调用（钩子静默失效）
2. weight_sync 钩子打在 worker 侧会破坏 @register 分发（RayWorkerGroup 找不到方法）
   → 改打驱动侧 `CheckpointEngineManager.update_weights`
3. schedule 钩子必须是普通函数（原方法 @auto_await，fit 同步调用拿到 coroutine）
4. merge 先排序后改写区间事件 ts（E04）/ segment_start 落盘时间晚于 span（E07）
   → 先归一化再排序 + segment_start 带显式事件时间

## 重启后下一步（按序）

1. `cd ~/rheo-a && git pull`（确认在 feat/m0-baseline 最新）
2. 清残留：`pkill -9 -f "ray::"`、删 `/mnt/c/Users/15985/tools/.locks/gpu.lock`（若持有者已死）
3. （若 r1 未完成）续跑正式跑：
   ```bash
   RHEO_TRACE=1 EXP=grpo-lora-qwen25-1.5b STEPS=40 TEST_FREQ=5 \
     ~/tools/bin/with-lock gpu 1800 -- bash bench/m0/run_grpo.sh \
     > bench/results/m0-baseline/logs/train-$(date +%H%M).log 2>&1
   ```
   - run_grpo.sh 已内置 RHEO_TRACE/HOOKS/DIR 环境变量传递
   - VAL_BEFORE_TRAIN 默认 true；test_freq=5 出 8 个验证点
4. 训练期间每 15 分钟用 15s 轮询脚本看进度（progress 行 + critic/rewards/mean）
5. 训完：`python bench/m0/plot_reward.py`（TB → reward_curve.png）
6. 合并 trace：`python bench/m0/merge_trace.py --spill-dir bench/results/m0-baseline/traces-spill
   --out bench/traces/m0-baseline.jsonl` → 必须过 validate（ok=True）
7. 统计摘要写 `bench/results/m0-baseline/summary.md`；推分支开 PR

## 运行期注意

- **断电保护（正式跑已启用）**：`SAVE_FREQ=5`（默认）每 5 步存 checkpoint 到
  `~/tools/rheo-checkpoints/<EXP>/`；中断后**重跑同一条命令**即可，verl 的
  `resume_mode=auto` 自动从最新 checkpoint 续跑（loss/优化器状态都在）。断电最多损失 4 步。
- GPU 锁：训练全程持 `with-lock gpu`；其他 AI 会话共用 GPU，慢是正常状况
- 若 venv 重装/升级过 verl：三个补丁要重打（env.md 表格有逐个 diff 说明）
- 下载一律先查两端 manifest；模型/数据集只放 ~/models、~/datasets
- Hydra resolved config 在日志头 "Error executing job with overrides" 行（正式跑后提取进 summary）
