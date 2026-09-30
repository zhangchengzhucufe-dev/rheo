# 云端 GPU 跑 40 步（AutoDL 等按小时计费平台）

> TASK-A2 修订版：并入《M0 云端 40 步训练复盘》避坑清单（G1-G7）。
> 适用：租 RTX 4090D/5090 等，从零到训练完成约 2-3 小时（含环境）。
> 本仓库的 WSL 专属开关在云端自动关闭（run_grpo.sh 检测 /proc/version）。

## 0. 租什么（复盘 G4/G5 教训）

- **GPU**：RTX 4090D 24GB 首选（空闲多、≈4090）；5090 需选 CUDA 12.8 镜像；
  **避开 12GB 卡**（会复刻本地显存腾挪问题）和 vGPU（虚拟化性能不可控）
- **镜像**：PyTorch 2.5.1 + CUDA 12.4 + Python 3.12 平台镜像
- **数据盘**：≥50GB（scratch：checkpoint 14GB 上限 + spill + 日志）
- 计费：按量；**40 步全程约 ¥5-8**

## 1. 环境安装（约 20 分钟）

```bash
git clone https://github.com/zhangchengzhucufe-dev/rheo.git && cd rheo
git checkout feat/m0-baseline   # 或已合并则 main

python -m venv ~/venv-rheo && source ~/venv-rheo/bin/activate
pip install -U pip uv
uv pip install torch==2.9.0
uv pip install -c <(echo 'torch==2.9.0') "verl[vllm]==0.8.0" "vllm==0.12.0" \
  "transformers>=4.56,<5" "huggingface-hub<1.0" bitsandbytes math-verify matplotlib
uv pip install -e '.[dev]'
```

> **huggingface-hub 必须 <1.0**（复盘 A3）：hub 1.x 与 transformers<5 不兼容，
> 报 `require_version` ImportError 即是此因。

插桩引导经 **PYTHONPATH 零拷贝**（run_grpo.sh 自动设置，无需 cp 到 site-packages）。

## 2. 预检（约 1 分钟，90 秒发现 90% 环境问题）

```bash
bash bench/m0/doctor.sh
```

全绿（ALL GREEN）才继续；✗ 项按提示修（版本漂移/路径缺失/磁盘不足都会在这里暴露）。

## 3. 模型 + 数据集（约 10 分钟）

```bash
# AutoDL 有学术加速；没有则 export HF_ENDPOINT=https://hf-mirror.com
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_dir='$HOME/models/Qwen2.5-1.5B-Instruct', ignore_patterns=['*.pth','original/*','*.gguf'])"
python bench/m0/prepare_gsm8k.py   # → ~/datasets/rheo/gsm8k
```

## 4. 启动（TASK-A2 G6 + D13）

```bash
# scratch 盘变量（checkpoint/日志/spill 统一走 /root/autodl-tmp，系统盘不塞爆）
export SCRATCH=/root/autodl-tmp/rheo-scratch
# 成功后自动关机（省钱关键，G6）：训练完成 60 秒后实例自动关机
export AUTO_SHUTDOWN=1

MAX_BATCHES=20 nohup bash bench/m0/supervise_run.sh > train.log 2>&1 &
tail -f train.log
```

- 每批开始前自动做 **df 预检**（<20GB 显式报 ENOSPC 并停）
- attempt 日志首行回显全部生效参数（G7），轮转保留 10 份
- 训练 trace spill 按 **run_id 分目录**，续跑绝不清理（G1 修复）
- checkpoint `max_ckpt_to_keep=2`（磁盘上限 ~14GB）
- 停止训练：`bash bench/m0/stop_supervisor.sh`

## 5. 收尾

```bash
python bench/m0/plot_reward.py                      # reward 曲线
python bench/m0/merge_trace.py \
  --spill-dir ~/rheo-scratch/spill/<RUN_ID> \
  --expected-steps 40                               # 必须 covered_steps: 1-40 (40 步)
# 产物：bench/traces/m0-baseline.rheotrace.jsonl + bench/results/m0-baseline/reward_curve.png
```

**下载这两个文件到本地后**再释放实例；`AUTO_SHUTDOWN=1` 时实例已自动关机（数据盘保留）。

## 6. 实例镜像（复盘 G6：不要为环境重装再花 1.5 小时）

首次环境配好后、训练前：AutoDL 控制台 → **保存镜像**（免费额度内）。
之后重跑/换卡直接从镜像开机，跳过第 1-3 步。

## 避坑速查（复盘编号）

| 编号 | 坑 | 对策（已内建） |
|---|---|---|
| G1 | 崩溃续跑前清 spill → 轨迹丢 1-29 步 | spill 按 run_id 分目录，续跑只续写（D14） |
| G3/E16 | 验证点丢失 | test_freq=10 + 恢复后 val 点照常落（test_freq 对齐） |
| G4 | 三段不同配置拼接 | 单一配置 micro=2/util=0.55 跑全程 |
| G6 | 实例忘关烧钱 / 环境重建 1.5h | AUTO_SHUTDOWN 钩子 + 存镜像 |
| A3 | huggingface-hub 1.x 冲突 | 安装命令已钉 <1.0 |
| E15 | 配置错误烧重试 | run_grpo 失败分类：配置类立即终止并高亮根因 |
| D13 | 磁盘写满毁 checkpoint | scratch 统一 + df 预检 + max_ckpt_to_keep=2 |

## 注意

- WSL 的 3 个 site-packages 补丁（CUDA IPC / FSDP 重试 / expandable segments）
  **云端不需要、也不要打**——原生 Linux 上 IPC 正常工作
- `val-aux/gsm8k/reward/mean@1` 恒为 2.0 是 verl 的重复计分派生指标（见 summary.md G5 审计），
  正确口径看 `val-core/gsm8k/acc/mean@1`
