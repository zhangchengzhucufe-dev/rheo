# 云端 GPU 跑 40 步（AutoDL 等按小时计费平台）

> 适用：租 RTX 4090 / A100 等，从零到训练完成约 2-4 小时（含环境）。
> 本仓库的 WSL 专属开关在云端自动关闭（run_grpo.sh 检测 /proc/version）。

## 0. 租什么

- 镜像：选平台自带的 **PyTorch 2.x + CUDA 12.x** 官方镜像（省去 CUDA 安装）
- 显存 **≥24GB**（4090/A5000/A100 均可；显存大可把 run_grpo.sh 里 util 提到 0.85、去掉腾挪）
- 计费：按量；数据盘 ≥30GB（模型 3GB + checkpoint）

## 1. 环境（约 20 分钟）

```bash
git clone https://github.com/zhangchengzhucufe-dev/rheo.git && cd rheo
# python 3.12 venv（平台镜像一般自带 python；没有就用 conda/uv 建 3.12）
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch==2.9.0
uv pip install -c <(echo 'torch==2.9.0') "verl[vllm]==0.8.0" "vllm==0.12.0" \
  "transformers>=4.56,<5" bitsandbytes math-verify matplotlib
uv pip install -e '.[dev]'
# 插桩引导：装进 venv 的 site-packages（一条 cp）
cp bench/m0/sitecustomize.py "$(python -c 'import site; print(site.getsitepackages()[0])')/sitecustomize.py"
```

## 2. 模型 + 数据集（约 10 分钟）

```bash
# 平台一般内置 hf-mirror 加速；没有就 export HF_ENDPOINT=https://hf-mirror.com
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_dir='$HOME/models/Qwen2.5-1.5B-Instruct', ignore_patterns=['*.pth','original/*','*.gguf'])"
python bench/m0/prepare_gsm8k.py   # 数据落到 ~/datasets/rheo/gsm8k
```

## 3. 启动（后台，断线不丢）

```bash
MAX_BATCHES=20 nohup bash bench/m0/supervise_run.sh > train.log 2>&1 &
tail -f train.log
```

 supervise_run.sh 每批内部 6 次重试、最多 20 批、每 5 步存档，
 崩溃自动从 checkpoint 续跑。云端无 WDDM 问题，大概率一批直接跑完。

## 4. 收尾

- reward 曲线：`python bench/m0/plot_reward.py`（自动找最新 tb 目录）
- 轨迹合并校验：`python bench/m0/merge_trace.py --spill-dir bench/results/m0-baseline/traces-spill`
- 结果文件：`bench/traces/m0-baseline.rheotrace.jsonl` + `bench/results/m0-baseline/reward_curve.png`
- 用完释放实例（先确认 checkpoint/trace 已下载回本地或推到仓库）

## 注意

- WSL 的 3 个 site-packages 补丁（CUDA IPC / FSDP 重试 / expandable segments）
  **云端不需要、也不要打**——原生 Linux 上 IPC 正常工作
- 本地 WSL 正在跑的同名实验与云端互不干扰，哪边先到 40 步用哪边
