# M0 Baseline — 环境与版本表

> 会话 A（`feat/m0-baseline`）· 生成于 2026-09-29 · WSL2 (Ubuntu) + Windows 11 宿主

## 硬件

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 **Laptop** GPU |
| 显存 | **6144 MiB**（注意：PLAN.md 假设 12GB；且 Windows 桌面常驻占用约 1–2.5GB，训练实际可用约 3.5–4.5GB） |
| 驱动 | 617.14（CUDA UMD 13.4） |
| 平台 | WSL2, kernel 6.18 x86_64 |
| Python | 3.12.14（venv: `~/tools/venvs/rheo`） |

## 主训练环境 `~/tools/venvs/rheo`（verl + vLLM 训练用）

| 包 | 版本 | 说明 |
|---|---|---|
| `torch` | 2.9.0+cu128 | 锁定（vllm 0.12.0 pin） |
| `vllm` | 0.12.0 | rollout 后端（3060 = sm86，vLLM 自带 attention kernel，无需 flashinfer） |
| `verl` | 0.8.0 | 训练框架（HybridFlow） |
| `transformers` | 4.57.6 | 锁 <5（verl 0.8.0 与 transformers 5.x 未验证兼容） |
| `flash-attn` | 2.8.1+cu12torch2.9 | 预编译轮子（`~/tools/downloads/` 留档），训练侧 remove_padding 用 |
| `bitsandbytes` | 0.50.2 | 8bit 优化器（verl 原生 `optim.optimizer_impl=bitsandbytes.optim` + `optimizer=AdamW8bit`） |
| `peft` | 0.21.0 | LoRA |
| `ray` | 2.58.0 | verl 编排 |
| `datasets` / `pyarrow` / `numpy` | 5.0.1 / 25.0.1 / 2.5.3 | 数据管线（datasets 2.x 与 pyarrow≥19 不兼容，已升级） |
| `tensordict` | 0.10.0 | |
| `hydra-core` | 1.3.7 | |
| `rheorl` | 0.0.1 | 本仓库，`pip install -e` |

## SGLang 环境 `~/tools/venvs/rheo-sglang`（M3+ 引擎基底用，M0 训练不依赖）

| 包 | 版本 |
|---|---|
| `sglang` | 0.5.8 |
| `torch` | 2.9.1+cu128 |

> **为什么分两个 venv**：verl 官方的 `vllm` extra（vllm 0.12.0 → torch==2.9.0）与
> `sglang` extra（sglang 0.5.8 → torch==2.9.1）的 torch pin 互斥，历史上从未能共存于
> 一个环境。M0 训练走 vLLM；SGLang 是 PLAN M3+ 的引擎基底，独立 venv 保证可导入可测。

## 复现命令

```bash
# 1. venv
uv venv --python 3.12 ~/tools/venvs/rheo
uv pip install --python ~/tools/venvs/rheo/bin/python torch==2.9.0
uv pip install --python ~/tools/venvs/rheo/bin/python -c <(echo 'torch==2.9.0') \
  "verl[vllm]==0.8.0" "vllm==0.12.0" "transformers>=4.56,<5" bitsandbytes math-verify
uv pip install --python ~/tools/venvs/rheo/bin/python -e '~/rheo-a[dev]'
# flash-attn 预编译轮子（避免 30+ min 源码编译）
uv pip install --python ~/tools/venvs/rheo/bin/python \
  ~/tools/downloads/flash_attn-2.8.1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# 2. 数据（HF 直连不通时走镜像）
HF_ENDPOINT=https://hf-mirror.com python bench/m0/prepare_gsm8k.py   # → ~/datasets/rheo/gsm8k/

# 3. 模型
HF_ENDPOINT=https://hf-mirror.com python -c \
  "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_dir='$HOME/models/Qwen2.5-1.5B-Instruct', ignore_patterns=['*.pth','original/*','*.gguf'])"

# 4. 训练（必须包 GPU 锁）
STEPS=3 EXP=smoke TEST_FREQ=-1 VAL_BEFORE_TRAIN=false \
  ~/tools/bin/with-lock gpu 1800 -- bash bench/m0/run_grpo.sh    # 冒烟
STEPS=60 TEST_FREQ=10 \
  ~/tools/bin/with-lock gpu 1800 -- bash bench/m0/run_grpo.sh    # 正式
```

## WSL2 本地补丁（`~/tools/venvs/rheo` 内的 site-packages 修改）

重装 verl 后需要重打。三个补丁都源于 WSL2/WDDM 的平台限制：

| # | 文件 | 触发开关 | 问题 | 做法 |
|---|---|---|---|---|
| 1 | `verl/utils/device.py` `is_support_ipc()` | `VERL_DISABLE_CUDA_IPC=1` | WSL2 不支持 CUDA IPC handle，verl 权重传输默认走 IPC 会在 vLLM worker 报 `invalid resource handle` | 环境变量下返回 False → 走宿主共享内存 bucket 传输 |
| 2 | `verl/utils/fsdp_utils.py` | 恒开 | vLLM sleep 刚释放数 GB 后立刻 H2D 装载 FSDP flat param，WDDM 偶发 `cudaErrorNotReady` | `_flat_param_to_device_wsl_safe`：等待+重试（最多 5 次） |
| 3 | `verl/utils/device.py` `set_expandable_segments()` | `RHEO_KEEP_EXPANDABLE=1` | vLLM 进程释放数 GB 与训练进程分配交错时，经典 CUDACachingAllocator 会 INTERNAL ASSERT（跨进程池损坏）；verl 每步还会运行时切回经典池 | 配合 `PYTORCH_ALLOC_CONF=expandable_segments:True`，禁掉 verl 的运行时切换，全程 VMM 分配 |

另两个 6GB 卡的关键配置（不需要补丁，但没它们跑不起来）：

- `+...model.lora.merge=true`：默认 LoRA-as-adapter 模式下 vLLM sleep 只到 level 1（保留 3.1GB 基础权重），训练期 FSDP 再装 3.1GB 必然超 6GB。merge 模式每步全量同步权重，sleep level 2 全量释放，显存才能周转。
- `rollout.gpu_memory_utilization=0.75`：训练期 vLLM 反正全量释放，生成期应让它独占 GPU；util 太小（0.58）时 KV cache 只剩 ~0.5GB，生成陷入抢占-重算循环。

**性能备注**：本机 GPU 与其他 AI 会话共享（Windows 侧另有进程在跑），实测全量权重同步与生成阶段都会被显著拉慢——慢是正常状况，不是故障。

## 网络备注

- huggingface.co 在本机路由不可达（Errno 101），用 `HF_ENDPOINT=https://hf-mirror.com` 替代；pypi 直连正常。
