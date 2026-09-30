# Rheo：原生 RL Rollout 引擎 —— 完整设计与实施蓝图

> 版本 v1.0（2026-09-29，经 10 轮自迭代纠正）
> 定位：一个原生为 RL 后训练设计的 rollout 引擎 + 开放 Rollout 协议 + 社区生态
> 目标形态：SGLang 之于 serving，Rheo 之于 RL rollout

## 命名（2026-09-29 定案）

**Rheo**，源自希腊语 ῥέω（"流动"）——轨迹如河流动，权重如水流周期性更新；词源同 rheology（流变学，研究万物如何流动）与 rheostat（调节流量的变阻器，对应 staleness 控制器给 rollout 调速）。
- 小写 `rheo`，与 vllm/sglang/triton 排在一起毫无违和
- PyPI 包名 **`rheorl`**（已验证空闲；裸名 `rheo` 被 Seven Bridges 的休眠占位包占用）
- GitHub 建议 `rheo-rl/rheo` 或个人账号下 `rheo`
- Slogan：**"Weights flow. Trajectories follow."** / 中文："让训练流动起来"
- Logo 意象：一条穿过版本阶梯的水流（梯田瀑布 = versioned flow）
- 曾用名 River 弃用原因：与 PyPI 知名在线机器学习库 `river`（online-ml/river）同域冲突
- 落选候选：Current（双关完美但不可搜索）、Sluice（语义契合但生僻）、Rill（气质太小）

---

## 0. 一句话命题

**Serving 引擎把权重当常量、把请求当原子单元；RL 后训练把权重当版本流、把轨迹当跨版本的生命体。现有系统都在用错误的抽象做 rollout——Rheo 用正确的抽象重做一遍。**

四大开放问题（当前无系统解决好）：
- P1: KV cache 跨权重版本失效（partial rollout 的根本矛盾）
- P2: 投机解码 draft/MTP 头与持续进化的 policy 的共进化
- P3: 异步 rollout 的 staleness 无理论边界
- P4: "Rollout Engine" 品类缺位（训练框架和 serving 引擎都在打补丁）

对应四大核心机制：
- M1: **版本化权重管理**（WeightManager，双缓冲热切换 + token 边界暂停/换权重/续跑原语）
- M2: **KV 版本容忍**（KVT：把"KV 是否可复用"从二值问题变成带在线探针测量的连续控制问题）
- M3: **影子补全模式**（shadow-finish：长尾轨迹按出生版本精确跑完，CPU offload 精确权重）
- M4: **Staleness 控制器**（每条轨迹带出生版本，控制器按 KL 预算自动调速/校正）

---

## 1. 系统架构（自底向上四层 + 一个协议）

### L0 Kernel 层（Triton 起步，热点后置 C++/CUDA）
- 带版本标签的 paged attention kernel（KV block 元数据：birth_version, drift_epoch）
- 投机解码树验证 kernel（tree attention + 多 token 并行验证）
- 融合采样 kernel：逐轨迹 temperature/seed + **token 级 logprob 捕获**（RL 刚需：采样时 logprob = 出生版本策略的 logprob，TIS 校正免费获得）
- 3060 上全部 Triton 实现；与用户 FlashAttention/Triton 学习路线合并

### L1 Runtime 层（Python-first，profile 证明必要后下沉 C++）
- **WeightManager**
  - HBM 双缓冲：buffer A 服务生成，trainer 经 NVLink/PCIe/RDMA 向 buffer B 流式写入下一步权重，写完在安全点指针翻转（原子换权重）
  - 安全点 = **token 边界**：所有在途轨迹在 token 边界暂停（新原语），换权重后按各自策略决定继续方式（见 Scheduler）
  - 版本账本：全局单调 version id；每个 KV block、每条轨迹段、每个 logprob 都盖版本戳
  - delta 模式：支持低秩/量化 delta 本地应用（省带宽）
  - 影子副本：完整精确权重 CPU pinned 常驻（给 shadow-finish 用），低精度副本进 HBM（给 ε-stale 续跑用）
- **KVManager**
  - SGLang radix cache 块扩展版本元数据
  - 选择性失效 API：按 (版本距离 × 漂移估计) 决定整链失效/区间重算/容忍复用
- **Trajectory Runtime**
  - 轨迹段状态机：running / paused(token边界) / env-wait / finished / aborted(早停)
  - 每段携带：birth_version、token 区间、logprob 摘要、累计 drift

### L2 编排层（Python）
- **Rollout API**（引擎对训练器的接口）：
  ```python
  river.submit(groups=[(prompt, G, sampling_params, max_len)], deadline=...)
  river.sync_weights(state_dict_or_path, version=v, mode="stream")
  river.collect(budget=n_tokens_or_n_groups)  # 异步流式返回轨迹事件
  river.abort(group_id, reason="zero_variance")  # DAPO 动态采样引擎侧支持
  ```
- **Env Gateway**：多轮工具调用的异步拦截；env-wait 时 KV 分级留存（HBM → host 分页池 → 4bit 压缩 → 磁盘 checkpoint），等待时长预测按工具学习
- **Staleness 控制器**：每 batch 计算有效 staleness 直方图（出生版本 → 当前版本的 KL），超预算时二选一：训练端吸收（TIS 指数自调）/ 生成端加速换权重
- **Drift 估计器**：廉价信号（trainer 每步上报各层 ||ΔW||_F/||W||_F）+ 在线探针（每 N token 用新权重对滑动 128-token 窗口重算 logits，与缓存生成时 logits 算 KL）→ 驱动 KVT 阈值
- 遥测：rollout MFU、投机接受率、KV 复用率、staleness 直方图、token 经济学 —— 全部落 **RheoTrace** 格式

### L3 适配层（社区切入点）
- verl 适配器（第一优先，HybridFlow 是事实标准）
- 之后：slime / OpenRLHF / AReaL 适配器
- 引擎后端基底：**SGLang**（radix paging + overlap scheduler 离目标最近，且与用户 SGLang 社区路线协同）；vLLM 二期移植

### Rollout Protocol（社区护城河）
- gRPC/Arrow 开放协议：TrajectoryService { SubmitGroups / StreamEvents / SyncWeights / Heartbeat(drift) / Collect }
- 定位：rollout 界的 OpenAI API——谁先定义协议，谁定义品类
- RheoTrace 格式随协议一起开放（追踪数据格式的通用语言）

---

## 2. 十轮自迭代记录（每轮：发现的缺陷 → 纠正）

**迭代1｜致命缺陷：直接造独立引擎 = 没用户没负载没公信力。**
纠正：倒转顺序——先做 verl 的 drop-in rollout 后端适配器（寄生式起步），先发测量研究（RolloutBench），引擎从适配器里长出来。SGLang 也是先靠真实负载采用的。

**迭代2｜致命缺陷：ε-stale KV 复用数值上危险，一旦静默污染训练，项目公信力即死。**
纠正：把正确性变成可测量量——引擎永远返回采样时 logprob；在线探针每 N token 量化新旧权重下的分布偏移；默认保守（严格 on-policy 场景默认 re-prefill），激进模式 opt-in；每阶段论文先做"何时安全"的实证研究。

**迭代3｜致命缺陷：双权重版本影子副本 HBM 翻倍，12GB 卡上只够 1.5B，集群上挤占 KV 内存。**
纠正：shadow 是调度模式非常驻状态——只给长尾掉队者（<5% token）用；精确副本放 CPU pinned，按层按需 PCIe 上送（慢但掉队者少）；若用低精度副本跑，则该轨迹自动降级为 ε-stale 数据（诚实标注），不冒充精确。

**迭代4｜致命缺陷：MTP 投机解码假设模型自带 MTP 头，但主流开源 RL 模型（Qwen2.5/Llama）没有。**
纠正：draft 分级注册表：t0 n-gram/prompt-lookup（零成本，数学题前缀重复强，今天就能用）→ t1 self-spec 层跳 → t2 MTP（配合联合训练配方）→ t3 EAGLE-3 随环自动共训练（研究特性：draft 头对进化 policy 的 KL 追踪训练）。接受率遥测同时充当 staleness 信号——一举两得。

**迭代5｜致命缺陷：多轮 agent rollout 打破单 KV 流假设——env 等待数秒，KV 全留 HBM 会爆，丢弃则重算贵。**
纠正：env-wait KV 分级留存 + 按工具学习等待时长预测，成本模型决定留哪级；组内跨 env 调用连续批处理（A 等工具时解码 B、C）。

**迭代6｜致命缺陷：调度器成本模型自由参数太多，第一天没法标定。**
纠正：先建离散事件仿真器——用 RheoTrace 真实轨迹回放，调度策略离线仿真验证后再上线。3060 采轨迹，仿真实验任意规模。这也是测量论文的方法论。

**迭代7｜致命缺陷：范围失控——光权重同步就是数月工程，贪大求全会烂尾。**
纠正：行走骨架式分期，每期终止于可用工件（详见 §4 里程碑）：适配器+遥测 → 调度器v1 → WeightManager → KVT+shadow → 共进化投机 → 独立引擎+协议。

**迭代8｜致命缺陷：硬件错配——卖得出去的结果需要规模，3060 只有 12GB。**
纠正：双轨制：(a) 算法/调度研究在 0.5B-1.5B 上做（学术合法，大量论文用 0.5-7B）；(b) 里程碑级云爆发（8×A100 spot 租 2-3 天验证规模，$50-100/次）；(c) 引擎把消费级卡当一等公民——"端侧 RL 微调"是无人认领的差异化叙事。

**迭代9｜致命缺陷：社区计划空洞——verl 维护者凭什么采纳外来组件？**
纠正：木马式贡献路径：先把 RheoTrace 遥测/追踪格式以普通 PR 贡进 verl（他们本就需要 rollout 可观测性），格式成为生态通用语之后，引擎就是"说这个标准的东西"。社区里程碑：适配器 PR 合入 → 复现基准仓库 → 3 个外部用户 → workshop 论文 → 协议 v1 草案 + 2 框架采纳。

**迭代10｜致命缺陷：单点赌注风险——押注"RL rollout 永远是主导负载"，且个人带宽有限。**
纠正：(a) 核心原语（版本化权重/版本感知 KV/轨迹调度）同样服务在线评测、蒸馏数据生成、合成数据管线——定位写宽，入口收窄；(b) 计划是依赖有序 DAG，任意阶段停下都有独立产出；(c) 明确 AI 分工（AI 写代码/跑分析/写文档，人做研究判断、长训练值守、社区署名沟通）。

---

## 3. 技术栈与代码结构

```
rheo/
├── kernels/          # Triton: paged-attn(版本戳) / tree-verify / fused-sampling+logprob
├── runtime/
│   ├── weight_manager.py   # 双缓冲、版本账本、token边界安全点
│   ├── kv_manager.py       # radix扩展、drift元数据、选择性失效
│   ├── trajectory.py       # 段状态机、版本戳、logprob摘要
│   └── scheduler.py        # 成本模型: continue | shadow | ε-stale | re-prefill
├── orchestration/
│   ├── api.py              # submit/sync_weights/collect/abort
│   ├── env_gateway.py      # 工具调用拦截、KV分级、等待预测
│   ├── staleness.py        # KL预算控制器、TIS自调
│   ├── drift.py            # 探针 + 权重范数信号
│   └── telemetry.py        # → RheoTrace
├── adapters/
│   ├── verl/               # 第一优先
│   └── sglang/             # 后端基底集成
├── bench/                  # RolloutBench: workloads + traces + 回放器
├── sim/                    # 离散事件仿真器（调度策略离线验证）
└── protocol/               # Rollout Protocol v0 (gRPC/Arrow schema)
```

技术栈：Python 3.11+ / PyTorch / Triton（kernel）/ SGLang（基底）/ verl（训练侧）/ Arrow+gRPC（协议与追踪）/ SQLite+Grafana（遥测面板）。
语言策略：Python-first；L1 热路径在 profile 证明 >15% 开销后才下沉 C++/Rust。AI 写代码效率 Python 最高，先跑通再优化。

---

## 4. 里程碑（每期终止于独立可用工件）

| 阶段 | 周期 | 交付物 | 验收标准 |
|---|---|---|---|
| M0 起步 | wk1-2 | 3060 上 verl+Qwen2.5-1.5B GRPO 跑通（LoRA+8bit优化器）| 完整训练循环出 reward 曲线；基线遥测数据落 RheoTrace |
| M1 测量 | wk3-6 | RheoTrace v0 + 测量研究 v0 + 第一篇博客 | 回答"rollout GPU 周期去哪了"：MFU 分解、停顿分解、长尾分布 |
| M2 调度v1 | wk7-12 | 组感知调度 + DAPO 动态采样引擎化 + env-wait 分级（verl 可 PR 补丁）| RolloutBench 吞吐 +30-50%，博客#2 |
| M3 权重 | wk13-20 | WeightManager 双缓冲 + token 边界换权重原语（PR verl/SGLang）| 换权重停顿 <500ms（1.5B，3060）；墙钟提升数据 |
| M4 KVT | wk21-32 | drift 探针 + ε-stale 实证研究 + shadow 模式；论文草稿#1 | "KV 何时可跨版本复用"的 KL-漂移曲线 + 等效训练质量对照 |
| M5 共进化 | wk33-44 | draft 注册表全量 + EAGLE-3 随环共训练 + 接受率→staleness 联动 | 投机加速 × staleness 信号论文草稿#2 |
| M6 独立引擎 | wk45+ | Rheo v0 独立引擎 + Rollout Protocol v1 + 投稿（MLSys/ATC 2027）| 2 个训练框架跑在协议上；RolloutBench 全量数字公开 |

云爆发点：M2 末（8×A100 验证调度规模化）、M4 末（7B KVT 对照）、M6（全量基准）。

---

## 5. RolloutBench 评测设计

工作负载：
1. GRPO 数学推理（Qwen2.5-1.5B / 7B，DeepSeek-R1-distill 数据配方）
2. 多轮 agent 工具调用（ALFWorld 类环境，真实 env 延迟分布）
3. 长尾压力（双峰长度分布人工构造，专打 partial rollout 弱点）

指标：rollout MFU、tokens/s/GPU、到目标 reward 的墙钟、staleness-KL、$/checkpoint、KV 复用率、投机接受率。

基线：verl+vLLM、verl+SGLang、OpenRLHF、AReaL。

---

## 6. 风险与对策

| 风险 | 对策 |
|---|---|
| ε-stale 数值污染 | 探针+保守默认+论文先行（迭代2） |
| verl 上游拒 PR | 遥测格式先行，是他们自己要的东西（迭代9） |
| 规模可信度 | 云爆发里程碑硬性排入（迭代8） |
| 单人带宽 | DAG 化分期，任意点停止有产出（迭代10） |
| 赛道变化 | 原语服务多负载，定位写宽（迭代10） |

---

## 7. 分工

**AI（ZCode）**：kernel/调度器/适配器代码、实验脚本、数据分析、论文初稿、文档。
**人**：研究判断与设计决策、长训练值守、GitHub issue/PR 署名沟通（社区人格必须是人）、里程碑级云租预算决策。

## 8. 关键参考

AReaL / AsyncFlow / Echo / StreamRL / OrchestrRL / DORA / verl(HybridFlow) / System-Aware Self-Speculative Decoding for RL Rollouts / DAPO / SGLang RadixAttention / vLLM PagedAttention / EAGLE-3 / LMCache / Mooncake

---

## 9. 硬件与预算事实（2026-09-30 实测修正，优先级高于本文早期假设）

M0 实测（详见 `bench/results/m0-baseline/env.md`）：

- 实机为 RTX 3060 Laptop **6GB**（非迭代 8 假设的 12GB）；WSL2 下 Windows 桌面常驻占 1–2.5GB，训练实际可用 **3.5–4.5GB**。
- 已验证：Qwen2.5-1.5B + LoRA + GRPO + vLLM 0.12 在此卡上可完整跑通（M0 达成）。

对后续里程碑的修正：

- **默认实验规模改为 0.5B**（1.5B 为可选档）：调度器、仿真器、工作负载的全部"相对结论"（加速比、停顿占比、长尾行为）在 0.5B 上成立；绝对吞吐数字一律标注模型规模。
- **M3 双缓冲**：fp16 双份权重 0.5B≈2GB 本地可行；1.5B≈6GB 仅权重，本地不可行——1.5B/7B 的双缓冲与热切换验证上云单卡 A100/A800 80GB。
- **M4 影子副本**（CPU pinned 精确副本）不受影响；HBM 低精度副本按 0.5B 预算。

云爆发预算（2026-09 国内行情，AutoDL/恒源云/矩池云等，支付宝直付）：

| 用途 | 卡型 | 价格 |
|---|---|---|
| 中间规模实验（3B 级） | 单卡 4090 24GB | ¥1.3–2.5/卡时 |
| 1.5B 双缓冲、7B LoRA 对照 | 单卡 A100/A800 80GB | ¥5–10/卡时 |
| 里程碑发布数字（需 NVLink 可比性） | 8×A800 NVLink 整机 | ¥60–100/时；6–12h 聚焦实验 ≈ **¥400–1200/次** |

原则：大规模结论先在 `sim/` 仿真器跑（零租金），真机只做点验证；8 卡机选 A800/H800（A100 国内特供等效，NVLink 在、数字可比，H20 已停产勿选）；云端模型权重即用即弃，只拉回 trace/checkpoint/结果。
