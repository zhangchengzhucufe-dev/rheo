# Rollout 分析指标 v0（口径定稿）

> 状态：v0 定稿，2026-09-29，分支 `feat/analysis` 第 1 天。
> 用途：M1 "rollout GPU 周期去哪了" 的统一口径。`bench/analysis` 的实现、输出报告，以及后续博客/论文引用的数字，**一律以本文为准**。
> 上游依赖：RheoTrace v0 trace 文件（规格 `docs/rheotrace-spec-v0.md`，B 负责，尚未定稿）。本文 §6 列出分析侧的最小字段需求，作为对 B 规格的对账清单；缺口已记 `docs/issues.md`。
> 范围：v0 只覆盖 rollout 阶段（prefill / decode / 停顿），不含训练侧（backward、optimizer）时间。投机解码、KV 复用率、staleness-KL、$/checkpoint 等留 v1（§8）。

## 0. 记号与基本约定

| 符号 | 含义 |
|---|---|
| `T_wall` | rollout 墙钟：trace 内最早事件开始到最晚事件结束 |
| `T_active` | 至少有一条轨迹在 GPU 上执行 prefill/decode 的时间（§3.1） |
| `T_pause` | `T_wall - T_active`：引擎零执行的时间 |
| `N_gpu` | 参与 rollout 的 GPU 数 |
| `peak` | 硬件稠密算力峰值（BF16/FP16 tensor core，TFLOPS），运行参数可覆盖 |
| `P` / `L` / `d` | 模型参数量 / 层数 / 隐藏维（hidden size，按 Q 侧算） |

**时间基准**：单条 trace 单一单调时钟，精度 ns；所有区间**左闭右开** `[t0, t1)`，长度 = `t1 - t0`。此约定必须写进 RheoTrace spec（见 §6）。

**Token 计数三条铁律**（所有指标共用，避免口径打架）：

1. prefill token 与 decode token **永不合并**；吞吐与 MFU 要么分开报告，要么注明口径。
2. decode 计数只算**提交给训练侧的 token**（v0 无投机解码时即全部 decode token；v1 引入投机后只算被接受的）。
3. 被 abort 的轨迹（如 DAPO 零方差中止）已生成的 token 计入"全部生成"，单独打标：MFU 分母包含它（GPU 周期确实花了），吞吐与长度分布给"含/不含 abort"两列。

**分析单元**：一个 trace 文件 = 一个分析单元。若 trace 跨多个 rollout 阶段（以权重同步事件为界自动切分），逐阶段出报告并给一行全程聚合。

## 1. 吞吐：tokens/s/GPU

**T1 端到端吞吐（headline）**

```
T1 = total_decode_tokens / T_wall / N_gpu
```

分母是整段墙钟，含一切停顿。这是训练侧感知到的真实数据供给速度，**所有系统对比用这个数**。

**T2 引擎活跃吞吐**

```
T2 = total_decode_tokens / T_active / N_gpu
```

T1/T2 = 引擎占空比（≤1），差值即停顿开销。T1 与 T2 必须同时报告。

**T3 prefill 吞吐**（诊断用，不进 headline）

```
T3 = total_prompt_tokens / |∪ prefill 区间| / N_gpu
```

分母取 prefill 活跃区间的**并集**：并发 prefill 共享一份 GPU 时间，按段求和会重复计时。re-prefill（暂停续跑重算）的 token 照实计入分子——它是真实计算量（实现见 `bench/analysis`，RheoTrace 的 `meta.reason="re-prefill"` span 同样计入）。

并行布局说明：张量并行时单 GPU 吞吐照此定义（总 token / 墙钟 / N_gpu）；数据并行分片（不同 GPU 跑不同轨迹）时同样成立。v0 不按布局细分，报告统一给 per-GPU 值，布局进元数据（§6）。

## 2. MFU 分解

### 2.1 有用 FLOPs（解析式，v0 不跑 profiler）

MAC = 2 FLOPs 计法。

- **decode** 每生成 1 个 token：

  ```
  flops_dec(s) = 2·P + 4·L·d·s
  ```

  第一项为稠密前向（2·P 近似，embedding 与 tied head 误差接受）；第二项为 attention 对全部缓存 KV 的 QK^T 与 softmax·V，`s` 为该 token 前向时的 KV 长度（含当前 token）。GQA 下该式按 Q 侧 d 计，K/V 头缩减影响的是访存不是分数计算。

- **prefill** 每 prompt token：`flops_pre = 2·P`。causal attention 的 `Σ s_i/2` 项 v0 略去（误差 <10%，接受；v1 可选 full 模式）。

- `s` 从 trace 段级 token 区间推得；trace 只给段起止长度时用段内均值。

**参数获取**：`P / L / d` 由运行参数 `--model-config <json>` 或内置常见模型表提供；trace 元数据若带 model 名则校验一致，不一致报错。

**废 token**：MFU 分母用"全部生成 token"（见铁律 3）。废 token 率 `waste = 1 − committed / generated` 单独报告，不混入 MFU。

### 2.2 两级 MFU（v0）

| ID | 名称 | 定义 | 回答的问题 |
|---|---|---|---|
| **M1** | 引擎 MFU | `useful_flops / (peak · T_active)` | 引擎在算的时候，算得效率如何（批形状、访存、kernel） |
| **M2** | rollout MFU | `useful_flops / (peak · T_wall)` | 训练侧看到的端到端算力利用率，**headline** |

恒等式（停顿分解的骨架，报告必须三项同列）：

```
M2 = M1 × (T_active / T_wall)
```

一眼可判"损失在不在引擎里"：M1 低 → 引擎内部问题（kernel/批形状）；M1 高但 T_active/T_wall 低 → 停顿问题（§3）。

kernel 级 MFU 需 kernel 级事件，留 v1。

### 2.3 peak 取值

`peak` 用硬件数据表稠密峰值（未计稀疏），与训练侧 MFU 的惯用口径一致，因此 M1/M2 是**保守上界**。支持 `--peak-tflops` 覆盖为实测 GEMM 峰值（**推荐**，数据表值偏乐观）。内置参考表（占位，实现时以实测校准）：

| GPU | peak（TFLOPS，稠密） |
|---|---|
| RTX 3060 | ≈ 25.3（FP16 稠密、FP16 累加；FP32 累加约减半） |
| A100 | 312（BF16） |
| H100 SXM | 989（BF16） |

## 3. 停顿分类口径（rollout 墙钟去哪了）

### 3.1 总纲：墙钟先切两块

```
T_wall = T_active + T_pause
```

- `T_active`：至少一条轨迹在 GPU 上执行 prefill/decode。
- `T_pause`：引擎零执行。

**长尾掉队不是 pause**：有轨迹在跑，GPU 就在 active。掉队表现为批占用率下降，在 T_active 内部继续分（§3.3）。把掉队算成 pause 是常见口径错误，v0 明确禁止。

### 3.2 T_pause 四分类（互斥，按固定优先级归属）

对 pause 时间轴做**基本区间扫描**：P1 覆盖段先剔除；剩余部分在轨迹出生/终止、env 等待进/出这些边界点切分，每个基本区间内"所有在途轨迹同时 env_wait"的谓词取值恒定，据此精确归入 P2/P3（不做中点近似）。

| 优先级 | ID | 类别 | 判定条件 | 说明 |
|---|---|---|---|---|
| 1 | **P1** | weight_sync | 权重同步事件区间覆盖该时刻 | 含传输 + 应用 + 指针翻转全程；即使有轨迹恰在 env-wait，也归 P1——同步是可行动作，优先完整暴露 |
| 2 | **P2** | env_wait | 所有在途轨迹同时处于 env-wait | 只统计**全局阻塞**的 env 等待；个别轨迹的等待在 §3.4 按轨迹统计，两边都报 |
| 3 | **P3** | schedule_gap | 以上都不是：引擎可调度但没派活 | 换批间隙、调度开销、collect 边界等 |
| 4 | **P4** | other | 兜底 | 应趋近 0；占比 >1% 触发数据质量告警（Q1），通常意味着 §6 字段缺失 |

输出：P1–P4 占 `T_pause` 与占 `T_wall` 的两张占比表 + **pause 瀑布图**（时间轴水平条形，按优先级配色）。

### 3.3 批内视角：占用率与掉队（S 类）

需要 trace 给每个执行段标 `batch_id`（§6）。定义：
- `B0(batch)` = 该批派发时的轨迹数；
- `k(t)` = 时刻 t 该批仍在执行的轨迹数；
- 批占用率 `occ(t) = k(t) / B0`。

| ID | 指标 | 定义 |
|---|---|---|
| **S1** | 掉队份额 | `T_straggler / T_active`，其中 `T_straggler = Σ 1[occ(t) < 0.5] · dt`（对每个活跃批分别累计后求和）。含义：GPU 在跑，但只服务零星掉队者——这正是 partial rollout / shadow-finish 要收割的时间。阈值 0.5 为 v0 固定值，报告注明 |
| **S2** | 批尾比 | 每批 `(t_last − t_p50) / t_p50`，报告跨批 P50 与 P90。衡量"一批等最慢者"的代价 |
| **S3** | 占用率曲线 | occ(t) 全程曲线（图），双峰长尾负载下应呈阶梯式跌落 |

batch 标注缺失时的降级口径（RheoTrace v0 冻结 spec 无 batch 概念，见 §6/`docs/issues.md`）：按轨迹包络时间重叠聚类出伪批计算 S1–S3，并在报告中打 `W-NO-BATCH` 警告——聚类会把跨批残留的长尾并进新批，**偏乐观，结果只当下界**。

### 3.4 轨迹级 env 等待（非全局）

- 每条轨迹的 env_wait 总时长分布（P50 / P90 / max）；
- env_wait 占该轨迹墙钟的比例分布。

多轮 agent 负载的核心诊断量：全局 P2 只捕捉"所有轨迹同时等"的时刻，轨迹级分布捕捉"谁在等、等多久"。

## 4. 轨迹长度分布与长尾统计

**长度定义**：单条轨迹的 decode token 数（铁律 2/3：只算提交的；abort 轨迹打标，给"含/不含 abort"两列）。trace 结束仍未完成的轨迹**不入长度分布**，计数进数据质量（Q2）。

| ID | 指标 | 定义 |
|---|---|---|
| **F1** | 基础统计 | n、mean、std、CV、min / P10 / P25 / P50 / P75 / P90 / P95 / P99 / max |
| **F2** | 长尾比 | `P99/P50`、`max/P50`、**尾部 token 份额** = 长度 > P90 的轨迹的 token 数 / 总 token 数（partial rollout 收益面的第一度量） |
| **F3** | 双峰检验 | Sarle 双峰系数 `BC = (g1² + 1) / (γ2 + 3(n−1)²/((n−2)(n−3)))`，g1 为样本偏度、γ2 为超额峰度（m4/m2²−3）。`BC > 5/9` 判双峰迹象（5/9 恰为均匀分布的 BC 值）；配合直方图（log 轴）人工复核。v0 不引入 scipy/Hartigan dip，BC + 直方图够用且零重依赖 |
| **F4** | 长度-时长散点 | 轨迹长度 vs 墙钟时长（图）。env 等待重的轨迹表现为"短长度、长时长"的离群带 |
| **F5** | stale 暴露面 | `stale_token_share` = 生成时刻权重版本 ≠ 该轨迹 birth_version 的 token 份额。需要 trace 的版本时间线（§6）。这是"跨版本继续生成"机制收益上界的第一近似 |

## 5. 报告契约（`bench/analysis` 输出）

```
python -m bench.analysis <trace> [--out DIR] [--peak-tflops X] [--model-config f.json]
```

`--out` 默认 `bench/results/<trace名>-<起始t0纳秒>/`，产出 `report.md` + `figures/*.png`（md 内相对引用）：

```
report.md
├── 0 元数据：模型、peak、N_gpu、token 总量、T_wall、trace 来源与哈希、FLOPs 模型声明
├── 1 吞吐：T1 / T2 / T3 + 占空比 T_active/T_wall
├── 2 MFU 分解：M1 / M2 + 恒等式数值校验 + waste
├── 3 停顿分解：P1–P4 两张占比表 + S1 / S2 + pause 瀑布图 + occupancy 曲线（S3）
├── 4 长度分布：F1 / F2 / F3 / F5 表 + 直方图 + 长度-时长散点（F4）
└── 5 数据质量：P4 占比告警（Q1）、未完成轨迹数（Q2）、时钟单调检查（Q3）、字段缺失告警
```

依赖：numpy、matplotlib（实现合入时加 `analysis` optional-dependencies，v0 文档阶段不装）。

## 6. 与 RheoTrace v0（B，spec v0.1 冻结稿）的对账结果

2026-09-29 与 `docs/rheotrace-spec-v0.md` 逐项核对。**字段命名以 B 的 spec 为准**；`bench/analysis/adapters.py` 完成 rheotrace-jsonl → canon 的映射，指标不动。

| 本文档需求（原 §6） | spec 对应 | 结论 |
|---|---|---|
| 轨迹段生命周期（seg/traj/group、状态机、区间） | `segment_start/state/end`、`phase_span`，状态机 §2 | ✅ 满足；traj:=seg 一一对应 |
| 权重同步**显式区间** | `weight_sync{t_start,t_end,version,mode}` | ✅ 满足；且"新版本自 t_end 生效"（§5.1）比我方口径更精确 |
| 版本时间线 + birth_version | `weight_sync` 账本 + `birth_version`/`end_version` | ✅ 满足，F5 按账本在 span.t_start 的生效版本重放 |
| env 调用区间 | `segment_state` env_wait 状态区间（+ 可选 span） | ✅ 满足；canon 取状态区间 |
| 元数据头（model/world_size/时钟） | `run_start{model,clock,n_workers,meta}` | ✅ 基本满足；P/L/d 不在 spec，走 `--model-config` 或内置表（模型名含 "1.5b" 等规模记号时按内置表近似） |
| 时钟约定（单调、ns、排序） | ns 整数、事件按 ts 非降序（E04/E18） | ✅ 满足 |
| abort 语义（reason） | `segment_end{state=aborted, reason 必填}`（E15） | ✅ 满足；committed 以"非 aborted"近似 |
| 投机解码预留 | §8 明确 M5 增量事件，兼容路径成立 | ✅ 接受其显式推迟 |
| **batch_id** | spec 全文无批概念 | ❌ **缺口**，S1–S3 核心依赖 → 已提 issues.md 请求向后兼容增量；过渡期按 §3.3 降级口径 |
| 区间开闭语义 | spec 未写明 | ⚠️ 小缺口 → 已提 issues.md；C 侧统一按左闭右开处理 |
| prefill/decode span 的 `n_tokens` | spec 标可选 | ⚠️ C 分析必需 → 已提 issues.md（请 A 插桩必填）；适配器遇缺失直接报错不静默估 |

冻结前的原始需求清单已被上表取代；对 B 的待办以 `docs/issues.md` 为准。

## 7. 验收对照（TASK-C 第 1 天项）

- [x] rollout MFU 分解口径（§2）
- [x] 停顿分类口径：权重同步 / env 等待 / 调度间隙 / 长尾掉队（§3）
- [x] tokens/s/GPU（§1）
- [x] 轨迹长度分布与长尾统计（§4）
- [x] 与 B 的 spec 对账清单（§6 + `docs/issues.md`）
- [ ] `bench/analysis` 按本口径实现出数（第 2 天起）

## 8. v1 预留（不做承诺）

kernel 级 MFU、投机接受率、KV 复用率、staleness-KL、Hartigan dip 检验、$/checkpoint、prefill attention full 模式、多阶段对比视图。
