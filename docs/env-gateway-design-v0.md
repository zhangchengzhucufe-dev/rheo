# Env Gateway v0 设计（接口冻结稿）

> 版本 v0.1 · 2026-10-01 · 会话 S4 起草
> 状态：**env-wait KV 分级与等待预测的接口冻结**。§3 拦截点、§4 分级层级与判据、§5 等待预测接口、
> §6 成本模型输入一经 merge 不再改动；只允许向后兼容增量（对齐 rheotrace-spec §8 的演进纪律）。
> 关联：PLAN.md §1 L2（Env Gateway）、§2 迭代 5、§4 M2 后半、§9 硬件事实；
> 轨迹状态机与事件语义见 `docs/rheotrace-spec-v0.md`（冻结）；
> 与 S1 调度器的对接见 `docs/scheduler-design-v0.md`（同日互审，§7 有逐条对账）。
> 消费方：**S1**（§7 对接点：唤醒后 re-prefill 判定、水位字段）、**S2**（仿真器可回放分级决策，TASK-S4 任务 4）、
> **S3**（agent-tool 延迟分布参数作预测器夹具，§5.4）。

---

## 1. 目的与范围

多轮 agent rollout 中，一条轨迹发起工具调用后进入 `env_wait`，等待从几十毫秒（快查询）到
几十秒（慢爬取/长计算）不等（PLAN 迭代 5）。此时它的 KV 既不能全留 HBM（批间累积直接爆池），
也不能一丢了之（唤醒后整段重算，prefill 贵且把"等待"变成了"计算"）。Env Gateway 解决的是：

1. **拦截**：在工具调用异步等待处接管该段 KV 的留存决策；
2. **分级留存**：HBM → host pinned → 4bit 压缩 → 磁盘 checkpoint 四级 + 丢弃（唤醒重算），
   由可插拔成本模型按（等待估计，KV 段大小，各级迁移带宽）决策；
3. **等待预测**：按工具学 EWMA，给成本模型供等待时长估计；
4. **等待中的自适应**：估计错了（预测短、实际长）能逐级下沉；轨迹 abort 能全级释放。

v0 交付（TASK-S4）：本文档 + `orchestration/env_gateway.py` 骨架 + mock 单测。
**明确不在 v0 范围**：不接真机引擎（集成排 M2 后半）；不做 4bit 量化/传输的本体实现（mock 层面
只表达"字节按压缩比缩减、按带宽计费"）；不改冻结 spec——KV 分级标记的事件化需求记
`docs/issues.md`（§8），由 spec 侧走增量。

设计立场（承接 PLAN 迭代 5/6 与 S1 设计立场）：

- **决策是纯函数**：分级决策输入全部显式（等待估计、段大小、水位、带宽表），无 IO、无时钟、
  无内部随机——S2 仿真器能逐位重放同一决策（与 S1 §4.4 同一契约）。
- **状态机不动**：`env_wait` 段的 KV 降级/升级是**资源动作**，不是轨迹状态转换——segment 始终
  停在 `env_wait`，不发 `segment_state` 事件。分级不与 S1 的 `paused` 语义纠缠（§7 对账 1）。
- **默认保守**：估计不可靠（样本少）时宁可留在高层级；成本模型算不过来时宁可重算——
  v0 的错误方向是"多留"，不是"丢数据"。

## 2. 术语

沿用 rheotrace-spec §2 与 scheduler-design §2：run / segment / group / version / batch / 决策点。
补充网关侧术语：

| 术语 | 含义 |
|---|---|
| **分级（tier）** | KV 段的存放层级，§4.1 的 T0–T3 + DROPPED |
| **降级（demote）** | KV 向低层级迁移（HBM → host → …），释放 HBM |
| **升级（promote）** | KV 向高层级迁回，唤醒前必须回到 T0（或转 DROPPED 重算） |
| **段大小** | 该段当前占用的 KV 字节数（prompt + 已生成 token 的 K/V 全量） |
| **等待估计** | 预测器对该（工具，历史）给出的等待时长分布摘要（§5.2） |
| **等待超支（overrun）** | 实际等待超过估计的分位界（§4.4），触发逐级下沉 |
| **版本风险** | 等待期间发生 weight_sync 的概率估计——决定"留存 vs 丢弃"的天平（§6.2） |

## 3. 拦截点（冻结）

```
running ──(工具调用发出)──► env_wait ──(工具结果返回)──► running
                │                            │
                │  on_env_call(seg, tool)    │  on_env_result(seg)
                │  → 分级决策 + 迁移          │  → 升级回 T0 / 转 re-prefill
                ▼                            ▼
          （等待期间 on_wait_tick 周期复查，超支则逐级下沉）
```

- **拦截时机**：引擎把工具请求发出、segment 状态转 `env_wait` 的那一刻（状态转换本身由引擎按
  spec §2 落 `segment_state`，gateway 不插手）。gateway 在此接管两件事：向预测器登记工具名、
  对该段 KV 执行分级决策。
- **唤醒时机**：工具结果返回、segment 转回 `running` 前——gateway 必须先完成升级（或判定
  重算），再放行状态转换。这是硬序：**先有可用 KV（或重算计划），后回 running**。
- **不做状态转换**：v0 不需要 `env_wait → paused`（S1 §11 开放项的回应见 §7 对账 1）；
  abort 的终态转换由引擎落 `segment_end`，gateway 只做资源释放（`on_abort`）。
- **与 S1 决策点的关系**：gateway 的决策发生在 D1–D4 之外（env_wait 段不可调度，S1 §2）；
  唤醒放行后该段重新可调度，若等待期间发生过 weight_sync，则按 S1 §8.3 进入 D2 判定
  （跨版本 → re-prefill），gateway 提供该段当时的 KV 层级作输入。

## 4. KV 分级判据（冻结）

### 4.1 层级表

| 级 | 名称 | 介质 | 有效容量系数 | 迁移带宽基准* | 说明 |
|---|---|---|---|---|---|
| **T0** | `HBM` | 引擎 KV 池 | 1.0（原样） | —（原地） | 唯一可直接续跑的层级；占用即挤压在途批 |
| **T1** | `HOST_PINNED` | host pinned 内存 | 1.0 | PCIe ≈ 8–16 GB/s（3060 Laptop 实测取 8） | 原样字节拷贝；容量充裕（GB 量级 vs HBM 池 1.5–2.5 GB） |
| **T2** | `COMPRESSED_4BIT` | host pinned（量化后） | 0.25 | 量化/反量化开销 + 0.25×PCIe | 精度损失 = ε-stale 之外的第二个降级源；M4 前只用于"留存后重验"路径，v0 不激活数值路径 |
| **T3** | `DISK_CHECKPOINT` | 本地磁盘 | 1.0 | NVMe ≈ 2–3 GB/s | 最廉价最慢；仅长等待 + 大段 |
| **D** | `DROPPED` | — | — | — | 丢弃；唤醒转 re-prefill（`prefill_len = n_prompt + n_gen`，签名对齐 S1 §7.2） |

\* 带宽是成本模型的**可配参数**（`TierSpec`），表值为 0.5B/6GB 档的默认依据；真机 A/B 回填。
容量系数：T2 存量化后字节（0.25×），迁移时按量化后字节计带宽。

### 4.2 决策规则（v0 语义，伪码冻结）

```python
def decide_tier(seg, wait, mem, p_sync, cfg) -> TierDecision:
    rt(t) = seg.eff_bytes(t) / cfg.eff_bw(t) + cfg.overhead_s(t)     # 双程迁移时间（T2 按量化后字节 + 量化开销）
    recompute = seg.prefill_len / cfg.prefill_tok_per_s              # 整段重算时间
    fits(t) = rt(t) <= wait.p90_s and cfg.capacity_ok(t, seg, mem)   # 窗口内迁得完且装得下

    if p_sync >= cfg.sync_risk_threshold:          # 等待期间大概率换权重：留存价值归零
        return DROPPED("version_risk")             # （对齐 S1 §8：v1 无 KVT，跨版本 KV 一律失效）
    if wait.p90_s <= cfg.hbm_hold_max_s and not mem.hbm_over_soft:
        return HBM("short_hold")                   # 短等待且池不紧：原地留住，迁移不值
    if best := argmin rt over {t ∈ (T1,T2,T3) | fits(t)}:
        return best                                # 需要腾池或等待长：窗口内迁得完的最快层级
    if rt(T3) < recompute and cfg.capacity_ok(T3, seg, mem):
        return T3("disk_fallback")                 # 窗口内迁不完但迁移仍远比重算便宜：兜底（保守方向"多留"）
    return DROPPED("window_or_cost")               # 极短窗口迁不完 / 重算更便宜：丢弃，唤醒重算
```

要点（单测逐条覆盖）：

1. **短等待 + 池不紧 → T0**：迁移有双程成本，等待短到迁移不划算时原地留住；
2. **池紧（软水位越过）→ 强制降级**：即使等待估计很短——HBM 是在途批的资源，env_wait 段
   没有理由占着（S1 §5.3：env_wait 段"不占批槽但占 KV"，gateway 的职责就是把这不占槽的占用
   压到最低）；
3. **降级目标 = 窗口内迁得完、双程最快的层级**：roundtrip ≤ wait.p90 才迁，否则白迁（还没迁完
   就唤醒了）；T1/T2/T3 按有效迁移时间取最优；
4. **极短窗口 / 重算更便宜 → DROPPED**：估计的 p90 比任何层级的双程迁移还短（估计错了或真的
   极短），或整段重算比迁到磁盘还快——丢弃，唤醒重算更诚实；
5. **兜底 T3**：长段 + 长等待、窗口判定失败但迁移成本仍远低于重算时，磁盘 checkpoint 总能
   放下——错误方向是"多留"，不是"丢数据"（§1 设计立场）；
6. **版本风险折扣**：`p_sync = 1 − exp(−wait.p90 / sync_interval)`（Poisson 近似，同步节奏由
   引擎账本供 `sync_interval`）；`p_sync ≥ 阈值`（默认 0.5）时留存价值归零直接 DROPPED——
   跨版本后 KV 反正要重算（对齐 S1 §8 "v1 无 KVT，跨版本 KV 一律视为失效"）。

### 4.3 迁移路径全集（单测覆盖矩阵）

| 路径 | 触发 | 断言 |
|---|---|---|
| T0 → 保持 T0 | 短等待 + 池不紧 | 无迁移事件 |
| T0 → T1 | 长等待 / 池紧，PCIe 足够 | nbytes 原样迁移，HBM 释放 |
| T0 → T2 | 池紧 + T1 带宽不够快（大段） | 有效字节 0.25× |
| T0 → T3 | 更长等待 + 更大段 | 磁盘落 checkpoint |
| T0 → D | 窗口内谁都迁不完 / drop_wins / 版本风险高 | 唤醒产 re-prefill 计划 |
| T1 → T2 / T2 → T3（下沉） | 等待超支（§4.4） | 从当前层级继续下迁，不回 T0 |
| T → T0（升级） | 唤醒，HBM 有位 | 放行 running |
| D → re-prefill | 唤醒 | WakePlan{mode:"re-prefill", prefill_len} |
| any → 释放 | abort（§4.5） | 全层级字节清零，后续唤醒报错 |

### 4.4 等待超支与逐级下沉（冻结）

预测给的是分布不是定值（§5.2）。gateway 在 env_wait 期间按引擎 tick（`on_wait_tick`，v0 由
mock 时钟驱动；真机集成后挂引擎的每步回调）复查：

- 实际等待 ≤ 估计 p90：不动；
- 实际等待 > 估计 p90 且当前层级 = T0：立即按 §4.2 以"已等时长为下界的更新估计"重新决策
  ——这是**估计错了的纠错路径**（预测短留了 T0，实际在拖）；
- 已在 T1–T3 且又超支（p99 级别）：继续下迁一级（T1→T2→T3）；
- 已在 T3：无处可去，停留（磁盘成本可忽略）。

### 4.5 边界行为（冻结）

| 边界 | 行为 |
|---|---|
| **显存满（硬）** | 池越过软水位：全部 env_wait 段按"剩余等待估计降序"逐段降级（还要等得越久越先放，快醒的留住），直到水位回落；全部降完仍超 → 报 `HbmExhausted`，由调用方（调度器 D4/v2 抢占）接手——v0 不抢在途段的 KV |
| **等待超时（引擎侧 timeout）** | 引擎放弃等待 → 走 abort 语义（reason 由引擎定），gateway `on_abort` 释放 |
| **轨迹 abort** | `on_abort(seg)`：从当前层级释放全部字节、清除预测器无关状态；此后 `on_env_result` 对同段报 `GatewayError`（幂等释放允许重复调用） |
| **唤醒时 HBM 仍满** | 升级推迟：WakePlan{can_resume: False}，段留在当前层级，gateway 在后续 tick 重试升级——不放行 running（硬序，§3） |
| **重复 on_env_call** | 报 `GatewayError`（一段同时只能有一次在途等待） |
| **未知工具** | 预测器回退全局估计（§5.3），决策照常 |

## 5. 等待时长预测接口（冻结）

### 5.1 定位

预测器是 gateway 的**可插拔组件**（`WaitPredictor` Protocol），v0 实现按工具 EWMA；
S2 仿真器与真机 A/B 用同一接口换更强预测器（分位数回归、按 (工具, 轮次) 条件化等）不加改动。

### 5.2 接口

```python
class WaitPredictor(Protocol):
    def predict(self, tool: str) -> WaitEstimate: ...
    def observe(self, tool: str, seconds: float, *, now: float | None = None) -> None: ...


@dataclass(frozen=True)
class WaitEstimate:
    mean_s: float  # 点估计
    p90_s: float  # 决策用上界（§4.2/§4.4 的窗口判定）
    p99_s: float  # 下沉判定（§4.4）
    n_samples: int  # 该工具自身样本数；0 = 未见过的工具（估计来自全局或先验回退）
```

- `observe` 由 gateway 在 `on_env_result` 时调用（真实等待时长回流）；
- 预测器无引擎依赖、无 IO；EWMA 状态在对象内，跨调用累积（与 S1 §4.4"策略无跨调用记忆"
  不冲突——预测器不是 policy，是引擎侧状态体，S1 §4.4 第 4 条正是这么安排的）。

### 5.3 按工具 EWMA（v0 实现）

- 每工具独立 EWMA：`m ← (1−α)·m + α·x`，二阶矩同法 → 方差 → p90/p99 用正态近似
  （`mean + z·σ`，z₉₀≈1.282、z₉₉≈2.326）。工具延迟分布右偏（lognormal 形态），正态分位数
  偏保守（高估 p90），对"迁得完"判定是安全方向；精确分位数留给更强预测器。
- **回退链**：该工具样本 < `min_samples` → 全局 EWMA（跨工具混合）→ 构造时给的先验
  （默认 1.0s）。回退是显式的（`n_samples` 如实上报），不冒充"学到了"。
- 非平稳追踪：α 默认 0.1（≈ 最近 10 个样本的有效窗口），分布漂移后 30 样本内收敛到新均值
  的 10% 以内（单测断言）。
- 数值约束：seconds 必须 > 0 且有限，违反报 `ValueError`（不让脏数据静默进 EWMA）。

### 5.4 测试夹具（S3 供料位）

TASK-S3 的 agent-tool 负载定义三类工具延迟（快查询 / 慢爬取 / 超长计算）。S3 的 D1 规格落定前，
单测用内置确定性合成（`random.Random(seed)` + 对数正态）：

| 工具类 | 中位延迟 | 用途 |
|---|---|---|
| `fast_query` | ~50 ms | 应留 T0 / 迁移不划算 |
| `slow_crawl` | ~2 s | 应降 T1 或 T2 |
| `xlong_compute` | ~30 s | 应降 T3 或 drop |

S3 规格落地后，以其分布参数替换夹具构造（接口不变）。

## 6. 成本模型（输入冻结，实现可插拔）

### 6.1 输入表（冻结）

| 输入 | 来源 | 用途 |
|---|---|---|
| 等待估计 `WaitEstimate` | §5 预测器 | 迁移窗口判定（p90/p99） |
| KV 段大小 `nbytes`、`prefill_len` | 引擎侧段元数据 | 迁移时间、重算成本 |
| 各级带宽 `TierSpec.bandwidth_bps` / 容量 | 配置（§4.1 表值起） | roundtrip 计算 |
| HBM 水位（已用/总量/软阈值） | 引擎 KV 池采样 | 强制降级触发；对齐 S1 `MemoryWatermark`（其 v2 预留的 `host_pinned_used_bytes` 由 gateway 状态回填） |
| host 侧预算（pinned 池余量、磁盘配额） | 配置 | T1/T2/T3 容量守卫 |
| 版本风险（同步节奏估计：run 内平均 sync 间隔 / 预测等待） | 引擎账本 | 留存价值折扣（§4.2 第 6 条） |
| prefill 吞吐基准 `prefill_tok_per_s` | 配置（0.5B 档真机回填） | 重算成本 |

### 6.2 可插拔面

`CostModel` Protocol：`decide(seg, wait, mem, risk, cfg) -> TierDecision`。
v0 默认实现 = §4.2 伪码的直接翻译（`ThresholdCostModel`）；S2 仿真器可实现同接口扫参
（带宽、软阈值、α）找拐点，产"待真机复核"结论——与 S1 §8.2 注记同一方法论。

### 6.3 与 S1 成本模型的分工（对齐 §8.3）

两个模型共享两类输入（等待估计、KV 段大小），各自独立决策、在 `resume` 处汇合：

- **gateway**（本文）：env_wait 期间"留哪级"——优化目标是等待结束时的总成本
  （迁移 + 留存挤压 + 重算风险）；
- **scheduler**（S1 §8）：唤醒后"怎么续跑"——v1 只有 re-prefill/abort 两支，M3/M4 激活
  shadow/ε-stale 后，gateway 的 T1/T2 存货才有"不回 T0 直接续跑"的消费方；
- 版本风险的口径两边一致：v0/M2 无 KVT，跨版本 = KV 全失效 = T1–T3 留存价值归零。
  KVT（M4）落地后此处是第一个要重标定的接缝（记开放问题 §9）。

## 7. 与 S1 轨迹状态机 / 调度器设计的对账（同日互审）

对 `docs/scheduler-design-v0.md`（v0.1，2026-09-30）逐条核对：

| # | S1 侧表述 | gateway 侧结论 | 状态 |
|---|---|---|---|
| 1 | §7.2/§11：`env_wait → paused` 直转非法；问 S4 分级留存是否需要"等待中冻结到 token 边界"语义 | **不需要**。分级迁移是资源动作，segment 全程停在 `env_wait`，无状态转换；S1 的开放项就此关闭，无需 spec 增量 | ✅ 无矛盾 |
| 2 | §8.3：唤醒后若跨版本 → D2 判定 re-prefill，"kv_action 由 S4 状态提供" | 唤醒硬序（§3）保证进 running 前 KV 已就绪或已判重算；gateway 的 `WakePlan.recompute` 与 S1 `ResumePlan.kv_action="drop"` 语义一致（drop 都指"不复用 KV"，触发源一个在成本模型、一个在版本失效） | ✅ 一致 |
| 3 | §4.2 `MemoryWatermark`：v2 预留 `host_pinned_used_bytes`（S4 分级留存），v1 恒缺省 | 确认：v1 无 gateway 时该字段缺省正确；M2 后半集成起由 gateway 回填，字段语义 = T1+T2 合计占用 | ✅ 预留对齐 |
| 4 | §5.3：env_wait 段"不占批槽但占 KV"，D4 水位"如实计入其占用" | 分级后 HBM 侧占用即时下降（T1–T3 不占池），D4 读到的 `kv_used_bytes` 自动反映降级收益——S1 的 v1 水位语义（仅 HBM 池）无需改动 | ✅ 一致，且互补 |
| 5 | §3 D4：v1 只观察不抢占；抢占留 v2 | gateway 的显存满边界（§4.5）同样只降 env_wait 段、不碰在途段，缺额上报 `HbmExhausted` 交给调度器——两层都不越权 | ✅ 职责边界一致 |
| 6 | §4.4 确定性契约（纯函数、无时钟） | gateway 决策同理纯函数化（时间以显式参数进 `on_wait_tick`/`on_env_call`，mock 时钟注入），S2 可重放 | ✅ 同一契约 |

**结论：无阻断性矛盾**。增量需求（trace 事件化）见 §8 与 `docs/issues.md`。

## 8. trace 事件缺口（记 issues.md，不改冻结 spec）

分级迁移在 v0 只产生**内存内 `GatewayEvent` 流**（kind / seg_id / from_tier / to_tier / t_ns /
reason），不落 rheotrace。要进 trace 需要增量（spec §8 途径，S4→B/S2 提出）：

1. `kv_tier` 事件类型（或 `segment_state.reason` 扩展枚举——后者混语义，不建议）；
2. `segment_start` / `segment_end` 上可选 `kv_tier_final` 标记（唤醒即重算的段，分析侧需要知道
   它没从分级里受益）。

v0 的 `GatewayEvent` 字段按未来事件形状设计，增量合入后直转。

## 9. 开放问题（不阻塞 v0）

1. **T2（4bit）的数值语义**：降级到 T2 的 KV 唤醒后直接续跑是否可接受（等价一次 ε-stale），
   还是必须重验/重算——M4 KVT 实证研究回答，v0 一律按"回来要重验"保守处理（不激活数值路径）；
2. **重算 vs 留存的版本风险定价**：§6.2 的 `risk` 输入 v0 用平均 sync 间隔近似，M3 WeightManager
   的同步节奏可预测后重标定；
3. **组内相关性**：同组 G 条常同时等同一工具（§5.4 同类延迟），批级联合降级 vs 逐段独立决策——
   迭代 5 后半"组内跨 env 调用连续批处理"的前置问题，预研文档另出；
4. **p90/p99 正态近似的偏差**：重尾工具（xlong_compute）上会低估分位数，更强预测器（经验分位数）
   是接口换件，不改冻结面。

## 10. 实施映射（v0）

| 设计物 | 代码落点 | 验收 |
|---|---|---|
| §4.1/§4.2 分级决策 | `orchestration/env_gateway.py`（`KVTier`/`TierSpec`/`ThresholdCostModel`） | mock 单测覆盖 §4.3 全路径矩阵 |
| §4.4/§4.5 边界 | 同上（`EnvGateway.on_wait_tick`/`on_abort`/`HbmExhausted`） | 单测：显存满逐段降级、等待超支下沉、abort 幂等释放 |
| §5 预测接口 | 同上（`WaitPredictor`/`EWMAWaitPredictor`） | 单测：三类工具收敛分离、回退链、非平稳追踪、脏输入拒绝 |
| gateway 状态机 | 同上（`on_env_call`/`on_env_result` 硬序） | 单测：重复拦截/迟到唤醒报错，全流程事件序可断言 |
| 真机集成 | M2 后半（不在本期） | — |
