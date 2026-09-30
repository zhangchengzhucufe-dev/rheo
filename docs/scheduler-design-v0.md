# Scheduler v0 设计（接口冻结稿）

> 版本 v0.1 · 2026-09-30 · 会话 S1 起草
> 状态：**M2 调度接口冻结**。§4 策略接口、§7 token 边界原语签名、§8 成本模型输入表一经 merge 不再改动；
> 后续只允许向后兼容增量（新增可选字段/新增决策点，对齐 rheotrace-spec §8 的演进纪律）。
> 关联：PLAN.md §1 L1/L2、§2 迭代6、§4 M2、§9 硬件事实；trace 语义见 `docs/rheotrace-spec-v0.md`（冻结）；
> 指标口径见 `docs/metrics-v0.md`（冻结）；env-wait 分级见 S4 的 `docs/env-gateway-design-v0.md`（同日互审）。
> 消费方：**S2**（仿真器按 §4 接口实现策略 harness）、**S4**（按 §7 对齐 pause/resume 与 env_wait 的边界）、
> **A 插桩**（按 §5.3/§6.4 填 `batch_id` 与 abort reason）。

---

## 1. 目的与范围

本文冻结调度器的**对外接口与判定输入**，不冻结实现。v1（M2 前半）交付：

1. 组感知调度：同组 prompt 共批（§5）；
2. DAPO 零方差组 abort，引擎侧中止、reason 落 trace、废 token 可对账（§6）；
3. token 边界暂停原语的**接口位**（本体 M3 WeightManager 实现，§7）；
4. 成本模型四分支中**只激活 continue 与 re-prefill**（含 DAPO abort 路径），shadow/ε-stale 留空位（§8）。

明确不在 v1 范围：WeightManager/KVT/kernels 本体（M3/M4）；shadow/ε-stale 的实现与数值实验（只留判定输入表与接口位）；
显存抢占式换入换出（§3 D4 只观察上报）；改 rheotrace spec（需求走 `docs/issues.md`）。

设计立场（承接 PLAN 迭代 2/3/6）：

- **策略是纯函数**：`policy(observation) → decision` 无 IO、无时钟、无内部随机（seed 显式注入）。
  这是 S2 能在仿真器里逐位重放同一策略的前提，也是真机 A/B 可归因的前提。
- **默认保守**：v1 无 KVT，凡跨权重版本，KV 一律视为失效——跨版本续跑只有 re-prefill 一条路
  （或按策略 abort）。ε-stale 是 M4 的 opt-in，v1 不冒充精确。
- **组是调度与对账的最小逻辑单元**：批是调度单元（`batch_id`），组是采样单元（`group_id`），
  两者概念分离（spec §4.4 已冻结此区分），调度器负责把前者对齐到后者。

## 2. 术语

沿用 rheotrace-spec §2：run / segment / group / version / batch。补充调度侧术语：

| 术语 | 含义 |
|---|---|
| **决策点** | 调度器调用 policy 的时机（§3，D1–D4） |
| **token 边界** | 一条轨迹两个 decode token 之间的安全点；weight_sync 安全点 = 全体在途轨迹到达 token 边界 |
| **在途（in-flight）** | 状态 ∈ {running, paused, env_wait} 的段；env_wait 段不可调度（等工具），但占 KV |
| **组生命期** | 从组内首段 segment_start 到组内全部段终态（finished/aborted） |
| **废 token** | aborted 段已产出、未提交训练侧的 token（metrics-v0 铁律 3 的口径） |

## 3. 决策点（v1 全集）

调度器只在以下四个时机调用 policy；其余时间不干预，引擎按 vLLM/SGLang 原生连续批处理运转。

| ID | 决策点 | 触发 | v1 行为 |
|---|---|---|---|
| **D1** | 批组装 | 引擎有空槽且有待派段 | 组感知共批（§5.2）；observation 含候选组与显存水位 |
| **D2** | token 边界 | `weight_sync.t_start` 到来（全体强制到界）；或策略后续扩展的主动抢占 | 全体在途段 → `paused`（reason=`token_boundary`）；新权重生效后逐段决定续跑方式（§8：v1 = re-prefill 或 abort） |
| **D3** | 组完成回调 | 组内每有一条段 finished 且奖励可得 | 零方差检测（§6.3）；命中则 `abort(group, reason)` |
| **D4** | 显存压力 | KV 水位越过软阈值（周期采样） | v1 只把水位写进 observation 与遥测，**不**主动抢占；抢占留 v2 |

> D4 的"软阈值越过"事件本身 v1 也不改变调度——它存在于决策点清单里是为了冻结 observation 形状，
> 避免 v2 加抢占时破坏接口。

## 4. 策略接口（冻结）

### 4.1 签名

```python
Policy = Callable[[Observation], Decision]
```

策略即纯函数。命名策略用薄类包装以便携带参数，但 `__call__` 遵守同一签名与纯度要求。
代码位于 `runtime/scheduler.py`（v1 纯逻辑 + mock 单测，不依赖 torch/SGLang）；
S2 仿真器 `from runtime.scheduler import Observation, Decision, Policy` 直接复用同一类型。

### 4.2 Observation（只读快照，冻结字段）

```python
@dataclass(frozen=True)
class MemoryWatermark:
    kv_used_bytes: int  # 引擎 KV 池已用
    kv_total_bytes: int  # 引擎 KV 池总量
    # v2 预留：host_pinned_used_bytes 等（S4 分级留存），v1 恒缺省


@dataclass(frozen=True)
class SegmentView:
    seg_id: str
    group_id: str
    state: Literal["running", "paused", "env_wait"]
    n_prompt_tokens: int
    n_gen_tokens: int
    max_new_tokens: int | None  # 引擎可得时填写
    birth_version: int
    current_version: int  # 该段最近一次前向所用的权重版本（v1 == birth_version 或上一同步版本）
    batch_id: str | None  # 已派发段才有
    finish_mode: Literal["exact", "shadow", "stale"] | None  # paused 段的预定续跑方式（§8）


@dataclass(frozen=True)
class GroupView:
    group_id: str
    n_total: int  # G
    n_queued: int  # 已登记未派发（调度核心私有初态：trace 上无存在，不进 Observation.segments）
    n_running: int
    n_paused: int
    n_env_wait: int
    n_finished: int
    n_aborted: int
    finished_rewards: tuple[float, ...]  # 已完成成员奖励，按完成序；引擎拿不到奖励时为空 tuple
    dispatched: bool  # 是否已有段进入过 GPU


@dataclass(frozen=True)
class Observation:
    t_now_ns: int  # 决策时刻（wall ns；仿真器为虚拟钟——策略不得据此做绝对时间判断）
    current_version: int  # 当前生效权重版本
    pending_sync: bool  # True = weight_sync 安全点已到/正在逼近（D2 上下文）
    segments: tuple[SegmentView, ...]  # 全部在途段
    groups: tuple[GroupView, ...]  # 全部未收尾组（含待派与在途）
    memory: MemoryWatermark
    candidates: tuple[str, ...]  # D1 时：可派组 id 列表；其余决策点为空
```

冻结规则：

- 字段只增不改。v2 新增字段一律可选（有默认值），旧策略读到未知字段忽略。
- `Observation` 是**值快照**：策略不得假定两次调用之间的关联，也不得持有引用做写操作。
- `t_now_ns` 仅供日志；策略内禁止绝对时间判断（仿真重放时虚拟钟会骗人），相对时长一律用
  token 计数/版本距离表达。

### 4.3 Decision（冻结枚举与载荷）

```python
@dataclass(frozen=True)
class Decision:
    action: Literal["continue", "pause", "re-prefill", "abort"]
    targets: tuple[str, ...] = ()  # seg_id 列表；abort 时为空（组级，见 group_id）
    group_id: str | None = None  # 仅 abort：目标组，组内全部未终态段中止
    reason: str | None = None  # abort 必填（落 segment_end.reason）；pause/re-prefill 可选备注
```

| action | 语义 | 合法载荷 | 引擎侧效果（trace 映射） |
|---|---|---|---|
| `continue` | 维持现状（默认） | 无 | 无事件（v1 批内常态） |
| `pause` | 令 targets 在最近 token 边界暂停 | targets | `segment_state running→paused`，reason 如 `token_boundary` |
| `re-prefill` | targets 以当前生效版本重算 prefill 后续跑 | targets | `paused→running`；随后 `phase_span prefill` 且 `meta.reason="re-prefill"` |
| `abort` | 组级中止 | group_id + reason（必填） | 组内未终态段逐段 `segment_end{state:"aborted", reason}` |

约束（引擎校验，违反即错误）：

- `abort.reason` 必须 ∈ §6.4 的 reason 注册表（对齐 spec E15：aborted 必带 reason）。
- `re-prefill` 的 targets 必须处于 `paused`（安全点后续跑）**或 `env_wait`**（工具唤醒时发现
  跨版本，§8.3：唤醒即重算，无需 paused 中转）；`pause` 的 targets 必须处于 `running`。
- `abort` 只作用于未收尾组；组内已 finished 段不受影响（其数据去留见 §6.5）。
- D2 上下文（`pending_sync=True`）下，只要仍有 running 段，policy 就不得返回 `continue`——
  安全点是强制的；全体在途段已到界（running 集为空）后，`continue` 合法且表示"放行指针翻转"。
  policy 返回后引擎仍有一道防线：非法决策按异常处理并落 `schedule` span 备注。

### 4.4 确定性要求（S2 重放契约）

1. 纯函数：无 IO、无全局可变状态、无 `time`/`random` 直接调用；
2. 需要随机性（如 ε-探索）时从 `Observation` 无法获得——**v1 策略不引入随机**；
   未来需要时以显式 `seed` 参数进构造器，并在 trace `meta` 记录；
3. 同一 `Observation` 必产同一 `Decision`（值相等 ⇒ 决策相等），这是仿真器逐位重放与真机行为
   可比的充要条件；
4. 策略不得依赖调用序（无跨调用记忆）；需要状态（如 EWMA）的状态体放在引擎侧、经 observation
   传入——v1 无此需求，接口位不加。

## 5. 组感知调度（D1）

### 5.1 为什么共批

GRPO/DAPO 的优势在组内归一化，组是数据有效性单位：组内成员长度相近（同 prompt）、
早停与零方差检测都以组为单位生效。共批把三类开销同时压掉：

1. **批占用率**：组内 G 条同生同灭，批占用率曲线不呈长尾阶梯（metrics-v0 S1/S3 的掉队来源之一）；
2. **零方差 abort 的收益面**：整组同批，abort 时无跨批牵连，释放的是整段连续 KV 与完整算力槽；
3. **对账**：batch_id × group_id 的对应关系简单（一至多），废 token 率可按组精确归因。

### 5.2 共批规则（v1 冻结）

- 批的装配单位是**整组**：一个 batch 含 k 个完整组（k ≥ 1），**禁止把组拆到多个批**；
- 装配顺序：待派组按提交序（FIFO）；policy 可在 D1 通过不派发候选组来表达错峰
  （返回 `continue` 即"本波不派"），显式错峰策略由 S2 在 harness 里实现，引擎不内置；
- 容量约束：k 的上限由 KV 预算推得——`Σ_group (G × E[max_new_tokens])` 不超过当前可用 KV 池
  （预算表见 §9）；超限则减 k，不拆组；
- **queued 初态**：submit 时组内段登记为 `queued`（已登记未派发；trace 上无存在——`segment_start`
  未发，故不进 `Observation.segments`），派发（进批）才转 `running`，并以**派发时刻的生效版本**
  盖 `birth_version`（排队段可能在 weight_sync 之后才派发；引擎据此回填 `segment_start.birth_version`）。
  安全点 pause 与 re-prefill 均不触及 queued 段；组 abort 覆盖 queued 段（零 token 记账，不产生
  trace 事件），已 abort 的组不再入批（防僵尸派发）；
- 同组内所有段写入同一 `batch_id`；`batch_id` 由调度器生成（`b-<run 内序号>`），经
  `segment_start.batch_id`（spec v0.1.1 已就位）落 trace，A 插桩从调度器取值回填。

### 5.3 引擎不入批的段

`env_wait` 段不占批槽但占 KV；其 KV 分级去留由 S4 的 env-gateway 决策（成本模型对接点见 §8.3），
调度器只保证：env_wait 段的 KV 状态变化不违反共批组的批内一致性假设（v1：组内任一成员在
env_wait，整组视为"未完全在批"，D4 水位统计如实计入其占用）。

## 6. DAPO 零方差组 abort（D3）

### 6.1 机制

DAPO 动态采样：优势恒为零的组（组内奖励方差 = 0）对梯度无贡献，其生成算力是纯浪费。
引擎侧支持 = 在组生命期内检测零方差并**中止在途成员**，把废 token 率从"事后才知道"变成
"事中可截断"。

### 6.2 奖励从哪来

调度器**不计算奖励**。训练侧（v1 = verl 适配层，沿用 A 的 hooks 模式）在段 finished 时回填：

```python
scheduler.report_reward(group_id: str, seg_id: str, reward: float) -> None
```

- 奖励可得性是数据流属性：规则奖励（数学验证）在完成即可得，reward model 有额外延迟；
  `GroupView.finished_rewards` 为空 tuple 时，D3 上的零方差检测自动退化为不触发（诚实降级，不猜）。
- 回填顺序即 `finished_rewards` 序；重复回填同一段是错误。

### 6.3 检测点（v1 两个，均可配）

| 检测 | 触发条件 | 动作 | 误伤面 |
|---|---|---|---|
| **exact**（默认开） | 组内全部 G 段 finished 且奖励数 ≥ 2 且 `variance == 0`（G=1 组方差无定义，不判拒收，由训练侧过滤） | 组内已无在途段，无 abort 收益；**标记组数据拒收**（§6.5），对账用 | 无 |
| **early**（默认关，A/B 开） | 奖励可得数 ≥ `max(2, ceil(ρ·G))` 且已得奖励全等（奖励可得可能滞后于完成数——reward model 延迟——按可得数保守计） | `abort(group, reason="zero_variance_early")`，在途成员立即中止 | 在途成员本可能产出不同奖励 → 白白损失有效组；ρ 越小误伤越大 |

early 的默认关闭是刻意保守：v1 先用 exact 验证对账链路（abort 语义、trace、废 token 统计），
A/B 实验里再开 early 测真实收益/误伤曲线。ρ 进配置，扫描留给仿真器（S2 可零成本扫 ρ）。

### 6.4 abort reason 注册表（v1）

| reason | 含义 | 产生点 |
|---|---|---|
| `zero_variance_early` | 组内已完成成员奖励全等，early 检测命中 | D3 |
| `weight_skip` | D2 后策略判定该组不值得在新版本上重算（如同步瞬间组已大面积完成且判定边际收益低于 re-prefill 成本） | D2 |
| `policy_preempt` | 显存压力下的策略性中止（v2 预留，v1 不产生） | D4（预留） |

`zero_variance`（spec §4.4 示例里的拼写）保留为 exact 检测的**数据拒收**标记，不出现在
`segment_end.reason` 里（exact 检测无在途段可中止）；组级拒收的落盘方式见 §6.5。

### 6.5 废 token 对账

三层口径，全部可从 trace 直接推得：

1. **段级**：`segment_end{state:"aborted", reason, n_gen_tokens}` → 段级废 token（metrics 铁律 3）；
2. **组级拒收**（exact 检测 + 组内全部 finished 但数据被弃）：这是 C 在 `docs/issues.md` 已指出的
   spec 缺口（finished 段的 `trainer_committed` 不可区分）——**不实现绕路，走 spec §8 增量**：
   已在 issues.md 记 S1→S2 条目（见 §11）。增量落地前，适配层以**自定义事件**
   `scheduler_group_reject{ts, group_id, reason:"zero_variance", meta:{n_gen_tokens_group}}` 过渡
   （spec §8 允许 schema_version 0 内新增事件类型：旧 reader 跳过并报 W06，validator strict 照常通过；
   集成测试已验证该兼容路径）。`trainer_committed` 增量合入后废弃；
3. **聚合**：`bench/analysis` 现有 waste 口径不变；A/B 报告给
   `waste = aborted_tokens / generated_tokens` 及含/不含 abort 双列（口径已在 metrics-v0 冻结，不新增）。

## 7. 轨迹状态机扩展：token 边界 pause/resume 原语（签名冻结，本体 M3）

### 7.1 定位

`paused` 状态与 `running→paused→running` 转换在 rheotrace-spec §2 已冻结；本节冻结的是
**触发它的引擎原语签名**。v1 提供 stub：状态机转换、trace 事件、账本语义全部真实生效，
仅"暂停发生在逐 token 边界"这一执行细节由 M3 的 WeightManager 兑现（v1 的安全点 = 引擎批边界，
语义上仍是"在途轨迹停在 token 边界"，粒度粗于逐 token）。

### 7.2 签名（冻结）

```python
class PauseReceipt:
    seg_id: str
    paused_at_version: int  # 暂停时生效版本
    n_prompt_tokens: int  # 暂停段的 prompt 长度（resume 处推 prefill_len 用）
    n_gen_tokens: int  # 已产出 token 数（KV 内）


class TokenBoundaryPauser(Protocol):
    def pause(
        self,
        segs: Sequence[SegmentHandle],  # 在途段句柄（引擎侧对象）
        reason: str = "token_boundary",
    ) -> list[PauseReceipt]: ...
    def resume(
        self,
        receipt: PauseReceipt,
        mode: Literal["re-prefill", "shadow", "stale"],
        version: int
        | None = None,  # None = 暂停时版本（v1 stub 无账本视图；M3 本体默认查当前生效版本）
    ) -> ResumePlan: ...


@dataclass(frozen=True)
class ResumePlan:
    seg_id: str
    mode: Literal["re-prefill", "shadow", "stale"]
    target_version: int
    prefill_len: int  # re-prefill 需重算的长度 = n_prompt_tokens + n_gen_tokens
    kv_action: Literal["drop", "keep", "demote"]  # v1 恒 "drop"；shadow/stale 的 KV 留存 = M3/M4
```

- `resume.mode` 的三分支即 §8 成本模型的输出面；v1 只会产生 `re-prefill`（`kv_action="drop"`）。
- `prefill_len` 是 re-prefill 成本模型（§8.2）的判定输入，在此一并冻结——S2 仿真器按它计重算开销。
- 与 S4 的对齐：`env_wait` 与 `paused` 是**不同状态、不同原语**。env_wait 由 env-gateway 触发
  （工具调用拦截），其 KV 分级留存走 S4 的成本模型；`paused` 由本原语触发（版本边界/抢占）。
  两者可叠加的合法路径是 `env_wait → running → paused`（状态机表已禁止 `env_wait → paused` 直转），
  S4 设计文档同日互审确认无矛盾。

### 7.3 trace 事件映射（v1 stub 也必须过 validator）

| 原语调用 | 落盘事件 |
|---|---|
| `pause(segs)` | 每段 `segment_state{from:"running", to:"paused", reason}` |
| `resume(receipt, "re-prefill")` | `segment_state{paused→running}` + `phase_span{prefill, meta:{reason:"re-prefill"}, n_tokens=prefill_len}` |
| `resume(receipt, "shadow"/"stale")` | `segment_state{paused→running}` + 段内新 logprob 块 `version=target_version` + `segment_end.finish_mode` 相应标注（M3/M4 兑现） |
| `abort`（D2/D3 任一路径） | 每段 `segment_end{state:"aborted", reason, n_gen_tokens, end_version}` |
| 组级拒收（exact，§6.5 口径 2） | 自定义事件 `scheduler_group_reject{ts, group_id, reason:"zero_variance", meta}`（W06 前向兼容，`trainer_committed` 增量后废弃） |

## 8. 成本模型：四分支判定输入表（输入冻结；v1 只激活两支）

成本模型回答一个问题：**一个 paused 段（或待决策段）应当以哪种方式继续/终止**。
v0 冻结的是每支的**判定输入**（表左列）；实现与数值实验 v1 只做前两支。

### 8.1 输入总表

| 分支 | 判定输入（冻结） | 前置条件 | 激活 |
|---|---|---|---|
| **continue** | ① `current_version == birth_version`（未跨版本）；② KV 有效；③ 显存水位 < 软阈值 | 无版本边界 | ✅ v1 |
| **re-prefill** | ① 跨版本（`current_version > birth_version`）且 KV 判失效；② 重算成本 `prefill_len = n_prompt + n_gen`；③ 段剩余 token 估计 `max_new_tokens − n_gen`（可得时）；④ `重算成本 < 剩余价值估计`（v1 简化：恒真，除非 ③ ≤ 0）；⑤ 组未被 §6 拒收 | 已 `paused` 于 token 边界 | ✅ v1 |
| **shadow** | ① 长尾判定：所在批占用率 `occ < θ_shadow`（v0 建议 0.5，对齐 metrics S1 阈值）且引擎进入收尾窗口；② 段剩余 token 估计；③ CPU pinned 精确副本对该 birth_version 可得；④ PCIe 上送带宽估计 | M3（副本本体） | 留位 |
| **ε-stale** | ① drift 探针 KL ≤ 预算 ε；② 版本距离 `current − birth ≤ k_stale`；③ HBM 低精度副本可用；④ 该段此前未 stale 过（一次降级） | M4（探针本体） | 留位 |

### 8.2 v1 判定序（伪码，冻结语义）

```python
def decide_resume(seg, obs) -> Decision:  # D2 安全点之后逐段调用（仅 paused 段；env_wait 见 §8.3）
    if group_rejected(seg.group_id):  # §6（防御路径）
        return abort_group(seg.group_id, reason=...)
    if seg.current_version == obs.current_version:  # 未跨版本
        return continue_()  # 原地续跑，KV 未失效
    if remaining_tokens_estimate(seg) <= 0:
        # 已到 max_new_tokens：不进入本判定——max_len 完成由引擎 finish 收尾（非 abort）。
        # Decision 的 abort 只有组级，段级提前终止不经过调度器（§6.4 weight_skip 是组级判定）
        return skip_engine_finishes()
    return re_prefill([seg.seg_id])  # v1 唯一续跑路径
```

注：`weight_skip`（§6.4）是**组级**判定——"整组不值得在新版本上重算"，由带成本意识的策略
产生（v1 默认策略不产生）；段级"到长"是引擎的 finish 通道，不走 abort。
实现注记：到长段进入本判定序属于调用方契约违反，`decide_resume` 以 `SchedulerError` 拒绝
（而非伪码中的良性跳过——伪码表达语义，代码强制前置）。

真值简化注记：④"重算成本 < 剩余价值"在 v1 恒真的理由——0.5B 档 re-prefill 单段开销
（毫秒级 prefill）远小于该段剩余 decode 价值；带阈值的判定留给仿真器先扫（S2 有 `prefill_len`
与剩余估计两个输入，可在合成 trace 上找拐点，产"待真机复核"结论）。

### 8.3 与 S4 的成本模型对接

S4 的 KV 分级（HBM → host pinned → 4bit → 磁盘）发生在 `env_wait`，分支输出是"留哪级"；
本表分支输出是"怎么续跑"。两模型共享两类输入（等待时长估计、KV 段大小），各自独立决策、
在 `resume` 处汇合：env_wait 段被工具唤醒后回到 running，若其间发生 weight_sync，则进入本表
D2 判定（此时其 KV 可能已被 S4 降级到 host——`kv_action` 由 S4 状态提供，v1 无 KVT，恒全失效）。

## 9. 显存预算表（6GB / 0.5B 默认档）

依据 PLAN §9：RTX 3060 Laptop 6GB，Windows 桌面常驻占 1–2.5GB，训练实际可用 **3.5–4.5GB**。
默认实验规模 Qwen2.5-**0.5B**（1.5B 为可选档，结论相对值成立、绝对吞吐标注规模）。

| 预算项 | 估算 | 依据/备注 |
|---|---|---|
| rollout 引擎权重（fp16） | ≈ 1.0 GB | 0.5B × 2 B/param |
| LoRA 权重 + 8bit 优化器状态（训练侧） | ≈ 0.1–0.2 GB | M0 同款配置，r ≥ 8–16 量级 |
| 框架/激活/Workspace（torch + vLLM 运行时） | ≈ 0.5–1.0 GB | 经验区间，A/B 实测回填 |
| **KV cache 预算（调度器的 `kv_total_bytes`）** | **≈ 1.5–2.5 GB** | 引擎 `gpu_memory_utilization` 扣除上述项后的池；D1 容量约束按此推 k |
| 合计（引擎侧） | ≈ 3.1–4.7 GB | 须落在可用 3.5–4.5 GB 内；超配即 OOM，`gpu_memory_utilization` 从 0.55 起调 |
| Windows 桌面常驻（不可控） | 1–2.5 GB | PLAN §9 实测 |

调度侧含义：

1. **D1 的 k 上限是硬约束**：KV 池 1.5–2.5GB 在 0.5B/4k KV 头配置下大约容纳 10⁴–10⁵ token 级别的
   在途 KV（精确值随模型 config 与序列长度分布浮动，A/B 实测回填本表）——整组共批的 k 通常在
   个位数组量级，够用但不宽裕，这正是"禁止拆组"规则必须配"减 k 不拆组"逃逸口的原因；
2. **D2 安全点的隐性收益**：weight_sync 期间在途段全部 paused，KV 池可整段让渡给同步缓冲
   （v1 无双缓冲，靠暂停腾挪），预算表按"生成态"计，同步态另有 0.5B 权重一份的瞬态开销
   （≈1.0 GB，state_dict 上传），A/B 时验证不 OOM；
3. shadow/ε-stale 的显存代价（CPU pinned 副本 / HBM 低精度副本）按 0.5B 计入 M3/M4 预算，
   本表不展开（PLAN §9 已有结论）。

## 10. 实施映射（v1）

| 设计物 | 代码落点 | 验收 |
|---|---|---|
| §4 接口 + §8.2 判定序 | `runtime/scheduler.py`（纯逻辑，零重依赖） | mock 单测：四分支判定输入全覆盖（shadow/ε-stale 测到"留位不激活"的拒绝路径） |
| §5 共批 | 同上，`Scheduler.plan_batch`（返回 `BatchPlan`） | 单测：整组不拆、减 k 截断（FIFO 队头公平）、batch_id 一致性 |
| §6 DAPO | 同上，`report_reward`（exact 拒收）+ `early_abort_candidate`（early，纯函数） | 单测：exact/early、ρ 边界、重复回填报错；真机 A/B 验证对账链路 |
| §7 原语 stub | 同上，`TokenBoundaryPauser` stub | 产出的每条 trace 过 `rheotrace.validate` |
| verl 接入 | `bench/m0/rheo_trace_hooks.py` 同款 hooks 模式（不改 verl 本体）；等 `feat/m0-baseline` 合入 main 后 rebase | A/B：scheduler on/off，0.5B，with-lock 包锁，trace 落 `bench/traces/`、报告落 `bench/results/scheduler-v1/` |

## 11. 开放问题与跨板块记录

已记 `docs/issues.md`（本日新增）：

- **S1→S2**：spec §8 增量候选——`segment_end.trainer_committed: bool`（可选字段，向后兼容），
  用于组级拒收（§6.5 口径 2）的精确对账；过渡期用 `run_start.meta.scheduler` 携带拒收清单。
- **S1↔S4**：`env_wait → paused` 直转仍非法（§7.2）；若 S4 的分级留存需要"等待中冻结到 token 边界"
  语义，走增量讨论，不动已冻结状态机。

未决（不阻塞本冻结）：re-prefill 判定 ④ 的真实阈值（S2 仿真扫描后回填 §8.2 注记）；
`gpu_memory_utilization` 与 KV 池实测值（A/B 回填 §9）。
