# RheoTrace v0 格式规格（冻结稿）

> 版本 v0.1 · 2026-09-29 · 会话 B 起草，供 A（verl 插桩）与 C（分析流水线）引用
> 状态：**M1 核心接缝冻结**。字段与语义一经 merge 不再改动；只允许向后兼容的增量（见 §8）。
> 修订：v0.1 校准 validator 规则表与实现的一致性（E01 措辞、新增 E18/W09 落实 §4.0 的 ts 约定、initial_version 必填措辞消歧），无字段/语义变更。
> 关联：PLAN.md §1 L2 遥测 / §4 里程碑 M1；指标口径见 C 的 `docs/metrics-v0.md`。

---

## 1. 目的与范围

RheoTrace 是 Rheo 的**轨迹追踪与遥测格式**：把一次 rollout 过程中"时间花在哪、权重是哪个版本、
每个 token 的 logprob 是多少"以事件流形式忠实落盘。

服务三个消费方：

| 消费方 | 用途 | 依赖的字段 |
|---|---|---|
| A：verl 插桩 | 在 rollout 路径产生 trace | 全部事件类型 |
| C：分析流水线 | MFU 分解、停顿瀑布、长尾统计、tokens/s/GPU | `phase_span`、`weight_sync`、`segment_*`、`run_*` |
| M4/M5（未来） | staleness-KL、投机接受率 | `birth_version`/`end_version`、`token_logprob.version` |

**不在范围内**（写出边界）：gRPC 传输协议（M6）、KV block 级事件（M4 再加）、
投机解码 draft 事件（M5 再加，见 §8 兼容性）、调度器逻辑本身。
本 spec 同时冻结 `rheotrace` Python 包的 API（§9），A 插桩直接 import。

核心设计立场（对应 PLAN 四大机制）：

- **轨迹是跨版本的生命体**：一条轨迹段带 `birth_version` 出生戳，结束时带 `end_version`，
  两者之差就是 staleness 的原始度量——格式天然支持 partial rollout 与异步采集。
- **引擎永远诚实标注 logprob**：token 级 logprob 必须盖"产生它的权重版本"戳，
  exact / shadow / ε-stale 三种完成方式用 `finish_mode` 明示，不冒充（迭代 2、3）。

---

## 2. 模型与术语

```
run（一次 rollout 会话，= 一个 trace 文件）
├── version ledger：全局单调版本号，weight_sync 事件推进
├── group（rollout 组：一个 prompt 采 G 条）——仅是 seg 上的标签，不是独立事件
└── segment（轨迹段，状态机：running / paused / env_wait / finished / aborted）
    ├── phase_span：prefill / decode / env_wait / schedule 计时区间
    ├── token_logprob：token 级 logprob 分块记录
    └── segment_end：终态（finished | aborted）+ 完成方式
```

- **run**：一次 rollout 会话，对应**一个 trace 文件**（一文件一 run，v0 不做分卷）。
- **segment（轨迹段）**：一条轨迹的一个连续生成段。一条轨迹跨权重版本被暂停续跑时，
  v0 仍记为**同一个 segment**（段内 token 可能来自不同版本，靠 logprob 的 version 戳区分）。
  segment 归属 `group_id`（一个 prompt 的 G 条采样共享组号）。
- **version（权重版本）**：run 内全局单调递增整数。权重双缓冲在安全点翻转一次 = 一个新版本。
- **event（事件）**：JSONL 的一行，携带时间戳与类型特定字段。事件按发生顺序追加。

### 轨迹段状态机

```
             ┌──────────► paused ──────────┐
             │   (token 边界安全点)         │
(start) ──► running ──► env_wait ──────────┤──► running ...
             │    (多轮工具调用)           │
             └──────────► finished (segment_end)
                          aborted  (segment_end)
```

| from \ to | running | paused | env_wait | finished | aborted |
|---|---|---|---|---|---|
| *(start)* | ✓ | | | | |
| running | —(自身) | ✓ | ✓ | ✓ | ✓ |
| paused | ✓ | | | | ✓ |
| env_wait | ✓ | | | | ✓ |
| finished / aborted | 终态，任何后续事件都是错误 | | | | |

- `paused`：在 **token 边界安全点**暂停（换权重、抢占）。这是 Rheo 的新原语（PLAN M3）。
- `env_wait`：多轮 agent 的工具调用等待（PLAN 迭代 5）。
- 进入终态**只能**通过 `segment_end` 事件（`segment_state` 不发终态转换，见 §4）。

---

## 3. 存储格式选型：JSONL 事件流（v0 结论）

**结论：v0 采用 JSONL 事件流**（UTF-8，每行一个 JSON 对象，换行分隔）。
Arrow 列存推迟到 v1，理由与迁移路径见下表和 §8。

| 维度 | JSONL 事件流（选定） | Arrow 列存（v1 候选） |
|---|---|---|
| 写入模式 | 追加式，天然匹配 live rollout 流式产出；崩溃时最多丢最后一行（可检测） | record batch 需缓冲/flush 策略，流式追加别扭 |
| 依赖 | **零依赖**（stdlib `json`），verl 混乱环境 / CI / 任意语言都能写读 | 硬依赖 pyarrow（原生库 ~50MB wheel），环境冲突风险 |
| 可调试性 | `head`/`grep`/`jq` 直接看，插桩 bug 第一周就能人肉排查 | 二进制，必须工具才能看 |
| 体积 | logprob 冗长（double 文本 ~10–20 B/token），gzip 后 −70% 左右 | float32 原生 4 B/token，最紧凑 |
| 分析友好 | C 端一次 `read` 进 pandas，M1 规模（单 run ≤ GB 级）无压力 | 列式零拷贝，大规模更优 |
| 演进成本 | 字段级演进零成本，适合格式年轻期 | schema 演进有摩擦，适合格式冻结后 |

**迁移路径**：所有字段名设计为扁平、列友好（§4 字段表即未来的 Arrow 列名），
每个事件类型对应 v1 的一张 Arrow 表（键：run_id / seg_id / ts）。
v1 提供 `rheotrace.arrow` 转换器；事件 schema 不变，只是物理布局升级。
C 端分析代码只依赖 `rheotrace.read`，对布局无感。

**附加约定**：

- 一行 = 一个事件 = 一个 JSON object；禁止嵌套事件数组。
- 浮点数 = JSON number（双精度）；**writer 必须以 `allow_nan=False` 序列化**（NaN/Inf 非法）。
- writer 应当用紧凑分隔符与 `ensure_ascii=False`（体积与可读性）。
- 文件名约定：`*.rheotrace.jsonl`，gzip 压缩时 `*.rheotrace.jsonl.gz`（reader/writer 按后缀透明处理）。
- 每行必须以 `\n` 结尾；reader 遇到末尾无换行的残行视为截断（见 §6 E20）。
- 时间戳一律 **纳秒整数**。`clock` 默认 `wall_ns_epoch`（`time.time_ns()`，跨进程可对齐 trainer 日志）。
  纯本机时长测量可用 `mono_ns_raw`（单调钟，跨进程不可比）。合成 trace 用虚拟钟（锚定生成时刻的 epoch 值），如实标注。

---

## 4. 事件模型与字段表

### 4.0 公共信封（每个事件必有）

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `ts` | int | ✓ | 事件发生时刻（ns）。**区间型事件（span / weight_sync / segment_end）取区间结束时刻**。文件内事件按 `ts` 非降序排列 |
| `type` | string | ✓ | 事件类型，§4.1–4.7 注册表内的 snake_case 标识符 |
| `run_id` | string | ✓ | 由 `run_start` 定义，后续所有事件必须一致 |

其余字段按类型附加。**所有事件都可以带任意额外字段**（reader 保留、validator 忽略），
新增可选字段是兼容性变更（§8）。

### 4.1 `run_start`（文件第一个事件）

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `format` | string | ✓ | 恒为 `"rheotrace-jsonl"` |
| `schema_version` | int | ✓ | 恒为 `0`（v0） |
| `initial_version` | int | ✓ | run 开始时生效的权重版本（全新训练从 0 起；中途挂到已训练进程时填当前版本） |
| `engine` | string | ✓ | 引擎标识，如 `"verl+vllm-0.6.3"`、`"rheotrace-gen"`（合成） |
| `model` | string | ✓ | 模型标识，如 `"Qwen2.5-1.5B-Instruct"` |
| `clock` | string | ✓ | `"wall_ns_epoch"`（默认）\| `"mono_ns_raw"` |
| `n_workers` | int | | rollout worker 数（DP size） |
| `meta` | object | | 自由元数据：git sha、采样参数、GPU 型号、seed… |

### 4.2 `weight_sync`（版本账本 + 权重同步停顿）

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `version` | int | ✓ | **同步后的新版本**，必须严格大于此前任何版本 |
| `t_start` | int | ✓ | 生成停顿开始（所有在途轨迹到达 token 边界） |
| `t_end` | int | ✓ | 新权重生效（指针翻转）。**新版本自 `t_end` 起生效**；`t_end ≥ t_start` |
| `mode` | string | ✓ | `"full"` \| `"delta"` \| `"load"`（load = 从 checkpoint 恢复） |
| `trainer_step` | int | | 对应的 trainer 全局 step（版本↔step 映射，见 §5.2） |
| `meta` | object | | 传输字节数、压缩方式、带宽等 |

> **weight-sync 阶段的计时就在本事件的 `[t_start, t_end]`**，不发 `phase_span`。
> 五个阶段（prefill / decode / weight-sync / env-wait / schedule）的计时表达：
> 前四个走 `phase_span` / 状态机（§4.3、§4.4），weight-sync 走本事件——这是有意为之，
> 因为版本翻转是账本事件，天然携带窗口。

### 4.3 `phase_span`（阶段计时区间）

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `phase` | string | ✓ | `"prefill"` \| `"decode"` \| `"env_wait"` \| `"schedule"` |
| `seg_id` | string | | 归属的轨迹段；**缺省 = engine/worker 级区间** |
| `t_start` / `t_end` | int | ✓ | 区间，`t_end ≥ t_start` |
| `n_tokens` | int | | 本区间处理/产出的 token 数 |
| `worker` | int | | worker（DP rank）id |
| `meta` | object | | 如 `{"reason": "re-prefill"}` |

阶段语义（C 的停顿瀑布以此为原料）：

| phase | 计什么 | 归属 |
|---|---|---|
| `prefill` | prompt 预填充；暂停后续跑重算也发（`meta.reason="re-prefill"`） | 段级 |
| `decode` | 增量解码；一条段可发多个（每次暂停/恢复切开） | 段级 |
| `env_wait` | 工具调用等待；可选精化——等待时长已由状态机 `env_wait` 状态区间记录，span 用于 worker 级归因 | 段级，可选 |
| `schedule` | 调度间隙（批间空转、组内等待）；**engine 级，不带 seg_id** | run 级 |
| *weight-sync* | 不发 span，见 §4.2 | — |

### 4.4 `segment_start` / `segment_state` / `segment_end`（轨迹段生命周期）

`segment_start`：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `seg_id` | string | ✓ | run 内唯一 |
| `group_id` | string | ✓ | rollout 组 id（一个 prompt 的 G 条共享） |
| `birth_version` | int | ✓ | **出生版本**：本段第一个生成 token 时的生效版本，`initial_version ≤ birth_version ≤ 当前版本` |
| `t_start` | int | ✓ | 段开始时刻 |
| `n_prompt_tokens` | int | ✓ | prompt 长度 |
| `meta` | object | | 采样参数、seed、max_new_tokens 等 |

`segment_state`（仅 running/paused/env_wait 之间的转换）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `seg_id` | string | ✓ | |
| `from_state` / `to_state` | string | ✓ | 必须 ∈ {running, paused, env_wait}，且转换合法（§2 状态机表） |
| `reason` | string | | 如 `"token_boundary"`、`"env_call:search"`、`"preempt"` |

`segment_end`（终态，兼作 from→终态 的转换事件）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `seg_id` | string | ✓ | |
| `state` | string | ✓ | `"finished"` \| `"aborted"` |
| `from_state` | string | ✓ | 终态前的状态 |
| `reason` | string | `aborted` 时必填 | 如 `"zero_variance"`（DAPO 动态采样）、`"max_len"`、`"weight_skip"` |
| `t_end` | int | ✓ | 段结束 |
| `n_gen_tokens` | int | ✓ | 本段累计生成 token 数（含 aborted 前已产出的） |
| `birth_version` | int | ✓ | 回显，validator 与 segment_start 一致性校验 |
| `end_version` | int | ✓ | 段结束时生效版本（= 当前账本版本），**staleness = end_version − birth_version** |
| `finish_mode` | string | | `"exact"`（默认）\| `"shadow"`（按出生版本精确跑完，M3）\| `"stale"`（ε-stale 续跑，诚实降级标注） |
| `meta` | object | | |

### 4.5 `token_logprob`（token 级 logprob，分块）

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `seg_id` | string | ✓ | |
| `version` | int | ✓ | **产生这些 token 的策略版本**（TIS 校正与 staleness-KL 的依据）。exact/shadow 完成时 = birth_version；ε-stale 续跑的 token = 续跑时版本 |
| `start_idx` | int | ✓ | 本块第一个 token 在该段生成流中的下标（0 起） |
| `n` | int | ✓ | 块内 token 数，必须 `n == len(lp)` |
| `lp` | [float] | ✓ | 自然对数概率 |
| `tok` | [int] | | token id（可选，遥测通常不需要） |
| `entropy` | [float] | | 逐 token 熵（可选） |

规则：同一段内各块 `start_idx` 必须首尾相接从 0 铺起（下一块 = 上一块 start_idx + n）；
块间不允许重叠；可选数组若出现，长度必须等于 `n`。
对 `finished` 段，全部块应恰好覆盖 `[0, n_gen_tokens)`，有洞 → 警告；超出 → 错误。
writer 可对 `lp` 做固定位数舍入以省体积——这是 **writer 的选择**，格式本身按双精度无损。

### 4.6 `run_end`（文件最后一个事件）

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `summary` | object | | 自由计数，建议 `{"segments": n, "weight_syncs": k, "gen_tokens": t}` |

### 4.7 示例（一段完整生命周期）

```json
{"ts":1727600000000000000,"type":"run_start","run_id":"r-9f3c1a2b","format":"rheotrace-jsonl","schema_version":0,"initial_version":0,"engine":"verl+vllm-0.6.3","model":"Qwen2.5-1.5B-Instruct","clock":"wall_ns_epoch","meta":{"seed":42}}
{"ts":1727600001000000000,"type":"weight_sync","run_id":"r-9f3c1a2b","version":1,"t_start":1727600000900000000,"t_end":1727600001000000000,"mode":"full","trainer_step":1}
{"ts":1727600001010000000,"type":"segment_start","run_id":"r-9f3c1a2b","seg_id":"s-000123","group_id":"g-000031","birth_version":1,"t_start":1727600001010000000,"n_prompt_tokens":112}
{"ts":1727600001100000000,"type":"phase_span","run_id":"r-9f3c1a2b","seg_id":"s-000123","phase":"prefill","t_start":1727600001010000000,"t_end":1727600001100000000,"n_tokens":112}
{"ts":1727600001300000000,"type":"phase_span","run_id":"r-9f3c1a2b","seg_id":"s-000123","phase":"decode","t_start":1727600001100000000,"t_end":1727600001300000000,"n_tokens":64}
{"ts":1727600001300000001,"type":"token_logprob","run_id":"r-9f3c1a2b","seg_id":"s-000123","version":1,"start_idx":0,"n":64,"lp":[-0.42,-1.13,-0.07]}
{"ts":1727600001350000000,"type":"segment_state","run_id":"r-9f3c1a2b","seg_id":"s-000123","from_state":"running","to_state":"env_wait","reason":"env_call:search"}
{"ts":1727600002000000000,"type":"segment_state","run_id":"r-9f3c1a2b","seg_id":"s-000123","from_state":"env_wait","to_state":"running"}
{"ts":1727600002010000000,"type":"weight_sync","run_id":"r-9f3c1a2b","version":2,"t_start":1727600002000000000,"t_end":1727600002010000000,"mode":"full","trainer_step":2}
{"ts":1727600002020000000,"type":"segment_state","run_id":"r-9f3c1a2b","seg_id":"s-000123","from_state":"running","to_state":"paused","reason":"token_boundary"}
{"ts":1727600002030000000,"type":"segment_state","run_id":"r-9f3c1a2b","seg_id":"s-000123","from_state":"paused","to_state":"running"}
{"ts":1727600003000000000,"type":"phase_span","run_id":"r-9f3c1a2b","seg_id":"s-000123","phase":"prefill","t_start":1727600002030000000,"t_end":1727600003000000000,"n_tokens":176,"meta":{"reason":"re-prefill"}}
{"ts":1727600003100000000,"type":"phase_span","run_id":"r-9f3c1a2b","phase":"schedule","t_start":1727600003010000000,"t_end":1727600003100000000}
{"ts":1727600003200000000,"type":"segment_end","run_id":"r-9f3c1a2b","seg_id":"s-000123","state":"finished","from_state":"running","t_end":1727600003200000000,"n_gen_tokens":289,"birth_version":1,"end_version":2,"finish_mode":"exact"}
{"ts":1727600003200000001,"type":"run_end","run_id":"r-9f3c1a2b","summary":{"segments":1,"weight_syncs":2,"gen_tokens":289}}
```

（示例中 `lp` 截断为 3 个值示意。）

---

## 5. 版本语义（冻结）

### 5.1 版本账本

1. 版本空间是 **per-run** 的：`run_start.initial_version` 起步（默认 0），`weight_sync` 推进，
   全 run 严格单调递增，**任何回退都是错误**（validator 拒绝）。
2. **新版本在 `weight_sync.t_end` 生效**。任何"生效版本"的判定 = 按 `ts` 顺序重放
   `weight_sync` 事件（`ts == t_end`）。
3. 版本推进时刻 = 权重指针在安全点（token 边界）翻转的时刻，**不是**同步开始传输的时刻。
4. `birth_version`：段内第一个生成 token 时的生效版本。writer 负责如实填写；
   validator 校验 `initial_version ≤ birth_version ≤ (该时刻生效版本)`。
5. `end_version`：段结束时的生效版本，validator 要求与账本重放结果一致（错 = writer bug）。
6. **staleness = end_version − birth_version**（每段一个数；C 按 batch 聚成直方图喂 staleness 控制器）。
7. 版本**不**因 abort / 重放 / 段暂停而变化；版本只由 `weight_sync` 驱动。

### 5.2 版本号 ↔ trainer step

- 常规训练：**每个更新了策略权重的 trainer step 对应恰好一次 `weight_sync`，version 增 1**。
  此时 `version == initial_version + trainer_step`（initial_version=0 时 `version == trainer_step`）。
- 允许的偏离（都有显式记录，validator 不报错）：
  - trainer 每 k 步才同步一次（省带宽）→ version 单步跳 >1，`trainer_step` 字段标明真实 step；
  - 一次 step 内多次同步（如中间探针 load）→ 允许，各自 `trainer_step` 相同；
  - eval / 回滚 load → `mode="load"`，版本仍须递增（新 run 想从 checkpoint 续，就在新 run 的
    `initial_version` 里表达，不破坏本 run 单调性）。
- 结论：**版本号是"本 run 见过几次新权重"的账本；trainer_step 是外部坐标**。两者不强制相等，
  但 1:1 是 SHOULD（validator 对 `version` 跳变发 info 级提示，帮助发现漏同步）。

---

## 6. 校验规则（validator 实现即本表）

`rheotrace.validate` 按本表逐条检查。**E = error（拒绝文件）**，**W = warning（放行但报告）**。

| # | 级别 | 规则 |
|---|---|---|
| E01 | E | 文件第一个事件必须是 `run_start`；全文件恰好一个 run（`run_end` 缺失按 W01 处理——最常见成因是截断，不按 E 级拒绝） |
| E02 | E | 每个 JSON 行解析失败（末尾截断残行除外，见 W01） |
| E03 | E | 公共信封缺失/类型错：`ts` 非 int、`type` 缺失、`run_id` 与 run_start 不一致 |
| E04 | E | 事件 `ts` 相比前一事件**倒退**（乱序） |
| E05 | E | 必填字段缺失或类型错误（按 §4 各表逐字段） |
| E06 | E | 枚举值非法：`phase`、`state`、`mode`、`finish_mode`、`clock` 超出注册表 |
| E07 | E | `seg_id` 引用了未 `segment_start` 的段；`seg_id` 重复开启 |
| E08 | E | 状态机转换不合法（§2 表）；终态后仍有该段事件 |
| E09 | E | 版本回退：`weight_sync.version` 未严格递增；`version`(logprob) 超前于当时账本；`birth_version` 超出 run 最终账本版本（文件尾统一检查——出生版本取决于首个 token 时刻，流式阶段不可判定） |
| E10 | E | `birth_version` < `initial_version`；`segment_end.end_version` ≠ 当时账本版本 |
| E11 | E | 区间非法：`t_end < t_start`（span / weight_sync / segment_end 一律） |
| E12 | E | 段级 span 落在段生命周期之外（`t_start` 早于段开始或 `t_end` 晚于段结束）；engine 级 span 只允许 `schedule` |
| E13 | E | logprob 块重叠（`start_idx` 回跳）或 `n != len(lp)`；可选数组长度 ≠ `n` |
| E14 | E | logprob 块 `version` < 段的 `birth_version` |
| E15 | E | `aborted` 段的 `segment_end` 缺 `reason` |
| E16 | E | `run_end` 之后仍有事件 |
| E17 | E | `weight_sync` 窗口与上一窗口重叠（`t_start` 早于上一 `t_end`） |
| E18 | E | 区间型事件（weight_sync / phase_span / segment_end）的 `ts` 早于其 `t_end`——时间线矛盾（§4.0：区间事件以结束时刻为 ts） |
| W01 | W | 文件无 `run_end` 结尾（截断）；末尾残行被 lenient 模式跳过 |
| W02 | W | `run_end` 时仍有未终态的段（崩溃/未正常收尾） |
| W03 | W | logprob 覆盖有洞（仅 `finished` 段：块未铺满 `[0, n_gen_tokens)`） |
| W04 | W | 段级 `decode`/`prefill` span 横跨某个 `weight_sync` 窗口（插桩粒度可疑） |
| W05 | W | `env_wait` span 未落在该段 `env_wait` 状态区间内 |
| W06 | W | 未知事件类型（前向兼容：新版本写的旧 reader 跳过并警告） |
| W07 | W | 整个 run 没有任何 segment；或 run 没有任何 `weight_sync` 而有 segment 产出 |
| W08 | W | run 内 `version` 跳变 >1（§5.2，提示可能漏同步） |
| W09 | W | 区间型事件的 `ts` 晚于其 `t_end`（迟写/缓冲未及时 flush；时间核算仍以 `t_end` 为准）。`TraceWriter` 对区间型事件自动取 `t_end` 作 ts，合规默认 |

validator 报告对象：`ValidationReport(ok, errors[], warnings[])`，每条含规则号、行号、事件摘要。
`strict=True` 时存在任一 error 即抛 `ValidationError`（携带完整 report）。

---

## 7. 合成 trace 生成器与样例

`rheotrace.gen`：虚拟时钟离散采样，`random.Random(seed)` 全程确定性，参数化：

| 参数 | 含义 |
|---|---|
| `seed` | 随机种子（同 seed 同字节输出） |
| `n_steps` / `groups_per_step` / `group_size` | trainer 步数、每步组数、组内采样数 G |
| `length_dist` | `"lognorm"`（单峰）或 `"bimodal"`（双峰长尾：`short_frac` 比例走短峰） |
| `short_mean` / `long_mean` / `short_frac` | 双峰两峰的均值与配比（专打 partial rollout 弱点，PLAN §5 负载 3） |
| `weight_sync_ms` / `schedule_gap_ms` | 权重同步停顿 / 调度间隙 |
| `env_wait_prob` / `env_wait_ms` | agent 场景：进入 env 等待的概率与时长 |
| `abort_prob` / `pause_at_sync` | 早停比例；换权重时是否令在途段暂停续跑（partial rollout） |
| `lp_round` | logprob 舍入位数（writer 端省体积） |

预置三档负载，样例落 `bench/traces/synthetic/`（小文件 `git add -f` 进库，供 C 直接用）：

| 文件 | 预置 | 模拟什么 |
|---|---|---|
| `synthetic-grpo-small.jsonl` | `grpo` | 单峰长度、无 env、多次 weight_sync 的朴素 GRPO |
| `synthetic-bimodal-longtail.jsonl` | `bimodal` | 双峰长尾 + 换权重暂停续跑 + 早停（M1 压力负载） |
| `synthetic-agent-envwait.jsonl` | `agent` | 多轮工具调用 env_wait、反复暂停/恢复 |

再生成命令：`python -m rheotrace gen --preset bimodal --seed 7 --out <path>`（样例文件头 `meta.preset` 记录参数，可复现）。

---

## 8. 兼容性与演进

- **schema_version 0 内**：允许**新增可选字段、新增事件类型**（旧 reader 跳过未知类型/字段，W06）；
  禁止删除字段、改字段类型/语义、收窄枚举——那类破坏性变更必须 `schema_version → 1` 并新开 spec。
- reader 对 `schema_version > 自己认识的版本`：警告并尽力解析（不硬拒）。
- 未来增量候选（**现在不定义**，只预告）：KV block 生命周期事件（M4）、
  drift 探针读数事件（M4）、投机解码 draft/验证事件与接受率（M5）、per-worker weight-sync 分解（M3）。
- gRPC 协议（M6）将把本事件模型作为 TrajectoryService.StreamEvents 的载荷 schema，
  字段一一映射，本 spec 是协议的语义底座。

---

## 9. `rheotrace` 包 API（随本 spec 一并冻结）

```python
import rheotrace

# 常量
rheotrace.FORMAT_NAME == "rheotrace-jsonl"
rheotrace.FORMAT_VERSION == 0
rheotrace.SEGMENT_STATES  # {"running","paused","env_wait","finished","aborted"}
rheotrace.PHASES  # {"prefill","decode","env_wait","schedule"}

# 写：低层一次性落盘（测试/生成器用；纯序列化，不做校验）
rheotrace.write(path, events)  # path 以 .gz 结尾自动压缩

# 写：插桩用流式 writer（自动补 ts / run_id / run_start / run_end）
w = rheotrace.TraceWriter(path, engine=..., model=..., meta={...})
w.emit("weight_sync", version=1, t_start=..., t_end=..., mode="full")
#   ts 自动填充：区间型事件（weight_sync/phase_span/segment_end）默认取 t_end（§4.0 合规默认），
#   其余取当前时刻；显式传 ts 则尊重调用方
w.emit_raw({...})  # 已构造好的 dict，补缺失的 ts/run_id
w.close()  # 落 run_end(summary)；支持 with 语法；关闭后再 emit 报错
#   注意：TraceWriter 非线程安全。多 worker（DP rank）各写各的 trace 文件；
#   若必须并发写同一文件，由调用方在外层串行化，且显式传 ts 保证有序

# 读：无损还原事件列表（dict），支持 .gz
events = rheotrace.read(path)
for ev in rheotrace.iread(path):
    ...  # 流式逐行

# 校验
report = rheotrace.validate(path)  # 严格模式：有 error 抛 ValidationError(report)
report = rheotrace.validate(path, strict=False)  # 宽松：返回报告不抛
report.ok / report.errors / report.warnings

# 合成
events = rheotrace.generate(preset="bimodal", seed=7)
rheotrace.generate_file(out_path, preset="agent", seed=3)
```

CLI：`python -m rheotrace validate <file...>`（打印报告，error 时退出码 1）、
`python -m rheotrace gen --preset ... --seed ... --out ...`。

**Round-trip 保证**：`write(events)` → `read(path)` 逐事件相等（dict 相等含浮点精确相等）；
键序不保证、语义无损保证。

---

## 10. 已知限制（v0 明示不做）

1. 单文件单 run，无分卷/轮转（大 run 建议 gzip；分卷留给 v1）。
2. 跨节点时钟不校正：多机部署时各 worker 的 `ts` 有 NTP 偏差，v0 只保证单机内有序。
3. `token_logprob` 存采样时 logprob（出生版本策略），**不含** trainer 侧重算的 new-logprob
   ——那是训练数据管线的事，不是遥测。
4. 无 KV block / draft 头事件（M4/M5 增量加入，§8）。
