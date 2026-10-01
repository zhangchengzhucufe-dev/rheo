# Sim Study v0：调度策略离线仿真研究

> 会话 S2（TASK-S2 #1/#3）· 2026-10-01 起稿
> 状态：**回放核心 + 校准完成**；策略 harness 等 S1 的 D1（`policy(observation) → decision`
> 接口冻结）合入后填充 §4–§6。
> 复现：`sim.calibrate(trace_path)` → `Calibration.report()`；本文所有数字由该函数产出。

---

## 1. 问题与结论速览

**问题**：调度策略（组感知、错峰批）真机实验昂贵且不可复现，先用离散事件仿真器在
RheoTrace 真实/合成轨迹上筛选，真机只做点验证（PLAN 迭代 6）。

**当前结论**（v0，回放核心阶段）：

1. 单参数吞吐模型（每 GPU 聚合解码率恒定 + step 屏障 + sync 实测值）在**干净窗口**上
   复现生成期墙钟误差 **+3.7%**——足以支撑策略间的**相对**比较。
2. 全 run 口径误差 **-34%**，差额主体是混入训练窗口的验证/eval 负载（未建模，也不受
   调度策略控制）。**策略对比必须用干净窗口或无 eval 的合成 trace。**
3. 真实系统吞吐在 run 内有系统性下滑（窗口 1–9 平均 ~26 tok/s/GPU → 窗口 12–16
   平均 ~14），恒率模型吸收的是整段均值。归因假设见 §3.4。

---

## 2. 仿真器（sim/）

### 2.1 模型（v0，自由参数最少）

| 组件 | 模型 | 参数 |
|---|---|---|
| 调度原子 | **版本窗口**（spec §5.1）：版本 i 的生成期 = sync[i].t_end → sync[i+1].t_start，窗口末落 sync[i+1] 停顿 | — |
| worker 分配 | 池模式（全局 FIFO，工作守恒，默认）\| 分区模式（`assign_fn`，策略接缝） | — |
| 段服务 | decode/prefill 占用 worker：`tokens / 每 GPU 聚合吞吐`；env_wait **释放** worker（唤醒后回队首） | `decode_tok_per_s`、`prefill_tok_per_s` |
| 段内结构 | 从 trace 织出有序任务列：decode span（缺 n_tokens 按时长占比分摊）+ env_wait 区间 | — |
| 同步停顿 | 每窗口实测时长（或全局中位数） | `weight_sync_s` |
| 步首间隙 | sync 结束 → 首段到达（实测 ~0.1s） | `schedule_gap_s` |
| 确定性 | 纯 heapq 事件循环，无随机数；同输入同结果 | `seed`（供策略用） |

设计立场：**每 GPU 聚合吞吐是唯一核心自由参数**。连续批处理下单段速率（~6 tok/s）
远低于聚合速率（~26 tok/s/GPU），逐段建模批调度会引入一堆不可标定参数（PLAN 迭代 6
的教训）——恒率池模型在窗口粒度上等效（§3.2 实证）。

### 2.2 已知简化（诚实清单）

- **eval/验证负载不建模**：混入训练窗口的验证生成未插桩（m0 trace 无对应段），
  该部分时间不进仿真。全 run 口径的 -34% 差额即源于此（§3.3）。
- **prefill 折入 decode 口径**：m0 trace 无 prefill span（A 插桩未发），prompt 预填充
  时间被吞吐率吸收。prompt 均 111 tok，占比小；agent 负载启用 `prefill_tok_per_s` 可分离。
- **paused 不产生任务**：token 边界暂停续跑的时间由 step 屏障 + sync 窗口承担。
- **段不可跨 worker 切分**；单 worker 内串行执行任务（env_wait 除外）。
- 窗口内到达时刻不建模（段在步首全部就绪）：实测 gap ~0.1s，影响可忽略。

### 2.3 接口

```python
import sim

wl = sim.load_workload("bench/traces/synthetic/synthetic-bimodal-longtail.jsonl")
cfg = sim.ClusterConfig(n_gpu=8, decode_tok_per_s=22.5, weight_sync_s=11.6)
res = sim.run_step_barrier(wl, cfg)  # 池模式（默认）
res = sim.run_step_barrier(wl, cfg, assign_fn=...)  # 分区模式（策略接缝）
res.wall_s / res.stall_breakdown() / res.step_table()

cal = sim.calibrate("真实trace.jsonl")  # 拟合 + 双口径回放 + 误差报告
print(cal.report())
```

策略 harness（S1 D1 冻结后）：`assign_fn` 换成策略对象；观察面 = 窗口内段的
(group_id, token 负载, 长度分布)，决策面 = 段 → worker 的分配与开跑时机（错峰批）。

---

## 3. 校准：m0 pilot 真实 trace（8×GPU，verl-0.8.0+vLLM-0.12，Qwen2.5-1.5B）

数据：`m0-baseline-pilot.rheotrace.jsonl`（崩溃续跑段，覆盖 step 30–40，
meta.resume_from_step=29 语义；17 个 sync 窗口、2248 段、386,509 gen tokens、墙钟 3376s）。

### 3.1 校准结果（`calibrate()` 产出）

```
fitted decode throughput: 22.51 tok/s/GPU（10/16 干净窗口）
replay fidelity   （逐窗口实测率重放，生成期墙钟）：sim 3308s vs real 3193s（+3.6%）
policy-grade      （单一拟合率，干净窗口，生成期）：sim 1270s  vs real 1225s（+3.7%）
policy-grade      （单一拟合率，全 run 含 sync）：   sim 2217s  vs real 3376s（-34.3%）
```

三个口径的分工：

- **replay fidelity +3.6%**：给每个窗口用它自己的实测率，剩下的误差纯是引擎机制
  （段不可切分的尾部量化 + 池排空效应）。这是引擎机制误差的上界——**+3.6%**。
- **policy-grade +3.7%（干净窗口）**：单参数模型在干净窗口上与机制上界持平——
  单率假设本身在干净窗口上没有额外代价。**策略对比在这个精度下做相对比较是安全的。**
- **全 run -34.3%**：被 flag 的 6 个 eval 污染窗口里，真实系统在训练生成之外还跑了
  验证负载（未插桩，仿真器看不到），真实窗口因此远长于仿真。这不是模型缺陷，
  是口径问题——见 §3.3 的处理规则。

### 3.2 逐窗口对照（fidelity 口径，秒）

| 窗口 | 段 | gen tok | sim | real | 备注 |
|---|---|---|---|---|---|
| 1–4, 6–9 | 128/窗 | 19–24k | 91–126 | 87–121 | 干净，sim 高 ~4%（尾部量化） |
| 5 | 128 | 20.8k | 676 | 658 | **eval 污染**（step 30 验证） |
| 10–11 | 128 | 20–23k | 448 / 269 | 429 / 261 | **eval 污染**（step 35 双 sync） |
| 12–13 | 128 | 20–21k | 219 / 192 | 208 / 184 | 吞吐下滑档（14 tok/s/GPU） |
| 14–16 | 128 | 23–25k | 185–233 | 179–227 | 同上 |
| 0 | 0 | 0 | 0.1 | 0 | 初始 load sync |
| 17 | 200 | 35.8k | — | 截断 | trace 止于生成中段，不参与对账 |

### 3.3 eval 污染窗口的处理规则

窗口速率呈双模：干净窗口 ~22–29 tok/s/GPU，污染窗口 4–14。识别规则（确定性、非循环）：
**窗口速率 ≥ 0.75 × 中位速率 → 干净**。拟合与误差对账只用干净窗口 + 非截断窗口。
被 flag 的窗口（5, 10, 11, 12, 13, 16）中 5/10/11 有明确外部证据（TASK-A2：test_freq=10
的验证点 + step 35 的双 sync 异常），12/13/16 是吞吐下滑档（§3.4）。

**给后续实验的规则**：完整 40 步重跑 trace 到手后，eval 步（每 10 步一个）的窗口
继续按此规则剔除；策略对比报告必须注明剔除清单。

### 3.4 误差归因（全量清单）

| 来源 | 方向 | 量级 | 证据 |
|---|---|---|---|
| 段不可切分 + 池排空尾部量化 | sim 偏高 | +3~4% | fidelity 口径逐窗口一致偏高 |
| eval/验证负载未建模 | sim 偏低 | ~-1100s | 污染窗口 real/sim 差额合计 |
| 吞吐 run 内下滑（26→14 tok/s/GPU） | 双向 | 窗口级 ±40% | §3.2 表；单率模型取均值 |
| prefill 折入吞吐率 | sim 偏高（若 prefill 实际更快） | 未分离 | 无 prefill span；prompt 短，占比小 |
| sync 窗口取中位数（11.6s；实测 6.7–17.9s） | 双向 | 窗口级 ±6s | 全 run 合计 ±<1% |

吞吐下滑假设（待完整 40 步 trace + GPU 计数器复核）：① 响应长度方差随训练增大，
批内长尾挤占（vLLM preemption/重算）；② KV 内存压力随上下文增长；③ 热降频。
**归因不改变策略结论的相对有效性**——所有策略面对同样的下滑趋势。

---

## 4. 策略对比实验（待 S1 D1 冻结后执行）

- 策略：FIFO 基线 / 组感知（同组同窗聚合）/ 错峰批（staggered batch）
- trace 集：真实 m0（干净窗口）+ 合成双峰长尾 + 合成 agent env-wait
- 指标：吞吐、停顿分解（`stall_breakdown()`）、长尾掉队份额（组内最慢段 vs 组中位）
- 产出：对比表 + ≥1 个"仿真先证明、待真机复核"结论

## 5. 云规模外推（提前完工的加餐）

8×A800 80GB 配置外推：把 `ClusterConfig` 换成云参数（吞吐率按 A800/3060 实测比缩放），
仿真阶段②末爆发实验的负载混合，给租卡决策（¥60–100/时）提供提前量。

## 6. 变更记录

- 2026-10-01：回放核心 + 校准（本文 §1–§3）。仿真器代码 `sim/`，测试 `tests/test_sim.py`。
