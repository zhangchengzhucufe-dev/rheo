"""离散事件仿真核心（TASK-S2 #1）：step 屏障回放 + env_wait 释放 worker 的段生命周期重演。

时间模型（v0，自由参数最少——PLAN 迭代 6）：
- 每步为一个同步屏障：step 的全部段在步首就绪（步前间隙 = schedule gap），
  全部完成后落 weight_sync 停顿，进入下一步——与 verl GRPO 的步末同步语义一致。
- 段任务序列（sim.workload.SegmentJob.tasks）逐个执行：decode/prefill 占用 worker
  （时长 = tokens / 每 GPU 聚合吞吐），env_wait **释放** worker（工具调用等待期间
  worker 服务其他段——agent 负载的连续批处理本质）。
- 两种 worker 分配：
  * 池模式（默认，assign_fn=None）：全部 worker 从全局 FIFO 拉段，工作守恒——
    与连续批处理引擎的聚合行为一致，也是回放校准的口径；
  * 分区模式（assign_fn 给出 per-worker 队列）：策略 harness 的接缝——S1 冻结
    policy(observation)→decision 后，组感知 / 错峰批策略在这里落地。
- 确定性：纯 heapq 事件循环 + 确定性平局序号，无随机数；同输入同结果
  （cfg.seed 供策略 harness 的随机化策略使用）。
"""

from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from .cluster import ClusterConfig
from .workload import SegmentJob, Workload

# 策略接缝：输入 step 的全部段 + 集群，输出 per-worker 段队列（v0；S1 的 D1 冻结后换正式接口）
AssignFn = Callable[[list[SegmentJob], int, ClusterConfig], list[list[SegmentJob]]]


def assign_fifo_greedy(
    jobs: list[SegmentJob], n_gpu: int, cfg: ClusterConfig
) -> list[list[SegmentJob]]:
    """分区式 FIFO 贪心：到达序遍历，每段给预估工作量最小的 worker（确定性平局按 worker 序）。"""
    loads = [0.0] * n_gpu
    queues: list[list[SegmentJob]] = [[] for _ in range(n_gpu)]
    for job in jobs:
        est = sum(
            cfg.decode_s(t) if k == "decode" else cfg.prefill_s(t) if k == "prefill" else 0.0
            for k, t in job.tasks
        )
        w = min(range(n_gpu), key=lambda i: (loads[i], i))
        queues[w].append(job)
        loads[w] += est
    return queues


@dataclass
class StepRecord:
    step: int
    n_segments: int
    gen_tokens: int
    span_s: float  # 仿真步长（步首 → 末段完成）
    gap_s: float
    sync_s: float
    busy_decode_s: float
    busy_prefill_s: float
    env_wait_s: float  # 不占 worker 的外部等待总量（wall 时钟仍被消耗）
    idle_s: float  # worker 空转（含等 env 唤醒的富余）
    real_span_s: float | None = None  # 真实步长（版本窗口，含真实 gap）
    real_sync_s: float | None = None

    @property
    def total_s(self) -> float:
        return self.gap_s + self.span_s + self.sync_s


@dataclass
class SimResult:
    wall_s: float
    steps: list[StepRecord] = field(default_factory=list)
    real_wall_s: float | None = None
    config: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def wall_error(self) -> float | None:
        """仿真墙钟相对真实墙钟的误差（真实墙钟未知时为 None）。"""
        if self.real_wall_s is None:
            return None
        return (self.wall_s - self.real_wall_s) / self.real_wall_s

    def stall_breakdown(self) -> dict[str, float]:
        """全 run 停顿分解（秒）：decode / prefill / sync / gap / env_wait / idle。"""
        agg = {"decode": 0.0, "prefill": 0.0, "sync": 0.0, "gap": 0.0, "env_wait": 0.0, "idle": 0.0}
        for s in self.steps:
            agg["decode"] += s.busy_decode_s
            agg["prefill"] += s.busy_prefill_s
            agg["sync"] += s.sync_s
            agg["gap"] += s.gap_s
            agg["env_wait"] += s.env_wait_s
            agg["idle"] += s.idle_s
        return agg

    def step_table(self) -> str:
        cols = ("step", "n_seg", "gen_tok", "sim_span", "gap", "sync", "sim_total", "real_span")
        rows = ["\t".join(cols)]
        for s in self.steps:
            rows.append(
                "\t".join(
                    [
                        str(s.step),
                        str(s.n_segments),
                        str(s.gen_tokens),
                        f"{s.span_s:.1f}",
                        f"{s.gap_s:.1f}",
                        f"{s.sync_s:.1f}",
                        f"{s.total_s:.1f}",
                        f"{s.real_span_s:.1f}" if s.real_span_s is not None else "-",
                    ]
                )
            )
        return "\n".join(rows)


@dataclass
class _Sched:
    """事件循环里的段执行状态。"""

    job: SegmentJob
    task_idx: int


def _run_step(
    jobs: list[SegmentJob],
    queues: list[list[SegmentJob]] | None,
    n_gpu: int,
    cfg: ClusterConfig,
) -> tuple[float, float, float, float]:
    """跑一个 step 的全部段：返回 (span_s, busy_decode_s, busy_prefill_s, env_wait_s)。

    池模式（queues=None）：n_gpu 个 worker 从共享 FIFO 取段（工作守恒）。
    分区模式（queues 给定）：worker w 只服务 queues[w]，各自 FIFO。
    env_wait 释放 worker，到点唤醒后回到**队首**（比未开跑的段先服务）。
    """
    if queues is None:
        sources: list[deque[tuple[SegmentJob, int]]] = [deque((j, 0) for j in jobs)]
        src_of = [0] * n_gpu
    else:
        sources = [deque((j, 0) for j in q) for q in queues]
        src_of = list(range(n_gpu))

    suspended: list[tuple[float, int, int, SegmentJob, int]] = []  # (wake_t, seq, src, job, idx)
    hand: list[tuple[SegmentJob, int] | None] = [None] * n_gpu
    busy_until: list[float | None] = [None] * n_gpu
    t = 0.0
    seq = 0
    busy_d = busy_p = wait_total = 0.0
    finished = 0
    n_jobs = len(jobs)

    def start_task(w: int, job: SegmentJob, idx: int) -> None:
        """worker w 开始执行 job 的第 idx 个任务（env_wait 则挂起并保持空闲）。"""
        nonlocal seq, busy_d, busy_p, wait_total
        kind, amount = job.tasks[idx]
        if kind == "env_wait":
            wait_total += amount / 1e9
            heapq.heappush(suspended, (t + amount / 1e9, seq, src_of[w], job, idx + 1))
            seq += 1
            return
        if kind == "prefill":
            dur = cfg.prefill_s(amount)
            busy_p += dur
        else:  # decode
            dur = cfg.decode_s(amount)
            busy_d += dur
        hand[w] = (job, idx)
        busy_until[w] = t + dur

    def finish_or_chain(w: int, job: SegmentJob, idx: int) -> None:
        """hand 里的任务完成：段收尾或推进到下一任务（decode/prefill 链式续跑）。"""
        nonlocal finished, wait_total, seq
        hand[w] = None
        busy_until[w] = None
        nxt = idx + 1
        if nxt >= len(job.tasks):
            finished += 1
            return
        kind, _ = job.tasks[nxt]
        if kind == "env_wait":
            wait_src = job.tasks[nxt][1] / 1e9
            wait_total += wait_src
            heapq.heappush(suspended, (t + wait_src, seq, src_of[w], job, nxt + 1))
            seq += 1
        else:
            start_task(w, job, nxt)

    while finished < n_jobs:
        # 1) 空闲 worker 从自己的源取段
        for w in range(n_gpu):
            if hand[w] is None and sources[src_of[w]]:
                job, idx = sources[src_of[w]].popleft()
                start_task(w, job, idx)
        # 2) 推进时钟：worker 完成 / 挂起唤醒
        cand = [u for u in busy_until if u is not None]
        if suspended:
            cand.append(suspended[0][0])
        if not cand:
            break  # 防御：无在忙、无可唤醒——剩余段必在源里（不可能到达）
        t = min(cand)
        # 3) 到点的挂起段回源队首（比未开跑的段先服务）
        while suspended and suspended[0][0] <= t:
            _, _, s, job, idx = heapq.heappop(suspended)
            if idx >= len(job.tasks):
                finished += 1
                continue
            sources[s].appendleft((job, idx))
        # 4) 到点完成的 worker 推进
        for w in range(n_gpu):
            if busy_until[w] is not None and busy_until[w] == t:
                job, idx = hand[w]  # type: ignore[misc]
                finish_or_chain(w, job, idx)
    span = t
    return span, busy_d, busy_p, wait_total


def run_step_barrier(
    workload: Workload,
    cfg: ClusterConfig,
    *,
    assign_fn: AssignFn | None = None,
    rate_by_step: dict[int, float] | None = None,
) -> SimResult:
    """step 屏障回放：逐段重演 workload，产出墙钟与停顿分解。

    rate_by_step：窗口号 → 每 GPU 解码吞吐。给出时按窗口取率（"实测率重放"——
    分离引擎机制误差与吞吐漂移，校准报告的两个口径之一）；缺省用 cfg 的单一全局率
    （策略级模型：策略对比时的口径）。
    """
    result = SimResult(
        wall_s=0.0,
        config={
            "n_gpu": cfg.n_gpu,
            "decode_tok_per_s": cfg.decode_tok_per_s,
            "prefill_tok_per_s": cfg.prefill_tok_per_s,
            "weight_sync_s": cfg.weight_sync_s,
            "schedule_gap_s": cfg.schedule_gap_s,
            "seed": cfg.seed,
            "mode": "partitioned" if assign_fn else "pool",
        },
        meta={"workload": workload.meta, "assign_fn": assign_fn.__name__ if assign_fn else None},
    )
    t_wall = 0.0
    for step in workload.steps:
        gap = cfg.schedule_gap_s if cfg.schedule_gap_s is not None else (step.gap_s or 0.0)
        sync = cfg.weight_sync_s if cfg.weight_sync_s is not None else (step.sync_s or 0.0)
        eff_cfg = cfg
        if rate_by_step and step.index in rate_by_step:
            eff_cfg = replace(cfg, decode_tok_per_s=rate_by_step[step.index])
        queues = assign_fn(step.jobs, cfg.n_gpu, eff_cfg) if assign_fn else None
        span, busy_d, busy_p, wait_total = _run_step(step.jobs, queues, cfg.n_gpu, eff_cfg)
        if not step.jobs and step.real_span_s is not None:
            # 无段窗口（eval 等未建模内部结构）：真实时长透传，保证墙钟可比
            span = step.real_span_s
        idle = max(cfg.n_gpu * span - busy_d - busy_p, 0.0)
        result.steps.append(
            StepRecord(
                step=step.index,
                n_segments=len(step.jobs),
                gen_tokens=sum(j.n_gen_tokens for j in step.jobs),
                span_s=span,
                gap_s=gap,
                sync_s=sync,
                busy_decode_s=busy_d,
                busy_prefill_s=busy_p,
                env_wait_s=wait_total,
                idle_s=idle,
                real_span_s=step.real_span_s,
                real_sync_s=step.sync_s,
            )
        )
        t_wall += gap + span + sync
    result.wall_s = t_wall
    real_spans = [s.real_span_s for s in workload.steps if s.real_span_s is not None]
    real_syncs = [s.sync_s for s in workload.steps if s.sync_s is not None]
    if real_spans:
        result.real_wall_s = sum(real_spans) + sum(real_syncs or [])
    return result
