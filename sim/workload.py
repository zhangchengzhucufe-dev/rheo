"""trace → 仿真作业：把 RheoTrace 事件流解析成按 trainer step 分组的段作业。

一段（segment）的生成生命周期被建模为有序任务列表：
  ("decode", tokens)   — 占用 worker 的解码工作量
  ("prefill", tokens)  — 占用 worker 的预填充工作量（m0 trace 无 prefill span 时为空）
  ("env_wait", ns)     — 不占用 worker 的外部等待（工具调用；worker 转去服务别的段）
paused（token 边界暂停续跑）不产生任务：其停顿由 step 屏障 + weight_sync 窗口承担
（v0 简化，原因见 docs/sim-study-v0.md 校准一节）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rheotrace import read

Task = tuple[str, float]  # (kind, tokens | ns)


@dataclass
class SegmentJob:
    seg_id: str
    group_id: str
    step: int  # 出生窗口（= birth_version；spec §5.1 版本即调度原子）
    n_prompt_tokens: int
    n_gen_tokens: int
    tasks: list[Task]
    arrival_idx: int  # 段开始事件在 run 内的先后（策略 harness 的 FIFO 依据）
    birth_version: int
    real: dict = field(default_factory=dict)  # 真实 trace 的观测值（校准/误差对照用）


@dataclass
class Step:
    index: int
    jobs: list[SegmentJob]
    sync_s: float | None = None  # 步末 weight_sync 实测时长
    gap_s: float | None = None  # 步首调度间隙实测（sync 结束 → 首段开始）
    real_span_s: float | None = None  # 真实步长（sync[k-1].t_end → sync[k].t_start）
    meta: dict = field(default_factory=dict)  # trainer_step / mode 等对照信息


@dataclass
class Workload:
    steps: list[Step]
    n_gpu: int
    meta: dict = field(default_factory=dict)

    @property
    def n_segments(self) -> int:
        return sum(len(s.jobs) for s in self.steps)

    @property
    def gen_tokens(self) -> int:
        return sum(j.n_gen_tokens for s in self.steps for j in s.jobs)


def _seg_tasks(seg_events: list[dict], n_gen: int) -> list[Task]:
    """按时间序把段内 decode span 与 env_wait 区间织成有序任务列表。"""
    span_items: list[tuple[int, int, float]] = []  # (t_start, t_end, n_tokens)
    for e in seg_events:
        if e["type"] == "phase_span" and e["phase"] == "decode":
            span_items.append((e["t_start"], e["t_end"], float(e.get("n_tokens", 0))))
    waits: list[tuple[int, int]] = []
    open_wait: int | None = None
    for e in sorted(seg_events, key=lambda x: x["ts"]):
        if e["type"] == "segment_state":
            if e["to_state"] == "env_wait":
                open_wait = e["ts"]
            elif e["from_state"] == "env_wait" and open_wait is not None:
                waits.append((open_wait, e["ts"]))
                open_wait = None
    span_items.sort()
    waits.sort()

    tasks: list[Task] = []
    total_span_tokens = sum(t for *_, t in span_items)
    cursor = 0.0
    for t_start, t_end, n_tok in span_items:
        # 该 span 之前的 env_wait 先入队（spec：区间左闭右开，只取先于 span 的部分）
        while waits and waits[0][1] <= t_start:
            w0, w1 = waits.pop(0)
            tasks.append(("env_wait", float(w1 - w0)))
        if n_tok <= 0 and total_span_tokens > 0:
            # span 缺 n_tokens：按 span 时长占比分摊 n_gen（有损回退，注记在 real）
            n_tok = n_gen * (t_end - t_start) / max(sum(b - a for a, b, _ in span_items), 1)
        tasks.append(("decode", n_tok))
        cursor += n_tok
    # 尾部残余 env_wait
    for w0, w1 in waits:
        tasks.append(("env_wait", float(w1 - w0)))
    # 整段没有任何 decode span（插桩缺口）：整段 n_gen 一次 decode
    if not any(k == "decode" for k, _ in tasks) and n_gen > 0:
        return [("decode", float(n_gen))]
    # span 分摊后与 n_gen 有残差时，差额补到最后一个 decode 任务（保证守恒）
    if total_span_tokens > 0 and n_gen > 0:
        drift = n_gen - cursor
        for i in range(len(tasks) - 1, -1, -1):
            if tasks[i][0] == "decode" and tasks[i][1] + drift > 0:
                tasks[i] = ("decode", tasks[i][1] + drift)
                break
    return tasks


def load_workload(path: str | Path) -> Workload:
    """读一份 RheoTrace（真实或合成），解析成按 sync 窗口分组的仿真作业。

    调度原子 = **版本窗口**：版本 i 的生成期 = sync[i].t_end（版本 i 生效，spec §5.1）
    → sync[i+1].t_start，窗口末的同步停顿即 sync[i+1]。step.index = version。
    verl 在一个 trainer step 内可能多次同步（m0 pilot 实测 17 sync / 11 步，且
    trainer_step 标签复用不可信），版本窗口才是调度语义的可靠边界。窗口内无段的
    （如混入验证的 eval 窗）按真实时长透传，保证墙钟可比。
    """
    events = read(path)
    run_start = next(e for e in events if e["type"] == "run_start")
    run_end_ts = next((e["ts"] for e in events if e["type"] == "run_end"), None)
    n_gpu = int(run_start.get("n_workers") or 1)

    syncs = sorted((e for e in events if e["type"] == "weight_sync"), key=lambda e: e["t_end"])

    seg_events: dict[str, list[dict]] = {}
    order: list[str] = []
    for e in events:
        if e["type"] == "segment_start":
            seg_events[e["seg_id"]] = [e]
            order.append(e["seg_id"])
        elif e["type"] in ("phase_span", "segment_state", "token_logprob"):
            sid = e.get("seg_id")
            if sid in seg_events:
                seg_events[sid].append(e)

    ends = {e["seg_id"]: e for e in events if e["type"] == "segment_end"}

    # 段 → 出生窗口（= birth_version）
    by_version: dict[int, list[SegmentJob]] = {}
    for idx, sid in enumerate(order):
        evs = seg_events[sid]
        start = evs[0]
        bv = int(start["birth_version"])
        end = ends.get(sid, {})
        job = SegmentJob(
            seg_id=sid,
            group_id=start["group_id"],
            step=bv,
            n_prompt_tokens=int(start["n_prompt_tokens"]),
            n_gen_tokens=int(end.get("n_gen_tokens", 0)),
            tasks=_seg_tasks(evs, int(end.get("n_gen_tokens", 0))),
            arrival_idx=idx,
            birth_version=bv,
            real={
                "t_start": start["t_start"],
                "t_end": end.get("t_end"),
                "decode_s": sum(
                    e["t_end"] - e["t_start"]
                    for e in evs
                    if e["type"] == "phase_span" and e["phase"] == "decode"
                )
                / 1e9,
            },
        )
        by_version.setdefault(bv, []).append(job)

    # 每个窗口：版本 i 的生成期 = sync[i].t_end（版本 i 生效）→ sync[i+1].t_start，
    # 窗口末的同步停顿 = sync[i+1]。窗口内无段的（如 eval 窗）按真实时长透传。
    steps: list[Step] = []
    if syncs:
        steps.append(
            Step(
                index=0,
                jobs=[],
                sync_s=(syncs[0]["t_end"] - syncs[0]["t_start"]) / 1e9,
                gap_s=0.0,
                real_span_s=max((syncs[0]["t_start"] - run_start["ts"]) / 1e9, 0.0),
                meta={"trainer_step": syncs[0].get("trainer_step"), "mode": syncs[0].get("mode")},
            )
        )
    for i, e in enumerate(syncs, start=1):
        anchor = e["t_end"]  # 版本 i 自此刻生效（spec §5.1）
        jobs = by_version.get(i, [])
        nxt = syncs[i] if i < len(syncs) else None
        if nxt is not None:
            span_end: int | None = nxt["t_start"]
        else:
            # 末窗口：无后续 sync，用 run_end 收口；截断 trace 则无实测跨度
            span_end = run_end_ts
        real_span = max((span_end - anchor) / 1e9, 0.0) if span_end is not None else None
        gap = None
        if jobs:
            first_arr = min(j.real["t_start"] for j in jobs)
            gap = max((first_arr - anchor) / 1e9, 0.0)
        # 末窗口由 run_end 收口：trace 截断于生成中段时，其真实跨度不完整，
        # 不参与拟合与误差对账（meta.truncated 标记）
        truncated = nxt is None
        steps.append(
            Step(
                index=i,
                jobs=jobs,
                sync_s=((nxt["t_end"] - nxt["t_start"]) / 1e9) if nxt else None,
                gap_s=gap,
                real_span_s=real_span,
                meta={
                    "trainer_step": nxt.get("trainer_step") if nxt else None,
                    "mode": nxt.get("mode") if nxt else None,
                    "truncated": truncated,
                },
            )
        )

    return Workload(
        steps=steps,
        n_gpu=n_gpu,
        meta={
            "run_id": run_start["run_id"],
            "engine": run_start.get("engine"),
            "model": run_start.get("model"),
            "n_workers": n_gpu,
            "gen_meta": run_start.get("meta", {}).get("preset"),
        },
    )
