"""合成 trace 生成器：虚拟时钟 + 确定性随机，参数化负载（规格 §7）。

负载模型：n_workers 条 lane 并行产出轨迹段；trainer 每"步"结束后做一次 weight_sync，
同步落在该步规划时长的 straddle_frac 分位上——长尾段自然横跨同步窗口，被切出
pause/resume/re-prefill/weight_skip，即 partial rollout 的真实遥测形态。
同 seed 同字节输出（锚点固定，按 seed 错开时间线）。
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .core import FORMAT_NAME, FORMAT_VERSION, MS_NS, RUN_START
from .writer import write

# 固定锚点（规格 §4.7 示例同时刻）：保证同 seed 字节级可复现
ANCHOR_NS = 1_727_600_000_000_000_000

ABORT_REASONS = ("max_len", "zero_variance", "truncate")


@dataclass(frozen=True)
class GenParams:
    seed: int = 0
    n_steps: int = 3
    groups_per_step: int = 12
    group_size: int = 4
    n_workers: int = 4
    length_dist: str = "lognorm"  # "lognorm" | "bimodal"
    short_mean: float = 160.0
    short_sigma: float = 0.35
    long_mean: float = 1024.0
    long_sigma: float = 0.45
    short_frac: float = 0.7
    prefill_ms_per_token: float = 0.05
    decode_ms_per_token: float = 10.0
    weight_sync_ms: float = 800.0
    schedule_gap_ms: tuple[float, float] = (2.0, 30.0)
    env_wait_prob: float = 0.0
    env_wait_ms: tuple[float, float] = (200.0, 3000.0)
    max_env_waits: int = 3
    abort_prob: float = 0.02
    pause_at_sync: bool = True
    resume_prob: float = 0.9
    straddle_frac: float = 0.85  # 同步落在本步规划时长的分位（制造 partial rollout）
    lp_round: int | None = 5
    model: str = "synthetic-1.5B"


PRESETS: dict[str, GenParams] = {
    "grpo": GenParams(),
    "bimodal": replace(
        GenParams(),
        length_dist="bimodal",
        n_steps=4,
        groups_per_step=8,
        short_mean=64.0,
        long_mean=1024.0,
        short_frac=0.7,
        abort_prob=0.05,
    ),
    "agent": replace(
        GenParams(),
        env_wait_prob=0.55,
        n_steps=3,
        groups_per_step=10,
        short_mean=256.0,
    ),
}


def generate(*, preset: str = "grpo", seed: int = 0, **overrides: Any) -> list[dict]:
    """生成一份合成 trace 的事件列表。overrides 为 GenParams 字段。"""
    if preset not in PRESETS:
        raise ValueError(f"未知 preset {preset!r}，可选：{sorted(PRESETS)}")
    params = replace(PRESETS[preset], seed=seed, **overrides)
    return _simulate(params, preset)


def generate_file(
    path: str | Path, *, preset: str = "grpo", seed: int = 0, **overrides: Any
) -> int:
    """生成并落盘（.gz 后缀自动压缩）。返回事件数。"""
    events = generate(preset=preset, seed=seed, **overrides)
    return write(path, events)


def _sample_length(rng: random.Random, p: GenParams) -> int:
    if p.length_dist == "bimodal":
        mean, sigma = (
            (p.short_mean, p.short_sigma)
            if rng.random() < p.short_frac
            else (
                p.long_mean,
                p.long_sigma,
            )
        )
    else:
        mean, sigma = p.short_mean, p.short_sigma
    mu = math.log(max(mean, 1.0))  # 中位数≈mean 的 lognormal，合成负载无需精确矩匹配
    return max(1, int(rng.lognormvariate(mu, sigma)))


def _version_at(syncs: list[dict], ts: int) -> int:
    return max((s["version"] for s in syncs if s["t_end"] <= ts), default=0)


def _jsonify(obj: Any) -> Any:
    """tuple → list 等，保证 meta 落 JSON 再读回后逐值相等（round-trip 无损）。"""
    return json.loads(json.dumps(obj, ensure_ascii=False))


def _plan_segment(rng: random.Random, p: GenParams, t0: int) -> dict:
    """规划一条段的原始时间线（尚未按同步窗口切割）。"""
    n_prompt = max(8, int(rng.gauss(128, 32)))
    aborted = rng.random() < p.abort_prob
    gen_len = _sample_length(rng, p)
    abort_reason = None
    if aborted:
        gen_len = max(1, int(gen_len * rng.uniform(0.15, 0.95)))
        abort_reason = rng.choice(ABORT_REASONS)
    wait_points: list[int] = []
    if p.env_wait_prob > 0 and rng.random() < p.env_wait_prob and gen_len > 2:
        k = rng.randint(1, p.max_env_waits)
        wait_points = sorted(rng.sample(range(1, gen_len), min(k, gen_len - 1)))
    ivs: list[dict] = []
    t = t0
    pf = max(1, int(n_prompt * p.prefill_ms_per_token * MS_NS))
    ivs.append({"kind": "prefill", "t0": t, "t1": t + pf, "n": n_prompt})
    t += pf
    pos = 0
    for wp in wait_points:
        dur = max(1, int((wp - pos) * p.decode_ms_per_token * MS_NS))
        ivs.append({"kind": "decode", "t0": t, "t1": t + dur, "n": wp - pos})
        t += dur
        wdur = int(rng.uniform(*p.env_wait_ms) * MS_NS)
        ivs.append({"kind": "env_wait", "t0": t, "t1": t + wdur, "n": 0})
        t += wdur
        pos = wp
    dur = max(1, int((gen_len - pos) * p.decode_ms_per_token * MS_NS))
    ivs.append({"kind": "decode", "t0": t, "t1": t + dur, "n": gen_len - pos})
    t += dur
    return {
        "intervals": ivs,
        "n_prompt": n_prompt,
        "gen_len": gen_len,
        "aborted": aborted,
        "abort_reason": abort_reason,
        "start": t0,
        "end": t,
    }


def _cut_plan(pl: dict, syncs: list[dict], p: GenParams, rng: random.Random) -> None:
    """把同步窗口切进段时间线，产出带 pause/resume/weight_skip 标记的线性 script。

    顺序推进 floor_ts：env 等待被窗口顺延、running 区间被切后重排时，后续区间一律
    不得早于前一区间的实际结束——保证状态机在文件序下合法。
    段起点是否落在同步窗口内由 _simulate 的 lane 推进负责，这里不再处理。
    """
    out: list[tuple] = []
    floor_ts = pl["start"]
    for iv0 in pl["intervals"]:
        if not p.pause_at_sync:  # 不参与安全点暂停：段原样冲过同步窗口（span 会触发 W04）
            out.append(("iv", dict(iv0)))
            continue
        iv = dict(iv0)
        shift = max(0, floor_ts - iv["t0"])
        iv["t0"] += shift
        iv["t1"] += shift
        if iv["kind"] == "env_wait":
            for s in syncs:  # 等待中撞上同步：env 返回了也进不了 GPU，出口顺延
                if iv["t0"] < s["t_start"] < iv["t1"]:
                    new_t1 = max(iv["t1"], s["t_end"])
                    iv["t1"] = new_t1
            out.append(("iv", iv))
            floor_ts = iv["t1"]
            continue
        cur: dict | None = iv
        for s in syncs:
            if cur is None:
                break
            gs, ge = s["t_start"], s["t_end"]
            if not (cur["t0"] < gs < cur["t1"]):
                continue
            per_ms = p.prefill_ms_per_token if cur["kind"] == "prefill" else p.decode_ms_per_token
            n_head = int((gs - cur["t0"]) / (per_ms * MS_NS))
            n_rem = cur["n"] - n_head
            if n_head < 1 or n_rem < 1:
                continue  # 切不出有意义的两段，视为冲过边界
            out.append(("iv", {**cur, "t1": gs, "n": n_head}))
            out.append(("pause", gs))
            if cur["kind"] == "decode" and rng.random() > p.resume_prob:
                out.append(("weight_skip", gs, s["version"] - 1))
                cur = None
                break
            nxt = {**cur, "t0": ge, "t1": ge + int(n_rem * per_ms * MS_NS), "n": n_rem}
            if cur["kind"] == "prefill":
                nxt["reason"] = "re-prefill"
            # 续跑 span 的 token 版本由 emit 时按账本取（t0=ge 落在新版本下），不在计划期标记
            cur = nxt
            out.append(("resume", ge))
        if cur is not None:
            out.append(("iv", cur))
            floor_ts = cur["t1"]
    skip = next((i for i, it in enumerate(out) if it[0] == "weight_skip"), None)
    if skip is not None:
        out = out[: skip + 1]
        pl["aborted"] = True
        pl["abort_reason"] = "weight_skip"
        pl["actual_end"] = out[skip][1]  # 在安全点被放弃，实际止于暂停时刻
    else:
        pl["actual_end"] = out[-1][1]["t1"]
    pl["script"] = out
    pl["gen_len"] = sum(it[1]["n"] for it in out if it[0] == "iv" and it[1]["kind"] == "decode")


def _lp_value(rng: random.Random, p: GenParams) -> float:
    v = max(-6.0, min(-0.001, rng.gauss(-0.8, 0.7)))
    return round(v, p.lp_round) if p.lp_round is not None else v


def _emit_segment(
    emit: Callable[..., None], pl: dict, syncs: list[dict], p: GenParams, rng: random.Random
) -> None:
    script: list[tuple] = pl["script"]
    seg_id, group_id = pl["seg_id"], pl["group_id"]
    first_t0 = script[0][1]["t0"]
    first_decode_t0 = next(
        it[1]["t0"] for it in script if it[0] == "iv" and it[1]["kind"] == "decode"
    )
    birth = _version_at(syncs, first_decode_t0)
    emit(
        first_t0,
        "segment_start",
        seg_id=seg_id,
        group_id=group_id,
        birth_version=birth,
        t_start=first_t0,
        n_prompt_tokens=pl["n_prompt"],
        meta={"lane": pl["lane"]},
    )
    cursor = 0
    state = "running"
    for item in script:
        if item[0] == "pause":
            emit(
                item[1],
                "segment_state",
                seg_id=seg_id,
                from_state="running",
                to_state="paused",
                reason="token_boundary",
            )
            state = "paused"
        elif item[0] == "resume":
            emit(item[1], "segment_state", seg_id=seg_id, from_state="paused", to_state="running")
            state = "running"
        elif item[0] == "weight_skip":
            ts, prev_version = item[1], item[2]
            emit(
                ts,
                "segment_end",
                seg_id=seg_id,
                state="aborted",
                from_state="paused",
                reason="weight_skip",
                t_end=ts,
                n_gen_tokens=pl["gen_len"],
                birth_version=birth,
                end_version=prev_version,
            )
            return
        else:
            iv = item[1]
            if iv["kind"] == "env_wait":
                emit(
                    iv["t0"],
                    "segment_state",
                    seg_id=seg_id,
                    from_state="running",
                    to_state="env_wait",
                    reason="env_call:tool",
                )
                emit(
                    iv["t1"],
                    "phase_span",
                    seg_id=seg_id,
                    phase="env_wait",
                    t_start=iv["t0"],
                    t_end=iv["t1"],
                )
                emit(
                    iv["t1"],
                    "segment_state",
                    seg_id=seg_id,
                    from_state="env_wait",
                    to_state="running",
                )
                continue
            kw: dict[str, Any] = {}
            if iv.get("reason"):
                kw["meta"] = {"reason": iv["reason"]}
            emit(
                iv["t1"],
                "phase_span",
                seg_id=seg_id,
                phase=iv["kind"],
                t_start=iv["t0"],
                t_end=iv["t1"],
                n_tokens=iv["n"],
                **kw,
            )
            if iv["kind"] == "decode":
                emit(
                    iv["t1"],
                    "token_logprob",
                    seg_id=seg_id,
                    version=_version_at(syncs, iv["t0"]),  # 该 span 开跑时生效的版本
                    start_idx=cursor,
                    n=iv["n"],
                    lp=[_lp_value(rng, p) for _ in range(iv["n"])],
                )
                cursor += iv["n"]
    end_ts = script[-1][1]["t1"]
    kw: dict[str, Any] = {}
    if pl["aborted"]:
        kw["reason"] = pl["abort_reason"]
    emit(
        end_ts,
        "segment_end",
        seg_id=seg_id,
        state="aborted" if pl["aborted"] else "finished",
        from_state=state,
        t_end=end_ts,
        n_gen_tokens=pl["gen_len"],
        birth_version=birth,
        end_version=_version_at(syncs, end_ts),
        **kw,
    )


def _simulate(p: GenParams, preset: str) -> list[dict]:
    rng = random.Random(p.seed)
    run_id = f"r-gen-{p.seed:04d}"
    anchor = ANCHOR_NS + p.seed * 3_600_000_000_000  # 每个 seed 错开 1h
    events: list[dict] = []

    def emit(ts: int, type_: str, **f: Any) -> None:
        events.append({"ts": ts, "type": type_, "run_id": run_id, **f})

    emit(
        anchor,
        RUN_START,
        format=FORMAT_NAME,
        schema_version=FORMAT_VERSION,
        initial_version=0,
        engine="rheotrace-gen",
        model=p.model,
        clock="wall_ns_epoch",
        meta={"preset": preset, "params": _jsonify(asdict(p))},
    )

    lane_end: dict[int, int] = {}  # pass 1：lane 的计划空闲时刻
    syncs: list[dict] = []
    seg_n = 0
    total_gen = 0
    n_segments = 0
    plans: list[dict] = []
    for step in range(1, p.n_steps + 1):
        base_t = syncs[-1]["t_end"] if syncs else anchor
        gap = int(rng.uniform(*p.schedule_gap_ms) * MS_NS)
        # 信封 ts 按规格 §4.0 取区间结束时刻
        emit(base_t + gap, "phase_span", phase="schedule", t_start=base_t, t_end=base_t + gap)
        floor_t = base_t + gap  # 本步最早可开段时刻（调度间隙之后）
        for i in range(p.groups_per_step * p.group_size):
            lane = i % p.n_workers
            pl = _plan_segment(rng, p, max(lane_end.get(lane, anchor), floor_t))
            pl["lane"] = lane
            pl["group_id"] = f"g-{step:03d}-{i // p.group_size:03d}"
            pl["seg_id"] = f"s-{seg_n:06d}"
            seg_n += 1
            lane_end[lane] = pl["end"]
            plans.append(pl)
        if step < p.n_steps:
            # 同步放在本步"规划时长"的 straddle_frac 分位（不是绝对时刻的分位！），
            # 长尾段因此横跨同步窗口，被切出 pause/resume
            step_base = base_t + gap
            g_start = step_base + int(
                (max(pl["end"] for pl in plans) - step_base) * p.straddle_frac
            )
            if syncs:
                g_start = max(g_start, syncs[-1]["t_end"] + MS_NS)
            syncs.append(
                {
                    "t_start": g_start,
                    "t_end": g_start + int(p.weight_sync_ms * MS_NS),
                    "version": step,
                    "prev_version": step - 1,
                }
            )

    # pass 2：按 lane 顺序推进实际时间线——被同步切断的段实际结束晚于计划端点，
    # 同 lane 下一段必须等它真正跑完，且不得在同步窗口内开段
    by_lane: dict[int, list[dict]] = {}
    for pl in plans:  # plans 追加顺序 = 同 lane 内的时间顺序
        by_lane.setdefault(pl["lane"], []).append(pl)
    for lane in sorted(by_lane):
        cursor = anchor
        for pl in by_lane[lane]:
            start = max(pl["start"], cursor)
            for s in syncs:
                if s["t_start"] < start < s["t_end"]:
                    start = s["t_end"]  # 不在同步窗口内开段
                    break
            delta = start - pl["start"]
            if delta:
                for iv in pl["intervals"]:
                    iv["t0"] += delta
                    iv["t1"] += delta
                pl["start"] = start
            _cut_plan(pl, syncs, p, rng)
            _emit_segment(emit, pl, syncs, p, rng)
            cursor = max(cursor, pl["actual_end"])
            total_gen += pl["gen_len"]
            n_segments += 1

    for s in syncs:
        emit(
            s["t_end"],
            "weight_sync",
            version=s["version"],
            t_start=s["t_start"],
            t_end=s["t_end"],
            mode="full",
            trainer_step=s["version"],
        )

    events.append(
        {
            "ts": max(e["ts"] for e in events) + MS_NS,
            "type": "run_end",
            "run_id": run_id,
            "summary": {
                "segments": n_segments,
                "weight_syncs": len(syncs),
                "gen_tokens": total_gen,
            },
        }
    )
    # 账本先行：同一时刻上 weight_sync 必须先于续跑/新段事件被读到
    events.sort(key=lambda e: (e["ts"], e["type"] not in ("run_start", "run_end", "weight_sync")))
    return events
