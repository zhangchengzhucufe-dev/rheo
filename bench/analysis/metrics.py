"""指标计算：docs/metrics-v0.md 的 T1-T3 / M1-M2 / P1-P4 / S1-S3 / F1-F5 / Q 全套。

只对 canon（见 bench/analysis/canon.py）计算，纯函数、无 IO。
时间单位：内部一律 ns（整数），Analysis 输出的标量时间为秒（float）。

性能注记：P2 停顿分类是 O(停顿基本块 × 轨迹数) 的精确扫描，实测
47MB / 12 万事件（~1.1 万段）全流程 ~20s。M1 规模（单 run ≤ GB 级）
可用；更大规模先分 run 或再优化扫描。
"""

from __future__ import annotations

import json
import math
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .canon import PHASE_DECODE, PHASE_PREFILL, Trace

# ---------------------------------------------------------------------------
# 模型与硬件参数


@dataclass(frozen=True)
class ModelConfig:
    P: float
    L: int
    d: int


# 内置模型表：参数量为公开资料近似值，仅用于没带 model-config 的快速分析。
# 键做子串匹配（大小写不敏感）：B 的合成 trace 模型名形如 "synthetic-1.5B"，
# 按 "1.5b" 等规模记号落到对应几何；正式分析请用 --model-config 提供精确参数。
BUILTIN_MODELS: dict[str, ModelConfig] = {
    "qwen2.5-0.5b": ModelConfig(P=0.49e9, L=24, d=896),
    "qwen2.5-1.5b": ModelConfig(P=1.54e9, L=28, d=1536),
    "qwen2.5-3b": ModelConfig(P=3.09e9, L=36, d=2048),
    "qwen2.5-7b": ModelConfig(P=7.62e9, L=28, d=3584),
}
MODEL_SIZE_ALIASES: dict[str, str] = {
    "0.5b": "qwen2.5-0.5b",
    "1.5b": "qwen2.5-1.5b",
    "3b": "qwen2.5-3b",
    "7b": "qwen2.5-7b",
}

# RTX 3060 FP16 稠密 tensor 峰值（数据表占位值，推荐 --peak-tflops 传实测 GEMM 峰值）
DEFAULT_PEAK_TFLOPS = 25.3

BC_BIMODAL_THRESHOLD = 5 / 9  # Sarle 双峰系数判定线
STRAGGLER_OCC_THRESHOLD = 0.5  # S1 占用率阈值（metrics-v0 §3.3，v0 固定）


class MetricsError(ValueError):
    """参数无法解析（模型未知、峰值缺失等），trace 本身合法。"""


def resolve_model_config(trace: Trace, model_config_path: str | Path | None) -> ModelConfig:
    """--model-config > header 内嵌参数 > 内置表（按 header.model 名）。"""
    if model_config_path is not None:
        obj = json.loads(Path(model_config_path).read_text("utf-8"))
        return ModelConfig(P=float(obj["P"]), L=int(obj["L"]), d=int(obj["d"]))
    h = trace.header
    if h.params is not None:
        return ModelConfig(P=h.params.P, L=h.params.L, d=h.params.d)
    if h.model:
        key = h.model.lower().replace("_", "-")
        if key in BUILTIN_MODELS:
            return BUILTIN_MODELS[key]
        for size, full in MODEL_SIZE_ALIASES.items():
            # 词边界防误配："13b" 含 "3b" 但前缀是数字 → 不算命中
            start = key.find(size)
            while start != -1:
                if start == 0 or not key[start - 1].isdigit():
                    return BUILTIN_MODELS[full]
                start = key.find(size, start + 1)
    raise MetricsError(
        f"无法确定模型参数（P/L/d）：header.model={h.model!r} 不在内置表，"
        '请用 --model-config 提供 {"P":…,"L":…,"d":…}'
    )


# ---------------------------------------------------------------------------
# 区间代数（闭开区间，ns）


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """并集：合并重叠/相邻区间。"""
    out: list[tuple[int, int]] = []
    for a, b in sorted(spans):
        if a == b:
            continue
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def subtract_spans(
    base: list[tuple[int, int]], cuts: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """base ∖ cuts（两者各自不重叠）。"""
    out: list[tuple[int, int]] = []
    for a, b in base:
        cur = a
        for c0, c1 in cuts:
            if c1 <= cur or c0 >= b:
                continue
            if c0 > cur:
                out.append((cur, c0))
            cur = max(cur, c1)
            if cur >= b:
                break
        if cur < b:
            out.append((cur, b))
    return out


def intersect_spans(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    i = j = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if lo < hi:
            out.append((lo, hi))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return out


def total_dur(spans: list[tuple[int, int]]) -> int:
    return sum(b - a for a, b in spans)


# ---------------------------------------------------------------------------
# 分析结果容器


@dataclass(frozen=True)
class PausePiece:
    t0: int
    t1: int
    label: str  # P1 weight_sync | P2 env_wait | P3 schedule_gap | P4 other


@dataclass(frozen=True)
class BatchOcc:
    batch: str
    b0: int
    points: tuple[tuple[int, float], ...]  # (t_ns, occ) 阶梯点
    tail_ratio: float  # S2：(t_last - t_p50)/t_p50


@dataclass(frozen=True)
class Analysis:
    # 元数据
    model: str | None
    cfg: ModelConfig
    peak_tflops: float
    world_size: int
    t0_ns: int
    t1_ns: int
    n_trajs: int
    source: str | None
    sha256: str | None
    # token 计数（铁律：pre/decode 分开）
    prompt_tokens: int
    gen_tokens_all: int
    gen_tokens_committed: int
    n_aborted: int
    # 吞吐
    t_wall_s: float
    t_active_s: float
    thr_e2e: float  # T1
    thr_active: float  # T2
    thr_prefill: float  # T3
    duty: float  # T_active/T_wall
    # MFU
    flops_total: float
    m1_engine: float
    m2_rollout: float
    waste: float
    # 停顿
    pause_pieces: tuple[PausePiece, ...]
    pause_totals: dict[str, float]  # label -> 秒
    active_spans: tuple[tuple[int, int], ...]  # ns
    # 批内
    batch_occs: tuple[BatchOcc, ...]
    t_straggler_s: float
    straggler_share: float  # S1
    n_staggered_batches: int  # 从未同时满员的伪批数（S1 对其不适用）
    # env 等待（轨迹级）
    env_wait_s: tuple[float, ...]  # 每轨迹总等待，秒
    env_wait_share: tuple[float, ...]  # 占该轨迹墙钟比例
    traj_wall_s: tuple[float, ...]  # 每轨迹墙钟，秒（与 lengths_all 同序）
    # 长度分布
    lengths_all: np.ndarray  # 含 abort
    lengths_committed: np.ndarray  # 仅提交
    lengths_all_valid: np.ndarray  # 含 abort、仅含已完成轨迹
    lengths_committed_valid: np.ndarray
    tail_token_share: float  # F2
    sarle_bc: float  # F3（nan 当 n<4）
    stale_token_share: float  # F5
    # 数据质量
    warnings: tuple[str, ...]
    n_unfinished: int


# ---------------------------------------------------------------------------
# 主入口


def analyze(
    trace: Trace,
    *,
    peak_tflops: float | None = None,
    model_config_path: str | Path | None = None,
) -> Analysis:
    cfg = resolve_model_config(trace, model_config_path)
    peak = (
        float(peak_tflops)
        if peak_tflops is not None
        else (
            trace.header.peak_tflops
            if trace.header.peak_tflops is not None
            else DEFAULT_PEAK_TFLOPS
        )
    )
    world = max(1, trace.header.world_size)

    execs = trace.execs
    if not execs:
        raise MetricsError("trace 中没有任何执行段，无法分析")

    t0 = min(e.t0 for e in execs)
    t1 = max(
        max((e.t1 for e in execs), default=t0),
        max((w.t1 for w in trace.env_waits), default=t0),
        max((s.t1 for s in trace.syncs), default=t0),
        max((d.t for d in trace.traj_ends), default=t0),
    )

    # ---- 轨迹生命周期 -----------------------------------------------------
    trajs: dict[str, list] = {}
    for e in execs:
        trajs.setdefault(e.traj, []).append(e)
    birth: dict[str, int] = {}
    last_ev: dict[str, int] = {}
    for w in trace.env_waits:
        birth.setdefault(w.traj, w.t0)
        last_ev[w.traj] = max(last_ev.get(w.traj, w.t0), w.t1)
    for e in execs:
        birth.setdefault(e.traj, e.t0)
        last_ev[e.traj] = max(last_ev.get(e.traj, e.t0), e.t1)
    end = dict(last_ev)
    status: dict[str, str] = {}
    for d in trace.traj_ends:
        end[d.traj] = max(end.get(d.traj, d.t), d.t)
        status[d.traj] = d.status
    n_unfinished = sum(1 for t in trajs if t not in status)

    # ---- active / pause ---------------------------------------------------
    active = merge_spans([(e.t0, e.t1) for e in execs])
    pause = invert_within(active, t0, t1)
    t_active = total_dur(active)

    # P1：与 sync 的交集优先收走（固定优先级，metrics-v0 §3.2）
    sync_spans = merge_spans([(s.t0, s.t1) for s in trace.syncs])
    if intersect_spans(sync_spans, active):
        warn = "W-SYNC-OVERLAP：权重同步区间与执行区间重叠；canon 语义里 sync 应为阻塞区间"
    else:
        warn = None
    p1_spans = intersect_spans(pause, sync_spans)
    rest = subtract_spans(pause, sync_spans)

    env_by_traj: dict[str, list[tuple[int, int]]] = {}
    for w in trace.env_waits:
        env_by_traj.setdefault(w.traj, []).append((w.t0, w.t1))
    for spans in env_by_traj.values():
        spans.sort()

    pieces: list[PausePiece] = [PausePiece(a, b, "P1") for a, b in p1_spans]
    for a0, b0 in rest:
        # "所有在途轨迹同时 env_wait"的谓词只在这些边界点间变化：
        # 出生/终止/env 进出，全部收进来做基本区间扫描（精确，不是中点近似）
        pts = {a0, b0}
        for tr in trajs:
            if a0 < birth[tr] < b0:
                pts.add(birth[tr])
            if a0 < end[tr] < b0:
                pts.add(end[tr])
            for s, e2 in env_by_traj.get(tr, []):
                if a0 < s < b0:
                    pts.add(s)
                if a0 < e2 < b0:
                    pts.add(e2)
        ts_sorted = sorted(pts)
        for p, q in zip(ts_sorted, ts_sorted[1:], strict=False):
            inflight = [tr for tr in trajs if birth[tr] <= p < end[tr]]
            if inflight and all(_covers(env_by_traj.get(tr, []), p) for tr in inflight):
                pieces.append(PausePiece(p, q, "P2"))
            else:
                pieces.append(PausePiece(p, q, "P3"))
    pieces.sort(key=lambda p_: (p_.t0, p_.t1))
    pause_totals = {"P1": 0.0, "P2": 0.0, "P3": 0.0, "P4": 0.0}
    for p in pieces:
        pause_totals[p.label] += (p.t1 - p.t0) / 1e9

    # ---- token 计数 -------------------------------------------------------
    prompt_tokens = sum(e.n_tok for e in execs if e.phase == PHASE_PREFILL)
    gen_all = sum(e.n_tok for e in execs if e.phase == PHASE_DECODE)
    gen_committed = sum(e.n_tok for e in execs if e.phase == PHASE_DECODE and e.committed)
    n_aborted = sum(1 for t in trajs if status.get(t) == "aborted")

    # ---- 吞吐 -------------------------------------------------------------
    t_wall = (t1 - t0) / 1e9
    t_act = t_active / 1e9
    # T3 用 prefill 活跃区间的并集：并发 prefill 不能按段重复计时
    prefill_union = merge_spans([(e.t0, e.t1) for e in execs if e.phase == PHASE_PREFILL])
    prefill_dur = total_dur(prefill_union) / 1e9
    thr_e2e = gen_all / t_wall / world if t_wall > 0 else 0.0
    thr_active = gen_all / t_act / world if t_act > 0 else 0.0
    thr_prefill = prompt_tokens / prefill_dur / world if prefill_dur > 0 else 0.0
    duty = t_act / t_wall if t_wall > 0 else 0.0

    # ---- MFU（metrics-v0 §2）----------------------------------------------
    flops, n_no_prefill = _useful_flops(trajs, cfg)
    flops_total = flops
    m1 = flops_total / (peak * 1e12 * t_act) if t_act > 0 else 0.0
    m2 = flops_total / (peak * 1e12 * t_wall) if t_wall > 0 else 0.0
    waste = 1.0 - gen_committed / gen_all if gen_all > 0 else 0.0

    # ---- 批内占用与掉队（S1-S3）--------------------------------------------
    batches: dict[str, list[str]] = {}
    for e in execs:
        batches.setdefault(e.batch, [])
        if e.traj not in batches[e.batch]:
            batches[e.batch].append(e.traj)
    batch_occs: list[BatchOcc] = []
    straggler_spans: list[tuple[int, int]] = []
    overlap_flag = False
    batch_ranges: list[tuple[int, int, str]] = []
    n_staggered = 0
    for bname, members in sorted(batches.items()):
        b0 = len(members)
        events: list[tuple[int, int]] = []
        for m in members:
            events.append((birth[m], 1))
            events.append((end[m], -1))
        events.sort()
        bs, be = events[0][0], max(end[m] for m in members)
        batch_ranges.append((bs, be, bname))
        # 阶梯扫描：0 < k < 0.5·B0 的时间即掉队；峰值占用率到过 1.0（同时派发）才有效
        pts: list[tuple[int, float]] = [(bs, 0.0)]
        k = 0
        max_occ = 0.0
        cur_span_start: int | None = None
        spans_b: list[tuple[int, int]] = []
        for t, delta in events:
            prev_occ = k / b0
            k += delta
            occ = k / b0
            pts.append((t, occ))
            max_occ = max(max_occ, occ)
            was_strag = 0 < prev_occ < STRAGGLER_OCC_THRESHOLD
            is_strag = 0 < occ < STRAGGLER_OCC_THRESHOLD
            if is_strag and not was_strag:
                cur_span_start = t
            elif was_strag and not is_strag and cur_span_start is not None:
                spans_b.append((cur_span_start, t))
                cur_span_start = None
        if cur_span_start is not None:
            spans_b.append((cur_span_start, be))
        if max_occ < 1.0 - 1e-9:
            # 伪批从未同时满员（如分 lane 流水派发）：占用率口径不适用，剔除
            n_staggered += 1
        else:
            straggler_spans.extend(spans_b)
        ends = sorted(end[m] for m in members)
        # 中位数保持整数域（float64 在 wall_ns_epoch 量级只有 ~256ns 精度）
        n_e = len(ends)
        p50 = ends[n_e // 2] if n_e % 2 else (ends[n_e // 2 - 1] + ends[n_e // 2]) // 2
        # 相对批起点取比值：分母用绝对 epoch 时刻（wall_ns_epoch ~1.7e18ns）会退化成 ~1e-8 废值
        span = p50 - bs
        tail_ratio = (ends[-1] - p50) / span if span > 0 else 0.0
        batch_occs.append(BatchOcc(batch=bname, b0=b0, points=tuple(pts), tail_ratio=tail_ratio))
    batch_ranges.sort()
    for (_s0, s1_, _n1), (o0, _o1, _n2) in zip(batch_ranges, batch_ranges[1:], strict=False):
        if o0 < s1_:
            overlap_flag = True
    strag = intersect_spans(merge_spans(straggler_spans), active)
    t_strag = total_dur(strag) / 1e9
    straggler_share = t_strag / t_act if t_act > 0 else 0.0

    # ---- env 等待（轨迹级，§3.4）-------------------------------------------
    env_s: list[float] = []
    env_share: list[float] = []
    traj_wall: list[float] = []
    for t in trajs:
        tot = sum(b - a for a, b in env_by_traj.get(t, []))
        wall = end[t] - birth[t]
        env_s.append(tot / 1e9)
        env_share.append(tot / wall if wall > 0 else 0.0)
        traj_wall.append(wall / 1e9)

    # ---- 长度分布（F1-F3）--------------------------------------------------
    len_all: list[float] = []
    len_committed: list[float] = []
    len_all_valid: list[float] = []
    len_committed_valid: list[float] = []
    for t, execs_t in trajs.items():
        a = sum(e.n_tok for e in execs_t if e.phase == PHASE_DECODE)
        c = sum(e.n_tok for e in execs_t if e.phase == PHASE_DECODE and e.committed)
        len_all.append(float(a))
        len_committed.append(float(c))
        if t in status:  # 未完成轨迹不入分布（Q2）
            len_all_valid.append(float(a))
            if status[t] != "aborted":  # "不含 abort" 列：整条剔除，而非记 0
                len_committed_valid.append(float(c))
    arr_all_v = np.asarray(len_all_valid, dtype=float)
    arr_com_v = np.asarray(len_committed_valid, dtype=float)
    tail_share = _tail_token_share(arr_all_v)
    bc = _sarle_bc(arr_all_v)

    # ---- stale 暴露面（F5）-------------------------------------------------
    stale_share = _stale_token_share(trace, execs)

    # ---- 数据质量（Q）------------------------------------------------------
    warns: list[str] = []
    if warn:
        warns.append(warn)
    if n_unfinished:
        warns.append(f"Q2：{n_unfinished} 条轨迹未终止（缺 traj_end），已从长度分布剔除")
    if overlap_flag:
        warns.append("W-BATCH-OVERLAP：批区间在时间上重叠，S1 掉队份额可能重复计入")
    p4_share = pause_totals["P4"] / t_wall if t_wall > 0 else 0.0
    if p4_share > 0.01:
        warns.append("Q1：P4 other 占墙钟 >1%，通常意味着 trace 字段不足")
    if batch_occs and all(b.batch.startswith("cluster-") for b in batch_occs):
        warns.append(
            "W-NO-BATCH：trace 无 batch 标注，S1–S3 按轨迹时间重叠聚类近似"
            "（方向不定，详见 W-BATCH-STAGGERED）；见 issues.md 对 B 的 batch_id 请求"
        )
    if batch_occs and n_staggered == len(batch_occs):
        warns.append(
            "W-BATCH-STAGGERED：所有伪批从未同时满员派发（如分 lane 流水执行），"
            "占用率/掉队口径不适用，S1 已置 0；待 batch_id 增量后恢复"
        )
    elif n_staggered:
        warns.append(
            f"W-BATCH-STAGGERED：{n_staggered}/{len(batch_occs)} 个批从未同时满员，"
            "其掉队时间未计入 S1"
        )
    if n_no_prefill:
        warns.append(
            f"W-NO-PREFILL：{n_no_prefill} 条轨迹有 decode 无 prefill，attention 项按 KV=0 估计"
        )

    return Analysis(
        model=trace.header.model,
        cfg=cfg,
        peak_tflops=peak,
        world_size=world,
        t0_ns=t0,
        t1_ns=t1,
        n_trajs=len(trajs),
        source=str(trace.path) if trace.path else None,
        sha256=trace.sha256,
        prompt_tokens=prompt_tokens,
        gen_tokens_all=gen_all,
        gen_tokens_committed=gen_committed,
        n_aborted=n_aborted,
        t_wall_s=t_wall,
        t_active_s=t_act,
        thr_e2e=thr_e2e,
        thr_active=thr_active,
        thr_prefill=thr_prefill,
        duty=duty,
        flops_total=flops_total,
        m1_engine=m1,
        m2_rollout=m2,
        waste=waste,
        pause_pieces=tuple(pieces),
        pause_totals=pause_totals,
        active_spans=tuple(active),
        batch_occs=tuple(batch_occs),
        t_straggler_s=t_strag,
        straggler_share=straggler_share,
        n_staggered_batches=n_staggered,
        env_wait_s=tuple(env_s),
        env_wait_share=tuple(env_share),
        traj_wall_s=tuple(traj_wall),
        lengths_all=np.asarray(len_all, dtype=float),
        lengths_committed=np.asarray(len_committed, dtype=float),
        lengths_all_valid=arr_all_v,
        lengths_committed_valid=arr_com_v,
        tail_token_share=tail_share,
        sarle_bc=bc,
        stale_token_share=stale_share,
        warnings=tuple(warns),
        n_unfinished=n_unfinished,
    )


# ---------------------------------------------------------------------------
# 分解实现


def invert_within(spans: list[tuple[int, int]], t0: int, t1: int) -> list[tuple[int, int]]:
    """spans 在 [t0, t1) 内的补集。"""
    out: list[tuple[int, int]] = []
    cur = t0
    for a, b in spans:
        if a >= t1:
            break
        if a > cur:
            out.append((cur, min(a, t1)))
        cur = max(cur, b)
    if cur < t1:
        out.append((cur, t1))
    return out


def _covers(spans: list[tuple[int, int]], t: int) -> bool:
    """t 是否落在任一区间内（区间已排序）。"""
    for a, b in spans:
        if a <= t < b:
            return True
        if a > t:
            break
    return False


def _useful_flops(trajs: dict[str, list], cfg: ModelConfig) -> tuple[float, int]:
    """解析式 FLOPs（metrics-v0 §2.1）。返回 (flops, 无 prefill 的轨迹数)。"""
    total = 0.0
    n_no_prefill = 0
    for execs_t in trajs.values():
        s = 0  # 当前 KV 长度
        saw_prefill = False
        flagged = False
        for e in sorted(execs_t, key=lambda x: x.t0):
            if e.phase == PHASE_PREFILL:
                total += 2.0 * cfg.P * e.n_tok
                s = e.n_tok
                saw_prefill = True
            else:
                if not saw_prefill:
                    if not flagged:
                        n_no_prefill += 1
                        flagged = True
                s_end = s + e.n_tok
                # decode：2P/token + attention 对全部缓存 KV（梯形近似段内均值）
                total += 2.0 * cfg.P * e.n_tok + 4.0 * cfg.L * cfg.d * e.n_tok * (s + s_end) / 2.0
                s = s_end
    return total, n_no_prefill


def _tail_token_share(lengths: np.ndarray) -> float:
    """F2 尾部 token 份额：长度 > P90 的轨迹 token 数 / 总 token 数。"""
    if lengths.size == 0 or lengths.sum() == 0:
        return 0.0
    p90 = float(np.percentile(lengths, 90))
    return float(lengths[lengths > p90].sum() / lengths.sum())


def _sarle_bc(lengths: np.ndarray) -> float:
    """F3 Sarle 双峰系数 BC=(g1²+1)/(γ2+修正项)，>5/9 判双峰迹象；n<4 返回 nan。

    γ2 为超额峰度（m4/m2²−3）；5/9 恰为均匀分布的 BC 值（SAS 同款口径）。
    """
    n = lengths.size
    if n < 4:
        return math.nan
    mu = lengths.mean()
    m2 = float(((lengths - mu) ** 2).mean())
    if m2 == 0:
        return math.nan
    m3 = float(((lengths - mu) ** 3).mean())
    m4 = float(((lengths - mu) ** 4).mean())
    g1 = m3 / m2**1.5
    excess = m4 / m2**2 - 3.0
    return (g1 * g1 + 1.0) / (excess + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def _stale_token_share(trace: Trace, execs: tuple) -> float:
    """F5：生成时当前版本 > birth_version 的 token 份额，段内 bump 按时间线性切分。"""
    bump_ts = [b.t for b in trace.bumps]
    bump_vs = [b.ver for b in trace.bumps]

    def ver_at(t: int) -> int:
        i = bisect_right(bump_ts, t) - 1
        return bump_vs[i] if i >= 0 else 0

    total = 0
    stale = 0.0
    for e in execs:
        if e.phase != PHASE_DECODE:
            continue
        total += e.n_tok
        if e.t1 == e.t0:
            continue
        bounds = [e.t0] + [b.t for b in trace.bumps if e.t0 < b.t < e.t1] + [e.t1]
        rate = e.n_tok / (e.t1 - e.t0)
        for a, b in zip(bounds, bounds[1:], strict=False):
            if ver_at(a) > e.bv:
                stale += rate * (b - a)
    return stale / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# 长度分布统计表（F1）


def length_stats(vals: np.ndarray) -> dict[str, float]:
    if vals.size == 0:
        return {"n": 0.0}
    qs = np.percentile(vals, [10, 25, 50, 75, 90, 95, 99])
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    return {
        "n": float(vals.size),
        "mean": mean,
        "std": std,
        "cv": std / mean if mean > 0 else 0.0,
        "min": float(vals.min()),
        "p10": float(qs[0]),
        "p25": float(qs[1]),
        "p50": float(qs[2]),
        "p75": float(qs[3]),
        "p90": float(qs[4]),
        "p95": float(qs[5]),
        "p99": float(qs[6]),
        "max": float(vals.max()),
        "p99/p50": float(qs[6] / qs[2]) if qs[2] > 0 else 0.0,
        "max/p50": float(vals.max() / qs[2]) if qs[2] > 0 else 0.0,
    }
