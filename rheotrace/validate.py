"""RheoTrace validator：规格 docs/rheotrace-spec-v0.md §6 规则表（E01–E18 / W01–W09）的实现。"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .core import (
    CLOCKS,
    ENGINE_LEVEL_PHASES,
    FINISH_MODES,
    INTERVAL_END_TYPES,
    LEGAL_TRANSIENT,
    PHASE_SPAN,
    PHASES,
    RUN_END,
    RUN_START,
    SEGMENT_END,
    SEGMENT_LEVEL_PHASES,
    SEGMENT_START,
    SEGMENT_STATE,
    SEGMENT_STATES,
    SYNC_MODES,
    TOKEN_LOGPROB,
    TRANSIENT_STATES,
    WEIGHT_SYNC,
    ValidationReport,
)
from .reader import open_text

Source = str | Path | Iterable[dict]

# 各事件类型的必填字段与类型（规格 §4 各表）。类型标记：int/str/numlist
_REQUIRED: dict[str, dict[str, str]] = {
    RUN_START: {
        "format": "str",
        "schema_version": "int",
        "initial_version": "int",
        "engine": "str",
        "model": "str",
        "clock": "str",
    },
    WEIGHT_SYNC: {"version": "int", "t_start": "int", "t_end": "int", "mode": "str"},
    PHASE_SPAN: {"phase": "str", "t_start": "int", "t_end": "int"},
    SEGMENT_START: {
        "seg_id": "str",
        "group_id": "str",
        "birth_version": "int",
        "t_start": "int",
        "n_prompt_tokens": "int",
    },
    SEGMENT_STATE: {"seg_id": "str", "from_state": "str", "to_state": "str"},
    SEGMENT_END: {
        "seg_id": "str",
        "state": "str",
        "from_state": "str",
        "t_end": "int",
        "n_gen_tokens": "int",
        "birth_version": "int",
        "end_version": "int",
    },
    TOKEN_LOGPROB: {
        "seg_id": "str",
        "version": "int",
        "start_idx": "int",
        "n": "int",
        "lp": "numlist",
    },
    RUN_END: {},
}


def _is_int(v: object) -> bool:
    return type(v) is int  # bool 是 int 子类，显式排除


def _is_num(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


@dataclass
class _Seg:
    line: int
    birth_version: int
    t_start: int
    state: str
    t_end: int | None = None
    closed: bool = False
    lp_chunks: list[tuple[int, int]] = field(default_factory=list)  # (start_idx, end_idx)
    lp_max: int = 0
    env_wait_open: int | None = None
    env_wait_intervals: list[tuple[int, int]] = field(default_factory=list)


def _iter_events(source: Source, rep: ValidationReport) -> Iterator[tuple[int, dict]]:
    """统一来源：路径 → 行解析（含截断尾行处理 W01/E02）；可迭代 → 假定行号=序号。"""
    if isinstance(source, (str, Path)):
        with open_text(source) as fh:
            raw = fh.readlines()
        truncated_tail = bool(raw) and not raw[-1].endswith("\n")
        for i, line in enumerate(raw, 1):
            s = line.strip()
            if not s:
                continue
            try:
                yield i, json.loads(s)
            except json.JSONDecodeError:
                if i == len(raw) and truncated_tail:
                    rep.add_warning("W01", "末尾残行无法解析，已跳过（文件截断）", i)
                    return
                rep.add_error("E02", "JSON 行解析失败", i)
    else:
        for i, ev in enumerate(source, 1):
            yield i, ev


def validate(source: Source, *, strict: bool = True) -> ValidationReport:
    """按规格 §6 校验一份 trace。

    strict=True：存在任一 error 即抛 ValidationError（携带完整报告）；
    strict=False：总是返回报告。
    """
    rep = ValidationReport()
    _check(source, rep)
    if strict:
        rep.raise_if_errors()
    return rep


def _check(source: Source, rep: ValidationReport) -> None:
    first = True
    saw_run_end = False
    run_id: str | None = None
    initial_version = 0
    current_version = 0
    final_version = 0
    last_ts: int | None = None
    prev_sync_end: int | None = None
    n_syncs = 0
    sync_windows: list[tuple[int, int, int]] = []  # (t_start, t_end, version)
    segs: dict[str, _Seg] = {}

    for line, ev in _iter_events(source, rep):
        if not isinstance(ev, dict):
            rep.add_error("E03", "事件不是 JSON object", line)
            continue
        ts = ev.get("ts")
        etype = ev.get("type")
        rid = ev.get("run_id")
        if not _is_int(ts):
            rep.add_error("E03", "公共信封：ts 缺失或非整数", line)
            continue
        if not isinstance(etype, str) or not isinstance(rid, str):
            rep.add_error("E03", "公共信封：type / run_id 缺失或非字符串", line)
            continue
        if last_ts is not None and ts < last_ts:
            rep.add_error("E04", f"事件乱序：ts={ts} 早于前一事件的 ts={last_ts}", line)
        last_ts = ts

        if first:
            if etype != RUN_START:
                rep.add_error("E01", f"文件必须以 run_start 开头，实际以 {etype} 开头", line)
                return
            first = False
            run_id = rid
            bad = _check_required(ev, RUN_START, line, rep)
            if bad:
                continue
            if ev["clock"] not in CLOCKS:
                rep.add_error("E06", f"非法 clock: {ev['clock']!r}", line)
            initial_version = ev["initial_version"]
            current_version = initial_version
            final_version = initial_version
            continue

        if etype == RUN_START:
            rep.add_error("E01", "出现第二个 run_start（一文件一 run）", line)
            continue
        if rid != run_id:
            rep.add_error("E03", f"run_id 不一致：{rid!r} ≠ {run_id!r}", line)
            continue
        if etype == RUN_END:
            if saw_run_end:
                rep.add_error("E16", "run_end 之后再次出现 run_end", line)
                continue
            saw_run_end = True
            for sid, seg in segs.items():
                if not seg.closed:
                    rep.add_warning(
                        "W02", f"run 结束时段 {sid} 仍处于 {seg.state}（未正常收尾）", line
                    )
            continue
        if saw_run_end:
            rep.add_error("E16", f"run_end 之后仍有事件 {etype}", line)
            continue

        if etype not in _REQUIRED:
            rep.add_warning("W06", f"未知事件类型 {etype!r}（前向兼容，已跳过）", line)
            continue
        if _check_required(ev, etype, line, rep):
            continue

        if etype in INTERVAL_END_TYPES:
            # 规格同步单调（§4.0）：区间型事件以结束时刻为 ts；早于 t_end 是时间线矛盾，
            # 晚于 t_end 视为迟写（时间核算仍以 t_end 为准）
            if ts < ev["t_end"]:
                rep.add_error("E18", f"{etype} 的 ts={ts} 早于区间结束 t_end={ev['t_end']}", line)
            elif ts > ev["t_end"]:
                rep.add_warning("W09", f"{etype} 的 ts={ts} 晚于 t_end={ev['t_end']}（迟写）", line)

        if etype == WEIGHT_SYNC:
            version, t_start, t_end = ev["version"], ev["t_start"], ev["t_end"]
            if ev["mode"] not in SYNC_MODES:
                rep.add_error("E06", f"非法 mode: {ev['mode']!r}", line)
            if t_end < t_start:
                rep.add_error(
                    "E11", f"weight_sync 区间倒挂：t_end={t_end} < t_start={t_start}", line
                )
            if version <= current_version:
                rep.add_error(
                    "E09",
                    f"版本回退：weight_sync version={version} 未严格递增（当前 {current_version}）",
                    line,
                )
            elif version > current_version + 1:
                rep.add_warning("W08", f"版本跳变 {current_version}→{version}（可能漏同步）", line)
            if prev_sync_end is not None and t_start < prev_sync_end:
                rep.add_error(
                    "E17", f"同步窗口与上一窗口重叠：t_start={t_start} < {prev_sync_end}", line
                )
            current_version = max(current_version, version)
            final_version = max(final_version, version)
            prev_sync_end = t_end if prev_sync_end is None else max(prev_sync_end, t_end)
            if t_end >= t_start:
                sync_windows.append((t_start, t_end, version))
            n_syncs += 1

        elif etype == PHASE_SPAN:
            phase, t_start, t_end = ev["phase"], ev["t_start"], ev["t_end"]
            seg_id = ev.get("seg_id")
            if phase not in PHASES:
                rep.add_error("E06", f"非法 phase: {phase!r}", line)
                continue
            if t_end < t_start:
                rep.add_error("E11", f"span 区间倒挂：t_end={t_end} < t_start={t_start}", line)
            if phase in ENGINE_LEVEL_PHASES and seg_id is not None:
                rep.add_error("E12", f"engine 级 phase {phase!r} 不应带 seg_id", line)
                continue
            if phase in SEGMENT_LEVEL_PHASES and seg_id is None:
                rep.add_error("E12", f"段级 phase {phase!r} 缺 seg_id", line)
                continue
            if seg_id is None:
                continue
            seg = segs.get(seg_id)
            if seg is None:
                rep.add_error("E07", f"span 引用未开启的段 {seg_id}", line)
                continue
            if seg.closed:
                rep.add_error("E08", f"段 {seg_id} 已终态，仍有 span", line)
                continue
            if t_start < seg.t_start:
                rep.add_error("E12", f"span 早于段开始（seg t_start={seg.t_start}）", line)
            if seg.t_end is not None and t_end > seg.t_end:
                rep.add_error("E12", f"span 晚于段结束（seg t_end={seg.t_end}）", line)
            if phase in {"decode", "prefill"}:
                for gs, ge, _v in sync_windows:
                    if t_start < ge and gs < t_end:
                        rep.add_warning(
                            "W04",
                            f"{phase} span [{t_start},{t_end}] 横跨 weight_sync [{gs},{ge}]",
                            line,
                        )
                        break
            if phase == "env_wait":
                inside_closed = any(a <= t_start and t_end <= b for a, b in seg.env_wait_intervals)
                inside_open = seg.env_wait_open is not None and t_start >= seg.env_wait_open
                if not (inside_closed or inside_open):
                    rep.add_warning(
                        "W05",
                        f"env_wait span [{t_start},{t_end}] 不落在该段 env_wait 状态区间内",
                        line,
                    )

        elif etype == SEGMENT_START:
            seg_id = ev["seg_id"]
            if seg_id in segs:
                rep.add_error("E07", f"段 {seg_id} 重复开启", line)
                continue
            if ev["n_prompt_tokens"] < 0:
                rep.add_error("E05", f"n_prompt_tokens 为负: {ev['n_prompt_tokens']}", line)
            if ev["birth_version"] < initial_version:
                rep.add_error(
                    "E10",
                    f"birth_version={ev['birth_version']} < initial_version={initial_version}",
                    line,
                )
            segs[seg_id] = _Seg(
                line=line,
                birth_version=ev["birth_version"],
                t_start=ev["t_start"],
                state="running",
            )

        elif etype == SEGMENT_STATE:
            seg = segs.get(ev["seg_id"])
            if seg is None:
                rep.add_error("E07", f"状态转换引用未开启的段 {ev['seg_id']}", line)
                continue
            if seg.closed:
                rep.add_error("E08", f"段 {ev['seg_id']} 已终态，仍有状态转换", line)
                continue
            from_state, to_state = ev["from_state"], ev["to_state"]
            if from_state not in TRANSIENT_STATES or to_state not in TRANSIENT_STATES:
                rep.add_error(
                    "E06", f"非法状态 {from_state!r}/{to_state!r}（终态须经 segment_end）", line
                )
                continue
            if from_state != seg.state:
                rep.add_error(
                    "E08", f"from_state={from_state!r} 与段当前状态 {seg.state!r} 不符", line
                )
                continue
            if to_state not in LEGAL_TRANSIENT[from_state]:
                rep.add_error("E08", f"非法转换 {from_state}→{to_state}", line)
                continue
            if from_state == "env_wait" and seg.env_wait_open is not None:
                seg.env_wait_intervals.append((seg.env_wait_open, ts))
                seg.env_wait_open = None
            if to_state == "env_wait":
                seg.env_wait_open = ts
            seg.state = to_state

        elif etype == SEGMENT_END:
            seg = segs.get(ev["seg_id"])
            if seg is None:
                rep.add_error("E07", f"segment_end 引用未开启的段 {ev['seg_id']}", line)
                continue
            if seg.closed:
                rep.add_error("E08", f"段 {ev['seg_id']} 重复结束", line)
                continue
            state, from_state = ev["state"], ev["from_state"]
            if state not in SEGMENT_STATES or state not in {"finished", "aborted"}:
                rep.add_error("E06", f"非法终态 {state!r}", line)
                continue
            if from_state != seg.state:
                rep.add_error(
                    "E08", f"from_state={from_state!r} 与段当前状态 {seg.state!r} 不符", line
                )
                continue
            if state == "finished" and seg.state != "running":
                rep.add_error("E08", f"只有 running 可转入 finished，当前 {seg.state!r}", line)
            if ev.get("finish_mode") is not None and ev["finish_mode"] not in FINISH_MODES:
                rep.add_error("E06", f"非法 finish_mode: {ev['finish_mode']!r}", line)
            if state == "aborted" and not ev.get("reason"):
                rep.add_error("E15", "aborted 段缺少 reason", line)
            if ev["t_end"] < seg.t_start:
                rep.add_error(
                    "E11", f"segment_end t_end={ev['t_end']} 早于段开始 {seg.t_start}", line
                )
            if ev["birth_version"] != seg.birth_version:
                rep.add_error(
                    "E10",
                    f"birth_version 回显不一致：{ev['birth_version']} ≠ {seg.birth_version}",
                    line,
                )
            if ev["end_version"] != current_version:
                rep.add_error(
                    "E10", f"end_version={ev['end_version']} ≠ 当时账本版本 {current_version}", line
                )
            if seg.env_wait_open is not None:
                seg.env_wait_intervals.append((seg.env_wait_open, ev["t_end"]))
                seg.env_wait_open = None
            seg.closed = True
            seg.t_end = ev["t_end"]
            _check_lp_coverage(seg, ev["n_gen_tokens"], state, line, rep)

        elif etype == TOKEN_LOGPROB:
            seg = segs.get(ev["seg_id"])
            if seg is None:
                rep.add_error("E07", f"logprob 引用未开启的段 {ev['seg_id']}", line)
                continue
            if seg.closed:
                rep.add_error("E08", f"段 {ev['seg_id']} 已终态，仍有 logprob", line)
                continue
            version, start_idx, n, lp = ev["version"], ev["start_idx"], ev["n"], ev["lp"]
            if version > current_version:
                rep.add_error(
                    "E09", f"logprob version={version} 超前于当前账本版本 {current_version}", line
                )
            if version < seg.birth_version:
                rep.add_error(
                    "E14", f"logprob version={version} < 段 birth_version={seg.birth_version}", line
                )
            if start_idx < 0 or n < 0:
                rep.add_error("E05", f"start_idx/n 为负: {start_idx}/{n}", line)
            if n != len(lp):
                rep.add_error("E13", f"n={n} 与 len(lp)={len(lp)} 不符", line)
            for opt in ("tok", "entropy"):
                if opt in ev and len(ev[opt]) != n:
                    rep.add_error("E13", f"{opt} 长度 {len(ev[opt])} ≠ n={n}", line)
            for a, b in seg.lp_chunks:
                if start_idx < b and a < start_idx + max(n, 1) and n > 0:
                    rep.add_error(
                        "E13", f"logprob 块重叠：[{start_idx},{start_idx + n}) 撞上 [{a},{b})", line
                    )
                    break
            if n > 0:
                seg.lp_chunks.append((start_idx, start_idx + n))
                seg.lp_max = max(seg.lp_max, start_idx + n)

    # ---- 文件级收尾 ----
    if first:
        rep.add_error("E01", "空文件：没有 run_start")
        return
    if not saw_run_end:
        rep.add_warning("W01", "文件无 run_end 结尾（可能截断）")
    for seg in segs.values():
        if seg.birth_version > final_version:
            rep.add_error(
                "E09", f"birth_version={seg.birth_version} 超出 run 最终账本版本 {final_version}"
            )
    if not segs:
        rep.add_warning("W07", "整个 run 没有任何 segment")
    elif n_syncs == 0:
        rep.add_warning("W07", "run 有 segment 产出但没有任何 weight_sync")


def _check_required(ev: dict, etype: str, line: int, rep: ValidationReport) -> bool:
    """按字段表校验必填字段；返回 True 表示有缺失/类型错误（调用方跳过后续检查）。"""
    bad = False
    for name, kind in _REQUIRED[etype].items():
        if name not in ev:
            rep.add_error("E05", f"{etype} 缺字段 {name!r}", line)
            bad = True
        elif kind == "int" and not _is_int(ev[name]):
            rep.add_error("E05", f"{etype}.{name} 应为 int，实为 {type(ev[name]).__name__}", line)
            bad = True
        elif kind == "str" and not isinstance(ev[name], str):
            rep.add_error("E05", f"{etype}.{name} 应为 str，实为 {type(ev[name]).__name__}", line)
            bad = True
        elif kind == "numlist" and not (
            isinstance(ev[name], list) and all(_is_num(x) for x in ev[name])
        ):
            rep.add_error("E05", f"{etype}.{name} 应为数值数组", line)
            bad = True
    return bad


def _check_lp_coverage(
    seg: _Seg, n_gen_tokens: int, state: str, line: int, rep: ValidationReport
) -> None:
    if seg.lp_max > n_gen_tokens:
        rep.add_error("E13", f"logprob 覆盖到 {seg.lp_max}，超出 n_gen_tokens={n_gen_tokens}", line)
    if state != "finished":
        return
    merged: list[list[int]] = []
    for a, b in sorted(seg.lp_chunks):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    pos = 0
    for a, b in merged:
        if a > pos:
            rep.add_warning("W03", f"finished 段 logprob 覆盖有洞：[{pos},{a}) 缺失", line)
            return
        pos = b
    if pos < n_gen_tokens:
        rep.add_warning(
            "W03", f"finished 段 logprob 只覆盖到 {pos}，n_gen_tokens={n_gen_tokens}", line
        )
