"""trace 格式适配层：外部格式 → canon。

这是**唯一的**格式接缝。metrics / report 只认 canon（TASK-C 验收：
A 的真实 trace 到位后只许改这里，不许改指标定义）。

当前支持：
- rheo-canon-v0（C 自有 JSONL 规范格式，测试与联调用）
- rheotrace-jsonl（B 的 RheoTrace v0 冻结 spec，docs/rheotrace-spec-v0.md）：
  先过 rheotrace.validate 拒坏文件，再映射到 canon。

RheoTrace → canon 映射约定：
- traj := seg_id（spec：一条轨迹 v0 记为同一个 segment，一一对应）
- phase_span(prefill/decode) → exec；bv 按 weight_sync 账本在 t_start 的生效版本重放
- phase_span.n_tokens 是分析必填语义（spec 标可选）：缺了直接报错，不静默估
- segment_state 的 env_wait 状态区间 → env_wait；segment_end → traj_end
- weight_sync → sync 区间 + ver_bump（新版本自 t_end 生效，spec §5.1）
- abort 段的 decode token 全部 committed=False；re-prefill span 照实计入 prefill
- engine 级 schedule span 不映射（canon 的 P3 由"非 P1/P2 的 pause"推导）；
  run 首尾的 schedule/warmup 因此不计入 T_wall——这是有意的（不算预热/收尾空转）
- spec 无 batch 概念：segment_start 若带 batch_id（可选增量，见 issues.md）则用之，
  否则按轨迹时间重叠聚类出伪批（analyze 会打 W-NO-BATCH 警告）
"""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from dataclasses import replace
from pathlib import Path

from . import canon
from .canon import Trace, TraceError

__all__ = ["read_trace", "TraceError"]

_RHEOTRACE_FORMAT = "rheotrace-jsonl"


def _sniff_format(path: Path) -> str | None:
    try:
        with path.open("rb") as f:
            head = f.readline()
        first = json.loads(head.decode("utf-8"))
        return first.get("format") if isinstance(first, dict) else None
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None


def read_trace(path: str | Path) -> Trace:
    """读入 trace 文件并转为 canon。格式按 header 的 format 字段分发。

    `.gz` 压缩文件不走字节嗅探（gzip 二进制猜格式必错，B↔C 互校结论）：
    直接路由给 rheotrace reader 透明解压；canon v0 不支持压缩。
    """
    path = Path(path)
    if not path.is_file():
        raise TraceError(f"trace 文件不存在：{path}")
    if path.name.endswith(".gz"):
        return _read_rheotrace(path)
    fmt = _sniff_format(path)
    if fmt == canon.CANON_FORMAT:
        return canon.read_canon(path)
    if fmt == _RHEOTRACE_FORMAT:
        return _read_rheotrace(path)
    raise TraceError(
        f"不认识的 trace 格式（format={fmt!r}）：{path}。"
        f"支持：{canon.CANON_FORMAT}、{_RHEOTRACE_FORMAT}（及 .gz 压缩）"
    )


# ---------------------------------------------------------------------------
# RheoTrace v0 → canon


def _read_rheotrace(path: Path) -> Trace:
    try:
        import rheotrace
    except ImportError as e:  # 同仓分发，正常都在
        raise TraceError("rheotrace 包不可导入，无法读取 rheotrace-jsonl 格式") from e
    report = rheotrace.validate(path, strict=False)
    if not report.ok:
        head = ""
        if getattr(report, "errors", None):
            head = "，前几条：" + "; ".join(
                f"{getattr(err, 'rule', '?')}@L{getattr(err, 'line', '?')}"
                for err in report.errors[:3]
            )
        raise TraceError(
            f"RheoTrace 校验失败（{len(report.errors)} 个 error{head}）；"
            "可先跑 python -m rheotrace validate 看完整报告"
        )
    return _rheotrace_to_canon(rheotrace.read(path), path)


def _rheotrace_to_canon(events: list[dict], path: Path) -> Trace:
    run_start = events[0]
    if run_start.get("type") != "run_start":
        raise TraceError("rheotrace：第一个事件不是 run_start")
    initial_ver = int(run_start.get("initial_version", 0))

    # 预扫描：段终态、批标注、版本账本
    seg_aborted: dict[str, bool] = {}
    seg_batch: dict[str, str | None] = {}
    sync_ts: list[int] = []
    sync_ver: list[int] = []
    for e in events:
        typ = e.get("type")
        if typ == "segment_end":
            seg_aborted[e["seg_id"]] = e["state"] == "aborted"
        elif typ == "segment_start":
            bid = e.get("batch_id") or (e.get("meta") or {}).get("batch_id")
            seg_batch[e["seg_id"]] = str(bid) if bid is not None else None
        elif typ == "weight_sync":
            sync_ts.append(int(e["t_end"]))
            sync_ver.append(int(e["version"]))

    def ver_at(t: int) -> int:
        i = bisect_right(sync_ts, t) - 1
        return sync_ver[i] if i >= 0 else initial_ver

    has_batch = any(v is not None for v in seg_batch.values())
    if not has_batch:
        seg_batch = _cluster_batches(events)

    mapped: list[dict] = [
        {
            "ev": "header",
            "format": canon.CANON_FORMAT,
            "clock": str(run_start.get("clock", "wall_ns_epoch")),
            "model": run_start.get("model"),
            "world_size": int(run_start.get("n_workers", 1) or 1),
        }
    ]
    if initial_ver > 0:
        mapped.append({"ev": "ver_bump", "t": int(run_start["ts"]), "ver": initial_ver})

    env_open: dict[str, int] = {}
    for e in events:
        typ = e.get("type")
        ts = int(e["ts"])
        if typ == "weight_sync":
            mapped.append({"ev": "sync", "t0": int(e["t_start"]), "t1": int(e["t_end"]),
                           "ver": int(e["version"])})
            mapped.append({"ev": "ver_bump", "t": int(e["t_end"]), "ver": int(e["version"])})
        elif typ == "phase_span":
            phase = e.get("phase")
            seg = e.get("seg_id")
            if phase not in ("prefill", "decode") or seg is None:
                continue  # engine 级 schedule（及其他）不入 canon
            n_tokens = e.get("n_tokens")
            if n_tokens is None:
                raise TraceError(
                    f"phase_span(seg={seg}, phase={phase}) 缺 n_tokens："
                    "分析必需该字段（spec 标可选），请插桩端必填——见 docs/issues.md"
                )
            ev: dict = {
                "ev": "exec",
                "traj": seg,
                "batch": seg_batch.get(seg, "cluster-000"),
                "phase": phase,
                "t0": int(e["t_start"]),
                "t1": int(e["t_end"]),
                "bv": ver_at(int(e["t_start"])),
            }
            aborted = seg_aborted.get(seg, False)
            if phase == "prefill":
                ev["n_prompt"] = int(n_tokens)
            else:
                ev["n_gen"] = int(n_tokens)
                ev["committed"] = not aborted
                ev["aborted"] = aborted
            mapped.append(ev)
        elif typ == "segment_state":
            seg = e["seg_id"]
            if e.get("to_state") == "env_wait":
                env_open[seg] = ts
            elif e.get("from_state") == "env_wait":
                opened = env_open.pop(seg, None)
                if opened is not None:
                    mapped.append({"ev": "env_wait", "traj": seg, "t0": opened, "t1": ts})
        elif typ == "segment_end":
            seg = e["seg_id"]
            opened = env_open.pop(seg, None)
            if opened is not None:
                mapped.append({"ev": "env_wait", "traj": seg, "t0": opened, "t1": int(e["t_end"])})
            mapped.append({
                "ev": "traj_end", "traj": seg, "t": int(e["t_end"]), "status": str(e["state"]),
            })
        # run_start/run_end/token_logprob/未知类型：v0 指标不用，跳过

    # canon 要求事件按主时间非降序；同刻 ver_bump 必须先于 exec（bv 校验依赖）
    mapped[1:] = sorted(
        mapped[1:],
        key=lambda ev: (ev.get("t0", ev.get("t", 0)), 0 if ev["ev"] == "ver_bump" else 1),
    )
    lines = [json.dumps(ev, ensure_ascii=False) for ev in mapped]
    trace = canon.parse_events(lines)
    return replace(trace, path=path, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _cluster_batches(events: list[dict]) -> dict[str, str]:
    """无 batch 标注时，按轨迹包络时间重叠聚类出伪批（精确的区间图连通分量）。"""
    env: dict[str, list[int]] = {}
    for e in events:
        if e.get("type") == "phase_span" and e.get("phase") in ("prefill", "decode"):
            seg = e["seg_id"]
            lo, hi = env.get(seg, [int(e["t_start"]), int(e["t_start"])])
            env[seg] = [min(lo, int(e["t_start"])), max(hi, int(e["t_end"]))]
    out: dict[str, str] = {}
    batch_idx = -1
    cur_end = -1
    for seg, (lo, hi) in sorted(env.items(), key=lambda kv: kv[1][0]):
        if batch_idx < 0 or lo >= cur_end:
            batch_idx += 1
            cur_end = hi
        else:
            cur_end = max(cur_end, hi)
        out[seg] = f"cluster-{batch_idx:03d}"
    return out
