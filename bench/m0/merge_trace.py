"""Merge per-process RheoTrace spill files into one validated trace.

verl spreads the rollout across several Ray worker processes (driver,
WorkerDict, AgentLoop workers); each instrumented site appends raw events to
$RHEO_TRACE_DIR/spill-<pid>.jsonl (wall-clock ns). The merger:

1. sorts all events globally by ts (same host clock),
2. rebuilds the version ledger in time order (verl may sync twice within one
   global step, so version = "how many new weights this run has seen", spec
   §5.2; the verl step is kept in trainer_step),
3. derives each segment's birth_version / end_version by ledger replay
   (spec §5.1.2: effective version = last weight_sync whose t_end <= t),
4. writes the final JSONL via rheotrace.write() and runs rheotrace.validate.

每次启动（含崩溃后续跑）各占一个 run_id 子目录：把同一次训练的所有启动目录
依次传入 --spill-dir（可多次），按时间序合成一条完整 trace；单个目录内混入
多个 run_id 会被拒绝（TASK-A2 D14）。

Usage:
    python bench/m0/merge_trace.py --spill-dir DIR [--spill-dir DIR2 ...] [--out FILE]
"""

import argparse
import glob
import json
import math
from pathlib import Path

from rheotrace import validate, write


def load_spill(spill_dir: Path) -> list[dict]:
    events: list[dict] = []
    bad = 0
    for path in sorted(glob.glob(str(spill_dir / "spill-*.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    # 进程被 kill -9 时可能留下半行；跳过并计数
                    bad += 1
                    continue
                if ev.get("type") == "trace_boot":
                    continue
                events.append(ev)
    if bad:
        print(f"note: skipped {bad} unparseable spill line(s)")
    return events


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--spill-dir",
        type=Path,
        action="append",
        required=True,
        help="spill 目录，可多次传入（同一次训练的每次启动各一个，按时间序合并）",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=here / ".." / "traces" / "m0-baseline.rheotrace.jsonl",
    )
    ap.add_argument("--model", default="Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument(
        "--expected-steps",
        type=int,
        default=None,
        help="计划训练步数（如 40）；与 weight_sync 覆盖步数不符时报 WARNING",
    )
    args = ap.parse_args()

    raw: list[dict] = []
    dir_run_ids: list[str] = []
    seen_dirs: set[str] = set()
    for d in args.spill_dir:
        # 重复目录去重：同一目录传两次会把事件翻倍（E07/E13 假错）
        key = str(d.resolve()) if d.exists() else str(d)
        if key in seen_dirs:
            print(f"note: 忽略重复的 spill 目录 {d}")
            continue
        seen_dirs.add(key)
        evs = load_spill(d)
        if not evs:
            raise SystemExit(f"no spill events under {d}")
        ids = {e.get("run_id") for e in evs}
        ids.discard(None)
        if len(ids) > 1:
            raise SystemExit(
                f"REFUSED: {d} 内混入多个 run_id ({sorted(ids)})——"
                "TASK-A2 D14 要求每次启动一个 run_id 目录；"
                "崩溃续跑的多个启动目录请依次多次传 --spill-dir"
            )
        rid = ids.pop() if ids else f"r-unnamed-{d.name}"
        dir_run_ids.append(rid)
        raw.extend(evs)
    if not raw:
        raise SystemExit("no spill events")
    run_id = dir_run_ids[0]

    # 覆盖步数注记：weight_sync 的 trainer_step 集合（热身同步记 0，不计入）
    covered = sorted(
        {int(e.get("trainer_step", 0)) for e in raw if e["type"] == "weight_sync"} - {0}
    )
    if args.expected_steps:
        # 实测 verl 的步末同步带自增后的编号（2 步跑同步出现在 2,3），
        # 所以按「数量 + 最大步 ≥ 预期」判定，不做 1..N 的精确集合比对
        if len(covered) < args.expected_steps or (covered and max(covered) < args.expected_steps):
            print(
                f"WARNING: weight_sync 只覆盖 {len(covered)} 步"
                f"（steps={covered}，预期 {args.expected_steps} 步）——存在缺口，曲线可能缺验证点"
            )
    lo = covered[0] if covered else 0
    hi = covered[-1] if covered else 0
    print(f"covered_steps: {lo}-{hi} ({len(covered)} 步)")

    # 先把区间事件的时间戳归一到 t_end（规格 §4.0），再排序——否则排序后被
    # 改写的 ts 会重新失序（E04）
    for ev in raw:
        if ev["type"] in ("weight_sync", "phase_span", "segment_end"):
            ev["ts"] = ev["t_end"]
    raw.sort(key=lambda e: e["ts"])

    # 1) rebuild ledger: weight_sync in t_end order, sequential versions
    syncs = sorted((e for e in raw if e["type"] == "weight_sync"), key=lambda e: e["t_end"])
    ledger: list[tuple[int, int]] = []
    for i, ev in enumerate(syncs, start=1):
        ledger.append((ev["t_end"], i))

    def effective_version(t: int) -> int:
        v = 0
        for t_end, ver in ledger:
            if t_end <= t:
                v = ver
            else:
                break
        return v

    # 2) fill derived fields
    for ev in raw:
        t = ev["type"]
        if t == "weight_sync":
            ev["version"] = effective_version(ev["t_end"]) or 1
            ev["ts"] = ev["t_end"]
        elif t == "segment_start":
            ev["birth_version"] = effective_version(ev["t_start"])
        elif t == "segment_end":
            ev["end_version"] = effective_version(ev["t_end"])
            ev.setdefault("finish_mode", "exact")
            ev["ts"] = ev["t_end"]
        elif t == "phase_span":
            ev["ts"] = ev["t_end"]
        elif t == "token_logprob":
            # resolved after birth_version pass below
            pass

    # token_logprob version = its segment's birth_version
    births = {e["seg_id"]: e["birth_version"] for e in raw if e["type"] == "segment_start"}
    for ev in raw:
        if ev["type"] == "token_logprob":
            ev["version"] = births.get(ev.get("seg_id"), 0)

    # segment_end needs birth_version echo: take from segment_start
    for ev in raw:
        if ev["type"] == "segment_end":
            ev["birth_version"] = births.get(ev.get("seg_id"), 0)

    # 2.5) 双保险：非有限浮点（-inf/NaN）会让 rheotrace.write 崩溃（allow_nan=False），
    # 钩子层已清洗，这里对历史/异常 spill 再钳一次
    for ev in raw:
        if ev["type"] == "token_logprob":
            ev["lp"] = [v if math.isfinite(v) else -1e30 for v in ev.get("lp", [])]

    # 2.6) 引擎级 schedule span：由段边界推导（批间空转 = 上一版段最晚结束 →
    # 下一版段最早开始）。manager 级钩子在真实运行中静默不触发，段事件天然
    # 携带边界，merge 推导更可靠且少一个 verl 内部补丁面
    derived_spans = []
    by_birth: dict[int, list[dict]] = {}
    for ev in raw:
        if ev["type"] == "segment_start":
            by_birth.setdefault(ev.get("birth_version", 0), []).append(ev)
        elif ev["type"] == "segment_end":
            by_birth.setdefault(effective_version(ev["t_end"]), []).append(ev)
    births_sorted = sorted(b for b in by_birth if b > 0)
    for prev_b, next_b in zip(births_sorted, births_sorted[1:], strict=False):
        prev_ends = [e["t_end"] for e in by_birth[prev_b] if e["type"] == "segment_end"]
        next_starts = [e["t_start"] for e in by_birth[next_b] if e["type"] == "segment_start"]
        if not prev_ends or not next_starts:
            continue
        gap_start, gap_end = max(prev_ends), min(next_starts)
        if gap_end >= gap_start:
            derived_spans.append(
                {
                    "type": "phase_span",
                    "phase": "schedule",
                    "ts": gap_end,
                    "t_start": gap_start,
                    "t_end": gap_end,
                }
            )
    raw.extend(derived_spans)
    raw.sort(key=lambda e: e["ts"])

    # 3) emit final trace
    run_start = {
        "type": "run_start",
        "ts": raw[0]["ts"],
        "run_id": run_id,
        "format": "rheotrace-jsonl",
        "schema_version": 0,
        "initial_version": 0,
        "engine": "verl-0.8.0+vllm-0.12.0",
        "model": args.model,
        "clock": "wall_ns_epoch",
        "n_workers": args.n_workers,
        "meta": {
            "spill_dirs": [str(d) for d in args.spill_dir],
            "instrumentation": "bench/m0/rheo_trace_hooks.py",
            "covered_steps": covered,
        },
    }
    out_events = [run_start]
    for ev in raw:
        clean = {k: v for k, v in ev.items() if k not in ("fmt", "pid")}
        # 统一覆盖为最终 run_id（spill 里的启动级 run_id 是合并前的簿记，
        # 保留在 launch_run_id 附加字段便于溯源；规格允许任意附加字段）
        clean["run_id"] = run_start["run_id"]
        if ev.get("run_id") and ev["run_id"] != run_start["run_id"]:
            clean["launch_run_id"] = ev["run_id"]
        out_events.append(clean)
    gen_tokens = sum(e.get("n_gen_tokens", 0) for e in raw if e["type"] == "segment_end")
    out_events.append(
        {
            "type": "run_end",
            "ts": max(e["ts"] for e in raw) + 1,
            "run_id": run_id,
            "summary": {
                "segments": sum(1 for e in raw if e["type"] == "segment_start"),
                "weight_syncs": len(syncs),
                "gen_tokens": gen_tokens,
            },
        }
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = write(args.out, out_events)
    print(f"wrote {n} events -> {args.out}")

    report = validate(args.out, strict=False)
    print(f"validate: ok={report.ok} errors={len(report.errors)} warnings={len(report.warnings)}")
    for err in report.errors:
        print("  E:", err)
    for warn in report.warnings:
        print("  W:", warn)
    if not report.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
