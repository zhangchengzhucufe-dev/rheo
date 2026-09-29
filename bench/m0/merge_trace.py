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

Usage:
    python bench/m0/merge_trace.py --spill-dir DIR [--out FILE]
"""

import argparse
import glob
import json
from pathlib import Path

from rheotrace import validate, write


def load_spill(spill_dir: Path) -> list[dict]:
    events: list[dict] = []
    for path in sorted(glob.glob(str(spill_dir / "spill-*.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                ev = json.loads(line)
                if ev.get("type") == "trace_boot":
                    continue
                events.append(ev)
    return events


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--spill-dir", type=Path, required=True)
    ap.add_argument(
        "--out",
        type=Path,
        default=here / ".." / "traces" / "m0-baseline.jsonl",
    )
    ap.add_argument("--model", default="Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n-workers", type=int, default=8)
    args = ap.parse_args()

    raw = load_spill(args.spill_dir)
    if not raw:
        raise SystemExit(f"no spill events under {args.spill_dir}")
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

    # 3) emit final trace
    run_start = {
        "type": "run_start",
        "ts": raw[0]["ts"],
        "run_id": "r-m0baseline",
        "format": "rheotrace-jsonl",
        "schema_version": 0,
        "initial_version": 0,
        "engine": "verl-0.8.0+vllm-0.12.0",
        "model": args.model,
        "clock": "wall_ns_epoch",
        "n_workers": args.n_workers,
        "meta": {
            "spill_dir": str(args.spill_dir),
            "instrumentation": "bench/m0/rheo_trace_hooks.py",
        },
    }
    out_events = [run_start]
    for ev in raw:
        clean = {k: v for k, v in ev.items() if k not in ("fmt", "pid")}
        clean.setdefault("run_id", run_start["run_id"])
        out_events.append(clean)
    gen_tokens = sum(e.get("n_gen_tokens", 0) for e in raw if e["type"] == "segment_end")
    out_events.append(
        {
            "type": "run_end",
            "ts": max(e["ts"] for e in raw) + 1,
            "run_id": "r-m0baseline",
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
