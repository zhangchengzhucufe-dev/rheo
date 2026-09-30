"""Merge + validate pipeline test for the M0 trace instrumentation."""

import json
import subprocess
import sys
from pathlib import Path
from time import time_ns

import pytest

from rheotrace import validate

MERGE = Path(__file__).resolve().parents[1] / "bench" / "m0" / "merge_trace.py"


def _write_spill(spill_dir: Path) -> None:
    t0 = time_ns()

    def spill(pid: int, ts: int, **ev: dict) -> None:
        ev.update({"type": ev.get("type"), "ts": ts, "fmt": "x", "pid": pid})
        with open(spill_dir / f"spill-{pid}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(ev) + "\n")

    # two syncs sharing trainer_step=1 (warmup + step sync) must get distinct versions
    spill(
        1,
        t0,
        type="weight_sync",
        version=1,
        t_start=t0,
        t_end=t0 + 1000,
        mode="full",
        trainer_step=1,
    )
    spill(1, t0 + 2000, type="phase_span", phase="schedule", t_start=t0 + 1000, t_end=t0 + 2000)
    spill(
        2,
        t0 + 2500,
        type="segment_start",
        seg_id="s-a",
        group_id="g-1",
        # 真实钩子里 segment_start 是生成结束后补写的：ts 晚于 decode span 的
        # t_end；merge 归一化排序后 segment_start 仍须排在 decode span 之前
        t_start=t0 + 1500,
        n_prompt_tokens=10,
    )
    spill(
        2,
        t0 + 2500,
        type="phase_span",
        seg_id="s-a",
        phase="decode",
        t_start=t0 + 1500,
        t_end=t0 + 2500,
        n_tokens=4,
    )
    spill(
        2,
        t0 + 2501,
        type="token_logprob",
        seg_id="s-a",
        start_idx=0,
        n=4,
        lp=[-0.1, -0.2, -0.3, -0.4],
    )
    spill(
        2,
        t0 + 2600,
        type="segment_end",
        seg_id="s-a",
        state="finished",
        from_state="running",
        t_end=t0 + 2600,
        n_gen_tokens=4,
    )
    spill(
        3,
        t0 + 3000,
        type="weight_sync",
        version=1,
        t_start=t0 + 2500,
        t_end=t0 + 3000,
        mode="full",
        trainer_step=1,
    )
    spill(
        3,
        t0 + 3100,
        type="segment_start",
        seg_id="s-b",
        group_id="g-2",
        t_start=t0 + 3100,
        n_prompt_tokens=12,
    )
    spill(
        3,
        t0 + 3300,
        type="phase_span",
        seg_id="s-b",
        phase="decode",
        t_start=t0 + 3100,
        t_end=t0 + 3300,
        n_tokens=2,
    )
    spill(
        3,
        t0 + 3400,
        type="segment_end",
        seg_id="s-b",
        state="finished",
        from_state="running",
        t_end=t0 + 3400,
        n_gen_tokens=2,
    )


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_merge_produces_valid_trace(tmp_path: Path) -> None:
    spill_dir = tmp_path / "spill"
    spill_dir.mkdir()
    _write_spill(spill_dir)
    out = tmp_path / "trace.jsonl"

    r = subprocess.run(
        [sys.executable, str(MERGE), "--spill-dir", str(spill_dir), "--out", str(out)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "ok=True errors=0" in r.stdout

    events = _read(out)
    assert events[0]["type"] == "run_start" and events[-1]["type"] == "run_end"
    steps = [e["ts"] for e in events]
    assert steps == sorted(steps), "events must be ts-ordered (E04)"


def test_merge_derives_versions(tmp_path: Path) -> None:
    spill_dir = tmp_path / "spill"
    spill_dir.mkdir()
    _write_spill(spill_dir)
    out = tmp_path / "trace.jsonl"

    subprocess.run(
        [sys.executable, str(MERGE), "--spill-dir", str(spill_dir), "--out", str(out)],
        capture_output=True,
        text=True,
        check=True,
    )
    events = {e["type"]: e for e in _read(out)}
    syncs = [e for e in _read(out) if e["type"] == "weight_sync"]
    segs = {e["seg_id"]: e for e in _read(out) if e["type"] == "segment_start"}
    ends = {e["seg_id"]: e for e in _read(out) if e["type"] == "segment_end"}

    # warmup + step sync => versions 1,2 even though trainer_step duplicated
    assert [s["version"] for s in syncs] == [1, 2]
    assert segs["s-a"]["birth_version"] == 1 and ends["s-a"]["end_version"] == 1
    assert segs["s-b"]["birth_version"] == 2 and ends["s-b"]["end_version"] == 2
    lps = [e for e in _read(out) if e["type"] == "token_logprob"]
    assert lps[0]["version"] == 1
    assert events["run_end"]["summary"]["segments"] == 2


def test_validate_rejects_broken_trace(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "run_start",
                        "ts": 1,
                        "run_id": "r",
                        "format": "rheotrace-jsonl",
                        "schema_version": 0,
                        "initial_version": 0,
                        "engine": "e",
                        "model": "m",
                        "clock": "wall_ns_epoch",
                    }
                ),
                json.dumps(
                    {
                        "type": "weight_sync",
                        "ts": 2,
                        "run_id": "r",
                        "version": 5,
                        "t_start": 1,
                        "t_end": 2,
                        "mode": "full",
                    }
                ),
                json.dumps(
                    {
                        "type": "weight_sync",
                        "ts": 3,
                        "run_id": "r",
                        "version": 3,
                        "t_start": 2,
                        "t_end": 3,
                        "mode": "full",
                    }
                ),
            ]
        )
        + "\n"
    )
    report = validate(bad, strict=False)
    assert not report.ok
    assert any(e.rule == "E09" for e in report.errors), "version regression must be rejected"


if __name__ == "__main__":
    pytest.main([__file__])


def test_merge_derives_schedule_spans(tmp_path: Path) -> None:
    """schedule span = gap between one version's last segment end and the next
    version's earliest segment start (engine-level, no seg_id)."""
    spill_dir = tmp_path / "spill"
    spill_dir.mkdir()
    t0 = time_ns()

    def spill(pid: int, ts: int, **ev: dict) -> None:
        ev.update({"type": ev.get("type"), "ts": ts, "fmt": "x", "pid": pid})
        with open(spill_dir / f"spill-{pid}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(ev) + "\n")

    spill(
        1,
        t0 + 500,
        type="weight_sync",
        version=1,
        t_start=t0,
        t_end=t0 + 500,
        mode="full",
        trainer_step=1,
    )
    # v1 段：结束于 t0+1000
    spill(
        2,
        t0 + 600,
        type="segment_start",
        seg_id="s-a",
        group_id="g-1",
        t_start=t0 + 600,
        n_prompt_tokens=5,
    )
    spill(
        2,
        t0 + 1000,
        type="segment_end",
        seg_id="s-a",
        state="finished",
        from_state="running",
        t_end=t0 + 1000,
        n_gen_tokens=3,
    )
    # v2 同步 + v2 段：最早开始于 t0+1500
    spill(
        3,
        t0 + 1200,
        type="weight_sync",
        version=2,
        t_start=t0 + 1000,
        t_end=t0 + 1200,
        mode="full",
        trainer_step=2,
    )
    spill(
        2,
        t0 + 1500,
        type="segment_start",
        seg_id="s-b",
        group_id="g-2",
        t_start=t0 + 1500,
        n_prompt_tokens=5,
    )
    spill(
        2,
        t0 + 1800,
        type="segment_end",
        seg_id="s-b",
        state="finished",
        from_state="running",
        t_end=t0 + 1800,
        n_gen_tokens=2,
    )

    out = tmp_path / "trace.jsonl"
    subprocess.run(
        [sys.executable, str(MERGE), "--spill-dir", str(spill_dir), "--out", str(out)],
        capture_output=True,
        text=True,
        check=True,
    )
    events = [json.loads(line) for line in out.read_text().splitlines()]
    sched = [e for e in events if e["type"] == "phase_span" and e["phase"] == "schedule"]
    assert len(sched) == 1
    # gap = v1 段最晚结束(t0+1000) -> v2 段最早开始(t0+1500)
    assert sched[0]["t_start"] == t0 + 1000 and sched[0]["t_end"] == t0 + 1500
    assert "seg_id" not in sched[0], "schedule is engine-level (no seg_id)"
    r = validate(out, strict=False)
    assert r.ok


def test_merge_refuses_mixed_run_ids(tmp_path: Path) -> None:
    """TASK-A2 D14/G1: spill 目录混入两个 run_id 必须被拒绝。"""
    spill_dir = tmp_path / "spill"
    spill_dir.mkdir()
    t0 = time_ns()

    def spill(pid: int, ts: int, run_id: str, **ev: dict) -> None:
        ev.update({"type": ev.get("type"), "ts": ts, "run_id": run_id})
        with open(spill_dir / f"spill-{pid}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(ev) + "\n")

    spill(
        1,
        t0,
        "r-aaa",
        type="weight_sync",
        version=1,
        t_start=t0,
        t_end=t0 + 100,
        mode="full",
        trainer_step=1,
    )
    spill(
        2,
        t0 + 200,
        "r-bbb",
        type="weight_sync",
        version=1,
        t_start=t0 + 100,
        t_end=t0 + 200,
        mode="full",
        trainer_step=2,
    )

    out = tmp_path / "trace.jsonl"
    r = subprocess.run(
        [sys.executable, str(MERGE), "--spill-dir", str(spill_dir), "--out", str(out)],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    assert "REFUSED" in r.stderr + r.stdout


def test_merge_reports_covered_steps(tmp_path: Path) -> None:
    """TASK-A2 D14: covered_steps 注记 + 缺步 WARNING。"""
    spill_dir = tmp_path / "spill"
    spill_dir.mkdir()
    t0 = time_ns()

    def spill(pid: int, ts: int, **ev: dict) -> None:
        ev.update({"type": ev.get("type"), "ts": ts, "run_id": "r-x"})
        with open(spill_dir / f"spill-{pid}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(ev) + "\n")

    # 只有 step 1 和 step 3 的同步：step 2 缺口必须被报告
    spill(
        1,
        t0 + 100,
        type="weight_sync",
        version=1,
        t_start=t0,
        t_end=t0 + 100,
        mode="full",
        trainer_step=1,
    )
    spill(
        1,
        t0 + 300,
        type="weight_sync",
        version=2,
        t_start=t0 + 200,
        t_end=t0 + 300,
        mode="full",
        trainer_step=3,
    )

    out = tmp_path / "trace.jsonl"
    r = subprocess.run(
        [
            sys.executable,
            str(MERGE),
            "--spill-dir",
            str(spill_dir),
            "--out",
            str(out),
            "--expected-steps",
            "3",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "covered_steps: 1-3 (2 步)" in r.stdout
    assert "WARNING" in r.stdout and "缺 [2]" in r.stdout
    events = [json.loads(line) for line in out.read_text().splitlines()]
    run_start = next(e for e in events if e["type"] == "run_start")
    assert run_start["meta"]["covered_steps"] == [1, 3]
