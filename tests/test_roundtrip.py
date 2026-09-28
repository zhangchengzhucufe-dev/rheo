"""验收：write → read round-trip 无损（含 gzip、浮点精确、TraceWriter 自动字段）。"""

import rheotrace

PRESETS = ("grpo", "bimodal", "agent")


def test_roundtrip_all_presets(tmp_path):
    for preset in PRESETS:
        events = rheotrace.generate(preset=preset, seed=7)
        path = tmp_path / f"{preset}.jsonl"
        assert rheotrace.write(path, events) == len(events)
        assert rheotrace.read(path) == events


def test_roundtrip_gzip(tmp_path):
    events = rheotrace.generate(preset="agent", seed=3)
    path = tmp_path / "trace.jsonl.gz"
    rheotrace.write(path, events)
    assert rheotrace.read(path) == events


def test_roundtrip_float_precision(tmp_path):
    """双精度经 JSON 文本往返必须逐位相等（格式承诺：无损）。"""
    events = [
        {
            "ts": 1727600000000000000,
            "type": "run_start",
            "run_id": "r-x",
            "format": rheotrace.FORMAT_NAME,
            "schema_version": 0,
            "initial_version": 0,
            "engine": "test",
            "model": "m",
            "clock": "wall_ns_epoch",
        },
        {
            "ts": 1727600000000000001,
            "type": "token_logprob",
            "run_id": "r-x",
            "seg_id": "s-1",
            "version": 0,
            "start_idx": 0,
            "n": 3,
            "lp": [-1.0 / 3.0, 2.0 / 7.0, -(0.1**20)],
        },
    ]
    path = tmp_path / "f.jsonl"
    rheotrace.write(path, events)
    assert rheotrace.read(path) == events


def test_roundtrip_unicode(tmp_path):
    events = rheotrace.generate(preset="grpo", seed=1)
    events[0]["meta"]["中文"] = "值 ✓"
    path = tmp_path / "u.jsonl"
    rheotrace.write(path, events)
    assert rheotrace.read(path) == events


def test_iread_matches_read(tmp_path):
    events = rheotrace.generate(preset="bimodal", seed=2)
    path = tmp_path / "t.jsonl"
    rheotrace.write(path, events)
    assert list(rheotrace.iread(path)) == events


def test_trace_writer_autofill(tmp_path):
    """插桩用 writer：自动补 ts/run_id、自动落 run_start/run_end。"""
    path = tmp_path / "w.jsonl"
    with rheotrace.TraceWriter(path, engine="test-engine", model="test-model") as w:
        ev = w.emit("weight_sync", version=1, t_start=1, t_end=2, mode="full")
        assert ev["run_id"] == w.run_id and isinstance(ev["ts"], int)
        w.emit_raw(
            {"type": "phase_span", "seg_id": "s-1", "phase": "decode", "t_start": 1, "t_end": 2}
        )
    events = rheotrace.read(path)
    types = [e["type"] for e in events]
    assert types[0] == "run_start" and types[-1] == "run_end"
    assert events[0]["engine"] == "test-engine"
    assert all(e["run_id"] == w.run_id for e in events)
    assert events[-1]["summary"]["events_by_type"]["weight_sync"] == 1


def test_trace_writer_gz_and_close_idempotent(tmp_path):
    path = tmp_path / "w.jsonl.gz"
    w = rheotrace.TraceWriter(path, engine="e", model="m")
    w.emit("phase_span", seg_id="s", phase="schedule", t_start=0, t_end=1)
    w.close()
    w.close()  # 幂等
    assert len(rheotrace.read(path)) == 3  # run_start + span + run_end
