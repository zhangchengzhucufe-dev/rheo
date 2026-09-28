"""仓库自检：bench/traces/synthetic/ 下的入库样例必须全部通过 validator。"""

from pathlib import Path

import pytest

import rheotrace

SAMPLES = Path(__file__).resolve().parent.parent / "bench" / "traces" / "synthetic"


def test_committed_samples_validate():
    files = sorted(SAMPLES.glob("*.jsonl"))
    if not files:
        pytest.skip("bench/traces/synthetic/ 下暂无样例")
    for f in files:
        rep = rheotrace.validate(f, strict=False)
        assert rep.errors == [], f"{f.name}: {[str(e) for e in rep.errors]}"
        assert rep.warnings == [], f"{f.name}: {[str(w) for w in rep.warnings]}"


def test_committed_samples_roundtrip(tmp_path):
    files = sorted(SAMPLES.glob("*.jsonl"))
    if not files:
        pytest.skip("bench/traces/synthetic/ 下暂无样例")
    for f in files:
        events = rheotrace.read(f)
        out = tmp_path / f.name
        rheotrace.write(out, events)
        assert rheotrace.read(out) == events
        # 头部自描述：preset 与参数足以复现
        assert events[0]["type"] == "run_start"
        assert "preset" in events[0]["meta"]
