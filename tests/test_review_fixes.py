"""复查补充：区间型事件 ts 约定（§4.0）的强制（E18/W09）与 TraceWriter 合规默认。"""

import time

import pytest

import rheotrace
from rheotrace.core import RheotraceError, ValidationError


def _good(preset="grpo", seed=0):
    return rheotrace.generate(preset=preset, seed=seed)


def _idx(events, etype):
    return next(i for i, e in enumerate(events) if e["type"] == etype)


def test_interval_ts_before_end_rejected():
    ev = _good()
    i = _idx(ev, "weight_sync")
    ev[i]["ts"] = ev[i]["t_end"] - 1
    with pytest.raises(ValidationError) as ei:
        rheotrace.validate(ev)
    assert any(x.rule == "E18" for x in ei.value.report.errors)


def test_interval_ts_after_end_warns_not_rejects():
    """迟写（ts > t_end）只警告：真实插桩的 flush 延迟不应炸掉分析流水线。

    选 ts 最大的区间事件只 +1ns：其后仅有相隔 1ms 的 run_end，不会引发 E04 乱序。
    """
    ev = _good()
    ends = [e for e in ev if e["type"] == "segment_end"]
    target = max(ends, key=lambda e: e["ts"])
    target["ts"] = target["t_end"] + 1
    rep = rheotrace.validate(ev, strict=False)
    assert rep.ok
    assert any(w.rule == "W09" for w in rep.warnings)


def test_writer_interval_events_take_t_end_as_ts(tmp_path):
    """TraceWriter 对区间型事件默认取 t_end 作 ts：插桩方无需手工传 ts 即合规。"""
    path = tmp_path / "w.jsonl"
    with rheotrace.TraceWriter(path, engine="e", model="m") as w:
        w.emit("phase_span", seg_id="s1", phase="decode", t_start=10, t_end=20)
    events = rheotrace.read(path)
    span = next(e for e in events if e["type"] == "phase_span")
    assert span["ts"] == 20  # 区间事件 → t_end，而不是 writer 构造时刻


def test_writer_point_events_take_now_as_ts(tmp_path):
    """点事件（segment_start）仍取当前时刻，且不早于其标记的时刻。

    t_start 回退 1 分钟：即使 wall clock 因 NTP 微调回退，断言也不会偶发失败。
    """
    path = tmp_path / "w.jsonl"
    t_start = time.time_ns() - 60_000_000_000
    with rheotrace.TraceWriter(path, engine="e", model="m") as w:
        ev = w.emit(
            "segment_start",
            seg_id="s1",
            group_id="g1",
            birth_version=0,
            t_start=t_start,
            n_prompt_tokens=8,
        )
    assert ev["ts"] >= t_start


def test_validate_rejects_single_event_dict():
    """误用守卫：传入单个事件 dict（而非序列）应立刻 TypeError，而不是按键迭代产生误导报告。"""
    ev = rheotrace.generate(preset="grpo", seed=0)
    with pytest.raises(TypeError, match="单个事件"):
        rheotrace.validate(ev[0])


def test_writer_emit_after_close_raises(tmp_path):
    w = rheotrace.TraceWriter(tmp_path / "w.jsonl", engine="e", model="m")
    w.close()
    with pytest.raises(RheotraceError):
        w.emit("phase_span", seg_id="s", phase="decode", t_start=0, t_end=1)
