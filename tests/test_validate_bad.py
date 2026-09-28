"""验收：validator 拒绝坏文件——缺字段 / 乱序事件 / 版本回退各至少一例，外加状态机等规则。

构造方式：基于一份合法合成 trace 做单点 mutation，断言目标规则号出现在 errors 中。
"""

import pytest

import rheotrace
from rheotrace.core import ValidationError


def _good(preset="grpo", seed=0):
    return rheotrace.generate(preset=preset, seed=seed)


def _idx(events, etype, nth=0):
    return [i for i, e in enumerate(events) if e["type"] == etype][nth]


def _mutated(events, i, **changes):
    out = [dict(e) for e in events]
    out[i].update(changes)
    return out


def _removed(events, i):
    return [e for j, e in enumerate(events) if j != i]


def _expect_rules(events, *rules):
    with pytest.raises(ValidationError) as ei:
        rheotrace.validate(events)
    got = {x.rule for x in ei.value.report.errors}
    for r in rules:
        assert r in got, f"缺少规则 {r}，实际 {got}"


# ---- 验收三项：缺字段 / 乱序 / 版本回退 ----


def test_missing_field_rejected():
    ev = _good()
    del ev[_idx(ev, "weight_sync")]["mode"]
    _expect_rules(ev, "E05")


def test_missing_envelope_ts_rejected():
    ev = _good()
    del ev[_idx(ev, "phase_span")]["ts"]
    _expect_rules(ev, "E03")


def test_out_of_order_ts_rejected():
    ev = _good()
    i = _idx(ev, "segment_end")
    ev = _mutated(ev, i, ts=1)  # 比任何前序事件都早
    _expect_rules(ev, "E04")


def test_version_regression_rejected():
    ev = _good()
    i = _idx(ev, "weight_sync", nth=1)
    ev = _mutated(ev, i, version=0)  # 低于上一版本
    _expect_rules(ev, "E09")


def test_logprob_version_ahead_rejected():
    ev = _good()
    i = _idx(ev, "token_logprob")
    ev = _mutated(ev, i, version=99)
    _expect_rules(ev, "E09")


# ---- 状态机与引用完整性 ----


def test_illegal_state_transition_rejected():
    ev = _good("agent", seed=0)
    # env_wait → paused 非法（env_wait 只能回 running 或转 aborted）
    j = [
        k
        for k, e in enumerate(ev)
        if e["type"] == "segment_state" and e["from_state"] == "env_wait"
    ][0]
    ev = _mutated(ev, j, to_state="paused")
    _expect_rules(ev, "E08")


def test_event_after_terminal_rejected():
    ev = _good()
    lp = ev[_idx(ev, "token_logprob")]
    ghost = dict(lp)
    ghost["ts"] = ev[-1]["ts"] - 1  # run_end 之前、该段终态之后
    ev.insert(len(ev) - 1, ghost)
    _expect_rules(ev, "E08")


def test_unknown_segment_reference_rejected():
    ev = _good()
    i = [k for k, e in enumerate(ev) if e["type"] == "phase_span" and e.get("seg_id")][0]
    ev = _mutated(ev, i, seg_id="s-does-not-exist")
    _expect_rules(ev, "E07")


def test_duplicate_segment_start_rejected():
    ev = _good()
    i = _idx(ev, "segment_start")
    ev.insert(i + 1, dict(ev[i]))
    _expect_rules(ev, "E07")


# ---- 区间 / 数值一致性 ----


def test_span_end_before_start_rejected():
    ev = _good()
    i = _idx(ev, "phase_span")
    t_start = ev[i]["t_start"]
    ev = _mutated(ev, i, t_end=t_start - 1)
    _expect_rules(ev, "E11")


def test_weight_sync_window_inverted_rejected():
    ev = _good()
    i = _idx(ev, "weight_sync")
    ev = _mutated(ev, i, t_end=ev[i]["t_start"] - 1)
    _expect_rules(ev, "E11")


def test_end_version_mismatch_rejected():
    ev = _good()
    i = _idx(ev, "segment_end")
    ev = _mutated(ev, i, end_version=ev[i]["end_version"] + 5)
    _expect_rules(ev, "E10")


def test_birth_version_echo_mismatch_rejected():
    ev = _good()
    i = _idx(ev, "segment_end")
    ev = _mutated(ev, i, birth_version=ev[i]["birth_version"] + 1)
    _expect_rules(ev, "E10")


def test_logprob_n_mismatch_rejected():
    ev = _good()
    i = _idx(ev, "token_logprob")
    ev = _mutated(ev, i, n=ev[i]["n"] + 1)
    _expect_rules(ev, "E13")


def test_logprob_overlap_rejected():
    ev = _good()
    seg = ev[_idx(ev, "token_logprob")]["seg_id"]
    j = [k for k, e in enumerate(ev) if e["type"] == "token_logprob" and e["seg_id"] == seg]
    assert len(j) >= 1
    # 同段第二块回跳 1 个 token，与首块重叠；无第二块则复制首块制造重叠
    if len(j) >= 2:
        ev = _mutated(ev, j[1], start_idx=ev[j[1]]["start_idx"] - 1)
    else:
        ghost = dict(ev[j[0]])
        ghost["start_idx"] = ev[j[0]]["start_idx"] + ev[j[0]]["n"] - 1
        ev.insert(j[0] + 1, ghost)
    _expect_rules(ev, "E13")


def test_logprob_below_birth_version_rejected():
    ev = _good()
    i = _idx(ev, "token_logprob")
    ev = _mutated(ev, i, version=ev[i]["version"] - 1)
    _expect_rules(ev, "E14")


def test_abort_without_reason_rejected():
    ev = _good(preset="bimodal", seed=0)
    idxs = [k for k, e in enumerate(ev) if e["type"] == "segment_end" and e["state"] == "aborted"]
    assert idxs, "bimodal 预置应含 aborted 段"
    i = idxs[0]
    del ev[i]["reason"]
    _expect_rules(ev, "E15")


def test_event_after_run_end_rejected():
    ev = _good()
    ev.append(dict(ev[-1]))  # 第二个 run_end
    _expect_rules(ev, "E16")


# ---- 文件级 ----


def test_first_event_not_run_start_rejected(tmp_path):
    ev = _good()
    p = tmp_path / "x.jsonl"
    rheotrace.write(p, _removed(ev, _idx(ev, "run_start")))
    with pytest.raises(ValidationError) as ei:
        rheotrace.validate(p)
    assert any(x.rule == "E01" for x in ei.value.report.errors)


def test_missing_run_end_is_warning_not_error(tmp_path):
    ev = _good()
    p = tmp_path / "x.jsonl"
    rheotrace.write(p, _removed(ev, len(ev) - 1))
    rep = rheotrace.validate(p, strict=False)
    assert rep.ok and any(w.rule == "W01" for w in rep.warnings)


def test_truncated_tail_line_is_warning(tmp_path):
    ev = _good()
    p = tmp_path / "t.jsonl"
    rheotrace.write(p, ev)
    lines = p.read_text().splitlines(keepends=True)
    p.write_text("".join(lines[:-1]) + lines[-1][: len(lines[-1]) // 2])
    rep = rheotrace.validate(p, strict=False)
    assert rep.ok and any(w.rule == "W01" for w in rep.warnings)


def test_corrupt_middle_line_rejected(tmp_path):
    ev = _good()
    p = tmp_path / "t.jsonl"
    rheotrace.write(p, ev)
    lines = p.read_text().splitlines(keepends=True)
    lines[len(lines) // 2] = "{not json\n"
    p.write_text("".join(lines))
    rep = rheotrace.validate(p, strict=False)
    assert any(x.rule == "E02" for x in rep.errors)


def test_unknown_event_type_is_forward_compatible():
    ev = _good()
    ev.insert(
        len(ev) - 1,
        {"ts": ev[-1]["ts"] - 1, "type": "spec_decode_probe", "run_id": ev[0]["run_id"], "v": 1},
    )
    rep = rheotrace.validate(ev, strict=False)
    assert rep.ok and any(w.rule == "W06" for w in rep.warnings)


def test_wrong_type_field_rejected():
    ev = _good()
    i = _idx(ev, "segment_start")
    ev = _mutated(ev, i, n_prompt_tokens="many")
    _expect_rules(ev, "E05")
