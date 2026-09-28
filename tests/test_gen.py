"""验收：生成器确定性、参数生效（轨迹数 / 双峰长度 / 延迟注入）。"""

import pytest

import rheotrace
from rheotrace.gen import PRESETS, GenParams


def test_deterministic_same_seed():
    a = rheotrace.generate(preset="agent", seed=3)
    b = rheotrace.generate(preset="agent", seed=3)
    assert a == b


def test_different_seeds_differ():
    a = rheotrace.generate(preset="grpo", seed=1)
    b = rheotrace.generate(preset="grpo", seed=2)
    assert a != b


def test_segment_count_matches_params():
    ev = rheotrace.generate(preset="grpo", n_steps=2, groups_per_step=5, group_size=3)
    n_start = sum(1 for e in ev if e["type"] == "segment_start")
    n_end = sum(1 for e in ev if e["type"] == "segment_end")
    assert n_start == n_end == 2 * 5 * 3


def test_bimodal_length_distribution():
    """双峰：短峰与长峰都应有大样本落点（固定 seed 下）。"""
    ev = rheotrace.generate(preset="bimodal", n_steps=1, groups_per_step=25, group_size=4)
    lens = [e["n_gen_tokens"] for e in ev if e["type"] == "segment_end"]
    assert len(lens) == 100
    short = sum(1 for x in lens if x < 200)
    long = sum(1 for x in lens if x > 500)
    assert short > 30 and long > 10, f"双峰形态不对：short={short}, long={long}"


def test_env_wait_injection():
    on = rheotrace.generate(preset="agent", seed=3)
    assert any(e["type"] == "phase_span" and e["phase"] == "env_wait" for e in on)
    off = rheotrace.generate(preset="grpo", seed=3)  # grpo 预置 env_wait_prob=0
    assert not any(e["type"] == "phase_span" and e["phase"] == "env_wait" for e in off)


def test_weight_sync_and_partial_rollout_present():
    """grpo/bimodal 预置应产出 weight_sync，且存在跨版本续跑（partial rollout 语义）。"""
    ev = rheotrace.generate(preset="bimodal", seed=7)
    syncs = [e for e in ev if e["type"] == "weight_sync"]
    assert len(syncs) == 3  # n_steps=4
    versions = [e["version"] for e in syncs]
    assert versions == sorted(versions) and len(set(versions)) == len(versions)
    # 有段的生命周期覆盖过 weight_sync 边界（存在 paused 事件）
    assert any(e["type"] == "segment_state" and e["to_state"] == "paused" for e in ev)
    # 存在同段内多版本 logprob（出生版本 != 最后块版本）
    lp_versions: dict[str, set[int]] = {}
    for e in ev:
        if e["type"] == "token_logprob":
            lp_versions.setdefault(e["seg_id"], set()).add(e["version"])
    assert any(len(v) > 1 for v in lp_versions.values()), "应有段产出跨版本 token"


def test_version_never_decreases():
    ev = rheotrace.generate(preset="agent", seed=9)
    cur = 0
    for e in ev:
        if e["type"] == "weight_sync":
            assert e["version"] > cur
            cur = e["version"]


def test_pause_at_sync_off_produces_straddle_warnings():
    """关闭安全点暂停 → decode span 横跨同步窗口，validator 应发 W04。"""
    ev = rheotrace.generate(preset="grpo", seed=1, pause_at_sync=False)
    rep = rheotrace.validate(ev, strict=False)
    assert rep.ok
    assert any(w.rule == "W04" for w in rep.warnings)


def test_gen_params_preset_override_isolation():
    base = rheotrace.generate(preset="grpo", seed=0)
    tweaked = rheotrace.generate(preset="grpo", seed=0, n_workers=1)
    assert base != tweaked
    # 预置对象本身不被 overrides 污染（frozen dataclass，replace 返回新对象）
    assert GenParams().n_workers == 4 and PRESETS["grpo"].n_workers == 4


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        rheotrace.generate(preset="nope")
