"""验收：合成 trace 与手工构造的合法 trace 通过 validator（零 error 零 warning）。"""

import pytest

import rheotrace


@pytest.mark.parametrize("preset", ["grpo", "bimodal", "agent"])
@pytest.mark.parametrize("seed", [0, 7, 42])
def test_generated_traces_are_clean(preset, seed, tmp_path):
    events = rheotrace.generate(preset=preset, seed=seed)
    path = tmp_path / f"{preset}-{seed}.jsonl"
    rheotrace.write(path, events)
    rep = rheotrace.validate(path, strict=False)
    assert rep.errors == []
    assert rep.warnings == []


def test_validate_list_input():
    events = rheotrace.generate(preset="grpo", seed=5)
    assert rheotrace.validate(events, strict=False).ok


def test_validate_strict_returns_report_on_clean():
    events = rheotrace.generate(preset="grpo", seed=5)
    assert rheotrace.validate(events).ok


def test_minimal_hand_written_trace(tmp_path):
    """规格 §4.7 形态的最小合法 trace（无 logprob 也应通过，仅可能无警告）。"""
    events = [
        {
            "ts": 1,
            "type": "run_start",
            "run_id": "r",
            "format": "rheotrace-jsonl",
            "schema_version": 0,
            "initial_version": 0,
            "engine": "hand",
            "model": "m",
            "clock": "wall_ns_epoch",
        },
        {
            "ts": 2,
            "type": "segment_start",
            "run_id": "r",
            "seg_id": "s1",
            "group_id": "g1",
            "birth_version": 0,
            "t_start": 2,
            "n_prompt_tokens": 10,
        },
        {
            "ts": 3,
            "type": "phase_span",
            "run_id": "r",
            "seg_id": "s1",
            "phase": "prefill",
            "t_start": 2,
            "t_end": 3,
            "n_tokens": 10,
        },
        {
            "ts": 4,
            "type": "phase_span",
            "run_id": "r",
            "seg_id": "s1",
            "phase": "decode",
            "t_start": 3,
            "t_end": 4,
            "n_tokens": 1,
        },
        {
            "ts": 4,
            "type": "token_logprob",
            "run_id": "r",
            "seg_id": "s1",
            "version": 0,
            "start_idx": 0,
            "n": 1,
            "lp": [-0.5],
        },
        {
            "ts": 5,
            "type": "segment_end",
            "run_id": "r",
            "seg_id": "s1",
            "state": "finished",
            "from_state": "running",
            "t_end": 5,
            "n_gen_tokens": 1,
            "birth_version": 0,
            "end_version": 0,
        },
        {
            "ts": 6,
            "type": "weight_sync",
            "run_id": "r",
            "version": 1,
            "t_start": 5,
            "t_end": 6,
            "mode": "full",
            "trainer_step": 1,
        },
        {"ts": 7, "type": "run_end", "run_id": "r", "summary": {}},
    ]
    path = tmp_path / "hand.jsonl"
    rheotrace.write(path, events)
    rep = rheotrace.validate(path, strict=False)
    assert rep.ok and rep.warnings == []


def test_mid_training_attach_and_finish_modes():
    """规格 §4.1 initial_version（中途挂到已训练进程）与 §4.4 finish_mode 的合法形态。"""
    events = [
        {
            "ts": 1,
            "type": "run_start",
            "run_id": "r",
            "format": "rheotrace-jsonl",
            "schema_version": 0,
            "initial_version": 5,
            "engine": "hand",
            "model": "m",
            "clock": "wall_ns_epoch",
        },
        # shadow-finish：按出生版本精确跑完（PLAN M3），不占新版本
        {
            "ts": 2,
            "type": "segment_start",
            "run_id": "r",
            "seg_id": "s1",
            "group_id": "g1",
            "birth_version": 5,
            "t_start": 2,
            "n_prompt_tokens": 10,
        },
        {
            "ts": 3,
            "type": "phase_span",
            "run_id": "r",
            "seg_id": "s1",
            "phase": "decode",
            "t_start": 2,
            "t_end": 3,
            "n_tokens": 2,
        },
        {
            "ts": 3,
            "type": "token_logprob",
            "run_id": "r",
            "seg_id": "s1",
            "version": 5,
            "start_idx": 0,
            "n": 2,
            "lp": [-0.5, -1.5],
        },
        {
            "ts": 4,
            "type": "segment_end",
            "run_id": "r",
            "seg_id": "s1",
            "state": "finished",
            "from_state": "running",
            "t_end": 4,
            "n_gen_tokens": 2,
            "birth_version": 5,
            "end_version": 5,
            "finish_mode": "shadow",
        },
        # ε-stale 续跑：token 由当前版本产出，诚实降级标注（迭代 3）
        {
            "ts": 5,
            "type": "segment_start",
            "run_id": "r",
            "seg_id": "s2",
            "group_id": "g1",
            "birth_version": 5,
            "t_start": 5,
            "n_prompt_tokens": 8,
        },
        {
            "ts": 6,
            "type": "weight_sync",
            "run_id": "r",
            "version": 6,
            "t_start": 5,
            "t_end": 6,
            "mode": "delta",
            "trainer_step": 6,
        },
        {
            "ts": 7,
            "type": "phase_span",
            "run_id": "r",
            "seg_id": "s2",
            "phase": "decode",
            "t_start": 6,
            "t_end": 7,
            "n_tokens": 1,
        },
        {
            "ts": 7,
            "type": "token_logprob",
            "run_id": "r",
            "seg_id": "s2",
            "version": 6,
            "start_idx": 0,
            "n": 1,
            "lp": [-0.9],
        },
        {
            "ts": 8,
            "type": "segment_end",
            "run_id": "r",
            "seg_id": "s2",
            "state": "finished",
            "from_state": "running",
            "t_end": 8,
            "n_gen_tokens": 1,
            "birth_version": 5,
            "end_version": 6,
            "finish_mode": "stale",
        },
        {"ts": 9, "type": "run_end", "run_id": "r", "summary": {}},
    ]
    rep = rheotrace.validate(events, strict=False)
    assert rep.ok and rep.warnings == []
