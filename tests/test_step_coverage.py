"""§8 v0.1.2 增量：validator step 覆盖率断言（W10 / report.covered_steps）+ meta 约定键。"""

import json

from rheotrace import covered_steps, validate
from rheotrace.cli import main, parse_step_spec


def make_trace(tmp_path, trainer_steps, resume_from_step=None):
    """构造带 weight_sync.trainer_step 的最小合法 trace。"""
    events = [
        {
            "type": "run_start",
            "ts": 1000,
            "run_id": "r-cov",
            "format": "rheotrace-jsonl",
            "schema_version": 0,
            "initial_version": 0,
            "engine": "test",
            "model": "test",
            "clock": "wall_ns_epoch",
            "meta": (
                {"resume_from_step": resume_from_step} if resume_from_step is not None else {}
            ),
        }
    ]
    t = 1000
    for i, step in enumerate(trainer_steps, start=1):
        t += 100
        events.append(
            {
                "type": "weight_sync",
                "ts": t,
                "run_id": "r-cov",
                "version": i,
                "t_start": t - 10,
                "t_end": t,
                "mode": "full",
                "trainer_step": step,
            }
        )
    events.append({"type": "run_end", "ts": t + 1, "run_id": "r-cov"})
    path = tmp_path / "cov.rheotrace.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    return path


def test_covered_steps_report_full(tmp_path):
    path = make_trace(tmp_path, trainer_steps=range(1, 41))
    rep = validate(path, strict=False)
    assert rep.covered_steps == list(range(1, 41))


def test_covered_steps_excludes_zero(tmp_path):
    """0 = 热身同步，不算训练步（与 merge 头部注记同口径）。"""
    path = make_trace(tmp_path, trainer_steps=[0, 1, 2, 3])
    rep = validate(path, strict=False)
    assert rep.covered_steps == [1, 2, 3]
    assert covered_steps([{"type": "weight_sync", "trainer_step": 0}]) == []


def test_w10_fires_on_gap(tmp_path):
    """TASK-A2 G1 场景：崩溃丢步 → 缺失步 WARNING，不拒绝文件。"""
    path = make_trace(tmp_path, trainer_steps=[30, 31, 40])
    rep = validate(path, strict=False, expected_steps=range(1, 41))
    assert rep.ok  # 覆盖缺口是 WARNING 级，文件本身合法
    w10 = [w for w in rep.warnings if w.rule == "W10"]
    assert len(w10) == 1
    assert "37" in w10[0].message  # 缺失步清单里点名具体步号


def test_w10_silent_when_full_coverage(tmp_path):
    path = make_trace(tmp_path, trainer_steps=range(1, 11))
    rep = validate(path, strict=False, expected_steps=range(1, 11))
    assert not [w for w in rep.warnings if w.rule == "W10"]


def test_w10_resume_from_step_deducts_expected(tmp_path):
    """续跑 run：meta.resume_from_step=29 时，预期应扣除 ≤29 的步（调用方扣，spec §4.1）。"""
    path = make_trace(tmp_path, trainer_steps=range(30, 41), resume_from_step=29)
    rep = validate(path, strict=False, expected_steps=range(30, 41))
    assert not [w for w in rep.warnings if w.rule == "W10"]
    # 但若按完整 1-40 对账，30 之前全部报缺——G1 的"不静默"正是要这个
    rep2 = validate(path, strict=False, expected_steps=range(1, 41))
    assert [w for w in rep2.warnings if w.rule == "W10"]


def test_trainer_step_type_checked(tmp_path):
    path = make_trace(tmp_path, trainer_steps=[1, 2])
    # 手工注入非 int trainer_step（bool 是 int 子类，也须拒）；须插在 run_end 之前
    lines = path.read_text(encoding="utf-8").splitlines()
    ev = json.loads(lines[1])
    lines.insert(3, json.dumps({**ev, "trainer_step": True}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rep = validate(path, strict=False)
    assert any(e.rule == "E05" and "trainer_step" in e.message for e in rep.errors)
    assert rep.covered_steps == [1, 2]


def test_resume_from_step_is_just_meta(tmp_path):
    """meta 是自由对象：类型不合法也不拦（§4.0：validator 忽略额外字段）。"""
    path = make_trace(tmp_path, trainer_steps=[1])
    lines = path.read_text(encoding="utf-8").splitlines()
    ev = json.loads(lines[0])
    ev["meta"]["resume_from_step"] = "29"
    lines[0] = json.dumps(ev)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert validate(path).ok


def test_cli_expected_steps_flag(tmp_path, capsys):
    path = make_trace(tmp_path, trainer_steps=[1, 2, 3])
    rc = main(["validate", str(path), "--lenient", "--expected-steps", "1-5"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "W10" in out
    assert "covered_steps: 1-3 (3 步)" in out


def test_parse_step_spec():
    assert parse_step_spec("1-40") == set(range(1, 41))
    assert parse_step_spec("1,2,5-7") == {1, 2, 5, 6, 7}
    assert parse_step_spec("3") == {3}
