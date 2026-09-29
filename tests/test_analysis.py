"""bench/analysis 指标与管线测试：口径对照 docs/metrics-v0.md 逐条验证。"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

from synth import (  # noqa: E402
    MODEL_D,
    MODEL_L,
    MODEL_P,
    SynthConfig,
    expected_flops,
    generate_events,
    write_trace,
)

from bench.analysis import canon  # noqa: E402
from bench.analysis.__main__ import main  # noqa: E402
from bench.analysis.adapters import read_trace  # noqa: E402
from bench.analysis.metrics import (  # noqa: E402
    BC_BIMODAL_THRESHOLD,
    _sarle_bc,
    analyze,
)


def hdr(**kw):
    base = {
        "ev": "header", "format": "rheo-canon-v0", "clock": "mono_ns",
        "model": "qwen2.5-1.5b", "P": MODEL_P, "L": MODEL_L, "d": MODEL_D,
        "world_size": 1,
    }
    base.update(kw)
    return base


def ex(traj, batch, phase, t0, t1, bv, n, **kw):
    ev = {
        "ev": "exec", "traj": traj, "batch": batch, "phase": phase,
        "t0": t0, "t1": t1, "bv": bv,
        "n_prompt" if phase == "prefill" else "n_gen": n,
    }
    ev.update(kw)
    return ev


def parse(events):
    return canon.parse_events([json.dumps(e) for e in events])


def approx(x, y, rel=1e-9):
    assert abs(x - y) <= rel * max(abs(x), abs(y), 1e-30), f"{x} != {y}"


# ---------------------------------------------------------------------------
# 端到端：生成 → 读取 → 分析，真值逐项对照


def test_roundtrip_clean(tmp_path):
    cfg = SynthConfig(n_batches=2, group_size=4, seed=3, len_mode="normal",
                      short_len=300, long_len=600, long_std=50)
    events, truth = generate_events(cfg)
    path = tmp_path / "roundtrip.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")

    trace = read_trace(path)
    a = analyze(trace, peak_tflops=10.0)

    approx(a.t_wall_s, truth["t_wall"] / 1e9)
    approx(a.t_active_s, truth["t_active"] / 1e9)
    assert a.gen_tokens_all == truth["gen_all"]
    approx(a.thr_e2e, truth["gen_all"] / (truth["t_wall"] / 1e9))
    approx(a.pause_totals["P1"], truth["p1"] / 1e9)
    approx(a.pause_totals["P2"], truth["p2"] / 1e9)
    approx(a.pause_totals["P3"], truth["p3"] / 1e9)
    approx(a.pause_totals["P4"], 0.0)
    # T_pause = T_wall - T_active（生成器语义保证）
    approx(sum(a.pause_totals.values()), a.t_wall_s - a.t_active_s)
    # FLOPs 与独立实现对照；恒等式 M2 = M1 × duty
    approx(a.flops_total, expected_flops(events), rel=1e-6)
    approx(a.m2_rollout, a.m1_engine * a.duty, rel=1e-12)
    assert a.waste == 0.0
    assert a.n_aborted == 0
    assert a.stale_token_share == 0.0
    # 长度分布逐条一致
    assert sorted(a.lengths_all_valid.tolist()) == truth["lengths"]


def test_global_env_wait_p2():
    cfg = SynthConfig(n_batches=2, group_size=4, seed=5, len_mode="normal",
                      short_len=200, long_len=400, long_std=30,
                      global_env_wait_batch=0, env_wait_dur_ns=80_000_000)
    events, truth = generate_events(cfg)
    a = analyze(parse(events), peak_tflops=10.0)
    approx(a.pause_totals["P2"], truth["p2"] / 1e9)
    approx(a.pause_totals["P1"], truth["p1"] / 1e9)
    approx(a.pause_totals["P3"], truth["p3"] / 1e9)
    approx(sum(a.pause_totals.values()), a.t_wall_s - a.t_active_s)
    assert a.env_wait_s and max(a.env_wait_s) > 0


def test_abort_waste_and_two_columns():
    cfg = SynthConfig(n_batches=1, group_size=4, seed=1, len_mode="normal",
                      short_len=300, long_len=500, long_std=40, abort_frac=0.5)
    events, truth = generate_events(cfg)
    a = analyze(parse(events), peak_tflops=10.0)
    assert a.n_aborted == truth["n_aborted"]
    if truth["gen_all"] > 0:
        approx(a.waste, 1 - truth["gen_committed"] / truth["gen_all"])
    if a.n_aborted:
        # "不含 abort" 列：abort 轨迹整条剔除，而非记 0
        assert a.lengths_committed_valid.size == a.lengths_all_valid.size - a.n_aborted
        assert a.lengths_committed_valid.min() > 0


# ---------------------------------------------------------------------------
# 手工 trace：停顿分类优先级、掉队、stale、双峰


def test_pause_priority_p1_over_p2_and_p3():
    events = [
        hdr(),
        # b0：两条轨迹，decode 中间全员 env 等待 [300,500)，sync [350,450) 插在中间
        ex("a", "b0", "prefill", 100, 200, 0, 100),
        ex("b", "b0", "prefill", 100, 200, 0, 100),
        ex("a", "b0", "decode", 200, 300, 0, 100),
        ex("b", "b0", "decode", 200, 300, 0, 100),
        {"ev": "env_wait", "traj": "a", "t0": 300, "t1": 500},
        {"ev": "env_wait", "traj": "b", "t0": 300, "t1": 500},
        {"ev": "sync", "t0": 350, "t1": 450, "ver": 1},
        {"ev": "ver_bump", "t": 450, "ver": 1},
        ex("a", "b0", "decode", 500, 600, 1, 100),
        ex("b", "b0", "decode", 500, 600, 1, 100),
        {"ev": "traj_end", "traj": "a", "t": 600, "status": "finished"},
        {"ev": "traj_end", "traj": "b", "t": 600, "status": "finished"},
        # b1：批间空档 [600,700) → P3
        ex("c", "b1", "prefill", 700, 800, 1, 100),
        ex("c", "b1", "decode", 800, 1000, 1, 200),
        {"ev": "traj_end", "traj": "c", "t": 1000, "status": "finished"},
    ]
    a = analyze(parse(events), peak_tflops=10.0)
    approx(a.t_wall_s, 900e-9)
    approx(a.pause_totals["P1"], 100e-9)  # sync 优先于 env_wait
    approx(a.pause_totals["P2"], 100e-9)  # [300,350) + [450,500)
    approx(a.pause_totals["P3"], 100e-9)  # [600,700) 无在途轨迹
    approx(a.pause_totals["P4"], 0.0)
    approx(sum(a.pause_totals.values()), 300e-9)


def test_straggler_s1_s2():
    events = [
        hdr(),
        ex("x0", "b0", "prefill", 0, 100, 0, 100),
        ex("x1", "b0", "prefill", 0, 100, 0, 100),
        ex("x2", "b0", "prefill", 0, 100, 0, 100),
        ex("x3", "b0", "prefill", 0, 100, 0, 100),
        ex("x0", "b0", "decode", 100, 1000, 0, 900),
        ex("x1", "b0", "decode", 100, 1000, 0, 900),
        ex("x2", "b0", "decode", 100, 1000, 0, 900),
        ex("x3", "b0", "decode", 100, 3000, 0, 2900),
    ]
    events += [{"ev": "traj_end", "traj": f"x{i}",
                "t": 3000 if i == 3 else 1000, "status": "finished"}
               for i in range(4)]
    a = analyze(parse(events), peak_tflops=10.0)
    # T_active = 执行区间并集 = [0,100)∪[100,3000) = 3000；掉队 = [1000,3000) = 2000
    approx(a.t_straggler_s, 2000e-9)
    approx(a.straggler_share, 2000 / 3000)
    assert len(a.batch_occs) == 1
    approx(a.batch_occs[0].tail_ratio, (3000 - 1000) / 1000)  # S2


def test_stale_token_share():
    events = [
        hdr(),
        ex("s", "b0", "prefill", 0, 100, 0, 100),
        ex("s", "b0", "decode", 100, 1100, 0, 1000),  # t=600 版本推进 → 后半 stale
        {"ev": "ver_bump", "t": 600, "ver": 1},
        ex("s", "b0", "decode", 1100, 1300, 1, 200),  # bv=1，新版本下生成 → 不 stale
        {"ev": "traj_end", "traj": "s", "t": 1300, "status": "finished"},
    ]
    a = analyze(parse(events), peak_tflops=10.0)
    approx(a.stale_token_share, 500 / 1200)


def test_sarle_bc():
    bimodal = np.array([1.0] * 50 + [10.0] * 10)
    assert _sarle_bc(bimodal) > BC_BIMODAL_THRESHOLD
    rng = np.random.default_rng(0)
    unimodal = rng.normal(0.0, 1.0, 200)
    assert _sarle_bc(unimodal) < 0.5
    assert np.isnan(_sarle_bc(np.array([1.0, 2.0, 3.0])))


# ---------------------------------------------------------------------------
# validator：坏文件必须拒


def _expect_err(events, frag):
    import pytest

    with pytest.raises(canon.TraceError, match=frag):
        parse(events)


def test_validator_rejects():
    import pytest

    _expect_err([ex("a", "b0", "prefill", 0, 1, 0, 5)], "header")
    _expect_err([hdr(), ex("a", "b0", "prefill", 500, 600, 0, 5),
                 ex("a", "b0", "decode", 100, 200, 0, 5)], "乱序")
    _expect_err([hdr(), {"ev": "ver_bump", "t": 10, "ver": 2},
                 {"ev": "ver_bump", "t": 20, "ver": 1}], "版本回退")
    _expect_err([hdr(), ex("a", "b0", "decode", 0, 10, 0, 5),
                 {"ev": "traj_end", "traj": "a", "t": 10, "status": "finished"},
                 ex("a", "b0", "decode", 20, 30, 0, 5)], "终止")
    _expect_err([hdr(), ex("a", "b0", "decode", 0, 10, 1, 5)], "当前版本")
    _expect_err([hdr(), ex("a", "b0", "decode", 10, 5, 0, 5)], "倒置")

    from bench.analysis.metrics import MetricsError

    with pytest.raises(MetricsError):
        analyze(parse([hdr()]))  # 空 trace：合法文件但无法分析


# ---------------------------------------------------------------------------
# CLI：一条命令出完整报告（双峰长尾负载）


def test_cli_bimodal_end_to_end(tmp_path):
    cfg = SynthConfig(n_batches=5, group_size=8, seed=7)
    trace_path, _ = write_trace(cfg, tmp_path / "bimodal.jsonl")
    out = tmp_path / "report"
    rc = main([str(trace_path), "--out", str(out)])
    assert rc == 0

    report = (out / "report.md").read_text(encoding="utf-8")
    for fig in ("pause_waterfall", "occupancy", "length_hist", "len_vs_dur"):
        assert (out / "figures" / f"{fig}.png").exists(), fig
        assert f"figures/{fig}.png" in report

    # 双峰长尾指标
    assert "有双峰迹象" in report
    events, truth = generate_events(cfg)
    a = analyze(read_trace(trace_path), peak_tflops=None)
    assert a.tail_token_share > 0.2
    assert a.lengths_all_valid.size == truth["n_trajs"]
    stats = a.lengths_all_valid
    p99, p50 = np.percentile(stats, 99), np.percentile(stats, 50)
    assert p99 / p50 > 3


# ---------------------------------------------------------------------------
# 口径修正回归：P2 精确扫描（非中点近似）、T3 prefill 并集


def test_p2_exact_sweep_partial_overlap():
    # pause [300,600)：A env [300,600)，B env [300,500)。
    # [300,500) 双双等待 → P2；[500,600) B 不在等待 → P3（中点近似会整体误判 P2）
    events = [
        hdr(),
        ex("a", "b0", "prefill", 100, 200, 0, 100),
        ex("b", "b0", "prefill", 100, 200, 0, 100),
        ex("a", "b0", "decode", 200, 300, 0, 100),
        ex("b", "b0", "decode", 200, 300, 0, 100),
        {"ev": "env_wait", "traj": "a", "t0": 300, "t1": 600},
        {"ev": "env_wait", "traj": "b", "t0": 300, "t1": 500},
        ex("a", "b0", "decode", 600, 700, 0, 100),
        ex("b", "b0", "decode", 600, 700, 0, 100),
        {"ev": "traj_end", "traj": "a", "t": 700, "status": "finished"},
        {"ev": "traj_end", "traj": "b", "t": 700, "status": "finished"},
    ]
    a = analyze(parse(events), peak_tflops=10.0)
    approx(a.pause_totals["P2"], 200e-9)
    approx(a.pause_totals["P3"], 100e-9)
    approx(a.pause_totals["P1"], 0.0)


def test_t3_prefill_union_not_sum():
    # 两条轨迹并发 prefill [0,100)：GPU 只花 100ns，T3 = 256 token / 100ns
    events = [
        hdr(),
        ex("a", "b0", "prefill", 0, 100, 0, 128),
        ex("b", "b0", "prefill", 0, 100, 0, 128),
        ex("a", "b0", "decode", 100, 200, 0, 50),
        ex("b", "b0", "decode", 100, 200, 0, 50),
        {"ev": "traj_end", "traj": "a", "t": 200, "status": "finished"},
        {"ev": "traj_end", "traj": "b", "t": 200, "status": "finished"},
    ]
    a = analyze(parse(events), peak_tflops=10.0)
    approx(a.thr_prefill, 256 / 100e-9)


# ---------------------------------------------------------------------------
# B 格式适配器：rheotrace-jsonl → canon（docs/rheotrace-spec-v0.md）


def _btrace_events():
    base = 1_727_600_000_000_000_000

    def T(ms: int) -> int:
        return base + ms * 1_000_000

    return [
        {"ts": T(0), "type": "run_start", "run_id": "r-t", "format": "rheotrace-jsonl",
         "schema_version": 0, "initial_version": 0, "engine": "test", "n_workers": 2,
         "model": "Qwen2.5-1.5B-Instruct", "clock": "wall_ns_epoch"},
        {"ts": T(100), "type": "segment_start", "run_id": "r-t", "seg_id": "s1",
         "group_id": "g1", "birth_version": 0, "t_start": T(100), "n_prompt_tokens": 112},
        {"ts": T(110), "type": "phase_span", "run_id": "r-t", "seg_id": "s1",
         "phase": "prefill", "t_start": T(100), "t_end": T(110), "n_tokens": 112},
        {"ts": T(205), "type": "phase_span", "run_id": "r-t", "seg_id": "s1",
         "phase": "decode", "t_start": T(110), "t_end": T(205), "n_tokens": 190},
        {"ts": T(205), "type": "segment_end", "run_id": "r-t", "seg_id": "s1",
         "state": "finished", "from_state": "running", "t_end": T(205),
         "n_gen_tokens": 190, "birth_version": 0, "end_version": 0, "finish_mode": "exact"},
        {"ts": T(290), "type": "weight_sync", "run_id": "r-t", "version": 1,
         "t_start": T(210), "t_end": T(290), "mode": "full", "trainer_step": 1},
        {"ts": T(300), "type": "segment_start", "run_id": "r-t", "seg_id": "s2",
         "group_id": "g2", "birth_version": 1, "t_start": T(300), "n_prompt_tokens": 100},
        {"ts": T(310), "type": "phase_span", "run_id": "r-t", "seg_id": "s2",
         "phase": "prefill", "t_start": T(300), "t_end": T(310), "n_tokens": 100},
        {"ts": T(410), "type": "phase_span", "run_id": "r-t", "seg_id": "s2",
         "phase": "decode", "t_start": T(310), "t_end": T(410), "n_tokens": 90},
        {"ts": T(410), "type": "segment_end", "run_id": "r-t", "seg_id": "s2",
         "state": "finished", "from_state": "running", "t_end": T(410),
         "n_gen_tokens": 90, "birth_version": 1, "end_version": 1, "finish_mode": "exact"},
        {"ts": T(410), "type": "run_end", "run_id": "r-t",
         "summary": {"segments": 2, "weight_syncs": 1, "gen_tokens": 280}},
    ]


def test_rheotrace_adapter_mapping(tmp_path):
    import json

    p = tmp_path / "b.rheotrace.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in _btrace_events()) + "\n", encoding="utf-8")
    a = analyze(read_trace(p), peak_tflops=10.0)
    # 真值（手算）：wall=T(100)..T(410)=310ms，active=[100,205)∪[300,410)=215ms，
    # P1=sync[210,290)=80ms，P3=[205,210)+[290,300)=15ms
    approx(a.t_wall_s, 0.310)
    approx(a.t_active_s, 0.215)
    approx(a.pause_totals["P1"], 0.080)
    approx(a.pause_totals["P3"], 0.015)
    approx(a.thr_e2e, 280 / 0.310 / 2)  # n_workers=2 → per-GPU
    assert a.gen_tokens_all == 280
    assert a.prompt_tokens == 212
    assert a.stale_token_share == 0.0
    assert a.waste == 0.0
    assert any(w.startswith("W-NO-BATCH") for w in a.warnings)  # 无 batch 标注 → 聚类降级


def test_rheotrace_adapter_gz(tmp_path):
    """`.gz` 不走字节嗅探，直接由 rheotrace reader 透明解压（B↔C 互校结论）。"""
    import gzip
    import json

    p = tmp_path / "b.rheotrace.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in _btrace_events()) + "\n", encoding="utf-8")
    gz = tmp_path / "b.rheotrace.jsonl.gz"
    gz.write_bytes(gzip.compress(p.read_bytes()))
    a = analyze(read_trace(gz), peak_tflops=10.0)
    approx(a.t_wall_s, 0.310)
    approx(a.thr_e2e, 280 / 0.310 / 2)


def test_rheotrace_adapter_on_main_synthetic(tmp_path):
    """main 上 B 已交付的合成 trace 全流程跑通（TASK-C 任务 3 用 B 的 trace）。"""
    syn = REPO / "bench" / "traces" / "synthetic"
    bimodal = syn / "synthetic-bimodal-longtail.jsonl"
    grpo = syn / "synthetic-grpo-small.jsonl"
    if not bimodal.exists() or not grpo.exists():
        import pytest

        pytest.skip("bench/traces/synthetic 未就位")

    a_bim = analyze(read_trace(bimodal))
    assert a_bim.thr_e2e > 0
    assert a_bim.pause_totals["P1"] > 0  # weight_sync_ms=800 × 多步
    assert a_bim.sarle_bc == a_bim.sarle_bc and a_bim.sarle_bc > 5 / 9
    assert a_bim.tail_token_share > 0.2

    a_grpo = analyze(read_trace(grpo))
    assert a_grpo.thr_e2e > 0

    # CLI 一条命令直接吃 B 格式
    out = tmp_path / "rep"
    assert main([str(bimodal), "--out", str(out)]) == 0
    assert (out / "report.md").exists()
