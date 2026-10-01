"""sim 回放核心测试：确定性、池/分区两种分配、env_wait 释放 worker、step 屏障、校准。"""

import math
from pathlib import Path

from rheotrace import generate, write
from sim import (
    ClusterConfig,
    assign_fifo_greedy,
    calibrate,
    load_workload,
    run_step_barrier,
)
from sim.engine import _run_step
from sim.workload import SegmentJob, Step, Workload


def make_job(seg_id: str, tasks, n_gen: int = 0, step: int = 1) -> SegmentJob:
    return SegmentJob(
        seg_id=seg_id,
        group_id=f"g-{seg_id}",
        step=step,
        n_prompt_tokens=0,
        n_gen_tokens=n_gen or int(sum(t for k, t in tasks if k == "decode")),
        tasks=list(tasks),
        arrival_idx=0,
        birth_version=step,
    )


def workload_from_gen(preset: str, seed: int = 3):
    """gen 合成 trace → workload（临时文件，测试自足）。"""
    events = generate(preset=preset, seed=seed)
    path = Path(__file__).parent / "_tmp_sim_workload.jsonl"
    write(path, events)
    try:
        return load_workload(path)
    finally:
        path.unlink()


def test_deterministic_replay():
    """同输入两次运行结果逐字段一致（无随机源）。"""
    wl = workload_from_gen("grpo")
    cfg = ClusterConfig(n_gpu=4, decode_tok_per_s=50.0, weight_sync_s=1.0, schedule_gap_s=0.5)
    r1 = run_step_barrier(wl, cfg)
    r2 = run_step_barrier(wl, cfg)
    assert r1 == r2


def test_pool_mode_makespan_analytic():
    """池模式：等长段作业的 span ≈ 总工作量 / (n_gpu × 率)，尾段量化有界。"""
    n_jobs, n_gpu, rate = 16, 4, 10.0
    jobs = [make_job(f"s{i}", [("decode", 100)]) for i in range(n_jobs)]
    cfg = ClusterConfig(n_gpu=n_gpu, decode_tok_per_s=rate)
    span, busy_d, _, _ = _run_step(jobs, None, n_gpu, cfg)
    ideal = n_jobs * 100 / (n_gpu * rate)
    assert math.isclose(busy_d, n_jobs * 100 / rate, rel_tol=1e-9)
    assert ideal <= span < ideal + 100 / rate  # 尾段量化不超过一个段的服务时长


def test_env_wait_releases_worker():
    """env_wait 不占 worker：等待期间其他段推进（agent 连续批处理的本质）。"""
    # A: decode 1s → env_wait 10s → decode 1s；B: decode 5s。单 worker
    a = make_job("A", [("decode", 10), ("env_wait", 10e9), ("decode", 10)])
    b = make_job("B", [("decode", 50)])
    cfg = ClusterConfig(n_gpu=1, decode_tok_per_s=10.0)
    span, busy_d, _, wait = _run_step([a, b], None, 1, cfg)
    assert math.isclose(span, 12.0, rel_tol=1e-9)  # B 在 A 等待期间跑完，A 再 1s 收尾
    assert math.isclose(busy_d, 7.0, rel_tol=1e-9)
    assert math.isclose(wait, 10.0, rel_tol=1e-9)


def test_step_barrier_serializes_windows():
    """step 屏障：窗口严格串行，wall = Σ(gap + span + sync)。"""
    wl = workload_from_gen("grpo")
    n_steps = len(wl.steps)
    cfg = ClusterConfig(n_gpu=4, decode_tok_per_s=100.0, weight_sync_s=2.0, schedule_gap_s=1.0)
    res = run_step_barrier(wl, cfg)
    assert len(res.steps) == n_steps
    assert math.isclose(res.wall_s, sum(s.total_s for s in res.steps), rel_tol=1e-9)
    assert res.wall_s >= n_steps * (1.0 + 2.0)  # 下界：每步至少 gap+sync


def test_partitioned_assign_respects_queues():
    """分区模式：worker 只服务自己的队列（段间隔离，策略 harness 的接缝）。"""
    jobs = [make_job(f"s{i}", [("decode", 100)]) for i in range(4)]
    cfg = ClusterConfig(n_gpu=2, decode_tok_per_s=10.0)
    queues = assign_fifo_greedy(jobs, 2, cfg)
    assert sorted(len(q) for q in queues) == [2, 2]
    span, _, _, _ = _run_step(jobs, queues, 2, cfg)
    assert math.isclose(span, 20.0, rel_tol=1e-9)  # 每 worker 2 段 × 10s


def test_empty_window_passthrough():
    """无段窗口（eval 窗）：span 用真实时长透传，保证墙钟可比。"""
    wl = Workload(
        steps=[
            Step(index=0, jobs=[], sync_s=5.0, gap_s=0.0, real_span_s=42.0),
            Step(index=1, jobs=[make_job("x", [("decode", 80)])], sync_s=5.0, gap_s=0.0),
        ],
        n_gpu=2,
    )
    cfg = ClusterConfig(n_gpu=2, decode_tok_per_s=10.0, weight_sync_s=5.0, schedule_gap_s=0.0)
    res = run_step_barrier(wl, cfg)
    assert math.isclose(res.steps[0].span_s, 42.0)
    # 单段不可跨 worker 切分：80 tok / 10 tok/s = 8s
    assert math.isclose(res.steps[1].span_s, 8.0)
    assert math.isclose(res.wall_s, (42.0 + 5.0) + (8.0 + 5.0))


def test_calibration_on_synthetic_grpo():
    """合成 trace 端到端校准：机制误差有界（gen 虚拟钟与恒率池模型的口径差）。"""
    events = generate(preset="grpo", seed=7)
    path = Path(__file__).parent / "_tmp_sim_cal.jsonl"
    write(path, events)
    try:
        cal = calibrate(path)
    finally:
        path.unlink()
    assert "replay fidelity" in cal.report()
    s, r = cal._wall(cal.sim_per_window)
    assert abs(s - r) / r < 0.35


def test_calibration_agent_envwait_runs():
    """agent 负载（env_wait + 反复暂停）走通校准全链路，env_wait 是独立停顿类。"""
    events = generate(preset="agent", seed=7)
    path = Path(__file__).parent / "_tmp_sim_agent.jsonl"
    write(path, events)
    try:
        cal = calibrate(path)
    finally:
        path.unlink()
    breakdown = cal.sim.stall_breakdown()
    assert breakdown["env_wait"] > 0
    assert breakdown["decode"] > 0


def test_bimodal_longtail_tail_dominates():
    """双峰长尾：span 由最长段构成下界、不低于均值工作量口径。"""
    wl = workload_from_gen("bimodal")
    step = next(s for s in wl.steps if s.jobs)
    cfg = ClusterConfig(n_gpu=wl.n_gpu, decode_tok_per_s=100.0)
    span, busy_d, _, _ = _run_step(step.jobs, None, wl.n_gpu, cfg)
    longest = max(sum(t for k, t in j.tasks if k == "decode") for j in step.jobs)
    assert span >= longest / cfg.decode_tok_per_s
    assert span >= busy_d / wl.n_gpu
