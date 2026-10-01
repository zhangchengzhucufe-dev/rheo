"""EWMA 等待预测器单测：回退链、按工具分离、收敛性、非平稳追踪、脏输入拒绝。

收敛夹具：TASK-S3 agent-tool 负载的三类工具延迟（快查询 ~50ms / 慢爬取 ~2s /
超长计算 ~30s），用 seeded 对数正态合成（确定性）。S3 的 D1 规格落地后以其
分布参数替换 `sample_stream`（design §5.4）。
"""

import math
import random

import pytest

from orchestration.env_gateway import EWMAWaitPredictor

# 三类工具：中位延迟与对数正态形状（σ 越大越重尾）
TOOL_CLASSES = {
    "fast_query": 0.05,
    "slow_crawl": 2.0,
    "xlong_compute": 30.0,
}
SIGMA = 0.3


def sample_stream(median: float, n: int, seed: int) -> list[float]:
    """对数正态样本流：median = exp(mu)，真均值 = exp(mu + σ²/2)。"""
    rnd = random.Random(seed)
    mu = math.log(median)
    return [math.exp(mu + SIGMA * rnd.gauss(0, 1)) for _ in range(n)]


def true_mean(median: float) -> float:
    return median * math.exp(SIGMA**2 / 2)


def test_prior_when_tool_unseen():
    p = EWMAWaitPredictor(prior_mean_s=0.5)
    est = p.predict("unknown")
    assert est.n_samples == 0
    assert est.mean_s == pytest.approx(0.5)
    assert est.p90_s >= est.mean_s
    assert est.p99_s >= est.p90_s


def test_alpha_validation():
    with pytest.raises(ValueError, match="alpha"):
        EWMAWaitPredictor(alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        EWMAWaitPredictor(alpha=1.5)


def test_rejects_bad_observations():
    p = EWMAWaitPredictor()
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="等待时长"):
            p.observe("t", bad)


def test_global_fallback_for_unseen_tool():
    """看过 8 个样本后，未见工具回退全局 EWMA（n_samples 如实报 0）。"""
    p = EWMAWaitPredictor(min_samples=5)
    for x in sample_stream(0.05, 8, seed=1):
        p.observe("fast_query", x)
    est = p.predict("other_tool")
    assert est.n_samples == 0
    assert est.mean_s == pytest.approx(true_mean(0.05), rel=0.5)  # 全局流就是快查询


def test_partial_tool_samples_reported_honestly():
    """该工具样本数不足 min_samples：数值走全局，n_samples 仍报该工具自身数量。"""
    p = EWMAWaitPredictor(min_samples=5)
    for x in sample_stream(2.0, 3, seed=2):
        p.observe("slow_crawl", x)
    est = p.predict("slow_crawl")
    assert est.n_samples == 3


def test_per_tool_separation():
    """三类工具各自学各自的：估计互不串台且与真值同序。"""
    p = EWMAWaitPredictor(min_samples=5)
    for tool, median in TOOL_CLASSES.items():
        for x in sample_stream(median, 100, seed=hash(tool) % 2**31):
            p.observe(tool, x)
    fast = p.predict("fast_query")
    slow = p.predict("slow_crawl")
    xlong = p.predict("xlong_compute")
    assert fast.n_samples == 100 and slow.n_samples == 100
    assert fast.mean_s * 10 < slow.mean_s < xlong.mean_s / 10
    assert fast.p90_s < slow.p90_s < xlong.p90_s
    assert fast.p99_s >= fast.p90_s >= fast.mean_s


@pytest.mark.parametrize("tool,median", sorted(TOOL_CLASSES.items()))
def test_convergence_to_true_mean(tool, median):
    """收敛性：300 个合成样本后 EWMA 均值落在真均值 ±20% 内。"""
    p = EWMAWaitPredictor(alpha=0.1, min_samples=5)
    for x in sample_stream(median, 300, seed=42):
        p.observe(tool, x)
    est = p.predict(tool)
    rel_err = abs(est.mean_s - true_mean(median)) / true_mean(median)
    assert rel_err < 0.20, f"{tool}: est={est.mean_s:.4f} true={true_mean(median):.4f}"


def test_tracks_distribution_shift():
    """非平稳：延迟分布漂移后，EWMA 在 100 个样本内跟上新均值。"""
    p = EWMAWaitPredictor(alpha=0.1, min_samples=5)
    for x in sample_stream(0.05, 100, seed=7):
        p.observe("tool", x)
    for x in sample_stream(2.0, 100, seed=8):
        p.observe("tool", x)
    est = p.predict("tool")
    assert est.mean_s == pytest.approx(true_mean(2.0), rel=0.20)
