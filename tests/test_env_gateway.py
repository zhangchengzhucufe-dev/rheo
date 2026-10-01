"""Env Gateway mock 单测：分级决策全路径矩阵（design §4.3）+ 边界（§4.5）+ 网关状态机。

KV 迁移本体用 MockKVStore 替身；时间用显式 ns 整数（无时钟依赖，S2 可重放）。
"""

from dataclasses import dataclass, field

import pytest

from orchestration.env_gateway import (
    EnvGateway,
    GatewayConfig,
    GatewayError,
    HbmExhausted,
    KVSegment,
    KVTier,
    MemoryPressure,
    ThresholdCostModel,
    TierDecision,
    TierSpec,
    WaitEstimate,
)

# ---------------------------------------------------------------------------
# 夹具：小尺度确定性配置
#
# 1000 B 段的双程迁移时间：T1 = 2·1000/10⁴ = 0.2 s，T2 = 2·250/10⁴ + 0.01 = 0.06 s，
# T3 = 2·1000/2.5·10³ = 0.8 s；重算 = prefill_len / 1000 tok/s。
# ---------------------------------------------------------------------------

T1, T2, T3 = KVTier.HOST_PINNED, KVTier.COMPRESSED_4BIT, KVTier.DISK_CHECKPOINT


def tiny_cfg(**kw) -> GatewayConfig:
    tiers = {
        T1: TierSpec(T1, capacity_bytes=10_000, bandwidth_bps=10_000.0),
        T2: TierSpec(
            T2, capacity_bytes=10_000, bandwidth_bps=10_000.0, size_ratio=0.25, overhead_s=0.01
        ),
        T3: TierSpec(T3, capacity_bytes=10_000, bandwidth_bps=2_500.0),
    }
    base = dict(
        hbm_total_bytes=2000,
        hbm_soft_bytes=700,
        hbm_hold_max_s=0.2,
        tiers=tiers,
        prefill_tok_per_s=1000.0,
        sync_risk_threshold=0.5,
    )
    base.update(kw)
    return GatewayConfig(**base)


def mk_seg(sid="s-1", nbytes=1000, n_prompt=80, n_gen=20) -> KVSegment:
    return KVSegment(seg_id=sid, nbytes=nbytes, n_prompt_tokens=n_prompt, n_gen_tokens=n_gen)


def mem(used=500, total=2000, soft=700) -> MemoryPressure:
    return MemoryPressure(used, total, soft)


@dataclass
class FixedWaitPredictor:
    """确定性预测替身：永远返回同一估计，记录 observe 回流。"""

    est: WaitEstimate
    observed: list = field(default_factory=list)

    def predict(self, tool: str) -> WaitEstimate:
        return self.est

    def observe(self, tool: str, seconds: float) -> None:
        self.observed.append((tool, seconds))


@dataclass
class StubCostModel:
    """固定决策替身（测 reclaim/abort 路径时绕开真实判定）。"""

    decision: TierDecision

    def decide(self, seg, wait, mem, p_sync, cfg, tier_used) -> TierDecision:
        return self.decision


@dataclass
class MockKVStore:
    moves: list = field(default_factory=list)

    def demote(self, seg, frm, to):
        self.moves.append(("demote", seg.seg_id, frm, to))

    def promote(self, seg, frm, to):
        self.moves.append(("promote", seg.seg_id, frm, to))

    def free(self, seg, tier):
        self.moves.append(("free", seg.seg_id, tier))


def est(mean=0.05, p90=0.1, p99=0.2, n=10) -> WaitEstimate:
    return WaitEstimate(mean, p90, p99, n)


def gateway(predictor=None, cfg=None, store=None, **kw) -> EnvGateway:
    return EnvGateway(
        cfg=cfg or tiny_cfg(),
        predictor=predictor or FixedWaitPredictor(est()),
        store=store if store is not None else MockKVStore(),
        **kw,
    )


# ---------------------------------------------------------------------------
# 分级决策纯函数（design §4.3 路径矩阵，T0 行 + 降级目标行）
# ---------------------------------------------------------------------------

model = ThresholdCostModel()


def decide(seg, wait, used=500, tier_used=None, p_sync=0.0, cfg=None):
    return model.decide(seg, wait, mem(used), p_sync, cfg or tiny_cfg(), tier_used or {})


def test_short_hold_keeps_hbm():
    """短等待 + 池不紧 → 留 T0。"""
    d = decide(mk_seg(), est(p90=0.1))
    assert d.tier is KVTier.HBM and d.reason == "short_hold"


def test_pressure_forces_demote_even_short_wait():
    """池越过软水位 → 即使等待极短也强制降级（T2 双程最快且窗口内迁得完）。"""
    d = decide(mk_seg(), est(p90=0.1), used=800)
    assert d.tier is T2 and d.reason == "migrate_fastest_fit"


def test_long_wait_demotes_to_fastest_fit():
    d = decide(mk_seg(), est(p90=2.0))
    assert d.tier is T2


def test_t1_when_t2_capacity_full():
    d = decide(mk_seg(), est(p90=2.0), tier_used={T2: 10_000})
    assert d.tier is T1


def test_t3_when_t1_t2_capacity_full():
    d = decide(mk_seg(), est(p90=2.0), tier_used={T1: 10_000, T2: 10_000})
    assert d.tier is T3


def test_disk_fallback_when_window_missed_but_recompute_expensive():
    """大段：窗口内谁都迁不完，但重算（200 s）远贵于 T3 双程（80 s）→ 磁盘兜底。

    需要放大各级容量（100 KB 段超出 tiny 档 10 KB 池），只保留窗口/成本判定。
    """
    big_tiers = {
        t: TierSpec(
            t,
            capacity_bytes=1_000_000,
            bandwidth_bps=s.bandwidth_bps,
            size_ratio=s.size_ratio,
            overhead_s=s.overhead_s,
        )
        for t, s in tiny_cfg().tiers.items()
    }
    cfg = tiny_cfg(tiers=big_tiers)
    seg = mk_seg(nbytes=100_000, n_prompt=199_900, n_gen=100)
    d = model.decide(seg, est(p90=2.0), mem(500), 0.0, cfg, {})
    assert d.tier is T3 and d.reason == "disk_fallback"


def test_drop_when_recompute_cheaper():
    """窗口内迁不完且重算更便宜 → 丢弃。"""
    seg = mk_seg(nbytes=100_000)
    d = decide(seg, est(p90=2.0))
    assert d.tier is KVTier.DROPPED and d.reason == "window_or_cost"


def test_drop_on_version_risk():
    """等待窗口内大概率换权重 → 留存价值归零，直接丢弃。"""
    d = decide(mk_seg(), est(p90=0.1), p_sync=0.6)
    assert d.tier is KVTier.DROPPED and d.reason == "version_risk"


def test_drop_when_all_tiers_full():
    seg = mk_seg(nbytes=100_000)
    d = decide(seg, est(p90=2.0), tier_used={T1: 10_000, T2: 10_000, T3: 9_950})
    assert d.tier is KVTier.DROPPED


# ---------------------------------------------------------------------------
# 网关状态机：place / promote 主路径
# ---------------------------------------------------------------------------


def test_lifecycle_hbm_hold_then_promote():
    g = gateway()
    assert g.on_env_call(mk_seg(), "fast_query", t_ns=0, hbm_used_bytes=500) is None
    assert g.tier_of("s-1") is KVTier.HBM
    plan = g.on_env_result("s-1", t_ns=50_000_000, hbm_used_bytes=500)  # 50 ms 后唤醒
    assert plan.can_resume and not plan.recompute
    assert g.waiting_seg_ids() == ()
    # 真实等待回流给预测器
    assert g.predictor.observed == [("fast_query", 0.05)]


def test_lifecycle_place_t2_then_promote():
    store = MockKVStore()
    g = gateway(predictor=FixedWaitPredictor(est(p90=2.0)), store=store)
    ev = g.on_env_call(mk_seg(), "slow_crawl", t_ns=0, hbm_used_bytes=500)
    assert (ev.kind, ev.from_tier, ev.to_tier) == ("place", KVTier.HBM, T2)
    assert store.moves == [("demote", "s-1", KVTier.HBM, T2)]
    assert g.tier_used_bytes()[T2] == 250  # 4bit 有效字节
    plan = g.on_env_result("s-1", t_ns=3_000_000_000, hbm_used_bytes=100)
    assert plan.can_resume
    assert g.tier_used_bytes()[T2] == 0
    assert ("promote", "s-1", T2, KVTier.HBM) in store.moves


def test_duplicate_env_call_raises():
    g = gateway()
    g.on_env_call(mk_seg(), "t", 0, 500)
    with pytest.raises(GatewayError, match="重复拦截"):
        g.on_env_call(mk_seg(), "t", 1, 500)


def test_result_on_unregistered_seg_raises():
    g = gateway()
    with pytest.raises(GatewayError):
        g.on_env_result("ghost", 0, 500)


def test_tick_on_unregistered_seg_raises():
    g = gateway()
    with pytest.raises(GatewayError):
        g.on_wait_tick("ghost", 0, 500)


# ---------------------------------------------------------------------------
# 等待超支：下沉与逐级 step-down（design §4.4）
# ---------------------------------------------------------------------------


def test_overrun_escalates_hbm_to_offhbm():
    """预测短留了 T0，实际拖长超过 p90 → 以已等时长为下界重新决策下沉。"""
    g = gateway()  # est p90 = 0.1 → 留 HBM
    g.on_env_call(mk_seg(), "fast_query", t_ns=0, hbm_used_bytes=500)
    events = g.on_wait_tick("s-1", t_ns=1_000_000_000, hbm_used_bytes=500)  # 已等 1 s
    assert len(events) == 1
    assert (events[0].kind, events[0].from_tier, events[0].to_tier) == ("demote", KVTier.HBM, T2)
    assert events[0].reason.startswith("overrun:")


def test_overrun_steps_down_t2_to_t3():
    g = gateway(predictor=FixedWaitPredictor(est(p90=2.0, p99=4.0)))
    g.on_env_call(mk_seg(), "slow_crawl", t_ns=0, hbm_used_bytes=500)
    assert g.tier_of("s-1") is T2
    events = g.on_wait_tick("s-1", t_ns=5_000_000_000, hbm_used_bytes=500)  # 已等 5 s > p99
    assert [(e.kind, e.from_tier, e.to_tier) for e in events] == [("demote", T2, T3)]


def test_no_escalation_within_estimate():
    g = gateway()
    g.on_env_call(mk_seg(), "fast_query", t_ns=0, hbm_used_bytes=500)
    events = g.on_wait_tick("s-1", t_ns=50_000_000, hbm_used_bytes=500)  # 50 ms < p90
    assert events == []


# ---------------------------------------------------------------------------
# 显存满：池压回收与 HbmExhausted（design §4.5）
# ---------------------------------------------------------------------------


def test_reclaim_demotes_longest_remaining_first():
    """软水位越过：env_wait 段按剩余等待估计降序降级——还要等得久的先放。"""
    g = gateway()
    g.on_env_call(mk_seg("s-a"), "t", t_ns=0, hbm_used_bytes=500)
    g.on_env_call(mk_seg("s-b"), "t", t_ns=40_000_000, hbm_used_bytes=500)
    events = g.on_wait_tick("s-a", t_ns=50_000_000, hbm_used_bytes=1500)
    # s-a 剩余 0.05 s，s-b 剩余 0.09 s → s-b 先降；释放 1000 B 后已低于软水位，s-a 留住
    assert [(e.seg_id, e.to_tier) for e in events] == [("s-b", T2)]
    assert g.tier_of("s-a") is KVTier.HBM
    assert g.tier_of("s-b") is T2


def test_reclaim_demotes_until_pressure_cleared():
    g = gateway()
    for sid in ("s-a", "s-b"):
        g.on_env_call(mk_seg(sid), "t", t_ns=0, hbm_used_bytes=500)
    events = g.on_wait_tick("s-a", t_ns=50_000_000, hbm_used_bytes=1800)
    assert len(events) == 2  # 两段都放掉才回到软水位以下
    assert all(g.tier_of(sid) is T2 for sid in ("s-a", "s-b"))


def test_hbm_exhausted_when_decisions_cannot_help():
    """全部降级决策仍留 HBM（如窗口迁不完且重算贵）→ 上报调度器接手。"""
    g = gateway(
        cost_model=StubCostModel(TierDecision(KVTier.HBM, "short_hold")),
    )
    g.on_env_call(mk_seg("s-a"), "t", 0, 500)
    g.on_env_call(mk_seg("s-b"), "t", 0, 500)
    with pytest.raises(HbmExhausted, match="调度器"):
        g.on_wait_tick("s-a", t_ns=50_000_000, hbm_used_bytes=1800)


# ---------------------------------------------------------------------------
# 唤醒边界：HBM 满推迟升级 / DROPPED 转重算
# ---------------------------------------------------------------------------


def test_wake_hbm_full_defers_then_tick_promotes():
    store = MockKVStore()
    g = gateway(
        predictor=FixedWaitPredictor(est(p90=2.0)), store=store, cfg=tiny_cfg(hbm_total_bytes=2000)
    )
    g.on_env_call(mk_seg(nbytes=1500), "t", 0, hbm_used_bytes=500)
    plan = g.on_env_result("s-1", t_ns=3_000_000_000, hbm_used_bytes=800)
    # 800 + 1500 > 2000：升级推迟，不放行 running
    assert not plan.can_resume and plan.reason == "hbm_full_deferred"
    assert g.tier_of("s-1") is T2
    events = g.on_wait_tick("s-1", t_ns=4_000_000_000, hbm_used_bytes=200)
    assert [(e.kind, e.to_tier) for e in events] == [("promote", KVTier.HBM)]
    assert g.waiting_seg_ids() == ()


def test_dropped_wake_asks_recompute():
    g = gateway(cost_model=StubCostModel(TierDecision(KVTier.DROPPED, "window_or_cost")))
    ev = g.on_env_call(mk_seg(), "t", 0, 500)
    assert (ev.kind, ev.to_tier) == ("drop", KVTier.DROPPED)
    plan = g.on_env_result("s-1", t_ns=100, hbm_used_bytes=500)
    assert not plan.can_resume and plan.recompute
    assert plan.prefill_len == 100  # n_prompt + n_gen，对齐 S1 ResumePlan.prefill_len
    assert g.waiting_seg_ids() == ()


# ---------------------------------------------------------------------------
# abort：全层级释放、幂等、迟到唤醒报错
# ---------------------------------------------------------------------------


def test_abort_frees_from_offhbm_and_is_idempotent():
    store = MockKVStore()
    g = gateway(predictor=FixedWaitPredictor(est(p90=2.0)), store=store)
    g.on_env_call(mk_seg(), "t", 0, 500)
    ev = g.on_abort("s-1", t_ns=100)
    assert (ev.kind, ev.from_tier) == ("free", T2)
    assert g.tier_used_bytes()[T2] == 0
    assert store.moves[-1] == ("free", "s-1", T2)
    assert g.on_abort("s-1", 200) is None  # 幂等
    with pytest.raises(GatewayError, match="abort"):
        g.on_env_result("s-1", 300, 500)


def test_abort_while_hbm_touched_no_store():
    store = MockKVStore()
    g = gateway(store=store)
    g.on_env_call(mk_seg(), "t", 0, 500)
    ev = g.on_abort("s-1", 100)
    assert ev.from_tier is KVTier.HBM
    assert store.moves == []  # 从未迁移，无需释放动作


def test_abort_unregistered_raises():
    g = gateway()
    with pytest.raises(GatewayError):
        g.on_abort("ghost", 0)


# ---------------------------------------------------------------------------
# 版本风险接线（gateway 层 p_sync 由 sync_interval 算出）
# ---------------------------------------------------------------------------


def test_sync_interval_drives_version_risk_drop():
    """p_sync = 1 − exp(−p90/interval)；interval=1s、p90=2s → 0.865 ≥ 0.5 → 丢弃。"""
    g = gateway(
        predictor=FixedWaitPredictor(est(p90=2.0)),
        cost_model=ThresholdCostModel(),
        sync_interval_s=1.0,
    )
    ev = g.on_env_call(mk_seg(), "t", 0, 500)
    assert ev.to_tier is KVTier.DROPPED


def test_no_sync_interval_never_drops_on_risk():
    g = gateway(predictor=FixedWaitPredictor(est(p90=2.0)))
    assert g.on_env_call(mk_seg(), "t", 0, 500).to_tier is T2


def test_event_log_is_append_only():
    store = MockKVStore()
    g = gateway(predictor=FixedWaitPredictor(est(p90=2.0)), store=store)
    g.on_env_call(mk_seg(), "t", 0, 500)
    g.on_env_result("s-1", 3_000_000_000, 100)
    assert [(e.kind, e.to_tier) for e in g.events] == [
        ("place", T2),
        ("promote", KVTier.HBM),
    ]
