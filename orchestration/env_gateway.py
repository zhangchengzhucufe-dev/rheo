"""Env Gateway：多轮工具调用的异步拦截 + env-wait KV 分级留存 + 等待时长预测。

设计依据 `docs/env-gateway-design-v0.md`（接口冻结稿 v0.1）。要点：

- 拦截点在引擎把工具请求发出、segment 转 ``env_wait`` 的时刻（``on_env_call``）；
  唤醒硬序：先升级回 HBM 或判定重算（``on_env_result``），后放行 ``running``。
- 分级 T0(HBM) → T1(host pinned) → T2(4bit) → T3(disk checkpoint) + DROPPED；
  决策是纯函数（``ThresholdCostModel``），时间与水位全部显式传参，S2 仿真器可重放。
- 等待预测按工具 EWMA（``EWMAWaitPredictor``），带全局/先验回退链。
- v0 不接真机引擎（集成排 M2 后半）；KV 迁移本体由可插拔 ``KVStore`` 承担，单测用 mock。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable


class GatewayError(RuntimeError):
    """网关状态机 misuse（重复拦截、迟到唤醒等）。"""


class HbmExhausted(GatewayError):
    """全部 env_wait 段降级后 HBM 仍越过软水位——交调度器（D4/v2 抢占）接手。"""


class KVTier(Enum):
    """KV 段存放层级（design §4.1）。"""

    HBM = "hbm"
    HOST_PINNED = "host_pinned"
    COMPRESSED_4BIT = "compressed_4bit"
    DISK_CHECKPOINT = "disk_checkpoint"
    DROPPED = "dropped"


OFFHBM_TIERS: tuple[KVTier, ...] = (
    KVTier.HOST_PINNED,
    KVTier.COMPRESSED_4BIT,
    KVTier.DISK_CHECKPOINT,
)


@dataclass(frozen=True)
class TierSpec:
    """一级的容量与带宽参数（0.5B/6GB 档默认依据 design §4.1 表，真机 A/B 回填）。"""

    tier: KVTier
    capacity_bytes: int
    bandwidth_bps: float  # 双程迁移按此带宽计（上行/下行同速的近似）
    size_ratio: float = 1.0  # 有效字节系数（T2 = 0.25）
    overhead_s: float = 0.0  # 固定开销（T2 的量化/反量化）


@dataclass(frozen=True)
class GatewayConfig:
    hbm_total_bytes: int = 2 * 1024**3  # 0.5B 档 KV 池（S1 §9 预算 1.5–2.5GB 取中）
    hbm_soft_bytes: int = int(1.5 * 1024**3)  # 软水位（越过即强制降级）
    hbm_hold_max_s: float = 0.2  # 等待 p90 不超过此值且池不紧 → 留 HBM
    tiers: Mapping[KVTier, TierSpec] = field(
        default_factory=lambda: {
            KVTier.HOST_PINNED: TierSpec(KVTier.HOST_PINNED, 8 * 1024**3, 8e9),
            KVTier.COMPRESSED_4BIT: TierSpec(
                KVTier.COMPRESSED_4BIT, 4 * 1024**3, 8e9, size_ratio=0.25, overhead_s=0.01
            ),
            KVTier.DISK_CHECKPOINT: TierSpec(KVTier.DISK_CHECKPOINT, 64 * 1024**3, 2.5e9),
        }
    )
    prefill_tok_per_s: float = 50_000.0  # 0.5B 档 prefill 吞吐基准（真机回填）
    sync_risk_threshold: float = 0.5  # p_sync ≥ 此值 → 留存价值归零，直接 DROPPED

    def eff_bytes(self, nbytes: int, tier: KVTier) -> int:
        return max(1, round(nbytes * self.tiers[tier].size_ratio))

    def roundtrip_s(self, nbytes: int, tier: KVTier) -> float:
        """降级 + 升级的双程迁移时间（秒）。"""
        spec = self.tiers[tier]
        return 2.0 * self.eff_bytes(nbytes, tier) / spec.bandwidth_bps + spec.overhead_s

    def recompute_s(self, prefill_len: int) -> float:
        return prefill_len / self.prefill_tok_per_s


@dataclass(frozen=True)
class KVSegment:
    """env-wait 段的 KV 元数据（引擎侧提供；v0 为纯数据，无句柄）。"""

    seg_id: str
    nbytes: int  # 当前 KV 占用字节（prompt + 已生成 token 的 K/V 全量）
    n_prompt_tokens: int
    n_gen_tokens: int

    @property
    def prefill_len(self) -> int:
        """丢弃后重算的长度（对齐 S1 §7.2 ``ResumePlan.prefill_len``）。"""
        return self.n_prompt_tokens + self.n_gen_tokens


@dataclass(frozen=True)
class MemoryPressure:
    """HBM 池快照（调用方采样传入；决策纯函数，不持引用）。"""

    hbm_used_bytes: int
    hbm_total_bytes: int
    hbm_soft_bytes: int

    @property
    def hbm_over_soft(self) -> bool:
        return self.hbm_used_bytes > self.hbm_soft_bytes


@dataclass(frozen=True)
class WaitEstimate:
    """等待时长分布摘要（design §5.2）。"""

    mean_s: float
    p90_s: float
    p99_s: float
    n_samples: int  # 该工具样本数（0 = 纯先验回退）


@runtime_checkable
class WaitPredictor(Protocol):
    def predict(self, tool: str) -> WaitEstimate: ...

    def observe(self, tool: str, seconds: float) -> None: ...


@runtime_checkable
class CostModel(Protocol):
    def decide(
        self,
        seg: KVSegment,
        wait: WaitEstimate,
        mem: MemoryPressure,
        p_sync: float,
        cfg: GatewayConfig,
        tier_used: Mapping[KVTier, int],
    ) -> TierDecision: ...


@runtime_checkable
class KVStore(Protocol):
    """KV 迁移本体（真机 = 引擎/pinned 池/磁盘；v0 单测用 mock 替身）。"""

    def demote(self, seg: KVSegment, from_tier: KVTier, to_tier: KVTier) -> None: ...

    def promote(self, seg: KVSegment, from_tier: KVTier, to_tier: KVTier) -> None: ...

    def free(self, seg: KVSegment, tier: KVTier) -> None: ...


@dataclass(frozen=True)
class TierDecision:
    tier: KVTier
    reason: str


@dataclass(frozen=True)
class GatewayEvent:
    """分级迁移的内存内遥测（design §8：字段按未来 rheotrace 增量事件的形状设计）。"""

    kind: str  # place | demote | drop | promote | free
    seg_id: str
    from_tier: KVTier
    to_tier: KVTier
    t_ns: int
    reason: str


@dataclass(frozen=True)
class WakePlan:
    """唤醒裁决（design §3 硬序：先有可用 KV 或重算计划，后放行 running）。"""

    seg_id: str
    can_resume: bool
    recompute: bool  # True = KV 已 DROPPED，转 re-prefill（prefill_len 给出）
    prefill_len: int | None = None
    reason: str = ""


# ---------------------------------------------------------------------------
# 等待预测（design §5）
# ---------------------------------------------------------------------------

_Z90 = 1.2815515655446004
_Z99 = 2.3263478740408408


class _Ewma:
    """单流 EWMA 均值 + 二阶矩（方差取相邻偏差的 EWMA，重尾下分位数偏保守）。"""

    __slots__ = ("mean", "var", "n")

    def __init__(self) -> None:
        self.mean = 0.0
        self.var = 0.0
        self.n = 0

    def update(self, x: float, alpha: float) -> None:
        if self.n == 0:
            self.mean = x
        else:
            prev = self.mean
            self.mean = (1.0 - alpha) * self.mean + alpha * x
            self.var = (1.0 - alpha) * self.var + alpha * (x - prev) ** 2
        self.n += 1

    def estimate(self) -> WaitEstimate:
        std = math.sqrt(self.var)
        return WaitEstimate(self.mean, self.mean + _Z90 * std, self.mean + _Z99 * std, self.n)


class EWMAWaitPredictor:
    """按工具 EWMA 等待预测器（design §5.3）。

    回退链：该工具样本 ≥ ``min_samples`` → 全局 EWMA → 构造先验。
    ``WaitEstimate.n_samples`` 如实上报回退状态，不冒充"学到了"。
    """

    def __init__(
        self,
        alpha: float = 0.1,
        min_samples: int = 5,
        prior_mean_s: float = 1.0,
        prior_std_s: float | None = None,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha 必须在 (0, 1]，得到 {alpha}")
        self._alpha = alpha
        self._min_samples = min_samples
        self._prior_mean = prior_mean_s
        self._prior_std = prior_std_s if prior_std_s is not None else prior_mean_s * 0.5
        self._tools: dict[str, _Ewma] = {}
        self._global = _Ewma()

    def observe(self, tool: str, seconds: float) -> None:
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(f"等待时长必须为正有限数，得到 {seconds!r}")
        ewma = self._tools.setdefault(tool, _Ewma())
        ewma.update(seconds, self._alpha)
        self._global.update(seconds, self._alpha)

    def predict(self, tool: str) -> WaitEstimate:
        ewma = self._tools.get(tool)
        if ewma is not None and ewma.n >= self._min_samples:
            return ewma.estimate()
        own_n = ewma.n if ewma is not None else 0
        if self._global.n >= self._min_samples:
            est = self._global.estimate()
            return WaitEstimate(est.mean_s, est.p90_s, est.p99_s, own_n)
        return WaitEstimate(
            self._prior_mean,
            self._prior_mean + _Z90 * self._prior_std,
            self._prior_mean + _Z99 * self._prior_std,
            own_n,
        )


# ---------------------------------------------------------------------------
# 分级决策（design §4.2 / §6）
# ---------------------------------------------------------------------------


class ThresholdCostModel:
    """design §4.2 伪码的直接翻译。决策纯函数：占用/水位全部经参数传入。"""

    def decide(
        self,
        seg: KVSegment,
        wait: WaitEstimate,
        mem: MemoryPressure,
        p_sync: float,
        cfg: GatewayConfig,
        tier_used: Mapping[KVTier, int],
    ) -> TierDecision:
        def capacity_ok(tier: KVTier) -> bool:
            return tier_used.get(tier, 0) + cfg.eff_bytes(seg.nbytes, tier) <= (
                cfg.tiers[tier].capacity_bytes
            )

        def fits(tier: KVTier) -> bool:
            return cfg.roundtrip_s(seg.nbytes, tier) <= wait.p90_s and capacity_ok(tier)

        if p_sync >= cfg.sync_risk_threshold:
            return TierDecision(KVTier.DROPPED, "version_risk")
        if wait.p90_s <= cfg.hbm_hold_max_s and not mem.hbm_over_soft:
            return TierDecision(KVTier.HBM, "short_hold")
        candidates = [t for t in OFFHBM_TIERS if fits(t)]
        if candidates:
            best = min(candidates, key=lambda t: cfg.roundtrip_s(seg.nbytes, t))
            return TierDecision(best, "migrate_fastest_fit")
        disk_rt = cfg.roundtrip_s(seg.nbytes, KVTier.DISK_CHECKPOINT)
        if disk_rt < cfg.recompute_s(seg.prefill_len) and capacity_ok(KVTier.DISK_CHECKPOINT):
            return TierDecision(KVTier.DISK_CHECKPOINT, "disk_fallback")
        return TierDecision(KVTier.DROPPED, "window_or_cost")


# ---------------------------------------------------------------------------
# 网关状态机（design §3 拦截点 + §4.4 超支下沉 + §4.5 边界）
# ---------------------------------------------------------------------------

_STEP_DOWN_ORDER: dict[KVTier, KVTier] = {
    KVTier.HOST_PINNED: KVTier.COMPRESSED_4BIT,
    KVTier.COMPRESSED_4BIT: KVTier.DISK_CHECKPOINT,
}


@dataclass
class _WaitingEntry:
    seg: KVSegment
    tool: str
    called_at_ns: int
    tier: KVTier
    est: WaitEstimate
    result_pending: bool = False  # 工具已返回但 HBM 满未升级（等 tick 重试）


class EnvGateway:
    """env-wait 拦截器：登记工具、分级留存、超支下沉、唤醒升级、abort 释放。

    时间一律显式传参（``t_ns``），决策可被 S2 仿真器逐位重放。
    ``store`` 为 None 时干跑（只记账不迁移）；真机集成时注入引擎侧实现。
    """

    def __init__(
        self,
        cfg: GatewayConfig | None = None,
        cost_model: CostModel | None = None,
        predictor: WaitPredictor | None = None,
        store: KVStore | None = None,
        sync_interval_s: float | None = None,
    ) -> None:
        self.cfg = cfg or GatewayConfig()
        self.cost_model = cost_model or ThresholdCostModel()
        self.predictor = predictor or EWMAWaitPredictor()
        self.store = store
        self.sync_interval_s = sync_interval_s
        self.events: list[GatewayEvent] = []
        self._waiting: dict[str, _WaitingEntry] = {}
        self._tier_used: dict[KVTier, int] = dict.fromkeys(OFFHBM_TIERS, 0)
        self._aborted: set[str] = set()

    # -- 查询 ------------------------------------------------------------

    def tier_of(self, seg_id: str) -> KVTier | None:
        entry = self._waiting.get(seg_id)
        return entry.tier if entry else None

    def tier_used_bytes(self) -> Mapping[KVTier, int]:
        return dict(self._tier_used)

    def waiting_seg_ids(self) -> tuple[str, ...]:
        return tuple(self._waiting)

    # -- 拦截点（design §3） ---------------------------------------------

    def on_env_call(
        self, seg: KVSegment, tool: str, t_ns: int, hbm_used_bytes: int
    ) -> GatewayEvent | None:
        """工具请求发出、segment 转 env_wait 时调用。返回落位事件（留 HBM 时为 None）。"""
        if seg.seg_id in self._waiting:
            raise GatewayError(f"段 {seg.seg_id} 已在 env_wait 中，重复拦截")
        if seg.seg_id in self._aborted:
            raise GatewayError(f"段 {seg.seg_id} 已 abort，不应再进入 env_wait")
        est = self.predictor.predict(tool)
        p_sync = self._p_sync(est.p90_s)
        mem = MemoryPressure(hbm_used_bytes, self.cfg.hbm_total_bytes, self.cfg.hbm_soft_bytes)
        decision = self.cost_model.decide(seg, est, mem, p_sync, self.cfg, self._tier_used)
        entry = _WaitingEntry(seg, tool, t_ns, KVTier.HBM, est)
        self._waiting[seg.seg_id] = entry
        if decision.tier is KVTier.HBM:
            return None
        return self._move(entry, decision.tier, t_ns, decision.reason, kind="place")

    def on_wait_tick(self, seg_id: str, t_ns: int, hbm_used_bytes: int) -> list[GatewayEvent]:
        """env_wait 期间周期复查：池压回收、超支下沉、升级重试。"""
        entry = self._require(seg_id)
        if entry.result_pending:
            return self._try_promote(entry, t_ns, hbm_used_bytes)
        events: list[GatewayEvent] = []
        # 池压回收（design §4.5）：全部 env_wait 段按剩余等待估计降序逐段降级
        if hbm_used_bytes > self.cfg.hbm_soft_bytes:
            events.extend(self._reclaim(t_ns, hbm_used_bytes))
        # 超支下沉（design §4.4）；reclaim 可能已把本段挪走，按挪后层级判断
        elapsed_s = (t_ns - entry.called_at_ns) / 1e9
        if entry.tier is KVTier.HBM and elapsed_s > entry.est.p90_s:
            events.extend(self._escalate_hbm(entry, t_ns, elapsed_s, hbm_used_bytes))
        elif entry.tier in _STEP_DOWN_ORDER and elapsed_s > entry.est.p99_s:
            events.append(self._move(entry, _STEP_DOWN_ORDER[entry.tier], t_ns, "overrun"))
        return events

    def on_env_result(self, seg_id: str, t_ns: int, hbm_used_bytes: int) -> WakePlan:
        """工具结果返回时调用。升级成功/重算裁决后，调用方才放行 running。"""
        entry = self._require(seg_id)
        actual_s = (t_ns - entry.called_at_ns) / 1e9
        self.predictor.observe(entry.tool, actual_s)
        if entry.tier is KVTier.DROPPED:
            self._waiting.pop(seg_id, None)
            return WakePlan(
                seg_id,
                can_resume=False,
                recompute=True,
                prefill_len=entry.seg.prefill_len,
                reason="dropped_recompute",
            )
        if entry.tier is KVTier.HBM:  # 一直在原地，无升级动作
            self._waiting.pop(seg_id, None)
            return WakePlan(seg_id, can_resume=True, recompute=False, reason="held_hbm")
        entry.result_pending = True
        self._try_promote(entry, t_ns, hbm_used_bytes)
        if seg_id not in self._waiting:  # 已升级收尾
            return WakePlan(seg_id, can_resume=True, recompute=False, reason="promoted")
        return WakePlan(seg_id, can_resume=False, recompute=False, reason="hbm_full_deferred")

    def on_abort(self, seg_id: str, t_ns: int) -> GatewayEvent | None:
        """轨迹中止：全层级释放。幂等——重复调用返回 None。"""
        entry = self._waiting.pop(seg_id, None)
        if entry is None:
            if seg_id in self._aborted:
                return None
            raise GatewayError(f"段 {seg_id} 未在网关登记，无法 abort")
        self._aborted.add(seg_id)
        event = GatewayEvent("free", seg_id, entry.tier, KVTier.DROPPED, t_ns, "abort")
        if entry.tier not in (KVTier.HBM, KVTier.DROPPED):
            self._tier_used[entry.tier] -= self._placed_bytes(entry.seg, entry.tier)
            if self.store is not None:
                self.store.free(entry.seg, entry.tier)
        self.events.append(event)
        return event

    # -- 内部 ------------------------------------------------------------

    def _require(self, seg_id: str) -> _WaitingEntry:
        entry = self._waiting.get(seg_id)
        if entry is None:
            raise GatewayError(f"段 {seg_id} 未在 env_wait 中（未拦截、已唤醒或已 abort）")
        return entry

    def _p_sync(self, p90_s: float) -> float:
        """等待窗口内发生 weight_sync 的概率（Poisson 近似，design §4.2 第 6 条）。"""
        if self.sync_interval_s is None or self.sync_interval_s <= 0:
            return 0.0
        return 1.0 - math.exp(-p90_s / self.sync_interval_s)

    def _placed_bytes(self, seg: KVSegment, tier: KVTier) -> int:
        return self.cfg.eff_bytes(seg.nbytes, tier)

    def _move(
        self, entry: _WaitingEntry, to_tier: KVTier, t_ns: int, reason: str, kind: str = "demote"
    ) -> GatewayEvent:
        from_tier = entry.tier
        if from_tier is not KVTier.HBM and from_tier is not KVTier.DROPPED:
            self._tier_used[from_tier] -= self._placed_bytes(entry.seg, from_tier)
        if to_tier is KVTier.DROPPED:
            kind = "drop"
        else:
            self._tier_used[to_tier] += self._placed_bytes(entry.seg, to_tier)
        entry.tier = to_tier
        if self.store is not None and to_tier is not KVTier.DROPPED:
            self.store.demote(entry.seg, from_tier, to_tier)
        event = GatewayEvent(kind, entry.seg.seg_id, from_tier, to_tier, t_ns, reason)
        self.events.append(event)
        return event

    def _reclaim(self, t_ns: int, hbm_used_bytes: int) -> list[GatewayEvent]:
        hbm_entries = [e for e in self._waiting.values() if e.tier is KVTier.HBM]
        # 还要等得越久越先放（剩余等待估计 = p90 − 已等时长，下界 0；已快醒的留住）
        hbm_entries.sort(
            key=lambda e: max(0.0, e.est.p90_s - (t_ns - e.called_at_ns) / 1e9), reverse=True
        )
        used = hbm_used_bytes
        events: list[GatewayEvent] = []
        for entry in hbm_entries:
            if used <= self.cfg.hbm_soft_bytes:
                break
            est = self._shifted_est(entry, (t_ns - entry.called_at_ns) / 1e9)
            p_sync = self._p_sync(est.p90_s)
            mem = MemoryPressure(used, self.cfg.hbm_total_bytes, self.cfg.hbm_soft_bytes)
            d = self.cost_model.decide(entry.seg, est, mem, p_sync, self.cfg, self._tier_used)
            if d.tier is KVTier.HBM:
                continue  # 决策说留（窗口迁不完且重算更贵）——跳过，看下一段
            if d.tier is not KVTier.DROPPED:
                used -= entry.seg.nbytes
            events.append(self._move(entry, d.tier, t_ns, f"reclaim:{d.reason}"))
        if used > self.cfg.hbm_soft_bytes:
            raise HbmExhausted(
                f"全部 {len(hbm_entries)} 个 env_wait 段降级后 HBM 仍超软水位"
                f"（used={used} > soft={self.cfg.hbm_soft_bytes}），需调度器接手"
            )
        return events

    def _escalate_hbm(
        self, entry: _WaitingEntry, t_ns: int, elapsed_s: float, hbm_used_bytes: int
    ) -> list[GatewayEvent]:
        est = self._shifted_est(entry, elapsed_s)
        p_sync = self._p_sync(est.p90_s)
        mem = MemoryPressure(hbm_used_bytes, self.cfg.hbm_total_bytes, self.cfg.hbm_soft_bytes)
        d = self.cost_model.decide(entry.seg, est, mem, p_sync, self.cfg, self._tier_used)
        if d.tier is KVTier.HBM:
            return []
        return [self._move(entry, d.tier, t_ns, f"overrun:{d.reason}")]

    def _try_promote(
        self, entry: _WaitingEntry, t_ns: int, hbm_used_bytes: int
    ) -> list[GatewayEvent]:
        if hbm_used_bytes + entry.seg.nbytes > self.cfg.hbm_total_bytes:
            # HBM 满：升级推迟，留在当前层级等下一个 tick（design §4.5），不放行 running
            return []
        from_tier = entry.tier
        if from_tier not in (KVTier.HBM, KVTier.DROPPED):
            self._tier_used[from_tier] -= self._placed_bytes(entry.seg, from_tier)
            if self.store is not None:
                self.store.promote(entry.seg, from_tier, KVTier.HBM)
        event = GatewayEvent("promote", entry.seg.seg_id, from_tier, KVTier.HBM, t_ns, "wake")
        self.events.append(event)
        self._waiting.pop(entry.seg.seg_id, None)
        return [event]

    def _shifted_est(self, entry: _WaitingEntry, elapsed_s: float) -> WaitEstimate:
        """以已等时长为下界的更新估计（design §4.4）。"""
        return WaitEstimate(
            mean_s=max(entry.est.mean_s, elapsed_s),
            p90_s=max(entry.est.p90_s, elapsed_s),
            p99_s=max(entry.est.p99_s, elapsed_s),
            n_samples=entry.est.n_samples,
        )
