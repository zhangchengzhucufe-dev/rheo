"""Scheduler v1 核心：组感知调度 + DAPO 零方差组 abort + token 边界暂停 stub。

接口冻结见 docs/scheduler-design-v0.md（§4 策略接口 / §7 原语签名 / §8 成本模型输入表）。
纯逻辑、零重依赖：S2 仿真器直接 import 本模块的类型实现策略 harness；
引擎侧（verl hooks 适配，S1 任务 3）负责把本模块的决策/计划映射成 RheoTrace 事件。

v1 只激活 continue 与 re-prefill（含 DAPO abort）；shadow/ε-stale 仅留判定输入位（§8.1），
任何试图激活它们的路径都会抛 SchedulerError。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import count
from math import ceil
from typing import Literal

# ---------------------------------------------------------------------------
# §4 策略接口（冻结）
# ---------------------------------------------------------------------------

FinishMode = Literal["exact", "shadow", "stale"]
ResumeMode = Literal["re-prefill", "shadow", "stale"]
Action = Literal["continue", "pause", "re-prefill", "abort"]

#: abort reason 注册表（设计 §6.4；落 segment_end.reason，对齐 rheotrace-spec E15）
ABORT_REASONS: tuple[str, ...] = ("zero_variance_early", "weight_skip", "policy_preempt")

SegState = Literal["running", "paused", "env_wait", "finished", "aborted"]

#: 段状态机（rheotrace-spec §2 表的运行时镜像；终态转换只经 finish/abort 通道）
_SEG_TRANSITIONS: dict[str, frozenset[str]] = {
    "running": frozenset({"paused", "env_wait", "finished", "aborted"}),
    "paused": frozenset({"running", "aborted"}),
    "env_wait": frozenset({"running", "aborted"}),
    "finished": frozenset(),
    "aborted": frozenset(),
}

_INFLIGHT: frozenset[str] = frozenset({"running", "paused", "env_wait"})


class SchedulerError(Exception):
    """调度核心的契约违反：非法决策、非法状态转换、留位分支被激活等。"""


@dataclass(frozen=True)
class MemoryWatermark:
    kv_used_bytes: int
    kv_total_bytes: int


@dataclass(frozen=True)
class SegmentView:
    seg_id: str
    group_id: str
    state: Literal["running", "paused", "env_wait"]
    n_prompt_tokens: int
    n_gen_tokens: int
    max_new_tokens: int | None
    birth_version: int
    current_version: int
    batch_id: str | None
    finish_mode: FinishMode | None


@dataclass(frozen=True)
class GroupView:
    group_id: str
    n_total: int
    n_running: int
    n_paused: int
    n_env_wait: int
    n_finished: int
    n_aborted: int
    finished_rewards: tuple[float, ...]
    dispatched: bool


@dataclass(frozen=True)
class Observation:
    t_now_ns: int
    current_version: int
    pending_sync: bool
    segments: tuple[SegmentView, ...]
    groups: tuple[GroupView, ...]
    memory: MemoryWatermark
    candidates: tuple[str, ...]


@dataclass(frozen=True)
class Decision:
    action: Action
    targets: tuple[str, ...] = ()
    group_id: str | None = None
    reason: str | None = None


Policy = Callable[[Observation], Decision]


# ---------------------------------------------------------------------------
# §7 token 边界 pause/resume 原语（签名冻结；本体 M3 兑现，v1 stub）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PauseReceipt:
    seg_id: str
    paused_at_version: int
    n_prompt_tokens: int
    n_gen_tokens: int


@dataclass(frozen=True)
class ResumePlan:
    seg_id: str
    mode: ResumeMode
    target_version: int
    prefill_len: int
    kv_action: Literal["drop", "keep", "demote"]


class TokenBoundaryPauser:
    """token 边界暂停原语 stub（设计 §7.1）。

    v1：安全点 = 引擎批边界（语义仍是"在途轨迹停在 token 边界"，粒度粗于逐 token）；
    状态机转换与 trace 事件由 Scheduler.apply / 适配层落实。逐 token 边界与
    shadow/stale 续跑本体由 M3 WeightManager 兑现。
    """

    def pause(
        self, segs: Sequence[SegmentView], reason: str = "token_boundary"
    ) -> list[PauseReceipt]:
        del reason  # reason 由调用方落 segment_state 事件；原语本身只负责收据
        return [
            PauseReceipt(
                seg_id=s.seg_id,
                paused_at_version=s.current_version,
                n_prompt_tokens=s.n_prompt_tokens,
                n_gen_tokens=s.n_gen_tokens,
            )
            for s in segs
            if s.state == "running"
        ]

    def resume(
        self,
        receipt: PauseReceipt,
        mode: ResumeMode,
        version: int | None = None,
    ) -> ResumePlan:
        if mode != "re-prefill":
            raise SchedulerError(
                f"resume mode={mode!r} 属 shadow/ε-stale 分支，v1 留位未激活（§8.1，M3/M4 兑现）"
            )
        target = version if version is not None else receipt.paused_at_version
        return ResumePlan(
            seg_id=receipt.seg_id,
            mode=mode,
            target_version=target,
            prefill_len=receipt.n_prompt_tokens + receipt.n_gen_tokens,
            kv_action="drop",
        )


# ---------------------------------------------------------------------------
# §8 成本模型：v1 判定序 + 分支激活位
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchedulerConfig:
    """v1 可配项。shadow/stale 分支的激活位恒 False（留位，§8.1）。"""

    #: DAPO early 检测的完成比例阈值 ρ；None = early 检测关闭（§6.3 默认）
    early_abort_rho: float | None = None
    shadow_branch_enabled: bool = False
    stale_branch_enabled: bool = False
    #: max_new_tokens 未知的段的 KV 估算用默认值（§5.2 容量推算）
    default_max_new_tokens: int = 1024


def decide_resume(
    seg: SegmentView,
    current_version: int,
    group_rejected: bool,
    cfg: SchedulerConfig | None = None,
) -> Decision:
    """§8.2 v1 判定序：paused/env_wait 段在版本边界后的续跑决策。

    输入即 §8.1 re-prefill 分支的冻结判定输入；shadow/ε-stale 分支在本函数中
    显式留位——激活位开着也抛错（本体不存在，诚实地不可用）。
    """
    cfg = cfg or SchedulerConfig()
    if cfg.shadow_branch_enabled or cfg.stale_branch_enabled:
        raise SchedulerError("shadow/ε-stale 分支 v1 留位未激活（设计 §8.1）")
    if group_rejected:
        return Decision("abort", group_id=seg.group_id, reason="zero_variance_early")
    if seg.current_version == current_version:
        return Decision("continue", targets=(seg.seg_id,))
    if seg.max_new_tokens is not None and seg.n_gen_tokens >= seg.max_new_tokens:
        # 防御路径：到长未收尾（正常应由引擎 finish），不值得为其重算
        return Decision("abort", group_id=seg.group_id, reason="weight_skip")
    return Decision("re-prefill", targets=(seg.seg_id,))


# ---------------------------------------------------------------------------
# §5/§6 引擎侧组件
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchPlan:
    """D1 批组装结果（§5.2）：整组共批，batch_id 经 segment_start.batch_id 落 trace。"""

    batch_id: str
    group_ids: tuple[str, ...]
    seg_ids: tuple[str, ...]
    est_kv_tokens: int


def early_abort_candidate(group: GroupView, rho: float | None) -> bool:
    """§6.3 early 零方差检测（纯函数，policy 与引擎共用）。

    触发：已完成且奖励可得数 ≥ max(2, ceil(ρ·G)) 且奖励全等（精确相等；
    规则奖励是离散值，浮点容差反而引入误判面）。ρ=None 或奖励数据缺失 → 不触发
    （诚实降级，§6.2）。
    """
    if rho is None or not group.finished_rewards:
        return False
    threshold = max(2, ceil(rho * group.n_total))
    if len(group.finished_rewards) < threshold:
        return False
    rewards = group.finished_rewards
    return min(rewards) == max(rewards)


def validate_decision(obs: Observation, decision: Decision) -> None:
    """§4.3 决策合法性校验（引擎防线，S2 可复用）。违反即 SchedulerError。"""
    if decision.action == "continue":
        if obs.pending_sync and any(s.state == "running" for s in obs.segments):
            raise SchedulerError(
                "pending_sync=True 且仍有 running 段时 continue 非法：安全点未达成（§4.3）；"
                "全体已到界后 continue 合法（引擎执行指针翻转）"
            )
        return
    if decision.action == "abort":
        if not decision.group_id:
            raise SchedulerError("abort 必须携带 group_id（组级中止）")
        if decision.reason not in ABORT_REASONS:
            raise SchedulerError(f"abort.reason={decision.reason!r} 不在注册表 {ABORT_REASONS}")
        if decision.targets:
            raise SchedulerError("abort 是组级决策，不得携带 targets")
        group = next((g for g in obs.groups if g.group_id == decision.group_id), None)
        if group is None:
            raise SchedulerError(f"abort 目标组 {decision.group_id!r} 不在途或不存在")
        return

    by_id = {s.seg_id: s for s in obs.segments}
    if decision.action == "pause":
        for seg_id in decision.targets:
            seg = by_id.get(seg_id)
            if seg is None:
                raise SchedulerError(f"pause 目标段 {seg_id!r} 不在途")
            if seg.state != "running":
                raise SchedulerError(f"pause 目标段 {seg_id!r} 状态为 {seg.state}，须为 running")
    elif decision.action == "re-prefill":
        for seg_id in decision.targets:
            seg = by_id.get(seg_id)
            if seg is None:
                raise SchedulerError(f"re-prefill 目标段 {seg_id!r} 不在途")
            if seg.state != "paused":
                raise SchedulerError(
                    f"re-prefill 目标段 {seg_id!r} 状态为 {seg.state}，须为 paused"
                )
    else:
        raise SchedulerError(f"未知 action：{decision.action!r}")


@dataclass
class _SegState:
    seg_id: str
    group_id: str
    n_prompt_tokens: int
    max_new_tokens: int | None
    birth_version: int
    state: SegState = "running"
    n_gen_tokens: int = 0
    batch_id: str | None = None
    current_version: int = -1  # -1 = 未初始化，注册时置为 birth_version

    def __post_init__(self) -> None:
        if self.current_version == -1:
            self.current_version = self.birth_version

    def view(self) -> SegmentView:
        assert self.state in _INFLIGHT  # 观察快照只含在途段（§4.2）
        return SegmentView(
            seg_id=self.seg_id,
            group_id=self.group_id,
            state=self.state,  # type: ignore[arg-type]
            n_prompt_tokens=self.n_prompt_tokens,
            n_gen_tokens=self.n_gen_tokens,
            max_new_tokens=self.max_new_tokens,
            birth_version=self.birth_version,
            current_version=self.current_version,
            batch_id=self.batch_id,
            finish_mode=None,
        )


@dataclass
class _GroupState:
    group_id: str
    seg_ids: list[str]
    rewards: dict[str, float] = field(default_factory=dict)  # seg_id → 奖励，按完成序回填
    rejected: bool = False


class Scheduler:
    """调度核心（引擎侧状态 + 决策产生/应用）。

    适配层职责：把本类的状态推进（set_state/advance/finish）与决策产出映射成
    RheoTrace 事件（设计 §7.3 映射表）；奖励经 report_reward 回填（§6.2）。
    """

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        self.cfg = config or SchedulerConfig()
        self._segs: dict[str, _SegState] = {}
        self._groups: dict[str, _GroupState] = {}
        self._batch_seq = count()
        self._aborted_tokens = 0
        self._aborted_by_reason: dict[str, int] = {}

    # -- 登记（适配层在 submit/rollout 开始时调用） --------------------------

    def register_group(self, group_id: str, seg_ids: Sequence[str]) -> None:
        if group_id in self._groups:
            raise SchedulerError(f"组重复登记：{group_id!r}")
        if not seg_ids:
            raise SchedulerError(f"组 {group_id!r} 至少要有一个段")
        self._groups[group_id] = _GroupState(group_id=group_id, seg_ids=list(seg_ids))

    def register_segment(
        self,
        seg_id: str,
        group_id: str,
        n_prompt_tokens: int,
        max_new_tokens: int | None = None,
        birth_version: int = 0,
    ) -> None:
        if seg_id in self._segs:
            raise SchedulerError(f"段重复登记：{seg_id!r}")
        group = self._groups.get(group_id)
        if group is None:
            raise SchedulerError(f"段 {seg_id!r} 的组 {group_id!r} 未登记")
        if seg_id not in group.seg_ids:
            raise SchedulerError(f"段 {seg_id!r} 不在组 {group_id!r} 的成员清单里")
        self._segs[seg_id] = _SegState(
            seg_id=seg_id,
            group_id=group_id,
            n_prompt_tokens=n_prompt_tokens,
            max_new_tokens=max_new_tokens,
            birth_version=birth_version,
        )

    # -- 状态推进（适配层从引擎回调里驱动） ---------------------------------

    def set_state(self, seg_id: str, state: SegState) -> None:
        seg = self._segs[seg_id]
        if state not in _SEG_TRANSITIONS.get(seg.state, frozenset()):
            raise SchedulerError(f"非法状态转换 {seg.seg_id}: {seg.state} → {state}")
        seg.state = state

    def advance(self, seg_id: str, n_tokens: int = 1) -> None:
        if n_tokens < 0:
            raise SchedulerError("n_tokens 不得为负")
        self._segs[seg_id].n_gen_tokens += n_tokens

    def finish(self, seg_id: str) -> None:
        self.set_state(seg_id, "finished")

    def report_reward(self, group_id: str, seg_id: str, reward: float) -> GroupView | None:
        """§6.2 奖励回填。exact 零方差命中时标记组拒收并返回该组视图（§6.5 口径 2）。

        组级拒收不影响已 finished 段的 trace 终态——数据去留是训练侧的事，
        调度器只负责把"哪些组被拒收"记账并经 waste_report 暴露。
        """
        group = self._groups.get(group_id)
        if group is None:
            raise SchedulerError(f"奖励回填的组不存在：{group_id!r}")
        if seg_id not in group.seg_ids:
            raise SchedulerError(f"段 {seg_id!r} 不属于组 {group_id!r}")
        if seg_id in group.rewards:
            raise SchedulerError(f"段 {seg_id!r} 重复回填奖励")
        if self._segs[seg_id].state != "finished":
            raise SchedulerError(f"段 {seg_id!r} 未 finished，不能回填奖励")
        group.rewards[seg_id] = reward
        if len(group.rewards) == len(group.seg_ids):
            values = list(group.rewards.values())
            if min(values) == max(values):
                group.rejected = True
                return self.group_view(group_id)
        return None

    # -- D1：批组装（§5.2） ---------------------------------------------------

    def plan_batch(
        self, candidate_group_ids: Sequence[str], kv_free_tokens: int
    ) -> BatchPlan | None:
        """整组共批：FIFO 装配完整组，超预算减 k 不拆组（§5.2 冻结规则）。

        容量估算 = Σ_seg E[max_new_tokens]（设计 §5.2 口径；prompt 占用另计，
        这里只按生成 token 推）。连一个组都放不下时返回 None（本波不派）。
        """
        picked: list[str] = []
        seg_ids: list[str] = []
        est = 0
        for group_id in candidate_group_ids:
            group = self._groups.get(group_id)
            if group is None:
                raise SchedulerError(f"候选组不存在：{group_id!r}")
            pending = [
                self._segs[s]
                for s in group.seg_ids
                if self._segs[s].batch_id is None and self._segs[s].state == "running"
            ]
            if len(pending) != len(group.seg_ids):
                continue  # 组已（部分）派发或不在批装状态，跳过
            group_est = sum(
                s.max_new_tokens
                if s.max_new_tokens is not None
                else self.cfg.default_max_new_tokens
                for s in pending
            )
            if est + group_est > kv_free_tokens:
                continue  # 超预算：减 k 不拆组（§5.2），尝试更小的后续组
            picked.append(group_id)
            seg_ids.extend(s.seg_id for s in pending)
            est += group_est
        if not picked:
            return None
        batch_id = f"b-{next(self._batch_seq)}"
        for seg_id in seg_ids:
            self._segs[seg_id].batch_id = batch_id
        return BatchPlan(
            batch_id=batch_id, group_ids=tuple(picked), seg_ids=tuple(seg_ids), est_kv_tokens=est
        )

    # -- D2/D3：决策产生与应用 ------------------------------------------------

    def on_sync(self, current_version: int) -> list[Decision]:
        """weight_sync 安全点后的逐段续跑决策（§8.2 判定序）。

        前置：适配层已把全部 running 段置 paused（D2 强制安全点）。env_wait 段
        跨版本同样进入判定（其 KV 在 v1 恒失效，§8.3）。
        """
        decisions: list[Decision] = []
        for seg in self._segs.values():
            if seg.state not in ("paused", "env_wait"):
                continue
            group = self._groups[seg.group_id]
            d = decide_resume(seg.view(), current_version, group.rejected, self.cfg)
            decisions.append(d)
        return self._coalesce(decisions)

    @staticmethod
    def _coalesce(decisions: list[Decision]) -> list[Decision]:
        """同 action+reason 的组级 abort 合并（一组一条）；其余保持逐段。"""
        aborts: dict[str, Decision] = {}
        out: list[Decision] = []
        for d in decisions:
            if d.action == "abort" and d.group_id is not None:
                aborts.setdefault(d.group_id, d)
            else:
                out.append(d)
        out.extend(aborts.values())
        return out

    def apply(self, decision: Decision, current_version: int | None = None) -> None:
        """把决策应用到内部状态（trace 事件由适配层按 §7.3 映射落盘）。"""
        if decision.action == "continue":
            return
        if decision.action == "pause":
            for seg_id in decision.targets:
                self.set_state(seg_id, "paused")
            return
        if decision.action == "re-prefill":
            for seg_id in decision.targets:
                self.set_state(seg_id, "running")
                if current_version is not None:
                    self._segs[seg_id].current_version = current_version
            return
        # abort：组级
        assert decision.group_id is not None and decision.reason is not None
        group = self._groups[decision.group_id]
        for seg_id in group.seg_ids:
            seg = self._segs[seg_id]
            if seg.state in _INFLIGHT:
                self.set_state(seg_id, "aborted")
                self._aborted_tokens += seg.n_gen_tokens
                self._aborted_by_reason[decision.reason] = (
                    self._aborted_by_reason.get(decision.reason, 0) + 1
                )

    # -- 观察快照与对账 --------------------------------------------------------

    def group_view(self, group_id: str) -> GroupView:
        group = self._groups[group_id]
        counts = {"running": 0, "paused": 0, "env_wait": 0, "finished": 0, "aborted": 0}
        dispatched = False
        for seg_id in group.seg_ids:
            seg = self._segs[seg_id]
            counts[seg.state] += 1
            dispatched = dispatched or seg.batch_id is not None
        return GroupView(
            group_id=group_id,
            n_total=len(group.seg_ids),
            n_running=counts["running"],
            n_paused=counts["paused"],
            n_env_wait=counts["env_wait"],
            n_finished=counts["finished"],
            n_aborted=counts["aborted"],
            finished_rewards=tuple(group.rewards.values()),
            dispatched=dispatched,
        )

    def observation(
        self,
        current_version: int,
        kv_used_bytes: int,
        kv_total_bytes: int,
        *,
        t_now_ns: int = 0,
        pending_sync: bool = False,
        candidates: Sequence[str] = (),
    ) -> Observation:
        segments = tuple(s.view() for s in self._segs.values() if s.state in _INFLIGHT)
        groups = tuple(
            self.group_view(g)
            for g in self._groups
            if any(self._segs[s].state in _INFLIGHT for s in self._groups[g].seg_ids)
        )
        return Observation(
            t_now_ns=t_now_ns,
            current_version=current_version,
            pending_sync=pending_sync,
            segments=segments,
            groups=groups,
            memory=MemoryWatermark(kv_used_bytes=kv_used_bytes, kv_total_bytes=kv_total_bytes),
            candidates=tuple(candidates),
        )

    def waste_report(self) -> dict[str, object]:
        """§6.5 废 token 对账（段级口径；组级拒收清单一并暴露）。"""
        return {
            "aborted_segments": sum(self._aborted_by_reason.values()),
            "aborted_tokens": self._aborted_tokens,
            "by_reason": dict(self._aborted_by_reason),
            "rejected_groups": [g for g, s in self._groups.items() if s.rejected],
        }


# ---------------------------------------------------------------------------
# v1 默认策略（D2/D3；D1 批组装走 plan_batch，策略只表达"本波不派"= continue）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class V1Policy:
    """默认策略：early 零方差 abort（§6.3，ρ 取配置）+ 安全点 pause + 跨版本 re-prefill。

    纯函数（§4.4）：无内部状态；early 检测是 Observation 的纯函数
    （early_abort_candidate）。单次调用至多返回一条决策，引擎循环重复调用
    直到 continue。
    """

    config: SchedulerConfig = field(default_factory=SchedulerConfig)

    def __call__(self, obs: Observation) -> Decision:
        for group in obs.groups:
            if early_abort_candidate(group, self.config.early_abort_rho):
                return Decision("abort", group_id=group.group_id, reason="zero_variance_early")
        if obs.pending_sync:
            running = tuple(s.seg_id for s in obs.segments if s.state == "running")
            if running:
                return Decision("pause", targets=running, reason="token_boundary")
        stale_paused = tuple(
            s.seg_id
            for s in obs.segments
            if s.state == "paused" and s.current_version < obs.current_version
        )
        if stale_paused:
            return Decision("re-prefill", targets=stale_paused)
        return Decision("continue")
