"""runtime/scheduler.py 单测：设计文档 §4/§5/§6/§7/§8 的契约与判定输入覆盖。

重点：成本模型四分支判定输入全覆盖（shadow/ε-stale 测"留位不激活"的拒绝路径，
设计 §10 验收行）。
"""

import pytest

from runtime.scheduler import (
    ABORT_REASONS,
    Decision,
    GroupView,
    MemoryWatermark,
    Observation,
    Scheduler,
    SchedulerConfig,
    SchedulerError,
    SegmentView,
    TokenBoundaryPauser,
    V1Policy,
    decide_resume,
    early_abort_candidate,
    validate_decision,
)

# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------


def make_seg(
    seg_id="s1",
    group_id="g1",
    state="paused",
    n_prompt_tokens=100,
    n_gen_tokens=50,
    max_new_tokens=200,
    birth_version=1,
    current_version=1,
) -> SegmentView:
    return SegmentView(
        seg_id=seg_id,
        group_id=group_id,
        state=state,
        n_prompt_tokens=n_prompt_tokens,
        n_gen_tokens=n_gen_tokens,
        max_new_tokens=max_new_tokens,
        birth_version=birth_version,
        current_version=current_version,
        batch_id=None,
        finish_mode=None,
    )


def make_group(n_total=4, n_finished=0, rewards=(), dispatched=False) -> GroupView:
    return GroupView(
        group_id="g1",
        n_total=n_total,
        n_running=n_total - n_finished,
        n_paused=0,
        n_env_wait=0,
        n_finished=n_finished,
        n_aborted=0,
        finished_rewards=tuple(rewards),
        dispatched=dispatched,
    )


def fresh_scheduler(group_size=4, n_groups=2, cfg: SchedulerConfig | None = None) -> Scheduler:
    sch = Scheduler(cfg)
    for g in range(n_groups):
        gid = f"g{g}"
        segs = [f"{gid}-s{i}" for i in range(group_size)]
        sch.register_group(gid, segs)
        for s in segs:
            sch.register_segment(s, gid, n_prompt_tokens=100, max_new_tokens=200)
    return sch


class TestConfigAndRegistration:
    def test_rho_bounds(self):
        assert SchedulerConfig(early_abort_rho=0.5).early_abort_rho == 0.5
        assert SchedulerConfig(early_abort_rho=None).early_abort_rho is None
        for bad in (0.0, -0.1, 1.5):
            with pytest.raises(SchedulerError, match="early_abort_rho"):
                SchedulerConfig(early_abort_rho=bad)

    def test_default_max_new_tokens_must_be_positive(self):
        with pytest.raises(SchedulerError, match="default_max_new_tokens"):
            SchedulerConfig(default_max_new_tokens=0)

    def test_negative_birth_version_rejected(self):
        sch = Scheduler()
        sch.register_group("g0", ["s0"])
        with pytest.raises(SchedulerError, match="birth_version"):
            sch.register_segment("s0", "g0", n_prompt_tokens=1, birth_version=-1)

    def test_duplicate_seg_ids_in_group_rejected(self):
        sch = Scheduler()
        with pytest.raises(SchedulerError, match="重复段 id"):
            sch.register_group("g0", ["s0", "s0"])


# ---------------------------------------------------------------------------
# §5 组感知共批
# ---------------------------------------------------------------------------


class TestBatching:
    def test_whole_groups_never_split(self):
        sch = fresh_scheduler(group_size=4)
        plan = sch.plan_batch(["g0", "g1"], kv_free_tokens=10**9)
        assert plan is not None
        assert plan.group_ids == ("g0", "g1")
        assert len(plan.seg_ids) == 8
        assert all(sch._segs[s].batch_id == plan.batch_id for s in plan.seg_ids)

    def test_budget_overflow_reduces_k_never_splits(self):
        sch = fresh_scheduler(group_size=4, n_groups=2)
        # 每组估算 = 4 × 200 = 800；只放得下一组
        plan = sch.plan_batch(["g0", "g1"], kv_free_tokens=1000)
        assert plan is not None
        assert plan.group_ids == ("g0",) and len(plan.seg_ids) == 4

    def test_budget_overflow_truncates_fifo_no_starvation_skipping(self):
        sch = Scheduler()
        # g0 大组（2×900=1800），g1 小组（1×100=100）：预算 1000 连小组都够，
        # 但大组在队头装不下即整波不派（减 k 是截断不是跳过，FIFO 队头公平）
        sch.register_group("g0", ["g0-s0", "g0-s1"])
        for s in ("g0-s0", "g0-s1"):
            sch.register_segment(s, "g0", n_prompt_tokens=10, max_new_tokens=900)
        sch.register_group("g1", ["g1-s0"])
        sch.register_segment("g1-s0", "g1", n_prompt_tokens=10, max_new_tokens=100)
        assert sch.plan_batch(["g0", "g1"], kv_free_tokens=1000) is None

    def test_single_group_too_large_returns_none(self):
        sch = fresh_scheduler(group_size=4)
        assert sch.plan_batch(["g0"], kv_free_tokens=799) is None

    def test_batch_ids_increase_across_waves(self):
        sch = fresh_scheduler(group_size=4, n_groups=2)
        p1 = sch.plan_batch(["g0"], kv_free_tokens=10**9)
        p2 = sch.plan_batch(["g1"], kv_free_tokens=10**9)
        assert p1.batch_id != p2.batch_id

    def test_already_dispatched_group_skipped(self):
        sch = fresh_scheduler(group_size=4)
        sch.plan_batch(["g0"], kv_free_tokens=10**9)
        plan = sch.plan_batch(["g0", "g1"], kv_free_tokens=10**9)
        assert plan is not None
        assert plan.group_ids == ("g1",)

    def test_unknown_candidate_raises(self):
        sch = fresh_scheduler()
        with pytest.raises(SchedulerError, match="候选组不存在"):
            sch.plan_batch(["nope"], kv_free_tokens=10**9)

    def test_partial_group_mixed_batches_is_impossible(self):
        # 组一旦部分派发（这里人为把一个段挂到别的批），plan_batch 必须整组跳过
        sch = fresh_scheduler(group_size=4)
        sch._segs["g0-s0"].batch_id = "b-999"
        assert sch.plan_batch(["g0"], kv_free_tokens=10**9) is None


# ---------------------------------------------------------------------------
# §8 成本模型：四分支判定输入（验收：四分支全覆盖）
# ---------------------------------------------------------------------------


class TestCostModelBranches:
    def test_continue_when_not_crossed_version(self):
        d = decide_resume(make_seg(current_version=1), current_version=1, group_rejected=False)
        assert d.action == "continue" and d.targets == ("s1",)

    def test_re_prefill_when_crossed_version(self):
        d = decide_resume(make_seg(current_version=1), current_version=2, group_rejected=False)
        assert d.action == "re-prefill" and d.targets == ("s1",)

    def test_re_prefill_when_max_new_unknown(self):
        # max_new_tokens 未知 → 剩余价值按可得处理，走 re-prefill（§8.2 输入③缺省）
        d = decide_resume(
            make_seg(current_version=1, max_new_tokens=None),
            current_version=2,
            group_rejected=False,
        )
        assert d.action == "re-prefill"

    def test_max_len_segment_never_enters_decision(self):
        # 到长段是引擎 finish 通道（§8.2 注）：不产生 abort（组级 abort 会连坐全组），
        # 而是契约错误——适配层必须在进入判定前收尾
        with pytest.raises(SchedulerError, match="finish"):
            decide_resume(
                make_seg(current_version=1, n_gen_tokens=200, max_new_tokens=200),
                current_version=2,
                group_rejected=False,
            )

    def test_group_rejected_aborts(self):
        d = decide_resume(make_seg(current_version=1), current_version=2, group_rejected=True)
        assert d.action == "abort" and d.reason == "zero_variance_early"

    def test_shadow_branch_reserved_not_activatable(self):
        cfg = SchedulerConfig(shadow_branch_enabled=True)
        with pytest.raises(SchedulerError, match="留位未激活"):
            decide_resume(make_seg(), current_version=2, group_rejected=False, cfg=cfg)

    def test_stale_branch_reserved_not_activatable(self):
        cfg = SchedulerConfig(stale_branch_enabled=True)
        with pytest.raises(SchedulerError, match="留位未激活"):
            decide_resume(make_seg(), current_version=2, group_rejected=False, cfg=cfg)

    def test_shadow_stale_branches_disabled_by_default(self):
        # 默认配置下判定序照常走 continue/re-prefill，不会触碰留位分支
        assert SchedulerConfig().shadow_branch_enabled is False
        assert SchedulerConfig().stale_branch_enabled is False
        assert decide_resume(make_seg(), 2, False).action == "re-prefill"


# ---------------------------------------------------------------------------
# §7 token 边界原语 stub
# ---------------------------------------------------------------------------


class TestTokenBoundaryPauser:
    def test_pause_collects_running_only(self):
        segs = [make_seg("s1", state="running"), make_seg("s2", state="paused")]
        receipts = TokenBoundaryPauser().pause(segs)
        assert [r.seg_id for r in receipts] == ["s1"]
        assert receipts[0].paused_at_version == 1
        assert receipts[0].n_prompt_tokens == 100
        assert receipts[0].n_gen_tokens == 50

    def test_resume_re_prefill_plan(self):
        receipt = TokenBoundaryPauser().pause([make_seg("s1", state="running")])[0]
        plan = TokenBoundaryPauser().resume(receipt, "re-prefill", version=7)
        assert plan.seg_id == "s1"
        assert plan.mode == "re-prefill"
        assert plan.target_version == 7
        assert plan.prefill_len == 150  # n_prompt + n_gen（冻结输入）
        assert plan.kv_action == "drop"

    def test_resume_default_version_is_paused_at(self):
        receipt = TokenBoundaryPauser().pause([make_seg("s1", state="running")])[0]
        assert TokenBoundaryPauser().resume(receipt, "re-prefill").target_version == 1

    @pytest.mark.parametrize("mode", ["shadow", "stale"])
    def test_resume_shadow_stale_reserved(self, mode):
        receipt = TokenBoundaryPauser().pause([make_seg("s1", state="running")])[0]
        with pytest.raises(SchedulerError, match="留位未激活"):
            TokenBoundaryPauser().resume(receipt, mode)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# §6 DAPO 零方差
# ---------------------------------------------------------------------------


class TestZeroVariance:
    def test_early_disabled_by_default(self):
        assert not early_abort_candidate(make_group(n_finished=4, rewards=(1, 1, 1, 1)), None)

    def test_early_needs_at_least_two(self):
        assert not early_abort_candidate(make_group(rewards=(1,)), 0.5)

    def test_early_threshold_is_ceil_rho_times_g(self):
        # G=4, ρ=0.5 → ceil=2 → max(2, 2)=2：两个全等即触发
        assert early_abort_candidate(make_group(rewards=(0.0, 0.0)), 0.5)
        # ρ=0.75 → ceil=3：两个不够
        assert not early_abort_candidate(make_group(rewards=(0.0, 0.0)), 0.75)
        assert early_abort_candidate(make_group(rewards=(0.0, 0.0, 0.0)), 0.75)

    def test_early_not_triggered_when_rewards_differ(self):
        assert not early_abort_candidate(make_group(rewards=(0.0, 1.0, 1.0)), 0.5)

    def test_early_requires_reward_data(self):
        assert not early_abort_candidate(make_group(n_finished=4, rewards=()), 0.5)

    def test_exact_rejection_marks_group(self):
        sch = fresh_scheduler(group_size=2, n_groups=1)
        for s in ("g0-s0", "g0-s1"):
            sch.finish(s)
        sch.report_reward("g0", "g0-s0", 1.0)
        assert sch.group_view("g0").finished_rewards == (1.0,)
        assert sch.waste_report()["rejected_groups"] == []
        sch.report_reward("g0", "g0-s1", 1.0)
        assert sch.waste_report()["rejected_groups"] == ["g0"]

    def test_exact_no_rejection_when_variance_nonzero(self):
        sch = fresh_scheduler(group_size=2, n_groups=1)
        for s in ("g0-s0", "g0-s1"):
            sch.finish(s)
        sch.report_reward("g0", "g0-s0", 1.0)
        sch.report_reward("g0", "g0-s1", 0.0)
        assert sch.waste_report()["rejected_groups"] == []

    def test_report_reward_validations(self):
        sch = fresh_scheduler(group_size=2, n_groups=1)
        with pytest.raises(SchedulerError, match="不存在"):
            sch.report_reward("nope", "g0-s0", 1.0)
        with pytest.raises(SchedulerError, match="不属于"):
            sch.report_reward("g0", "g1-s0", 1.0)
        with pytest.raises(SchedulerError, match="未 finished"):
            sch.report_reward("g0", "g0-s0", 1.0)
        sch.finish("g0-s0")
        sch.report_reward("g0", "g0-s0", 1.0)
        with pytest.raises(SchedulerError, match="重复回填"):
            sch.report_reward("g0", "g0-s0", 1.0)


# ---------------------------------------------------------------------------
# §4.3 决策校验
# ---------------------------------------------------------------------------


class TestValidateDecision:
    def _obs(self, pending_sync=False, include_group=True):
        segs = (make_seg("s1", state="running"), make_seg("s2", state="paused"))
        groups = (make_group(),) if include_group else ()
        return Observation(
            t_now_ns=0,
            current_version=1,
            pending_sync=pending_sync,
            segments=segs,
            groups=groups,
            memory=MemoryWatermark(kv_used_bytes=1, kv_total_bytes=10),
            candidates=(),
        )

    def test_continue_illegal_under_pending_sync(self):
        with pytest.raises(SchedulerError, match="安全点"):
            validate_decision(self._obs(pending_sync=True), Decision("continue"))

    def test_abort_requires_reason_in_registry(self):
        with pytest.raises(SchedulerError, match="注册表"):
            validate_decision(self._obs(), Decision("abort", group_id="g1", reason="whatever"))
        assert "zero_variance_early" in ABORT_REASONS

    def test_abort_requires_group_and_rejects_targets(self):
        with pytest.raises(SchedulerError, match="group_id"):
            validate_decision(self._obs(), Decision("abort", reason="weight_skip"))
        with pytest.raises(SchedulerError, match="targets"):
            validate_decision(
                self._obs(), Decision("abort", group_id="g1", reason="weight_skip", targets=("s1",))
            )
        with pytest.raises(SchedulerError, match="不在途"):
            validate_decision(
                self._obs(), Decision("abort", group_id="ghost", reason="weight_skip")
            )

    def test_state_preconditions(self):
        with pytest.raises(SchedulerError, match="running"):
            validate_decision(self._obs(), Decision("pause", targets=("s2",)))
        with pytest.raises(SchedulerError, match="须为 paused 或 env_wait"):
            validate_decision(self._obs(), Decision("re-prefill", targets=("s1",)))
        with pytest.raises(SchedulerError, match="不在途"):
            validate_decision(self._obs(), Decision("pause", targets=("s9",)))

    def test_re_prefill_from_env_wait_is_legal(self):
        # §8.3 唤醒路径：env_wait 段跨版本 → re-prefill 合法（无需 paused 中转）
        obs = self._obs()
        env_wait_obs = Observation(
            t_now_ns=0,
            current_version=2,
            pending_sync=False,
            segments=(make_seg("s1", state="env_wait", current_version=1),),
            groups=obs.groups,
            memory=obs.memory,
            candidates=(),
        )
        validate_decision(env_wait_obs, Decision("re-prefill", targets=("s1",)))
        d = decide_resume(
            make_seg("s1", state="env_wait", current_version=1), 2, group_rejected=False
        )
        assert d.action == "re-prefill" and d.targets == ("s1",)

    def test_valid_decisions_pass(self):
        obs = self._obs(pending_sync=True)
        validate_decision(obs, Decision("pause", targets=("s1",)))
        validate_decision(self._obs(), Decision("re-prefill", targets=("s2",)))
        validate_decision(self._obs(), Decision("abort", group_id="g1", reason="weight_skip"))
        validate_decision(self._obs(), Decision("continue"))


# ---------------------------------------------------------------------------
# 状态机与废 token 对账
# ---------------------------------------------------------------------------


class TestStateMachineAndWaste:
    def test_advance_requires_running_state(self):
        # 非 running 段推进 token 会让 n_gen_tokens 与 trace 失真，污染废 token 对账（§6.5）
        sch = fresh_scheduler(group_size=1, n_groups=1)
        with pytest.raises(SchedulerError, match="未知段"):
            sch.advance("nope", 1)  # 未知段
        sch.set_state("g0-s0", "paused")
        with pytest.raises(SchedulerError, match="只有 running 段"):
            sch.advance("g0-s0", 1)
        sch.set_state("g0-s0", "running")
        sch.advance("g0-s0", 5)
        sch.finish("g0-s0")
        with pytest.raises(SchedulerError, match="只有 running 段"):
            sch.advance("g0-s0", 1)
        assert sch._segs["g0-s0"].n_gen_tokens == 5

    def test_illegal_transition_rejected(self):
        sch = fresh_scheduler(group_size=1, n_groups=1)
        sch.set_state("g0-s0", "env_wait")
        with pytest.raises(SchedulerError, match="非法状态转换"):
            sch.set_state("g0-s0", "paused")  # env_wait → paused 直转非法（§7.2）

    def test_terminal_state_is_final(self):
        sch = fresh_scheduler(group_size=1, n_groups=1)
        sch.finish("g0-s0")
        with pytest.raises(SchedulerError, match="非法状态转换"):
            sch.set_state("g0-s0", "running")

    def test_abort_accounting(self):
        sch = fresh_scheduler(group_size=2, n_groups=1, cfg=SchedulerConfig(early_abort_rho=0.5))
        sch.plan_batch(["g0"], kv_free_tokens=10**9)
        for s in ("g0-s0", "g0-s1"):
            sch.advance(s, 30)
        sch.finish("g0-s0")
        sch.report_reward("g0", "g0-s0", 0.0)
        sch.advance("g0-s1", 17)
        sch.apply(Decision("abort", group_id="g0", reason="zero_variance_early"))
        waste = sch.waste_report()
        assert waste["aborted_segments"] == 1
        assert waste["aborted_tokens"] == 47  # 仅在途段的已产出 token（§6.5 口径 1）
        assert waste["by_reason"] == {"zero_variance_early": 1}

    def test_abort_skips_terminal_segments(self):
        sch = fresh_scheduler(group_size=2, n_groups=1)
        sch.advance("g0-s0", 10)
        sch.finish("g0-s0")
        sch.apply(Decision("abort", group_id="g0", reason="weight_skip"))
        assert sch.group_view("g0").n_finished == 1
        assert sch.group_view("g0").n_aborted == 1
        assert sch.waste_report()["aborted_tokens"] == 0  # finished 段不算废 token

    def test_registration_validations(self):
        sch = fresh_scheduler(group_size=1, n_groups=1)
        with pytest.raises(SchedulerError, match="重复登记"):
            sch.register_group("g0", ["x"])
        with pytest.raises(SchedulerError, match="至少要有一个段"):
            sch.register_group("gx", [])
        with pytest.raises(SchedulerError, match="未登记"):
            sch.register_segment("sx", "ghost", n_prompt_tokens=1)
        with pytest.raises(SchedulerError, match="成员清单"):
            sch.register_segment("sx", "g0", n_prompt_tokens=1)


class TestOnSync:
    def test_env_wait_excluded_decided_at_wake(self):
        # §8.3：env_wait 段不进安全点判定（唤醒后由适配层调 decide_resume）
        sch = fresh_scheduler(group_size=2, n_groups=1)
        sch.plan_batch(["g0"], kv_free_tokens=10**9)
        sch.set_state("g0-s0", "paused")
        sch.set_state("g0-s1", "env_wait")
        decisions = sch.on_sync(current_version=2)
        # 只有 paused 段出 re-prefill；env_wait 段不出决策
        assert decisions == [Decision("re-prefill", targets=("g0-s0",))]

    def test_on_sync_decisions_pass_own_validation(self):
        # 产出 → 校验往返：on_sync 的每条决策必须过得了 §4.3 防线
        sch = fresh_scheduler(group_size=2, n_groups=1)
        sch.plan_batch(["g0"], kv_free_tokens=10**9)
        sch.apply(Decision("pause", targets=("g0-s0", "g0-s1")))
        obs = sch.observation(2, 0, 10**9)
        for d in sch.on_sync(2):
            validate_decision(obs, d)

    def test_on_sync_coalesces_group_abort(self):
        sch = Scheduler(SchedulerConfig(early_abort_rho=0.5))
        segs = [f"g0-s{i}" for i in range(2)]
        sch.register_group("g0", segs)
        for s in segs:
            sch.register_segment(s, "g0", n_prompt_tokens=10, max_new_tokens=100)
        sch.set_state("g0-s0", "paused")
        sch.set_state("g0-s1", "paused")
        sch._groups["g0"].rejected = True  # exact 拒收后残留 paused 段（防御路径）
        decisions = sch.on_sync(2)
        assert decisions == [Decision("abort", group_id="g0", reason="zero_variance_early")]

    def test_apply_rejects_bad_abort_payload(self):
        # 适配层绕过 validate_decision 直呼 apply 时的第二道防线
        sch = fresh_scheduler(group_size=1, n_groups=1)
        with pytest.raises(SchedulerError, match="载荷非法"):
            sch.apply(Decision("abort", group_id="g0", reason="made_up"))
        with pytest.raises(SchedulerError, match="载荷非法"):
            sch.apply(Decision("abort", reason="weight_skip"))


# ---------------------------------------------------------------------------
# V1Policy + 引擎循环（决策 → 校验 → 应用）
# ---------------------------------------------------------------------------


class TestV1PolicyLoop:
    def _run_until_continue(self, sch, obs):
        """§4.3：单次调用至多一条决策，引擎循环重复调用直到 continue。"""
        policy = V1Policy(SchedulerConfig(early_abort_rho=0.5))
        applied = []
        for _ in range(16):
            d = policy(obs)
            validate_decision(obs, d)
            if d.action == "continue":
                break
            sch.apply(d, current_version=obs.current_version)
            applied.append(d)
            obs = sch.observation(
                obs.current_version,
                kv_used_bytes=0,
                kv_total_bytes=10**9,
                pending_sync=obs.pending_sync,
            )
        return applied

    def test_pending_sync_pauses_all_running(self):
        sch = fresh_scheduler(group_size=2, n_groups=1)
        sch.plan_batch(["g0"], kv_free_tokens=10**9)
        obs = sch.observation(1, 0, 10**9, pending_sync=True)
        applied = self._run_until_continue(sch, obs)
        assert applied and applied[0].action == "pause"
        assert set(applied[0].targets) == {"g0-s0", "g0-s1"}
        assert all(sch._segs[s].state == "paused" for s in ("g0-s0", "g0-s1"))

    def test_post_sync_re_prefill_then_version_stamp(self):
        sch = fresh_scheduler(group_size=1, n_groups=1)
        sch.plan_batch(["g0"], kv_free_tokens=10**9)
        sch.apply(Decision("pause", targets=("g0-s0",)))
        obs = sch.observation(2, 0, 10**9)  # 同步完成，版本 2
        applied = self._run_until_continue(sch, obs)
        assert [d.action for d in applied] == ["re-prefill"]
        assert sch._segs["g0-s0"].current_version == 2  # re-prefill 后按新版本记账

    def test_fully_terminal_group_not_in_observation(self):
        # 组全员终态后不再出现在 obs.groups（§4.2"未收尾组"），策略对其返回 continue
        sch = fresh_scheduler(group_size=2, n_groups=1)
        for s in ("g0-s0", "g0-s1"):
            sch.advance(s, 5)
            sch.finish(s)
            sch.report_reward("g0", s, 1.0)
        obs = sch.observation(1, 0, 10**9)
        assert obs.groups == ()
        assert V1Policy()(obs).action == "continue"

    def test_early_abort_path_in_loop(self):
        sch = Scheduler(SchedulerConfig(early_abort_rho=0.5))
        segs = [f"g0-s{i}" for i in range(4)]
        sch.register_group("g0", segs)
        for s in segs:
            sch.register_segment(s, "g0", n_prompt_tokens=10, max_new_tokens=100)
        sch.plan_batch(["g0"], kv_free_tokens=10**9)
        for s in segs:
            sch.advance(s, 9)
        for s in segs[:2]:  # ceil(0.5*4)=2，两个全等即触发
            sch.finish(s)
            sch.report_reward("g0", s, 0.0)
        obs = sch.observation(1, 0, 10**9)
        applied = self._run_until_continue(sch, obs)
        assert [d.action for d in applied] == ["abort"]
        assert applied[0].reason == "zero_variance_early"
        assert sch.group_view("g0").n_aborted == 2
        assert sch.waste_report()["aborted_tokens"] == 18

    def test_full_cycle_trace_shaped(self):
        """端到端：D1 共批 → D2 安全点 → 同步 → re-prefill → 完成，状态全程合法。"""
        sch = fresh_scheduler(group_size=2, n_groups=1)
        plan = sch.plan_batch(["g0"], kv_free_tokens=10**9)
        assert plan is not None
        policy = V1Policy()
        version = 1
        # D2：安全点
        obs = sch.observation(version, 0, 10**9, pending_sync=True)
        d = policy(obs)
        validate_decision(obs, d)
        sch.apply(d)
        # 同步完成 → 续跑
        version = 2
        obs = sch.observation(version, 0, 10**9)
        d = policy(obs)
        validate_decision(obs, d)
        assert d.action == "re-prefill"
        sch.apply(d, current_version=version)
        for s in plan.seg_ids:
            sch.advance(s, 50)
            sch.finish(s)
        assert sch.group_view("g0").n_finished == 2
        assert sch.observation(version, 0, 10**9).segments == ()  # 无在途段
