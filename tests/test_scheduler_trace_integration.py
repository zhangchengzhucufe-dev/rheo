"""端到端集成：Scheduler 决策 → §7.3 事件映射 → RheoTrace 落盘 → rheotrace.validate。

验证设计文档 §10 验收行"§7 stub 产出的每条 trace 过 rheotrace.validate"，
同时检验 scheduler_group_reject 自定义事件走 spec §8 的 W06 前向兼容路径。
本文件中的 TraceAdapter 是任务 3（verl hooks 接入）适配层的参考实现：
虚拟钟、全事件显式 ts（spec §9：与 auto-now 混流必乱序）、终态只经 segment_end。
"""

from rheotrace import read, validate
from runtime.scheduler import (
    Decision,
    PauseReceipt,
    ResumePlan,
    Scheduler,
    SchedulerConfig,
    TokenBoundaryPauser,
    V1Policy,
    validate_decision,
)


class TraceAdapter:
    """把 Scheduler 的状态推进与决策映射成 §7.3 事件（虚拟钟，显式 ts 流）。"""

    def __init__(self, path: str, *, model: str = "Qwen2.5-0.5B-Instruct") -> None:
        from rheotrace import TraceWriter

        self.w = TraceWriter(path, engine="rheo-scheduler-v1-test", model=model)
        self.t: int = self.w.run_start_ts
        self.sch = Scheduler()
        self.policy = V1Policy(SchedulerConfig(early_abort_rho=0.5))
        self.pauser = TokenBoundaryPauser()  # §7 原语 stub：pause 收据 / resume 计划
        self.version = 0  # 账本：weight_sync 推进，segment_end.end_version 依据（spec §5.1）
        self.state: dict[str, str] = {}  # seg_id → 对外轨迹状态（适配层自有账）
        self.seg_meta: dict[str, dict] = {}
        self.receipts: dict[str, PauseReceipt] = {}
        self.resume_plans: dict[str, ResumePlan] = {}

    def tick(self, delta_ns: int = 1_000_000) -> int:
        self.t += delta_ns
        return self.t

    # -- submit / D1 派发 ------------------------------------------------------

    def submit_group(self, group_id: str, prompt_lens: list[int], max_new: int) -> None:
        segs = [f"{group_id}-s{i}" for i in range(len(prompt_lens))]
        self.sch.register_group(group_id, segs)
        for s, np_ in zip(segs, prompt_lens, strict=True):
            self.sch.register_segment(
                s, group_id, n_prompt_tokens=np_, max_new_tokens=max_new, birth_version=self.version
            )
            self.state[s] = "queued"  # 已登记未派发：trace 上无存在
            self.seg_meta[s] = {"n_prompt": np_}

    def dispatch(self, candidate_groups: list[str], kv_free_tokens: int = 10**9) -> str | None:
        plan = self.sch.plan_batch(candidate_groups, kv_free_tokens, birth_version=self.version)
        if plan is None:
            return None
        for seg_id in plan.seg_ids:
            self.w.emit(
                "segment_start",
                ts=self.t,  # 全文件显式 ts：segment_start 非区间型，auto-now 会混流乱序（spec §9）
                seg_id=seg_id,
                group_id=self.sch._segs[seg_id].group_id,
                batch_id=plan.batch_id,
                birth_version=self.version,
                t_start=self.t,
                n_prompt_tokens=self.seg_meta[seg_id]["n_prompt"],
            )
            self.state[seg_id] = "running"
        return plan.batch_id

    # -- 执行阶段（引擎回调） ---------------------------------------------------

    def prefill(self, seg_id: str, *, reason: str | None = None) -> None:
        t0, t1 = self.tick(), self.tick()
        if reason == "re-prefill":
            # §7.3 冻结：KV 已 drop，重算全长 = n_prompt + 已生成（ResumePlan.prefill_len）
            n = self.resume_plans[seg_id].prefill_len
        else:
            n = self.seg_meta[seg_id]["n_prompt"]
        fields = {"meta": {"reason": reason}} if reason else {}
        self.w.emit(
            "phase_span",
            seg_id=seg_id,
            phase="prefill",
            t_start=t0,
            t_end=t1,
            n_tokens=n,
            **fields,
        )

    def decode(self, seg_id: str, n_tokens: int) -> None:
        t0, t1 = self.tick(), self.tick()
        start_idx = self.sch._segs[seg_id].n_gen_tokens  # 覆盖铺接：下一块起点
        self.sch.advance(seg_id, n_tokens)
        self.w.emit(
            "phase_span",
            seg_id=seg_id,
            phase="decode",
            t_start=t0,
            t_end=t1,
            n_tokens=n_tokens,
        )
        self.w.emit(
            "token_logprob",
            ts=t1,
            seg_id=seg_id,
            version=self.version,
            start_idx=start_idx,
            n=n_tokens,
            lp=[-0.5] * n_tokens,
        )

    # -- 决策循环与 §7.3 映射 ---------------------------------------------------

    def run_policy_until_continue(
        self, *, current_version: int, pending_sync: bool = False
    ) -> list[Decision]:
        applied: list[Decision] = []
        for _ in range(32):
            obs = self.sch.observation(
                current_version,
                0,
                10**9,
                t_now_ns=self.t,
                pending_sync=pending_sync,
                # §4.2 D1 上下文；V1Policy 不消费，批组装走 plan_batch
                candidates=self.sch.pending_groups(),
            )
            d = self.policy(obs)
            validate_decision(obs, d)
            if d.action == "continue":
                break
            self.apply(d, current_version=current_version)
            applied.append(d)
        return applied

    def apply(self, d: Decision, *, current_version: int) -> None:
        if d.action == "pause":
            views = [self.sch._segs[s].view() for s in d.targets]  # 须在置 paused 前取（§7 契约）
            for receipt in self.pauser.pause(views):
                self.receipts[receipt.seg_id] = receipt
            self.sch.apply(d)
            for seg_id in d.targets:
                self.w.emit(
                    "segment_state",
                    ts=self.tick(),
                    seg_id=seg_id,
                    from_state="running",
                    to_state="paused",
                    reason=d.reason or "token_boundary",
                )
                self.state[seg_id] = "paused"
        elif d.action == "re-prefill":
            for seg_id in d.targets:
                plan = self.pauser.resume(
                    self.receipts[seg_id], "re-prefill", version=current_version
                )
                self.resume_plans[seg_id] = plan
            self.sch.apply(d, current_version=current_version)
            for seg_id in d.targets:
                self.w.emit(
                    "segment_state",
                    ts=self.tick(),
                    seg_id=seg_id,
                    from_state="paused",
                    to_state="running",
                )
                self.state[seg_id] = "running"
        elif d.action == "abort":
            group = self.sch._groups[d.group_id]
            started = [
                s for s in group.seg_ids if self.state[s] not in ("finished", "aborted", "queued")
            ]
            self.sch.apply(d)  # 调度器记账：在途 + 排队段（排队零 token，§5.2）
            for seg_id in started:  # 终态转换只经 segment_end（spec §2，无中间 state 事件）
                self._emit_segment_end(seg_id, aborted=True, reason=d.reason)
            # queued 段：trace 上无存在（无 segment_start），不发任何事件
        else:
            self.sch.apply(d)

    def sync_weights(self, version: int, trainer_step: int) -> None:
        t0, t1 = self.tick(), self.tick()
        self.w.emit(
            "weight_sync",
            version=version,
            t_start=t0,
            t_end=t1,
            mode="full",
            trainer_step=trainer_step,
        )
        self.version = version

    def finish(self, seg_id: str) -> None:
        self.sch.finish(seg_id)
        self._emit_segment_end(seg_id, aborted=False, reason=None)

    def report_reward(self, group_id: str, seg_id: str, reward: float) -> None:
        view = self.sch.report_reward(group_id, seg_id, reward)
        if view is not None:  # exact 零方差拒收 → §6.5 口径 2 过渡事件（W06 前向兼容）
            total = sum(self.sch._segs[s].n_gen_tokens for s in self.sch._groups[group_id].seg_ids)
            self.w.emit(
                "scheduler_group_reject",
                ts=self.tick(),
                group_id=group_id,
                reason="zero_variance",
                meta={"n_gen_tokens_group": total},  # 全组已生成 token：拒收即全废（§6.5）
            )

    def _emit_segment_end(self, seg_id: str, *, aborted: bool, reason: str | None) -> None:
        seg = self.sch._segs[seg_id]
        self.w.emit(
            "segment_end",
            seg_id=seg_id,
            state="aborted" if aborted else "finished",
            from_state=self.state[seg_id],
            t_end=self.tick(),
            n_gen_tokens=seg.n_gen_tokens,
            birth_version=seg.birth_version,
            end_version=self.version,
            reason=reason,
            finish_mode="exact",
        )
        self.state[seg_id] = "aborted" if aborted else "finished"

    def close(self) -> None:
        self.w.close(end_ts=self.tick())


# ---------------------------------------------------------------------------


class TestSchedulerTraceIntegration:
    def test_scenario_a_sync_re_prefill_full_cycle(self, tmp_path):
        """D1 共批 → D2 安全点 pause → weight_sync → re-prefill → finish → 下一波。"""
        path = str(tmp_path / "sched-a.rheotrace.jsonl")
        ad = TraceAdapter(path)
        ad.submit_group("g0", [100, 120], max_new=200)
        ad.submit_group("g1", [80, 90], max_new=64)
        ad.tick()

        assert ad.dispatch(["g0"]) == "b-0"
        for seg in ("g0-s0", "g0-s1"):
            ad.prefill(seg)
            ad.decode(seg, 50)

        # D2：安全点（先全体到界，再落 weight_sync）
        applied = ad.run_policy_until_continue(current_version=0, pending_sync=True)
        assert [d.action for d in applied] == ["pause"]
        assert set(applied[0].targets) == {"g0-s0", "g0-s1"}
        ad.sync_weights(1, trainer_step=1)

        # D2 续跑：全体跨版本 → re-prefill（§7 原语给出冻结输入 prefill_len = n_prompt + n_gen）
        applied = ad.run_policy_until_continue(current_version=1)
        assert [d.action for d in applied] == ["re-prefill"]
        assert ad.resume_plans["g0-s0"].prefill_len == 150  # 100 + 50
        assert ad.resume_plans["g0-s1"].prefill_len == 170  # 120 + 50
        assert ad.resume_plans["g0-s0"].kv_action == "drop"
        for seg in ("g0-s0", "g0-s1"):
            ad.prefill(seg, reason="re-prefill")  # span n_tokens = prefill_len（§7.3）
            ad.decode(seg, 40 if seg.endswith("s0") else 30)

        ad.finish("g0-s0")
        ad.finish("g0-s1")
        ad.report_reward("g0", "g0-s0", 0.9)
        ad.report_reward("g0", "g0-s1", 0.1)  # 方差非零 → 不拒收

        # 第二波：g1 出生于版本 1（G=2，奖励有方差 → 不拒收）
        assert ad.dispatch(["g1"]) == "b-1"
        for seg in ("g1-s0", "g1-s1"):
            ad.prefill(seg)
            ad.decode(seg, 64)
            ad.finish(seg)
        ad.report_reward("g1", "g1-s0", 0.5)
        ad.report_reward("g1", "g1-s1", 0.7)

        ad.close()
        report = validate(path, strict=True)  # 有 error 即抛
        assert report.warnings == [], f"场景 A 应零警告：{report.warnings}"

        # 回读落盘 trace：re-prefill span 的 n_tokens 必须是 prefill_len（§7.3 冻结映射）
        spans = [
            e
            for e in read(path)
            if e["type"] == "phase_span"
            and e.get("meta", {}).get("reason") == "re-prefill"
            and e["seg_id"] == "g0-s0"
        ]
        assert len(spans) == 1 and spans[0]["n_tokens"] == 150

    def test_scenario_b_dapo_early_abort_and_exact_reject(self, tmp_path):
        """early abort（组级中止在途段）+ exact 拒收（scheduler_group_reject，W06）。"""
        path = str(tmp_path / "sched-b.rheotrace.jsonl")
        ad = TraceAdapter(path)
        ad.submit_group("g0", [100, 110], max_new=200)  # exact 路径
        ad.submit_group("g1", [90, 95, 85, 105], max_new=200)  # early 路径（ρ=0.5, G=4）
        ad.tick()

        assert ad.dispatch(["g0", "g1"]) == "b-0"
        for seg in ("g0-s0", "g0-s1", "g1-s0", "g1-s1", "g1-s2", "g1-s3"):
            ad.prefill(seg)
            ad.decode(seg, 10)

        # g0：全部完成且奖励全等 → exact 拒收（自定义事件，无 abort）
        ad.finish("g0-s0")
        ad.report_reward("g0", "g0-s0", 1.0)
        ad.finish("g0-s1")
        ad.report_reward("g0", "g0-s1", 1.0)

        # g1：2/4 完成且全等 → early 命中，中止在途两段
        ad.finish("g1-s0")
        ad.report_reward("g1", "g1-s0", 0.0)
        ad.finish("g1-s1")
        ad.report_reward("g1", "g1-s1", 0.0)

        applied = ad.run_policy_until_continue(current_version=0)
        assert [d.action for d in applied] == ["abort"]
        assert applied[0].group_id == "g1"
        assert applied[0].reason == "zero_variance_early"

        ad.close()
        report = validate(path, strict=True)
        codes = {i.rule for i in report.warnings}
        # W06：scheduler_group_reject 前向兼容；W07：本场景无 weight_sync（刻意最小化）
        assert codes == {"W06", "W07"}, f"警告码意外：{report.warnings}"

        # §6.5 对账：仅 g1 在途两段的已产出 token 记废（10×2），g0 finished 不算废
        waste = ad.sch.waste_report()
        assert waste["aborted_segments"] == 2
        assert waste["aborted_tokens"] == 20
        assert waste["by_reason"] == {"zero_variance_early": 2}
        assert waste["rejected_groups"] == ["g0"]

        # 回读拒收事件：组级废 token = 全组已生成（g0 两段各 10）
        rejects = [e for e in read(path) if e["type"] == "scheduler_group_reject"]
        assert len(rejects) == 1
        assert rejects[0]["group_id"] == "g0"
        assert rejects[0]["meta"]["n_gen_tokens_group"] == 20
