"""Scheduler seeded fuzz：随机操作序列下的不变量（对齐 tests/test_validator_fuzz.py 先例）。

确定性：random.Random(seed)，同 seed 同序列。每步断言：
- 状态机合法（非法操作以 SchedulerError 拒绝，其他异常即失败）；
- observation 与内部状态一致（在途段过滤、组视图计数守恒）；
- waste_report 与驱动方跟踪的期望严格相等；
- V1Policy 决策循环有限终止，且每条决策过得了 validate_decision（产出→校验往返）；
- on_sync 产出的每条决策同样过得了校验（D2 路径的往返）。

咬合力经变异测试验证：废 token 记账漂移 38/40 种子被抓住、advance 计数漂移 24/40；
守卫子句类变异（如去掉 advance 的 running 检查）对合法操作驱动器不可见，由单测覆盖
（test_advance_requires_running_state）——fuzz 管合法交织下的一致性，单测管契约拒绝。
"""

import random

import pytest

from runtime.scheduler import (
    Decision,
    Scheduler,
    SchedulerConfig,
    SchedulerError,
    V1Policy,
    validate_decision,
)

SEEDS = range(40)
STEPS_PER_SEED = 80


class Driver:
    """随机操作驱动器：维护期望对账账本，驱动 Scheduler 走过全部决策点。"""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.sch = Scheduler(SchedulerConfig(early_abort_rho=0.5))
        self.policy = V1Policy(SchedulerConfig(early_abort_rho=0.5))
        self.version = 0
        self.group_seq = 0
        self.live: dict[str, dict] = {}  # group_id → {"seg_ids", "max_new", "prompt"}
        self.expected_aborted_tokens = 0
        self.expected_aborted_segments = 0
        self.expected_rejected: list[str] = []

    # -- 不变量 ----------------------------------------------------------------

    def check_invariants(self) -> None:
        obs = self.sch.observation(self.version, 0, 10**9)
        for seg in obs.segments:
            assert seg.state in ("running", "paused", "env_wait")
            assert seg.n_gen_tokens >= 0
            if seg.max_new_tokens is not None:
                assert seg.n_gen_tokens <= seg.max_new_tokens
        for group in obs.groups:
            total = (
                group.n_queued
                + group.n_running
                + group.n_paused
                + group.n_env_wait
                + group.n_finished
                + group.n_aborted
            )
            assert total == group.n_total
            # 奖励序与驱动器账本逐位一致（按完成序回填）
            assert group.finished_rewards == tuple(self.live[group.group_id]["rewards"].values())
        waste = self.sch.waste_report()
        assert waste["aborted_tokens"] == self.expected_aborted_tokens
        assert waste["aborted_segments"] == self.expected_aborted_segments
        assert waste["rejected_groups"] == self.expected_rejected

    def run_policy_loop(self, *, pending_sync: bool = False) -> list[Decision]:
        """决策→校验→应用循环：有限终止 + 每条决策过校验。"""
        applied: list[Decision] = []
        for _ in range(64):
            obs = self.sch.observation(self.version, 0, 10**9, pending_sync=pending_sync)
            d = self.policy(obs)
            validate_decision(obs, d)
            if d.action == "continue":
                break
            if d.action == "abort":
                self.expect_abort(d.group_id)
            self.sch.apply(d, current_version=self.version)
            applied.append(d)
        else:
            raise AssertionError("V1Policy 循环 64 步未终止")
        return applied

    def expect_abort(self, group_id: str) -> None:
        for gid, info in self.live.items():
            if gid != group_id:
                continue
            for seg_id, st in info["seg_states"].items():
                if st not in ("finished", "aborted"):
                    self.expected_aborted_segments += 1
                    self.expected_aborted_tokens += info["gen"][seg_id]
                    info["seg_states"][seg_id] = "aborted"

    # -- 随机操作 ---------------------------------------------------------------

    def op_submit(self) -> None:
        gid = f"g{self.group_seq}"
        self.group_seq += 1
        g = self.rng.randint(1, 4)
        segs = [f"{gid}-s{i}" for i in range(g)]
        self.sch.register_group(gid, segs)
        for s in segs:
            self.sch.register_segment(
                s, gid, n_prompt_tokens=self.rng.randint(1, 500), max_new_tokens=200
            )
        self.live[gid] = {
            "seg_ids": segs,
            "seg_states": {s: "queued" for s in segs},
            "gen": {s: 0 for s in segs},
            "rewards": {},
        }

    def groups_with(self, pred) -> list[str]:
        return [
            gid
            for gid, info in self.live.items()
            if pred(info) and gid not in self.expected_rejected
        ]

    def op_dispatch(self) -> None:
        candidates = self.sch.pending_groups()
        if not candidates:
            return
        budget = self.rng.choice([10**9, 10**9, 300, 800])
        plan = self.sch.plan_batch(
            candidates[: self.rng.randint(1, len(candidates))],
            budget,
            birth_version=self.version,
        )
        if plan is None:
            return
        for seg_id in plan.seg_ids:
            gid = next(g for g, i in self.live.items() if seg_id in i["seg_ids"])
            self.live[gid]["seg_states"][seg_id] = "running"

    def op_advance(self) -> None:
        running = [
            (gid, s)
            for gid, i in self.live.items()
            for s in i["seg_ids"]
            if i["seg_states"][s] == "running"
        ]
        if not running:
            return
        gid, seg_id = self.rng.choice(running)
        k = self.rng.randint(1, 30)
        self.sch.advance(seg_id, k)
        self.live[gid]["gen"][seg_id] += k

    def op_finish_and_reward(self) -> None:
        running = self.groups_with(
            lambda i: any(st == "running" for st in i["seg_states"].values())
        )
        if not running:
            return
        gid = self.rng.choice(running)
        info = self.live[gid]
        seg_id = self.rng.choice([s for s, st in info["seg_states"].items() if st == "running"])
        self.sch.finish(seg_id)
        info["seg_states"][seg_id] = "finished"
        # 奖励：一半概率与已有奖励相同（触发 exact/early 路径）
        existing = list(info["rewards"].values())
        reward = (
            existing[0] if existing and self.rng.random() < 0.5 else round(self.rng.random(), 2)
        )
        info["rewards"][seg_id] = reward
        view = self.sch.report_reward(gid, seg_id, reward)
        if view is not None:
            assert gid not in self.expected_rejected
            self.expected_rejected.append(gid)

    def op_sync(self) -> None:
        self.version += 1
        self.sch.apply(Decision("pause", targets=()))
        # 安全点：驱动 V1Policy 把 running 段全部 pause（模拟 D2 强制到界）
        obs = self.sch.observation(self.version, 0, 10**9, pending_sync=True)
        d = self.policy(obs)
        validate_decision(obs, d)
        if d.action == "pause":
            for seg_id in d.targets:
                gid = next(g for g, i in self.live.items() if seg_id in i["seg_ids"])
                self.live[gid]["seg_states"][seg_id] = "paused"
            self.sch.apply(d)
        self.run_policy_loop(pending_sync=True)
        # 指针翻转 → 新版本生效 → 续跑判定（re-prefill 决策的往返校验）
        for decision in self.sch.on_sync(self.version):
            post = self.sch.observation(self.version, 0, 10**9)
            validate_decision(post, decision)
            if decision.action == "abort":
                self.expect_abort(decision.group_id)
            self.sch.apply(decision, current_version=self.version)
        for info in self.live.values():
            for seg_id, st in info["seg_states"].items():
                if st == "paused" and self.sch._segs[seg_id].state == "running":
                    info["seg_states"][seg_id] = "running"
                elif st == "running" and self.sch._segs[seg_id].state == "paused":
                    info["seg_states"][seg_id] = "paused"
        self.run_policy_loop()

    def op_abort_random(self) -> None:
        # 直接对一个有在途/排队段的组发 abort（V1Policy 之外的路径）
        candidates = [
            gid
            for gid, i in self.live.items()
            if any(st not in ("finished", "aborted") for st in i["seg_states"].values())
        ]
        if not candidates:
            return
        gid = self.rng.choice(candidates)
        d = Decision(
            "abort", group_id=gid, reason=self.rng.choice(["weight_skip", "policy_preempt"])
        )
        obs = self.sch.observation(self.version, 0, 10**9)
        if any(g.group_id == gid for g in obs.groups):
            validate_decision(obs, d)
        self.expect_abort(gid)
        self.sch.apply(d)

    def step(self) -> None:
        op = self.rng.choices(
            [
                self.op_submit,
                self.op_dispatch,
                self.op_advance,
                self.op_finish_and_reward,
                self.op_sync,
                self.op_abort_random,
            ],
            weights=[3, 3, 6, 4, 2, 1],
        )[0]
        try:
            op()
            self.run_policy_loop()
            self.check_invariants()
        except SchedulerError:
            # 契约拒绝是合法结果（如对已收尾组 abort）；状态必须仍然自洽
            self.check_invariants()


@pytest.mark.parametrize("seed", SEEDS)
def test_scheduler_fuzz_invariants(seed):
    rng = random.Random(seed)
    driver = Driver(rng)
    for _ in range(STEPS_PER_SEED):
        driver.step()
