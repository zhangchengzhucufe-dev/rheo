"""校准（TASK-S2 验收 #1）：用真实 trace 拟合 ClusterConfig，量化回放墙钟误差。

方法：真实 trace（m0 pilot，8 GPU，verl+vLLM）的调度原子是版本窗口；
每窗口 gen tokens / (窗口生成期 × n_gpu) = 每 GPU 聚合吞吐的窗口级估计。
窗口吞吐在真实 trace 里呈双模（干净窗口高速档、混入验证负载的窗口低一个量级），
拟合在干净窗口上做；误差分三个口径报告（见 Calibration.report）：
  1. replay fidelity——逐窗口实测率重放，分离引擎机制误差（尾部量化、分配不均）；
  2. policy-grade 干净窗口——单一拟合率的生成期误差（策略对比时的可用精度）；
  3. policy-grade 全 run——含 eval 窗的未建模验证负载，说明单率模型的适用边界。
"""

from __future__ import annotations

import statistics as st
from dataclasses import dataclass, field
from pathlib import Path

from .cluster import ClusterConfig
from .engine import SimResult, run_step_barrier
from .workload import Workload, load_workload


@dataclass
class Calibration:
    config: ClusterConfig
    workload: Workload
    fitted_decode_tok_per_s: float
    per_step_rate: dict[int, float]  # 窗口 → 每 GPU 吞吐的窗口级估计
    clean_windows: set[int] = field(default_factory=set)  # 未被 eval 等污染的窗口
    sim: SimResult | None = None  # 全局率回放（策略级口径）
    sim_per_window: SimResult | None = None  # 逐窗口实测率回放（引擎机制误差上界）

    def _wall(
        self, sim: SimResult, windows: set[int] | None = None, *, with_sync: bool = False
    ) -> tuple[float, float]:
        """（sim, real）墙钟对，秒。默认只算生成期（gap+span，不含 sync）——
        真实 real_span_s 的口径即不含 sync；with_sync=True 时两侧都加 sync。"""
        sim_t = real_t = 0.0
        for step, rec in zip(self.workload.steps, sim.steps, strict=True):
            if windows is not None and step.index not in windows:
                continue
            if step.real_span_s is None or step.meta.get("truncated"):
                continue  # 截断窗口：sim 跑完 vs real 被切，不可比
            sim_t += rec.gap_s + rec.span_s + (rec.sync_s if with_sync else 0.0)
            real_sync = (step.sync_s or 0.0) if with_sync else 0.0
            real_t += step.real_span_s + real_sync
        return sim_t, real_t

    def report(self) -> str:
        """校准报告（docs/sim-study-v0.md 的数字全部出自这里）。"""
        lines = [
            f"workload: {self.workload.n_segments} segments, "
            f"{self.workload.gen_tokens} gen tokens, {len(self.workload.steps)} windows, "
            f"n_gpu={self.workload.n_gpu}",
            f"fitted decode throughput: {self.fitted_decode_tok_per_s:.2f} tok/s/GPU "
            f"({len(self.clean_windows)}/{len(self.per_step_rate)} clean windows; "
            f"eval-flagged: {sorted(set(self.per_step_rate) - self.clean_windows)})",
        ]
        if self.sim_per_window is not None:
            s, r = self._wall(self.sim_per_window)
            lines.append(
                f"replay fidelity (per-window fitted rate), gen-phase wall: "
                f"sim {s:.0f}s vs real {r:.0f}s ({(s - r) / r:+.1%})"
            )
        if self.sim is not None:
            s, r = self._wall(self.sim, self.clean_windows)
            lines.append(
                f"policy-grade (single fitted rate), clean windows, gen-phase: "
                f"sim {s:.0f}s vs real {r:.0f}s ({(s - r) / r:+.1%})"
            )
            s, r = self._wall(self.sim, with_sync=True)
            lines.append(
                f"policy-grade, full run (incl. sync): sim {s:.0f}s vs real {r:.0f}s "
                f"({(s - r) / r:+.1%}; 差额主体 = eval 窗的未建模验证负载)"
            )
        return "\n".join(lines)

    def summary(self) -> str:
        return self.report()


def fit(workload: Workload) -> Calibration:
    """从 workload（含 real 观测）拟合最小参数集。

    干净窗口 = 窗口速率 ≥ 0.75 × 中位速率（启发式；eval/验证污染窗远低于阈值）。
    拟合率 = 干净窗口的 Σtokens / Σ(生成期 × n_gpu)（调和均值，直接对齐生成期墙钟）。
    """
    rates: dict[int, float] = {}
    for step in workload.steps:
        if step.meta.get("truncated"):
            continue  # 截断窗口的真实跨度不完整，速率估计不可信
        tokens = sum(j.n_gen_tokens for j in step.jobs)
        if step.real_span_s and step.real_span_s > 0 and tokens > 0:
            rates[step.index] = tokens / (step.real_span_s * workload.n_gpu)
    clean: set[int] = set()
    if not rates:
        rate = 14.0
    else:
        med = st.median(rates.values())
        clean = {k for k, v in rates.items() if v >= 0.75 * med}
        num = sum(sum(j.n_gen_tokens for j in s.jobs) for s in workload.steps if s.index in clean)
        den = sum(
            (s.real_span_s or 0.0) * workload.n_gpu for s in workload.steps if s.index in clean
        )
        rate = num / den if den > 0 else med
    syncs = [s.sync_s for s in workload.steps if s.sync_s is not None]
    gaps = [s.gap_s for s in workload.steps if s.gap_s is not None]
    cfg = ClusterConfig(
        n_gpu=workload.n_gpu,
        decode_tok_per_s=rate,
        weight_sync_s=st.median(syncs) if syncs else 0.0,
        schedule_gap_s=st.median(gaps) if gaps else 0.0,
    )
    return Calibration(
        config=cfg,
        workload=workload,
        fitted_decode_tok_per_s=rate,
        per_step_rate=rates,
        clean_windows=clean,
    )


def calibrate(trace_path: str | Path, *, run_sim: bool = True) -> Calibration:
    """端到端：读真实 trace → 拟合 → 双口径回放 → 误差量化。"""
    wl = load_workload(trace_path)
    cal = fit(wl)
    if run_sim:
        cal.sim = run_step_barrier(wl, cal.config)
        cal.sim_per_window = run_step_barrier(wl, cal.config, rate_by_step=cal.per_step_rate)
    return cal
