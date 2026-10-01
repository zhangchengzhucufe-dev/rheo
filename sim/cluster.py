"""集群参数化（TASK-S2 #1）：仿真侧唯一的多Free参数面，拟合值由 sim.calibrate 产出。

设计立场（PLAN 迭代 6 的教训：自由参数太多没法标定）——
v0 吞吐模型只有**一个**核心参数：每 GPU 聚合解码吞吐（tok/s/GPU）。
连续批处理下单段速率远低于聚合速率（真实 m0 trace：单段 ~6 tok/s，
8 GPU 聚合 ~14 tok/s/GPU），以聚合口径建模、用真实 trace 拟合，避免发明不可测的批调度细节。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ClusterConfig:
    """虚拟集群。时间单位秒，token 数为 spec §4 的 n_tokens 口径。"""

    n_gpu: int = 8
    # 每 GPU 聚合吞吐：批内并发共享该速率预算（连续批处理的有效值，非单段速率）
    decode_tok_per_s: float = 14.0
    # 每 GPU 预填充吞吐；None = prefill 折入 decode 口径（m0 trace 无 prefill span 时的默认）
    prefill_tok_per_s: float | None = None
    # 每步权重同步停顿（秒）。None = 用 workload 自带的每步实测值
    weight_sync_s: float | None = None
    # 步首调度间隙（秒）。None = 用 workload 自带的每步实测值
    schedule_gap_s: float | None = None
    # 确定性种子：回放模式本身无随机性，供后续策略 harness（错峰抖动等）使用
    seed: int = 0
    meta: dict = field(default_factory=dict)

    def decode_s(self, n_tokens: int | float) -> float:
        return n_tokens / self.decode_tok_per_s

    def prefill_s(self, n_tokens: int | float) -> float:
        if self.prefill_tok_per_s is None:
            return 0.0
        return n_tokens / self.prefill_tok_per_s
