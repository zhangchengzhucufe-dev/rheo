"""RheoTrace v0 核心常量、错误类型与工具函数。

格式规格见 docs/rheotrace-spec-v0.md；本模块是格式的机器可读注册表，
validator 与生成器都从这里取常量，保证与规格单一来源。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

FORMAT_NAME = "rheotrace-jsonl"
FORMAT_VERSION = 0

RUN_START = "run_start"
RUN_END = "run_end"
WEIGHT_SYNC = "weight_sync"
PHASE_SPAN = "phase_span"
SEGMENT_START = "segment_start"
SEGMENT_STATE = "segment_state"
SEGMENT_END = "segment_end"
TOKEN_LOGPROB = "token_logprob"

EVENT_TYPES = frozenset(
    {
        RUN_START,
        RUN_END,
        WEIGHT_SYNC,
        PHASE_SPAN,
        SEGMENT_START,
        SEGMENT_STATE,
        SEGMENT_END,
        TOKEN_LOGPROB,
    }
)

SEGMENT_STATES = frozenset({"running", "paused", "env_wait", "finished", "aborted"})
TRANSIENT_STATES = frozenset({"running", "paused", "env_wait"})
TERMINAL_STATES = frozenset({"finished", "aborted"})

# 轨迹段状态机（规格 §2 表），segment_state 只承载暂态间的转换；终态一律走 segment_end
LEGAL_TRANSIENT: dict[str, frozenset[str]] = {
    "running": frozenset({"paused", "env_wait"}),
    "paused": frozenset({"running"}),
    "env_wait": frozenset({"running"}),
}

PHASES = frozenset({"prefill", "decode", "env_wait", "schedule"})
ENGINE_LEVEL_PHASES = frozenset({"schedule"})  # 不允许带 seg_id 的 phase
SEGMENT_LEVEL_PHASES = PHASES - ENGINE_LEVEL_PHASES
SYNC_MODES = frozenset({"full", "delta", "load"})
FINISH_MODES = frozenset({"exact", "shadow", "stale"})
CLOCKS = frozenset({"wall_ns_epoch", "mono_ns_raw"})

# 规格同步单调（§4.0）：区间型事件以结束时刻为信封 ts
INTERVAL_END_TYPES = frozenset({WEIGHT_SYNC, PHASE_SPAN, SEGMENT_END})

MS_NS = 1_000_000


class RheotraceError(Exception):
    """rheotrace 基类错误。"""


@dataclass
class Issue:
    """一条校验发现（error 或 warning）。"""

    rule: str
    message: str
    line: int | None = None

    def __str__(self) -> str:
        loc = f"line {self.line}: " if self.line is not None else ""
        return f"[{self.rule}] {loc}{self.message}"


@dataclass
class ValidationReport:
    """校验报告：errors 非空即拒绝文件。"""

    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, rule: str, message: str, line: int | None = None) -> None:
        self.errors.append(Issue(rule, message, line))

    def add_warning(self, rule: str, message: str, line: int | None = None) -> None:
        self.warnings.append(Issue(rule, message, line))

    def raise_if_errors(self) -> None:
        if self.errors:
            raise ValidationError(self)

    def __str__(self) -> str:
        head = "OK" if self.ok else "REJECTED"
        return f"{head}: {len(self.errors)} errors, {len(self.warnings)} warnings"


class ValidationError(RheotraceError):
    """校验拒绝；携带完整报告。"""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        super().__init__(str(report))


def now_ns() -> int:
    """默认时钟：wall clock 纳秒（epoch）。"""
    return time.time_ns()


def dumps_line(event: dict) -> str:
    """单行序列化：紧凑、非 ASCII 保留、禁止 NaN/Inf。"""
    return json.dumps(event, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
