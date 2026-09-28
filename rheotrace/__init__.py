"""RheoTrace v0：RL rollout 轨迹追踪格式的写入 / 读取 / 校验。

格式规格见 docs/rheotrace-spec-v0.md（M1 冻结接口，A 插桩与 C 分析均以此为准）：

- ``write`` / ``TraceWriter``：事件流落盘（.gz 透明压缩；writer 自动补 ts/run_id/run_start/run_end）
- ``read`` / ``iread``：无损读取（逐事件 dict 相等，浮点精确）
- ``validate``：规格 §6 规则表（E/W）校验，strict 模式拒绝坏文件
- ``generate`` / ``generate_file``：确定性合成 trace（grpo / bimodal / agent 预置负载）
"""

from .core import (
    CLOCKS,
    ENGINE_LEVEL_PHASES,
    FINISH_MODES,
    FORMAT_NAME,
    FORMAT_VERSION,
    PHASES,
    SEGMENT_STATES,
    SYNC_MODES,
    RheotraceError,
    ValidationError,
    ValidationReport,
)
from .gen import PRESETS, generate, generate_file
from .reader import iread, read
from .validate import validate
from .writer import TraceWriter, write

__version__ = "0.0.1"

__all__ = [
    "CLOCKS",
    "ENGINE_LEVEL_PHASES",
    "FINISH_MODES",
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "PHASES",
    "PRESETS",
    "SEGMENT_STATES",
    "SYNC_MODES",
    "RheotraceError",
    "TraceWriter",
    "ValidationReport",
    "ValidationError",
    "generate",
    "generate_file",
    "iread",
    "read",
    "validate",
    "write",
    "__version__",
]
