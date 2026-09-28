"""trace 格式适配层：外部格式 → canon。

这是**唯一的**格式接缝。A 的真实 trace（RheoTrace spec v0，B 负责）到位后，
只需在这里加一个 spec→canon 的 reader，metrics / report 不动（TASK-C 验收）。

当前支持：
- rheo-canon-v0（C 自有的 JSONL 规范格式，测试与 B spec 定稿前的联调用）
"""

from __future__ import annotations

import json
from pathlib import Path

from . import canon
from .canon import Trace, TraceError

__all__ = ["read_trace", "TraceError"]


def read_trace(path: str | Path) -> Trace:
    """读入 trace 文件并转为 canon。格式按 header 的 format 字段分发。"""
    path = Path(path)
    if not path.is_file():
        raise TraceError(f"trace 文件不存在：{path}")
    head = path.read_bytes().splitlines()[0] if path.stat().st_size else b""
    try:
        first = json.loads(head.decode("utf-8"))
        fmt = first.get("format") if isinstance(first, dict) else None
    except (json.JSONDecodeError, UnicodeDecodeError):
        fmt = None

    if fmt == canon.CANON_FORMAT:
        return canon.read_canon(path)
    raise TraceError(
        f"不认识的 trace 格式（format={fmt!r}）：{path}。"
        "RheoTrace spec v0 的适配器尚未实现——spec 定稿后在 bench/analysis/adapters.py 加 reader"
    )
