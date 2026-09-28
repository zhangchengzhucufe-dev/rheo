"""RheoTrace 写入：低层一次性 write() 与插桩用 TraceWriter。"""

from __future__ import annotations

import gzip
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from uuid import uuid4

from .core import FORMAT_NAME, FORMAT_VERSION, RUN_END, RUN_START, dumps_line, now_ns


def _open_sink(path: str | Path) -> Any:
    """按后缀透明 gzip；返回带 close() 的文本句柄。"""
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, "wt", encoding="utf-8")
    return p.open("w", encoding="utf-8")


def write(path: str | Path, events: Iterable[dict]) -> int:
    """把事件序列原样落盘：纯序列化，不补字段、不校验（坏样本也要能写出来给 validator 测）。

    path 以 .gz 结尾时自动压缩。返回写入条数。
    """
    n = 0
    with _open_sink(path) as fh:
        for ev in events:
            fh.write(dumps_line(ev) + "\n")
            n += 1
    return n


class TraceWriter:
    """流式 writer：自动补 ts / run_id，自动落 run_start / run_end。插桩方（会话 A）直接用这个。

    with TraceWriter(path, engine=..., model=...) as w:
        w.emit("segment_start", seg_id=..., group_id=..., birth_version=..., t_start=..., ...)
    """

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str | None = None,
        engine: str = "unknown",
        model: str = "unknown",
        initial_version: int = 0,
        clock: str = "wall_ns_epoch",
        meta: dict | None = None,
    ) -> None:
        self.path = Path(path)
        self.run_id = run_id or f"r-{uuid4().hex[:8]}"
        self._fh = _open_sink(path)
        self._counts: dict[str, int] = {}
        self._closed = False
        self.emit(
            RUN_START,
            format=FORMAT_NAME,
            schema_version=FORMAT_VERSION,
            initial_version=initial_version,
            engine=engine,
            model=model,
            clock=clock,
            meta=dict(meta or {}),
        )

    def emit(self, type: str, **fields: Any) -> dict:
        """发一条事件：补 type / run_id / ts（调用方显式给了 ts 则尊重）。"""
        ev: dict[str, Any] = dict(fields)
        ev["type"] = type
        ev["run_id"] = self.run_id
        ev.setdefault("ts", now_ns())
        return self._write(ev)

    def emit_raw(self, event: dict) -> dict:
        """发一条已构造好的事件 dict：只补缺失的 run_id / ts，其余原样保留。"""
        ev = dict(event)
        ev.setdefault("run_id", self.run_id)
        ev.setdefault("ts", now_ns())
        return self._write(ev)

    def _write(self, ev: dict) -> dict:
        self._fh.write(dumps_line(ev) + "\n")
        self._counts[ev["type"]] = self._counts.get(ev["type"], 0) + 1
        return ev

    def close(self) -> None:
        """落 run_end（带按类型计数）并关文件；幂等。"""
        if self._closed:
            return
        self._closed = True
        self.emit(RUN_END, summary={"events_by_type": dict(sorted(self._counts.items()))})
        self._fh.close()

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
