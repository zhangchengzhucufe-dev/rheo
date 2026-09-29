"""RheoTrace 写入：低层一次性 write() 与插桩用 TraceWriter。"""

from __future__ import annotations

import gzip
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from uuid import uuid4

from .core import (
    FORMAT_NAME,
    FORMAT_VERSION,
    INTERVAL_END_TYPES,
    RUN_END,
    RUN_START,
    RheotraceError,
    dumps_line,
    now_ns,
)


def _open_sink(path: str | Path) -> Any:
    """按后缀透明 gzip（不区分大小写）；返回带 close() 的文本句柄。"""
    p = Path(path)
    if p.suffix.lower() == ".gz":
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

    时钟约定：自动补的 ts 是墙钟（now_ns），因此调用方自带的时间戳必须与之同源；
    事件 ts 早于本 run 起始时刻立即报错——否则乱序要到 validator 才暴露，离病因很远。
    显式安排时间戳时以 ``w.run_start_ts`` 为基准，且**同一文件内全部事件显式传 ts**
    （auto-now 与显式未来/过去时刻混流必乱序），close 时用 ``close(end_ts=...)`` 收尾。
    虚拟时钟（合成/回放）也可以改用 ``write()`` 并自带完整时间线。
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
        self.run_start_ts: int | None = None
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
        """发一条事件：补 type / run_id / ts。

        ts 自动填充规则（规格 §4.0）：区间型事件（weight_sync / phase_span / segment_end）
        默认取其 t_end；其余取当前时刻。显式传了 ts 则尊重调用方。
        """
        ev: dict[str, Any] = dict(fields)
        ev["type"] = type
        ev["run_id"] = self.run_id
        if "ts" not in ev:
            if type in INTERVAL_END_TYPES and "t_end" in ev:
                ev["ts"] = ev["t_end"]
            else:
                ev["ts"] = now_ns()
        return self._write(ev)

    def emit_raw(self, event: dict) -> dict:
        """发一条已构造好的事件 dict：只补缺失的 run_id / ts（规则同 emit），其余原样保留。"""
        ev = dict(event)
        ev.setdefault("run_id", self.run_id)
        if "ts" not in ev and ev.get("type") in INTERVAL_END_TYPES and "t_end" in ev:
            ev["ts"] = ev["t_end"]
        ev.setdefault("ts", now_ns())
        return self._write(ev)

    def _write(self, ev: dict) -> dict:
        if self._closed:
            raise RheotraceError("TraceWriter 已关闭，不能再写入事件")
        if self.run_start_ts is None:
            self.run_start_ts = ev["ts"]  # 第一个事件即 run_start
        elif ev["ts"] < self.run_start_ts:
            raise RheotraceError(
                f"事件 {ev['type']} ts={ev['ts']} 早于本 run 起始时刻 {self.run_start_ts}："
                "TraceWriter 以墙钟补 ts，调用方时间戳必须与之同源；"
                "虚拟时钟请改用 write() 并自带完整时间线"
            )
        self._fh.write(dumps_line(ev) + "\n")
        self._counts[ev["type"]] = self._counts.get(ev["type"], 0) + 1
        return ev

    def close(self, *, end_ts: int | None = None) -> None:
        """落 run_end（带按类型计数）并关文件；幂等。

        end_ts：显式指定 run_end 时刻。手写时间线（显式 ts 的事件流）必须用它——
        默认 auto-now 会早于此前显式的未来时刻，产生 E04。
        """
        if self._closed:
            return
        summary = {"events_by_type": dict(sorted(self._counts.items()))}
        if end_ts is None:
            self.emit(RUN_END, summary=summary)
        else:
            self.emit(RUN_END, ts=end_ts, summary=summary)
        self._closed = True  # 置位须在 run_end 落盘之后，否则被自己的写入守卫拦下
        self._fh.close()

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
