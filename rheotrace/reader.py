"""RheoTrace 读取：无损事件流 reader，按后缀透明解压 gzip。"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .core import RheotraceError


def open_text(path: str | Path) -> Any:
    """打开 trace 文本句柄；.gz 后缀（不区分大小写）自动解压。"""
    p = Path(path)
    if p.suffix.lower() == ".gz":
        return gzip.open(p, "rt", encoding="utf-8")
    return p.open("r", encoding="utf-8")


def iread(path: str | Path, *, skip_truncated: bool = False) -> Iterator[dict]:
    """逐事件流式读取。

    skip_truncated=True 时，仅当损坏行是文件最后一行（写入中途崩溃的残行）才静默跳过；
    中间行损坏一律抛 RheotraceError。
    """
    with open_text(path) as fh:
        lineno = 0
        while True:
            line = fh.readline()
            if not line:
                return
            lineno += 1
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError:
                rest = fh.read()
                if skip_truncated and not rest.strip():
                    return
                raise RheotraceError(f"第 {lineno} 行 JSON 解析失败") from None


def read(path: str | Path, *, skip_truncated: bool = False) -> list[dict]:
    """读入全部事件，无损还原（dict 逐项相等，浮点精确）。"""
    return list(iread(path, skip_truncated=skip_truncated))
