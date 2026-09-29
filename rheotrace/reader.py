"""RheoTrace 读取：无损事件流 reader，按后缀透明解压 gzip。

损坏/截断策略（与规格 §6 W01/E02 对齐）：
- gzip 流提前结束（EOFError）= 写入中途崩溃的截断 → validate 记 W01；
  iread 默认抛 RheotraceError，skip_truncated=True 时静默停止
- gzip 流损坏（CRC/结构错）→ validate 记 E02；iread 恒抛 RheotraceError
- 末行 JSON 残缺（无换行的截断残行）→ validate 记 W01；iread 同上
- UTF-8 BOM 容忍（PowerShell 重定向/记事本产物），写出端永不产生 BOM
"""

from __future__ import annotations

import gzip
import json
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .core import RheotraceError


def open_text(path: str | Path) -> Any:
    """打开 trace 文本句柄；.gz 后缀（不区分大小写）自动解压。utf-8-sig 兼容带 BOM 文件。"""
    p = Path(path)
    if p.suffix.lower() == ".gz":
        return gzip.open(p, "rt", encoding="utf-8-sig")
    return p.open("r", encoding="utf-8-sig")


def _readline_lenient(fh: Any) -> str:
    """读一行，把 gzip 层的截断/损坏翻译成领域异常。

    - EOFError（流提前结束）→ RheotraceError("TRUNCATED: ...")，由调用方按策略处理
    - BadGzipFile / zlib.error（CRC 或结构损坏）→ RheotraceError("CORRUPT: ...")，恒为错误
    """
    try:
        return fh.readline()
    except EOFError as e:
        raise RheotraceError("TRUNCATED: gzip 流提前结束，文件截断") from e
    except (gzip.BadGzipFile, zlib.error) as e:
        raise RheotraceError(f"CORRUPT: gzip 流损坏（{e}）") from e


def iread(path: str | Path, *, skip_truncated: bool = False) -> Iterator[dict]:
    """逐事件流式读取。

    skip_truncated=True 时，仅当损坏是截断形态（末行 JSON 残缺 / gzip 流提前结束）
    才静默停止；gzip 层的结构性损坏（CORRUPT）恒抛 RheotraceError。
    """
    with open_text(path) as fh:
        lineno = 0
        while True:
            try:
                line = _readline_lenient(fh)
            except RheotraceError as e:
                if skip_truncated and str(e).startswith("TRUNCATED"):
                    return
                raise
            if not line:
                return
            lineno += 1
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError:
                try:
                    rest = _readline_lenient(fh)
                except RheotraceError as e:
                    if str(e).startswith("TRUNCATED") and skip_truncated:
                        return
                    raise
                if rest.strip():
                    raise RheotraceError(f"第 {lineno} 行 JSON 解析失败") from None
                if skip_truncated:
                    return
                raise RheotraceError(
                    f"第 {lineno} 行 JSON 残缺且无后续内容（末尾截断）；"
                    "如需容忍请用 skip_truncated=True 或 validate 的宽松模式"
                ) from None


def read(path: str | Path, *, skip_truncated: bool = False) -> list[dict]:
    """读入全部事件，无损还原（dict 逐项相等，浮点精确）。"""
    return list(iread(path, skip_truncated=skip_truncated))
