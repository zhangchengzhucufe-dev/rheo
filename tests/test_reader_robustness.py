"""第八轮复查：gzip 截断/损坏分层处理 + BOM/CRLF 真实世界文件形态。"""

import json

import rheotrace
from rheotrace.core import RheotraceError


def _good_events():
    return rheotrace.generate(preset="grpo", seed=0)


def _write_gz(path, events):
    rheotrace.write(path, events)
    return open(path, "rb").read()


def test_truncated_gz_is_warning_with_partial_content(tmp_path):
    """gzip 流截断 = 写入中途崩溃：validate 记 W01 并保留截断点前的内容。"""
    p = tmp_path / "t.jsonl.gz"
    raw = _write_gz(p, _good_events())
    p.write_bytes(raw[: len(raw) // 2])
    rep = rheotrace.validate(p, strict=False)
    assert rep.ok and any(w.rule == "W01" for w in rep.warnings)
    partial = rheotrace.read(p, skip_truncated=True)
    assert 0 < len(partial) < len(_good_events())


def test_truncated_gz_read_raises_by_default(tmp_path):
    p = tmp_path / "t.jsonl.gz"
    raw = _write_gz(p, _good_events())
    p.write_bytes(raw[: len(raw) // 2])
    try:
        rheotrace.read(p)
        raise AssertionError("默认应抛 RheotraceError")
    except RheotraceError as e:
        assert "TRUNCATED" in str(e)


def test_corrupt_gz_is_rejected_even_with_skip_truncated(tmp_path):
    """结构性损坏不是干净的截断：skip_truncated 也不吞。"""
    p = tmp_path / "t.jsonl.gz"
    raw = bytearray(_write_gz(p, _good_events()))
    mid = len(raw) // 2
    raw[mid] ^= 0xFF
    raw[mid + 1] ^= 0xFF
    p.write_bytes(bytes(raw))
    rep = rheotrace.validate(p, strict=False)
    assert not rep.ok
    try:
        rheotrace.read(p, skip_truncated=True)
        raise AssertionError("损坏 gz 应抛")
    except RheotraceError:
        pass


def test_utf8_bom_tolerated(tmp_path):
    """PowerShell 重定向/记事本产物带 BOM：读端容忍，写出端永不产生 BOM。"""
    p = tmp_path / "bom.jsonl"
    with open(p, "wb") as f:
        f.write(b"\xef\xbb\xbf")
        for e in _good_events():
            f.write(json.dumps(e).encode() + b"\n")
    rep = rheotrace.validate(p, strict=False)
    assert rep.ok and not rep.warnings
    assert rheotrace.read(p) == _good_events()


def test_crlf_line_endings_tolerated(tmp_path):
    p = tmp_path / "crlf.jsonl"
    with open(p, "wb") as f:
        for e in _good_events():
            f.write(json.dumps(e).encode() + b"\r\n")
    rep = rheotrace.validate(p, strict=False)
    assert rep.ok and not rep.warnings
    assert rheotrace.read(p) == _good_events()


def test_corrupt_last_line_with_newline_is_error_not_warning(tmp_path):
    """带换行的损坏末行 = 完整写入的坏数据（E02），不得因'恰好是末行'降级为 W01。"""
    p = tmp_path / "g.jsonl"
    lines = [rheotrace.core.dumps_line(e) for e in _good_events()]
    lines[-1] = "garbage\n"
    p.write_text("\n".join(lines))
    rep = rheotrace.validate(p, strict=False)
    assert any(x.rule == "E02" for x in rep.errors)
    try:
        rheotrace.read(p, skip_truncated=True)
        raise AssertionError("带换行的损坏行不该被 skip_truncated 吞掉")
    except RheotraceError:
        pass
