"""规范事件模型（canon v0）+ JSONL 解析与校验。

C 侧内部表示：所有指标只对着这里的 canon 算。B 的 RheoTrace spec 定稿后，
在 adapters.py 里加一个 spec→canon 的映射即可，metrics 不动（TASK-C 验收口径）。

事件语义（对应 docs/metrics-v0.md §6）：
- header      ：trace 元数据，必须且只能是第一行
- exec        ：一段 GPU 执行区间（prefill 或 decode），左闭右开 [t0, t1)，ns
- env_wait    ：单条轨迹的 env 等待区间
- sync        ：权重同步的阻塞区间（传输+应用+翻转全程）
- ver_bump    ：权重版本推进（时间点）；版本号必须严格递增
- traj_end    ：轨迹终止（finished / aborted）；无此事件的轨迹视为未完成

时间基准：单 trace 单一单调时钟、ns 整数、区间左闭右开。
事件按主时间（区间的 t0 / 点事件的 t）非降序排列，乱序即拒。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

CANON_FORMAT = "rheo-canon-v0"

PHASE_PREFILL = "prefill"
PHASE_DECODE = "decode"


class TraceError(ValueError):
    """trace 文件不合法：缺字段 / 乱序 / 版本回退等。"""


@dataclass(frozen=True)
class ModelParams:
    P: float  # 参数量
    L: int  # 层数
    d: int  # hidden size（Q 侧）


@dataclass(frozen=True)
class Header:
    format: str
    clock: str
    model: str | None
    params: ModelParams | None
    world_size: int
    peak_tflops: float | None


@dataclass(frozen=True)
class Exec:
    traj: str
    batch: str
    phase: str  # prefill | decode
    t0: int
    t1: int
    bv: int  # birth_version：执行时的权重版本
    n_tok: int  # prefill: n_prompt；decode: n_gen
    committed: bool  # decode token 是否提交给训练侧
    aborted: bool


@dataclass(frozen=True)
class EnvWait:
    traj: str
    t0: int
    t1: int


@dataclass(frozen=True)
class Sync:
    t0: int
    t1: int
    ver: int | None


@dataclass(frozen=True)
class VerBump:
    t: int
    ver: int


@dataclass(frozen=True)
class TrajEnd:
    traj: str
    t: int
    status: str  # finished | aborted


@dataclass(frozen=True)
class Trace:
    header: Header
    execs: tuple[Exec, ...]
    env_waits: tuple[EnvWait, ...]
    syncs: tuple[Sync, ...]
    bumps: tuple[VerBump, ...]
    traj_ends: tuple[TrajEnd, ...]
    path: Path | None = None
    sha256: str | None = None


def _require(obj: dict, key: str, line_no: int) -> object:
    if key not in obj or obj[key] is None:
        raise TraceError(f"第 {line_no} 行：缺字段 {key!r}")
    return obj[key]


def _interval(obj: dict, line_no: int) -> tuple[int, int]:
    t0 = _require(obj, "t0", line_no)
    t1 = _require(obj, "t1", line_no)
    if not isinstance(t0, int) or not isinstance(t1, int):
        raise TraceError(f"第 {line_no} 行：t0/t1 必须是整数 ns")
    if t1 < t0:
        raise TraceError(f"第 {line_no} 行：t1 < t0（区间为空或倒置）")
    return t0, t1


def _parse_header(obj: dict, line_no: int) -> Header:
    fmt = obj.get("format")
    if fmt != CANON_FORMAT:
        raise TraceError(f"第 {line_no} 行：format={fmt!r} 不是 {CANON_FORMAT}")
    clock = str(obj.get("clock", "mono_ns"))
    params = None
    if obj.get("P") is not None:
        params = ModelParams(P=float(obj["P"]), L=int(obj["L"]), d=int(obj["d"]))
    return Header(
        format=fmt,
        clock=clock,
        model=obj.get("model"),
        params=params,
        world_size=int(obj.get("world_size", 1)),
        peak_tflops=float(obj["peak_tflops"]) if obj.get("peak_tflops") is not None else None,
    )


def parse_events(lines: list[str]) -> Trace:
    """解析 JSONL 文本行列表，带完整校验。"""
    header: Header | None = None
    execs: list[Exec] = []
    env_waits: list[EnvWait] = []
    syncs: list[Sync] = []
    bumps: list[VerBump] = []
    traj_ends: list[TrajEnd] = []
    ended: dict[str, TrajEnd] = {}
    last_t: int | None = None
    last_ver = 0

    for line_no, raw in enumerate(lines, start=1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            raise TraceError(f"第 {line_no} 行：JSON 解析失败（{e}）") from e
        if not isinstance(obj, dict):
            raise TraceError(f"第 {line_no} 行：事件必须是 JSON 对象")
        ev = obj.get("ev")
        if ev == "header":
            if header is not None:
                raise TraceError(f"第 {line_no} 行：重复的 header")
            if last_t is not None:
                raise TraceError(f"第 {line_no} 行：header 必须是第一行")
            header = _parse_header(obj, line_no)
            continue
        if header is None:
            raise TraceError(f"第 {line_no} 行：header 之前出现了事件 {ev!r}")

        # 区间事件用 t0/t1，点事件用 t；主时间键统一为 key（非降序校验）
        if ev in ("exec", "env_wait", "sync"):
            t0, t1 = _interval(obj, line_no)
            key = t0
        elif ev in ("ver_bump", "traj_end"):
            key = int(_require(obj, "t", line_no))
            if key < 0:
                raise TraceError(f"第 {line_no} 行：t 必须为非负整数")
            t0 = t1 = key
        else:
            raise TraceError(f"第 {line_no} 行：未知事件类型 {ev!r}")
        if last_t is not None and key < last_t:
            raise TraceError(f"第 {line_no} 行：事件乱序（t={key} < 上一事件 t={last_t}）")
        last_t = key

        if ev == "exec":
            traj = str(_require(obj, "traj", line_no))
            batch = str(_require(obj, "batch", line_no))
            phase = str(_require(obj, "phase", line_no))
            if phase not in (PHASE_PREFILL, PHASE_DECODE):
                raise TraceError(f"第 {line_no} 行：phase={phase!r} 不合法")
            bv = int(_require(obj, "bv", line_no))
            if bv > last_ver:
                raise TraceError(f"第 {line_no} 行：bv={bv} 超过当前版本 {last_ver}（版本超前）")
            if traj in ended:
                raise TraceError(f"第 {line_no} 行：轨迹 {traj!r} 已终止，不能再有执行段")
            if phase == PHASE_PREFILL:
                n_tok = int(_require(obj, "n_prompt", line_no))
            else:
                n_tok = int(_require(obj, "n_gen", line_no))
            if n_tok <= 0:
                raise TraceError(f"第 {line_no} 行：token 数必须为正，得到 {n_tok}")
            aborted = bool(obj.get("aborted", False))
            committed = bool(obj.get("committed", True))
            if aborted and phase == PHASE_PREFILL:
                raise TraceError(f"第 {line_no} 行：prefill 段不能标记 aborted")
            execs.append(
                Exec(traj=traj, batch=batch, phase=phase, t0=t0, t1=t1, bv=bv, n_tok=n_tok,
                     committed=committed, aborted=aborted)
            )
        elif ev == "env_wait":
            env_waits.append(EnvWait(traj=str(_require(obj, "traj", line_no)), t0=t0, t1=t1))
        elif ev == "sync":
            ver = obj.get("ver")
            syncs.append(Sync(t0=t0, t1=t1, ver=int(ver) if ver is not None else None))
        elif ev == "ver_bump":
            ver = int(_require(obj, "ver", line_no))
            if ver <= last_ver:
                raise TraceError(f"第 {line_no} 行：版本回退（{ver} <= {last_ver}）")
            last_ver = ver
            bumps.append(VerBump(t=key, ver=ver))
        elif ev == "traj_end":
            traj = str(_require(obj, "traj", line_no))
            status = str(_require(obj, "status", line_no))
            if status not in ("finished", "aborted"):
                raise TraceError(f"第 {line_no} 行：status={status!r} 不合法")
            if traj in ended:
                raise TraceError(f"第 {line_no} 行：轨迹 {traj!r} 重复终止")
            ended[traj] = TrajEnd(traj=traj, t=t0, status=status)
            traj_ends.append(ended[traj])
        else:
            raise TraceError(f"第 {line_no} 行：未知事件类型 {ev!r}")

    if header is None:
        raise TraceError("trace 缺少 header 行")
    return Trace(
        header=header,
        execs=tuple(execs),
        env_waits=tuple(env_waits),
        syncs=tuple(syncs),
        bumps=tuple(bumps),
        traj_ends=tuple(traj_ends),
    )


def read_canon(path: Path) -> Trace:
    """读取 canon v0 JSONL 文件。"""
    data = path.read_bytes()
    lines = data.decode("utf-8").splitlines()
    trace = parse_events(lines)
    return replace(trace, path=path, sha256=hashlib.sha256(data).hexdigest())
