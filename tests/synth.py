"""C 侧临时合成 trace 生成器（canon v0 JSONL）。

用途：B 的 rheotrace 生成器（TASK-B）与 spec 定稿前，给 bench/analysis 提供
确定性测试输入。生成器语义与 docs/metrics-v0.md 对齐，落盘为 canon 格式；
B 的 trace 就位后这里只服务回归测试，不进 bench/traces/。

时间线结构（顺序批，批间不重叠）：
  [batch0] gap sync [batch1] gap sync ...
  批内：全员同时 prefill → decode（可选 env 等待切分）→ traj_end
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

# 与内置表 qwen2.5-1.5b 一致
MODEL_P, MODEL_L, MODEL_D = 1.54e9, 28, 1536


@dataclass
class SynthConfig:
    n_batches: int = 3
    group_size: int = 8
    prompt_tokens: int = 128
    len_mode: str = "bimodal"  # bimodal | normal
    short_len: int = 200
    short_std: int = 30
    short_weight: float = 0.8
    long_len: int = 1500
    long_std: int = 150
    decode_tps: float = 1000.0  # 每轨迹 decode 速率 token/s
    prefill_dur_ns: int = 10_000_000
    sync_dur_ns: int = 50_000_000
    gap_dur_ns: int = 20_000_000  # 批间 schedule gap（P3）
    env_wait_frac: float = 0.0  # 个别轨迹 env 等待比例（切分其 decode，不造成全局 pause）
    env_wait_dur_ns: int = 100_000_000
    global_env_wait_batch: int = -1  # 指定批全员同时 env 等待（产生 P2）；-1 = 无
    abort_frac: float = 0.0
    model: str = "qwen2.5-1.5b"
    world_size: int = 1
    seed: int = 0


def _sample_length(cfg: SynthConfig, rng: random.Random) -> int:
    if cfg.len_mode == "bimodal":
        if rng.random() < cfg.short_weight:
            return max(1, round(rng.gauss(cfg.short_len, cfg.short_std)))
        return max(1, round(rng.gauss(cfg.long_len, cfg.long_std)))
    return max(1, round(rng.gauss((cfg.short_len + cfg.long_len) / 2, cfg.long_std * 2)))


def _decode_exec(traj, batch, t0, t1, bv, n_gen, committed=True):
    return {
        "ev": "exec",
        "traj": traj,
        "batch": batch,
        "phase": "decode",
        "t0": int(t0),
        "t1": int(t1),
        "bv": bv,
        "n_gen": int(n_gen),
        "committed": committed,
    }


def generate_events(cfg: SynthConfig) -> tuple[list[dict], dict]:
    """生成 canon 事件列表（按时间排序）与真值字典。"""
    rng = random.Random(cfg.seed)
    header = {
        "ev": "header",
        "format": "rheo-canon-v0",
        "clock": "mono_ns",
        "model": cfg.model,
        "P": MODEL_P,
        "L": MODEL_L,
        "d": MODEL_D,
        "world_size": cfg.world_size,
    }
    body: list[dict] = []
    t = 1_000_000_000  # 起点 1s，避开 0 附近边界
    ver = 0
    sync_total = 0
    gap_total = 0
    env_global_total = 0

    for i in range(cfg.n_batches):
        if i > 0:
            t += cfg.gap_dur_ns  # 无事件空档 → P3
            gap_total += cfg.gap_dur_ns
            body.append({"ev": "sync", "t0": t, "t1": t + cfg.sync_dur_ns, "ver": ver + 1})
            sync_total += cfg.sync_dur_ns
            t += cfg.sync_dur_ns
            ver += 1
            body.append({"ev": "ver_bump", "t": t, "ver": ver})

        batch = f"b{i}"
        batch_start = t
        d0 = batch_start + cfg.prefill_dur_ns
        is_global_env = i == cfg.global_env_wait_batch

        lengths = [_sample_length(cfg, rng) for _ in range(cfg.group_size)]
        aborted = [rng.random() < cfg.abort_frac for _ in range(cfg.group_size)]
        if is_global_env:
            aborted = [False] * cfg.group_size

        # prefill 全员同时
        for j in range(cfg.group_size):
            body.append(
                {
                    "ev": "exec",
                    "traj": f"t{i}-{j}",
                    "batch": batch,
                    "phase": "prefill",
                    "t0": batch_start,
                    "t1": d0,
                    "bv": ver,
                    "n_prompt": cfg.prompt_tokens,
                }
            )

        batch_max_end = 0
        if is_global_env:
            # 全员在 T_mid 同时进入 env 等待：decode1 压满到 T_mid，等待后跑剩余
            n1 = round(0.4 * min(lengths))  # 短于所有 n，保证人人有剩余
            t_mid = d0 + n1 / cfg.decode_tps * 1e9
            for j in range(cfg.group_size):
                traj = f"t{i}-{j}"
                body.append(_decode_exec(traj, batch, d0, t_mid, ver, n1))
                end = t_mid + cfg.env_wait_dur_ns
                if lengths[j] > n1:
                    dur2 = (lengths[j] - n1) / cfg.decode_tps * 1e9
                    body.append(_decode_exec(traj, batch, end, end + dur2, ver, lengths[j] - n1))
                    end += dur2
                body.append(
                    {"ev": "env_wait", "traj": traj, "t0": t_mid, "t1": t_mid + cfg.env_wait_dur_ns}
                )
                body.append({"ev": "traj_end", "traj": traj, "t": int(end), "status": "finished"})
                batch_max_end = max(batch_max_end, end)
            env_global_total += cfg.env_wait_dur_ns
            t = batch_max_end
            continue

        for j in range(cfg.group_size):
            traj = f"t{i}-{j}"
            n = lengths[j]
            ab = aborted[j]
            committed = not ab
            if ab:
                n = max(1, int(n * 0.4))
            dur = n / cfg.decode_tps * 1e9
            if not ab and rng.random() < cfg.env_wait_frac:
                n1 = n // 2
                dur1 = n1 / cfg.decode_tps * 1e9
                body.append(_decode_exec(traj, batch, d0, d0 + dur1, ver, n1, committed))
                ew0 = d0 + dur1
                body.append(
                    {"ev": "env_wait", "traj": traj, "t0": ew0, "t1": ew0 + cfg.env_wait_dur_ns}
                )
                dur2 = (n - n1) / cfg.decode_tps * 1e9
                body.append(
                    _decode_exec(
                        traj,
                        batch,
                        ew0 + cfg.env_wait_dur_ns,
                        ew0 + cfg.env_wait_dur_ns + dur2,
                        ver,
                        n - n1,
                        committed,
                    )
                )
                end = ew0 + cfg.env_wait_dur_ns + dur2
            else:
                body.append(_decode_exec(traj, batch, d0, d0 + dur, ver, n, committed))
                end = d0 + dur
            status = "aborted" if ab else "finished"
            body.append({"ev": "traj_end", "traj": traj, "t": int(end), "status": status})
            batch_max_end = max(batch_max_end, end)
        t = batch_max_end

    events = [header] + sorted(body, key=lambda e: e.get("t0", e.get("t", 0)))
    # 时间统一收敛为整数 ns（同一浮点值截断后保持事件间对齐一致）
    for e in events:
        for k in ("t0", "t1", "t"):
            if k in e:
                e[k] = int(e[k])
    truth = _truth(events, sync_total, gap_total, env_global_total)
    return events, truth


def _truth(events: list[dict], sync_total: int, gap_total: int, env_global_total: int) -> dict:
    execs = [e for e in events if e["ev"] == "exec"]
    t0 = min(e["t0"] for e in execs)
    ends = [e["t1"] for e in execs] + [e["t1"] for e in events if e["ev"] == "env_wait"]
    ends += [e["t1"] for e in events if e["ev"] == "sync"] + [
        e["t"] for e in events if e["ev"] == "traj_end"
    ]
    t1 = max(ends)
    decode_execs = [e for e in execs if e["phase"] == "decode"]
    gen_all = sum(e["n_gen"] for e in decode_execs)
    gen_committed = sum(e["n_gen"] for e in decode_execs if e.get("committed", True))
    # T_active = 执行区间并集（同批轨迹并发，不能按段求和）
    merged: list[list[int]] = []
    for a, b in sorted((e["t0"], e["t1"]) for e in execs):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    active = sum(b - a for a, b in merged)
    per_traj: dict[str, list[dict]] = {}
    for e in execs:
        per_traj.setdefault(e["traj"], []).append(e)
    lengths = sorted(
        sum(e["n_gen"] for e in exs if e["phase"] == "decode") for exs in per_traj.values()
    )
    n_aborted = sum(1 for e in events if e["ev"] == "traj_end" and e["status"] == "aborted")
    return {
        "t0": t0,
        "t1": t1,
        "t_wall": t1 - t0,
        "t_active": active,
        "gen_all": gen_all,
        "gen_committed": gen_committed,
        "p1": sync_total,
        "p2": env_global_total,
        "p3": gap_total,
        "lengths": lengths,
        "n_aborted": n_aborted,
        "n_trajs": len(per_traj),
    }


def expected_flops(
    events: list[dict], P: float = MODEL_P, L: int = MODEL_L, d: int = MODEL_D
) -> float:
    """按 metrics-v0 §2.1 解析式从事件重算 FLOPs（独立实现，供测试对照）。"""
    total = 0.0
    per_traj: dict[str, list[dict]] = {}
    for e in events:
        if e["ev"] == "exec":
            per_traj.setdefault(e["traj"], []).append(e)
    for execs in per_traj.values():
        s = 0
        for e in sorted(execs, key=lambda x: x["t0"]):
            if e["phase"] == "prefill":
                total += 2.0 * P * e["n_prompt"]
                s = e["n_prompt"]
            else:
                total += 2.0 * P * e["n_gen"]
                total += 4.0 * L * d * e["n_gen"] * (s + s + e["n_gen"]) / 2.0
                s += e["n_gen"]
    return total


def write_trace(cfg: SynthConfig, path: Path) -> tuple[Path, dict]:
    events, truth = generate_events(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    return path, truth
