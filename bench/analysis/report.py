"""Markdown 报告组装：严格按 docs/metrics-v0.md §5 的报告契约分节。"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .metrics import Analysis, length_stats


def _f(x: float, sig: int = 4) -> str:
    if x != x:  # nan
        return "n/a"
    if x == 0:
        return "0"
    if abs(x) >= 1e6 or abs(x) < 1e-3:
        return f"{x:.3e}"
    return f"{x:.{sig}g}"


def _tok(x: float) -> str:
    if x >= 1e9:
        return f"{x / 1e9:.2f}G"
    if x >= 1e6:
        return f"{x / 1e6:.2f}M"
    if x >= 1e3:
        return f"{x / 1e3:.1f}k"
    return f"{x:.0f}"


def _stats_table(title: str, stats: dict[str, float]) -> list[str]:
    if stats.get("n", 0) == 0:
        return [f"**{title}**：无样本", ""]
    order = [
        "n", "mean", "std", "cv", "min", "p10", "p25", "p50",
        "p75", "p90", "p95", "p99", "max", "p99/p50", "max/p50",
    ]
    lines = [f"**{title}**", "", "| 统计量 | 值 |", "|---|---|"]
    for k in order:
        if k in stats:
            lines.append(f"| {k} | {_f(stats[k])} |")
    lines.append("")
    return lines


PAUSE_NAMES = {
    "P1": "P1 weight_sync",
    "P2": "P2 env_wait",
    "P3": "P3 schedule_gap",
    "P4": "P4 other",
}


def build_report(a: Analysis, figure_files: dict[str, str]) -> str:
    L: list[str] = []
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    L.append("# Rollout 分析报告（metrics-v0）")
    L.append("")
    L.append(f"- 生成时间：{now}")
    L.append(f"- trace：`{a.source}`" + (f"（sha256 `{a.sha256[:12]}…`）" if a.sha256 else ""))
    L.append("")

    # 0 元数据
    L.append("## 0 元数据")
    L.append("")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append(f"| 模型 | {a.model or '（未标注）'} |")
    L.append(f"| P / L / d | {_f(a.cfg.P)} / {a.cfg.L} / {a.cfg.d} |")
    L.append(f"| peak（TFLOPS，稠密） | {_f(a.peak_tflops)} |")
    L.append(f"| N_gpu（world size） | {a.world_size} |")
    L.append(f"| T_wall | {_f(a.t_wall_s)} s |")
    L.append(f"| 轨迹数 | {a.n_trajs}（abort {a.n_aborted}，未完成 {a.n_unfinished}） |")
    L.append(
        f"| prompt / 生成 token | {_tok(a.prompt_tokens)} / {_tok(a.gen_tokens_all)}"
        f"（提交 {_tok(a.gen_tokens_committed)}） |"
    )
    L.append("")
    L.append("FLOPs 模型：decode `2P + 4·L·d·s`，prefill `2P`（metrics-v0 §2.1）。")
    L.append("")

    # 1 吞吐
    L.append("## 1 吞吐")
    L.append("")
    L.append("| 指标 | 值 |")
    L.append("|---|---|")
    L.append(f"| T1 tokens/s/GPU（端到端，headline） | {_f(a.thr_e2e)} |")
    L.append(f"| T2 tokens/s/GPU（引擎活跃） | {_f(a.thr_active)} |")
    L.append(f"| T3 prefill tokens/s/GPU | {_f(a.thr_prefill)} |")
    L.append(f"| 占空比 T_active/T_wall | {_f(a.duty)} |")
    L.append("")

    # 2 MFU
    L.append("## 2 MFU 分解")
    L.append("")
    L.append("| 指标 | 值 |")
    L.append("|---|---|")
    L.append(f"| M1 引擎 MFU | {_f(a.m1_engine)} |")
    L.append(f"| M2 rollout MFU（headline） | {_f(a.m2_rollout)} |")
    L.append(f"| 有用 FLOPs 合计 | {_f(a.flops_total)} |")
    L.append(f"| 废 token 率 waste | {_f(a.waste)} |")
    L.append("")
    ident = a.m1_engine * a.duty
    rel = abs(ident - a.m2_rollout) / a.m2_rollout if a.m2_rollout > 0 else 0.0
    L.append(
        f"恒等式校验：M1 × 占空比 = {_f(ident)} vs M2 = {_f(a.m2_rollout)}（相对差 {_f(rel)}）。"
    )
    L.append("")

    # 3 停顿
    L.append("## 3 停顿分解")
    L.append("")
    t_pause = sum(a.pause_totals.values())
    L.append("| 类别 | 时长 (s) | 占 T_pause | 占 T_wall |")
    L.append("|---|---|---|---|")
    for k in ("P1", "P2", "P3", "P4"):
        v = a.pause_totals[k]
        share_pause = v / t_pause if t_pause > 0 else 0.0
        share_wall = v / a.t_wall_s if a.t_wall_s > 0 else 0.0
        L.append(f"| {PAUSE_NAMES[k]} | {_f(v)} | {share_pause:.1%} | {share_wall:.1%} |")
    if a.t_wall_s > 0:
        L.append(f"| 合计 T_pause | {_f(t_pause)} | 100% | {t_pause / a.t_wall_s:.1%} |")
    L.append("")
    if a.batch_occs and a.n_staggered_batches == len(a.batch_occs):
        L.append(
            "**S1 掉队份额**：不适用——伪批从未同时满员派发（见 §5 W-BATCH-STAGGERED），S1 置 0。"
        )
    else:
        L.append(
            f"**S1 掉队份额**：T_straggler = {_f(a.t_straggler_s)} s，"
            f"占 T_active = {a.straggler_share:.1%}（阈值 occ<0.5）。"
        )
    if a.batch_occs:
        ratios = sorted(b.tail_ratio for b in a.batch_occs)
        p50 = float(np.median(ratios))
        p90 = float(np.percentile(ratios, 90))
        L.append(
            f"**S2 批尾比**：跨批 P50 = {_f(p50)}，P90 = {_f(p90)}（共 {len(a.batch_occs)} 批）。"
        )
    L.append("")
    if "pause_waterfall" in figure_files:
        L.append(f"![pause waterfall](figures/{figure_files['pause_waterfall']})")
        L.append("")
    if "occupancy" in figure_files:
        L.append(f"![occupancy](figures/{figure_files['occupancy']})")
        L.append("")

    # 3.4 env 等待
    if a.env_wait_s:
        env_arr = np.asarray(a.env_wait_s)
        L.append("### 3.4 轨迹级 env 等待")
        L.append("")
        L.append("| 统计量 | 值 |")
        L.append("|---|---|")
        L.append(
            f"| P50 / P90 / max 等待 (s) | {_f(float(np.percentile(env_arr, 50)))} / "
            f"{_f(float(np.percentile(env_arr, 90)))} / {_f(float(env_arr.max()))} |"
        )
        L.append(f"| 等待占轨迹墙钟比例 P50 | {float(np.percentile(a.env_wait_share, 50)):.1%} |")
        L.append("")

    # 4 长度分布
    L.append("## 4 轨迹长度分布")
    L.append("")
    L.extend(_stats_table("F1 全部长度（含 abort）", length_stats(a.lengths_all_valid)))
    if a.n_aborted:
        L.extend(_stats_table("F1 仅提交（不含 abort）", length_stats(a.lengths_committed_valid)))
    L.append(f"**F2 尾部 token 份额**（长度 > P90 的轨迹）：{a.tail_token_share:.1%}。")
    bc = a.sarle_bc
    if bc == bc:
        verdict = "有双峰迹象" if bc > 5 / 9 else "无双峰迹象"
        L.append(f"**F3 Sarle 双峰系数**：BC = {_f(bc)}（阈值 5/9 ≈ 0.556 → {verdict}）。")
    else:
        L.append("**F3 Sarle 双峰系数**：样本不足（n<4），无法计算。")
    L.append(f"**F5 stale token 份额**（生成时版本 > birth_version）：{a.stale_token_share:.1%}。")
    L.append("")
    if "length_hist" in figure_files:
        L.append(f"![length hist](figures/{figure_files['length_hist']})")
        L.append("")
    if "len_vs_dur" in figure_files:
        L.append(f"![len vs dur](figures/{figure_files['len_vs_dur']})")
        L.append("")

    # 5 数据质量
    L.append("## 5 数据质量")
    L.append("")
    if a.warnings:
        for w in a.warnings:
            L.append(f"- ⚠️ {w}")
    else:
        L.append("- 无告警。")
    L.append("")

    return "\n".join(L)


def write_report(a: Analysis, out_dir: Path, figure_files: dict[str, str] | None = None) -> Path:
    """产出 report.md（figure_files 为 None 时报告不含图）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    md = build_report(a, figure_files or {})
    path = out_dir / "report.md"
    path.write_text(md, encoding="utf-8")
    return path
