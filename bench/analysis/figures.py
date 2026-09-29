"""报告插图（matplotlib，Agg 后端）。图内文字用英文，避免无中文字体环境出豆腐块。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .metrics import Analysis  # noqa: E402

NS = 1e9
DPI = 150

PAUSE_COLORS = {
    "P1": "#d62728",  # weight_sync
    "P2": "#ff7f0e",  # env_wait
    "P3": "#7f7f7f",  # schedule_gap
    "P4": "#bcbd22",  # other
}
PAUSE_NAMES = {
    "P1": "P1 weight_sync",
    "P2": "P2 env_wait",
    "P3": "P3 schedule_gap",
    "P4": "P4 other",
}


def fig_pause_waterfall(a: Analysis, path: Path) -> None:
    """墙钟瀑布：active + 四类 pause 在时间轴上铺开。"""
    fig, ax = plt.subplots(figsize=(9, 2.6))
    t0 = a.t0_ns
    rows: list[tuple[str, str, list[tuple[int, int]]]] = [
        ("active", "#2ca02c", [(s, e) for s, e in a.active_spans])
    ]
    for label in ("P1", "P2", "P3", "P4"):
        spans = [(p.t0, p.t1) for p in a.pause_pieces if p.label == label]
        if spans:
            rows.append((PAUSE_NAMES[label], PAUSE_COLORS[label], spans))
    for i, (_name, color, spans) in enumerate(rows):
        for s, e in spans:
            ax.barh(i, (e - s) / NS, left=(s - t0) / NS, height=0.6, color=color)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("time (s)")
    ax.set_title("Rollout wall-clock waterfall")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_occupancy(a: Analysis, path: Path) -> None:
    """批占用率曲线（S3）：0.5 阈值线下方即掉队时间。"""
    fig, ax = plt.subplots(figsize=(9, 3))
    t0 = a.t0_ns
    for occ in a.batch_occs:
        ts = [(p[0] - t0) / NS for p in occ.points]
        vs = [p[1] for p in occ.points]
        ax.step(ts, vs, where="post", alpha=0.7)
    ax.axhline(0.5, color="r", ls="--", lw=1, label="straggler threshold (0.5)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("batch occupancy k/B0")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Per-batch occupancy (S3)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_length_hist(a: Analysis, path: Path) -> None:
    """轨迹长度直方图（F3 辅证）：双峰负载应呈双驼峰。"""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    vals = a.lengths_all_valid
    if vals.size:
        ax.hist(vals, bins=min(40, max(10, vals.size // 2)), color="#4c72b0")
    ax.set_yscale("log")
    ax.set_xlabel("trajectory length (decode tokens)")
    ax.set_ylabel("count (log)")
    ax.set_title("Length distribution")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_len_vs_dur(a: Analysis, path: Path) -> None:
    """长度-时长散点（F4）：env 等待重的轨迹呈"短长度、长时长"离群带。"""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.scatter(a.lengths_all, a.traj_wall_s, s=12, alpha=0.6)
    ax.set_xlabel("length (decode tokens)")
    ax.set_ylabel("trajectory wall time (s)")
    ax.set_title("Length vs duration (F4)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def save_all(a: Analysis, out_dir: Path) -> dict[str, str]:
    """产出全部图，返回名字→文件名（report.md 同目录相对引用）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = {
        "pause_waterfall": fig_pause_waterfall,
        "occupancy": fig_occupancy,
        "length_hist": fig_length_hist,
        "len_vs_dur": fig_len_vs_dur,
    }
    files: dict[str, str] = {}
    for name, fn in figs.items():
        p = out_dir / f"{name}.png"
        fn(a, p)
        files[name] = p.name
    return files
