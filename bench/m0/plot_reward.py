"""Plot the M0 reward curve from verl's tensorboard event files.

Usage: python bench/m0/plot_reward.py [--tb-dir DIR] [--out PNG]

Reads every event file under the tensorboard dir (verl layout:
$TENSORBOARD_DIR/<project>/<experiment>/events.out.*), extracts scalar tags
related to reward/score/length, and renders one figure: training reward and
validation score vs global step.
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalars(tb_dir: str) -> dict[str, dict[int, float]]:
    series: dict[str, dict[int, float]] = {}
    for path in glob.glob(os.path.join(tb_dir, "**", "events.out.*"), recursive=True):
        acc = EventAccumulator(os.path.dirname(path))
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            series.setdefault(tag, {})
            for ev in acc.Scalars(tag):
                series[tag][ev.step] = float(ev.value)
    return series


def pick(series: dict[str, dict[int, float]], *keywords: str) -> list[str]:
    return [t for t in series if any(k in t.lower() for k in keywords)]


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    results = os.path.join(here, "..", "results", "m0-baseline")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb-dir", default=os.path.join(results, "tb"))
    ap.add_argument("--out", default=os.path.join(results, "reward_curve.png"))
    args = ap.parse_args()

    series = load_scalars(args.tb_dir)
    if not series:
        raise SystemExit(f"no tensorboard scalars found under {args.tb_dir}")
    print("available tags:")
    for tag in sorted(series):
        pts = series[tag]
        print(f"  {tag}: {len(pts)} points, steps {min(pts)}..{max(pts)}")

    train_tags = pick(series, "reward", "score")
    train_tags = [t for t in train_tags if not t.startswith("val/")]
    val_tags = pick(series, "val/")
    len_tags = pick(series, "response_length")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, height_ratios=[3, 1])
    for tag in sorted(train_tags):
        pts = series[tag]
        ax1.plot(*zip(*sorted(pts.items()), strict=True), marker="o", ms=3, label=f"train: {tag}")
    for tag in sorted(val_tags):
        pts = series[tag]
        ax1.plot(
            *zip(*sorted(pts.items()), strict=True),
            marker="s",
            ms=5,
            ls="--",
            label=f"val: {tag}",
        )
    ax1.set_ylabel("reward / score")
    ax1.set_title("M0 baseline: GRPO + LoRA on Qwen2.5-1.5B-Instruct (GSM8K)")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    for tag in sorted(len_tags):
        pts = series[tag]
        ax2.plot(*zip(*sorted(pts.items()), strict=True), marker="o", ms=3, label=tag)
    ax2.set_xlabel("trainer step")
    ax2.set_ylabel("resp length")
    ax2.grid(alpha=0.3)
    if len_tags:
        ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
