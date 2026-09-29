"""`python -m bench.analysis <trace>` —— 一条命令出完整报告（TASK-C 验收项）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .adapters import TraceError, read_trace
from .figures import save_all
from .metrics import MetricsError, analyze
from .report import write_report

DEFAULT_OUT_ROOT = Path(__file__).resolve().parents[2] / "bench" / "results"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m bench.analysis",
        description="trace → rollout 分析报告（Markdown + 图），口径见 docs/metrics-v0.md",
    )
    p.add_argument("trace", help="trace 文件路径（JSONL）")
    p.add_argument("--out", type=Path, default=None,
                   help="输出目录（默认 bench/results/<trace名>-<起始t0ns>/）")
    p.add_argument("--peak-tflops", type=float, default=None,
                   help="实测 GEMM 峰值，覆盖默认/元数据峰值")
    p.add_argument("--model-config", type=Path, default=None,
                   help='模型参数 json：{"P":…,"L":…,"d":…}')
    p.add_argument("--no-figures", action="store_true", help="只出 Markdown，不画图")
    args = p.parse_args(argv)

    try:
        trace = read_trace(args.trace)
    except TraceError as e:
        print(f"错误：{e}", file=sys.stderr)
        return 2

    try:
        analysis = analyze(trace, peak_tflops=args.peak_tflops, model_config_path=args.model_config)
    except MetricsError as e:
        print(f"错误：{e}", file=sys.stderr)
        return 2

    if args.out is not None:
        out_dir = args.out
    else:
        stem = Path(args.trace).stem
        out_dir = DEFAULT_OUT_ROOT / f"{stem}-{analysis.t0_ns}"

    figure_files = None
    if not args.no_figures:
        figure_files = save_all(analysis, out_dir / "figures")
    report_path = write_report(analysis, out_dir, figure_files)

    print(f"T1 tokens/s/GPU（端到端）: {analysis.thr_e2e:.2f}")
    print(f"M2 rollout MFU: {analysis.m2_rollout:.3f}"
          f"  (M1 {analysis.m1_engine:.3f} × duty {analysis.duty:.3f})")
    t_pause = sum(analysis.pause_totals.values())
    print(f"T_pause {t_pause:.3f}s / T_wall {analysis.t_wall_s:.3f}s；"
          f"S1 掉队份额 {analysis.straggler_share:.1%}")
    print(f"报告：{report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
