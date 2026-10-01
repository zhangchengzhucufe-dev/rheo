"""命令行入口：python -m rheotrace {validate,gen}。"""

from __future__ import annotations

import argparse
import sys

from .core import ValidationError
from .gen import PRESETS, generate_file
from .validate import validate


def parse_step_spec(spec: str) -> set[int]:
    """'1-40' / '1,2,5-10' → step 集合（--expected-steps 用）。"""
    steps: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        if sep:
            steps.update(range(int(lo), int(hi) + 1))
        else:
            steps.add(int(part))
    return steps


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rheotrace", description="RheoTrace v0 trace 工具")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="校验 trace 文件（.jsonl / .jsonl.gz）")
    v.add_argument("files", nargs="+")
    v.add_argument(
        "--lenient",
        action="store_true",
        help="数据问题只报告不拒绝；无法读取的文件仍以退出码 2 报告",
    )
    v.add_argument(
        "--expected-steps",
        type=parse_step_spec,
        default=None,
        metavar="SPEC",
        help=(
            "训练日志的 step 集合，如 '1-40' 或 '1,2,5-10'；"
            "与 trace 内 weight_sync.trainer_step 对账，缺失步报 WARNING"
        ),
    )

    g = sub.add_parser("gen", help="生成合成 trace")
    g.add_argument("--preset", choices=sorted(PRESETS), default="grpo")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", required=True)
    g.add_argument("--n-steps", type=int, default=None)
    g.add_argument("--groups-per-step", type=int, default=None)
    g.add_argument("--group-size", type=int, default=None)

    args = ap.parse_args(argv)

    if args.cmd == "validate":
        rc = 0
        for f in args.files:
            try:
                rep = validate(f, strict=not args.lenient, expected_steps=args.expected_steps)
            except ValidationError as e:
                rep = e.report
            except OSError as e:
                print(f"{f}: 无法读取（{e.strerror or e}）")
                rc = 2
                continue
            verdict = "OK" if rep.ok else "REJECTED"
            print(f"{f}: {verdict} ({len(rep.errors)} errors, {len(rep.warnings)} warnings)")
            if rep.covered_steps:
                lo, hi = rep.covered_steps[0], rep.covered_steps[-1]
                print(f"  covered_steps: {lo}-{hi} ({len(rep.covered_steps)} 步)")
            for i in rep.errors:
                print(f"  E {i}")
            for w in rep.warnings:
                print(f"  W {w}")
            if not rep.ok:
                rc = 1
        return rc

    overrides = {
        k: v
        for k, v in {
            "n_steps": args.n_steps,
            "groups_per_step": args.groups_per_step,
            "group_size": args.group_size,
        }.items()
        if v is not None
    }
    try:
        n = generate_file(args.out, preset=args.preset, seed=args.seed, **overrides)
    except OSError as e:
        print(f"{args.out}: 无法写入（{e.strerror or e}）")
        return 2
    print(f"{args.out}: {n} events (preset={args.preset}, seed={args.seed})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
