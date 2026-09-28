"""命令行入口：python -m rheotrace {validate,gen}。"""

from __future__ import annotations

import argparse
import sys

from .core import ValidationError
from .gen import PRESETS, generate_file
from .validate import validate


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rheotrace", description="RheoTrace v0 trace 工具")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="校验 trace 文件（.jsonl / .jsonl.gz）")
    v.add_argument("files", nargs="+")
    v.add_argument("--lenient", action="store_true", help="只报告不拒绝（退出码恒 0）")

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
                rep = validate(f, strict=not args.lenient)
            except ValidationError as e:
                rep = e.report
            except OSError as e:
                print(f"{f}: 无法读取（{e.strerror or e}）")
                rc = 2
                continue
            verdict = "OK" if rep.ok else "REJECTED"
            print(f"{f}: {verdict} ({len(rep.errors)} errors, {len(rep.warnings)} warnings)")
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
    n = generate_file(args.out, preset=args.preset, seed=args.seed, **overrides)
    print(f"{args.out}: {n} events (preset={args.preset}, seed={args.seed})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
