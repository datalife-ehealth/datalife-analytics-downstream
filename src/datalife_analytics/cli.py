"""Command line entry point: ``datalife-fixtures generate`` and ``datalife-fixtures validate``."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from .contract import ContractError, validate
from .generator import GeneratorConfig, generate
from .io import FORMATS, read_dataset, write_dataset


def _date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {text!r}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datalife-fixtures",
        description="Generate and validate synthetic DataLife longitudinal datasets.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate a synthetic dataset")
    gen.add_argument("--seed", type=int, required=True)
    gen.add_argument("--subjects", type=int, default=200)
    gen.add_argument("--start", type=_date, default=date(2024, 1, 1))
    gen.add_argument("--end", type=_date, default=date(2025, 12, 31))
    gen.add_argument("--not-resulted-rate", type=float, default=0.03)
    gen.add_argument("--format", choices=FORMATS, default="parquet")
    gen.add_argument("--out", type=Path, default=Path("data/generated"))
    gen.add_argument("--prefix", default="", help="optional filename prefix")

    val = sub.add_parser("validate", help="validate a generated dataset directory")
    val.add_argument("directory", type=Path)
    val.add_argument("--prefix", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "generate":
        try:
            cfg = GeneratorConfig(
                seed=args.seed,
                n_subjects=args.subjects,
                start=args.start,
                end=args.end,
                not_resulted_rate=args.not_resulted_rate,
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        tables = generate(cfg)
        validate(tables)
        manifest = write_dataset(tables, cfg, args.out, fmt=args.format, prefix=args.prefix)
        counts = {k: len(v) for k, v in tables.items()}
        print(f"wrote {manifest.parent} {json.dumps(counts)}")
        return 0

    try:
        validate(read_dataset(args.directory, prefix=args.prefix))
    except ContractError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"ok: {args.directory} satisfies the contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
