import argparse
import json
import math
import sys
from pathlib import Path

from datalife_analytics.baseline.diff import compare_reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datalife-baseline-diff",
        description="Compare two baseline study reports to detect reproducibility drift.",
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument(
        "--tolerances",
        type=Path,
        required=True,
        help="Path to the JSON tolerance map",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable JSON summary",
    )

    return parser


def reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def validate_json_numbers(value: object) -> None:
    if isinstance(value, dict):
        for item in value.values():
            validate_json_numbers(item)

    elif isinstance(value, list):
        for item in value:
            validate_json_numbers(item)

    elif isinstance(value, bool):
        raise ValueError("Boolean JSON values are not allowed")

    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite JSON number")

    elif isinstance(value, int):
        try:
            float(value)
        except OverflowError as exc:
            raise ValueError("JSON integer is too large") from exc


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        with args.baseline.open(encoding="utf-8") as file:
            baseline_report = json.load(file, parse_constant=reject_json_constant)

        with args.current.open(encoding="utf-8") as file:
            current_report = json.load(file, parse_constant=reject_json_constant)

        with args.tolerances.open(encoding="utf-8") as file:
            tolerances = json.load(file, parse_constant=reject_json_constant)

        validate_json_numbers(baseline_report)
        validate_json_numbers(current_report)
        validate_json_numbers(tolerances)

        if not isinstance(baseline_report, dict):
            raise ValueError("Invalid baseline report")

        if not isinstance(current_report, dict):
            raise ValueError("Invalid current report")

        if not isinstance(tolerances, dict) or not tolerances:
            raise ValueError("Invalid tolerance map")

        if not all(isinstance(key, str) and key for key in tolerances):
            raise ValueError("Invalid tolerance map keys")

        summary = compare_reports(
            baseline_report,
            current_report,
            tolerances,
        )

    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError, KeyError):
        print("error: invalid or unreadable comparison input", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(summary, sort_keys=True))
    else:
        print(f"Overall: {summary['status']}")
        for name, status in summary["checks"].items():
            print(f"{name}: {status}")

    if any(status == "MISSING" for status in summary["checks"].values()):
        return 2

    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
