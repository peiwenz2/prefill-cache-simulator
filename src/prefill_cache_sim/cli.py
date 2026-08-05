"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analyzer import analyze_trace
from .config import git_provenance, load_scenario
from .trace import load_trace


def _analyze(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario, schema_path=args.schema)
    loaded_trace = load_trace(
        scenario.trace_path,
        block_size_tokens=scenario.value["trace"]["block_size_tokens"],
    )
    trace_sha = loaded_trace.sha256
    expected_trace_sha = scenario.value["trace"]["sha256"]
    if trace_sha != expected_trace_sha:
        raise ValueError(
            f"trace.sha256 mismatch: expected {expected_trace_sha}, got {trace_sha}"
        )
    code = git_provenance(Path(__file__).resolve().parents[2])
    report = {
        "provenance": {
            "schema_version": scenario.value["schema_version"],
            "scenario_id": scenario.value["scenario_id"],
            "seed": scenario.value["seed"],
            "trace_sha256": trace_sha,
            "config_sha256": scenario.config_sha256,
            "git_sha": code.sha,
            "git_dirty": code.dirty,
        },
        "analysis": analyze_trace(loaded_trace).to_dict(),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prefill-cache-sim")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze", help="analyze a Mooncake trace")
    analyze.add_argument("--scenario", type=Path, required=True)
    packaged_schema = Path(__file__).with_name("scenario.schema.json")
    source_schema = Path(__file__).parents[2] / "scenario.schema.json"
    default_schema = packaged_schema if packaged_schema.exists() else source_schema
    analyze.add_argument(
        "--schema",
        type=Path,
        default=default_schema,
    )
    analyze.add_argument("--output", type=Path)
    analyze.set_defaults(handler=_analyze)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.handler(args)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
