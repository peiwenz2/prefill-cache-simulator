#!/usr/bin/env python3
"""Run the M12.2 placement grid over the frozen Mooncake trace."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prefill_cache_sim.config import git_provenance  # noqa: E402
from prefill_cache_sim.m12_placement import (  # noqa: E402
    TraceRequestInput,
    build_kernel_requests,
    build_m12_2_cases,
    run_placement_case,
)
from prefill_cache_sim.trace import load_trace  # noqa: E402

BLOCK_SIZE_TOKENS = 512
ARRIVAL_SCALE = 5.0
OBSERVATION_START_WORK = 0.0
OBSERVATION_END_WORK = 30_000_000.0
TENANT_BUCKETS = 16
TRUTH_BASIS = "SYNTHETIC_TENANT_TIER_ON_TRACE"
TIER_SLO_WORK = {
    "STRICT": 5_000.0,
    "STANDARD": 20_000.0,
    "RELAXED": 100_000.0,
}
MANIFEST_SCHEMA = "m12-placement-manifest-v1"
EXPECTED_CASE_COUNT = 15
EXPECTED_RESULT_COUNT = 90
EXPECTED_VERDICT_COUNT = 15


class ArtifactGridFailure(RuntimeError):
    """The diagnostic artifact set was published, but the grid is incomplete."""


class ArtifactSetInvalid(RuntimeError):
    """A reader observed an incomplete or digest-inconsistent artifact set."""


RESULT_FIELDS = (
    "case_id",
    "regime",
    "kvs_mode",
    "kvs_contention_multiplier",
    "decode_binding_requested",
    "strategy_id",
    "offered_requests",
    "attempt_count",
    "strict_useful_token_goodput",
    "strict_useful_output_token_goodput",
    "request_goodput",
    "minimum_tier_slo_attainment",
    "jain_fairness",
    "fairness_floor_pass",
    "local_hit_tokens",
    "remote_hit_tokens",
    "uncached_tokens",
    "token_hit_rate",
    "request_load_max_mean",
    "p_queue_p95",
    "kvs_normalized_work",
    "spill_count",
    "decode_normalized_utilization",
    "decode_queue_p95",
    "completion_max_work",
)


def build_trace_workload(trace_path: Path):
    trace = load_trace(trace_path, block_size_tokens=BLOCK_SIZE_TOKENS)
    rows: list[TraceRequestInput] = []
    tier_counts = {tier: 0 for tier in TIER_SLO_WORK}
    tenant_counts = {f"tenant-{index:02d}": 0 for index in range(TENANT_BUCKETS)}
    for record in trace.records:
        tenant, tier = _synthetic_fairness_truth(record.request_id)
        tier_counts[tier] += 1
        tenant_counts[tenant] += 1
        rows.append(
            TraceRequestInput(
                record.request_id,
                tenant,
                tier,
                record.timestamp_ms * ARRIVAL_SCALE,
                tuple(f"trace:{value}" for value in record.prefix_blocks),
                tuple(record.block_token_sizes),
                record.output_tokens,
                "mooncake-model",
                "no-adapter",
                "default-work-shape",
                OBSERVATION_END_WORK,
                ("p0", "p1"),
            )
        )
    workload = build_kernel_requests(rows)
    return workload, {
        "truth_basis": TRUTH_BASIS,
        "trace_sha256": trace.sha256,
        "record_count": len(trace.records),
        "input_tokens": sum(record.input_tokens for record in trace.records),
        "output_tokens": sum(record.output_tokens for record in trace.records),
        "unique_raw_blocks": len(
            {value for record in trace.records for value in record.prefix_blocks}
        ),
        "tenant_counts": tenant_counts,
        "tier_counts": tier_counts,
    }


def run_artifacts(trace_path: Path, output_dir: Path) -> dict[str, bytes]:
    workload, trace_metadata = build_trace_workload(trace_path)
    cases = build_m12_2_cases(
        horizon=OBSERVATION_END_WORK,
        tier_slo_work=TIER_SLO_WORK,
    )
    result_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    for case in cases:
        try:
            pair = run_placement_case(workload, case)
            if case.decode_binding:
                participants = (pair.hybrid, pair.priced_spill)
                unproven = [
                    report.strategy_id
                    for report in participants
                    if report.decode_normalized_utilization < 0.8
                    and report.decode_queue_p95 <= 0
                ]
                if unproven:
                    raise ValueError(
                        "decode-binding is not proven for comparison participants: "
                        + ",".join(unproven)
                    )
        except (ValueError, RuntimeError) as exc:
            failure_rows.append(
                {
                    "case_id": case.case_id,
                    "regime": case.regime.regime_id.value,
                    "kvs_mode": case.kvs_mode.value,
                    "decode_binding_requested": case.decode_binding,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
        for report in pair.result_table:
            metrics = report.kernel_metrics
            result_rows.append(
                {
                    "case_id": case.case_id,
                    "regime": case.regime.regime_id.value,
                    "kvs_mode": case.kvs_mode.value,
                    "kvs_contention_multiplier": _number(
                        case.kvs_contention_multiplier
                    ),
                    "decode_binding_requested": case.decode_binding,
                    "strategy_id": report.strategy_id,
                    "offered_requests": metrics.offered_logical_requests,
                    "attempt_count": metrics.attempt_count,
                    "strict_useful_token_goodput": _number(
                        metrics.strict_useful_token_goodput
                    ),
                    "strict_useful_output_token_goodput": _number(
                        metrics.strict_useful_output_token_goodput
                    ),
                    "request_goodput": _number(metrics.request_goodput),
                    "minimum_tier_slo_attainment": _number(
                        metrics.minimum_tier_slo_attainment
                    ),
                    "jain_fairness": _number(metrics.jain_fairness),
                    "fairness_floor_pass": metrics.fairness_floor_pass,
                    "local_hit_tokens": report.local_hit_tokens,
                    "remote_hit_tokens": report.remote_hit_tokens,
                    "uncached_tokens": report.uncached_tokens,
                    "token_hit_rate": _number(report.token_hit_rate),
                    "request_load_max_mean": _number(report.request_load_max_mean),
                    "p_queue_p95": _number(report.p_queue_p95),
                    "kvs_normalized_work": _number(report.kvs_normalized_work),
                    "spill_count": report.spill_count,
                    "decode_normalized_utilization": _number(
                        report.decode_normalized_utilization
                    ),
                    "decode_queue_p95": _number(report.decode_queue_p95),
                    "completion_max_work": _number(report.completion_max_work),
                }
            )
        comparison_rows.append(
            {
                "case_id": case.case_id,
                "regime": case.regime.regime_id.value,
                "kvs_mode": case.kvs_mode.value,
                "verdict": pair.verdict.verdict,
                "strictly_improved_axes": "|".join(pair.verdict.strictly_improved_axes),
                "violated_axes": "|".join(pair.verdict.violated_axes),
                "baseline_also_fails_floor": pair.verdict.baseline_also_fails_floor,
                "cause": pair.verdict.cause or "",
            }
        )

    provenance = git_provenance(ROOT)
    source_provenance = _source_provenance()
    contract = {
        "schema_version": "m12-placement-contract-v1",
        "truth_basis": TRUTH_BASIS,
        "time_unit": "NORMALIZED_WORK",
        "wall_clock_claim": False,
        "tenant_tier_assignment": (
            "sha256 domain-separated request_id; tenant modulo 16; "
            "tier buckets STRICT 20%, STANDARD 60%, RELAXED 20%"
        ),
        "ranking": "strict_useful_token_goodput",
        "tier_slo_work": TIER_SLO_WORK,
        "placement_candidate_slo_slack_work": OBSERVATION_END_WORK,
        "observation_window": [OBSERVATION_START_WORK, OBSERVATION_END_WORK],
        "arrival_scale": ARRIVAL_SCALE,
    }
    config = {
        **trace_metadata,
        "trace_record_count": trace_metadata["record_count"],
        "block_size_tokens": BLOCK_SIZE_TOKENS,
        "arrival_scale": ARRIVAL_SCALE,
        "observation_start_work": OBSERVATION_START_WORK,
        "observation_end_work": OBSERVATION_END_WORK,
        "tier_slo_work": TIER_SLO_WORK,
        "case_count": len(cases),
        "successful_case_count": len(comparison_rows),
        "failed_case_count": len(failure_rows),
        "result_count": len(result_rows),
        "verdict_count": len(comparison_rows),
        "git_sha": provenance.sha,
        "git_dirty": provenance.dirty,
        **source_provenance,
    }
    artifacts = {
        "contract.json": _json_bytes(contract),
        "config.json": _json_bytes(config),
        "results.csv": _csv_bytes(result_rows, RESULT_FIELDS),
        "comparisons/g12-verdicts.csv": _csv_bytes(
            comparison_rows,
            (
                "case_id",
                "regime",
                "kvs_mode",
                "verdict",
                "strictly_improved_axes",
                "violated_axes",
                "baseline_also_fails_floor",
                "cause",
            ),
        ),
        "failures.csv": _csv_bytes(
            failure_rows,
            (
                "case_id",
                "regime",
                "kvs_mode",
                "decode_binding_requested",
                "error_type",
                "error",
            ),
        ),
    }
    artifacts["MANIFEST.json"] = _manifest_bytes(artifacts)
    _write_atomic(output_dir, artifacts)
    if (
        len(cases) != EXPECTED_CASE_COUNT
        or len(result_rows) != EXPECTED_RESULT_COUNT
        or len(comparison_rows) != EXPECTED_VERDICT_COUNT
        or failure_rows
    ):
        raise ArtifactGridFailure(
            f"{len(failure_rows)} placement cases failed; expected "
            f"{EXPECTED_CASE_COUNT} cases/{EXPECTED_RESULT_COUNT} results/"
            f"{EXPECTED_VERDICT_COUNT} verdicts, got {len(cases)}/"
            f"{len(result_rows)}/{len(comparison_rows)}"
        )
    return artifacts


def _synthetic_fairness_truth(request_id: str) -> tuple[str, str]:
    tenant_value = _stable_hash_int("tenant", request_id)
    tier_value = _stable_hash_int("tier", request_id) % 100
    tenant = f"tenant-{tenant_value % TENANT_BUCKETS:02d}"
    tier = "STRICT" if tier_value < 20 else "STANDARD" if tier_value < 80 else "RELAXED"
    return tenant, tier


def _stable_hash_int(domain: str, value: str) -> int:
    digest = hashlib.sha256(f"m12-placement-{domain}-v1\x00".encode())
    encoded = value.encode()
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], "big")


def _number(value: float) -> str:
    return format(value, ".12g")


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _csv_bytes(rows, fields) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def _manifest_bytes(artifacts: dict[str, bytes]) -> bytes:
    return _json_bytes(
        {
            "schema_version": MANIFEST_SCHEMA,
            "algorithm": "sha256",
            "files": {
                name: hashlib.sha256(payload).hexdigest()
                for name, payload in sorted(artifacts.items())
            },
        }
    )


def _source_provenance() -> dict[str, object]:
    paths = (Path(__file__).resolve(), *sorted((ROOT / "src").rglob("*.py")))
    fingerprints = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    combined = _fingerprint_mapping(fingerprints)
    source_names = tuple(fingerprints)
    diff = subprocess.run(
        ["git", "diff", "--binary", "--no-ext-diff", "HEAD", "--", *source_names],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *source_names],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.splitlines()
    dirty_payload = {
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_source_fingerprints": {
            name: fingerprints[name] for name in sorted(untracked)
        },
    }
    return {
        "source_fingerprints": fingerprints,
        "source_combined_sha256": combined,
        "dirty_patch_sha256": _fingerprint_mapping(dirty_payload),
    }


def _fingerprint_mapping(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def read_valid_artifact_set(output_dir: Path) -> dict[str, object]:
    """Read the commit marker and reject every incomplete/mixed artifact set."""
    manifest_path = output_dir / "MANIFEST.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != MANIFEST_SCHEMA:
            raise ArtifactSetInvalid("manifest schema mismatch")
        files = manifest["files"]
        if not isinstance(files, dict) or not files:
            raise ArtifactSetInvalid("manifest has no files")
        for name, expected in files.items():
            payload = (output_dir / name).read_bytes()
            actual = hashlib.sha256(payload).hexdigest()
            if actual != expected:
                raise ArtifactSetInvalid(f"digest mismatch for {name}")
    except ArtifactSetInvalid:
        raise
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ArtifactSetInvalid(f"artifact set is incomplete: {exc}") from exc
    return manifest


def _write_atomic(output_dir: Path, artifacts: dict[str, bytes]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=output_dir.parent, prefix=".m12-placement-"))
    try:
        ordered = [name for name in sorted(artifacts) if name != "MANIFEST.json"]
        ordered.append("MANIFEST.json")
        for name in ordered:
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(artifacts[name])
        for name in ordered:
            destination = output_dir / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / name, destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main() -> int:
    trace = ROOT / "mooncake_trace.jsonl"
    output = ROOT / "results" / "m12-placement"
    try:
        run_artifacts(trace, output)
    except ArtifactGridFailure as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
