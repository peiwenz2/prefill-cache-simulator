#!/usr/bin/env python3
"""Deterministically repair the single failed M12 final-grid cell.

This tool is intentionally narrow.  It accepts only the frozen a469688 base
artifact and the da59431 repair bundle documented by M12.  It never mutates the
base directory and publishes through the runner's atomic writer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import pickle
import posixpath
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    from scripts import run_m12_final as m12
except ModuleNotFoundError:  # direct `python scripts/repair_m12_final.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts import run_m12_final as m12

BASE_SHA = "a469688e2225c87ab18a454b1e12e7addf6a4a7f"
BASE_MANIFEST_SHA256 = (
    "182a70aeba26c761323742b44dc8925a2c0af641f252273eee0c45742358f281"
)
REPAIR_SHA = "da59431"
REPAIR_MANIFEST_SHA256 = (
    "9298e760d759f7291c33d297f4812ab320a378e21a9a265f62066b700bc18a19"
)
MISSING_CELL = "MIXED-1.5x-DECODE_CAUSAL"
INVARIANCE_CELLS = (
    "COMPUTE_BOUND-0.8x-BASELINE",
    "MEMORY_BOUND-2.0x-DECODE_CAUSAL",
    "MIXED-1.5x-BASELINE",
    "MIXED-1.5x-PRICED_SPILL",
)
EXPECTED_BASE_FILES = frozenset(
    {
        "attribution.json",
        "config.json",
        "constraints.csv",
        "contract.json",
        "crossovers.csv",
        "diagnostics/rss-watchdog.json",
        "explanation.csv",
        "failures.csv",
        "falsification/visibility-delay.json",
        "gates/g12-3.json",
        "gates/g12-4.json",
        "pareto.json",
        "primary.csv",
        "provenance.json",
        "sensitivities.csv",
    }
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _safe_name(name: str) -> bool:
    return (
        name == posixpath.normpath(name)
        and not name.startswith("/")
        and name not in ("", ".", "..")
        and not name.startswith("../")
        and "\\" not in name
    )


def _load_pickle(path: Path, expected_sha256: str) -> dict[str, Any]:
    # Inputs are locally produced, private experiment artifacts.  Never use this
    # loader for downloaded or otherwise untrusted pickle files.
    payload = path.read_bytes()
    if _sha256(payload) != expected_sha256:
        raise ValueError(f"repair payload hash mismatch: {path.name}")
    with io.BytesIO(payload) as stream:
        value = pickle.load(stream)  # noqa: S301
    if not isinstance(value, dict) or "result" not in value:
        raise ValueError(f"invalid repair payload: {path}")
    return value


def _validate_manifest(
    base: Path,
) -> tuple[dict[str, Any], dict[str, bytes], bytes]:
    manifest_bytes = (base / "MANIFEST.json").read_bytes()
    if _sha256(manifest_bytes) != BASE_MANIFEST_SHA256:
        raise ValueError("untrusted base manifest digest")
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError("base manifest must be a JSON object")
    if manifest.get("schema_version") != "m12-final-manifest-v1":
        raise ValueError("unexpected base manifest schema")
    if manifest.get("algorithm") != "sha256":
        raise ValueError("base manifest must use sha256")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("base manifest files must be an object")
    if set(files) != EXPECTED_BASE_FILES:
        raise ValueError("base manifest artifact set mismatch")
    snapshot: dict[str, bytes] = {}
    for name, expected in files.items():
        if not isinstance(name, str) or not _safe_name(name):
            raise ValueError(f"unsafe base artifact name: {name!r}")
        path = base / str(name)
        if not path.is_file():
            raise ValueError(f"base artifact missing: {name}")
        payload = path.read_bytes()
        if _sha256(payload) != expected:
            raise ValueError(f"base manifest mismatch: {name}")
        snapshot[name] = payload
    return manifest, snapshot, manifest_bytes


def _validate_repair_manifest(path: Path, repair: Path) -> dict[str, Any]:
    manifest_bytes = path.read_bytes()
    if _sha256(manifest_bytes) != REPAIR_MANIFEST_SHA256:
        raise ValueError("untrusted repair manifest digest")
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError("repair manifest must be a JSON object")
    if manifest.get("schema_version") != "m12-repair-input-manifest-v1":
        raise ValueError("unexpected repair manifest schema")
    if manifest.get("git_sha") != "da594318b78c0901874d784bae87e8e197dddcf7":
        raise ValueError("unexpected repair git sha")
    if manifest.get("git_dirty") is not False:
        raise ValueError("repair manifest must attest a clean run")
    payloads = manifest.get("payloads")
    if not isinstance(payloads, dict):
        raise ValueError("repair payload manifest must be an object")
    expected = {
        "missing-cell.pkl",
        "missing-cell-delay1.pkl",
        "mixed-1.5-decode-no-gate.pkl",
        *(f"invariance/{cell_id}.pkl" for cell_id in INVARIANCE_CELLS),
    }
    if set(payloads) != expected:
        raise ValueError("repair payload set mismatch")
    root = repair.resolve()
    for name, digest in payloads.items():
        if (
            not isinstance(name, str)
            or not _safe_name(name)
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise ValueError(f"unsafe repair payload name: {name!r}")
        candidate = (repair / name).resolve()
        if candidate.parent != root and root not in candidate.parents:
            raise ValueError(f"repair payload escapes root: {name}")
        if not candidate.is_file() or _sha256(candidate.read_bytes()) != digest:
            raise ValueError(f"repair payload hash mismatch: {name}")
    source_hashes = manifest.get("source_sha256")
    if not isinstance(source_hashes, dict):
        raise ValueError("repair source manifest must be an object")
    if set(source_hashes) != {
        "scripts/run_m12_final.py",
        "src/prefill_cache_sim/m12_decode.py",
        "src/prefill_cache_sim/m12_kernel.py",
    }:
        raise ValueError("repair source manifest set mismatch")
    revision = str(manifest["git_sha"])
    for name, digest in source_hashes.items():
        if not isinstance(name, str) or not _safe_name(name):
            raise ValueError(f"unsafe repair source name: {name!r}")
        payload = subprocess.run(
            ("git", "show", f"{revision}:{name}"),
            check=True,
            capture_output=True,
        ).stdout
        if _sha256(payload) != digest:
            raise ValueError(f"repair source hash mismatch: {name}")
    return manifest


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _csv_payload_rows(payload: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(payload.decode())))


def _index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    indexed = {row["cell_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate cell_id in base CSV")
    return indexed


def _mapped_row(
    result: m12.CellResult,
    mapper: Callable[[m12.CellResult], Mapping[str, object]],
) -> dict[str, str]:
    return next(csv.DictReader(m12._csv_bytes([result], mapper).decode().splitlines()))


def _reconstruct_results(
    primary: list[dict[str, str]],
    constraints: Mapping[str, dict[str, str]],
    explanation: Mapping[str, dict[str, str]],
    repaired: m12.CellResult,
) -> list[m12.CellResult]:
    results: list[m12.CellResult] = []
    for row in primary:
        cell_id = row["cell_id"]
        if cell_id == repaired.cell.cell_id:
            results.append(repaired)
            continue
        c = constraints[cell_id]
        e = explanation[cell_id]
        if c["accounting_conserved"] != "True":
            raise ValueError(f"base work accounting failed: {cell_id}")
        results.append(
            m12.CellResult(
                m12.FinalCell(
                    row["regime"],
                    float(row["arrival_scale"]),
                    row["strategy"],
                    "PRIMARY",
                ),
                int(row["offered_requests"]),
                int(row["offered_tokens"]),
                float(row["strict_goodput"]),
                float(row["strict_output_goodput"]),
                float(row["request_goodput"]),
                float(c["minimum_tier"]),
                float(c["jain"]),
                json.loads(c["per_tier"]),
                float(c["queue_p95_normalized_work"]),
                float(e["token_hit_rate"]),
                float(e["waste_fraction"]),
                float(e["load_skew"]),
                float(e["kvs_work"]),
                float(e["p_utilization"]),
                float(e["d_utilization"]),
                float(e["p_to_d_debt"]),
                1.0,
                1.0,
                row["capacity_binding"] == "True",
                int(row["cache_capacity_entries"]),
                hit_ceiling=float(row["hit_ceiling"]),
                decision_fingerprint=e["decision_fingerprint"],
                census_age_work=(
                    None if e["census_age_work"] == "" else float(e["census_age_work"])
                ),
                visibility_delay_work=float(e["visibility_delay_work"]),
                attempt_count=int(row["attempt_count"]),
                retry_count=int(row["retry_count"]),
                congestion_action=row["congestion_action"] or None,
                gated_retry_count=int(row["gated_retry_count"]),
            )
        )
    return results


def _replace_group(
    document: dict[str, Any],
    key: str,
    replacement: Mapping[str, Any],
    predicate: Callable[[Mapping[str, Any]], bool],
) -> None:
    rows = document.get(key)
    if not isinstance(rows, list):
        raise ValueError(f"expected list: {key}")
    kept = [row for row in rows if not (isinstance(row, dict) and predicate(row))]
    kept.append(dict(replacement))
    document[key] = sorted(
        kept,
        key=lambda row: (
            str(row.get("regime", "")),
            float(row.get("arrival_scale", 0)),
            str(row.get("single_switch", row.get("strategy", ""))),
        ),
    )


def build_repair(
    base: Path, repair: Path, repair_manifest_path: Path
) -> dict[str, bytes]:
    base_manifest, base_snapshot, base_manifest_bytes = _validate_manifest(base)
    repair_manifest = _validate_repair_manifest(repair_manifest_path, repair)
    payload_hashes = repair_manifest["payloads"]
    provenance = json.loads(base_snapshot["provenance.json"])
    if (
        provenance.get("git_sha") != BASE_SHA
        or provenance.get("git_dirty") is not False
    ):
        raise ValueError("unexpected base provenance")

    missing_payload = _load_pickle(
        repair / "missing-cell.pkl", payload_hashes["missing-cell.pkl"]
    )
    delayed_payload = _load_pickle(
        repair / "missing-cell-delay1.pkl",
        payload_hashes["missing-cell-delay1.pkl"],
    )
    no_gate_payload = _load_pickle(
        repair / "mixed-1.5-decode-no-gate.pkl",
        payload_hashes["mixed-1.5-decode-no-gate.pkl"],
    )
    missing = missing_payload["result"]
    delayed = delayed_payload["result"]
    no_gate = no_gate_payload["result"]
    if missing.cell.cell_id != MISSING_CELL or delayed.cell != missing.cell:
        raise ValueError("repair payload cell mismatch")
    if no_gate.cell.cell_id != "MIXED-1.5x-DECODE_NO_GATE":
        raise ValueError("G12-3 baseline payload cell mismatch")
    trace_sha = repair_manifest.get("trace_sha256")
    for payload in (missing_payload, delayed_payload, no_gate_payload):
        metadata = payload.get("trace_metadata")
        if not isinstance(metadata, dict) or metadata.get("trace_sha256") != trace_sha:
            raise ValueError("repair payload trace mismatch")
    if missing.decode_report is None or no_gate.decode_report is None:
        raise ValueError("G12-3 repair requires DecodeRunReport")

    invariance: list[dict[str, Any]] = []
    samples: dict[str, m12.CellResult] = {}
    for cell_id in INVARIANCE_CELLS:
        name = f"invariance/{cell_id}.pkl"
        payload = _load_pickle(repair / name, payload_hashes[name])
        result = payload["result"]
        if result.cell.cell_id != cell_id:
            raise ValueError(f"mislabeled invariance payload: {cell_id}")
        metadata = payload.get("trace_metadata")
        if not isinstance(metadata, dict) or metadata.get("trace_sha256") != trace_sha:
            raise ValueError(f"invariance payload trace mismatch: {cell_id}")
        samples[cell_id] = result

    primary_rows = _csv_payload_rows(base_snapshot["primary.csv"])
    constraints_rows = _csv_payload_rows(base_snapshot["constraints.csv"])
    explanation_rows = _csv_payload_rows(base_snapshot["explanation.csv"])
    if len(primary_rows) != 44 or any(
        row["cell_id"] == MISSING_CELL for row in primary_rows
    ):
        raise ValueError("base primary grid is not the expected 44-row partial grid")
    base_primary = _index(primary_rows)
    base_constraints = _index(constraints_rows)
    base_explanation = _index(explanation_rows)
    for cell_id, result in samples.items():
        checks = {
            "primary.csv": _mapped_row(result, m12._primary_row)
            == base_primary[cell_id],
            "constraints.csv": _mapped_row(result, m12._constraint_row)
            == base_constraints[cell_id],
            "explanation.csv": _mapped_row(result, m12._explanation_row)
            == base_explanation[cell_id],
        }
        if not all(checks.values()):
            raise ValueError(f"invariance gate failed: {cell_id}: {checks}")
        if result.decision_fingerprint != base_explanation[cell_id][
            "decision_fingerprint"
        ]:
            raise ValueError(f"decision fingerprint mismatch: {cell_id}")
        invariance.append(
            {
                "cell_id": cell_id,
                "row_bytes_equal": True,
                "decision_fingerprint": result.decision_fingerprint,
                "artifact_sha256": payload_hashes[f"invariance/{cell_id}.pkl"],
            }
        )
    primary_rows.append(_mapped_row(missing, m12._primary_row))
    constraints_rows.append(_mapped_row(missing, m12._constraint_row))
    explanation_rows.append(_mapped_row(missing, m12._explanation_row))
    order = {
        cell.cell_id: index
        for index, cell in enumerate(m12.build_cell_plan(set()))
    }
    primary_rows.sort(key=lambda row: order[row["cell_id"]])
    constraints_rows.sort(key=lambda row: order[row["cell_id"]])
    explanation_rows.sort(key=lambda row: order[row["cell_id"]])

    reconstructed = _reconstruct_results(
        primary_rows,
        _index(constraints_rows),
        _index(explanation_rows),
        missing,
    )

    artifacts = {
        name: base_snapshot[name]
        for name in base_manifest["files"]
        if name != "MANIFEST.json"
    }
    artifacts["primary.csv"] = m12._csv_mapping_bytes(primary_rows)
    artifacts["constraints.csv"] = m12._csv_mapping_bytes(constraints_rows)
    artifacts["explanation.csv"] = m12._csv_mapping_bytes(explanation_rows)
    old_failure_header = base_snapshot["failures.csv"].decode().splitlines()[0]
    artifacts["failures.csv"] = f"{old_failure_header}\n".encode()
    artifacts["pareto.json"] = m12._json_bytes(
        m12._pareto(reconstructed, expected_cells=m12.build_cell_plan(set()))
    )
    artifacts["crossovers.csv"] = m12._csv_mapping_bytes(
        m12._crossovers(reconstructed, expected_cells=m12.build_cell_plan(set()))
    )

    g12_3 = json.loads(base_snapshot["gates/g12-3.json"])
    verdict = m12.evaluate_g12_3(
        no_gate=no_gate.decode_report,
        candidate=missing.decode_report,
        arrival_scale=1.5,
    )
    causal_row = {
        "regime": "MIXED",
        "arrival_scale": 1.5,
        "strategy": "DECODE_CAUSAL",
        "status": verdict.conclusion,
        "passed": verdict.passed,
        "sensitivity_passed": verdict.passed,
        "deployable": verdict.deployable_conclusion,
        "canonical_verdict": asdict(verdict),
        "attempt_count": missing.attempt_count,
        "retry_count": missing.retry_count,
        "congestion_action": missing.congestion_action,
        "gated_retry_count": missing.gated_retry_count,
    }
    _replace_group(
        g12_3,
        "cells",
        causal_row,
        lambda row: row.get("regime") == "MIXED"
        and row.get("strategy") == "DECODE_CAUSAL",
    )
    g12_3["retry_pressure_covered"] = missing.gated_retry_count > 0
    causal = [row for row in g12_3["cells"] if row.get("strategy") == "DECODE_CAUSAL"]
    g12_3["overall_verdict"] = (
        "PASS"
        if causal and all(bool(row.get("passed")) for row in causal)
        else "NARROW_OVERLOAD_ONLY"
    )
    artifacts["gates/g12-3.json"] = m12._json_bytes(g12_3)

    attribution = json.loads(base_snapshot["attribution.json"])
    priced = samples["MIXED-1.5x-PRICED_SPILL"]
    partial = m12._attribution(
        [priced, missing], expected_cells=(priced.cell, missing.cell)
    )
    partial_records = partial.get("records")
    if not isinstance(partial_records, list):
        raise ValueError("partial attribution records must be a list")
    decode_record = next(
        row
        for row in partial_records
        if isinstance(row, dict) and row.get("single_switch") == "DECODE_CREDITS"
    )
    _replace_group(
        attribution,
        "records",
        decode_record,
        lambda row: row.get("regime") == "MIXED"
        and row.get("arrival_scale") == 1.5
        and row.get("single_switch") == "DECODE_CREDITS",
    )
    attribution["overall_verdict"] = "COMPLETE"
    artifacts["attribution.json"] = m12._json_bytes(attribution)

    visibility = json.loads(base_snapshot["falsification/visibility-delay.json"])
    visibility_row = {
        "cell_id": MISSING_CELL,
        "delta_work": 1.0,
        "decision_fingerprint_before": missing.decision_fingerprint,
        "decision_fingerprint_after": delayed.decision_fingerprint,
        "decision_sequence_unchanged": missing.decision_log == delayed.decision_log,
        "census_not_newer": None,
        "census_age_before_work": None,
        "census_age_after_work": None,
        "offered_requests_conserved": (
            delayed.offered_requests == missing.offered_requests
        ),
        "offered_tokens_conserved": delayed.offered_tokens == missing.offered_tokens,
        "work_conserved": m12.math.isclose(delayed.total_work, delayed.accounted_work),
    }
    rows = [row for row in visibility["rows"] if row["cell_id"] != MISSING_CELL]
    rows.append(visibility_row)
    visibility["rows"] = sorted(rows, key=lambda row: row["cell_id"])
    artifacts["falsification/visibility-delay.json"] = m12._json_bytes(visibility)

    config = json.loads(base_snapshot["config.json"])
    config.update(
        completed_cell_count=54,
        failed_cell_count=0,
        repair_mode="MIXED_PROVENANCE_SINGLE_CELL",
    )
    artifacts["config.json"] = m12._json_bytes(config)

    tool_sha256 = _sha256(Path(__file__).read_bytes())
    repair_evidence = {
        "schema_version": "m12-repair-evidence-v1",
        "base_run": {"git_sha": BASE_SHA, "cell_count": 53},
        "repair_run": {
            "git_sha": repair_manifest["git_sha"],
            "cell_ids": [MISSING_CELL, "MIXED-1.5x-DECODE_NO_GATE"],
            "python_version": repair_manifest["python_version"],
            "attestation_basis": repair_manifest["attestation_basis"],
            "trace_sha256": repair_manifest["trace_sha256"],
            "source_sha256": repair_manifest["source_sha256"],
            "randomness_contract": "DETERMINISTIC_FIXED_RUNNER_INPUT",
            "payload_sha256": repair_manifest["payloads"],
        },
        "invariance_evidence": {
            "verdict": "PASS",
            "method": "CSV_ROW_BYTES_AND_DECISION_FINGERPRINT",
            "samples": invariance,
        },
        "merge": {
            "tool_sha256": tool_sha256,
            "base_manifest_sha256": _sha256(base_manifest_bytes),
            "derived_recompute_scope": {
                "full_rows": ["pareto.json", "crossovers.csv"],
                "affected_group": [
                    "gates/g12-3.json:MIXED-1.5x",
                    "attribution.json:MIXED-1.5x-DECODE_CREDITS",
                    "falsification/visibility-delay.json:MIXED-1.5x-DECODE_CAUSAL",
                ],
            },
        },
    }
    artifacts["diagnostics/repair-evidence.json"] = m12._json_bytes(repair_evidence)
    artifacts["provenance.json"] = m12._json_bytes(
        {
            "schema_version": "m12-mixed-provenance-v1",
            "git_sha": repair_manifest["git_sha"],
            "git_dirty": False,
            "merge_tool_sha256": tool_sha256,
            "base_provenance": provenance,
            "repair_evidence_sha256": _sha256(
                artifacts["diagnostics/repair-evidence.json"]
            ),
        }
    )
    artifacts["MANIFEST.json"] = m12._manifest_bytes(artifacts)
    return artifacts


def _tree_digest(artifacts: Mapping[str, bytes]) -> str:
    return _sha256(
        b"".join(
            name.encode() + b"\0" + payload
            for name, payload in sorted(artifacts.items())
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--repair-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    artifacts = build_repair(args.base, args.repair, args.repair_manifest)
    print(
        json.dumps(
            {"file_count": len(artifacts), "tree_sha256": _tree_digest(artifacts)},
            sort_keys=True,
        )
    )
    if not args.dry_run:
        m12._write_atomic(args.output, artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
