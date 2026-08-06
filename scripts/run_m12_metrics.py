#!/usr/bin/env python3
"""Publish the frozen M12.1 metric contract and service-regime grid."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prefill_cache_sim.config import git_provenance  # noqa: E402
from prefill_cache_sim.m12_metrics import (  # noqa: E402
    MAX_TIER_REGRESSION_ABSOLUTE,
    MINIMUM_JAIN_FAIRNESS,
    MINIMUM_TIER_SLO_ATTAINMENT,
    SERVICE_REGIMES,
    AttemptOutcome,
    aggregate_m12_metrics,
)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _csv_bytes(rows: list[dict[str, object]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def _write_atomic(output: Path, artifacts: dict[str, bytes]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as staging_name:
        staging = Path(staging_name)
        ordered = [name for name in sorted(artifacts) if name != "MANIFEST.json"]
        ordered.append("MANIFEST.json")
        for name in ordered:
            path = staging / name
            path.write_bytes(artifacts[name])
            os.replace(path, output / name)


def _attempts(regime_index: int) -> list[AttemptOutcome]:
    regime = SERVICE_REGIMES[regime_index]
    attempts: list[AttemptOutcome] = []
    tenants = ("tenant-a", "tenant-b", "tenant-c")
    tiers = ("STRICT", "STANDARD", "RELAXED")
    for index, (tenant, tier) in enumerate(zip(tenants, tiers, strict=True)):
        p_work = regime.prefill_work(4096, batch_tokens=12_288)
        d_work = regime.decode_work(64, concurrent_sequences=3)
        if index == 1:
            attempts.append(
                AttemptOutcome(
                    f"logical-{index}",
                    0,
                    tenant,
                    tier,
                    0,
                    4096,
                    64,
                    8,
                    False,
                    False,
                    None,
                    p_work,
                    d_work / 8,
                    p_work,
                    d_work / 8,
                    32_768,
                )
            )
        attempts.append(
            AttemptOutcome(
                f"logical-{index}",
                1 if index == 1 else 0,
                tenant,
                tier,
                0,
                4096,
                64,
                64,
                True,
                True,
                800 + index,
                p_work,
                d_work,
                0,
                0,
                0,
            )
        )
    return attempts


def main() -> int:
    provenance = git_provenance(ROOT)
    grid_rows: list[dict[str, object]] = []
    validation: dict[str, object] = {}
    for regime_index, regime in enumerate(SERVICE_REGIMES):
        for batch_tokens in (4096, 16_384, 65_536):
            unit_work = regime.prefill_work(4096, batch_tokens=batch_tokens)
            grid_rows.append(
                {
                    "regime": regime.regime_id.value,
                    "dimension": "PREFILL",
                    "load": batch_tokens,
                    "unit_work": f"{unit_work:.12f}",
                    "truth_basis": "SYNTHETIC_SERVICE_REGIME",
                }
            )
        for concurrency in (1, 16, 64):
            unit_work = regime.decode_work(
                64, concurrent_sequences=concurrency
            )
            grid_rows.append(
                {
                    "regime": regime.regime_id.value,
                    "dimension": "DECODE",
                    "load": concurrency,
                    "unit_work": f"{unit_work:.12f}",
                    "truth_basis": "SYNTHETIC_SERVICE_REGIME",
                }
            )
        validation[regime.regime_id.value] = aggregate_m12_metrics(
            _attempts(regime_index),
            observation_start_work=0,
            observation_end_work=1000,
            prefill_nodes=2,
            decode_nodes=2,
        ).to_dict()

    contract = {
        "schema_version": "m12-metric-contract-v1",
        "truth_basis": "SYNTHETIC_SERVICE_REGIME",
        "primary_metric": "strict_useful_token_goodput",
        "efficiency_metric": "strict_useful_tokens_per_gpu_work",
        "fairness": {
            "minimum_tier_slo_attainment": MINIMUM_TIER_SLO_ATTAINMENT,
            "minimum_jain_fairness": MINIMUM_JAIN_FAIRNESS,
            "max_tier_regression_absolute": MAX_TIER_REGRESSION_ABSOLUTE,
        },
        "regimes": [asdict(regime) for regime in SERVICE_REGIMES],
        "claims": {
            "wall_clock_allowed": False,
            "mfu_allowed": False,
            "normalized_kill_final": True,
            "normalized_pass_provisional": True,
        },
        "provenance": {
            "git_sha": provenance.sha,
            "git_dirty": provenance.dirty,
        },
    }
    artifacts = {
        "contract.json": _json_bytes(contract),
        "regime_grid.csv": _csv_bytes(grid_rows),
        "validation.json": _json_bytes(validation),
    }
    artifacts["MANIFEST.json"] = _json_bytes(
        {
            "schema_version": "m12-metrics-manifest-v1",
            "files": {
                name: hashlib.sha256(content).hexdigest()
                for name, content in sorted(artifacts.items())
            },
        }
    )
    _write_atomic(ROOT / "results/m12-metrics", artifacts)
    print(
        json.dumps(
            {
                "regimes": len(SERVICE_REGIMES),
                "grid_rows": len(grid_rows),
                "validation_reports": len(validation),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
