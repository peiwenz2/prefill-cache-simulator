#!/usr/bin/env python3
"""Run the normalized minimum-prefill resource-sizing grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from prefill_cache_sim.config import git_provenance  # noqa: E402
from prefill_cache_sim.m12_metrics import SERVICE_REGIMES  # noqa: E402
from prefill_cache_sim.m12_placement import PlacementWorkload  # noqa: E402
from prefill_cache_sim.m12_sizing import (  # noqa: E402
    GateObservation,
    SizingCell,
    SizingGates,
    SizingRunRecord,
    SizingTopology,
    run_sizing_cell,
    select_minimum,
)
from scripts.run_m12_placement import (  # noqa: E402
    ARRIVAL_SCALE,
    OBSERVATION_END_WORK,
    TIER_SLO_WORK,
    TRUTH_BASIS,
    build_trace_workload,
)

SCHEMA_VERSION = "m12-sizing-v2.3"
DEFAULT_P_COUNTS = tuple(range(1, 9))
DEFAULT_TOPOLOGIES = tuple(SizingTopology)
BASELINE_GATES = SizingGates(1.0, 0.80, 0.90, 20_000.0, 1_000_000.0)
STRICT_95_GATES = SizingGates(1.0, 0.95, 0.90, 20_000.0, 1_000_000.0)
TIER_FLOOR_SENSITIVITY = tuple(value / 100 for value in range(70, 99))
DEADLINE_MULTIPLIERS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
SIZING_OBSERVATION_END_WORK = OBSERVATION_END_WORK / ARRIVAL_SCALE
RESULT_FIELDS = (
    "topology",
    "p_count",
    "completion_ratio",
    "minimum_tier_slo_attainment",
    "strict_slo_attainment",
    "standard_slo_attainment",
    "relaxed_slo_attainment",
    "jain_fairness",
    "p_queue_p95_work",
    "kvs_bytes_per_work",
    "baseline_failed_gates",
    "strict_95_failed_gates",
    "strict_useful_token_goodput",
    "strict_useful_output_token_goodput",
    "request_goodput",
    "token_hit_rate",
    "local_hit_tokens",
    "remote_hit_tokens",
    "recompute_tokens",
    "p_normalized_utilization",
    "d_normalized_utilization",
    "p_queue_wait_p50_work",
    "p_queue_wait_p95_work",
    "p_queue_wait_p99_work",
    "final_alive_blocks_mean_per_p",
    "final_alive_blocks_min_per_p",
    "final_alive_blocks_max_per_p",
    "normalized_kvs_work",
    "decision_fingerprint",
)
_FORK_WORKLOAD: PlacementWorkload | None = None


def build_plan(
    p_counts: Sequence[int] = DEFAULT_P_COUNTS,
    topologies: Sequence[SizingTopology] = DEFAULT_TOPOLOGIES,
) -> tuple[tuple[SizingTopology, int], ...]:
    counts = tuple(sorted(set(p_counts)))
    if not counts or any(type(value) is not int or value <= 0 for value in counts):
        raise ValueError("P counts must be non-empty plain positive integers")
    modes = tuple(sorted(set(topologies), key=lambda value: value.value))
    if not modes:
        raise ValueError("topology grid must be non-empty")
    return tuple((topology, count) for topology in modes for count in counts)


def run_grid(
    trace_path: Path,
    *,
    p_counts: Sequence[int] = DEFAULT_P_COUNTS,
    topologies: Sequence[SizingTopology] = DEFAULT_TOPOLOGIES,
    limit: int | None = None,
    jobs: int = 1,
) -> tuple[list[SizingRunRecord], Mapping[str, object]]:
    workload, trace_metadata = build_trace_workload(trace_path)
    workload = restore_raw_trace_timestamps(workload)
    unique_keys = len(
        {
            key
            for request in workload
            for key in request.prefix_cache_keys
        }
    )
    plan = build_plan(p_counts, topologies)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        plan = plan[:limit]
    if type(jobs) is not int or jobs <= 0:
        raise ValueError("jobs must be a plain positive integer")
    args = tuple(
        (topology, p_count, max(1, unique_keys))
        for topology, p_count in plan
    )
    records: list[SizingRunRecord] = []
    if jobs == 1:
        for index, value in enumerate(args, 1):
            print(f"[{index}/{len(plan)}] {value[0].value} P={value[1]}", flush=True)
            records.append(_run_grid_cell(value, workload))
    else:
        global _FORK_WORKLOAD
        _FORK_WORKLOAD = workload
        with ProcessPoolExecutor(
            max_workers=min(jobs, len(args)), mp_context=get_context("fork")
        ) as executor:
            pending = {
                executor.submit(_run_grid_cell_wire, value): value for value in args
            }
            for index, future in enumerate(as_completed(pending), 1):
                value = pending[future]
                records.append(_record_from_wire(future.result()))
                print(
                    f"[{index}/{len(plan)}] done {value[0].value} P={value[1]}",
                    flush=True,
                )
    return records, trace_metadata


def _run_grid_cell(
    value: tuple[SizingTopology, int, int],
    workload: PlacementWorkload | None = None,
) -> SizingRunRecord:
    topology, p_count, capacity = value
    active_workload = workload if workload is not None else _FORK_WORKLOAD
    if active_workload is None:
        raise RuntimeError("fork worker did not inherit the sizing workload")
    return run_sizing_cell(
        active_workload,
        p_count=p_count,
        topology=topology,
        gates=BASELINE_GATES,
        regime=SERVICE_REGIMES[2],
        observation_end_work=SIZING_OBSERVATION_END_WORK,
        tier_slo_work=TIER_SLO_WORK,
        cache_capacity_entries_per_p=capacity,
        decode_node_count=8,
    )


def _run_grid_cell_wire(value: tuple[SizingTopology, int, int]) -> dict[str, object]:
    return _record_to_wire(_run_grid_cell(value))


def _record_to_wire(record: SizingRunRecord) -> dict[str, object]:
    observation = record.cell.observation
    return {
        "topology": record.cell.topology.value,
        "p_count": record.cell.p_count,
        "observation": (
            observation.completion_ratio,
            observation.minimum_tier_slo_attainment,
            observation.jain_fairness,
            observation.p_queue_p95_work,
            observation.kvs_bytes_per_work,
        ),
        "failed_gates": record.cell.failed_gates,
        "scalars": (
            record.strict_useful_token_goodput,
            record.strict_useful_output_token_goodput,
            record.request_goodput,
            record.token_hit_rate,
            record.local_hit_tokens,
            record.remote_hit_tokens,
            record.recompute_tokens,
            record.p_normalized_utilization,
            record.d_normalized_utilization,
            record.p_queue_wait_p50_work,
            record.p_queue_wait_p95_work,
            record.p_queue_wait_p99_work,
            record.final_alive_blocks_mean_per_p,
            record.final_alive_blocks_min_per_p,
            record.final_alive_blocks_max_per_p,
            record.normalized_kvs_work,
        ),
        "per_tier_slo_attainment": dict(record.per_tier_slo_attainment),
        "per_tier_completion_elapsed_work": {
            tier: tuple(values)
            for tier, values in record.per_tier_completion_elapsed_work.items()
        },
        "completion_ledger": record.completion_ledger,
        "decision_fingerprint": record.decision_fingerprint,
    }


def _record_from_wire(value: Mapping[str, object]) -> SizingRunRecord:
    observation = GateObservation(*value["observation"])
    cell = SizingCell(
        int(value["p_count"]),
        SizingTopology(str(value["topology"])),
        observation,
        tuple(value["failed_gates"]),
    )
    return SizingRunRecord(
        cell,
        *value["scalars"],
        value["per_tier_slo_attainment"],
        value["per_tier_completion_elapsed_work"],
        tuple(value["completion_ledger"]),
        str(value["decision_fingerprint"]),
    )


def build_artifacts(
    records: Sequence[SizingRunRecord],
    trace_metadata: Mapping[str, object],
) -> dict[str, bytes]:
    records = tuple(
        sorted(
            records,
            key=lambda value: (value.cell.topology.value, value.cell.p_count),
        )
    )
    identities = tuple((value.cell.topology, value.cell.p_count) for value in records)
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate sizing cells cannot be published")
    baseline_cells = tuple(record.cell for record in records)
    strict_cells = tuple(
        _reevaluate(record.cell, STRICT_95_GATES) for record in records
    )
    rows = [_record_row(record, strict_cell) for record, strict_cell in zip(
        records, strict_cells, strict=True
    )]
    baseline_verdicts = _topology_verdicts(baseline_cells, BASELINE_GATES)
    strict_verdicts = _topology_verdicts(strict_cells, STRICT_95_GATES)
    provenance = git_provenance(ROOT)
    if provenance.dirty:
        raise RuntimeError("refusing to publish sizing artifacts from a dirty tree")
    contract = {
        "schema_version": SCHEMA_VERSION,
        "claim_basis": "NORMALIZED_SIMULATION",
        "wall_clock_claim": False,
        "gpu_capacity_claim": False,
        "trace_truth_basis": TRUTH_BASIS,
        "cache_capacity_mode": "NO_EVICTION_CAPACITY_CONTROL_PER_P_NODE",
        "decode_node_count": 8,
        "service_regime": SERVICE_REGIMES[2].regime_id.value,
        "service_costs": asdict(SERVICE_REGIMES[2]),
        "timestamp_mapping": "raw trace timestamp_ms, no replay slowdown",
        "observation_end_work": SIZING_OBSERVATION_END_WORK,
        "tier_slo_work": TIER_SLO_WORK,
        "baseline_gates": asdict(BASELINE_GATES),
        "strict_95_gates": asdict(STRICT_95_GATES),
        "zero_transfer_price_control_deployable": False,
        "queue_wait_definition": "actual ATTEMPT_READY to PREFILL_START elapsed work",
        "utilization_semantics": (
            "normalized issued GPU work divided by modeled pool capacity; not MFU"
        ),
        "alive_blocks_semantics": (
            "resident cache keys at observation end; mean/min/max across P nodes"
        ),
        "deadline_frontier_semantics": (
            "evaluator-only: scale tier deadlines post-hoc and recompute per-tier "
            "attainment plus Jain fairness while routing, execution, costs, queue "
            "gates, and KVS gates remain frozen"
        ),
    }
    summary = {
        "baseline": baseline_verdicts,
        "strict_95": strict_verdicts,
        "interpretation": (
            "minimum P under the frozen normalized model; not production GPU sizing"
        ),
    }
    provenance_payload = {
        **trace_metadata,
        "git_sha": provenance.sha,
        "git_dirty": provenance.dirty,
        "sizing_cell_count": len(records),
    }
    artifacts = {
        "contract.json": _json_bytes(contract),
        "cells.csv": _csv_bytes(rows),
        "threshold-frontier.csv": _csv_bytes(
            _threshold_frontier_rows(baseline_cells),
            ("tier_slo_floor", "topology", "minimum_feasible_p"),
        ),
        "deadline-frontier.csv": _csv_bytes(
            _deadline_frontier_rows(records),
            (
                "deadline_multiplier",
                "strict_deadline_work",
                "standard_deadline_work",
                "relaxed_deadline_work",
                "topology",
                "minimum_feasible_p",
            ),
        ),
        "verdict.json": _json_bytes(summary),
        "provenance.json": _json_bytes(provenance_payload),
    }
    artifacts["MANIFEST.json"] = _manifest_bytes(artifacts)
    return artifacts


def _threshold_frontier_rows(
    cells: Sequence[SizingCell],
) -> list[dict[str, object]]:
    """Post-hoc tier-floor sensitivity; all other gates remain frozen."""
    rows: list[dict[str, object]] = []
    for floor in TIER_FLOOR_SENSITIVITY:
        gates = replace(BASELINE_GATES, minimum_tier_slo_attainment=floor)
        reevaluated = tuple(_reevaluate(cell, gates) for cell in cells)
        for topology in sorted(SizingTopology, key=lambda value: value.value):
            verdict = select_minimum(
                reevaluated,
                gates,
                include_control=not topology.deployable,
                required_topologies=(topology,),
            )
            rows.append(
                {
                    "tier_slo_floor": floor,
                    "topology": topology.value,
                    "minimum_feasible_p": (
                        verdict.minimum_feasible_p
                        if verdict.minimum_feasible_p is not None
                        else "GRID_EXHAUSTED"
                    ),
                }
            )
    return rows


def _deadline_frontier_rows(
    records: Sequence[SizingRunRecord],
) -> list[dict[str, object]]:
    """Evaluator-only deadline sensitivity; routing and execution stay frozen."""
    rows: list[dict[str, object]] = []
    for multiplier in DEADLINE_MULTIPLIERS:
        scaled_cells: list[SizingCell] = []
        for record in records:
            per_tier = {
                tier: (
                    sum(
                        elapsed <= TIER_SLO_WORK[tier] * multiplier
                        for elapsed in samples
                    )
                    / len(samples)
                    if samples
                    else 0.0
                )
                for tier, samples in record.per_tier_completion_elapsed_work.items()
            }
            tenant_demand: dict[str, int] = {}
            tenant_service: dict[str, int] = {}
            for tenant, _tier, demand, elapsed in record.completion_ledger:
                tenant_demand[tenant] = tenant_demand.get(tenant, 0) + demand
                if elapsed <= TIER_SLO_WORK[_tier] * multiplier:
                    tenant_service[tenant] = tenant_service.get(tenant, 0) + demand
            ratios = tuple(
                tenant_service.get(tenant, 0) / demand
                for tenant, demand in sorted(tenant_demand.items())
            )
            observation = replace(
                record.cell.observation,
                minimum_tier_slo_attainment=min(per_tier.values()),
                jain_fairness=_jain(ratios),
            )
            scaled_cells.append(
                replace(
                    record.cell,
                    observation=observation,
                    failed_gates=(),
                )
            )
        evaluated = tuple(_reevaluate(cell, BASELINE_GATES) for cell in scaled_cells)
        for topology in sorted(SizingTopology, key=lambda value: value.value):
            verdict = select_minimum(
                evaluated,
                BASELINE_GATES,
                include_control=True,
                required_topologies=(topology,),
            )
            rows.append(
                {
                    "deadline_multiplier": multiplier,
                    "strict_deadline_work": TIER_SLO_WORK["STRICT"] * multiplier,
                    "standard_deadline_work": TIER_SLO_WORK["STANDARD"] * multiplier,
                    "relaxed_deadline_work": TIER_SLO_WORK["RELAXED"] * multiplier,
                    "topology": topology.value,
                    "minimum_feasible_p": (
                        verdict.minimum_feasible_p
                        if verdict.minimum_feasible_p is not None
                        else "GRID_EXHAUSTED"
                    ),
                }
            )
    return rows


def _jain(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    numerator = sum(values) ** 2
    denominator = len(values) * sum(value * value for value in values)
    return numerator / denominator if denominator else 0.0


def publish(output_dir: Path, artifacts: Mapping[str, bytes]) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}."
    ) as temporary:
        stage = Path(temporary)
        for relative, payload in artifacts.items():
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        if output_dir.exists():
            backup = output_dir.with_name(f".{output_dir.name}.previous")
            if backup.exists():
                raise RuntimeError(f"stale sizing backup exists: {backup}")
            os.replace(output_dir, backup)
            try:
                os.replace(stage, output_dir)
            except BaseException:
                os.replace(backup, output_dir)
                raise
            _remove_tree(backup)
        else:
            os.replace(stage, output_dir)


def _reevaluate(cell: SizingCell, gates: SizingGates) -> SizingCell:
    from prefill_cache_sim.m12_sizing import evaluate_gates

    evaluated = evaluate_gates(
        cell.observation,
        gates,
        p_count=cell.p_count,
        topology=cell.topology,
    )
    assert isinstance(evaluated, SizingCell)
    return evaluated


def restore_raw_trace_timestamps(workload: PlacementWorkload) -> PlacementWorkload:
    requests = tuple(
        replace(
            request,
            logical=replace(
                request.logical,
                arrival_work=request.logical.arrival_work / ARRIVAL_SCALE,
            ),
        )
        for request in workload
    )
    return PlacementWorkload(requests, workload.request_truth)


def _topology_verdicts(
    cells: Sequence[SizingCell], gates: SizingGates
) -> dict[str, object]:
    result: dict[str, object] = {}
    for topology in SizingTopology:
        selected = tuple(cell for cell in cells if cell.topology is topology)
        verdict = select_minimum(
            selected,
            gates,
            include_control=True,
            required_topologies=(topology,),
        )
        result[topology.value] = {
            "deployable": topology.deployable,
            "minimum_feasible_p": verdict.minimum_feasible_p,
            "grid_exhausted": verdict.grid_exhausted,
            "non_monotonic_p_counts": verdict.non_monotonic_p_counts,
            "predecessor_certificate": (
                _certificate_dict(verdict.predecessor_certificate)
                if verdict.predecessor_certificate
                else None
            ),
        }
    deployable = select_minimum(cells, gates)
    result["DEPLOYABLE_WINNER"] = {
        "minimum_feasible_p": deployable.minimum_feasible_p,
        "selected_topology": (
            deployable.selected_topology.value
            if deployable.selected_topology
            else None
        ),
        "grid_exhausted": deployable.grid_exhausted,
        "predecessor_certificates": tuple(
            _certificate_dict(value)
            for value in deployable.predecessor_certificates
        ),
    }
    return result


def _record_row(
    record: SizingRunRecord, strict_cell: SizingCell
) -> dict[str, object]:
    expected_tiers = {"STRICT", "STANDARD", "RELAXED"}
    if set(record.per_tier_slo_attainment) != expected_tiers:
        raise ValueError(
            "sizing artifact requires exactly STRICT, STANDARD, and RELAXED tiers"
        )
    observation = record.cell.observation
    return {
        "topology": record.cell.topology.value,
        "p_count": record.cell.p_count,
        "completion_ratio": observation.completion_ratio,
        "minimum_tier_slo_attainment": observation.minimum_tier_slo_attainment,
        "strict_slo_attainment": record.per_tier_slo_attainment["STRICT"],
        "standard_slo_attainment": record.per_tier_slo_attainment["STANDARD"],
        "relaxed_slo_attainment": record.per_tier_slo_attainment["RELAXED"],
        "jain_fairness": observation.jain_fairness,
        "p_queue_p95_work": observation.p_queue_p95_work,
        "kvs_bytes_per_work": observation.kvs_bytes_per_work,
        "baseline_failed_gates": "|".join(record.cell.failed_gates),
        "strict_95_failed_gates": "|".join(strict_cell.failed_gates),
        "strict_useful_token_goodput": record.strict_useful_token_goodput,
        "strict_useful_output_token_goodput": (
            record.strict_useful_output_token_goodput
        ),
        "request_goodput": record.request_goodput,
        "token_hit_rate": record.token_hit_rate,
        "local_hit_tokens": record.local_hit_tokens,
        "remote_hit_tokens": record.remote_hit_tokens,
        "recompute_tokens": record.recompute_tokens,
        "p_normalized_utilization": record.p_normalized_utilization,
        "d_normalized_utilization": record.d_normalized_utilization,
        "p_queue_wait_p50_work": record.p_queue_wait_p50_work,
        "p_queue_wait_p95_work": record.p_queue_wait_p95_work,
        "p_queue_wait_p99_work": record.p_queue_wait_p99_work,
        "final_alive_blocks_mean_per_p": record.final_alive_blocks_mean_per_p,
        "final_alive_blocks_min_per_p": record.final_alive_blocks_min_per_p,
        "final_alive_blocks_max_per_p": record.final_alive_blocks_max_per_p,
        "normalized_kvs_work": record.normalized_kvs_work,
        "decision_fingerprint": record.decision_fingerprint,
    }


def _certificate_dict(value) -> dict[str, object]:
    return {
        "p_count": value.p_count,
        "topology": value.topology.value,
        "failed_gates": value.failed_gates,
        "observed": dict(value.observed),
        "required": dict(value.required),
    }


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode()


def _csv_bytes(
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str] = RESULT_FIELDS,
) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def _manifest_bytes(artifacts: Mapping[str, bytes]) -> bytes:
    return _json_bytes(
        {
            "schema_version": "m12-sizing-manifest-v1",
            "files": {
                name: hashlib.sha256(payload).hexdigest()
                for name, payload in sorted(artifacts.items())
            },
        }
    )


def _remove_tree(path: Path) -> None:
    for entry in sorted(path.rglob("*"), reverse=True):
        if entry.is_dir():
            entry.rmdir()
        else:
            entry.unlink()
    path.rmdir()


def _parse_counts(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(","))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=ROOT / "mooncake_trace.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "results/m12-sizing")
    parser.add_argument("--p-counts", type=_parse_counts, default=DEFAULT_P_COUNTS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    records, trace_metadata = run_grid(
        args.trace,
        p_counts=args.p_counts,
        limit=args.limit,
        jobs=args.jobs,
    )
    publish(args.output, build_artifacts(records, trace_metadata))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
