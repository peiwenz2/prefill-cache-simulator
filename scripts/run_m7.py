#!/usr/bin/env python3
"""Run M7 shared-decoder lease and recovery experiments."""

from __future__ import annotations

import csv
from pathlib import Path

from prefill_cache_sim.config import git_provenance
from prefill_cache_sim.continuation import ContinuationPrefixBuilder
from prefill_cache_sim.decode_lease import (
    DecodePolicy,
    LeaseConfig,
    RecoveryMode,
    simulate_decode_leases,
)
from prefill_cache_sim.trace import load_trace, to_simulation_requests


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    loaded = load_trace(root / "mooncake_trace.jsonl", block_size_tokens=512)
    requests = to_simulation_requests(loaded)
    builder = ContinuationPrefixBuilder(
        loaded.sha256,
        512,
        {request.logical_request_id: index for index, request in enumerate(requests)},
    )
    provenance = git_provenance(root)
    cases = [
        (DecodePolicy.D0, RecoveryMode.R0, 4, False),
        (DecodePolicy.D1, RecoveryMode.R0, 4, False),
        (DecodePolicy.D1_5, RecoveryMode.R0, 4, False),
        (DecodePolicy.D0, RecoveryMode.R0, 1, False),
        (DecodePolicy.D1, RecoveryMode.R0, 1, False),
        (DecodePolicy.D0, RecoveryMode.R0, 2, False),
        (DecodePolicy.D1, RecoveryMode.R0, 2, False),
        (DecodePolicy.D0, RecoveryMode.R0, 8, False),
        (DecodePolicy.D1, RecoveryMode.R0, 8, False),
        (DecodePolicy.D1, RecoveryMode.R0, 4, True),
        (DecodePolicy.D1, RecoveryMode.R1, 4, True),
    ]
    rows: list[dict[str, object]] = []
    for policy, recovery, decode_nodes, migrate in cases:
        capacity = 256
        case_label = (
            f"{policy.value} {recovery.value} D={decode_nodes} migrate={migrate}"
        )
        print(f"running {case_label}", flush=True)
        config = LeaseConfig(
            policy,
            recovery,
            prefill_nodes=4,
            decode_nodes=decode_nodes,
            local_capacity_blocks=capacity,
            migrate_on_boundary=migrate,
        )
        report = simulate_decode_leases(requests, builder, config)
        rows.append(
            {
                "case_id": (
                    f"M7-{policy.value}-{recovery.value}-D{decode_nodes}"
                    f"-M{int(migrate)}"
                ),
                "policy": policy.value,
                "recovery": recovery.value,
                "local_capacity_blocks": capacity,
                "migrate_on_boundary": migrate,
                "prefill_nodes": config.prefill_nodes,
                "decode_nodes": config.decode_nodes,
                "lease_schedule": "/".join(map(str, config.fixed_schedule)),
                "max_rounds": config.max_rounds,
                "strict_completed_goodput": report.strict_completed_goodput,
                "completed_goodput": report.completed_goodput,
                "completed_requests": report.completed_requests,
                "completed_tokens": report.completed_tokens,
                "partial_tokens": report.partial_tokens,
                "ttft_p95": report.ttft_p95,
                "short_ttft_p95": report.short_ttft_p95,
                "recovery_stall_work": report.recovery_stall_work,
                "recovery_p95": report.recovery_p95,
                "decode_waste_work": report.decode_waste_work,
                "jain_fairness": report.jain_fairness,
                "boundaries_per_completed_request": (
                    report.boundaries_per_completed_request
                ),
                "trace_sha256": loaded.sha256,
                "git_sha": provenance.sha or "",
                "git_dirty": provenance.dirty,
                "metric_unit": "NORMALIZED_WORK",
            }
        )
    output = root / "results/m7/results.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
