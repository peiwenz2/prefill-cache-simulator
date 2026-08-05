#!/usr/bin/env python3
"""Run bounded M5 KVS bandwidth/capacity/threshold sensitivity."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import fields, replace
from pathlib import Path

from prefill_cache_sim.config import git_provenance
from prefill_cache_sim.kvs import TopologyMode
from prefill_cache_sim.m5_experiment import M5Case, M5Result, run_m5_case
from prefill_cache_sim.trace import load_trace, to_simulation_requests


def matrix() -> list[M5Case]:
    base = M5Case("M5-BASE")
    return (
        [base]
        + [
            replace(base, case_id=f"M5-BW-{bandwidth}", bandwidth_gbps=bandwidth)
            for bandwidth in (25, 50, 100, 200)
        ]
        + [
            replace(base, case_id=f"M5-CAP-{capacity}", remote_capacity_blocks=capacity)
            for capacity in (10_000, 50_000, 200_000)
        ]
        + [
            replace(
                base,
                case_id=f"M5-THRESHOLD-{threshold}",
                kvcache_balancing_threshold=threshold,
            )
            for threshold in (1.0, 1.5, 2.0, 4.0)
        ]
        + [
            replace(base, case_id=f"M5-OVERLAP-{overlap}", overlap_ratio=overlap)
            for overlap in (0, 0.5, 0.9)
        ]
        + [
            replace(base, case_id=f"M5-BYTES-{size}", bytes_per_token=size)
            for size in (16_384, 65_536, 327_680)
        ]
        + [
            replace(base, case_id=f"M5-LAT-{latency}", fixed_latency_ms=latency)
            for latency in (0.05, 0.2, 1.0, 5.0)
        ]
        + [
            replace(
                base,
                case_id=f"M5-TIMEOUT-{probability}",
                lookup_timeout_probability=probability,
            )
            for probability in (0.01, 0.05, 0.2)
        ]
        + [
            replace(
                base,
                case_id=f"M5-HYBRID-{capacity}",
                topology_mode=TopologyMode.HYBRID,
                local_dram_capacity_blocks=capacity,
            )
            for capacity in (25_000, 50_000, 100_000)
        ]
        + [
            replace(
                base,
                case_id="M5-WORST-CORNER",
                bandwidth_gbps=25,
                bytes_per_token=327_680,
                overlap_ratio=0,
            )
        ]
        + [
            replace(
                base,
                case_id="M5-LOCAL-ONLY",
                topology_mode=TopologyMode.LOCAL_ONLY,
            )
        ]
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    loaded = load_trace(root / "mooncake_trace.jsonl", block_size_tokens=512)
    requests = to_simulation_requests(loaded)
    provenance = git_provenance(root)
    output = root / "results/m5/results.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    cases = matrix()
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[field.name for field in fields(M5Result)]
        )
        writer.writeheader()
        for index, case in enumerate(cases, 1):
            print(f"[{index}/{len(cases)}] {case.case_id}", flush=True)
            config_sha = hashlib.sha256(
                json.dumps(
                    {
                        field.name: str(getattr(case, field.name))
                        for field in fields(case)
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            writer.writerow(
                run_m5_case(
                    case,
                    requests,
                    trace_sha256=loaded.sha256,
                    config_sha256=config_sha,
                    git_sha=provenance.sha,
                    git_dirty=provenance.dirty,
                ).to_dict()
            )
            handle.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
