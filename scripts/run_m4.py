#!/usr/bin/env python3
"""Run the bounded M4 experiment matrix with CSV checkpointing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import fields, replace
from pathlib import Path

from prefill_cache_sim.config import git_provenance
from prefill_cache_sim.domain import CacheVisibility, ReplayMode, ViewMode
from prefill_cache_sim.experiment import ExperimentCase, ExperimentResult, run_case
from prefill_cache_sim.trace import load_trace, to_simulation_requests

SELECTORS = (
    "S0_RANDOM",
    "S1_ROUND_ROBIN",
    "S2_LEAST_WORK",
    "S3_GB_PREFIX_BUCKET",
    "S4_SESSION_AFFINITY",
    "S5_CENTRALIZED_MASTER_TTFT",
    "S6_CALIBRATED_TTFT",
)
TOP3 = ("S3_GB_PREFIX_BUCKET", "S4_SESSION_AFFINITY", "S5_CENTRALIZED_MASTER_TTFT")


def base_case(case_id: str, selector: str) -> ExperimentCase:
    return ExperimentCase(
        case_id=case_id,
        selector_id=selector,
        eviction_id="E1_LRU",
        node_count=4,
        capacity_mode="FIXED_TOTAL",
        capacity_blocks=50_000,
        replay_mode=ReplayMode.QUEUED,
        visibility=CacheVisibility.INSERT_AT_COMPLETION,
        view_mode=ViewMode.DELAYED,
        view_delay_ms=100,
        view_loss_probability=0,
        replay_speed=1,
        seed=713,
        prefill_uncached_token_ms=0.06,
        prefill_concurrency_per_node=1,
        prefix_anchor_k=2,
        soft_alpha=1.25,
        minimum_shared_blocks=2,
        family_size_cap=128,
        hot_block_percentile=99,
    )


def matrix() -> list[ExperimentCase]:
    cases = [base_case(f"A1-{selector}", selector) for selector in SELECTORS]
    cases += [
        replace(base_case(f"K-S3-k{k}", "S3_GB_PREFIX_BUCKET"), prefix_anchor_k=k)
        for k in (1, 2, 4, 8, 12, 16)
    ]
    cases += [
        replace(
            base_case(f"A2-{selector}-{eviction}", selector),
            eviction_id=eviction,
        )
        for selector in ("S3_GB_PREFIX_BUCKET", "S5_CENTRALIZED_MASTER_TTFT")
        for eviction in (
            "E0_FIFO",
            "E1_LRU",
            "E2_SLRU",
            "E3_SECOND_HIT_ADMISSION",
        )
    ]
    cases += [
        replace(
            base_case(f"SEED-{selector}-{seed}", selector),
            seed=seed,
        )
        for selector in TOP3
        for seed in range(713, 721)
    ]
    cases += [
        replace(base_case(f"STALE-{selector}-{delay}", selector), view_delay_ms=delay)
        for selector in TOP3
        for delay in (0, 50, 500, 5_000)
    ]
    cases += [
        replace(
            base_case(f"CAP-TOTAL-{selector}-{capacity}", selector),
            capacity_blocks=capacity,
        )
        for selector in TOP3
        for capacity in (10_000, 30_000, 100_000)
    ]
    cases += [
        replace(
            base_case(f"CAP-PER-{selector}-{capacity}", selector),
            capacity_mode="FIXED_PER_NODE",
            capacity_blocks=capacity,
        )
        for selector in TOP3
        for capacity in (5_000, 10_000, 30_000)
    ]
    cases += [
        replace(
            base_case(f"N-{selector}-{nodes}", selector),
            node_count=nodes,
        )
        for selector in TOP3
        for nodes in (2, 8)
    ]
    cases += [
        replace(
            base_case(f"S4-CAP-{cap}", "S4_SESSION_AFFINITY"),
            family_size_cap=cap,
        )
        for cap in (8, 32, 128, 512)
    ]
    cases += [
        replace(
            base_case(f"S4-HOT-{percentile}", "S4_SESSION_AFFINITY"),
            hot_block_percentile=percentile,
        )
        for percentile in (90, 95, 99, 100)
    ]
    cases += [
        replace(
            base_case(f"S5-DISCOUNT-{discount}", "S5_CENTRALIZED_MASTER_TTFT"),
            cache_discount=discount,
        )
        for discount in (0.25, 0.7, 1.0, 2.0)
    ]
    cases += [
        replace(
            base_case(f"S5-BAND-{band}", "S5_CENTRALIZED_MASTER_TTFT"),
            similar_band_ratio=band,
        )
        for band in (0.01, 0.05, 0.1, 0.2)
    ]
    return cases


def case_sha(case: ExperimentCase) -> str:
    payload = json.dumps(
        {field.name: str(getattr(case, field.name)) for field in fields(case)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=Path("mooncake_trace.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/m4/results.csv"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    loaded = load_trace(args.trace, block_size_tokens=512)
    requests = to_simulation_requests(loaded)
    provenance = git_provenance(Path(__file__).resolve().parents[1])
    if provenance.dirty:
        raise RuntimeError("refusing to run M4 matrix from a dirty worktree")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()
    if args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as handle:
            completed = {row["case_id"] for row in csv.DictReader(handle)}
    selected = matrix()[: args.limit] if args.limit else matrix()
    fieldnames = [field.name for field in fields(ExperimentResult)]
    write_header = not args.output.exists()
    with args.output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for index, case in enumerate(selected, 1):
            if case.case_id in completed:
                continue
            print(f"[{index}/{len(selected)}] {case.case_id}", flush=True)
            result = run_case(
                case,
                requests,
                trace_sha256=loaded.sha256,
                config_sha256=case_sha(case),
                git_sha=provenance.sha,
                git_dirty=provenance.dirty,
            )
            if result.offered_load_ratio >= 1:
                raise RuntimeError(
                    f"unstable offered load for {case.case_id}: "
                    f"rho={result.offered_load_ratio:.3f}"
                )
            writer.writerow(result.to_dict())
            handle.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
