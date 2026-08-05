#!/usr/bin/env python3
"""Run Phase-C protocol/SLO overlay sensitivity."""

from __future__ import annotations

import csv
from pathlib import Path

from prefill_cache_sim.config import git_provenance
from prefill_cache_sim.lifecycle import (
    LifecycleConfig,
    LifecycleReport,
    ProtocolId,
    simulate_lifecycle,
)
from prefill_cache_sim.slo import RelativeSloOverlay, Tier, TieredSloOverlay
from prefill_cache_sim.trace import load_trace, to_simulation_requests


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    loaded = load_trace(root / "mooncake_trace.jsonl", block_size_tokens=512)
    requests = to_simulation_requests(loaded)
    provenance = git_provenance(root)
    output = root / "results/m6/results.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for protocol in ProtocolId:
        rows.append(
            row(
                f"M6-REL-{protocol.value}",
                simulate_lifecycle(
                    requests,
                    RelativeSloOverlay(2.5, 2.5),
                    LifecycleConfig(protocol, 4, 4),
                ),
                seed=None,
                overlay="RELATIVE",
                trace_sha256=loaded.sha256,
                git_sha=provenance.sha,
                git_dirty=provenance.dirty,
            )
        )
    for seed in range(713, 721):
        for protocol in (
            ProtocolId.P0_PD,
            ProtocolId.P1_DP,
            ProtocolId.P2_GATED_PD,
            ProtocolId.P3_GATED_DP,
        ):
            rows.append(
                row(
                    f"M6-TIER-{seed}-{protocol.value}",
                    simulate_lifecycle(
                        requests,
                        TieredSloOverlay(seed, tenant_count=16),
                        LifecycleConfig(protocol, 4, 4),
                    ),
                    seed=seed,
                    overlay="TIERED",
                    trace_sha256=loaded.sha256,
                    git_sha=provenance.sha,
                    git_dirty=provenance.dirty,
                )
            )
    for strict_weight in (1.0, 2.0, 4.0):
        weights = {
            Tier.STRICT: strict_weight,
            Tier.STANDARD: 1.0,
            Tier.RELAXED: 0.5,
        }
        report = simulate_lifecycle(
            requests,
            TieredSloOverlay(713, tenant_count=16),
            LifecycleConfig(ProtocolId.P2_GATED_PD, 4, 4, tier_weights=weights),
        )
        rows.append(
            row(
                f"M6-WEIGHT-{strict_weight}",
                report,
                seed=713,
                overlay="TIERED",
                trace_sha256=loaded.sha256,
                git_sha=provenance.sha,
                git_dirty=provenance.dirty,
            )
        )
    for name in ("lambda_gpu", "lambda_stall", "lambda_risk"):
        for value in (0.5, 1.0, 2.0):
            kwargs = {name: value}
            report = simulate_lifecycle(
                requests,
                TieredSloOverlay(713, tenant_count=16),
                LifecycleConfig(ProtocolId.P2_GATED_PD, 4, 4, **kwargs),
            )
            rows.append(
                row(
                    f"M6-OAT-{name}-{value}",
                    report,
                    seed=713,
                    overlay="TIERED",
                    trace_sha256=loaded.sha256,
                    git_sha=provenance.sha,
                    git_dirty=provenance.dirty,
                )
            )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")
    return 0


def row(
    case_id: str,
    report: LifecycleReport,
    *,
    seed: int | None,
    overlay: str,
    trace_sha256: str,
    git_sha: str | None,
    git_dirty: bool,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "protocol": report.protocol.value,
        "overlay": overlay,
        "seed": "" if seed is None else seed,
        "trace_sha256": trace_sha256,
        "git_sha": git_sha or "",
        "git_dirty": git_dirty,
        "prefill_nodes": 4,
        "decode_nodes": 4,
        "completed_requests": report.completed_requests,
        "rejected_requests": report.rejected_requests,
        "strict_completed_goodput": report.strict_completed_goodput,
        "lenient_completed_goodput": report.lenient_completed_goodput,
        "partial_output_tokens": report.partial_output_tokens,
        "prefill_waste_work": report.prefill_waste_work,
        "slo_missed_prefill_work": report.slo_missed_prefill_work,
        "decode_hold_work": report.decode_hold_work,
        "store_occupancy_work": report.store_occupancy_work,
        "hold_count": report.hold_count,
        "reserve_count": report.reserve_count,
        "commit_count": report.commit_count,
        "rollback_count": report.rollback_count,
        "reject_count": report.reject_count,
        "jain_fairness": report.jain_fairness
        if report.jain_fairness is not None
        else "",
        "strict_weight": report.scalar_weights["STRICT"],
        "standard_weight": report.scalar_weights["STANDARD"],
        "relaxed_weight": report.scalar_weights["RELAXED"],
        "lambda_gpu": report.scalar_weights["lambda_gpu"],
        "lambda_stall": report.scalar_weights["lambda_stall"],
        "lambda_risk": report.scalar_weights["lambda_risk"],
        "prefill_token_cost": report.scalar_weights["prefill_token_cost"],
        "decode_token_cost": report.scalar_weights["decode_token_cost"],
        "max_post_prefill_wait": report.scalar_weights["max_post_prefill_wait"],
        "probe_cost": report.scalar_weights["probe_cost"],
        "probe_staleness_work": report.scalar_weights["probe_staleness_work"],
        "store_put_work": report.scalar_weights["store_put_work"],
        "store_pull_work": report.scalar_weights["store_pull_work"],
        "metric_unit": "NORMALIZED_WORK",
    }


if __name__ == "__main__":
    raise SystemExit(main())
