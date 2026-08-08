#!/usr/bin/env python3
"""Run M8 cooperative-preemption decision sensitivity on Mooncake."""

from __future__ import annotations

import csv
from bisect import bisect_right
from pathlib import Path

from prefill_cache_sim.config import git_provenance
from prefill_cache_sim.preemption import (
    CooperativePreemptionController,
    DecodeAction,
    DecodeCheckpoint,
    KeepMoveInput,
    PreemptionConfig,
    R2CheckpointStore,
    preemption_gate,
)
from prefill_cache_sim.trace import load_trace, to_simulation_requests


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    loaded = load_trace(root / "mooncake_trace.jsonl", block_size_tokens=512)
    requests = to_simulation_requests(loaded)
    arrivals = [request.arrival_ms for request in requests]
    provenance = git_provenance(root)
    rows: list[dict[str, object]] = []
    for recovery_mode in ("R1_PREFIX_KVS", "R2_CHECKPOINT"):
        for margin in (0.1, 50.0, 100.0, 200.0, 500.0, 1000.0):
            config = PreemptionConfig(safety_margin=margin)
            controller = CooperativePreemptionController(config, R2CheckpointStore())
            aborts = 0
            twice = 0
            move_deltas: list[float] = []
            for request_index, request in enumerate(requests):
                request_aborts = 0
                for checkpoint_index in range(config.max_preemptions + 2):
                    epoch = request_aborts
                    emitted = min(
                        request.output_tokens,
                        (checkpoint_index + 1) * config.min_quantum,
                    )
                    if emitted >= request.output_tokens:
                        break
                    remaining = request.output_tokens - emitted
                    recovery = (
                        request.input_tokens * 0.06
                        if recovery_mode == "R1_PREFIX_KVS"
                        else request.input_tokens * 0.002
                    )
                    checkpoint = DecodeCheckpoint(
                        request.logical_request_id,
                        epoch,
                        emitted,
                        emitted,
                        f"r2:{request.logical_request_id}:{epoch}"
                        if recovery_mode == "R2_CHECKPOINT"
                        else None,
                        1.0,
                        config.min_quantum,
                    )
                    pressure = max(
                        0,
                        bisect_right(arrivals, request.arrival_ms + 100)
                        - request_index
                        - 1,
                    )
                    keep_probability = max(0.05, 1 - remaining / 1000)
                    inputs = KeepMoveInput(
                        completion_value=100,
                        keep_probability=keep_probability,
                        move_probability=min(
                            1.0, keep_probability + min(0.3, pressure / 20)
                        ),
                        remaining_gpu_work=remaining,
                        interference_gpu_work=min(100, pressure * 5),
                        checkpoint_gpu_work=0.5,
                        recovery_gpu_work=recovery,
                        migration_stall_work=recovery * 0.25,
                        duplicate_risk=0.001,
                        wait_work=max(0, remaining - 64),
                        tenant_weight=1,
                    )
                    decision, _ = controller.report_checkpoint(checkpoint, inputs)
                    if decision.reason in ("move_exceeds_margin", "keep_with_margin"):
                        move_deltas.append(decision.move_score - decision.keep_score)
                    if decision.action is not DecodeAction.ABORT_SELF:
                        break
                    aborts += 1
                    request_aborts += 1
                if request_aborts >= 2:
                    twice += 1
            completed = len(requests)
            rows.append(
                {
                    "case_id": f"M8-{recovery_mode}-M{margin}",
                    "recovery_mode": recovery_mode,
                    "safety_margin": margin,
                    "assumed_completed_requests": completed,
                    "preemptions": aborts,
                    "preemptions_per_completed_request": aborts / completed,
                    "preempted_twice_ratio": twice / completed,
                    "move_delta_mean": sum(move_deltas) / len(move_deltas),
                    "scored_decisions": len(move_deltas),
                    "hard_gate_pass_upper_bound": preemption_gate(
                        completed_requests=completed,
                        preemptions=aborts,
                        config=config,
                    ),
                    "min_quantum": config.min_quantum,
                    "max_preemptions": config.max_preemptions,
                    "hard_gate_limit": config.max_preemptions_per_completed,
                    "completion_model": "ASSUME_RESUME_SUCCESS_UPPER_BOUND",
                    "r1_recovery_per_input_token": 0.06,
                    "r2_recovery_per_input_token": 0.002,
                    "calibration_status": "SYNTHETIC_UNCALIBRATED",
                    "trace_sha256": loaded.sha256,
                    "git_sha": provenance.sha or "",
                    "git_dirty": provenance.dirty,
                    "metric_unit": "SYNTHETIC_SCORE",
                }
            )
    output = root / "results/m8/results.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
