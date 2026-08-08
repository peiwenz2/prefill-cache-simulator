from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from prefill_cache_sim.domain import (
    BlockRef,
    CacheVisibility,
    ReplayMode,
    Request,
    ViewMode,
)
from prefill_cache_sim.experiment import ExperimentCase, run_case
from prefill_cache_sim.trace import load_trace, to_simulation_requests


def request(request_id: str, arrival_ms: int, blocks: list[int]) -> Request:
    return Request(
        request_id,
        request_id,
        0,
        arrival_ms,
        512 * len(blocks),
        0,
        tuple(BlockRef.trace(block) for block in blocks),
        (512,) * len(blocks),
        None,
    )


def case(selector_id: str = "S1_ROUND_ROBIN") -> ExperimentCase:
    return ExperimentCase(
        case_id="test",
        selector_id=selector_id,
        eviction_id="E1_LRU",
        node_count=2,
        capacity_mode="FIXED_TOTAL",
        capacity_blocks=10,
        replay_mode=ReplayMode.QUEUED,
        visibility=CacheVisibility.INSERT_AT_COMPLETION,
        view_mode=ViewMode.DELAYED,
        view_delay_ms=100,
        view_loss_probability=0,
        replay_speed=1,
        seed=713,
        family_size_cap=2,
    )


def test_result_schema_is_flat_stable_and_conservative() -> None:
    result = run_case(
        case(),
        (request("a", 0, [1]), request("b", 1, [1])),
        trace_sha256="ab" * 32,
        config_sha256="cd" * 32,
        git_sha="ef" * 20,
        git_dirty=False,
    )
    row = result.to_dict()
    assert row["result_schema_version"] == "phase-a-result-v3"
    assert row["prefix_anchor_k"] == 2
    assert row["trace_sha256"] == "ab" * 32
    assert row["config_sha256"] == "cd" * 32
    assert row["completed_requests"] == 2
    assert row["request_load_max_mean"] == 1
    assert all(not isinstance(value, (list, dict, tuple)) for value in row.values())


def test_fixed_total_capacity_partition_preserves_exact_total() -> None:
    uneven = case()
    uneven = replace(uneven, node_count=3)
    assert uneven.capacities() == (4, 3, 3)


@pytest.mark.parametrize(
    "selector_id",
    [
        "S0_RANDOM",
        "S1_ROUND_ROBIN",
        "S2_LEAST_WORK",
        "S3_GB_PREFIX_BUCKET",
        "S4_SESSION_AFFINITY",
        "S5_CENTRALIZED_MASTER_TTFT",
        "S6_CALIBRATED_TTFT",
    ],
)
@pytest.mark.parametrize(
    "eviction_id",
    ["E0_FIFO", "E1_LRU", "E2_SLRU", "E3_SECOND_HIT_ADMISSION"],
)
def test_every_phase_a_policy_runs_end_to_end(
    selector_id: str, eviction_id: str
) -> None:
    path = Path(__file__).resolve().parents[1] / "mooncake_trace.jsonl"
    if not path.exists():
        pytest.skip("Mooncake trace fixture is not present")
    requests = to_simulation_requests(load_trace(path, block_size_tokens=512))[:100]
    experiment = replace(
        case(selector_id),
        eviction_id=eviction_id,
        capacity_blocks=100,
        family_size_cap=2,
    )
    result = run_case(
        experiment,
        requests,
        trace_sha256="ab" * 32,
        config_sha256="cd" * 32,
        git_sha=None,
        git_dirty=True,
    )
    assert result.completed_requests == 100
    assert 0 <= result.block_ref_hit_rate <= 1
    assert result.replica_factor >= 1
