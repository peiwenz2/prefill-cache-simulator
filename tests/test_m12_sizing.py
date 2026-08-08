from __future__ import annotations

from dataclasses import replace

import pytest

import prefill_cache_sim.m12_sizing as sizing
from prefill_cache_sim.m12_metrics import SERVICE_REGIMES
from prefill_cache_sim.m12_placement import TraceRequestInput, build_kernel_requests
from prefill_cache_sim.m12_sizing import (
    GateObservation,
    SizingGates,
    SizingTopology,
    evaluate_gates,
    run_sizing_cell,
    select_minimum,
)


def observation(**changes: float) -> GateObservation:
    base = GateObservation(
        completion_ratio=1.0,
        minimum_tier_slo_attainment=0.90,
        jain_fairness=0.95,
        p_queue_p95_work=10.0,
        kvs_bytes_per_work=0.25,
    )
    return replace(base, **changes)


def gates() -> SizingGates:
    return SizingGates(
        minimum_completion_ratio=1.0,
        minimum_tier_slo_attainment=0.80,
        minimum_jain_fairness=0.90,
        maximum_p_queue_p95_work=20.0,
        maximum_kvs_bytes_per_work=0.50,
    )


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"completion_ratio": 0.99}, ("COMPLETION_FLOOR",)),
        ({"minimum_tier_slo_attainment": 0.79}, ("TIER_SLO_FLOOR",)),
        ({"jain_fairness": 0.89}, ("FAIRNESS_FLOOR",)),
        ({"p_queue_p95_work": 20.01}, ("P_QUEUE_P95_CEILING",)),
        (
            {"kvs_bytes_per_work": 0.51},
            ("KVS_UTILIZATION_CEILING",),
        ),
    ],
)
def test_each_frozen_gate_can_independently_make_a_cell_infeasible(
    changes: dict[str, float], expected: tuple[str, ...]
) -> None:
    assert evaluate_gates(observation(**changes), gates()) == expected


def test_gate_boundaries_are_inclusive() -> None:
    at_boundary = GateObservation(1.0, 0.80, 0.90, 20.0, 0.50)
    assert evaluate_gates(at_boundary, gates()) == ()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum_completion_ratio": float("nan")},
        {"minimum_tier_slo_attainment": 1.1},
        {"minimum_jain_fairness": -0.1},
        {"maximum_p_queue_p95_work": float("inf")},
        {"maximum_kvs_bytes_per_work": -1.0},
    ],
)
def test_invalid_gate_contract_fails_closed(kwargs: dict[str, float]) -> None:
    values = {
        "minimum_completion_ratio": 1.0,
        "minimum_tier_slo_attainment": 0.8,
        "minimum_jain_fairness": 0.9,
        "maximum_p_queue_p95_work": 20.0,
        "maximum_kvs_bytes_per_work": 0.5,
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        SizingGates(**values)


def test_minimum_selection_has_exact_predecessor_certificate() -> None:
    cells = tuple(
        evaluate_gates(
            observation(completion_ratio=completion), gates(),
            p_count=p_count,
            topology=SizingTopology.LOCAL_ONLY,
        )
        for p_count, completion in ((1, 0.5), (2, 0.9), (3, 1.0), (4, 1.0))
    )
    result = select_minimum(
        cells,
        gates(),
        required_topologies=(SizingTopology.LOCAL_ONLY,),
    )
    assert result.minimum_feasible_p == 3
    assert result.predecessor_certificate is not None
    assert result.predecessor_certificate.p_count == 2
    assert result.predecessor_certificate.failed_gates == ("COMPLETION_FLOOR",)


def test_non_monotonic_feasibility_is_reported_without_early_stop() -> None:
    cells = tuple(
        evaluate_gates(
            observation(completion_ratio=completion), gates(),
            p_count=p_count,
            topology=SizingTopology.SHARED_KVS,
        )
        for p_count, completion in ((1, 0.5), (2, 1.0), (3, 0.9), (4, 1.0))
    )
    result = select_minimum(
        cells,
        gates(),
        required_topologies=(SizingTopology.SHARED_KVS,),
    )
    assert result.minimum_feasible_p == 2
    assert result.non_monotonic_p_counts == (3,)


def test_missing_literal_predecessor_fails_closed() -> None:
    only_p2 = (
        evaluate_gates(
            observation(),
            gates(),
            p_count=2,
            topology=SizingTopology.LOCAL_ONLY,
        ),
    )
    result = select_minimum(
        only_p2,
        gates(),
        required_topologies=(SizingTopology.LOCAL_ONLY,),
    )
    assert result.minimum_feasible_p is None
    assert result.grid_exhausted is True


def test_global_predecessor_certificate_covers_every_deployable_topology() -> None:
    cells = tuple(
        evaluate_gates(
            observation(completion_ratio=completion),
            gates(),
            p_count=p_count,
            topology=topology,
        )
        for topology in (SizingTopology.LOCAL_ONLY, SizingTopology.SHARED_KVS)
        for p_count, completion in ((1, 0.5), (2, 1.0))
    )
    result = select_minimum(cells, gates())
    assert result.minimum_feasible_p == 2
    assert {value.topology for value in result.predecessor_certificates} == {
        SizingTopology.LOCAL_ONLY,
        SizingTopology.SHARED_KVS,
    }


def test_global_verdict_fails_closed_when_a_deployable_topology_is_missing() -> None:
    local_only = tuple(
        evaluate_gates(
            observation(completion_ratio=completion),
            gates(),
            p_count=p_count,
            topology=SizingTopology.LOCAL_ONLY,
        )
        for p_count, completion in ((1, 0.5), (2, 1.0))
    )
    result = select_minimum(local_only, gates())
    assert result.minimum_feasible_p is None
    assert result.grid_exhausted is True


def test_zero_transfer_control_is_never_selected_as_deployable_minimum() -> None:
    cells = (
        evaluate_gates(
            observation(), gates(),
            p_count=1,
            topology=SizingTopology.ZERO_TRANSFER_PRICE_CONTROL,
        ),
        evaluate_gates(
            observation(completion_ratio=0.9), gates(),
            p_count=1,
            topology=SizingTopology.LOCAL_ONLY,
        ),
        evaluate_gates(
            observation(), gates(),
            p_count=2,
            topology=SizingTopology.LOCAL_ONLY,
        ),
    )
    result = select_minimum(
        cells,
        gates(),
        required_topologies=(SizingTopology.LOCAL_ONLY,),
    )
    assert result.minimum_feasible_p == 2
    assert result.selected_topology is SizingTopology.LOCAL_ONLY


def _trace_row(
    identity: str,
    arrival: float,
    keys: tuple[str, ...],
    sizes: tuple[int, ...],
) -> TraceRequestInput:
    return TraceRequestInput(
        identity,
        "tenant-a",
        "STANDARD",
        arrival,
        keys,
        sizes,
        0,
        "model",
        "adapter",
        "shape",
        100,
        ("p0", "p1"),
    )


def test_zero_transfer_control_allows_remote_hit_without_transfer_work() -> None:
    workload = build_kernel_requests(
        [
            _trace_row("seed", 0, ("A",), (10,)),
            _trace_row("blocker", 1, ("B",), (100,)),
            _trace_row("reuse", 1, ("A",), (10,)),
        ]
    )
    common = {
        "workload": workload,
        "p_count": 2,
        "gates": gates(),
        "regime": SERVICE_REGIMES[2],
        "observation_end_work": 100,
        "tier_slo_work": {"STANDARD": 100},
        "cache_capacity_entries_per_p": 8,
        "decode_node_count": 2,
        "kvs_bytes_per_token": 16,
    }
    ideal = run_sizing_cell(
        topology=SizingTopology.ZERO_TRANSFER_PRICE_CONTROL,
        **common,
    )
    local = run_sizing_cell(topology=SizingTopology.LOCAL_ONLY, **common)
    assert ideal.remote_hit_tokens > 0
    assert ideal.normalized_kvs_work == 0
    assert local.remote_hit_tokens == 0


def test_zero_transfer_control_changes_only_price_not_routing_threshold() -> None:
    workload = build_kernel_requests([_trace_row("one", 0, ("A",), (10,))])
    cost = sizing._sizing_cost(
        SERVICE_REGIMES[2], SizingTopology.ZERO_TRANSFER_PRICE_CONTROL, 1.0, 16
    )
    policy = sizing._sizing_policy(
        SizingTopology.ZERO_TRANSFER_PRICE_CONTROL, cost, workload
    )
    assert cost.kvs_work_per_token == 0
    assert policy.mooncake_balancing_threshold == 2.0


def test_sizing_cell_resizes_cohort_eligibility_to_requested_p_count() -> None:
    workload = build_kernel_requests(
        [_trace_row("one", 0, ("A",), (10,))]
    )
    record = run_sizing_cell(
        workload,
        p_count=3,
        topology=SizingTopology.LOCAL_ONLY,
        gates=gates(),
        regime=SERVICE_REGIMES[2],
        observation_end_work=100,
        tier_slo_work={"STANDARD": 100},
        cache_capacity_entries_per_p=8,
        decode_node_count=2,
    )
    assert len(record.decision_fingerprint) == 64
    assert record.recompute_tokens == 10
