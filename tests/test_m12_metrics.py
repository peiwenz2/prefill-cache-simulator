from __future__ import annotations

import math

import pytest

from prefill_cache_sim.m12_metrics import (
    SERVICE_REGIMES,
    AttemptOutcome,
    LogicalRequestSpec,
    ServiceRegimeId,
    _nonnegative_taxonomy_remainder,
    aggregate_m12_metrics,
)


def outcome(
    logical: str,
    attempt: int,
    *,
    completed: bool = True,
    strict: bool = True,
    emitted: int = 20,
    p_work: float = 10,
    d_work: float = 20,
    wasted_p: float = 0,
    wasted_d: float = 0,
    tenant: str = "tenant-a",
    tier: str = "STANDARD",
) -> AttemptOutcome:
    return AttemptOutcome(
        logical,
        attempt,
        tenant,
        tier,
        0,
        100,
        20,
        emitted,
        completed,
        strict,
        50 if completed else None,
        p_work,
        d_work,
        wasted_p,
        wasted_d,
    )


def workload_for(values: list[AttemptOutcome]) -> list[LogicalRequestSpec]:
    requests: dict[str, LogicalRequestSpec] = {}
    for value in values:
        requests.setdefault(
            value.logical_request_id,
            LogicalRequestSpec(
                value.logical_request_id,
                value.tenant_id,
                value.tier,
                value.arrival_work,
                value.input_tokens,
                value.requested_output_tokens,
            ),
        )
    return list(requests.values())


def aggregate(
    values: list[AttemptOutcome],
    *,
    workload: list[LogicalRequestSpec] | None = None,
):
    return aggregate_m12_metrics(
        workload if workload is not None else workload_for(values),
        values,
        observation_start_work=0,
        observation_end_work=100,
        prefill_nodes=2,
        decode_nodes=2,
    )


def test_retry_work_is_charged_but_logical_tokens_are_credited_once() -> None:
    failed = outcome(
        "r1",
        0,
        completed=False,
        strict=False,
        emitted=5,
        wasted_p=10,
        wasted_d=5,
    )
    winner = outcome("r1", 1)
    report = aggregate([failed, winner])
    assert report.offered_logical_requests == 1
    assert report.attempt_count == 2
    assert report.retry_amplification == 2
    assert report.strict_completed_requests == 1
    assert report.strict_useful_tokens == 120
    assert report.strict_useful_input_tokens == 100
    assert report.strict_useful_output_tokens == 20
    assert report.total_gpu_work == 60
    assert report.wasted_gpu_work == 15
    assert report.raw_output_token_throughput == pytest.approx(0.25)


def test_taxonomy_tolerates_only_roundoff_at_large_work_scale() -> None:
    values = [
        outcome(
            f"r{index}",
            0,
            strict=index % 2 == 0,
            p_work=1_000_000 + (index + 1) * 0.08,
            d_work=0,
        )
        for index in range(17)
    ]
    report = aggregate_m12_metrics(
        workload_for(values),
        values,
        observation_start_work=0,
        observation_end_work=20_000_000,
        prefill_nodes=2,
        decode_nodes=2,
    )
    assert report.unclassified_attempt_gpu_work == 0
    assert report.total_gpu_work == pytest.approx(
        report.winner_attributable_gpu_work + report.slo_missed_gpu_work
    )
    total = 17_000_012.24
    one_ulp_over = total + math.ulp(total)
    assert _nonnegative_taxonomy_remainder(total, one_ulp_over, summand_count=17) == 0
    with pytest.raises(ValueError, match="over-counted"):
        _nonnegative_taxonomy_remainder(
            total,
            total + 1e-5,
            summand_count=17,
        )


def test_duplicate_attempt_identity_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate attempt identity"):
        aggregate([outcome("r1", 0), outcome("r1", 0)])


def test_logical_shape_cannot_change_across_attempts() -> None:
    first = outcome("r1", 0)
    changed = AttemptOutcome(
        "r1",
        1,
        "tenant-a",
        "STANDARD",
        0,
        101,
        20,
        20,
        True,
        True,
        50,
        10,
        20,
        0,
        0,
    )
    with pytest.raises(ValueError, match="differs from workload"):
        aggregate([first, changed])


def test_offered_load_and_horizon_do_not_depend_on_attempt_count() -> None:
    baseline = aggregate([outcome("r1", 0)])
    retried = aggregate(
        [
            outcome("r1", 0, completed=False, strict=False, wasted_p=10),
            outcome("r1", 1),
        ]
    )
    assert baseline.offered_logical_requests == retried.offered_logical_requests
    assert baseline.offered_input_tokens == retried.offered_input_tokens
    assert baseline.offered_output_tokens == retried.offered_output_tokens
    assert baseline.observation_end_work == retried.observation_end_work


def test_work_and_utilization_conserve() -> None:
    report = aggregate([outcome("r1", 0), outcome("r2", 0)])
    assert report.total_gpu_work == report.prefill_gpu_work + report.decode_gpu_work
    assert report.waste_fraction == 0
    assert report.prefill_normalized_utilization == pytest.approx(0.1)
    assert report.decode_normalized_utilization == pytest.approx(0.2)
    assert report.cluster_normalized_utilization == pytest.approx(0.15)


def test_slo_missed_work_is_not_mixed_with_true_waste() -> None:
    report = aggregate([outcome("r1", 0, strict=False), outcome("r2", 0, wasted_p=4)])
    assert report.slo_missed_gpu_work == 30
    assert report.wasted_gpu_work == 4
    assert report.total_gpu_work == pytest.approx(
        report.wasted_gpu_work
        + report.slo_missed_gpu_work
        + report.winner_attributable_gpu_work
        + report.unclassified_attempt_gpu_work
    )


def test_zero_attempt_drop_stays_in_offered_and_fairness_denominators() -> None:
    frozen = [
        LogicalRequestSpec("served", "tenant-a", "STRICT", 0, 100, 20),
        LogicalRequestSpec("dropped", "tenant-b", "RELAXED", 0, 100, 20),
    ]
    report = aggregate([outcome("served", 0, tier="STRICT")], workload=frozen)
    assert report.offered_logical_requests == 2
    assert report.retry_amplification == 0.5
    assert report.per_tier_slo_attainment == {"RELAXED": 0.0, "STRICT": 1.0}
    assert report.jain_fairness == pytest.approx(0.5)
    assert not report.fairness_floor_pass


def test_completed_attempt_must_match_frozen_true_output() -> None:
    value = outcome("r1", 0, emitted=19)
    frozen = [LogicalRequestSpec("r1", "tenant-a", "STANDARD", 0, 100, 20)]
    with pytest.raises(ValueError, match="frozen true output"):
        aggregate([value], workload=frozen)


def test_kvs_work_is_reported_outside_gpu_denominator() -> None:
    value = outcome("r1", 0)
    value = AttemptOutcome(
        value.logical_request_id,
        value.attempt_index,
        value.tenant_id,
        value.tier,
        value.arrival_work,
        value.input_tokens,
        value.requested_output_tokens,
        value.emitted_output_tokens,
        value.completed,
        value.strict_slo_met,
        value.finish_work,
        value.prefill_gpu_work,
        value.decode_gpu_work,
        value.wasted_prefill_work,
        value.wasted_decode_work,
        1024,
        2.5,
    )
    report = aggregate([value])
    assert report.kvs_bytes_per_horizon == pytest.approx(10.24)
    assert report.kvs_normalized_work == 2.5
    assert report.total_gpu_work == 30


def test_fairness_uses_per_tenant_served_over_demand_ratios() -> None:
    values = [
        outcome("a", 0, tenant="tenant-a", tier="STRICT"),
        outcome(
            "b",
            0,
            tenant="tenant-b",
            tier="RELAXED",
            completed=False,
            strict=False,
        ),
    ]
    report = aggregate(values)
    assert report.per_tier_slo_attainment == {"RELAXED": 0.0, "STRICT": 1.0}
    assert report.minimum_tier_slo_attainment == 0
    assert report.jain_fairness == pytest.approx(0.5)
    assert not report.fairness_floor_pass


def test_service_regimes_are_positive_monotone_and_distinct() -> None:
    assert {regime.regime_id for regime in SERVICE_REGIMES} == set(ServiceRegimeId)
    curves = set()
    for regime in SERVICE_REGIMES:
        single = regime.prefill_work(4096, batch_tokens=4096)
        batched = regime.prefill_work(4096, batch_tokens=65_536)
        assert 0 < batched <= single
        decode_single = regime.decode_work(64, concurrent_sequences=1)
        decode_batched = regime.decode_work(64, concurrent_sequences=64)
        assert 0 < decode_batched <= decode_single
        assert regime.kvs_work(1024) > 0
        curves.add((single, batched, decode_single, decode_batched))
    assert len(curves) == len(SERVICE_REGIMES)


def test_v1_2_regime_costs_are_round_and_kvs_bytes_are_accounting_only() -> None:
    by_id = {regime.regime_id: regime for regime in SERVICE_REGIMES}
    expected = {
        ServiceRegimeId.COMPUTE_BOUND: (0.08, 1.0, 0.01),
        ServiceRegimeId.MEMORY_BOUND: (0.04, 1.4, 0.02),
        ServiceRegimeId.MIXED: (0.06, 1.0, 0.01),
    }
    for regime_id, (prefill, decode, remote) in expected.items():
        regime = by_id[regime_id]
        assert regime.prefill_token_work == prefill
        assert regime.decode_token_work == decode
        assert regime.kvs_token_work == remote
        assert regime.kvs_bytes_per_token == 65_536
        assert regime.kvs_work(65_536) == remote


def test_invalid_waste_and_completion_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="wasted prefill"):
        outcome("r", 0, p_work=1, wasted_p=2)
    with pytest.raises(ValueError, match="completed and finish_work"):
        AttemptOutcome(
            "r",
            0,
            "t",
            "STANDARD",
            0,
            1,
            1,
            0,
            False,
            False,
            1,
            1,
            1,
            0,
            0,
        )
    assert math.isfinite(aggregate([outcome("r", 0)]).strict_useful_token_goodput)


def test_observation_window_and_pool_capacity_are_enforced() -> None:
    late = outcome("late", 0)
    late = AttemptOutcome(
        late.logical_request_id,
        late.attempt_index,
        late.tenant_id,
        late.tier,
        100,
        late.input_tokens,
        late.requested_output_tokens,
        late.emitted_output_tokens,
        late.completed,
        late.strict_slo_met,
        100,
        late.prefill_gpu_work,
        late.decode_gpu_work,
        late.wasted_prefill_work,
        late.wasted_decode_work,
    )
    with pytest.raises(ValueError, match="arrival is outside"):
        aggregate([late])
    overloaded = outcome("overloaded", 0, p_work=201, d_work=20)
    with pytest.raises(ValueError, match="exceeds normalized pool capacity"):
        aggregate([overloaded])
