from __future__ import annotations

import math
from dataclasses import replace

import pytest

from prefill_cache_sim.m12_decode import (
    AbortFence,
    AdmissionAction,
    DecodeAdmissionConfig,
    DecodeAdmissionMode,
    DecodeCapacityPolicy,
    DecodeCreditLedger,
    DecodeRunReport,
    PrefixFamilyPredictor,
    evaluate_g12_3,
)
from prefill_cache_sim.m12_kernel import (
    CausalKernel,
    CausalView,
    FrozenKernelCostModel,
    KernelConfig,
)
from prefill_cache_sim.m12_metrics import LogicalRequestSpec
from prefill_cache_sim.m12_placement import M12PlacementPolicy, PlacementMode


def request(identity: str, arrival: float, output: int, family: str = "family"):
    from prefill_cache_sim.m12_kernel import KernelRequestSpec

    return KernelRequestSpec(
        LogicalRequestSpec(identity, "tenant", "STANDARD", arrival, 10, output),
        (family,),
        (10,),
    )


def view(now: float = 0, d_ready: float = 0) -> CausalView:
    return CausalView(
        now,
        frozenset(),
        {"p0": now},
        {"d0": d_ready},
        {"p0": frozenset()},
        {"p0": 0},
    )


def base() -> M12PlacementPolicy:
    return M12PlacementPolicy(
        PlacementMode.PRICED_SPILL,
        FrozenKernelCostModel(0.1, 0, 0, 0.5),
        kvs_enabled=False,
    )


def test_modes_mark_label_access_and_deployability_explicitly() -> None:
    assert DecodeAdmissionMode.NO_GATE.deployable
    assert DecodeAdmissionMode.CAUSAL.deployable
    assert not DecodeAdmissionMode.ORACLE.deployable
    assert not DecodeAdmissionMode.ORACLE_NOISED.deployable
    assert DecodeAdmissionMode.ORACLE_NOISED.uses_output_label


def test_causal_predictor_is_past_only_and_family_scoped() -> None:
    predictor = PrefixFamilyPredictor(default_output_tokens=8, history_limit=3)
    future = request("future", 10, 10_000, "A")
    assert predictor.predict(future, at_work=0) == 8
    predictor.observe(request("a1", 0, 4, "A"), completed_at_work=2)
    predictor.observe(request("b1", 0, 100, "B"), completed_at_work=2)
    assert predictor.predict(future, at_work=2) == 4
    with pytest.raises(ValueError, match="future"):
        predictor.predict(future, at_work=1)


def test_reservation_is_predicted_tokens_times_kernel_decode_cost() -> None:
    policy = DecodeCapacityPolicy(
        base(),
        DecodeAdmissionConfig(DecodeAdmissionMode.CAUSAL, capacity_credits=100),
        predictor=PrefixFamilyPredictor(default_output_tokens=7),
    )
    plan = policy.plan_attempts(request("r", 0, 999), view())[0]
    assert plan.emitted_output_tokens == 999  # kernel truth is not replaced
    assert policy.decisions[0].reserved_decode_credits == 3.5


@pytest.mark.parametrize(
    ("action", "expected_delay"),
    [
        (AdmissionAction.DEFER, 4.0),
        (AdmissionAction.GATED_PD, 0.0),
        (AdmissionAction.GATED_DP, 4.0),
    ],
)
def test_all_congestion_actions_gate_instead_of_dropping_offered_load(
    action: AdmissionAction, expected_delay: float
) -> None:
    policy = DecodeCapacityPolicy(
        base(),
        DecodeAdmissionConfig(
            DecodeAdmissionMode.CAUSAL,
            capacity_credits=4,
            congestion_action=action,
        ),
        predictor=PrefixFamilyPredictor(default_output_tokens=8),
    )
    plan = policy.plan_attempts(request("r", 0, 3), view(d_ready=4))[0]
    assert plan.arrival_work == expected_delay
    assert policy.decisions[0].action is action


def test_credit_ledger_accumulates_conserves_and_reconciles_actual() -> None:
    policy = DecodeCapacityPolicy(
        base(),
        DecodeAdmissionConfig(
            DecodeAdmissionMode.CAUSAL,
            capacity_credits=2,
            congestion_action=AdmissionAction.GATED_DP,
        ),
        predictor=PrefixFamilyPredictor(default_output_tokens=4),
    )
    first = policy.plan_attempts(request("a", 0, 6), view())[0]
    second = policy.plan_attempts(request("b", 0, 2), view())[0]
    assert first.arrival_work == 0
    assert second.arrival_work > 0
    assert policy.ledger.reserved_decode_credits == 4
    assert policy.ledger.peak_reserved_decode_credits <= 2


def test_oversized_reservation_is_serialized_without_hiding_predicted_work() -> None:
    ledger = DecodeCreditLedger(capacity_credits=4)

    first = ledger.reserve("large", 0, 10, now_work=0)
    chunks = ledger._active[("large", 0)]
    second = ledger.reserve("small", 0, 1, now_work=0)

    assert first.starts_at_work == 0
    assert [item.credits for item in chunks] == [4, 4, 2]
    assert sum(item.credits for item in chunks) == 10
    assert all(item.credits <= ledger.capacity_credits for item in chunks)
    assert second.starts_at_work >= chunks[0].releases_at_work
    assert ledger.peak_reserved_decode_credits <= ledger.capacity_credits

    ledger.activate("large", 0, 10, starts_at_work=0, finishes_at_work=9)
    assert ledger.reserved_decode_credits == 11
    assert ledger.decode_residency["large"] == 9
    assert ("small", 0) not in ledger._active
    assert ledger.available_at(1, now_work=8, exclude=("small", 0)) >= 9
    endpoints = sorted(
        {
            point
            for item in ledger._temporal_reservations()
            for point in (item.starts_at_work, item.releases_at_work)
        }
    )
    assert all(
        sum(
            item.credits
            for item in ledger._temporal_reservations()
            if item.starts_at_work <= point < item.releases_at_work
        )
        <= ledger.capacity_credits
        for point in endpoints
    )
    ledger.settle("large", 0, actual_decode_work=12)
    assert ledger.reconciled_credit_error == 2


def test_positive_reservation_rejects_zero_capacity_consistently() -> None:
    ledger = DecodeCreditLedger(capacity_credits=0)

    with pytest.raises(ValueError, match="positive capacity"):
        ledger.available_at(1, now_work=0)
    with pytest.raises(ValueError, match="positive capacity"):
        ledger.reserve("request", 0, 1, now_work=0)


@pytest.mark.parametrize("capacity", [-1, math.inf, math.nan])
def test_ledger_rejects_invalid_capacity(capacity: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        DecodeCreditLedger(capacity)


@pytest.mark.parametrize("credits", [-1, math.inf, math.nan])
def test_ledger_rejects_invalid_reservation_credits(credits: float) -> None:
    ledger = DecodeCreditLedger(4)
    with pytest.raises(ValueError, match="finite and non-negative"):
        ledger.available_at(credits, now_work=0)


def test_reservation_checks_future_starts_across_its_entire_interval() -> None:
    ledger = DecodeCreditLedger(capacity_credits=4)
    ledger.reserve("large", 0, 10, now_work=0)
    ledger.reserve("small", 0, 1, now_work=0)
    ledger.activate("large", 0, 10, starts_at_work=0, finishes_at_work=3)

    ledger.reserve("new", 0, 8, now_work=3)

    endpoints = sorted(
        {
            point
            for item in ledger._temporal_reservations()
            for point in (item.starts_at_work, item.releases_at_work)
        }
    )
    assert all(
        sum(
            item.credits
            for item in ledger._temporal_reservations()
            if item.starts_at_work <= point < item.releases_at_work
        )
        <= ledger.capacity_credits
        for point in endpoints
    )
    assert ledger._active[("new", 0)][1].starts_at_work >= 9


def test_rejected_activation_leaves_plans_and_debt_unchanged() -> None:
    ledger = DecodeCreditLedger(capacity_credits=4)
    ledger.reserve("running", 0, 4, now_work=0)
    ledger.activate("running", 0, 4, starts_at_work=0, finishes_at_work=10)
    ledger.reserve("next", 0, 4, now_work=0)
    before_active = dict(ledger._active)
    before_executing = dict(ledger._executing)
    before_debt = ledger.reserved_decode_credits

    with pytest.raises(ValueError, match="activation exceeds"):
        ledger.activate("next", 0, 4, starts_at_work=5, finishes_at_work=9)

    assert ledger._active == before_active
    assert ledger._executing == before_executing
    assert ledger.reserved_decode_credits == before_debt


def test_gated_dp_accepts_prediction_larger_than_shared_capacity() -> None:
    policy = DecodeCapacityPolicy(
        base(),
        DecodeAdmissionConfig(
            DecodeAdmissionMode.CAUSAL,
            capacity_credits=4,
            congestion_action=AdmissionAction.GATED_DP,
        ),
        predictor=PrefixFamilyPredictor(default_output_tokens=20),
    )

    first = policy.plan_attempts(request("large", 0, 20), view())[0]
    second = policy.plan_attempts(request("small", 0, 1), view())[0]

    assert first.arrival_work == 0
    assert policy.decisions[0].reserved_decode_credits == 10
    assert second.arrival_work > first.arrival_work
    assert policy.ledger.peak_reserved_decode_credits <= 4


def test_defer_moves_whole_attempt_when_d_is_idle_but_credits_are_full() -> None:
    policy = DecodeCapacityPolicy(
        base(),
        DecodeAdmissionConfig(
            DecodeAdmissionMode.CAUSAL,
            capacity_credits=3,
            congestion_action=AdmissionAction.DEFER,
        ),
        predictor=PrefixFamilyPredictor(default_output_tokens=6),
    )
    first = request("a", 0, 4)
    first_attempt = policy.plan_attempts(first, view())[0]
    assert first_attempt.arrival_work == 0
    policy.decode_started(
        first,
        first_attempt,
        view(now=1),
        finish_work=4,
    )
    delayed = policy.plan_attempts(request("b", 1, 4), view(now=1))[0]
    assert delayed.arrival_work > 0


def test_gated_pd_allows_prefill_then_kernel_gates_decode() -> None:
    config = KernelConfig(
        0,
        100,
        ("p0",),
        ("d0",),
        {"STANDARD": 100},
        8,
        FrozenKernelCostModel(0.1, 0, 0, 0.5),
    )
    policy = DecodeCapacityPolicy(
        base(),
        DecodeAdmissionConfig(
            DecodeAdmissionMode.CAUSAL,
            capacity_credits=2,
            congestion_action=AdmissionAction.GATED_PD,
        ),
        predictor=PrefixFamilyPredictor(default_output_tokens=4),
    )
    result = CausalKernel(config).run(
        (request("a", 0, 4), request("b", 0, 4)), policy
    )
    assert result.attempts[1].prefill_finish_work == 2
    assert any(event.kind == "DECODE_GATED" for event in result.events)


def test_gated_dp_reserves_decode_before_and_delays_prefill() -> None:
    policy = DecodeCapacityPolicy(
        base(),
        DecodeAdmissionConfig(
            DecodeAdmissionMode.CAUSAL,
            capacity_credits=2,
            congestion_action=AdmissionAction.GATED_DP,
        ),
        predictor=PrefixFamilyPredictor(default_output_tokens=4),
    )
    policy.plan_attempts(request("a", 0, 4), view())
    second = policy.plan_attempts(request("b", 0, 4), view())[0]
    assert second.arrival_work > 0


def test_action_lifecycles_have_distinct_kernel_event_traces() -> None:
    config = KernelConfig(
        0,
        100,
        ("p0",),
        ("d0",),
        {"STANDARD": 100},
        8,
        FrozenKernelCostModel(0.1, 0, 0, 0.5),
    )
    traces = {}
    for action in (
        AdmissionAction.DEFER,
        AdmissionAction.GATED_PD,
        AdmissionAction.GATED_DP,
    ):
        policy = DecodeCapacityPolicy(
            base(),
            DecodeAdmissionConfig(
                DecodeAdmissionMode.CAUSAL,
                capacity_credits=3,
                congestion_action=action,
            ),
            predictor=PrefixFamilyPredictor(default_output_tokens=6),
        )
        result = CausalKernel(config).run(
            (request("a", 0, 6), request("b", 1.5, 6)), policy
        )
        traces[action] = tuple(event.kind for event in result.events)
        assert policy.ledger.p_to_d_debt_credits == 0
        assert policy.ledger.actual_decode_work == result.metrics.decode_gpu_work
    assert "ADMISSION_DEFER" in traces[AdmissionAction.DEFER]
    assert "ADMISSION_GATED_PD" in traces[AdmissionAction.GATED_PD]
    assert "ADMISSION_GATED_DP" in traces[AdmissionAction.GATED_DP]
    assert len(set(traces.values())) == 3


def test_abort_requires_boundary_negative_value_authority_and_retry_budget() -> None:
    assert AbortFence(True, -0.1, True, 1).allows_abort
    assert not AbortFence(False, -0.1, True, 1).allows_abort
    assert not AbortFence(True, 0, True, 1).allows_abort
    assert not AbortFence(True, -0.1, False, 1).allows_abort
    assert not AbortFence(True, -0.1, True, 0).allows_abort


def test_lease_abort_is_real_waste_and_retry_without_duplicate_useful_credit() -> None:
    config = KernelConfig(
        0,
        100,
        ("p0",),
        ("d0",),
        {"STANDARD": 100},
        8,
        FrozenKernelCostModel(0.1, 0, 0, 0.5),
        retry_budget=1,
    )
    policy = DecodeCapacityPolicy(
        base(),
        DecodeAdmissionConfig(
            DecodeAdmissionMode.CAUSAL,
            capacity_credits=10,
            lease_boundary_tokens=2,
            abort_fences={"r": AbortFence(True, -1, True, 1)},
        ),
        predictor=PrefixFamilyPredictor(default_output_tokens=6),
    )
    result = CausalKernel(config).run((request("r", 0, 6),), policy)
    report = policy.summarize(result)
    assert [a.outcome.completed for a in result.attempts] == [False, True]
    assert report.useful_output_tokens == 6
    assert report.wasted_decode_work == 1
    assert report.preemptions == 1


def test_kernel_retry_budget_is_an_authoritative_abort_fence() -> None:
    config = KernelConfig(
        0,
        100,
        ("p0",),
        ("d0",),
        {"STANDARD": 100},
        8,
        FrozenKernelCostModel(0.1, 0, 0, 0.5),
        retry_budget=0,
    )
    policy = DecodeCapacityPolicy(
        base(),
        DecodeAdmissionConfig(
            DecodeAdmissionMode.CAUSAL,
            capacity_credits=10,
            abort_fences={"r": AbortFence(True, -1, True, 1)},
        ),
        predictor=PrefixFamilyPredictor(default_output_tokens=6),
    )
    result = CausalKernel(config).run((request("r", 0, 6),), policy)
    assert result.attempts[0].outcome.completed
    assert policy.summarize(result).preemptions == 0


def test_fully_emitted_output_is_never_aborted_at_lease_boundary() -> None:
    config = KernelConfig(
        0,
        100,
        ("p0",),
        ("d0",),
        {"STANDARD": 100},
        8,
        FrozenKernelCostModel(0.1, 0, 0, 0.5),
        retry_budget=1,
    )
    policy = DecodeCapacityPolicy(
        base(),
        DecodeAdmissionConfig(
            DecodeAdmissionMode.CAUSAL,
            capacity_credits=10,
            lease_boundary_tokens=6,
            abort_fences={"r": AbortFence(True, -1, True, 1)},
        ),
        predictor=PrefixFamilyPredictor(default_output_tokens=6),
    )
    result = CausalKernel(config).run((request("r", 0, 6),), policy)
    assert result.attempts[0].outcome.completed
    assert not any(event.kind == "LEASE_BOUNDARY" for event in result.events)


def test_zero_work_decode_start_still_counts_exact_lease_progress() -> None:
    config = KernelConfig(
        0,
        100,
        ("p0",),
        ("d0",),
        {"STANDARD": 100},
        8,
        FrozenKernelCostModel(0, 0, 0, 0.5),
        retry_budget=1,
    )
    policy = DecodeCapacityPolicy(
        base(),
        DecodeAdmissionConfig(
            DecodeAdmissionMode.NO_GATE,
            capacity_credits=10,
            lease_boundary_tokens=4,
            abort_fences={"r": AbortFence(True, -1, True, 1)},
        ),
    )
    result = CausalKernel(config).run((request("r", 0, 8),), policy)
    assert result.attempts[0].outcome.emitted_output_tokens == 4
    assert result.attempts[0].outcome.wasted_decode_work == 2


def test_early_abort_wakes_d_waiter_from_stale_future_start() -> None:
    config = KernelConfig(
        0,
        100,
        ("p0",),
        ("d0",),
        {"STANDARD": 100},
        8,
        FrozenKernelCostModel(0.1, 0, 0, 0.5),
        retry_budget=1,
    )
    policy = DecodeCapacityPolicy(
        base(),
        DecodeAdmissionConfig(
            DecodeAdmissionMode.NO_GATE,
            capacity_credits=10,
            lease_boundary_tokens=4,
            abort_fences={"a": AbortFence(True, -1, True, 1)},
        ),
    )
    result = CausalKernel(config).run(
        (request("a", 0, 12, "A"), request("b", 0, 2, "B")), policy
    )
    starts = [
        (event.logical_request_id, event.at_work)
        for event in result.events
        if event.kind == "DECODE_START"
    ]
    assert starts.count(("b", 3)) == 1
    assert ("b", 7) not in starts


def test_early_abort_does_not_bypass_waiter_independent_fence() -> None:
    class FencedPolicy(DecodeCapacityPolicy):
        def decode_not_before(self, request, attempt, causal_view):
            admission_fence = super().decode_not_before(
                request, attempt, causal_view
            )
            independent = 10 if request.logical.logical_request_id == "b" else 0
            return max(admission_fence, independent)

    config = KernelConfig(
        0,
        100,
        ("p0",),
        ("d0",),
        {"STANDARD": 100},
        8,
        FrozenKernelCostModel(0.1, 0, 0, 0.5),
        retry_budget=1,
    )
    policy = FencedPolicy(
        base(),
        DecodeAdmissionConfig(
            DecodeAdmissionMode.NO_GATE,
            capacity_credits=10,
            lease_boundary_tokens=4,
            abort_fences={"a": AbortFence(True, -1, True, 1)},
        ),
    )
    result = CausalKernel(config).run(
        (request("a", 0, 12, "A"), request("b", 0, 2, "B")), policy
    )
    starts = [
        event.at_work
        for event in result.events
        if event.kind == "DECODE_START" and event.logical_request_id == "b"
    ]
    assert starts == [10]


def test_kernel_ledger_charges_waste_but_credits_logical_output_once() -> None:
    config = KernelConfig(
        0,
        100,
        ("p0",),
        ("d0",),
        {"STANDARD": 100},
        8,
        FrozenKernelCostModel(0.1, 0, 0, 0.5),
    )
    policy = DecodeCapacityPolicy(
        base(),
        DecodeAdmissionConfig(DecodeAdmissionMode.NO_GATE, capacity_credits=1),
    )
    result = CausalKernel(config).run((request("r", 0, 6),), policy)
    report = policy.summarize(result)
    assert report.useful_output_tokens == 6
    assert report.decode_work == 3
    assert report.wasted_decode_work == 0
    assert report.preemptions == 0


def gate_report(mode: DecodeAdmissionMode, goodput: float) -> DecodeRunReport:
    return DecodeRunReport(
        mode,
        20,
        100,
        50,
        0,
        0,
        goodput,
        0.9,
        0.9,
        0,
        50,
        200,
        goodput,
        {"STRICT": 0.9},
        50,
        0,
        0,
    )


def test_gate_requires_five_percent_without_offered_load_or_fairness_cheat() -> None:
    baseline = gate_report(DecodeAdmissionMode.NO_GATE, 100)
    candidate = gate_report(DecodeAdmissionMode.CAUSAL, 105)
    passing = evaluate_g12_3(
        no_gate=baseline, candidate=candidate, arrival_scale=1.5
    )
    assert passing.passed and passing.deployable_conclusion
    assert not evaluate_g12_3(
        no_gate=baseline,
        candidate=replace(
            gate_report(DecodeAdmissionMode.ORACLE_NOISED, 106),
            offered_logical_requests=19,
        ),
        arrival_scale=1.5,
    ).passed
    with pytest.raises(ValueError, match="1.5"):
        evaluate_g12_3(
            no_gate=baseline,
            candidate=gate_report(DecodeAdmissionMode.CAUSAL, 110),
            arrival_scale=1.0,
        )


@pytest.mark.parametrize(
    "override",
    [
        {"offered_tokens": 199},
        {"strict_combined_goodput": 104.9},
        {"jain_fairness": 0.899},
        {"per_tier_slo_attainment": {"STRICT": 0.79}},
        {"per_tier_slo_attainment": {"STRICT": 0.87}},
        {"actual_decode_work": 49},
        {"credit_reconciliation_credits": 1},
        {"preemptions": 1, "aborted_attempts": 0, "wasted_decode_work": 1},
    ],
)
def test_gate_rejects_token_load_combined_goodput_and_fairness_cheats(override) -> None:
    baseline = gate_report(DecodeAdmissionMode.NO_GATE, 100)
    candidate = replace(
        gate_report(DecodeAdmissionMode.CAUSAL, 106), **override
    )
    assert not evaluate_g12_3(
        no_gate=baseline, candidate=candidate, arrival_scale=1.5
    ).passed


def test_gate_rejects_mislabeled_or_unaccounted_baseline_and_candidate() -> None:
    baseline = gate_report(DecodeAdmissionMode.NO_GATE, 100)
    candidate = gate_report(DecodeAdmissionMode.CAUSAL, 106)
    assert not evaluate_g12_3(
        no_gate=replace(baseline, mode=DecodeAdmissionMode.ORACLE),
        candidate=candidate,
        arrival_scale=1.5,
    ).passed
    assert not evaluate_g12_3(
        no_gate=replace(baseline, actual_decode_work=49),
        candidate=candidate,
        arrival_scale=1.5,
    ).passed
    assert not evaluate_g12_3(
        no_gate=baseline,
        candidate=replace(candidate, mode=DecodeAdmissionMode.NO_GATE),
        arrival_scale=1.5,
    ).passed
