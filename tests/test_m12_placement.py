from __future__ import annotations

import pytest

from prefill_cache_sim.m12_kernel import (
    CausalKernel,
    FrozenKernelCostModel,
    KernelConfig,
)
from prefill_cache_sim.m12_metrics import SERVICE_REGIMES
from prefill_cache_sim.m12_placement import (
    CohortTruth,
    KvsPriceMode,
    M12PlacementPolicy,
    PlacementMode,
    TraceRequestInput,
    _case_cost,
    build_kernel_requests,
    build_m12_2_cases,
    run_placement_case,
)


def trace_row(
    identity: str,
    *,
    arrival: float,
    tenant: str,
    tier: str,
    keys=("K",),
    sizes=(10,),
    output=2,
) -> TraceRequestInput:
    return TraceRequestInput(
        identity,
        tenant,
        tier,
        arrival,
        keys,
        sizes,
        output,
        "model-a",
        "adapter-a",
        "shape-a",
        100,
        ("p0", "p1"),
    )


def config(*, kvs=0.1, decode=0.5, end=100) -> KernelConfig:
    return KernelConfig(
        0,
        end,
        ("p0", "p1"),
        ("d0", "d1"),
        {"STANDARD": 20, "STRICT": 20},
        8,
        FrozenKernelCostModel(0.1, kvs, 10, decode),
    )


def test_trace_builder_preserves_explicit_identity_tenant_tier_and_prefix() -> None:
    rows = [
        trace_row("z-id", arrival=0, tenant="tenant-z", tier="STRICT"),
        trace_row(
            "a-id",
            arrival=1,
            tenant="tenant-a",
            tier="STANDARD",
            keys=("A", "B"),
            sizes=(4, 6),
        ),
    ]
    built = build_kernel_requests(rows)
    assert [item.logical.logical_request_id for item in built] == ["z-id", "a-id"]
    assert [(item.logical.tenant_id, item.logical.tier) for item in built] == [
        ("tenant-z", "STRICT"),
        ("tenant-a", "STANDARD"),
    ]
    assert built[1].prefix_cache_keys != ("A", "B")
    assert all(key.startswith("m12:v1:") for key in built[1].prefix_cache_keys)
    assert built.request_truth["a-id"].raw_prefix_keys == ("A", "B")
    assert built[1].prefix_token_sizes == (4, 6)


def test_positive_remote_transfer_is_causal_and_reported_from_execution() -> None:
    workload = build_kernel_requests(
        [
            trace_row("seed", arrival=0, tenant="t", tier="STANDARD"),
            trace_row(
                "blocker",
                arrival=2,
                tenant="t",
                tier="STANDARD",
                keys=("B",),
                sizes=(100,),
                output=0,
            ),
            trace_row("reuse", arrival=2, tenant="t", tier="STANDARD"),
        ]
    )
    policy = M12PlacementPolicy(
        PlacementMode.PRICED_SPILL,
        config().cost_model,
        kvs_enabled=True,
    )
    result = CausalKernel(config()).run(workload, policy)
    report = policy.summarize(result)
    assert report.remote_hit_tokens == 10
    assert report.kvs_normalized_work == 1
    assert report.local_hit_tokens == 0
    assert report.uncached_tokens == 110  # seed + blocker compulsory misses


def test_priced_spill_uses_only_p_queue_uncached_p_and_kvs_prices() -> None:
    policy = M12PlacementPolicy(
        PlacementMode.PRICED_SPILL,
        config().cost_model,
        kvs_enabled=True,
    )
    assert policy.eviction_regret_work == 0
    assert policy.decode_debt_work == 0


def test_load_skew_includes_idle_nodes_and_queue_p95_is_policy_observed() -> None:
    workload = build_kernel_requests(
        [trace_row("only", arrival=0, tenant="t", tier="STANDARD")]
    )
    policy = M12PlacementPolicy(
        PlacementMode.HYBRID, config().cost_model, kvs_enabled=True
    )
    report = policy.summarize(CausalKernel(config()).run(workload, policy))
    assert report.request_load_max_mean == 2
    assert report.p_queue_p95 == 0


def test_fixed_workload_and_horizon_are_identical_in_pair_runner() -> None:
    workload = build_kernel_requests(
        [
            trace_row("a", arrival=0, tenant="t1", tier="STANDARD"),
            trace_row("b", arrival=2, tenant="t2", tier="STANDARD"),
        ]
    )
    case = build_m12_2_cases(horizon=100, tier_slo_work={"STANDARD": 20})[0]
    pair = run_placement_case(workload, case)
    assert pair.hybrid.kernel_metrics.offered_logical_requests == 2
    assert pair.priced_spill.kernel_metrics.offered_logical_requests == 2
    assert pair.hybrid.kernel_metrics.observation_end_work == 100
    assert pair.priced_spill.kernel_metrics.observation_end_work == 100
    assert pair.verdict is not None
    assert {report.strategy_id for report in pair.result_table} == {
        "HYBRID",
        "PRICED_SPILL",
        "S3_GB_PREFIX_BUCKET",
        "S4_SESSION_AFFINITY",
        "S5_FLEXLB_TTFT",
        "S6_CALIBRATED_TTFT",
    }


def test_case_grid_has_all_regimes_kvs_extremes_and_decode_binding() -> None:
    cases = build_m12_2_cases(horizon=100, tier_slo_work={"STANDARD": 20})
    assert {case.regime.regime_id for case in cases} == {
        regime.regime_id for regime in SERVICE_REGIMES
    }
    assert {case.kvs_mode for case in cases} >= {
        KvsPriceMode.DISABLED,
        KvsPriceMode.NORMAL,
        KvsPriceMode.EXPENSIVE,
    }
    assert any(case.decode_binding for case in cases)
    assert any(case.kvs_contention_multiplier > 1 for case in cases)


def test_named_anchor_modes_are_stable_and_hybrid_is_mooncake_anchor() -> None:
    assert PlacementMode.HYBRID.value == "HYBRID"
    assert PlacementMode.S3.value == "S3_GB_PREFIX_BUCKET"
    assert PlacementMode.S4.value == "S4_SESSION_AFFINITY"
    assert PlacementMode.S5.value == "S5_FLEXLB_TTFT"
    assert PlacementMode.S6.value == "S6_CALIBRATED_TTFT"
    policy = M12PlacementPolicy(
        PlacementMode.S5, config().cost_model, kvs_enabled=False
    )
    assert (policy.flexlb_cache_discount, policy.flexlb_top_fraction) == (0.7, 0.3)


def test_s4_online_linker_sticks_only_from_past_prefix_history() -> None:
    workload = build_kernel_requests(
        [
            trace_row(
                "turn-1",
                arrival=0,
                tenant="t",
                tier="STANDARD",
                keys=("A", "B"),
                sizes=(5, 5),
            ),
            trace_row(
                "turn-2",
                arrival=2,
                tenant="t",
                tier="STANDARD",
                keys=("A", "B", "C"),
                sizes=(5, 5, 10),
            ),
        ]
    )
    policy = M12PlacementPolicy(
        PlacementMode.S4,
        config().cost_model,
        kvs_enabled=False,
        request_truth=workload.request_truth,
    )
    CausalKernel(config()).run(workload, policy)
    assert policy.decisions[0].node_id == policy.decisions[1].node_id


def test_builder_rejects_duplicate_identity_and_implicit_metadata() -> None:
    row = trace_row("same", arrival=0, tenant="t", tier="STANDARD")
    with pytest.raises(ValueError, match="duplicate"):
        build_kernel_requests([row, row])
    with pytest.raises(ValueError, match="tenant|tier"):
        TraceRequestInput("x", "", "", 0, ("K",), (10,), 2, "m", "a", "s", 1, ("p0",))


def test_case_uses_explicit_tier_slos_and_fairness_observes_divergence() -> None:
    workload = build_kernel_requests(
        [
            trace_row("strict", arrival=0, tenant="a", tier="STRICT"),
            trace_row("relaxed", arrival=0, tenant="b", tier="RELAXED"),
        ]
    )
    case = build_m12_2_cases(horizon=100, tier_slo_work={"STRICT": 1, "RELAXED": 10})[0]
    pair = run_placement_case(workload, case)
    attainment = pair.hybrid.kernel_metrics.per_tier_slo_attainment
    assert attainment == {"RELAXED": 1, "STRICT": 0}
    assert not pair.hybrid.kernel_metrics.fairness_floor_pass


def test_cohort_eligibility_filters_nodes_and_contention_changes_price() -> None:
    workload = build_kernel_requests(
        [
            TraceRequestInput(
                "r",
                "t",
                "STANDARD",
                0,
                ("K",),
                (10,),
                2,
                "m",
                "a",
                "shape",
                100,
                ("p1",),
            )
        ]
    )
    policy = M12PlacementPolicy(
        PlacementMode.PRICED_SPILL,
        config().cost_model,
        kvs_enabled=True,
        request_truth=workload.request_truth,
        kvs_contention_multiplier=3,
    )
    assert isinstance(workload.request_truth["r"], CohortTruth)
    CausalKernel(config()).run(workload, policy)
    assert policy.decisions[0].node_id == "p1"
    assert policy.kvs_contention_multiplier == 3


def test_destination_filter_preserves_remote_source_census() -> None:
    workload = build_kernel_requests(
        [
            TraceRequestInput(
                "seed",
                "t",
                "STANDARD",
                0,
                ("K",),
                (10,),
                2,
                "m",
                "a",
                "s",
                100,
                ("p0",),
            ),
            TraceRequestInput(
                "reuse",
                "t",
                "STANDARD",
                3,
                ("K",),
                (10,),
                2,
                "m",
                "a",
                "s",
                100,
                ("p1",),
            ),
        ]
    )
    policy = M12PlacementPolicy(
        PlacementMode.PRICED_SPILL,
        config().cost_model,
        kvs_enabled=True,
        request_truth=workload.request_truth,
    )
    report = policy.summarize(CausalKernel(config()).run(workload, policy))
    assert policy.decisions[1].node_id == "p1"
    assert report.remote_hit_tokens == 10
    assert report.kvs_normalized_work == 1


def test_cache_identity_is_namespaced_by_full_frozen_cohort() -> None:
    workload = build_kernel_requests(
        [
            TraceRequestInput(
                "seed",
                "t",
                "STANDARD",
                0,
                ("K",),
                (10,),
                2,
                "model-a",
                "adapter-a",
                "shape-a",
                100,
                ("p0",),
            ),
            TraceRequestInput(
                "cross",
                "t",
                "STANDARD",
                2,
                ("K",),
                (10,),
                2,
                "model-b",
                "adapter-a",
                "shape-a",
                100,
                ("p1",),
            ),
            TraceRequestInput(
                "same",
                "t",
                "STANDARD",
                3,
                ("K",),
                (10,),
                2,
                "model-a",
                "adapter-a",
                "shape-a",
                100,
                ("p1",),
            ),
        ]
    )
    policy = M12PlacementPolicy(
        PlacementMode.PRICED_SPILL,
        config().cost_model,
        kvs_enabled=True,
        request_truth=workload.request_truth,
    )
    result = CausalKernel(config()).run(workload, policy)
    outcomes = {item.logical_request_id: item.outcome for item in result.attempts}
    assert outcomes["cross"].prefill_gpu_work == 1
    assert outcomes["cross"].kvs_bytes == 0
    assert outcomes["same"].prefill_gpu_work == 0
    assert outcomes["same"].kvs_bytes == 100
    assert workload.request_truth["same"].raw_prefix_keys == ("K",)


def test_namespace_encoding_is_collision_safe_for_delimiter_like_values() -> None:
    workload = build_kernel_requests(
        [
            TraceRequestInput(
                "one",
                "t",
                "STANDARD",
                0,
                ("c|K",),
                (10,),
                2,
                "a|b",
                "c",
                "d",
                100,
                ("p0", "p1"),
            ),
            TraceRequestInput(
                "two",
                "t",
                "STANDARD",
                1,
                ("K",),
                (10,),
                2,
                "a",
                "b|c",
                "d",
                100,
                ("p0", "p1"),
            ),
        ]
    )
    assert workload[0].prefix_cache_keys != workload[1].prefix_cache_keys


def test_s4_cohort_identity_is_safe_for_adversarial_nul_fields() -> None:
    workload = build_kernel_requests(
        [
            TraceRequestInput(
                "one",
                "t",
                "STANDARD",
                0,
                ("X", "Y"),
                (5, 5),
                2,
                "a\x00b",
                "c",
                "d",
                100,
                ("p0", "p1"),
            ),
            TraceRequestInput(
                "two",
                "t",
                "STANDARD",
                2,
                ("X", "Y"),
                (5, 5),
                2,
                "a",
                "b\x00c",
                "d",
                100,
                ("p0", "p1"),
            ),
        ]
    )
    policy = M12PlacementPolicy(
        PlacementMode.S4,
        config().cost_model,
        kvs_enabled=False,
        request_truth=workload.request_truth,
    )
    CausalKernel(config()).run(workload, policy)
    assert policy.session_families == ("family:0", "family:1")
    assert isinstance(policy._session_families, list)


def test_slo_slack_applies_to_full_marginal_cost_not_queue_only() -> None:
    workload = build_kernel_requests(
        [
            TraceRequestInput(
                "tight",
                "t",
                "STANDARD",
                0,
                ("K",),
                (10,),
                2,
                "m",
                "a",
                "s",
                0.5,
                ("p0", "p1"),
            )
        ]
    )
    policy = M12PlacementPolicy(
        PlacementMode.PRICED_SPILL,
        config().cost_model,
        kvs_enabled=True,
        request_truth=workload.request_truth,
    )
    with pytest.raises(ValueError, match="SLO-eligible"):
        CausalKernel(config()).run(workload, policy)


def test_case_contention_changes_executed_kvs_and_completion() -> None:
    workload = build_kernel_requests(
        [
            TraceRequestInput(
                "seed",
                "t",
                "STANDARD",
                0,
                ("K",),
                (10,),
                2,
                "m",
                "a",
                "s",
                100_000,
                ("p0",),
            ),
            TraceRequestInput(
                "blocker",
                "t",
                "STANDARD",
                80_002,
                ("B",),
                (2,),
                2,
                "m",
                "a",
                "s",
                100_000,
                ("p0",),
            ),
            TraceRequestInput(
                "reuse",
                "t",
                "STANDARD",
                80_002.01,
                ("K",),
                (1_000_000,),
                2,
                "m",
                "a",
                "s",
                100_000,
                ("p0", "p1"),
            ),
        ]
    )
    cases = build_m12_2_cases(
        horizon=100_000, tier_slo_work={"STANDARD": 100_000}
    )
    normal = next(
        c
        for c in cases
        if c.kvs_mode is KvsPriceMode.NORMAL
        and c.kvs_contention_multiplier == 1
        and not c.decode_binding
        and c.regime.regime_id.value == "COMPUTE_BOUND"
    )
    contended = next(
        c
        for c in cases
        if c.kvs_contention_multiplier > 1
        and c.regime.regime_id.value == "COMPUTE_BOUND"
    )
    normal_report = run_placement_case(workload, normal).priced_spill
    contended_report = run_placement_case(workload, contended).priced_spill
    assert normal_report.remote_hit_tokens == 1_000_000
    assert contended_report.remote_hit_tokens == 0
    assert normal_report.spill_count == 1
    assert contended_report.spill_count == 0
    assert contended_report.kvs_normalized_work < normal_report.kvs_normalized_work
    assert contended_report.p_queue_p95 > normal_report.p_queue_p95
    assert contended_report.completion_max_work > normal_report.completion_max_work


def test_case_contention_is_applied_once_to_policy_and_kernel_costs() -> None:
    case = next(
        case
        for case in build_m12_2_cases(
            horizon=100, tier_slo_work={"STANDARD": 20}
        )
        if case.kvs_contention_multiplier > 1
        and case.kvs_mode is KvsPriceMode.NORMAL
    )
    executed = _case_cost(case)
    policy_base = _case_cost(case, include_contention=False)
    assert executed.kvs_work_per_token == pytest.approx(
        policy_base.kvs_work_per_token * case.kvs_contention_multiplier
    )


def test_s4_common_system_block_does_not_link_unrelated_requests() -> None:
    workload = build_kernel_requests(
        [
            trace_row(
                "a",
                arrival=0,
                tenant="t",
                tier="STANDARD",
                keys=("SYS", "A"),
                sizes=(5, 5),
            ),
            trace_row(
                "b",
                arrival=2,
                tenant="t",
                tier="STANDARD",
                keys=("SYS", "B"),
                sizes=(5, 5),
            ),
        ]
    )
    policy = M12PlacementPolicy(
        PlacementMode.S4,
        config().cost_model,
        kvs_enabled=False,
        request_truth=workload.request_truth,
    )
    CausalKernel(config()).run(workload, policy)
    assert policy.session_families == ("family:0", "family:1")


def test_s4_hot_prefix_handles_raw_truth_longer_than_count_keys() -> None:
    workload = build_kernel_requests(
        [trace_row("short", arrival=0, tenant="t", tier="STANDARD")]
    )
    policy = M12PlacementPolicy(
        PlacementMode.S4,
        config().cost_model,
        kvs_enabled=False,
        request_truth={
            "short": CohortTruth(
                "model-a",
                "adapter-a",
                "shape-a",
                100,
                frozenset({"p0", "p1"}),
                ("RAW-0", "RAW-1", "RAW-2"),
            )
        },
    )

    assert policy._causal_session_family(workload[0]) == "family:0"


def test_decode_binding_requires_executed_ledger_proof() -> None:
    workload = build_kernel_requests(
        [trace_row("small", arrival=0, tenant="t", tier="STANDARD")]
    )
    binding = next(
        case
        for case in build_m12_2_cases(horizon=100, tier_slo_work={"STANDARD": 100})
        if case.decode_binding
    )
    with pytest.raises(ValueError, match="not decode-binding"):
        run_placement_case(workload, binding)
