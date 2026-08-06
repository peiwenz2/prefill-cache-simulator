from __future__ import annotations

from dataclasses import replace

import pytest

from prefill_cache_sim.m12_eviction import (
    CensusConfig,
    ClusterCacheCensus,
    EvictionCandidate,
    EvictionMode,
    EvictionRunReport,
    PastOnlyReuseEstimator,
    choose_eviction_victims,
    evaluate_g12_4,
)
from prefill_cache_sim.m12_kernel import (
    AttemptExecutionSpec,
    CausalKernel,
    CausalView,
    FrozenKernelCostModel,
    KernelConfig,
    KernelPolicy,
    KernelRequestSpec,
)
from prefill_cache_sim.m12_metrics import LogicalRequestSpec


def test_census_is_bounded_stale_and_cohort_fenced() -> None:
    census = ClusterCacheCensus(CensusConfig(max_entries=2, max_staleness_work=3))
    census.observe("A", "cohort-a", "p0", at_work=0, recovery_work=1)
    census.observe("B", "cohort-a", "p1", at_work=1, recovery_work=1)
    census.observe("C", "cohort-a", "p2", at_work=2, recovery_work=1)
    assert census.lookup("A", "cohort-a", now_work=2) is None
    assert census.lookup("C", "wrong-cohort", now_work=2) is None
    assert census.lookup("B", "cohort-a", now_work=5) is None


def test_census_rejects_time_reversal_and_future_snapshot_lookup() -> None:
    census = ClusterCacheCensus(CensusConfig(2, 10))
    census.observe("A", "c", "p0", at_work=5, recovery_work=1)
    with pytest.raises(ValueError, match="monotonic"):
        census.observe("A", "c", "p1", at_work=4, recovery_work=1)
    assert census.lookup("A", "c", now_work=4) is None


@pytest.mark.parametrize("bad_time", [float("nan"), float("inf")])
def test_census_rejects_non_finite_time_without_poisoning_clock(
    bad_time: float,
) -> None:
    census = ClusterCacheCensus(CensusConfig(2, 10))
    with pytest.raises(ValueError, match="finite"):
        census.observe("A", "c", "p0", at_work=bad_time, recovery_work=1)
    assert census.observe("A", "c", "p0", at_work=1, recovery_work=1)
    with pytest.raises(ValueError, match="finite"):
        census.lookup("A", "c", now_work=bad_time)


def test_spill_replica_updates_once_without_double_counting() -> None:
    census = ClusterCacheCensus(CensusConfig(max_entries=4, max_staleness_work=10))
    assert census.observe("K", "c", "sender", at_work=0, recovery_work=2)
    assert census.observe("K", "c", "receiver", at_work=1, recovery_work=1)
    assert not census.observe("K", "c", "receiver", at_work=2, recovery_work=1)
    entry = census.lookup("K", "c", now_work=2)
    assert entry is not None and entry.holders == frozenset({"sender", "receiver"})
    assert census.first_holder_observations == 1
    assert census.replica_updates == 1
    assert census.refresh_observations == 1


def test_reuse_estimator_never_reads_future_or_other_family() -> None:
    reuse = PastOnlyReuseEstimator(history_limit=3, decay_window_work=5)
    reuse.observe("A", "c", at_work=1)
    assert reuse.estimate("A", "c", at_work=1) == 0
    reuse.observe("B", "c", at_work=2)
    reuse.observe("A", "c", at_work=2)
    assert reuse.estimate("A", "c", at_work=2) > 0
    assert reuse.estimate("A", "other", at_work=2) == 0
    assert reuse.estimate("A", "c", at_work=20) == 0
    with pytest.raises(ValueError, match="future"):
        reuse.estimate("A", "c", at_work=0)


def test_regret_subtracts_cheapest_replica_recovery_and_sender_congestion() -> None:
    candidates = (
        EvictionCandidate("unique", 0.5, 10, None, True),
        EvictionCandidate("replicated", 0.5, 10, 1, False),
    )
    victims = choose_eviction_victims(candidates, required=1)
    assert victims == ("replicated",)
    congested = replace(candidates[1], cheapest_recovery_work=8)
    assert congested.regret_work < candidates[1].regret_work


def test_unique_hot_is_protected_but_cold_unique_can_be_evicted() -> None:
    candidates = (
        EvictionCandidate("hot", 0.8, 10, None, True),
        EvictionCandidate("cold", 0, 10, None, True),
    )
    assert choose_eviction_victims(candidates, required=1) == ("cold",)
    with pytest.raises(ValueError, match="protected"):
        choose_eviction_victims((candidates[0],), required=1)


class FixedPolicy(KernelPolicy):
    def plan_attempts(self, request, view):
        return (
            AttemptExecutionSpec(
                request.logical.logical_request_id,
                0,
                request.logical.arrival_work,
                "p0",
                "d0",
                request.logical.true_output_tokens,
            ),
        )


def kernel_request(identity: str, key: str, arrival: float) -> KernelRequestSpec:
    return KernelRequestSpec(
        LogicalRequestSpec(identity, "tenant", "STANDARD", arrival, 1, 1),
        (key,),
        (1,),
    )


def test_candidate_hook_protects_past_hot_unique_key_under_binding_capacity() -> None:
    config = KernelConfig(
        0,
        20,
        ("p0",),
        ("d0",),
        {"STANDARD": 20},
        1,
        FrozenKernelCostModel(1, 0, 0, 1),
    )
    census = ClusterCacheCensus(CensusConfig(8, 10))
    from prefill_cache_sim.m12_eviction import M12EvictionConfig, M12EvictionPolicy

    policy = M12EvictionPolicy(
        FixedPolicy(),
        M12EvictionConfig(EvictionMode.CENSUS_REGRET, 1, 1, 2, {}),
        census,
        cache_key_cohorts={"A": "c", "B": "c"},
        reuse=PastOnlyReuseEstimator(history_limit=4, decay_window_work=10),
    )
    workload = (
        kernel_request("a1", "A", 0),
        kernel_request("a2", "A", 3),
        kernel_request("b", "B", 6),
        kernel_request("a3", "A", 9),
    )
    result = CausalKernel(config).run(workload, policy)
    assert "A" in result.completed_cache_keys
    assert policy.summarize(result).token_hit_rate == pytest.approx(1 / 2)


def test_first_touch_does_not_freeze_capacity_one_cache() -> None:
    config = KernelConfig(
        0,
        20,
        ("p0",),
        ("d0",),
        {"STANDARD": 20},
        1,
        FrozenKernelCostModel(1, 0, 0, 1),
    )
    from prefill_cache_sim.m12_eviction import M12EvictionConfig, M12EvictionPolicy

    policy = M12EvictionPolicy(
        FixedPolicy(),
        M12EvictionConfig(EvictionMode.CENSUS_REGRET, 1, 1, 2, {}),
        ClusterCacheCensus(CensusConfig(8, 10)),
        cache_key_cohorts={key: "c" for key in "ABC"},
    )
    result = CausalKernel(config).run(
        tuple(kernel_request(key, key, index * 3) for index, key in enumerate("ABC")),
        policy,
    )
    assert result.completed_cache_keys == frozenset({"C"})


def test_hit_and_lru_refresh_stop_at_first_prefix_miss() -> None:
    from prefill_cache_sim.m12_eviction import M12EvictionConfig, M12EvictionPolicy

    policy = M12EvictionPolicy(
        FixedPolicy(),
        M12EvictionConfig(EvictionMode.LRU, 2, 1, 1, {}),
        ClusterCacheCensus(CensusConfig(8, 10)),
        cache_key_cohorts={"X": "c", "Y": "c"},
    )
    request = KernelRequestSpec(
        LogicalRequestSpec("r", "tenant", "STANDARD", 0, 2, 1),
        ("X", "Y"),
        (1, 1),
    )
    causal_view = CausalView(
        0,
        frozenset({"Y"}),
        {"p0": 0},
        {"d0": 0},
        {"p0": frozenset({"Y"})},
        {"p0": 0},
    )
    policy.plan_attempts(request, causal_view)
    assert policy.hit_tokens == 0


def test_size_metadata_survives_first_miss_for_later_resident_candidate() -> None:
    from prefill_cache_sim.m12_eviction import M12EvictionConfig, M12EvictionPolicy

    policy = M12EvictionPolicy(
        FixedPolicy(),
        M12EvictionConfig(EvictionMode.CENSUS_REGRET, 2, 1, 1, {}),
        ClusterCacheCensus(CensusConfig(8, 10)),
        cache_key_cohorts={"X": "c", "Y": "c"},
    )
    request = KernelRequestSpec(
        LogicalRequestSpec("r", "tenant", "STANDARD", 0, 101, 1),
        ("X", "Y"),
        (1, 100),
    )
    causal_view = CausalView(
        0,
        frozenset({"Y"}),
        {"p0": 0},
        {"d0": 0},
        {"p0": frozenset({"Y"})},
        {"p0": 0},
    )
    policy.plan_attempts(request, causal_view)
    candidate = policy._candidate("Y", "p0", 0)
    assert policy.hit_tokens == 0
    assert candidate.recompute_work == 100


def report(mode: EvictionMode, hit: float, goodput: float) -> EvictionRunReport:
    return EvictionRunReport(
        mode,
        offered_requests=100,
        offered_tokens=1000,
        token_hit_rate=hit,
        strict_goodput=goodput,
        strict_output_goodput=goodput,
        jain_fairness=0.9,
        minimum_tier_slo_attainment=0.8,
        per_tier_slo_attainment={"STRICT": 0.8},
        total_gpu_work=100,
        accounted_gpu_work=100,
        wasted_gpu_work=0,
        kvs_work=1,
        replica_updates=1,
        duplicate_replica_updates=0,
        winner_gpu_work=100,
        slo_missed_gpu_work=0,
        unclassified_gpu_work=0,
    )


def test_g12_4_requires_binding_and_stops_only_when_both_gains_small() -> None:
    baseline = report(EvictionMode.LRU, 0.90, 100)
    candidate = report(EvictionMode.CENSUS_REGRET, 0.905, 102)
    verdict = evaluate_g12_4(baseline, candidate, hit_ceiling=0.94)
    assert verdict.capacity_binding and verdict.passed
    assert not verdict.stop_enforcement
    with pytest.raises(ValueError, match="capacity-binding"):
        evaluate_g12_4(baseline, candidate, hit_ceiling=0.92)
    stopped = evaluate_g12_4(
        baseline,
        replace(candidate, token_hit_rate=0.904, strict_goodput=101.9),
        hit_ceiling=0.94,
    )
    assert stopped.stop_enforcement and not stopped.passed


@pytest.mark.parametrize(
    "candidate",
    [
        replace(report(EvictionMode.CENSUS_REGRET, 0.91, 103), offered_tokens=999),
        replace(report(EvictionMode.CENSUS_REGRET, 0.91, 103), jain_fairness=0.89),
        replace(report(EvictionMode.CENSUS_REGRET, 0.91, 103), total_gpu_work=99),
        replace(report(EvictionMode.CENSUS_REGRET, 0.91, 103), winner_gpu_work=90),
    ],
)
def test_g12_4_rejects_load_fairness_work_and_replica_accounting_cheats(
    candidate: EvictionRunReport,
) -> None:
    baseline = report(EvictionMode.LRU, 0.90, 100)
    assert not evaluate_g12_4(baseline, candidate, hit_ceiling=0.94).passed


@pytest.mark.parametrize("ceiling", [-0.1, 1.1])
def test_g12_4_rejects_invalid_hit_domain(ceiling: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        evaluate_g12_4(
            report(EvictionMode.LRU, 0.9, 100),
            report(EvictionMode.CENSUS_REGRET, 0.91, 103),
            hit_ceiling=ceiling,
        )
