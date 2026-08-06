from __future__ import annotations

import pytest

from prefill_cache_sim.m12_priced_spill import (
    PlacementCandidate,
    PlacementStrategyMetrics,
    PricedSpillSelector,
    evaluate_g12_2,
)


def selector(*, kvs: float = 0.1, bias: float = 0) -> PricedSpillSelector:
    return PricedSpillSelector(1.0, kvs, bias)


def test_home_is_causal_and_deterministic() -> None:
    candidates = [PlacementCandidate("a", 0, 0, 0), PlacementCandidate("b", 0, 0, 0)]
    first = selector().select(
        prefix_family="family", input_tokens=100, candidates=candidates
    )
    second = selector().select(
        prefix_family="family", input_tokens=100, candidates=list(reversed(candidates))
    )
    assert first == second
    assert first.node_id == first.causal_home


def test_home_is_soft_and_spills_when_queue_price_dominates() -> None:
    base = [PlacementCandidate("a", 100, 0, 0), PlacementCandidate("b", 0, 0, 0)]
    probe = selector().select(prefix_family="family", input_tokens=100, candidates=base)
    home = probe.causal_home
    other = "b" if home == "a" else "a"
    candidates = [
        PlacementCandidate(home, 100, 0, 200),
        PlacementCandidate(other, 0, 0, 0),
    ]
    decision = selector().select(
        prefix_family="family", input_tokens=100, candidates=candidates
    )
    assert decision.node_id == other
    assert decision.spilled


def test_remote_reuse_charges_kvs_and_reduces_uncached_prefill_once() -> None:
    decision = selector(kvs=0.25).select(
        prefix_family="f",
        input_tokens=100,
        candidates=[PlacementCandidate("a", 20, 50, 7)],
    )
    assert decision.effective_prefill_work == 30
    assert decision.kvs_transfer_work == 12.5
    assert decision.total_marginal_work == 49.5


def test_kvs_expensive_extreme_prefers_recompute() -> None:
    candidates = [
        PlacementCandidate("a", 0, 100, 0),
        PlacementCandidate("b", 0, 0, 0),
    ]
    cheap = selector(kvs=0).select(
        prefix_family="f", input_tokens=100, candidates=candidates
    )
    expensive = selector(kvs=2).select(
        prefix_family="f", input_tokens=100, candidates=candidates
    )
    assert cheap.node_id == "a"
    assert expensive.node_id == "b"


def test_cohort_and_slo_constraints_are_hard() -> None:
    candidates = [
        PlacementCandidate("a", 100, 0, 0, model_id="other"),
        PlacementCandidate("b", 0, 0, 10, model_id="wanted"),
    ]
    decision = selector().select(
        prefix_family="f", input_tokens=100, candidates=candidates, model_id="wanted"
    )
    assert decision.node_id == "b"
    with pytest.raises(ValueError, match="no cohort-compatible"):
        selector().select(
            prefix_family="f",
            input_tokens=100,
            candidates=candidates,
            model_id="missing",
        )


def metrics(**changes: object) -> PlacementStrategyMetrics:
    values: dict[str, object] = {
        "strategy_id": "PRICED_SPILL",
        "token_hit_rate": 0.5,
        "request_load_max_mean": 1.2,
        "p_queue_p95": 100,
        "strict_useful_token_goodput": 10,
        "strict_useful_output_token_goodput": 2,
        "minimum_tier_slo_attainment": 0.9,
        "jain_fairness": 0.95,
        "per_tier_slo_attainment": {"STRICT": 0.9, "STANDARD": 0.9},
    }
    values.update(changes)
    return PlacementStrategyMetrics(**values)  # type: ignore[arg-type]


def test_g12_2_requires_two_strict_axis_improvements() -> None:
    baseline = metrics(strategy_id="HYBRID")
    assert (
        evaluate_g12_2(baseline, metrics(token_hit_rate=0.51)).verdict
        == "KILL_ENFORCEMENT"
    )
    assert (
        evaluate_g12_2(baseline, metrics(token_hit_rate=0.51, p_queue_p95=99)).verdict
        == "PASS_PROVISIONAL"
    )


def test_g12_2_fails_output_fairness_or_epsilon_regression() -> None:
    verdict = evaluate_g12_2(
        metrics(strategy_id="HYBRID"),
        metrics(
            token_hit_rate=0.51,
            p_queue_p95=99,
            strict_useful_output_token_goodput=1.99,
            jain_fairness=0.89,
        ),
    )
    assert verdict.verdict == "KILL_ENFORCEMENT"
    assert verdict.violated_axes == (
        "strict_useful_output_token_goodput",
        "jain_fairness",
    )


def test_g12_2_rejects_primary_goodput_and_relative_tier_regression() -> None:
    verdict = evaluate_g12_2(
        metrics(strategy_id="HYBRID"),
        metrics(
            token_hit_rate=0.51,
            p_queue_p95=99,
            strict_useful_token_goodput=9,
            minimum_tier_slo_attainment=0.87,
            per_tier_slo_attainment={"STRICT": 0.87, "STANDARD": 0.9},
        ),
    )
    assert verdict.violated_axes == (
        "strict_useful_token_goodput",
        "relative_tier_slo_attainment",
    )


def test_g12_2_requires_hybrid_and_valid_ratios() -> None:
    with pytest.raises(ValueError, match="HYBRID baseline"):
        evaluate_g12_2(metrics(), metrics(strategy_id="OTHER"))
    with pytest.raises(ValueError, match="ratio metrics"):
        metrics(token_hit_rate=1.01)
    with pytest.raises(ValueError, match="must match"):
        metrics(per_tier_slo_attainment={"STRICT": 0.5, "STANDARD": 0.9})


def test_per_tier_mapping_is_snapshotted() -> None:
    tiers = {"STRICT": 0.9}
    value = metrics(per_tier_slo_attainment=tiers)
    tiers["STRICT"] = 0.1
    assert value.per_tier_slo_attainment == {"STRICT": 0.9}
