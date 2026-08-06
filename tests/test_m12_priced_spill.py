from __future__ import annotations

import pytest

from prefill_cache_sim.m12_priced_spill import (
    PlacementCandidate,
    PricedSpillSelector,
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
