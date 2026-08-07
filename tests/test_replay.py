from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import subprocess
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from prefill_cache_sim.calibration import (
    CalibrationStatus,
    DishonestLabelError,
    EvidenceTier,
    MachineProvenance,
    TimeUnit,
)
from prefill_cache_sim.domain import BlockRef, Request
from prefill_cache_sim.replay import (
    ABSENT_VALUE,
    ARRIVAL_SCALES,
    BASELINE_ARM_ID,
    CANDIDATE_DOES_NOT_BEAT_BASELINE,
    DEFAULT_ARMS,
    DEFAULT_SHADOW_GATE,
    DISAGREEMENT_FRACTION_ABOVE_GATE,
    ENFORCEMENT_ENABLED,
    FROZEN_RANKING_STATISTIC,
    M4_WINNER_ARM_ID,
    RANKING_CONSISTENCY_BELOW_GATE,
    RANKING_CONSISTENCY_UNAVAILABLE,
    RANKING_SCHEMA_VERSION,
    RECONCILE_SCHEMA_VERSION,
    RECONCILED_FRACTION_BELOW_GATE,
    REPLAY_SCHEMA_VERSION,
    REPRODUCIBILITY_CLAIM,
    SHADOW_GATE_SCHEMA_VERSION,
    SHADOW_SCHEMA_VERSION,
    SHARED_FIELDS,
    SOURCE_ORDER,
    SOURCE_SCHEMA_VERSION,
    TPOT_UNMODELED_REASON,
    ArmRole,
    AttemptKey,
    AttemptTraceRecord,
    ClientLatencyRecord,
    EngineHitRecord,
    FaultPlan,
    FieldPerturbation,
    LedgerEntry,
    LedgerKind,
    MalformedPayloadError,
    RankingComparison,
    ReconciledRow,
    ReconciliationReport,
    ReplayPlan,
    ShadowDecision,
    ShadowEnforcementError,
    ShadowGate,
    ShadowOutcome,
    SourceBundle,
    SourceName,
    TruthBasis,
    UndefinedRankingError,
    apply_faults,
    case_fingerprint,
    combined_digest,
    concordance,
    kendall_tau_b,
    m10_source_paths,
    pairwise_winner_agreement,
    reconcile,
    run_replay,
    source_fingerprints,
    source_manifest,
    synthetic_bundle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def sim_request(index: int, arrival_ms: float, blocks: list[int]) -> Request:
    return Request(
        f"r{index}",
        f"synthetic:{index:020d}",
        0,
        arrival_ms,
        512 * len(blocks),
        64,
        tuple(BlockRef.trace(block) for block in blocks),
        (512,) * len(blocks),
        None,
    )


def tiny_requests() -> tuple[Request, ...]:
    families = ([1, 2], [1, 2, 3], [4, 5], [1, 2, 6], [4, 5, 7], [8])
    return tuple(
        sim_request(index, index * 40.0, blocks)
        for index, blocks in enumerate(families)
    )


def key(index: int, attempt: int = 0) -> AttemptKey:
    return AttemptKey(f"synthetic:{index:020d}", attempt)


def complete_machine() -> MachineProvenance:
    return MachineProvenance(
        host_id="bench-host-0",
        accelerator_model="L20",
        engine_version="vllm-0.0.0-test",
        captured_at_utc="2026-08-05T00:00:00Z",
    )


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------


def test_synthetic_bundle_is_deterministic_for_a_given_seed() -> None:
    first = synthetic_bundle(attempt_count=12, seed=99)
    second = synthetic_bundle(attempt_count=12, seed=99)
    other = synthetic_bundle(attempt_count=12, seed=100)
    assert first == second
    assert first != other


def test_synthetic_bundle_covers_every_source_with_matching_keys() -> None:
    bundle = synthetic_bundle(attempt_count=10, seed=7)
    engine = {record.key for record in bundle.engine_hits}
    client = {record.key for record in bundle.client_latencies}
    trace = {record.key for record in bundle.attempt_traces}
    assert len(engine) == 10
    assert engine == client == trace


def test_synthetic_bundle_exercises_retry_attempts() -> None:
    bundle = synthetic_bundle(attempt_count=24, seed=3)
    attempts = {record.key.attempt_index for record in bundle.attempt_traces}
    assert attempts == {0, 1}


def test_measured_engine_truth_requires_complete_machine_provenance() -> None:
    bundle = synthetic_bundle(attempt_count=4, seed=1)
    with pytest.raises(DishonestLabelError, match="MEASURED_ENGINE"):
        replace(bundle, truth_basis=TruthBasis.MEASURED_ENGINE)
    upgraded = replace(
        bundle,
        truth_basis=TruthBasis.MEASURED_ENGINE,
        machine=complete_machine(),
    )
    assert upgraded.truth_basis is TruthBasis.MEASURED_ENGINE


def test_bundle_round_trip_preserves_schema_version_and_records() -> None:
    bundle = synthetic_bundle(attempt_count=6, seed=11)
    restored = SourceBundle.from_dict(json.loads(json.dumps(bundle.to_dict())))
    assert restored.schema_version == SOURCE_SCHEMA_VERSION
    assert restored == bundle


def test_simulator_derived_client_source_declares_tpot_as_unmodelled() -> None:
    plan = ReplayPlan(arrival_scales=(1.0,), capacity_blocks=64)
    outcome = run_replay(
        plan,
        tiny_requests(),
        trace_sha256="ab" * 32,
        git_sha=None,
        git_dirty=None,
    )
    bundle = outcome.cells[0].bundle
    assert bundle.truth_basis is TruthBasis.MODELED_SIMULATOR
    assert bundle.tpot_reason == TPOT_UNMODELED_REASON
    assert all(record.tpot_work is None for record in bundle.client_latencies)


# --------------------------------------------------------------------------
# reconciliation
# --------------------------------------------------------------------------


def test_shared_fields_are_the_only_cross_source_observables() -> None:
    assert {field.field_name for field in SHARED_FIELDS} == {
        "input_tokens",
        "node_id",
    }
    by_name = {field.field_name: field.sources for field in SHARED_FIELDS}
    assert by_name["input_tokens"] == (
        SourceName.ENGINE_HIT,
        SourceName.CLIENT_LATENCY,
        SourceName.ATTEMPT_TRACE,
    )
    assert by_name["node_id"] == (SourceName.ENGINE_HIT, SourceName.ATTEMPT_TRACE)


def test_clean_bundle_reconciles_every_attempt_with_an_empty_ledger() -> None:
    bundle = synthetic_bundle(attempt_count=16, seed=5)
    report = reconcile(bundle)
    assert report.schema_version == RECONCILE_SCHEMA_VERSION
    assert report.ledger == ()
    assert report.reconciled_count == 16
    assert report.attempt_count == 16
    assert report.reconciled_fraction == 1.0
    assert report.disagreement_fraction == 0.0


def test_reconciled_rows_carry_the_joined_three_source_view() -> None:
    bundle = synthetic_bundle(attempt_count=4, seed=13)
    row = reconcile(bundle).reconciled[0]
    engine = {record.key: record for record in bundle.engine_hits}[
        AttemptKey(row.logical_request_id, row.attempt_index)
    ]
    assert row.node_id == engine.node_id
    assert row.input_tokens == engine.input_tokens
    assert row.hit_tokens == engine.hit_tokens


def test_reconciliation_report_round_trips_through_json() -> None:
    bundle = synthetic_bundle(attempt_count=5, seed=17)
    report = reconcile(bundle)
    restored = type(report).from_dict(json.loads(json.dumps(report.to_dict())))
    assert restored == report


def test_reconciliation_summary_keeps_the_ledger_and_drops_the_row_table() -> None:
    """The summary must stay the same size as the trace grows.

    ``to_dict`` serializes every reconciled attempt, which is correct for a
    round trip but unusable in a published artifact: a full-trace replay would
    embed hundreds of thousands of rows that ``reconciliation.csv`` already
    carries. The summary keeps the evidence (the ledger and the join rates) and
    drops the table.
    """
    small = reconcile(synthetic_bundle(attempt_count=8, seed=17)).summary()
    large = reconcile(synthetic_bundle(attempt_count=256, seed=17)).summary()

    assert "reconciled" not in small
    assert small["attempt_count"] == 8
    assert large["attempt_count"] == 256
    assert small["reconciled_count"] == 8
    assert small["reconciled_fraction"] == 1.0
    assert small["disagreement_fraction"] == 0.0
    assert small["ledger"] == []
    assert len(json.dumps(small)) == len(
        json.dumps(large | {"attempt_count": 8, "reconciled_count": 8})
    )


def test_reconciliation_summary_still_carries_injected_ledger_entries() -> None:
    """Dropping the row table must not drop the disagreement evidence."""
    clean = synthetic_bundle(attempt_count=16, seed=17)
    plan = FaultPlan(
        perturb=(
            FieldPerturbation(
                SourceName.ATTEMPT_TRACE, clean.engine_hits[2].key, "node_id", "node-x"
            ),
        )
    )
    summary = reconcile(apply_faults(clean, plan).bundle).summary()
    assert [entry["kind"] for entry in summary["ledger"]] == ["DISAGREEMENT"]
    assert summary["disagreement_fraction"] > 0.0


# --------------------------------------------------------------------------
# injected faults are recovered exactly
# --------------------------------------------------------------------------


def test_injected_missing_record_is_recovered_exactly() -> None:
    bundle = synthetic_bundle(attempt_count=8, seed=21)
    target = bundle.attempt_traces[3].key
    plan = FaultPlan(drop=((SourceName.CLIENT_LATENCY, target),))
    injected = apply_faults(bundle, plan)
    report = reconcile(injected.bundle)
    assert report.ledger == injected.expected_ledger
    assert report.ledger == (
        LedgerEntry(
            LedgerKind.MISSING,
            target.logical_request_id,
            target.attempt_index,
            "",
            (SourceName.CLIENT_LATENCY,),
            ("ABSENT",),
        ),
    )
    assert report.reconciled_count == 7


def test_injected_duplicate_record_is_recovered_exactly() -> None:
    bundle = synthetic_bundle(attempt_count=8, seed=23)
    target = bundle.engine_hits[2].key
    plan = FaultPlan(duplicate=((SourceName.ENGINE_HIT, target),))
    injected = apply_faults(bundle, plan)
    report = reconcile(injected.bundle)
    assert report.ledger == injected.expected_ledger
    assert report.ledger == (
        LedgerEntry(
            LedgerKind.DUPLICATE,
            target.logical_request_id,
            target.attempt_index,
            "",
            (SourceName.ENGINE_HIT,),
            ("count=2",),
        ),
    )
    assert report.reconciled_count == 7


def test_injected_disagreement_is_recovered_exactly() -> None:
    bundle = synthetic_bundle(attempt_count=8, seed=29)
    record = bundle.attempt_traces[5]
    plan = FaultPlan(
        perturb=(
            FieldPerturbation(
                SourceName.ATTEMPT_TRACE, record.key, "node_id", "node-injected"
            ),
        )
    )
    injected = apply_faults(bundle, plan)
    report = reconcile(injected.bundle)
    assert report.ledger == injected.expected_ledger
    assert len(report.ledger) == 1
    entry = report.ledger[0]
    assert entry.kind is LedgerKind.DISAGREEMENT
    assert entry.field_name == "node_id"
    assert entry.sources == (SourceName.ENGINE_HIT, SourceName.ATTEMPT_TRACE)
    assert entry.values == (record.node_id, "node-injected")
    assert report.reconciled_count == 7
    assert report.disagreement_fraction == pytest.approx(1 / 8)


def test_mixed_fault_plan_is_recovered_exactly_and_deterministically_ordered() -> None:
    bundle = synthetic_bundle(attempt_count=20, seed=31)
    plan = FaultPlan(
        drop=(
            (SourceName.ENGINE_HIT, bundle.attempt_traces[1].key),
            (SourceName.CLIENT_LATENCY, bundle.attempt_traces[1].key),
        ),
        duplicate=((SourceName.ATTEMPT_TRACE, bundle.attempt_traces[4].key),),
        perturb=(
            FieldPerturbation(
                SourceName.CLIENT_LATENCY,
                bundle.attempt_traces[9].key,
                "input_tokens",
                999_999,
            ),
            FieldPerturbation(
                SourceName.ENGINE_HIT,
                bundle.attempt_traces[12].key,
                "node_id",
                "node-wrong",
            ),
        ),
    )
    injected = apply_faults(bundle, plan)
    report = reconcile(injected.bundle)
    assert report.ledger == injected.expected_ledger
    kinds = [entry.kind for entry in report.ledger]
    assert kinds.count(LedgerKind.MISSING) == 1
    assert kinds.count(LedgerKind.DUPLICATE) == 1
    assert kinds.count(LedgerKind.DISAGREEMENT) == 2
    assert report.reconciled_count == 16
    sort_keys = [entry.sort_key for entry in report.ledger]
    assert sort_keys == sorted(sort_keys)


def test_missing_from_two_sources_is_one_entry_listing_both() -> None:
    bundle = synthetic_bundle(attempt_count=6, seed=37)
    target = bundle.attempt_traces[0].key
    plan = FaultPlan(
        drop=(
            (SourceName.CLIENT_LATENCY, target),
            (SourceName.ENGINE_HIT, target),
        )
    )
    injected = apply_faults(bundle, plan)
    report = reconcile(injected.bundle)
    assert report.ledger == injected.expected_ledger
    assert report.ledger[0].sources == (
        SourceName.ENGINE_HIT,
        SourceName.CLIENT_LATENCY,
    )


def test_perturbing_a_non_shared_field_is_honestly_invisible() -> None:
    bundle = synthetic_bundle(attempt_count=6, seed=41)
    target = bundle.engine_hits[2].key
    plan = FaultPlan(
        perturb=(
            FieldPerturbation(SourceName.ENGINE_HIT, target, "hit_tokens", 123_456),
        )
    )
    injected = apply_faults(bundle, plan)
    report = reconcile(injected.bundle)
    assert injected.expected_ledger == ()
    assert report.ledger == ()
    assert report.reconciled_count == 6


# --------------------------------------------------------------------------
# defect precedence
#
# The unit of damage is the ``(source, key)`` pair, not the key. Damaging two
# sources of one attempt differently is the only way to put two defect kinds in
# competition, so a plan that forbade it could never falsify the precedence at
# all. These tests exist because the plan now permits exactly that.
# --------------------------------------------------------------------------


def test_fault_plan_accepts_different_damage_to_two_sources_of_one_attempt() -> None:
    target = key(0)
    plan = FaultPlan(
        drop=((SourceName.ENGINE_HIT, target),),
        perturb=(
            FieldPerturbation(SourceName.CLIENT_LATENCY, target, "input_tokens", 1),
        ),
    )
    # Each source is named once; nothing about the plan is self-contradictory.
    assert plan.affected_keys() == (target,)


def _one_damaged_attempt(plan_for: Any, seed: int) -> tuple[Any, Any]:
    """Damage attempt 0 of a fresh bundle and return (expected, report)."""
    bundle = synthetic_bundle(attempt_count=6, seed=seed)
    injected = apply_faults(bundle, plan_for(bundle.attempt_traces[0].key))
    return injected.expected_ledger, reconcile(injected.bundle)


def test_duplicate_outranks_missing_on_the_same_attempt() -> None:
    expected, report = _one_damaged_attempt(
        lambda target: FaultPlan(
            duplicate=((SourceName.ENGINE_HIT, target),),
            drop=((SourceName.CLIENT_LATENCY, target),),
        ),
        seed=101,
    )
    assert report.ledger == expected
    assert len(report.ledger) == 1
    assert report.ledger[0].kind is LedgerKind.DUPLICATE
    # The MISSING the same attempt also earned is deliberately not charged: one
    # attempt is charged one defect, so the counts stay a partition.
    assert report.ledger[0].sources == (SourceName.ENGINE_HIT,)
    assert report.ledger[0].values == ("count=2",)
    assert report.reconciled_count == 5


def test_missing_outranks_disagreement_on_the_same_attempt() -> None:
    expected, report = _one_damaged_attempt(
        lambda target: FaultPlan(
            drop=((SourceName.ENGINE_HIT, target),),
            perturb=(
                FieldPerturbation(
                    SourceName.CLIENT_LATENCY, target, "input_tokens", 999_999
                ),
            ),
        ),
        seed=103,
    )
    assert report.ledger == expected
    assert len(report.ledger) == 1
    assert report.ledger[0].kind is LedgerKind.MISSING
    assert report.ledger[0].sources == (SourceName.ENGINE_HIT,)
    assert report.ledger[0].values == (ABSENT_VALUE,)


def test_duplicate_outranks_disagreement_on_the_same_attempt() -> None:
    expected, report = _one_damaged_attempt(
        lambda target: FaultPlan(
            duplicate=((SourceName.ATTEMPT_TRACE, target),),
            perturb=(
                FieldPerturbation(
                    SourceName.CLIENT_LATENCY, target, "input_tokens", 999_999
                ),
            ),
        ),
        seed=107,
    )
    assert report.ledger == expected
    assert len(report.ledger) == 1
    assert report.ledger[0].kind is LedgerKind.DUPLICATE
    assert report.ledger[0].sources == (SourceName.ATTEMPT_TRACE,)


def test_all_three_defects_at_once_charge_only_the_most_severe() -> None:
    expected, report = _one_damaged_attempt(
        lambda target: FaultPlan(
            duplicate=((SourceName.ENGINE_HIT, target),),
            drop=((SourceName.CLIENT_LATENCY, target),),
            perturb=(
                FieldPerturbation(
                    SourceName.ATTEMPT_TRACE, target, "input_tokens", 999_999
                ),
            ),
        ),
        seed=109,
    )
    assert report.ledger == expected
    assert [entry.kind for entry in report.ledger] == [LedgerKind.DUPLICATE]


def test_fault_plan_rejects_repeated_targets() -> None:
    target = key(0)
    with pytest.raises(ValueError, match="repeats"):
        FaultPlan(
            drop=(
                (SourceName.ENGINE_HIT, target),
                (SourceName.ENGINE_HIT, target),
            )
        )


def test_fault_plan_rejects_two_damages_to_the_same_source_and_key() -> None:
    # Dropping a record and then perturbing it leaves nothing to perturb. This
    # is the contradiction the collision check still exists to catch, now that
    # it no longer collapses distinct sources together.
    target = key(0)
    with pytest.raises(ValueError, match="repeats"):
        FaultPlan(
            drop=((SourceName.ENGINE_HIT, target),),
            perturb=(
                FieldPerturbation(SourceName.ENGINE_HIT, target, "input_tokens", 1),
            ),
        )


def test_apply_faults_rejects_targets_absent_from_the_bundle() -> None:
    bundle = synthetic_bundle(attempt_count=4, seed=43)
    plan = FaultPlan(drop=((SourceName.ENGINE_HIT, key(9999)),))
    with pytest.raises(ValueError, match="must appear exactly once"):
        apply_faults(bundle, plan)


def test_apply_faults_rejects_unknown_field_names() -> None:
    bundle = synthetic_bundle(attempt_count=4, seed=47)
    plan = FaultPlan(
        perturb=(
            FieldPerturbation(
                SourceName.ENGINE_HIT, bundle.engine_hits[0].key, "nope", 1
            ),
        )
    )
    with pytest.raises(ValueError, match="unknown field"):
        apply_faults(bundle, plan)


# --------------------------------------------------------------------------
# oracle independence
#
# ``report.ledger == injected.expected_ledger`` is the assertion every fault
# test rests on. It is only worth anything if it can fail, and only worth
# anything against a *shared* bug if the two sides were derived differently.
# --------------------------------------------------------------------------


def test_the_ledger_comparison_fails_when_the_plan_did_not_declare_the_damage() -> None:
    bundle = synthetic_bundle(attempt_count=6, seed=53)
    plan = FaultPlan(drop=((SourceName.ENGINE_HIT, bundle.attempt_traces[0].key),))
    injected = apply_faults(bundle, plan)
    assert reconcile(injected.bundle).ledger == injected.expected_ledger

    # Remove a record behind the plan's back. The oracle states the plan and
    # never reads the damaged bundle, so it cannot know about this -- and the
    # comparison above must therefore be capable of failing.
    smuggled = injected.bundle.with_records(
        SourceName.CLIENT_LATENCY, injected.bundle.client_latencies[:-1]
    )
    assert reconcile(smuggled).ledger != injected.expected_ledger


def test_the_expected_ledger_names_which_source_reported_the_wrong_value() -> None:
    bundle = synthetic_bundle(attempt_count=6, seed=59)
    target = bundle.attempt_traces[2].key

    def values_when_perturbed(source: SourceName) -> tuple[str, ...]:
        plan = FaultPlan(
            perturb=(FieldPerturbation(source, target, "node_id", "node-wrong"),)
        )
        injected = apply_faults(bundle, plan)
        report = reconcile(injected.bundle)
        assert report.ledger == injected.expected_ledger
        assert len(report.ledger) == 1
        return report.ledger[0].values

    engine_side = values_when_perturbed(SourceName.ENGINE_HIT)
    trace_side = values_when_perturbed(SourceName.ATTEMPT_TRACE)
    # ``node_id`` is observed by (ENGINE_HIT, ATTEMPT_TRACE) in that order, so
    # the *position* of the wrong value is what names the source that was
    # wrong. An oracle that only counted disagreements would pass both of these
    # identically and never notice it had blamed the wrong observer.
    assert engine_side[0] == "node-wrong"
    assert trace_side[1] == "node-wrong"
    assert engine_side != trace_side


# --------------------------------------------------------------------------
# client latency includes the queue
#
# The client hands a request over at ``arrival`` and sees the first token at
# ``finish``. ``start`` is the scheduler's private moment. Measuring from
# ``start`` would subtract the queueing delay -- which is precisely the thing an
# arrival-rate sweep exists to expose -- and call the result a client latency.
# --------------------------------------------------------------------------


def _queued_attempts(bundle: SourceBundle) -> list[tuple[Any, Any]]:
    traces = {record.key: record for record in bundle.attempt_traces}
    return [
        (record, traces[record.key])
        for record in bundle.client_latencies
        if traces[record.key].start_work > traces[record.key].arrival_work
    ]


def _assert_ttft_spans_arrival_to_finish(pairs: list[tuple[Any, Any]]) -> None:
    # Non-vacuous by construction: a bundle in which nobody ever queued would
    # satisfy both forms of the assertion and prove nothing.
    assert pairs
    for client, trace in pairs:
        assert client.ttft_work == pytest.approx(trace.finish_work - trace.arrival_work)
        assert client.ttft_work != pytest.approx(trace.finish_work - trace.start_work)


def test_the_fixture_client_latency_includes_the_queueing_delay() -> None:
    _assert_ttft_spans_arrival_to_finish(
        _queued_attempts(synthetic_bundle(attempt_count=32, seed=61))
    )


def test_the_modeled_client_latency_includes_the_queueing_delay() -> None:
    outcome = run_replay(
        ReplayPlan(capacity_blocks=64),
        tiny_requests(),
        trace_sha256="ab" * 32,
        git_sha=None,
        git_dirty=None,
    )
    pairs = [pair for cell in outcome.cells for pair in _queued_attempts(cell.bundle)]
    _assert_ttft_spans_arrival_to_finish(pairs)


# --------------------------------------------------------------------------
# ranking consistency statistic
# --------------------------------------------------------------------------


def test_frozen_statistic_is_kendall_tau_b() -> None:
    assert FROZEN_RANKING_STATISTIC == "KENDALL_TAU_B"
    assert RANKING_SCHEMA_VERSION == "m10-ranking-v1"


def test_identical_rankings_score_one() -> None:
    assert kendall_tau_b([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]) == 1.0
    assert pairwise_winner_agreement([1.0, 2.0, 3.0], [5.0, 6.0, 7.0]) == 1.0


def test_reversed_rankings_score_minus_one_and_zero_agreement() -> None:
    assert kendall_tau_b([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == -1.0
    assert pairwise_winner_agreement([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == 0.0


def test_tau_b_corrects_for_ties_on_one_side() -> None:
    assert kendall_tau_b([1.0, 1.0, 2.0], [1.0, 2.0, 3.0]) == pytest.approx(
        2 / (2 * 3) ** 0.5
    )


def test_tau_b_corrects_for_ties_on_both_sides() -> None:
    # n0 = 6, n1 = 1 (a ties 0-1), n2 = 1 (b ties 2-3)
    left = [1.0, 1.0, 2.0, 3.0]
    right = [1.0, 2.0, 3.0, 3.0]
    assert kendall_tau_b(left, right) == pytest.approx(4 / ((6 - 1) * (6 - 1)) ** 0.5)


def test_pairwise_agreement_ignores_pairs_tied_on_either_side() -> None:
    assert pairwise_winner_agreement([1.0, 1.0, 2.0], [1.0, 2.0, 3.0]) == 1.0
    assert pairwise_winner_agreement([1.0, 1.0, 2.0], [1.0, 2.0, 0.0]) == 0.0


def test_tau_b_is_undefined_when_one_side_is_fully_tied() -> None:
    with pytest.raises(ValueError, match="undefined"):
        kendall_tau_b([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])


def test_pairwise_agreement_is_undefined_without_a_strict_pair() -> None:
    with pytest.raises(ValueError, match="no strictly ordered pair"):
        pairwise_winner_agreement([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])


def test_ranking_statistics_reject_malformed_input() -> None:
    with pytest.raises(ValueError, match="equal length"):
        kendall_tau_b([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="at least two"):
        kendall_tau_b([1.0], [1.0])
    with pytest.raises(ValueError, match="equal length"):
        pairwise_winner_agreement([1.0, 2.0], [1.0])


def test_ranking_comparison_pairs_scores_by_shared_arm_id() -> None:
    comparison = RankingComparison.from_scores(
        "sim",
        "replay",
        {"S3": 0.54, "S0": 0.44, "S4": 0.53},
        {"S3": 0.55, "S0": 0.43, "S4": 0.52},
    )
    assert comparison.arm_ids == ("S0", "S3", "S4")
    assert comparison.statistic == FROZEN_RANKING_STATISTIC
    assert comparison.tau_b == 1.0
    assert comparison.pairwise_agreement == 1.0
    assert comparison.concordant_pairs == 3
    assert comparison.discordant_pairs == 0


def test_ranking_comparison_rejects_mismatched_arm_sets() -> None:
    with pytest.raises(ValueError, match="same arm ids"):
        RankingComparison.from_scores(
            "sim", "replay", {"S3": 1.0, "S0": 2.0}, {"S3": 1.0, "S5": 2.0}
        )


def test_ranking_comparison_round_trips_through_json() -> None:
    comparison = RankingComparison.from_scores(
        "sim", "replay", {"a": 1.0, "b": 2.0, "c": 3.0}, {"a": 3.0, "b": 2.0, "c": 1.0}
    )
    restored = RankingComparison.from_dict(json.loads(json.dumps(comparison.to_dict())))
    assert restored == comparison
    assert restored.tau_b == -1.0


# --------------------------------------------------------------------------
# a non-finite score is refused before it can be mistaken for a tie
#
# Every comparison against NaN is false, so an unchecked ``_sign`` would call
# the pair a *tie* -- the NaN arm would drop out of the counts and the arms that
# remain would report their own agreement as if it covered every arm. The
# damage is silent and it flatters: the one input the statistic cannot judge
# would produce its most confident answer.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_ranking_statistics_refuse_a_non_finite_score(bad: float) -> None:
    for left, right in (
        ([1.0, 2.0, bad], [1.0, 2.0, 3.0]),
        ([1.0, 2.0, 3.0], [1.0, 2.0, bad]),
    ):
        with pytest.raises(MalformedPayloadError, match="must be a finite number"):
            concordance(left, right)
        with pytest.raises(MalformedPayloadError, match="must be a finite number"):
            kendall_tau_b(left, right)
        with pytest.raises(MalformedPayloadError, match="must be a finite number"):
            pairwise_winner_agreement(left, right)


def test_a_non_finite_score_names_the_side_and_position_it_sits_in() -> None:
    with pytest.raises(MalformedPayloadError, match="right score at position 1"):
        kendall_tau_b([1.0, 2.0, 3.0], [1.0, math.nan, 3.0])


def test_refusing_nan_is_what_stops_a_forged_perfect_agreement() -> None:
    """The arms that survive a dropped NaN would have scored a perfect 1.0.

    This is the fabrication the refusal exists to prevent, so it is stated
    rather than assumed: the surviving pair really does agree perfectly, which
    is exactly why silently dropping the third arm would be so convincing.
    """
    assert kendall_tau_b([1.0, 2.0], [10.0, 20.0]) == 1.0
    assert pairwise_winner_agreement([1.0, 2.0], [10.0, 20.0]) == 1.0

    with pytest.raises(MalformedPayloadError):
        kendall_tau_b([1.0, 2.0, math.nan], [10.0, 20.0, 30.0])


def test_unrankable_and_unusable_are_different_failures() -> None:
    """A caller may record "no ranking evidence" without swallowing corruption.

    ``UndefinedRankingError`` and ``MalformedPayloadError`` are siblings under
    ``ValueError``, so neither ``except`` clause catches the other. A caller
    that collapsed them would report a corrupt artifact as an honest absence of
    evidence.
    """
    with pytest.raises(UndefinedRankingError):
        kendall_tau_b([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
    with pytest.raises(MalformedPayloadError):
        kendall_tau_b([1.0, math.nan, 1.0], [1.0, 2.0, 3.0])

    # The caller that matters is the one written to tolerate an unrankable pair.
    # Its ``except`` must not also absorb the corrupt one.
    try:
        kendall_tau_b([1.0, math.nan, 1.0], [1.0, 2.0, 3.0])
    except UndefinedRankingError:
        pytest.fail("corruption was reported as an honest absence of ranking evidence")
    except MalformedPayloadError:
        pass

    assert not issubclass(UndefinedRankingError, MalformedPayloadError)
    assert not issubclass(MalformedPayloadError, UndefinedRankingError)


def test_json_admits_nan_so_the_comparison_boundary_must_refuse_it() -> None:
    """``json.loads`` parses ``NaN`` even though JSON has no such literal.

    That is the whole reason the refusal has to live in ``from_dict`` and not
    only in the parser: a hand-edited artifact can carry a value the format
    does not define.
    """
    comparison = RankingComparison.from_scores(
        "sim", "replay", {"a": 1.0, "b": 2.0, "c": 3.0}, {"a": 1.0, "b": 2.0, "c": 3.0}
    )
    text = json.dumps(comparison.to_dict()).replace(
        '"left_scores": [1.0', '"left_scores": [NaN'
    )
    assert math.isnan(json.loads(text)["left_scores"][0])

    with pytest.raises(MalformedPayloadError, match="must be a finite number"):
        RankingComparison.from_dict(json.loads(text))


def test_a_non_finite_tau_b_is_refused_by_the_comparison_record() -> None:
    payload = RankingComparison.from_scores(
        "sim", "replay", {"a": 1.0, "b": 2.0, "c": 3.0}, {"a": 1.0, "b": 2.0, "c": 3.0}
    ).to_dict()
    for bad in ("NaN", "Infinity"):
        text = json.dumps(payload).replace('"tau_b": 1.0', f'"tau_b": {bad}')
        with pytest.raises(MalformedPayloadError, match="must be a finite number"):
            RankingComparison.from_dict(json.loads(text))


# --------------------------------------------------------------------------
# shadow decisions are recorded, never enforced
# --------------------------------------------------------------------------


def test_enforcement_is_disabled_at_module_scope() -> None:
    assert ENFORCEMENT_ENABLED is False


def test_shadow_decision_cannot_be_constructed_as_enforced() -> None:
    with pytest.raises(ShadowEnforcementError, match="never enforced"):
        ShadowDecision(
            schema_version="m10-shadow-v1",
            policy=ShadowGate(),
            baseline_arm_id="S0_RANDOM",
            candidate_arm_id="S3_GB_PREFIX_BUCKET",
            outcome=ShadowOutcome.SHADOW_RECOMMENDED,
            enforced=True,
            reasons=(),
            reconciled_fraction=1.0,
            disagreement_fraction=0.0,
            tau_b=1.0,
            baseline_score=0.4,
            candidate_score=0.5,
        )


def test_shadow_decision_enforce_always_raises() -> None:
    decision = ShadowGate().decide(
        baseline_arm_id="S0_RANDOM",
        candidate_arm_id="S3_GB_PREFIX_BUCKET",
        reconciled_fraction=1.0,
        disagreement_fraction=0.0,
        tau_b=1.0,
        baseline_score=0.44,
        candidate_score=0.54,
    )
    assert decision.enforced is False
    assert decision.outcome is ShadowOutcome.SHADOW_RECOMMENDED
    with pytest.raises(ShadowEnforcementError, match="never enforced"):
        decision.enforce()


def test_shadow_gate_withholds_and_records_every_failed_reason() -> None:
    decision = ShadowGate(
        minimum_reconciled_fraction=0.99,
        maximum_disagreement_fraction=0.005,
        minimum_tau_b=0.6,
    ).decide(
        baseline_arm_id="S0_RANDOM",
        candidate_arm_id="S5_FLEXLB_TTFT",
        reconciled_fraction=0.5,
        disagreement_fraction=0.4,
        tau_b=0.1,
        baseline_score=0.44,
        candidate_score=0.40,
    )
    assert decision.outcome is ShadowOutcome.SHADOW_WITHHELD
    assert decision.reasons == (
        "RECONCILED_FRACTION_BELOW_GATE",
        "DISAGREEMENT_FRACTION_ABOVE_GATE",
        "RANKING_CONSISTENCY_BELOW_GATE",
        "CANDIDATE_DOES_NOT_BEAT_BASELINE",
    )


def test_shadow_gate_withholds_when_ranking_consistency_is_undefined() -> None:
    decision = ShadowGate().decide(
        baseline_arm_id="S0_RANDOM",
        candidate_arm_id="S3_GB_PREFIX_BUCKET",
        reconciled_fraction=1.0,
        disagreement_fraction=0.0,
        tau_b=None,
        baseline_score=0.44,
        candidate_score=0.54,
    )
    assert decision.outcome is ShadowOutcome.SHADOW_WITHHELD
    assert decision.reasons == ("RANKING_CONSISTENCY_UNAVAILABLE",)


def test_shadow_decision_round_trips_through_json() -> None:
    decision = ShadowGate().decide(
        baseline_arm_id="S0_RANDOM",
        candidate_arm_id="S4_SESSION_AFFINITY",
        reconciled_fraction=1.0,
        disagreement_fraction=0.0,
        tau_b=0.9,
        baseline_score=0.44,
        candidate_score=0.53,
    )
    restored = ShadowDecision.from_dict(json.loads(json.dumps(decision.to_dict())))
    assert restored == decision
    assert restored.to_dict()["enforced"] is False


# --------------------------------------------------------------------------
# a non-finite input never reaches a threshold comparison
#
# Every gate is a ``<`` or a ``>`` against a float, and every such comparison
# with NaN is false. An unchecked NaN would therefore clear *all* the gates at
# once and arrive at SHADOW_RECOMMENDED -- the input the gate is least able to
# judge producing its most confident output.
# --------------------------------------------------------------------------


def _gate_inputs(**overrides: Any) -> dict[str, Any]:
    """Inputs that are recommended on their own, ready to be spoiled one at a time."""
    inputs: dict[str, Any] = {
        "baseline_arm_id": "S0_RANDOM",
        "candidate_arm_id": "S3_GB_PREFIX_BUCKET",
        "reconciled_fraction": 1.0,
        "disagreement_fraction": 0.0,
        "tau_b": 1.0,
        "baseline_score": 0.44,
        "candidate_score": 0.54,
    }
    inputs.update(overrides)
    return inputs


def test_the_unspoiled_gate_inputs_really_do_recommend() -> None:
    decision = ShadowGate().decide(**_gate_inputs())
    assert decision.outcome is ShadowOutcome.SHADOW_RECOMMENDED
    assert decision.reasons == ()


@pytest.mark.parametrize(
    "field",
    [
        "reconciled_fraction",
        "disagreement_fraction",
        "tau_b",
        "baseline_score",
        "candidate_score",
    ],
)
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_the_gate_refuses_a_non_finite_input_rather_than_recommending(
    field: str, bad: float
) -> None:
    with pytest.raises(MalformedPayloadError, match="ShadowGate.decide"):
        ShadowGate().decide(**_gate_inputs(**{field: bad}))


def test_a_nan_input_would_otherwise_have_cleared_every_gate_at_once() -> None:
    """Stated explicitly, because the failure mode is counter-intuitive.

    A NaN is below no minimum and above no maximum, so each individual gate
    would have passed it. The refusal is the only thing standing between an
    unjudgeable input and the gate's most confident verdict.
    """
    gate = ShadowGate()
    nan = math.nan
    assert not (nan < gate.minimum_reconciled_fraction)
    assert not (nan > gate.maximum_disagreement_fraction)
    assert not (nan < gate.minimum_tau_b)
    assert not (nan <= nan)

    with pytest.raises(MalformedPayloadError):
        gate.decide(
            **_gate_inputs(
                reconciled_fraction=nan,
                disagreement_fraction=nan,
                tau_b=nan,
                baseline_score=nan,
                candidate_score=nan,
            )
        )


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("minimum_reconciled_fraction", math.nan),
        ("minimum_reconciled_fraction", math.inf),
        ("maximum_disagreement_fraction", math.nan),
        ("maximum_disagreement_fraction", math.inf),
        ("minimum_tau_b", math.nan),
        ("minimum_tau_b", -math.inf),
    ],
)
def test_a_gate_cannot_be_built_with_a_non_finite_threshold(
    field: str, bad: float
) -> None:
    threshold: dict[str, Any] = {field: bad}
    with pytest.raises(MalformedPayloadError, match="ShadowGate"):
        ShadowGate(**threshold)


def test_a_recorded_decision_refuses_a_non_finite_measurement() -> None:
    payload = ShadowGate().decide(**_gate_inputs()).to_dict()
    for key in ("reconciled_fraction", "tau_b", "baseline_score"):
        text = json.dumps({**payload, key: 0.5}).replace(
            f'"{key}": 0.5', f'"{key}": NaN'
        )
        with pytest.raises(MalformedPayloadError, match="must be a finite number"):
            ShadowDecision.from_dict(json.loads(text))


# --------------------------------------------------------------------------
# replay orchestrator
# --------------------------------------------------------------------------


def test_default_plan_matches_the_m10_arm_matrix() -> None:
    assert ARRIVAL_SCALES == (1.0, 2.0)
    assert BASELINE_ARM_ID == "S0_RANDOM"
    assert M4_WINNER_ARM_ID == "S4_SESSION_AFFINITY"
    assert tuple(arm.arm_id for arm in DEFAULT_ARMS) == (
        "S0_RANDOM",
        "S3_GB_PREFIX_BUCKET",
        "S5_FLEXLB_TTFT",
        "S4_SESSION_AFFINITY",
    )
    roles = {arm.arm_id: arm.role for arm in DEFAULT_ARMS}
    assert roles["S0_RANDOM"] is ArmRole.BASELINE
    assert roles["S4_SESSION_AFFINITY"] is ArmRole.M4_WINNER
    assert roles["S5_FLEXLB_TTFT"] is ArmRole.STOP_GATED
    assert ReplayPlan().node_count == 4


def test_plan_expands_to_one_case_per_arm_and_arrival_scale() -> None:
    cases = ReplayPlan().cases()
    assert len(cases) == 8
    assert len({case.case_id for case in cases}) == 8
    assert {case.replay_speed for case in cases} == {1.0, 2.0}
    assert {case.node_count for case in cases} == {4}
    for case in cases:
        assert case.case_id.startswith("M10-")


def test_case_fingerprint_is_stable_and_config_sensitive() -> None:
    case = ReplayPlan().cases()[0]
    assert case_fingerprint(case) == case_fingerprint(case)
    assert case_fingerprint(case) != case_fingerprint(replace(case, seed=714))


def test_plan_rejects_an_unknown_arm_selector() -> None:
    with pytest.raises(ValueError, match="unknown selector"):
        ReplayPlan(arms=(replace(DEFAULT_ARMS[0], selector_id="S99_NOPE"),)).cases()


def test_plan_rejects_non_positive_arrival_scales() -> None:
    with pytest.raises(ValueError, match="arrival scales"):
        ReplayPlan(arrival_scales=(1.0, 0.0))


def test_replay_outcome_carries_only_synthetic_honesty_labels() -> None:
    outcome = run_replay(
        ReplayPlan(arrival_scales=(1.0,), capacity_blocks=64),
        tiny_requests(),
        trace_sha256="ab" * 32,
        git_sha="ef" * 20,
        git_dirty=True,
    )
    assert outcome.schema_version == REPLAY_SCHEMA_VERSION
    assert outcome.evidence_tier is EvidenceTier.SYNTHETIC_REPLAY
    assert outcome.calibration_status is CalibrationStatus.SYNTHETIC_UNCALIBRATED
    assert outcome.time_unit is TimeUnit.NORMALIZED_WORK
    assert outcome.machine.complete is False


def test_replay_refuses_hardware_labels_without_machine_provenance() -> None:
    with pytest.raises(DishonestLabelError):
        run_replay(
            ReplayPlan(arrival_scales=(1.0,), capacity_blocks=64),
            tiny_requests(),
            trace_sha256="ab" * 32,
            git_sha=None,
            git_dirty=None,
            evidence_tier=EvidenceTier.HW_VALIDATED,
        )


def test_replay_produces_one_cell_per_arm_and_scale() -> None:
    outcome = run_replay(
        ReplayPlan(capacity_blocks=64),
        tiny_requests(),
        trace_sha256="ab" * 32,
        git_sha=None,
        git_dirty=None,
    )
    assert len(outcome.cells) == 8
    assert outcome.cell("S3_GB_PREFIX_BUCKET", 2.0).result.replay_speed == 2.0
    assert outcome.cell("S0_RANDOM", 1.0).arm_role is ArmRole.BASELINE


def test_replay_cells_reconcile_cleanly_against_their_own_simulation() -> None:
    outcome = run_replay(
        ReplayPlan(capacity_blocks=64),
        tiny_requests(),
        trace_sha256="ab" * 32,
        git_sha=None,
        git_dirty=None,
    )
    for cell in outcome.cells:
        assert cell.reconciliation.ledger == ()
        assert cell.reconciliation.reconciled_count == cell.result.completed_requests


def test_replay_scores_expose_a_ranking_per_arrival_scale() -> None:
    outcome = run_replay(
        ReplayPlan(capacity_blocks=64),
        tiny_requests(),
        trace_sha256="ab" * 32,
        git_sha=None,
        git_dirty=None,
    )
    scores = outcome.scores("token_weighted_hit_rate", 1.0)
    assert set(scores) == {arm.arm_id for arm in DEFAULT_ARMS}
    doubled = outcome.scores("token_weighted_hit_rate", 2.0)
    assert set(doubled) == set(scores)


def test_replay_rejects_an_unknown_score_metric() -> None:
    outcome = run_replay(
        ReplayPlan(arrival_scales=(1.0,), capacity_blocks=64),
        tiny_requests(),
        trace_sha256="ab" * 32,
        git_sha=None,
        git_dirty=None,
    )
    with pytest.raises(ValueError, match="unknown metric"):
        outcome.scores("not_a_metric", 1.0)
    with pytest.raises(KeyError, match="no cell"):
        outcome.cell("S0_RANDOM", 7.0)


def test_replay_is_reproducible_for_a_fixed_plan_and_request_set() -> None:
    kwargs: dict[str, Any] = {
        "trace_sha256": "ab" * 32,
        "git_sha": None,
        "git_dirty": None,
    }
    plan = ReplayPlan(arrival_scales=(1.0,), capacity_blocks=64)
    first = run_replay(plan, tiny_requests(), **kwargs)
    second = run_replay(plan, tiny_requests(), **kwargs)
    assert [cell.result.to_dict() for cell in first.cells] == [
        cell.result.to_dict() for cell in second.cells
    ]


def test_replay_outcome_serialization_never_claims_milliseconds() -> None:
    outcome = run_replay(
        ReplayPlan(arrival_scales=(1.0,), capacity_blocks=64),
        tiny_requests(),
        trace_sha256="ab" * 32,
        git_sha=None,
        git_dirty=None,
    )
    payload = outcome.to_dict()
    assert payload["time_unit"] == "NORMALIZED_WORK"
    assert payload["evidence_tier"] == "SYNTHETIC_REPLAY"
    assert payload["calibration_status"] == "SYNTHETIC_UNCALIBRATED"
    assert "MILLISECONDS" not in json.dumps(payload)


def test_shadow_decisions_from_a_replay_are_recorded_but_not_enforced() -> None:
    outcome = run_replay(
        ReplayPlan(capacity_blocks=64),
        tiny_requests(),
        trace_sha256="ab" * 32,
        git_sha=None,
        git_dirty=None,
    )
    decisions = outcome.shadow_decisions("token_weighted_hit_rate")
    assert len(decisions) == 3
    assert {decision.candidate_arm_id for decision in decisions} == {
        "S3_GB_PREFIX_BUCKET",
        "S5_FLEXLB_TTFT",
        "S4_SESSION_AFFINITY",
    }
    assert all(decision.enforced is False for decision in decisions)
    assert all(decision.baseline_arm_id == BASELINE_ARM_ID for decision in decisions)


def test_records_expose_generic_field_access_for_the_reconciler() -> None:
    engine = EngineHitRecord(key(0), "node-0", 1024, 512)
    client = ClientLatencyRecord(key(0), 1024, 64, 12.5, None)
    trace = AttemptTraceRecord(key(0), "node-0", 1024, 0.0, 1.0, 12.5)
    assert engine.field("node_id") == "node-0"
    assert client.field("input_tokens") == 1024
    assert trace.field("node_id") == "node-0"
    with pytest.raises(ValueError, match="unknown field"):
        engine.field("nope")


# --------------------------------------------------------------------------
# hostile payloads
#
# Every ``from_dict`` here is a system boundary. A wrong *type* is more
# dangerous than a missing key: it does not crash, it manufactures a finding.
# These tests pin the two specific fabrications the ledger exists to rule out.
# --------------------------------------------------------------------------


def test_stringified_input_tokens_are_refused_rather_than_coerced() -> None:
    payload = EngineHitRecord(key(0), "node-0", 1024, 512).to_dict()
    payload["input_tokens"] = "1024"
    # Coercing this would make an engine reporting "1024" disagree with a client
    # reporting 1024, i.e. a DISAGREEMENT between two observers that agree.
    with pytest.raises(MalformedPayloadError, match="must be an integer"):
        EngineHitRecord.from_dict(payload)


def test_stringified_attempt_index_cannot_forge_a_second_attempt() -> None:
    payload = key(0).to_dict()
    payload["attempt_index"] = "0"
    # A string index hashes and sorts differently from its integer twin, so one
    # attempt would become two and each would be charged MISSING three times.
    with pytest.raises(MalformedPayloadError, match="must be an integer"):
        AttemptKey.from_dict(payload)


def test_booleans_are_refused_where_a_number_is_required() -> None:
    # bool subclasses int, so an unchecked `attempt_index: true` becomes 1.
    payload = key(0).to_dict()
    payload["attempt_index"] = True
    with pytest.raises(MalformedPayloadError, match="must be an integer"):
        AttemptKey.from_dict(payload)

    latency = ClientLatencyRecord(key(0), 1024, 64, 12.5, None).to_dict()
    latency["ttft_work"] = True
    with pytest.raises(MalformedPayloadError, match="must be a number"):
        ClientLatencyRecord.from_dict(latency)


def _disagreement_payload() -> dict[str, Any]:
    """A well-formed DISAGREEMENT entry, ready to be corrupted one field at a time."""
    return LedgerEntry(
        LedgerKind.DISAGREEMENT,
        "synthetic:0",
        0,
        "input_tokens",
        SOURCE_ORDER,
        ("1024", "2048", "1024"),
    ).to_dict()


def test_a_string_is_refused_where_an_array_is_required() -> None:
    entry = _disagreement_payload()
    # Iterating a string yields characters, so "abc" would silently become a
    # three-element ledger value list.
    entry["values"] = "abc"
    with pytest.raises(MalformedPayloadError, match="must be an array"):
        LedgerEntry.from_dict(entry)


def test_a_non_string_element_inside_an_array_is_named() -> None:
    entry = _disagreement_payload()
    entry["values"] = ["1024", 1024, "1024"]
    with pytest.raises(MalformedPayloadError, match="must be an array of strings"):
        LedgerEntry.from_dict(entry)


def test_a_missing_nested_key_is_blamed_on_the_parent_record() -> None:
    payload = EngineHitRecord(key(0), "node-0", 1024, 512).to_dict()
    del payload["key"]
    # Delegating straight to AttemptKey.from_dict would report AttemptKey's first
    # missing field, sending a reader looking in the wrong record entirely.
    with pytest.raises(MalformedPayloadError, match="EngineHitRecord.*'key'"):
        EngineHitRecord.from_dict(payload)


def test_a_nested_key_that_is_not_an_object_is_refused() -> None:
    payload = AttemptTraceRecord(key(0), "node-0", 1024, 0.0, 1.0, 12.5).to_dict()
    payload["key"] = "synthetic:0#0"
    with pytest.raises(MalformedPayloadError, match="must be an object"):
        AttemptTraceRecord.from_dict(payload)


def test_a_non_mapping_payload_is_refused_before_any_field_is_read() -> None:
    payloads: tuple[Any, ...] = ([], "{}", 7, None)
    for bad in payloads:
        with pytest.raises(MalformedPayloadError, match="expected an object"):
            EngineHitRecord.from_dict(bad)  # type: ignore[arg-type]


def test_null_is_accepted_only_where_the_schema_admits_it() -> None:
    payload = ClientLatencyRecord(key(0), 1024, 64, 12.5, None).to_dict()
    # tpot_work is legitimately null: the simulator does not model decode.
    assert ClientLatencyRecord.from_dict(payload).tpot_work is None
    payload["ttft_work"] = None
    with pytest.raises(MalformedPayloadError, match="must be a number"):
        ClientLatencyRecord.from_dict(payload)


def test_every_source_record_survives_its_own_round_trip() -> None:
    bundle = synthetic_bundle(attempt_count=8, seed=7)
    assert SourceBundle.from_dict(json.loads(json.dumps(bundle.to_dict()))) == bundle


# --------------------------------------------------------------------------
# hand-edited artifacts
#
# The tests above pin the *types* a payload may carry. These pin what the
# values must mean together. A published artifact is a text file: anyone can
# open it and change a number. A record that checked its fields but not its own
# conclusions would let an edited digit become a finding, which is worse than a
# crash because it is quotable.
#
# Each helper below produces a payload that is valid on its own, so that a test
# corrupting one key is testing that key and nothing else. Every helper is
# checked for validity first, otherwise a helper that silently broke would turn
# every test built on it into a tautology.
# --------------------------------------------------------------------------


def _comparison_payload(**overrides: Any) -> dict[str, Any]:
    payload = RankingComparison.from_scores(
        "sim",
        "replay",
        {"a": 1.0, "b": 2.0, "c": 3.0},
        {"a": 1.5, "b": 2.5, "c": 3.5},
    ).to_dict()
    payload.update(overrides)
    return payload


def _shadow_payload(**overrides: Any) -> dict[str, Any]:
    payload = ShadowGate().decide(**_gate_inputs()).to_dict()
    payload.update(overrides)
    return payload


def _ledger_payload(**overrides: Any) -> dict[str, Any]:
    payload = _disagreement_payload()
    payload.update(overrides)
    return payload


def _report_payload() -> dict[str, Any]:
    bundle = synthetic_bundle(attempt_count=8, seed=61)
    plan = FaultPlan(drop=((SourceName.CLIENT_LATENCY, bundle.attempt_traces[3].key),))
    return reconcile(apply_faults(bundle, plan).bundle).to_dict()


def _row_payload(**overrides: Any) -> dict[str, Any]:
    payload = ReconciledRow(
        "synthetic:0", 0, "node-0", 1024, 512, 64, 12.5, None, 0.0, 1.0, 12.5
    ).to_dict()
    payload.update(overrides)
    return payload


def test_every_uncorrupted_helper_payload_is_accepted() -> None:
    """Without this, a broken helper would make every test below vacuous."""
    assert RankingComparison.from_dict(_comparison_payload()).tau_b == 1.0
    assert ShadowDecision.from_dict(_shadow_payload()).reasons == ()
    assert LedgerEntry.from_dict(_ledger_payload()).kind is LedgerKind.DISAGREEMENT
    report = ReconciliationReport.from_dict(_report_payload())
    assert report.reconciled_count == 7
    assert len(report.ledger) == 1
    assert ReconciledRow.from_dict(_row_payload()).hit_tokens == 512


# -- a comparison must follow from the scores it carries --------------------


def test_an_edited_tau_b_is_refused_by_the_scores_it_claims_to_summarize() -> None:
    # The scores are the evidence. A tau_b they do not produce is the exact
    # claim this package exists not to publish.
    with pytest.raises(MalformedPayloadError, match="but the stored scores give"):
        RankingComparison.from_dict(_comparison_payload(tau_b=0.2))


def test_an_edited_pairwise_agreement_is_refused_the_same_way() -> None:
    with pytest.raises(MalformedPayloadError, match="but the stored scores give"):
        RankingComparison.from_dict(_comparison_payload(pairwise_agreement=0.5))


@pytest.mark.parametrize(
    "overrides",
    [
        {"concordant_pairs": 2},
        {"discordant_pairs": 1},
    ],
)
def test_edited_pair_counts_are_refused(overrides: dict[str, Any]) -> None:
    with pytest.raises(MalformedPayloadError, match="but the stored scores have"):
        RankingComparison.from_dict(_comparison_payload(**overrides))


def test_a_comparison_cannot_claim_a_statistic_that_was_not_frozen() -> None:
    with pytest.raises(MalformedPayloadError, match="must be the frozen"):
        RankingComparison.from_dict(_comparison_payload(statistic="SPEARMAN_RHO"))


def test_a_comparison_cannot_claim_the_wrong_schema() -> None:
    with pytest.raises(MalformedPayloadError, match="schema_version must be"):
        RankingComparison.from_dict(
            _comparison_payload(schema_version="m10-ranking-v9")
        )


def test_repeated_arm_ids_are_refused_because_they_forge_a_pair() -> None:
    # ("a", "a", "c") is still sorted, so only the distinctness check catches it.
    with pytest.raises(MalformedPayloadError, match="must be distinct"):
        RankingComparison.from_dict(_comparison_payload(arm_ids=["a", "a", "c"]))


def test_unsorted_arm_ids_are_refused_because_two_runs_must_agree_on_order() -> None:
    with pytest.raises(MalformedPayloadError, match="must be in sorted order"):
        RankingComparison.from_dict(
            _comparison_payload(
                arm_ids=["c", "b", "a"],
                left_scores=[3.0, 2.0, 1.0],
                right_scores=[3.5, 2.5, 1.5],
            )
        )


def test_a_comparison_of_fewer_than_two_arms_is_refused() -> None:
    with pytest.raises(MalformedPayloadError, match="at least two arms"):
        RankingComparison.from_dict(
            _comparison_payload(
                arm_ids=["a"], left_scores=[1.0], right_scores=[1.5], tau_b=1.0
            )
        )


def test_a_side_that_does_not_score_every_arm_is_refused() -> None:
    with pytest.raises(MalformedPayloadError, match="each side must score every arm"):
        RankingComparison.from_dict(_comparison_payload(left_scores=[1.0, 2.0]))


def test_a_tau_b_claimed_over_fully_tied_scores_is_refused() -> None:
    # An undefined statistic must not be smuggled in as a stored number.
    with pytest.raises(MalformedPayloadError, match="admit no ranking statistic"):
        RankingComparison.from_dict(
            _comparison_payload(
                left_scores=[1.0, 1.0, 1.0],
                right_scores=[1.5, 2.5, 3.5],
                tau_b=1.0,
                concordant_pairs=0,
                discordant_pairs=0,
                pairwise_agreement=1.0,
            )
        )


# -- a decision must agree with the measurements it records -----------------


def test_a_decision_claiming_enforcement_cannot_be_rebuilt() -> None:
    with pytest.raises(ShadowEnforcementError, match="never enforced"):
        ShadowDecision.from_dict(_shadow_payload(enforced=True))


def test_a_reason_the_gate_cannot_emit_is_refused() -> None:
    with pytest.raises(MalformedPayloadError, match="contains unknown member"):
        ShadowDecision.from_dict(
            _shadow_payload(outcome="SHADOW_WITHHELD", reasons=["LOOKED_WRONG_TO_ME"])
        )


def test_a_repeated_reason_is_refused() -> None:
    with pytest.raises(MalformedPayloadError, match="repeats a member"):
        ShadowDecision.from_dict(
            _shadow_payload(
                outcome="SHADOW_WITHHELD",
                reasons=[CANDIDATE_DOES_NOT_BEAT_BASELINE] * 2,
            )
        )


def test_reasons_out_of_canonical_order_are_refused() -> None:
    # Order is part of the record: two runs finding the same faults must
    # serialize to the same bytes.
    with pytest.raises(MalformedPayloadError, match="must be in canonical order"):
        ShadowDecision.from_dict(
            _shadow_payload(
                outcome="SHADOW_WITHHELD",
                reasons=[
                    CANDIDATE_DOES_NOT_BEAT_BASELINE,
                    RANKING_CONSISTENCY_UNAVAILABLE,
                ],
                tau_b=None,
            )
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"outcome": "SHADOW_WITHHELD", "reasons": []},
        {
            "outcome": "SHADOW_RECOMMENDED",
            "reasons": ["RECONCILED_FRACTION_BELOW_GATE"],
        },
    ],
)
def test_an_outcome_that_contradicts_its_reasons_is_refused(
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(MalformedPayloadError, match="does not match"):
        ShadowDecision.from_dict(_shadow_payload(**overrides))


def test_a_decision_cannot_claim_the_candidate_lost_while_it_won() -> None:
    # This reason is re-derivable from the stored scores, so it is re-derived.
    with pytest.raises(MalformedPayloadError, match="is stated for exactly the"):
        ShadowDecision.from_dict(
            _shadow_payload(
                outcome="SHADOW_WITHHELD",
                reasons=[CANDIDATE_DOES_NOT_BEAT_BASELINE],
            )
        )


def test_a_decision_cannot_claim_a_missing_tau_b_while_carrying_one() -> None:
    with pytest.raises(MalformedPayloadError, match="is stated for exactly the"):
        ShadowDecision.from_dict(
            _shadow_payload(
                outcome="SHADOW_WITHHELD",
                reasons=[RANKING_CONSISTENCY_UNAVAILABLE],
            )
        )


def test_a_decision_cannot_claim_a_low_tau_b_it_never_recorded() -> None:
    # RANKING_CONSISTENCY_UNAVAILABLE has to be listed too, or that check fires
    # first; what is under test is the one that follows it.
    with pytest.raises(MalformedPayloadError, match="requires a tau_b"):
        ShadowDecision.from_dict(
            _shadow_payload(
                outcome="SHADOW_WITHHELD",
                reasons=[
                    RANKING_CONSISTENCY_UNAVAILABLE,
                    "RANKING_CONSISTENCY_BELOW_GATE",
                ],
                tau_b=None,
            )
        )


def test_a_decision_cannot_compare_an_arm_with_itself() -> None:
    with pytest.raises(MalformedPayloadError, match="cannot be compared with itself"):
        ShadowDecision.from_dict(_shadow_payload(candidate_arm_id="S0_RANDOM"))


def test_a_decision_cannot_claim_the_wrong_schema() -> None:
    with pytest.raises(MalformedPayloadError, match="schema_version must be"):
        ShadowDecision.from_dict(_shadow_payload(schema_version="m10-shadow-v9"))
    assert SHADOW_SCHEMA_VERSION == "m10-shadow-v1"


# -- a decision must follow from the policy it states -----------------------
#
# The decision carries the gate that judged it, and every reason and the
# outcome are recomputed from that gate on construction. Before the policy was
# stored, none of the three threshold-dependent reasons could be re-derived at
# all, and the payload in the first test below -- measurements the default gate
# rejects on every count, carrying an empty reason list and a recommendation --
# was accepted without complaint.
# --------------------------------------------------------------------------


#: Measurements that pass the default gate with almost nothing to spare, so
#: that a threshold moved by a thousandth changes the verdict.
_NARROWLY_CLEARED = {
    "reconciled_fraction": 0.995,
    "disagreement_fraction": 0.005,
    "tau_b": 0.9,
}


def _forged_recommendation() -> dict[str, Any]:
    """The payload that used to be accepted: a recommendation over refuted data."""
    return _shadow_payload(
        outcome="SHADOW_RECOMMENDED",
        reasons=[],
        reconciled_fraction=0.0,
        disagreement_fraction=1.0,
        tau_b=-1.0,
        baseline_score=0.44,
        candidate_score=0.54,
    )


def test_the_forged_recommendation_is_refuted_by_its_own_policy() -> None:
    payload = _forged_recommendation()
    assert payload["policy"] == DEFAULT_SHADOW_GATE.to_dict()

    with pytest.raises(MalformedPayloadError, match="are not what this decision"):
        ShadowDecision.from_dict(payload)


def test_the_forged_measurements_really_do_fail_every_threshold() -> None:
    """Otherwise the test above could pass for some unrelated reason."""
    payload = _forged_recommendation()
    decision = ShadowGate().decide(
        **_gate_inputs(
            reconciled_fraction=payload["reconciled_fraction"],
            disagreement_fraction=payload["disagreement_fraction"],
            tau_b=payload["tau_b"],
        )
    )
    assert decision.outcome is ShadowOutcome.SHADOW_WITHHELD
    assert decision.reasons == (
        RECONCILED_FRACTION_BELOW_GATE,
        DISAGREEMENT_FRACTION_ABOVE_GATE,
        RANKING_CONSISTENCY_BELOW_GATE,
    )


def test_a_forger_who_relaxes_the_thresholds_has_to_say_so() -> None:
    """Restating the policy is the only way to make the forgery consistent.

    Which is the point: the permissive gate is now written down in the
    artifact, next to the verdict it produced, where the runner refuses it and
    a reader can see it.
    """
    permissive = ShadowGate(
        minimum_reconciled_fraction=0.0,
        maximum_disagreement_fraction=1.0,
        minimum_tau_b=-1.0,
    )
    payload = _forged_recommendation() | {"policy": permissive.to_dict()}

    restored = ShadowDecision.from_dict(payload)
    assert restored.outcome is ShadowOutcome.SHADOW_RECOMMENDED
    assert restored.policy == permissive
    assert restored.policy != DEFAULT_SHADOW_GATE


@pytest.mark.parametrize(
    ("threshold", "tightened"),
    [
        ("minimum_reconciled_fraction", 0.999),
        ("maximum_disagreement_fraction", 0.001),
        ("minimum_tau_b", 0.95),
    ],
)
def test_a_tampered_threshold_no_longer_produces_the_recorded_verdict(
    threshold: str, tightened: float
) -> None:
    # Measurements that clear the real gate but only just, so that tightening
    # any single threshold past one of them makes the recorded recommendation
    # underivable from the policy the record itself states.
    recommended = ShadowGate().decide(**_gate_inputs(**_NARROWLY_CLEARED)).to_dict()
    assert recommended["outcome"] == "SHADOW_RECOMMENDED"

    moved: dict[str, Any] = {threshold: tightened}
    tampered = replace(DEFAULT_SHADOW_GATE, **moved)
    with pytest.raises(MalformedPayloadError, match="are not what this decision"):
        ShadowDecision.from_dict({**recommended, "policy": tampered.to_dict()})


def test_a_decision_cannot_state_a_policy_of_an_unknown_version() -> None:
    with pytest.raises(MalformedPayloadError, match="policy_version must be"):
        ShadowDecision.from_dict(
            _shadow_payload(
                policy=DEFAULT_SHADOW_GATE.to_dict() | {"policy_version": "v9"}
            )
        )
    assert SHADOW_GATE_SCHEMA_VERSION == "m10-shadow-gate-v1"


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("minimum_reconciled_fraction", "0.99"),
        ("maximum_disagreement_fraction", None),
        ("minimum_tau_b", True),
        ("minimum_tau_b", math.nan),
        ("policy_version", 1),
    ],
)
def test_a_policy_field_of_the_wrong_type_or_range_is_refused(
    field: str, bad: Any
) -> None:
    with pytest.raises(MalformedPayloadError, match="ShadowGate"):
        ShadowDecision.from_dict(
            _shadow_payload(policy=DEFAULT_SHADOW_GATE.to_dict() | {field: bad})
        )


@pytest.mark.parametrize("missing", ["policy_version", "minimum_tau_b"])
def test_a_policy_missing_a_threshold_is_refused(missing: str) -> None:
    partial = {
        key: value
        for key, value in DEFAULT_SHADOW_GATE.to_dict().items()
        if key != missing
    }
    with pytest.raises(MalformedPayloadError, match="ShadowGate"):
        ShadowDecision.from_dict(_shadow_payload(policy=partial))


@pytest.mark.parametrize("bad", [None, "m10-shadow-gate-v1", 0.99, []])
def test_a_decision_with_no_policy_object_at_all_is_refused(bad: Any) -> None:
    with pytest.raises(MalformedPayloadError):
        ShadowDecision.from_dict(_shadow_payload(policy=bad))


def test_a_gate_that_is_not_a_gate_cannot_judge_a_decision() -> None:
    """A gate-like object can answer the re-derivation differently than it
    answered ``decide``, which is exactly the self-justifying record the exact
    type check exists to prevent. This one reports a threshold it does not
    store, so the limit it publishes is not the limit anything was judged
    against.
    """

    class LenientGate(ShadowGate):
        def __getattribute__(self, name: str) -> Any:
            if name == "minimum_tau_b":
                return -1.0
            return super().__getattribute__(name)

    assert LenientGate().minimum_tau_b != ShadowGate().minimum_tau_b
    with pytest.raises(MalformedPayloadError, match="policy must be a ShadowGate"):
        ShadowDecision(
            SHADOW_SCHEMA_VERSION,
            LenientGate(),
            "S0_RANDOM",
            "S3_GB_PREFIX_BUCKET",
            ShadowOutcome.SHADOW_RECOMMENDED,
            False,
            (),
            1.0,
            0.0,
            1.0,
            0.44,
            0.54,
        )


def test_the_gate_records_the_policy_it_applied_rather_than_a_copy() -> None:
    gate = ShadowGate(minimum_tau_b=0.5)
    decision = gate.decide(**_gate_inputs())
    assert decision.policy is gate


def test_a_withheld_decision_cannot_be_relabelled_a_recommendation() -> None:
    # Same policy, same measurements, verdict flipped by hand.
    withheld = ShadowGate().decide(**_gate_inputs(candidate_score=0.1)).to_dict()
    assert withheld["outcome"] == "SHADOW_WITHHELD"
    with pytest.raises(MalformedPayloadError, match="does not match"):
        ShadowDecision.from_dict({**withheld, "outcome": "SHADOW_RECOMMENDED"})


def test_emptying_the_reasons_does_not_launder_a_withheld_decision() -> None:
    # Withheld on a threshold alone, so that erasing the reason leaves a record
    # that is internally consistent and can only be refuted by re-deriving it
    # from the stated policy.
    withheld = ShadowGate().decide(**_gate_inputs(reconciled_fraction=0.5)).to_dict()
    assert withheld["reasons"] == [RECONCILED_FRACTION_BELOW_GATE]
    with pytest.raises(MalformedPayloadError, match="are not what this decision"):
        ShadowDecision.from_dict(
            {**withheld, "outcome": "SHADOW_RECOMMENDED", "reasons": []}
        )


def test_every_published_decision_states_the_default_gate() -> None:
    outcome = run_replay(
        ReplayPlan(capacity_blocks=64),
        tiny_requests(),
        trace_sha256="ab" * 32,
        git_sha=None,
        git_dirty=None,
    )
    decisions = outcome.shadow_decisions("token_weighted_hit_rate")
    assert decisions
    assert all(decision.policy == DEFAULT_SHADOW_GATE for decision in decisions)


# -- a ledger entry must be the kind of defect it says it is ----------------


def test_a_disagreement_about_an_unshared_field_is_refused() -> None:
    # Only one source ever observes ttft_work, so no two of them can differ.
    with pytest.raises(MalformedPayloadError, match="is not a shared field"):
        LedgerEntry.from_dict(_ledger_payload(field_name="ttft_work"))


def test_a_disagreement_must_name_exactly_the_fields_observers() -> None:
    # node_id is observed by ENGINE_HIT and ATTEMPT_TRACE only. Naming the
    # client as a third observer would blame a source that never looked.
    with pytest.raises(MalformedPayloadError, match="is between its observers"):
        LedgerEntry.from_dict(_ledger_payload(field_name="node_id"))


def test_a_disagreement_whose_values_all_agree_is_refused() -> None:
    with pytest.raises(MalformedPayloadError, match="every source reported"):
        LedgerEntry.from_dict(_ledger_payload(values=["1024", "1024", "1024"]))


def test_a_missing_entry_must_record_absence_and_nothing_else() -> None:
    with pytest.raises(MalformedPayloadError, match="for each absent"):
        LedgerEntry.from_dict(
            _ledger_payload(
                kind="MISSING",
                field_name="",
                sources=["CLIENT_LATENCY"],
                values=["1024"],
            )
        )


def test_a_whole_record_defect_cannot_name_a_field() -> None:
    with pytest.raises(MalformedPayloadError, match="charged against the record"):
        LedgerEntry.from_dict(
            _ledger_payload(
                kind="DUPLICATE",
                field_name="input_tokens",
                sources=["ENGINE_HIT"],
                values=["count=2"],
            )
        )


def test_an_entry_with_more_values_than_sources_is_refused() -> None:
    with pytest.raises(MalformedPayloadError, match="each named source reports one"):
        LedgerEntry.from_dict(_ledger_payload(sources=["ENGINE_HIT", "CLIENT_LATENCY"]))


def test_entry_sources_out_of_canonical_order_are_refused() -> None:
    with pytest.raises(MalformedPayloadError, match="must be in canonical order"):
        LedgerEntry.from_dict(
            _ledger_payload(sources=["CLIENT_LATENCY", "ENGINE_HIT", "ATTEMPT_TRACE"])
        )


def test_an_entry_that_names_no_source_is_refused() -> None:
    with pytest.raises(MalformedPayloadError, match="at least one source"):
        LedgerEntry.from_dict(_ledger_payload(sources=[], values=[]))


def test_an_entry_with_a_negative_attempt_index_is_refused() -> None:
    with pytest.raises(MalformedPayloadError, match="must not be negative"):
        LedgerEntry.from_dict(_ledger_payload(attempt_index=-1))


# -- a report must account for every attempt exactly once -------------------


def test_a_report_charging_two_kinds_against_one_attempt_is_refused() -> None:
    payload = _report_payload()
    charged = payload["ledger"][0]
    payload["ledger"].append(
        {
            "kind": "DUPLICATE",
            "logical_request_id": charged["logical_request_id"],
            "attempt_index": charged["attempt_index"],
            "field_name": "",
            "sources": ["ENGINE_HIT"],
            "values": ["count=2"],
        }
    )
    with pytest.raises(MalformedPayloadError, match="only the most severe"):
        ReconciliationReport.from_dict(payload)


def test_a_report_cannot_both_reconcile_and_charge_the_same_attempt() -> None:
    payload = _report_payload()
    settled = payload["reconciled"][0]
    payload["ledger"].append(
        {
            "kind": "DUPLICATE",
            "logical_request_id": settled["logical_request_id"],
            "attempt_index": settled["attempt_index"],
            "field_name": "",
            "sources": ["ENGINE_HIT"],
            "values": ["count=2"],
        }
    )
    with pytest.raises(MalformedPayloadError, match="both reconciled"):
        ReconciliationReport.from_dict(payload)


def test_an_attempt_count_that_does_not_match_the_tables_is_refused() -> None:
    # An undercounted denominator inflates reconciled_fraction, which is the
    # number the shadow gate reads.
    payload = _report_payload()
    payload["attempt_count"] -= 1
    with pytest.raises(MalformedPayloadError, match="accounted for but attempt_count"):
        ReconciliationReport.from_dict(payload)


def test_an_attempt_reconciled_twice_is_refused() -> None:
    payload = _report_payload()
    payload["reconciled"].append(payload["reconciled"][0])
    with pytest.raises(MalformedPayloadError, match="an attempt is reconciled once"):
        ReconciliationReport.from_dict(payload)


def test_a_ledger_out_of_sort_order_is_refused() -> None:
    bundle = synthetic_bundle(attempt_count=8, seed=67)
    plan = FaultPlan(
        drop=(
            (SourceName.CLIENT_LATENCY, bundle.attempt_traces[1].key),
            (SourceName.CLIENT_LATENCY, bundle.attempt_traces[5].key),
        )
    )
    payload = reconcile(apply_faults(bundle, plan).bundle).to_dict()
    assert len(payload["ledger"]) == 2
    payload["ledger"].reverse()
    with pytest.raises(MalformedPayloadError, match="sort_key order"):
        ReconciliationReport.from_dict(payload)


def test_a_report_cannot_claim_the_wrong_schema() -> None:
    with pytest.raises(MalformedPayloadError, match="schema_version must be"):
        ReconciliationReport.from_dict(
            {**_report_payload(), "schema_version": "m10-reconcile-v9"}
        )


# -- a row must be readable, and no more than that --------------------------


def test_a_row_with_a_negative_count_is_refused() -> None:
    with pytest.raises(MalformedPayloadError, match="must not be negative"):
        ReconciledRow.from_dict(_row_payload(hit_tokens=-1))


def test_a_row_without_an_identity_is_refused() -> None:
    with pytest.raises(MalformedPayloadError, match="node_id must not be empty"):
        ReconciledRow.from_dict(_row_payload(node_id=""))


def test_a_row_with_a_non_finite_duration_is_refused() -> None:
    text = json.dumps(_row_payload()).replace('"ttft_work": 12.5', '"ttft_work": NaN')
    with pytest.raises(MalformedPayloadError, match="must be a finite number"):
        ReconciledRow.from_dict(json.loads(text))


def test_a_row_that_contradicts_itself_is_reported_not_refused() -> None:
    """The scope boundary, pinned so that widening it has to be deliberate.

    A row whose finish precedes its arrival, or whose hit exceeds its prompt,
    is a real finding about the producing system. Refusing to hold it would
    destroy the evidence rather than report it, so the record accepts it and
    round-trips it unchanged.
    """
    impossible = _row_payload(
        arrival_work=99.0, start_work=99.0, finish_work=0.0, hit_tokens=4096
    )
    row = ReconciledRow.from_dict(impossible)
    assert row.finish_work < row.arrival_work
    assert row.hit_tokens > row.input_tokens
    assert ReconciledRow.from_dict(json.loads(json.dumps(row.to_dict()))) == row


# --------------------------------------------------------------------------
# source fingerprints
# --------------------------------------------------------------------------


def _plant_m10_tree(root: Path) -> None:
    """Plant a minimal tree with the shape the fingerprinter walks."""
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts/run_m10_synthetic.py").write_text("# gen\n", encoding="utf-8")
    package = root / "src/prefill_cache_sim/replay"
    package.mkdir(parents=True, exist_ok=True)
    (root / "src/prefill_cache_sim/__init__.py").write_text("", encoding="utf-8")
    (root / "src/prefill_cache_sim/domain.py").write_text("# d\n", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "orchestrator.py").write_text("# o\n", encoding="utf-8")


def test_source_paths_cover_the_generator_and_every_package_module() -> None:
    paths = set(m10_source_paths(REPO_ROOT))
    assert "scripts/run_m10_synthetic.py" in paths
    # Discovered independently here: a module added tomorrow must not be able to
    # silently escape the manifest, which is what a hand-maintained list allows.
    discovered = {
        item.relative_to(REPO_ROOT).as_posix()
        for item in (REPO_ROOT / "src/prefill_cache_sim").rglob("*.py")
        if "__pycache__" not in item.parts
    }
    assert discovered
    assert discovered <= paths


def test_source_paths_skip_derived_bytecode(tmp_path: Path) -> None:
    _plant_m10_tree(tmp_path)
    cache = tmp_path / "src/prefill_cache_sim/replay/__pycache__"
    cache.mkdir()
    (cache / "orchestrator.py").write_text("# derived\n", encoding="utf-8")
    assert not any("__pycache__" in path for path in m10_source_paths(tmp_path))


def test_source_paths_are_deterministic_and_sorted(tmp_path: Path) -> None:
    _plant_m10_tree(tmp_path)
    paths = m10_source_paths(tmp_path)
    assert paths == m10_source_paths(tmp_path)
    assert list(paths[1:]) == sorted(paths[1:])


def test_missing_package_or_source_is_refused_rather_than_skipped(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="simulator package not found"):
        m10_source_paths(tmp_path)
    _plant_m10_tree(tmp_path)
    (tmp_path / "scripts/run_m10_synthetic.py").unlink()
    # A source that vanished must fail the run, not quietly drop out of the map.
    with pytest.raises(FileNotFoundError, match="run_m10_synthetic.py"):
        source_fingerprints(tmp_path)


def test_editing_a_source_changes_its_digest_and_the_combined_one(
    tmp_path: Path,
) -> None:
    _plant_m10_tree(tmp_path)
    relative = "src/prefill_cache_sim/replay/orchestrator.py"
    before = source_manifest(tmp_path)
    (tmp_path / relative).write_text("# edited\n", encoding="utf-8")
    after = source_manifest(tmp_path)
    assert after["files"][relative] != before["files"][relative]
    assert after["combined_digest"] != before["combined_digest"]
    assert set(after["files"]) == set(before["files"])


def test_combined_digest_changes_when_the_file_set_changes(tmp_path: Path) -> None:
    _plant_m10_tree(tmp_path)
    covered = source_manifest(tmp_path)["files"]
    dropped = {
        path: digest
        for path, digest in covered.items()
        if path != "src/prefill_cache_sim/domain.py"
    }
    # Dropping a file must not be able to masquerade as the same code: paths are
    # hashed alongside their digests.
    assert combined_digest(dropped) != combined_digest(covered)


def test_manifest_states_what_the_digests_do_not_promise(tmp_path: Path) -> None:
    _plant_m10_tree(tmp_path)
    manifest = source_manifest(tmp_path)
    assert manifest["algorithm"] == "sha256"
    assert manifest["reproducibility_claim"] == REPRODUCIBILITY_CLAIM
    assert manifest["file_count"] == len(manifest["files"])
    assert set(manifest["runtime"]) == {"python_implementation", "python_version"}
    # The claim travels with the numbers rather than living in a review comment.
    assert "conditional" in manifest["note"]


# --------------------------------------------------------------------------
# staged, manifest-indexed artifact writes
# --------------------------------------------------------------------------


def _load_generator() -> ModuleType:
    path = REPO_ROOT / "scripts/run_m10_synthetic.py"
    spec = importlib.util.spec_from_file_location("m10_generator_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_generator()


def _sample_artifacts() -> dict[str, bytes]:
    artifacts = {
        "replay.json": GENERATOR._json_bytes({"schema_version": "x"}),
        "results.csv": GENERATOR._csv_bytes([{"a": 1, "b": 2}], ["a", "b"]),
        # An empty ledger is the *expected* M10 outcome, so it has to survive the
        # write path with its header intact rather than crash for want of a row.
        "ledger.csv": GENERATOR._csv_bytes([], ["kind", "field_name"]),
    }
    artifacts[GENERATOR.MANIFEST_NAME] = GENERATOR._manifest_bytes(artifacts)
    return artifacts


def test_json_bytes_refuses_to_emit_nan_or_infinity() -> None:
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="Out of range float"):
            GENERATOR._json_bytes({"tau_b": bad})


def test_json_and_csv_bytes_are_deterministic() -> None:
    payload: dict[str, Any] = {"b": 2, "a": [1, 2, 3]}
    assert GENERATOR._json_bytes(payload) == GENERATOR._json_bytes(dict(payload))
    rows = [{"x": 1, "y": "two"}, {"x": 3, "y": "four"}]
    fields = ["x", "y"]
    assert GENERATOR._csv_bytes(rows, fields) == GENERATOR._csv_bytes(
        list(rows), list(fields)
    )


def test_csv_bytes_match_what_a_file_handle_write_would_produce(
    tmp_path: Path,
) -> None:
    import csv

    rows = [{"x": 1, "y": "two"}, {"x": 3, "y": "four"}]
    reference = tmp_path / "reference.csv"
    with reference.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["x", "y"])
        writer.writeheader()
        writer.writerows(rows)
    assert GENERATOR._csv_bytes(rows, ["x", "y"]) == reference.read_bytes()


def test_csv_bytes_emit_a_header_for_zero_rows() -> None:
    # M10's ledger is legitimately empty, so field names are declared rather than
    # read off a first row that does not exist.
    assert GENERATOR._csv_bytes([], ["kind", "field_name"]) == b"kind,field_name\r\n"


def test_forbidden_labels_are_caught_before_anything_is_written() -> None:
    for label in GENERATOR.FORBIDDEN_LABELS:
        payload = GENERATOR._json_bytes({"calibration_status": label})
        with pytest.raises(RuntimeError, match="nothing here earned it"):
            GENERATOR._assert_no_stronger_claim("replay.json", payload)


def test_the_labels_this_run_does_use_are_not_forbidden() -> None:
    for name, blob in _sample_artifacts().items():
        GENERATOR._assert_no_stronger_claim(name, blob)
    honest = GENERATOR._json_bytes(
        {
            "calibration_status": CalibrationStatus.SYNTHETIC_UNCALIBRATED.value,
            "time_unit": TimeUnit.NORMALIZED_WORK.value,
            "evidence_tier": EvidenceTier.SYNTHETIC_REPLAY.value,
        }
    )
    GENERATOR._assert_no_stronger_claim("replay.json", honest)


def test_manifest_digests_every_other_artifact() -> None:
    artifacts = _sample_artifacts()
    manifest = json.loads(artifacts[GENERATOR.MANIFEST_NAME])
    assert manifest["schema_version"] == GENERATOR.MANIFEST_SCHEMA_VERSION
    assert set(manifest["files"]) == {"replay.json", "results.csv", "ledger.csv"}
    for name, digest in manifest["files"].items():
        assert digest == hashlib.sha256(artifacts[name]).hexdigest()


def test_write_artifacts_requires_a_manifest_in_the_set() -> None:
    with pytest.raises(RuntimeError, match="must include MANIFEST.json"):
        GENERATOR._write_artifacts(Path("/nonexistent"), {"replay.json": b"{}"})


def test_write_artifacts_creates_the_directory_and_writes_every_artifact(
    tmp_path: Path,
) -> None:
    output = tmp_path / "results" / "m10-synthetic"
    artifacts = _sample_artifacts()
    GENERATOR._write_artifacts(output, artifacts)
    for name, payload in artifacts.items():
        assert (output / name).read_bytes() == payload


def test_write_artifacts_never_deletes_files_it_did_not_generate(
    tmp_path: Path,
) -> None:
    output = tmp_path / "results" / "m10-synthetic"
    output.mkdir(parents=True)
    # PROVENANCE.md is hand-written and lives in the output directory, so a
    # whole-directory swap would destroy it.
    provenance = output / "PROVENANCE.md"
    provenance.write_text("hand written\n", encoding="utf-8")
    stale = output / "results.csv"
    stale.write_text("stale\n", encoding="utf-8")

    GENERATOR._write_artifacts(output, _sample_artifacts())

    assert provenance.read_text(encoding="utf-8") == "hand written\n"
    assert stale.read_bytes() == _sample_artifacts()["results.csv"]


def test_write_artifacts_leaves_no_staging_directory_behind(tmp_path: Path) -> None:
    output = tmp_path / "results" / "m10-synthetic"
    GENERATOR._write_artifacts(output, _sample_artifacts())
    assert list(output.parent.glob(".m10-staging-*")) == []
    assert sorted(item.name for item in output.iterdir()) == [
        "MANIFEST.json",
        "ledger.csv",
        "replay.json",
        "results.csv",
    ]


def test_write_artifacts_is_byte_identical_across_runs(tmp_path: Path) -> None:
    first = tmp_path / "a" / "m10-synthetic"
    second = tmp_path / "b" / "m10-synthetic"
    GENERATOR._write_artifacts(first, _sample_artifacts())
    GENERATOR._write_artifacts(second, _sample_artifacts())
    for name in _sample_artifacts():
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_round_trip_guard_accepts_a_genuine_replay() -> None:
    outcome = run_replay(
        ReplayPlan(capacity_blocks=64),
        tiny_requests(),
        trace_sha256="ab" * 32,
        git_sha=None,
        git_dirty=None,
    )
    rankings = {
        metric: outcome.ranking_comparison(metric) for metric in GENERATOR.SCORE_METRICS
    }
    shadow = {
        metric: outcome.shadow_decisions(metric) for metric in GENERATOR.SCORE_METRICS
    }
    # The guard is wired to real published records, not to a hand-built sample.
    GENERATOR._assert_round_trips(outcome, rankings, shadow)


def test_published_artifact_manifest_matches_the_committed_files() -> None:
    output = REPO_ROOT / "results/m10-synthetic"
    manifest_path = output / GENERATOR.MANIFEST_NAME
    if not manifest_path.is_file():
        pytest.skip("results/m10-synthetic has not been regenerated yet")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        actual = hashlib.sha256((output / name).read_bytes()).hexdigest()
        assert actual == digest, f"{name} does not match the manifest digest"


def test_published_replay_records_source_fingerprints() -> None:
    replay_path = REPO_ROOT / "results/m10-synthetic/replay.json"
    if not replay_path.is_file():
        pytest.skip("results/m10-synthetic has not been generated yet")
    payload = json.loads(replay_path.read_text(encoding="utf-8"))
    manifest = payload["provenance"]["source_fingerprints"]
    assert manifest["reproducibility_claim"] == REPRODUCIBILITY_CLAIM
    assert manifest["combined_digest"] == combined_digest(manifest["files"])
    # Historical artifacts are bound to the source tree recorded at generation
    # time, not to today's HEAD.  Verify every named file against that immutable
    # git object so later M12 work does not make valid M10 evidence fail closed.
    git_sha = payload["provenance"]["git_sha"]
    assert re.fullmatch(r"[0-9a-f]{40}", git_sha), "invalid historical git SHA"
    committed = {
        name: hashlib.sha256(
            subprocess.run(
                ["git", "show", f"{git_sha}:{name}"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
            ).stdout
        ).hexdigest()
        for name in manifest["files"]
    }
    assert manifest["files"] == committed
    assert manifest["combined_digest"] == combined_digest(committed)


def test_published_replay_never_claims_hardware_it_did_not_touch() -> None:
    replay_path = REPO_ROOT / "results/m10-synthetic/replay.json"
    if not replay_path.is_file():
        pytest.skip("results/m10-synthetic has not been generated yet")
    payload = json.loads(replay_path.read_text(encoding="utf-8"))
    outcome = payload["outcome"]
    assert outcome["calibration_status"] == CalibrationStatus.SYNTHETIC_UNCALIBRATED
    assert outcome["time_unit"] == TimeUnit.NORMALIZED_WORK
    assert outcome["evidence_tier"] == EvidenceTier.SYNTHETIC_REPLAY
    assert payload["provenance"]["hardware_validation"] == "BLOCKED_NO_ENGINE_ACCESS"
    assert all(value is None for value in outcome["machine"].values())
