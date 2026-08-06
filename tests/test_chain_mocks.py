"""Invariants for the M11 chain mocks.

Scope note: these tests assert cross-component protocol and state-machine
behaviour on constructed fixtures. They do not measure latency or throughput and
they are not a production end-to-end validation.
"""

from __future__ import annotations

import pytest

from prefill_cache_sim.chain import (
    CHAIN_TRUTH_BASIS,
    D2_GATE_CLOSED,
    FAIL_OPEN_COUNTERS,
    SCENARIO_ORDER,
    SCENARIOS,
    Capabilities,
    Capability,
    ChainConfig,
    ChainEventKind,
    ChainHarness,
    ChainProtocolError,
    EnforcementMode,
    FailOpenReason,
    LegRoute,
    RequiredModeFailure,
    RequiredModeRejected,
    SelectionOutcome,
    SelectorOwner,
    ViewSnapshot,
    baseline_host,
    enforce_config,
    fresh_view,
    negotiate,
    run_all,
    selector_host,
    stale_view,
)
from prefill_cache_sim.chain.scenarios import (
    BASELINE_PICK,
    INPUT_TOKENS,
    OUTPUT_TOKENS,
    SELECTOR_PICK,
)
from prefill_cache_sim.preemption import (
    DecodeAction,
    DecodeCheckpoint,
    KeepMoveInput,
    PreemptionConfig,
)
from prefill_cache_sim.replay.sources import TruthBasis


def move_economics() -> KeepMoveInput:
    """Economics that unambiguously favour moving, so a keep is a real decision."""
    return KeepMoveInput(
        completion_value=100.0,
        keep_probability=0.1,
        move_probability=0.95,
        remaining_gpu_work=40.0,
        interference_gpu_work=20.0,
        checkpoint_gpu_work=1.0,
        recovery_gpu_work=1.0,
        migration_stall_work=1.0,
        duplicate_risk=0.0,
        wait_work=0.0,
        tenant_weight=1.0,
    )


def full_handshake(harness: ChainHarness) -> None:
    """Negotiate every capability, the precondition for cache-aware scoring."""
    harness.handshake(Capabilities(), Capabilities())


# -- provenance ----------------------------------------------------------


def test_chain_output_is_labelled_synthetic() -> None:
    assert CHAIN_TRUTH_BASIS is TruthBasis.SYNTHETIC_FIXTURE


def test_exactly_the_seven_named_scenarios_exist() -> None:
    assert SCENARIO_ORDER == (
        "timeout",
        "stale_view",
        "p_rollback",
        "d_reject",
        "lease_expiry",
        "late_frame",
        "client_crash",
    )
    assert set(SCENARIOS) == set(SCENARIO_ORDER)


# -- invariants that hold across every scenario --------------------------


@pytest.mark.parametrize("name", SCENARIO_ORDER)
def test_every_scenario_has_exactly_one_visible_terminal(name: str) -> None:
    transcript = SCENARIOS[name]()
    assert transcript.terminals == ("STOP",)
    assert transcript.count("terminal_total") == 1


@pytest.mark.parametrize("name", SCENARIO_ORDER)
def test_every_scenario_delivers_the_full_output_exactly_once(name: str) -> None:
    transcript = SCENARIOS[name]()
    assert transcript.delivered_tokens == OUTPUT_TOKENS
    seqs = [seq for _, seq in transcript.user_frames]
    assert seqs == list(range(OUTPUT_TOKENS))


@pytest.mark.parametrize("name", SCENARIO_ORDER)
def test_no_scenario_reaches_a_hard_abort(name: str) -> None:
    transcript = SCENARIOS[name]()
    assert transcript.count("hard_abort_total") == 0


def test_run_all_builds_every_scenario_in_rfc_order() -> None:
    assert tuple(run_all()) == SCENARIO_ORDER


# -- 7.1 selector timeout ------------------------------------------------


def test_timeout_falls_open_to_the_baseline_without_charging_the_client() -> None:
    transcript = SCENARIOS["timeout"]()
    selection = transcript.selections[0]
    assert selection.outcome is SelectionOutcome.BASELINE_FAIL_OPEN
    assert selection.fail_open is FailOpenReason.SELECTOR_TIMEOUT
    assert selection.host == selection.baseline_host == BASELINE_PICK
    assert transcript.count("selector_timeout_total") == 1
    assert transcript.count("retry_budget_used") == 0
    assert transcript.count("decode_reject_total") == 0


def test_every_fail_open_reason_has_a_distinct_counter() -> None:
    assert set(FAIL_OPEN_COUNTERS) == set(FailOpenReason)
    assert len(set(FAIL_OPEN_COUNTERS.values())) == len(FailOpenReason)


def test_selector_error_also_falls_open_rather_than_rejecting() -> None:
    harness = ChainHarness(enforce_config(), scenario="error")
    selection = harness.select(fresh_view(), error=True)
    assert selection.outcome is SelectionOutcome.BASELINE_FAIL_OPEN
    assert harness.counters["selector_error_total"] == 1


# -- 7.2 stale view ------------------------------------------------------


def test_stale_view_degrades_enforce_to_shadow_and_still_records_scores() -> None:
    transcript = SCENARIOS["stale_view"]()
    selection = transcript.selections[0]
    assert selection.mode is EnforcementMode.ENFORCE
    assert selection.outcome is SelectionOutcome.SHADOW_RECORDED
    assert selection.fail_open is FailOpenReason.STALE_VIEW
    assert selection.host == BASELINE_PICK
    assert selection.selector_host == SELECTOR_PICK
    assert transcript.count("selector_stale_view_total") == 1
    assert transcript.count("selector_shadow_scores_total") == 3


def test_client_preference_is_advisory_and_its_rejection_is_counted() -> None:
    transcript = SCENARIOS["stale_view"]()
    assert transcript.selections[0].preference_ignored is True
    assert transcript.count("selector_pref_ignored_total") == 1


def test_exclusion_is_the_only_hard_client_constraint() -> None:
    snapshot = fresh_view()
    route = LegRoute(exclude_hosts=frozenset({SELECTOR_PICK}))
    assert selector_host(snapshot, route) != SELECTOR_PICK
    assert baseline_host(snapshot, route) != SELECTOR_PICK


def test_a_fresh_view_lets_enforce_apply_the_selector_choice() -> None:
    harness = ChainHarness(enforce_config(), scenario="fresh")
    full_handshake(harness)
    selection = harness.select(fresh_view())
    assert selection.outcome is SelectionOutcome.SELECTOR_APPLIED
    assert selection.host == SELECTOR_PICK != selection.baseline_host


def test_view_age_boundary_is_inclusive_of_the_configured_maximum() -> None:
    harness = ChainHarness(enforce_config(), scenario="boundary")
    at_limit = ViewSnapshot(1, 200, ("host-a", "host-b"), (1, 2))
    assert harness.select(at_limit).fail_open is None
    over_limit = ViewSnapshot(2, 201, ("host-a", "host-b"), (1, 2))
    assert harness.select(over_limit).fail_open is FailOpenReason.STALE_VIEW


# -- 7.3 prefill rollback ------------------------------------------------


def test_rollback_books_waste_without_moving_the_epoch_or_the_user_stream() -> None:
    transcript = SCENARIOS["p_rollback"]()
    assert transcript.prefill_waste_work == pytest.approx(2.5)
    assert transcript.count("prefill_rollback_total") == 1
    assert transcript.final_epoch == 0
    kinds = transcript.kinds()
    rollback_at = kinds.index(ChainEventKind.PREFILL_ROLLBACK)
    assert ChainEventKind.FRAME_ACCEPTED not in kinds[: rollback_at + 1]


def test_rollback_after_user_visible_output_is_refused() -> None:
    harness = ChainHarness(enforce_config(), scenario="illegal_rollback")
    harness.select(fresh_view())
    harness.commit_prefill()
    harness.admit_decode()
    harness.emit(3)
    with pytest.raises(ChainProtocolError, match="user-visible output"):
        harness.rollback_prefill()


# -- 7.4 decode reject ---------------------------------------------------


def test_decode_reject_costs_retry_budget_and_starts_a_new_attempt() -> None:
    transcript = SCENARIOS["d_reject"]()
    assert transcript.count("decode_reject_total") == 1
    assert transcript.count("retry_budget_used") == 1
    assert transcript.attempts == 1


def test_reselection_after_reject_goes_back_through_the_owner() -> None:
    transcript = SCENARIOS["d_reject"]()
    assert len(transcript.selections) == 2
    assert transcript.selections[1].host != SELECTOR_PICK
    kinds = transcript.kinds()
    reject_at = kinds.index(ChainEventKind.DECODE_REJECT)
    assert ChainEventKind.SELECT in kinds[reject_at:]


def test_decode_admission_before_prefill_commit_is_refused() -> None:
    harness = ChainHarness(enforce_config(), scenario="early_admit")
    harness.select(fresh_view())
    with pytest.raises(ChainProtocolError, match="before prefill commit"):
        harness.admit_decode()


def test_retry_budget_is_finite() -> None:
    harness = ChainHarness(enforce_config(), scenario="budget")
    harness.select(fresh_view())
    harness.commit_prefill()
    assert harness.reject_decode() is True
    assert harness.attempt == 1
    harness.select(fresh_view(epoch=8))
    harness.commit_prefill()
    assert harness.reject_decode() is False
    assert harness.attempt == 2
    assert harness.retry_budget_left == 0


def test_a_second_rejection_requires_a_full_new_admission_attempt() -> None:
    harness = ChainHarness(enforce_config(), scenario="repeat_reject")
    harness.select(fresh_view())
    harness.commit_prefill()
    assert harness.reject_decode() is True
    # The rejection consumed the selection and the commit: rejecting the
    # same decision twice is a protocol error, not a budget drain.
    with pytest.raises(ChainProtocolError, match="owner-selected host"):
        harness.reject_decode()
    harness.select(fresh_view(epoch=8))
    with pytest.raises(ChainProtocolError, match="no prefill commit"):
        harness.reject_decode()
    assert harness.counters["decode_reject_total"] == 1
    assert harness.counters["retry_budget_used"] == 1


# -- 7.5 lease expiry ----------------------------------------------------


def test_lease_expiry_is_not_a_terminal_and_costs_no_retry_budget() -> None:
    transcript = SCENARIOS["lease_expiry"]()
    assert transcript.count("lease_expiry_total") == 2
    assert transcript.count("retry_budget_used") == 0
    assert transcript.terminals == ("STOP",)
    assert transcript.kinds().count(ChainEventKind.TERMINAL) == 1


def test_lease_expiry_never_appears_after_the_terminal() -> None:
    transcript = SCENARIOS["lease_expiry"]()
    kinds = transcript.kinds()
    assert kinds.index(ChainEventKind.TERMINAL) == len(kinds) - 1


def test_continuation_length_is_original_input_plus_delivered_tokens() -> None:
    harness = ChainHarness(
        enforce_config(), scenario="continuation", input_tokens=INPUT_TOKENS
    )
    harness.select(fresh_view())
    harness.commit_prefill()
    harness.admit_decode()
    harness.emit(8)
    assert harness.expire_lease() == INPUT_TOKENS + 8
    assert harness.continuation_input_tokens == INPUT_TOKENS + 8


def test_lease_rounds_are_bounded_by_max_rounds() -> None:
    config = ChainConfig(
        mode=EnforcementMode.ENFORCE, lease_schedule=(4,), max_rounds=2
    )
    harness = ChainHarness(config, scenario="rounds")
    for _ in range(2):
        harness.select(fresh_view())
        harness.commit_prefill()
        harness.admit_decode()
        harness.expire_lease()
    harness.select(fresh_view())
    harness.commit_prefill()
    harness.admit_decode()
    with pytest.raises(ChainProtocolError, match="max_rounds"):
        harness.expire_lease()


def test_max_rounds_exhaustion_fences_the_expired_leg() -> None:
    config = ChainConfig(
        mode=EnforcementMode.ENFORCE, lease_schedule=(4,), max_rounds=1
    )
    harness = ChainHarness(config, scenario="rounds_fence")
    harness.select(fresh_view())
    harness.commit_prefill()
    harness.admit_decode()
    harness.emit(2)
    harness.expire_lease()
    harness.select(fresh_view(epoch=8))
    harness.commit_prefill()
    harness.admit_decode()
    exhausted_leg = harness.current_leg
    assert exhausted_leg is not None
    harness.emit(2)
    epoch_before = harness.epoch
    with pytest.raises(ChainProtocolError, match="max_rounds"):
        harness.expire_lease()
    # The refusal is a fenced transition: the epoch advanced, so the retained
    # leg reference cannot keep delivering to the user.
    assert harness.epoch == epoch_before + 1
    assert harness.current_leg is None
    assert exhausted_leg.emit(2) == 0
    assert harness.counters["stale_epoch_frame_dropped_total"] == 2
    assert harness.counters["lease_rounds_exhausted_total"] == 1
    with pytest.raises(ChainProtocolError, match="lease rounds exhausted"):
        harness.select(fresh_view(epoch=9))
    with pytest.raises(ChainProtocolError, match="only permitted terminal"):
        harness.finish("STOP")
    harness.finish("ERROR")
    assert harness.transcript().terminals == ("ERROR",)


def test_lease_expiry_requires_an_active_decode_leg() -> None:
    harness = ChainHarness(enforce_config(), scenario="no_leg")
    with pytest.raises(ChainProtocolError, match="active decode leg"):
        harness.expire_lease()


def test_lease_expiry_advances_the_epoch_and_fences_the_expired_leg() -> None:
    harness = ChainHarness(enforce_config(), scenario="lease_fence")
    harness.select(fresh_view())
    harness.commit_prefill()
    harness.admit_decode()
    expired_leg = harness.current_leg
    assert expired_leg is not None
    harness.emit(4)
    epoch_before = harness.epoch
    harness.expire_lease()
    assert harness.epoch == epoch_before + 1
    # The expired leg's in-flight frame arrives after the boundary.
    assert expired_leg.emit(1) == 0
    assert harness.counters["stale_epoch_frame_dropped_total"] == 1
    # The next leg picks up on the new epoch with no gap and no duplicate.
    harness.select(fresh_view(epoch=8))
    harness.commit_prefill()
    harness.admit_decode()
    assert harness.emit(2) == 2
    transcript = harness.transcript()
    assert [seq for _, seq in transcript.user_frames] == [0, 1, 2, 3, 4, 5]


def test_the_production_lease_schedule_is_eight_eight_fourteen() -> None:
    config = enforce_config()
    assert [config.lease_tokens(i) for i in range(4)] == [8, 8, 14, 14]


# -- 7.6 late frame ------------------------------------------------------


def test_a_superseded_epoch_frame_is_dropped_not_delivered() -> None:
    transcript = SCENARIOS["late_frame"]()
    assert transcript.count("late_frame_dropped_total") == 1
    assert len(transcript.dropped_frames) == 1
    dropped_epoch, dropped_seq = transcript.dropped_frames[0]
    assert dropped_epoch < transcript.final_epoch
    assert (dropped_epoch, dropped_seq) not in transcript.user_frames


def test_output_sequence_stays_monotone_across_the_epoch_change() -> None:
    transcript = SCENARIOS["late_frame"]()
    seqs = [seq for _, seq in transcript.user_frames]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)


def test_an_out_of_order_frame_on_the_live_epoch_is_also_dropped() -> None:
    harness = ChainHarness(enforce_config(), scenario="out_of_order")
    harness.select(fresh_view())
    harness.commit_prefill()
    harness.admit_decode()
    harness.emit(3)
    assert harness.deliver_frame(epoch=harness.epoch, output_seq=99) is False
    assert harness.counters["late_frame_dropped_total"] == 1


# -- 7.7 client crash ----------------------------------------------------


def test_restart_advances_the_epoch_and_settles_the_orphan_by_lease() -> None:
    transcript = SCENARIOS["client_crash"]()
    assert transcript.count("client_restart_total") == 1
    assert transcript.final_epoch == 1
    assert transcript.orphan_legs == (SELECTOR_PICK,)
    assert transcript.count("orphan_leg_lease_expiry_total") == 1
    assert ChainEventKind.ORPHAN_LEG_SETTLED in transcript.kinds()


def test_a_stale_r2_checkpoint_cannot_rewind_the_stream() -> None:
    transcript = SCENARIOS["client_crash"]()
    assert transcript.count("r2_recovery_total") == 1
    first_after_restart = next(
        event
        for event in transcript.events
        if event.kind is ChainEventKind.FRAME_ACCEPTED and event.epoch == 1
    )
    assert first_after_restart.detail.startswith("seq=12 ")


def test_recovery_resumes_from_delivered_when_r2_is_empty() -> None:
    harness = ChainHarness(enforce_config(), scenario="r2_miss")
    harness.select(fresh_view())
    harness.commit_prefill()
    harness.admit_decode()
    harness.emit(4)
    assert harness.crash_and_restart() == 4
    assert harness.counters.get("r2_recovery_total", 0) == 0


def test_a_frame_after_the_terminal_is_refused() -> None:
    harness = ChainHarness(enforce_config(), scenario="post_terminal")
    harness.finish("STOP")
    with pytest.raises(ChainProtocolError, match="after terminal"):
        harness.deliver_frame(epoch=0, output_seq=0)


def test_a_request_has_exactly_one_visible_terminal() -> None:
    harness = ChainHarness(enforce_config(), scenario="double_terminal")
    harness.finish("STOP")
    with pytest.raises(ChainProtocolError, match="exactly one visible terminal"):
        harness.finish("LENGTH")


# -- D2 stays gated and cooperative --------------------------------------


def test_hard_abort_is_not_part_of_the_protocol() -> None:
    harness = ChainHarness(enforce_config(), scenario="abort")
    with pytest.raises(ChainProtocolError, match="arbitrary hard abort"):
        harness.hard_abort("impatient scheduler")


def test_with_the_d2_gate_closed_only_report_checkpoint_is_reachable() -> None:
    harness = ChainHarness(enforce_config(), scenario="gate_closed")
    harness.select(fresh_view())
    harness.commit_prefill()
    harness.admit_decode()
    harness.emit(20)
    decision, grant = harness.report_checkpoint(
        move_economics(), leg=harness.current_leg, served_quantum=20
    )
    assert decision.action is DecodeAction.REPORT_CHECKPOINT
    assert decision.reason == D2_GATE_CLOSED
    assert grant.target_epoch == grant.source_epoch == harness.epoch
    assert harness.counters.get("abort_self_total", 0) == 0


def test_the_same_economics_move_only_once_the_gate_is_open() -> None:
    config = enforce_config(
        d2_gate_open=True, preemption=PreemptionConfig(min_quantum=4)
    )
    harness = ChainHarness(config, scenario="gate_open")
    harness.handshake(Capabilities(), Capabilities())
    harness.select(fresh_view())
    harness.commit_prefill()
    harness.admit_decode()
    harness.emit(6)
    decision, grant = harness.report_checkpoint(
        move_economics(), leg=harness.current_leg, served_quantum=6
    )
    assert decision.action is DecodeAction.ABORT_SELF
    assert grant.target_epoch == grant.source_epoch + 1


def test_an_unavailable_planner_fails_open_to_keep_not_to_abort() -> None:
    config = enforce_config(
        d2_gate_open=True, preemption=PreemptionConfig(min_quantum=4)
    )
    harness = ChainHarness(config, scenario="planner_down")
    harness.handshake(Capabilities(), Capabilities())
    harness.select(fresh_view())
    harness.commit_prefill()
    harness.admit_decode()
    harness.emit(6)
    decision, _ = harness.report_checkpoint(
        move_economics(),
        leg=harness.current_leg,
        served_quantum=6,
        planner_available=False,
    )
    assert decision.action is DecodeAction.REPORT_CHECKPOINT
    counter = FAIL_OPEN_COUNTERS[FailOpenReason.PLANNER_UNAVAILABLE]
    assert harness.counters[counter] == 1


def test_a_below_quantum_checkpoint_is_never_moved() -> None:
    config = enforce_config(
        d2_gate_open=True, preemption=PreemptionConfig(min_quantum=14)
    )
    harness = ChainHarness(config, scenario="min_quantum")
    harness.handshake(Capabilities(), Capabilities())
    harness.select(fresh_view())
    harness.commit_prefill()
    harness.admit_decode()
    harness.emit(3)
    decision, _ = harness.report_checkpoint(
        move_economics(), leg=harness.current_leg, served_quantum=3
    )
    assert decision.action is DecodeAction.REPORT_CHECKPOINT
    assert decision.reason == "minimum_quantum"


def test_the_r2_store_refuses_a_stale_checkpoint_write() -> None:
    harness = ChainHarness(enforce_config(), scenario="r2_fence")
    harness.store.put(DecodeCheckpoint("logical-0", 1, 10, 10, "kv-0", 1.0, 10))
    with pytest.raises(ValueError, match="stale"):
        harness.store.put(DecodeCheckpoint("logical-0", 1, 4, 4, "kv-0", 1.0, 4))


# -- checkpoint reports are bound to the reporting decode leg ---------------


def test_a_checkpoint_report_without_a_leg_is_rejected() -> None:
    harness = ChainHarness(enforce_config(), scenario="ckpt_no_leg")
    _run_one_leg(harness, 4)
    with pytest.raises(ChainProtocolError, match="must name the reporting"):
        harness.report_checkpoint(move_economics(), leg=None, served_quantum=4)
    assert harness.counters.get("checkpoint_report_total", 0) == 0


def test_a_superseded_leg_cannot_report_a_checkpoint() -> None:
    harness = ChainHarness(enforce_config(), scenario="ckpt_superseded")
    _run_one_leg(harness, 4)
    old_leg = harness.current_leg
    assert old_leg is not None
    harness.expire_lease()
    # No active leg at all: even the right identity has nothing to report on.
    with pytest.raises(ChainProtocolError, match="active decode leg"):
        harness.report_checkpoint(move_economics(), leg=old_leg, served_quantum=4)
    harness.select(fresh_view(epoch=8))
    harness.commit_prefill()
    harness.admit_decode()
    # A sibling of the active leg has no checkpoint authority either.
    with pytest.raises(ChainProtocolError, match="no checkpoint authority"):
        harness.report_checkpoint(move_economics(), leg=old_leg, served_quantum=4)
    assert harness.counters.get("checkpoint_report_total", 0) == 0


def test_a_leg_bound_checkpoint_carries_the_leg_local_sequence() -> None:
    config = enforce_config(
        d2_gate_open=True,
        preemption=PreemptionConfig(min_quantum=4, max_preemptions=2),
    )
    harness = ChainHarness(config, scenario="ckpt_leg_seq")
    harness.handshake(Capabilities(), Capabilities())
    _run_one_leg(harness, 6)
    leg = harness.current_leg
    assert leg is not None
    decision, _ = harness.report_checkpoint(move_economics(), leg=leg, served_quantum=6)
    assert decision.action is DecodeAction.ABORT_SELF
    stored = harness.store.get(harness.kv_handle)
    assert stored is not None
    assert stored.epoch == leg.epoch
    assert stored.output_seq == leg.next_seq == 6


# -- enforcement modes ---------------------------------------------------


def test_off_is_the_production_default_and_records_nothing() -> None:
    config = ChainConfig()
    assert config.mode is EnforcementMode.OFF
    harness = ChainHarness(config, scenario="off")
    selection = harness.select(fresh_view())
    assert selection.outcome is SelectionOutcome.DISABLED
    assert selection.host == selection.baseline_host == BASELINE_PICK
    assert selection.selector_host is None


def test_shadow_records_the_selector_choice_without_applying_it() -> None:
    config = ChainConfig(mode=EnforcementMode.SHADOW)
    harness = ChainHarness(config, scenario="shadow")
    full_handshake(harness)
    selection = harness.select(fresh_view())
    assert selection.outcome is SelectionOutcome.SHADOW_RECORDED
    assert selection.host == BASELINE_PICK
    assert selection.selector_host == SELECTOR_PICK
    assert harness.counters["selector_shadow_recorded_total"] == 1


def test_required_mode_is_rejected_outside_a_test_workspace() -> None:
    with pytest.raises(RequiredModeRejected, match="test-workspace only"):
        ChainConfig(mode=EnforcementMode.REQUIRED)


def test_required_mode_refuses_to_fail_open_instead_of_downgrading() -> None:
    config = ChainConfig(mode=EnforcementMode.REQUIRED, test_workspace=True)
    harness = ChainHarness(config, scenario="required")
    with pytest.raises(RequiredModeFailure, match="STALE_VIEW"):
        harness.select(stale_view())


# -- capability and version handshake ------------------------------------


def test_a_protocol_version_mismatch_removes_the_instance_from_scoring() -> None:
    result = negotiate(Capabilities(), Capabilities(protocol_version=2))
    assert result.accepted is False
    assert result.reason is not None and "protocol_version" in result.reason


def test_a_capability_gap_degrades_rather_than_rejects() -> None:
    instance = Capabilities(
        supported=frozenset(Capability) - {Capability.COOPERATIVE_PREEMPT}
    )
    result = negotiate(Capabilities(), instance)
    assert result.accepted is True
    assert result.degraded == (Capability.COOPERATIVE_PREEMPT,)
    assert result.d2_available is False


def test_score_version_affects_comparability_not_acceptance() -> None:
    result = negotiate(Capabilities(), Capabilities(score_version="other"))
    assert result.accepted is True
    assert result.comparable_scores is False


def test_a_rejected_handshake_makes_the_next_selection_fail_open() -> None:
    harness = ChainHarness(enforce_config(), scenario="handshake")
    harness.handshake(Capabilities(), Capabilities(protocol_version=99))
    selection = harness.select(fresh_view())
    assert selection.fail_open is FailOpenReason.CAPABILITY_MISMATCH
    assert selection.host == BASELINE_PICK
    assert harness.counters["selector_capability_mismatch_total"] == 1


# -- snapshot and route validation ---------------------------------------


def test_a_snapshot_with_mismatched_arity_is_rejected() -> None:
    with pytest.raises(ValueError, match="arity"):
        ViewSnapshot(1, 0, ("a", "b"), (1,))


def test_a_route_cannot_prefer_a_host_it_also_excludes() -> None:
    with pytest.raises(ValueError, match="cannot also be excluded"):
        LegRoute(prefer_host="host-a", exclude_hosts=frozenset({"host-a"}))


def test_a_fully_excluded_candidate_set_is_a_protocol_error() -> None:
    snapshot = fresh_view()
    route = LegRoute(exclude_hosts=frozenset(snapshot.candidates))
    with pytest.raises(ChainProtocolError, match="every candidate is excluded"):
        baseline_host(snapshot, route)


# -- H1: restart shares only durable state --------------------------------


def _run_one_leg(harness: ChainHarness, tokens: int) -> None:
    harness.select(fresh_view())
    harness.commit_prefill()
    harness.admit_decode()
    harness.emit(tokens)


def test_restart_without_durable_ack_fails_closed() -> None:
    config = ChainConfig(mode=EnforcementMode.ENFORCE, durable_output_ack=False)
    harness = ChainHarness(config, scenario="no_ack")
    _run_one_leg(harness, 4)
    with pytest.raises(ChainProtocolError, match="failing closed"):
        harness.crash_and_restart()
    assert harness.counters["crash_recovery_fail_closed_total"] == 1
    assert harness.counters.get("client_restart_total", 0) == 0


def test_an_eager_r2_checkpoint_cannot_skip_unacked_output() -> None:
    harness = ChainHarness(enforce_config(), scenario="eager_ckpt")
    _run_one_leg(harness, 12)
    harness.store.put(
        DecodeCheckpoint(harness.logical_request_id, 0, 20, 20, "kv-0", 1.0, 20)
    )
    assert harness.crash_and_restart() == 12
    harness.select(fresh_view(epoch=9))
    harness.commit_prefill()
    harness.admit_decode()
    assert harness.emit(1) == 1
    assert harness.transcript().user_frames[-1] == (1, 12)


def test_resume_position_is_the_durable_ack() -> None:
    harness = ChainHarness(enforce_config(), scenario="ack_resume")
    _run_one_leg(harness, 7)
    assert harness.durable_acked_seq == 7
    assert harness.crash_and_restart() == 7
    assert harness.continuation_input_tokens == INPUT_TOKENS + 7


# -- H3: leg-local sequence and fence-only deduplication ------------------


def test_a_racing_superseded_leg_cannot_double_deliver() -> None:
    harness = ChainHarness(enforce_config(), scenario="race")
    harness.select(fresh_view())
    harness.commit_prefill()
    leg_a_host = harness.admit_decode()
    leg_a = harness.current_leg
    assert leg_a is not None and leg_a.host == leg_a_host
    assert leg_a.emit(2) == 2
    # Supersession goes through the protocol: the lease ends leg A, and the
    # replacement is owner-mediated with its own attempt identity.
    harness.expire_lease()
    harness.select(fresh_view(epoch=8))
    harness.commit_prefill()
    harness.admit_decode()
    leg_b = harness.current_leg
    assert leg_b is not None and leg_b.leg_id != leg_a.leg_id
    assert leg_b.attempt != leg_a.attempt
    # Leg A races on: its frames arrive after the fence advanced.
    assert leg_a.emit(2) == 0
    assert leg_b.emit(2) == 2
    assert harness.counters["stale_epoch_frame_dropped_total"] == 2
    seqs = [seq for _, seq in harness.transcript().user_frames]
    assert seqs == [0, 1, 2, 3]


def test_a_sibling_admission_is_refused_outright() -> None:
    harness = ChainHarness(enforce_config(), scenario="sibling")
    harness.select(fresh_view())
    harness.commit_prefill()
    harness.admit_decode()
    harness.select(fresh_view(epoch=8))
    harness.commit_prefill()
    with pytest.raises(ChainProtocolError, match="already active"):
        harness.admit_decode()


def test_a_duplicate_sequence_on_the_live_epoch_is_dropped() -> None:
    harness = ChainHarness(enforce_config(), scenario="duplicate")
    _run_one_leg(harness, 3)
    assert harness.deliver_frame(epoch=harness.epoch, output_seq=0) is False
    assert harness.counters["duplicate_frame_dropped_total"] == 1
    assert harness.counters["late_frame_dropped_total"] == 1


def test_a_fenced_leg_still_advances_its_local_sequence() -> None:
    harness = ChainHarness(enforce_config(), scenario="local_seq")
    _run_one_leg(harness, 3)
    expired = harness.current_leg
    assert expired is not None
    harness.expire_lease()
    assert expired.emit(2) == 0
    assert expired.next_seq == 5
    assert harness.counters["stale_epoch_frame_dropped_total"] == 2
    assert harness.counters["late_frame_dropped_total"] == 2


def test_a_frame_claiming_an_unissued_epoch_is_a_forgery() -> None:
    harness = ChainHarness(enforce_config(), scenario="forged_epoch")
    _run_one_leg(harness, 1)
    with pytest.raises(ChainProtocolError, match="never issued"):
        harness.deliver_frame(epoch=5, output_seq=1)


def test_drop_kinds_have_distinct_counters() -> None:
    transcript = SCENARIOS["late_frame"]()
    assert transcript.count("stale_epoch_frame_dropped_total") == 1
    assert transcript.count("duplicate_frame_dropped_total") == 0
    assert transcript.count("out_of_order_frame_dropped_total") == 0
    assert transcript.count("late_frame_dropped_total") == 1


# -- H4: one epoch source of truth ----------------------------------------


def test_a_crash_then_checkpoint_fails_open_instead_of_raising() -> None:
    config = enforce_config(
        d2_gate_open=True, preemption=PreemptionConfig(min_quantum=4)
    )
    harness = ChainHarness(config, scenario="crash_then_ckpt")
    harness.handshake(Capabilities(), Capabilities())
    _run_one_leg(harness, 6)
    harness.crash_and_restart()
    harness.select(fresh_view(epoch=9))
    harness.commit_prefill()
    harness.admit_decode()
    harness.emit(4)
    decision, grant = harness.report_checkpoint(
        move_economics(), leg=harness.current_leg, served_quantum=6
    )
    assert decision.action is DecodeAction.REPORT_CHECKPOINT
    assert decision.reason == "controller_epoch_desync"
    assert grant.target_epoch == grant.source_epoch
    assert harness.counters["preemption_epoch_desync_fail_open_total"] == 1


def test_a_controller_granted_abort_keeps_both_epoch_views_in_sync() -> None:
    config = enforce_config(
        d2_gate_open=True,
        preemption=PreemptionConfig(min_quantum=4, max_preemptions=2),
    )
    harness = ChainHarness(config, scenario="epoch_sync")
    harness.handshake(Capabilities(), Capabilities())
    _run_one_leg(harness, 6)
    first, _ = harness.report_checkpoint(
        move_economics(), leg=harness.current_leg, served_quantum=6
    )
    assert first.action is DecodeAction.ABORT_SELF
    harness.select(fresh_view(epoch=9))
    harness.commit_prefill()
    harness.admit_decode()
    harness.emit(6)
    second, _ = harness.report_checkpoint(
        move_economics(), leg=harness.current_leg, served_quantum=6
    )
    assert second.reason != "controller_epoch_desync"
    assert second.action is DecodeAction.ABORT_SELF
    assert harness.epoch == 2


# -- M1: required mode counts, then hard-fails every fail-open reason ------


def _required_config(**kwargs: object) -> ChainConfig:
    return ChainConfig(
        mode=EnforcementMode.REQUIRED,
        test_workspace=True,
        **kwargs,  # type: ignore[arg-type]
    )


def test_required_mode_counts_then_fails_on_timeout() -> None:
    harness = ChainHarness(_required_config(), scenario="req_timeout")
    with pytest.raises(RequiredModeFailure, match="SELECTOR_TIMEOUT"):
        harness.select(fresh_view(), timeout=True)
    assert harness.counters["selector_timeout_total"] == 1


def test_required_mode_counts_then_fails_on_error() -> None:
    harness = ChainHarness(_required_config(), scenario="req_error")
    with pytest.raises(RequiredModeFailure, match="SELECTOR_ERROR"):
        harness.select(fresh_view(), error=True)
    assert harness.counters["selector_error_total"] == 1


def test_required_mode_counts_then_fails_on_stale_view() -> None:
    harness = ChainHarness(_required_config(), scenario="req_stale")
    with pytest.raises(RequiredModeFailure, match="STALE_VIEW"):
        harness.select(stale_view())
    assert harness.counters["selector_stale_view_total"] == 1


def test_required_mode_counts_then_fails_on_capability_mismatch() -> None:
    harness = ChainHarness(_required_config(), scenario="req_cap")
    harness.handshake(Capabilities(), Capabilities(protocol_version=99))
    with pytest.raises(RequiredModeFailure, match="CAPABILITY_MISMATCH"):
        harness.select(fresh_view())
    assert harness.counters["selector_capability_mismatch_total"] == 1


def test_required_mode_counts_then_fails_on_planner_unavailable() -> None:
    config = _required_config(
        d2_gate_open=True, preemption=PreemptionConfig(min_quantum=4)
    )
    harness = ChainHarness(config, scenario="req_planner")
    harness.handshake(Capabilities(), Capabilities())
    _run_one_leg(harness, 6)
    with pytest.raises(RequiredModeFailure, match="PLANNER_UNAVAILABLE"):
        harness.report_checkpoint(
            move_economics(),
            leg=harness.current_leg,
            served_quantum=6,
            planner_available=False,
        )
    counter = FAIL_OPEN_COUNTERS[FailOpenReason.PLANNER_UNAVAILABLE]
    assert harness.counters[counter] == 1


# -- M2: a zero quantum is invalid, not a default request ------------------


def test_zero_served_quantum_is_rejected_not_rewritten() -> None:
    harness = ChainHarness(enforce_config(), scenario="zero_quantum")
    _run_one_leg(harness, 5)
    with pytest.raises(ValueError, match="progress must be positive"):
        harness.report_checkpoint(
            move_economics(), leg=harness.current_leg, served_quantum=0
        )
    assert harness.counters.get("checkpoint_report_total", 0) == 0


# -- M3: exclusion ledger and admission host binding -----------------------


def test_exclude_hosts_must_come_from_the_failed_leg_ledger() -> None:
    harness = ChainHarness(enforce_config(), scenario="unproven_exclude")
    route = LegRoute(exclude_hosts=frozenset({SELECTOR_PICK}))
    with pytest.raises(ChainProtocolError, match="failed-leg ledger"):
        harness.select(fresh_view(), route)
    assert harness.counters["exclude_hosts_unproven_total"] == 1


def test_a_rejected_host_becomes_excludable() -> None:
    harness = ChainHarness(enforce_config(), scenario="proven_exclude")
    full_handshake(harness)
    harness.select(fresh_view())
    harness.commit_prefill()
    assert harness.reject_decode("kv_capacity") is True
    route = LegRoute(exclude_hosts=frozenset({SELECTOR_PICK}))
    selection = harness.select(fresh_view(epoch=8), route)
    assert selection.host != SELECTOR_PICK
    assert harness.counters.get("exclude_hosts_unproven_total", 0) == 0


def test_admission_host_cannot_override_the_owner_selection() -> None:
    harness = ChainHarness(enforce_config(), scenario="admit_override")
    full_handshake(harness)
    harness.select(fresh_view())
    harness.commit_prefill()
    with pytest.raises(ChainProtocolError, match="owner-selected host"):
        harness.admit_decode("host-c")
    assert harness.counters["admission_host_mismatch_total"] == 1
    assert harness.admit_decode(SELECTOR_PICK) == SELECTOR_PICK


# -- M4: no handshake means zero capabilities ------------------------------


def test_without_a_handshake_capabilities_default_to_zero() -> None:
    config = enforce_config(
        d2_gate_open=True, preemption=PreemptionConfig(min_quantum=4)
    )
    harness = ChainHarness(config, scenario="no_handshake")
    _run_one_leg(harness, 6)
    decision, grant = harness.report_checkpoint(
        move_economics(), leg=harness.current_leg, served_quantum=6
    )
    assert decision.action is DecodeAction.REPORT_CHECKPOINT
    assert decision.reason == "cooperative_preempt_not_negotiated"
    assert grant.target_epoch == grant.source_epoch
    assert harness.counters["preemption_capability_fail_open_total"] == 1


def test_selection_without_a_handshake_scores_no_cache_hits() -> None:
    harness = ChainHarness(enforce_config(), scenario="no_handshake_select")
    selection = harness.select(fresh_view())
    # No handshake means no PREFIX_CACHE_QUERY: the hit term is zero for
    # every candidate, so enforce picks the baseline and invents no score.
    assert selection.outcome is SelectionOutcome.SELECTOR_APPLIED
    assert selection.fail_open is None
    assert selection.host == BASELINE_PICK
    assert selection.selector_host is None
    assert harness.counters.get("selector_shadow_scores_total", 0) == 0


def test_a_degraded_prefix_cache_capability_zeroes_the_hit_term() -> None:
    harness = ChainHarness(enforce_config(), scenario="no_cache_capability")
    instance = Capabilities(
        supported=frozenset(Capability) - {Capability.PREFIX_CACHE_QUERY}
    )
    result = harness.handshake(Capabilities(), instance)
    assert result.accepted is True
    selection = harness.select(fresh_view())
    assert selection.outcome is SelectionOutcome.SELECTOR_APPLIED
    assert selection.host == BASELINE_PICK
    assert selection.selector_host is None


def test_shadow_without_a_handshake_records_no_selector_score() -> None:
    harness = ChainHarness(
        ChainConfig(mode=EnforcementMode.SHADOW), scenario="shadow_no_handshake"
    )
    selection = harness.select(fresh_view())
    assert selection.outcome is SelectionOutcome.SHADOW_RECORDED
    assert selection.host == BASELINE_PICK
    assert selection.selector_host is None
    assert harness.counters.get("selector_shadow_recorded_total", 0) == 0
    assert harness.counters.get("selector_shadow_scores_total", 0) == 0


def test_a_stale_view_without_a_handshake_shadows_no_score_either() -> None:
    harness = ChainHarness(enforce_config(), scenario="stale_no_handshake")
    selection = harness.select(stale_view())
    assert selection.fail_open is FailOpenReason.STALE_VIEW
    assert selection.selector_host is None
    assert harness.counters.get("selector_shadow_scores_total", 0) == 0


# -- M5: epoch-regressed views are stale regardless of age -----------------


def test_an_epoch_regressed_view_fails_open_even_when_fresh() -> None:
    harness = ChainHarness(enforce_config(), scenario="epoch_regress")
    full_handshake(harness)
    harness.select(fresh_view(epoch=7))
    regressed = harness.select(fresh_view(epoch=6, age_ms=1))
    assert regressed.fail_open is FailOpenReason.STALE_VIEW
    assert regressed.outcome is SelectionOutcome.SHADOW_RECORDED
    assert regressed.host == BASELINE_PICK
    assert harness.counters["selector_stale_view_total"] == 1


# -- M6: retry budget exhaustion is terminal -------------------------------


def test_budget_exhaustion_refuses_attempts_and_forces_one_error_terminal() -> None:
    harness = ChainHarness(enforce_config(), scenario="exhausted")
    harness.select(fresh_view())
    harness.commit_prefill()
    assert harness.reject_decode() is True
    harness.select(fresh_view(epoch=8))
    harness.commit_prefill()
    assert harness.reject_decode() is False
    assert harness.attempt == 2
    assert harness.counters["retry_budget_exhausted_total"] == 1
    with pytest.raises(ChainProtocolError, match="retry budget exhausted"):
        harness.select(fresh_view(epoch=9))
    with pytest.raises(ChainProtocolError, match="retry budget exhausted"):
        harness.reject_decode()
    with pytest.raises(ChainProtocolError, match="only permitted terminal"):
        harness.finish("STOP")
    harness.finish("ERROR")
    transcript = harness.transcript()
    assert transcript.terminals == ("ERROR",)
    assert transcript.count("terminal_total") == 1


# -- M7: the owner choice affects executable behaviour ---------------------


def test_no_owner_means_no_selector_attach_point() -> None:
    with pytest.raises(ValueError, match="attach point"):
        ChainConfig(owner=SelectorOwner.NONE, mode=EnforcementMode.ENFORCE)
    harness = ChainHarness(ChainConfig(owner=SelectorOwner.NONE), scenario="no_owner")
    assert harness.select(fresh_view()).outcome is SelectionOutcome.DISABLED


def test_turbo_pull_owner_has_no_push_preference_channel() -> None:
    turbo = ChainHarness(
        ChainConfig(
            mode=EnforcementMode.ENFORCE, owner=SelectorOwner.TURBO_CACHE_AWARE
        ),
        scenario="turbo_pull",
    )
    full_handshake(turbo)
    selection = turbo.select(fresh_view(), LegRoute(prefer_host=SELECTOR_PICK))
    assert selection.host == SELECTOR_PICK
    assert selection.preference_ignored is True
    assert turbo.counters["selector_pref_ignored_total"] == 1
    flexlb = ChainHarness(enforce_config(), scenario="flexlb_push")
    full_handshake(flexlb)
    matched = flexlb.select(fresh_view(), LegRoute(prefer_host=SELECTOR_PICK))
    assert matched.preference_ignored is False


# -- M8: output requires prefill commit and decode admission ---------------


def test_emit_before_decode_admission_is_refused() -> None:
    harness = ChainHarness(enforce_config(), scenario="early_emit")
    harness.select(fresh_view())
    harness.commit_prefill()
    with pytest.raises(ChainProtocolError, match="admitted decode leg"):
        harness.emit(1)


def test_a_raw_frame_before_any_admission_is_refused() -> None:
    harness = ChainHarness(enforce_config(), scenario="early_frame")
    with pytest.raises(ChainProtocolError, match="before any decode admission"):
        harness.deliver_frame(epoch=0, output_seq=0)


def test_admission_consumes_the_prefill_commit_and_the_selection() -> None:
    harness = ChainHarness(enforce_config(), scenario="commit_consumed")
    harness.select(fresh_view())
    harness.commit_prefill()
    harness.admit_decode()
    harness.expire_lease()
    with pytest.raises(ChainProtocolError, match="before prefill commit"):
        harness.admit_decode()
    harness.commit_prefill()
    with pytest.raises(ChainProtocolError, match="no host selected"):
        harness.admit_decode()


# -- low findings -----------------------------------------------------------


def test_a_timed_out_selection_publishes_no_selector_score() -> None:
    harness = ChainHarness(enforce_config(), scenario="timeout_score")
    selection = harness.select(fresh_view(), timeout=True)
    assert selection.selector_host is None
    assert harness.counters.get("selector_shadow_scores_total", 0) == 0
    errored = harness.select(fresh_view(), error=True)
    assert errored.selector_host is None


def test_rollback_after_decode_admission_is_refused() -> None:
    harness = ChainHarness(enforce_config(), scenario="rollback_admitted")
    harness.select(fresh_view())
    harness.commit_prefill()
    harness.admit_decode()
    with pytest.raises(ChainProtocolError, match="decode admission"):
        harness.rollback_prefill()


def test_reject_is_admission_time_only() -> None:
    harness = ChainHarness(enforce_config(), scenario="mid_leg_reject")
    harness.select(fresh_view())
    harness.commit_prefill()
    harness.admit_decode()
    with pytest.raises(ChainProtocolError, match="admission-time refusal"):
        harness.reject_decode()
