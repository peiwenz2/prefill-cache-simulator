from __future__ import annotations

from dataclasses import replace

import pytest

from prefill_cache_sim.preemption import (
    CooperativePreemptionController,
    DecodeAction,
    DecodeCheckpoint,
    KeepMoveInput,
    OutputAuthority,
    PreemptionConfig,
    R2CheckpointStore,
    preemption_gate,
)


def checkpoint(
    *, epoch: int = 0, seq: int = 14, quantum: int = 14, handle: str | None = "kv"
) -> DecodeCheckpoint:
    return DecodeCheckpoint("logical", epoch, seq, seq, handle, 1.0, quantum)


def economics(*, move_probability: float = 1.0, recovery: float = 0) -> KeepMoveInput:
    return KeepMoveInput(
        completion_value=100,
        keep_probability=0.2,
        move_probability=move_probability,
        remaining_gpu_work=50,
        interference_gpu_work=50,
        checkpoint_gpu_work=1,
        recovery_gpu_work=recovery,
        migration_stall_work=1,
        duplicate_risk=0,
        wait_work=0,
        tenant_weight=1,
    )


def test_move_requires_margin_and_emits_epoch_advancing_abort_grant() -> None:
    authorities: dict[str, OutputAuthority] = {}
    controller = CooperativePreemptionController(
        PreemptionConfig(), R2CheckpointStore(), authorities
    )
    decision, grant = controller.report_checkpoint(checkpoint(), economics())
    assert decision.action is DecodeAction.ABORT_SELF
    assert (grant.source_epoch, grant.target_epoch) == (0, 1)
    assert authorities["logical"].epoch == 1
    assert not authorities["logical"].accept(epoch=0, output_seq=14)


def test_minimum_quantum_and_max_preemptions_are_hard_bounds() -> None:
    controller = CooperativePreemptionController(
        PreemptionConfig(max_preemptions=1), R2CheckpointStore()
    )
    decision, _ = controller.report_checkpoint(checkpoint(quantum=8), economics())
    assert decision.reason == "minimum_quantum"
    _, first = controller.report_checkpoint(checkpoint(), economics())
    assert first.action is DecodeAction.ABORT_SELF
    decision, second = controller.report_checkpoint(
        checkpoint(epoch=1, seq=28), economics()
    )
    assert decision.reason == "bounded_preemption"
    assert second.action is DecodeAction.REPORT_CHECKPOINT


def test_planner_failure_fails_open_without_abort() -> None:
    controller = CooperativePreemptionController(
        PreemptionConfig(), R2CheckpointStore()
    )
    decision, grant = controller.report_checkpoint(
        checkpoint(), economics(), planner_available=False
    )
    assert decision.reason == "planner_fail_open"
    assert grant.action is DecodeAction.REPORT_CHECKPOINT


def test_old_epoch_late_checkpoint_and_output_frames_are_dropped() -> None:
    controller = CooperativePreemptionController(
        PreemptionConfig(), R2CheckpointStore()
    )
    controller.report_checkpoint(checkpoint(), economics())
    with pytest.raises(ValueError, match="controller-authorized"):
        controller.report_checkpoint(checkpoint(epoch=0), economics())
    authority = OutputAuthority(epoch=1, next_seq=14)
    assert not authority.accept(epoch=0, output_seq=14)
    assert authority.accept(epoch=1, output_seq=14)
    assert not authority.accept(epoch=1, output_seq=14)


def test_epoch_advance_preserves_logical_output_sequence_fence() -> None:
    authority = OutputAuthority(epoch=0, next_seq=14)
    assert authority.accept(epoch=0, output_seq=14)
    authority.advance_epoch(1, resume_output_seq=14)
    assert not authority.accept(epoch=1, output_seq=14)
    assert authority.accept(epoch=1, output_seq=15)


def test_r2_store_is_epoch_fenced_and_separate() -> None:
    store = R2CheckpointStore()
    store.put(checkpoint())
    assert store.get("kv") == checkpoint()
    with pytest.raises(ValueError, match="stale"):
        store.put(checkpoint())
    store.put(checkpoint(epoch=1, seq=28))
    assert store.get("kv").epoch == 1  # type: ignore[union-attr]


def test_recovery_cost_can_flip_move_to_keep() -> None:
    cheap = CooperativePreemptionController(PreemptionConfig(), R2CheckpointStore())
    costly = CooperativePreemptionController(PreemptionConfig(), R2CheckpointStore())
    assert (
        cheap.report_checkpoint(checkpoint(), economics(recovery=0))[0].action
        is DecodeAction.ABORT_SELF
    )
    assert (
        costly.report_checkpoint(checkpoint(), economics(recovery=500))[0].action
        is DecodeAction.REPORT_CHECKPOINT
    )


def test_aging_monotonically_protects_waiting_request() -> None:
    config = PreemptionConfig(aging_rate=1)
    fresh = CooperativePreemptionController(config, R2CheckpointStore())
    aged = CooperativePreemptionController(config, R2CheckpointStore())
    base = economics(move_probability=0.1)
    older = replace(base, wait_work=100)
    fresh_decision = fresh.report_checkpoint(checkpoint(), base)[0]
    aged_decision = aged.report_checkpoint(checkpoint(), older)[0]
    assert aged_decision.keep_score >= fresh_decision.keep_score
    assert (
        aged_decision.move_score - aged_decision.keep_score
        <= fresh_decision.move_score - fresh_decision.keep_score
    )


def test_failed_checkpoint_store_does_not_advance_epoch_or_counter() -> None:
    store = R2CheckpointStore()
    store.put(checkpoint(epoch=5, seq=99))
    controller = CooperativePreemptionController(PreemptionConfig(), store)
    with pytest.raises(ValueError, match="stale"):
        controller.report_checkpoint(checkpoint(), economics())
    assert controller.preemptions.get("logical", 0) == 0
    decision, grant = controller.report_checkpoint(checkpoint(handle=None), economics())
    assert decision.action is DecodeAction.ABORT_SELF
    assert grant.target_epoch == 1


def test_controller_rejects_reporter_declared_future_epoch() -> None:
    controller = CooperativePreemptionController(
        PreemptionConfig(), R2CheckpointStore()
    )
    with pytest.raises(ValueError, match="controller-authorized"):
        controller.report_checkpoint(checkpoint(epoch=10**9), economics())


def test_preemptions_per_completed_request_hard_gate() -> None:
    config = PreemptionConfig(max_preemptions_per_completed=0.25)
    assert preemption_gate(completed_requests=100, preemptions=25, config=config)
    assert not preemption_gate(completed_requests=100, preemptions=26, config=config)


def test_controller_enforces_bound_beyond_runner_loop() -> None:
    config = PreemptionConfig(max_preemptions=2)
    controller = CooperativePreemptionController(config, R2CheckpointStore())
    reasons: list[str] = []
    for epoch in range(3):
        decision, _ = controller.report_checkpoint(
            checkpoint(epoch=epoch, seq=14 * (epoch + 1), handle=f"kv-{epoch}"),
            economics(),
        )
        reasons.append(decision.reason)
    assert reasons.count("move_exceeds_margin") == 2
    assert reasons[-1] == "bounded_preemption"
