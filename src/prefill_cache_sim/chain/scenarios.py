"""The seven chain scenarios named in ``docs/m11-production-rfc.md`` §7.

Each builder drives :class:`~.harness.ChainHarness` through one exact message
order and returns the resulting transcript. The builders contain no assertions;
they are the *fixtures*, and ``tests/test_chain_mocks.py`` holds the invariants.
That split keeps a scenario honest — it cannot quietly weaken a claim by editing
the assertion that guards it.

Every value below is authored, not observed. The bundle is labelled
:data:`~.protocol.CHAIN_TRUTH_BASIS` and carries
:data:`~.protocol.CHAIN_EVIDENCE_NOTE`; none of it is a production end-to-end
result.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from ..preemption import DecodeCheckpoint, KeepMoveInput, PreemptionConfig
from .harness import ChainHarness, ChainTranscript
from .protocol import (
    Capabilities,
    ChainConfig,
    EnforcementMode,
    LegRoute,
    SelectorOwner,
    TerminalReason,
    ViewSnapshot,
)

CANDIDATES = ("host-a", "host-b", "host-c")
CACHED_TOKENS = (10, 90, 40)
BASELINE_PICK = CANDIDATES[0]
SELECTOR_PICK = CANDIDATES[1]
INPUT_TOKENS = 128
OUTPUT_TOKENS = 30


def fresh_view(epoch: int = 7, age_ms: int = 20) -> ViewSnapshot:
    """A view the owner may trust."""
    return ViewSnapshot(epoch, age_ms, CANDIDATES, CACHED_TOKENS)


def stale_view(epoch: int = 7, age_ms: int = 900) -> ViewSnapshot:
    """A view older than ``max_view_age_ms``, so its scores are unusable."""
    return ViewSnapshot(epoch, age_ms, CANDIDATES, CACHED_TOKENS)


def enforce_config(
    *,
    d2_gate_open: bool = False,
    preemption: PreemptionConfig | None = None,
) -> ChainConfig:
    """``enforce`` on centralized master, D2 gate closed. The rollout target state."""
    return ChainConfig(
        mode=EnforcementMode.ENFORCE,
        owner=SelectorOwner.CENTRALIZED_MASTER,
        max_view_age_ms=200,
        lease_schedule=(8, 8, 14),
        max_rounds=3,
        retry_budget=2,
        d2_gate_open=d2_gate_open,
        preemption=preemption or PreemptionConfig(),
    )


def _harness(scenario: str, config: ChainConfig) -> ChainHarness:
    harness = ChainHarness(
        config,
        scenario=scenario,
        input_tokens=INPUT_TOKENS,
        output_tokens=OUTPUT_TOKENS,
    )
    # Every RFC scenario starts from a completed full-capability handshake;
    # without one the instance has zero capabilities and cache-aware scoring
    # never engages (see the no-handshake tests).
    harness.handshake(Capabilities(), Capabilities())
    return harness


def timeout() -> ChainTranscript:
    """7.1 — the selector does not answer in time.

    The owner falls back to the choice it would have made anyway. A timeout is
    a scoring outage, not a request fault, so it must not surface as a rejection
    and must not cost the client anything.
    """
    harness = _harness("timeout", enforce_config())
    harness.select(fresh_view(), LegRoute(hint_input_tokens=INPUT_TOKENS), timeout=True)
    harness.commit_prefill()
    harness.admit_decode()
    harness.emit(OUTPUT_TOKENS)
    harness.finish(TerminalReason.STOP)
    return harness.transcript()


def stale_view_scenario() -> ChainTranscript:
    """7.2 — the map is too old to enforce on.

    ``enforce`` degrades to ``shadow`` for this leg: the score is still recorded
    so both arms stay comparable, but placement reverts to the baseline. The
    client's ``prefer_host`` is visibly discarded, which is the point.
    """
    harness = _harness("stale_view", enforce_config())
    harness.select(stale_view(), LegRoute(prefer_host=SELECTOR_PICK))
    harness.commit_prefill()
    harness.admit_decode()
    harness.emit(OUTPUT_TOKENS)
    harness.finish(TerminalReason.STOP)
    return harness.transcript()


def prefill_rollback() -> ChainTranscript:
    """7.3 — prefill aborts before committing anything user-visible.

    The wasted GPU work is booked against waste, not against goodput. The output
    epoch must not move, because nothing was ever fenced under it.
    """
    harness = _harness("p_rollback", enforce_config())
    harness.select(fresh_view())
    harness.rollback_prefill(work=2.5)
    harness.select(fresh_view(epoch=8))
    harness.commit_prefill()
    harness.admit_decode()
    harness.emit(OUTPUT_TOKENS)
    harness.finish(TerminalReason.STOP)
    return harness.transcript()


def decode_reject() -> ChainTranscript:
    """7.4 — decode refuses admission on capacity.

    This one *is* a fault: it costs retry budget and starts a new attempt. The
    replacement host comes back through the owner with the failed host excluded;
    the client never names it.
    """
    harness = _harness("d_reject", enforce_config())
    harness.select(fresh_view())
    harness.commit_prefill()
    first = harness.reject_decode("kv_capacity")
    if not first:  # pragma: no cover - budget is 2, the first refusal cannot exhaust it
        raise AssertionError("retry budget should survive one refusal")
    retry_route = LegRoute(exclude_hosts=frozenset({SELECTOR_PICK}))
    harness.select(fresh_view(epoch=8), retry_route)
    harness.commit_prefill()
    harness.admit_decode()
    harness.emit(OUTPUT_TOKENS)
    harness.finish(TerminalReason.STOP)
    return harness.transcript()


def lease_expiry() -> ChainTranscript:
    """7.5 — three legs under the production 8/8/14 schedule.

    Each expiry ends a leg and starts the next one. None of them is a terminal,
    none of them costs retry budget, and the continuation length is always
    ``original_input + tokens the user already saw``. Each expiry also advances
    the output epoch, so a frame the expired leg still had in flight is fenced.
    """
    config = enforce_config()
    harness = _harness("lease_expiry", config)
    remaining = OUTPUT_TOKENS
    while remaining > 0:
        harness.select(fresh_view(epoch=7 + harness.attempt))
        harness.commit_prefill()
        harness.admit_decode()
        lease = min(config.lease_tokens(harness.attempt), remaining)
        harness.emit(lease)
        remaining -= lease
        if remaining > 0:
            harness.expire_lease()
    harness.finish(TerminalReason.STOP)
    return harness.transcript()


def late_frame() -> ChainTranscript:
    """7.6 — a superseded leg keeps talking.

    The D2 gate is opened here on purpose: a cooperative ``ABORT_SELF`` is the
    cheapest way to advance the epoch. D2 requires a negotiated
    ``COOPERATIVE_PREEMPT`` capability, which the scenario handshake provides.
    The checkpoint report names the leg it comes from; after the grant the
    superseded leg then emits from its *own* counters, and the frame must be
    dropped by the output authority rather than reach the user twice.
    """
    config = enforce_config(
        d2_gate_open=True,
        preemption=PreemptionConfig(min_quantum=4, max_preemptions=2),
    )
    harness = _harness("late_frame", config)
    harness.select(fresh_view())
    harness.commit_prefill()
    harness.admit_decode()
    harness.emit(6)
    stale_leg = harness.current_leg
    economics = KeepMoveInput(
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
    harness.report_checkpoint(economics, leg=stale_leg, served_quantum=6)
    if stale_leg is not None:
        stale_leg.emit(1)
    harness.select(fresh_view(epoch=9))
    harness.commit_prefill()
    harness.admit_decode()
    harness.emit(OUTPUT_TOKENS - 6)
    harness.finish(TerminalReason.STOP)
    return harness.transcript()


def client_crash() -> ChainTranscript:
    """7.7 — the client dies mid-stream and comes back.

    A checkpoint older than what the user already received is present in the R2
    store. Recovery reads durable state only: the resume position is the
    client's durable output-ack ledger, and the checkpoint is KV recovery
    material — so a stale checkpoint can never rewind the stream, an eager one
    can never skip un-acked output, and the leg the crash orphaned ends on its
    own lease rather than on an abort.
    """
    harness = _harness("client_crash", enforce_config())
    harness.select(fresh_view())
    harness.commit_prefill()
    harness.admit_decode()
    harness.emit(12)
    harness.store.put(
        DecodeCheckpoint(
            harness.logical_request_id,
            harness.epoch,
            5,
            5,
            harness.kv_handle,
            1.0,
            5,
        )
    )
    harness.crash_and_restart()
    harness.settle_orphan_leg()
    harness.select(fresh_view(epoch=9))
    harness.commit_prefill()
    harness.admit_decode()
    harness.emit(OUTPUT_TOKENS - 12)
    harness.finish(TerminalReason.STOP)
    return harness.transcript()


SCENARIO_ORDER: tuple[str, ...] = (
    "timeout",
    "stale_view",
    "p_rollback",
    "d_reject",
    "lease_expiry",
    "late_frame",
    "client_crash",
)

SCENARIOS: Mapping[str, Callable[[], ChainTranscript]] = MappingProxyType(
    {
        "timeout": timeout,
        "stale_view": stale_view_scenario,
        "p_rollback": prefill_rollback,
        "d_reject": decode_reject,
        "lease_expiry": lease_expiry,
        "late_frame": late_frame,
        "client_crash": client_crash,
    }
)


def run_all() -> Mapping[str, ChainTranscript]:
    """Build every scenario once, in the order the RFC lists them."""
    return MappingProxyType({name: SCENARIOS[name]() for name in SCENARIO_ORDER})
