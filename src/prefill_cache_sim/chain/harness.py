"""Single-threaded chain state machine for the M11 protocol mocks.

The harness plays all four roles at once so a test can drive an exact message
order. It enforces the invariants the RFC claims rather than merely recording
them: a forbidden transition raises :class:`ChainProtocolError` instead of
producing a transcript that a later assertion has to notice.

Rules that are structural rather than configurable:

* :meth:`ChainHarness.hard_abort` always raises. There is no arbitrary abort in
  this protocol, only cooperative checkpointing and natural lease expiry.
* A lease expiry is never a user-visible terminal, so it neither appends to
  :attr:`ChainTranscript.terminals` nor consumes retry budget. It does end the
  leg's output authority: the epoch advances at the lease boundary, so a frame
  the expired leg still had in flight is fenced instead of trusted. Exhausting
  ``max_rounds`` is itself a fenced transition: the epoch still advances before
  the refusal, and only an ``ERROR`` terminal remains.
* Every decode leg is owner-mediated and has a distinct attempt identity:
  admission consumes both the owner selection and the prefill commit, sibling
  admissions are refused, and a capacity rejection consumes the pending
  admission so a second rejection needs a full new selection and commit.
* A checkpoint report names its decode leg. Reports with no leg, from a
  superseded leg, or from any leg other than the active one are rejected, and
  the checkpoint carries the leg's own epoch and local sequence.
* Cache-aware scoring requires a handshake that negotiated
  ``PREFIX_CACHE_QUERY``. Without one, the hit term is zero for every
  candidate — selection degenerates to the baseline order and no selector
  score is published or inferred.
* The epoch source of truth is the client's :class:`DurableClientLedger`. The
  in-memory :class:`~..preemption.OutputAuthority` mirrors it and is rebuilt
  from it on restart; the :class:`~..preemption.CooperativePreemptionController`
  only ever sees epochs it granted itself, and any client-driven epoch advance
  (lease expiry, crash) desynchronizes it until re-registration — which this
  mock does not model, so D2 fails open to ``REPORT_CHECKPOINT`` from then on.
* A restart shares nothing volatile with the crashed process. Recovery reads
  the durable ledger and the R2 store only; without a durable output ack the
  harness fails closed rather than pretending memory survived.

Nothing here measures anything. See :data:`~.protocol.CHAIN_EVIDENCE_NOTE`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from ..preemption import (
    ActionGrant,
    CooperativePreemptionController,
    Decision,
    DecodeAction,
    DecodeCheckpoint,
    KeepMoveInput,
    OutputAuthority,
    R2CheckpointStore,
)
from .protocol import (
    D2_GATE_CLOSED,
    FAIL_OPEN_COUNTERS,
    Capabilities,
    Capability,
    ChainConfig,
    ChainProtocolError,
    ChainRole,
    EnforcementMode,
    FailOpenReason,
    HandshakeResult,
    LegRoute,
    RequiredModeFailure,
    Selection,
    SelectionOutcome,
    SelectorOwner,
    ViewSnapshot,
    baseline_host,
    negotiate,
    score_report,
    selector_host,
)


class ChainEventKind(StrEnum):
    """Transcript vocabulary. One kind per observable protocol step."""

    HANDSHAKE = "HANDSHAKE"
    SELECT = "SELECT"
    FAIL_OPEN = "FAIL_OPEN"
    PREFILL_COMMIT = "PREFILL_COMMIT"
    PREFILL_ROLLBACK = "PREFILL_ROLLBACK"
    DECODE_ADMIT = "DECODE_ADMIT"
    DECODE_REJECT = "DECODE_REJECT"
    FRAME_ACCEPTED = "FRAME_ACCEPTED"
    FRAME_DROPPED = "FRAME_DROPPED"
    LEASE_EXPIRED = "LEASE_EXPIRED"
    CHECKPOINT_REPORT = "CHECKPOINT_REPORT"
    GRANT = "GRANT"
    CLIENT_RESTART = "CLIENT_RESTART"
    ORPHAN_LEG_SETTLED = "ORPHAN_LEG_SETTLED"
    TERMINAL = "TERMINAL"


@dataclass(frozen=True, slots=True)
class ChainEvent:
    seq: int
    kind: ChainEventKind
    role: ChainRole
    attempt: int
    epoch: int
    detail: str


@dataclass(slots=True)
class DurableClientLedger:
    """The client-owned durable record of what the user provably received.

    ``acked_seq`` is written when a frame is delivered, ``epoch`` when the
    output fence advances. This ledger — not any in-memory counter — is the
    only client state a restart may read back.
    """

    acked_seq: int = 0
    epoch: int = 0


@dataclass(slots=True)
class DecodeLeg:
    """One decode leg with its own frame counters, independent of the fence.

    The leg stamps every frame with the epoch and resume position it was
    admitted under and advances its *local* sequence whether or not the fence
    accepted the frame. Deduplication therefore lives in the fence alone: a
    superseded or racing leg keeps emitting and every frame it produces is
    individually judged.
    """

    harness: ChainHarness
    leg_id: str
    attempt: int
    host: str
    epoch: int
    next_seq: int

    def emit(self, tokens: int) -> int:
        if tokens <= 0:
            raise ValueError("emit count must be positive")
        accepted = 0
        for _ in range(tokens):
            if self.harness.deliver_frame(
                epoch=self.epoch, output_seq=self.next_seq, leg_id=self.leg_id
            ):
                accepted += 1
            self.next_seq += 1
        return accepted


@dataclass(frozen=True, slots=True)
class ChainTranscript:
    """Everything a test needs to assert, and nothing it has to recompute."""

    scenario: str
    config: ChainConfig
    events: tuple[ChainEvent, ...]
    selections: tuple[Selection, ...]
    user_frames: tuple[tuple[int, int], ...]
    dropped_frames: tuple[tuple[int, int], ...]
    terminals: tuple[str, ...]
    counters: Mapping[str, int]
    attempts: int
    final_epoch: int
    delivered_tokens: int
    prefill_waste_work: float
    orphan_legs: tuple[str, ...]

    def kinds(self) -> tuple[ChainEventKind, ...]:
        return tuple(event.kind for event in self.events)

    def count(self, name: str) -> int:
        return self.counters.get(name, 0)


class ChainHarness:
    """Drives one logical request across client, owner, prefill and decode."""

    def __init__(
        self,
        config: ChainConfig,
        *,
        scenario: str = "unnamed",
        logical_request_id: str = "logical-0",
        input_tokens: int = 128,
        output_tokens: int = 30,
        kv_handle: str = "kv-0",
    ) -> None:
        if input_tokens < 0 or output_tokens <= 0:
            raise ValueError("input tokens and output target must be sane")
        self.config = config
        self.scenario = scenario
        self.logical_request_id = logical_request_id
        self.original_input_tokens = input_tokens
        self.output_target = output_tokens
        self.kv_handle = kv_handle
        self.store = R2CheckpointStore()
        self._durable = DurableClientLedger()
        self._authority = OutputAuthority()
        self._controller = CooperativePreemptionController(
            config.preemption, self.store, {logical_request_id: self._authority}
        )
        # The epoch the controller last authorized. Client-driven epoch
        # advances (lease expiry, restart) move the authority without the
        # controller, and D2 must fail open until they re-converge.
        self._controller_epoch = 0
        # Absent a handshake, an instance has reported nothing, so it has
        # zero capabilities. Nothing is inferred: no handshake means no
        # PREFIX_CACHE_QUERY, so the cache hit term is zero and no selector
        # score exists to publish.
        self._handshake: HandshakeResult | None = None
        self._events: list[ChainEvent] = []
        self._selections: list[Selection] = []
        self._frames: list[tuple[int, int]] = []
        self._dropped: list[tuple[int, int]] = []
        self._terminals: list[str] = []
        self._orphans: list[str] = []
        self._counters: Counter[str] = Counter()
        self._attempt = 0
        self._retry_used = 0
        self._budget_exhausted = False
        self._rounds_exhausted = False
        self._delivered = 0
        self._waste = 0.0
        self._frames_this_attempt = 0
        self._prefill_committed = False
        self._current_host: str | None = None
        self._failed_hosts: set[str] = set()
        self._last_view_epoch = -1
        self._active_leg: DecodeLeg | None = None
        self._leg_counter = 0
        self._ever_admitted = False
        self._grant_seq = 0
        self._finished = False

    # -- inspection ------------------------------------------------------

    @property
    def epoch(self) -> int:
        return self._authority.epoch

    @property
    def attempt(self) -> int:
        return self._attempt

    @property
    def retry_budget_left(self) -> int:
        return self.config.retry_budget - self._retry_used

    @property
    def current_leg(self) -> DecodeLeg | None:
        return self._active_leg

    @property
    def durable_acked_seq(self) -> int:
        return self._durable.acked_seq

    @property
    def continuation_input_tokens(self) -> int:
        """The only continuation rule: original input plus what the user saw."""
        return self.original_input_tokens + self._delivered

    @property
    def counters(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self._counters))

    def transcript(self) -> ChainTranscript:
        return ChainTranscript(
            self.scenario,
            self.config,
            tuple(self._events),
            tuple(self._selections),
            tuple(self._frames),
            tuple(self._dropped),
            tuple(self._terminals),
            MappingProxyType(dict(self._counters)),
            self._attempt,
            self.epoch,
            self._delivered,
            self._waste,
            tuple(self._orphans),
        )

    # -- handshake -------------------------------------------------------

    def handshake(self, owner: Capabilities, instance: Capabilities) -> HandshakeResult:
        result = negotiate(owner, instance)
        self._handshake = result
        self._record(
            ChainEventKind.HANDSHAKE,
            ChainRole.SELECTOR_OWNER,
            f"accepted={result.accepted} degraded={','.join(result.degraded) or '-'}",
        )
        return result

    # -- selection -------------------------------------------------------

    def select(
        self,
        snapshot: ViewSnapshot,
        route: LegRoute | None = None,
        *,
        timeout: bool = False,
        error: bool = False,
    ) -> Selection:
        """Ask the existing map owner to place this leg.

        The client's ``route`` contributes a hard exclusion set and an advisory
        preference. The owner is free to discard the preference, which is what
        keeps placement authority in one place. The exclusion set is not free
        either: it may only name hosts this request's own failed legs put on
        the ledger, so a client cannot steer placement by inventing exclusions.
        """
        self._require_budget("select")
        route = route or LegRoute()
        unproven = route.exclude_hosts - self._failed_hosts
        if unproven:
            self._counters["exclude_hosts_unproven_total"] += len(unproven)
            raise ChainProtocolError(
                "exclude_hosts must come from this request's failed-leg ledger; "
                f"unproven: {','.join(sorted(unproven))}"
            )
        base = baseline_host(snapshot, route)
        mode = self.config.mode
        if mode is EnforcementMode.OFF:
            return self._finalize_selection(
                snapshot, route, base, None, SelectionOutcome.DISABLED, None
            )
        reason = self._fail_open_reason(snapshot, timeout=timeout, error=error)
        self._last_view_epoch = max(self._last_view_epoch, snapshot.epoch)
        cache_scores = self._cache_scores_negotiated
        if reason is not None:
            self._note_fail_open(reason)
            if reason is FailOpenReason.STALE_VIEW:
                return self._finalize_selection(
                    snapshot,
                    route,
                    base,
                    selector_host(snapshot, route) if cache_scores else None,
                    SelectionOutcome.SHADOW_RECORDED,
                    reason,
                )
            # A selector that timed out, threw, or never handshook produced
            # no usable score; publishing one would be an invention.
            return self._finalize_selection(
                snapshot, route, base, None, SelectionOutcome.BASELINE_FAIL_OPEN, reason
            )
        # Without a negotiated PREFIX_CACHE_QUERY the hit term is zero for
        # every candidate, so the selector's ranking degenerates to the
        # baseline order — and there is no cache-informed score to publish.
        chosen = selector_host(snapshot, route) if cache_scores else base
        scored = chosen if cache_scores else None
        if mode is EnforcementMode.SHADOW:
            if cache_scores:
                self._counters["selector_shadow_recorded_total"] += 1
            return self._finalize_selection(
                snapshot, route, base, scored, SelectionOutcome.SHADOW_RECORDED, None
            )
        return self._finalize_selection(
            snapshot, route, chosen, scored, SelectionOutcome.SELECTOR_APPLIED, None
        )

    @property
    def _cache_scores_negotiated(self) -> bool:
        """Cache-aware scoring exists only after a handshake negotiated it."""
        return (
            self._handshake is not None
            and self._handshake.accepted
            and Capability.PREFIX_CACHE_QUERY not in self._handshake.degraded
        )

    def _fail_open_reason(
        self, snapshot: ViewSnapshot, *, timeout: bool, error: bool
    ) -> FailOpenReason | None:
        if timeout:
            return FailOpenReason.SELECTOR_TIMEOUT
        if error:
            return FailOpenReason.SELECTOR_ERROR
        if snapshot.age_ms > self.config.max_view_age_ms:
            return FailOpenReason.STALE_VIEW
        if snapshot.epoch < self._last_view_epoch:
            # A regressed epoch is a stale view even when its age looks fresh.
            return FailOpenReason.STALE_VIEW
        if self._handshake is not None and not self._handshake.accepted:
            return FailOpenReason.CAPABILITY_MISMATCH
        return None

    def _note_fail_open(self, reason: FailOpenReason) -> None:
        # Count and record before raising: required mode must not hide the
        # very signal it exists to surface.
        self._counters[FAIL_OPEN_COUNTERS[reason]] += 1
        self._record(ChainEventKind.FAIL_OPEN, ChainRole.SELECTOR_OWNER, reason.value)
        if self.config.mode is EnforcementMode.REQUIRED:
            raise RequiredModeFailure(
                f"required mode refuses to fail open on {reason.value}"
            )

    def _finalize_selection(
        self,
        snapshot: ViewSnapshot,
        route: LegRoute,
        host: str,
        scored: str | None,
        outcome: SelectionOutcome,
        reason: FailOpenReason | None,
    ) -> Selection:
        # A pull-based owner has no push channel for a preference at all, so
        # under Turbo a stated preference is structurally unconsultable.
        ignored = route.prefer_host is not None and (
            host != route.prefer_host
            or self.config.owner is SelectorOwner.TURBO_CACHE_AWARE
        )
        if ignored:
            self._counters["selector_pref_ignored_total"] += 1
        if outcome is SelectionOutcome.SHADOW_RECORDED and scored is not None:
            self._counters["selector_shadow_scores_total"] += len(
                score_report(snapshot, route)
            )
        selection = Selection(
            host,
            outcome,
            self.config.mode,
            baseline_host(snapshot, route),
            scored,
            reason,
            ignored,
            snapshot.epoch,
        )
        self._selections.append(selection)
        self._current_host = host
        self._record(
            ChainEventKind.SELECT,
            ChainRole.SELECTOR_OWNER,
            f"{host} outcome={outcome.value} pref_ignored={ignored}",
        )
        return selection

    # -- prefill ---------------------------------------------------------

    def commit_prefill(self, work: float = 1.0) -> None:
        self._require_budget("commit_prefill")
        if work < 0:
            raise ValueError("prefill work must be non-negative")
        self._prefill_committed = True
        self._counters["prefill_commit_total"] += 1
        self._record(ChainEventKind.PREFILL_COMMIT, ChainRole.PREFILL, f"work={work}")

    def rollback_prefill(self, work: float = 1.0) -> None:
        """Undo an in-flight prefill and book its cost as waste.

        A rollback after any user-visible frame would mean the user saw output
        the chain then disowned, and a rollback after decode admission would
        disown a leg that is already running; both are refused. The output
        epoch is client state and a prefill rollback has no authority over it.
        """
        if work < 0:
            raise ValueError("prefill work must be non-negative")
        if self._frames_this_attempt:
            raise ChainProtocolError(
                "cannot roll back prefill after user-visible output"
            )
        if self._active_leg is not None:
            raise ChainProtocolError("cannot roll back prefill after decode admission")
        self._prefill_committed = False
        self._waste += work
        self._counters["prefill_rollback_total"] += 1
        self._record(
            ChainEventKind.PREFILL_ROLLBACK, ChainRole.PREFILL, f"waste={work}"
        )

    # -- decode ----------------------------------------------------------

    def admit_decode(self, host: str | None = None) -> str:
        """Admit one decode leg on the owner-selected host.

        ``host``, when given, is a verification token: it must equal the host
        the owner selected. The client cannot substitute a placement of its
        own at admission time. Admission consumes both the prefill commit and
        the owner selection, so every leg is owner-mediated: a second leg
        needs a second selection and a second commit. Sibling legs are
        refused outright — a new leg exists only after the previous one ended
        by lease, grant, or crash, and each of those paths advances the
        attempt index, so every leg carries a distinct attempt identity.
        """
        self._require_budget("admit_decode")
        if self._active_leg is not None:
            raise ChainProtocolError(
                "a decode leg is already active; a sibling admission is "
                "refused — a new leg requires supersession by lease expiry, "
                "checkpoint grant, or crash recovery"
            )
        if not self._prefill_committed:
            raise ChainProtocolError("decode admitted before prefill commit")
        if self._current_host is None:
            raise ChainProtocolError("no host selected for this leg")
        if host is not None and host != self._current_host:
            self._counters["admission_host_mismatch_total"] += 1
            raise ChainProtocolError(
                "admission host must be the owner-selected host, "
                f"not a client override ({host} != {self._current_host})"
            )
        target = self._current_host
        self._prefill_committed = False
        self._current_host = None
        self._ever_admitted = True
        self._frames_this_attempt = 0
        self._leg_counter += 1
        self._active_leg = DecodeLeg(
            self,
            f"leg-{self._leg_counter}",
            self._attempt,
            target,
            self.epoch,
            self._delivered,
        )
        self._counters["decode_admit_total"] += 1
        self._record(
            ChainEventKind.DECODE_ADMIT,
            ChainRole.DECODE,
            f"{target} leg={self._active_leg.leg_id}",
        )
        return target

    def reject_decode(self, reason: str = "capacity") -> bool:
        """Capacity refusal is a fault: it costs retry budget and a new attempt.

        Returns whether budget remains. A rejection refuses exactly one
        pending admission — an owner selection plus a prefill commit — and
        consumes both, so a second rejection requires a full new selection
        and prefill attempt through the owner; the client never names a
        replacement host itself. The refused host goes on the failed-leg
        ledger, which is what later entitles the client to exclude it.
        """
        self._require_budget("reject_decode")
        if self._active_leg is not None:
            raise ChainProtocolError(
                "decode reject is an admission-time refusal; an admitted leg "
                "ends by lease, checkpoint grant, or crash"
            )
        if self._current_host is None:
            raise ChainProtocolError("decode reject requires an owner-selected host")
        if not self._prefill_committed:
            raise ChainProtocolError(
                "decode reject refuses a pending admission; this attempt has "
                "no prefill commit to refuse"
            )
        self._failed_hosts.add(self._current_host)
        self._current_host = None
        self._prefill_committed = False
        self._counters["decode_reject_total"] += 1
        self._retry_used += 1
        self._counters["retry_budget_used"] = self._retry_used
        self._attempt += 1
        self._record(
            ChainEventKind.DECODE_REJECT,
            ChainRole.DECODE,
            f"{reason} budget_left={self.retry_budget_left}",
        )
        if self.retry_budget_left <= 0:
            self._budget_exhausted = True
            self._counters["retry_budget_exhausted_total"] += 1
            return False
        return True

    def emit(self, tokens: int) -> int:
        """Emit ``tokens`` frames from the currently admitted decode leg."""
        if self._active_leg is None:
            raise ChainProtocolError("emit requires an admitted decode leg")
        return self._active_leg.emit(tokens)

    def deliver_frame(self, *, epoch: int, output_seq: int, leg_id: str = "-") -> bool:
        """Route one frame through the output authority fence.

        The fence alone decides. A frame from a superseded epoch, a duplicate
        sequence, or a racing sequence is dropped — never surfaced, never
        counted toward delivered tokens — and each drop kind has its own
        counter. A frame claiming an epoch the authority never issued is a
        forgery and raises.
        """
        if self._finished:
            raise ChainProtocolError("frame delivered after terminal")
        if not self._ever_admitted:
            raise ChainProtocolError("frame delivered before any decode admission")
        if epoch > self._authority.epoch:
            raise ChainProtocolError("frame claims an epoch the fence never issued")
        if self._authority.accept(epoch=epoch, output_seq=output_seq):
            self._frames.append((epoch, output_seq))
            self._delivered += 1
            if self.config.durable_output_ack:
                self._durable.acked_seq = self._delivered
            self._frames_this_attempt += 1
            self._counters["user_frames_total"] += 1
            self._record(
                ChainEventKind.FRAME_ACCEPTED,
                ChainRole.DECODE,
                f"seq={output_seq} leg={leg_id}",
            )
            return True
        if epoch < self._authority.epoch:
            drop_kind = "stale_epoch_frame_dropped_total"
        elif output_seq < self._authority.next_seq:
            drop_kind = "duplicate_frame_dropped_total"
        else:
            drop_kind = "out_of_order_frame_dropped_total"
        self._dropped.append((epoch, output_seq))
        self._counters[drop_kind] += 1
        self._counters["late_frame_dropped_total"] += 1
        self._record(
            ChainEventKind.FRAME_DROPPED,
            ChainRole.CLIENT,
            f"epoch={epoch} seq={output_seq} leg={leg_id} kind={drop_kind}",
        )
        return False

    def expire_lease(self) -> int:
        """End the active decode leg by lease, invisibly to the user.

        Returns the continuation input length for the next leg. This path must
        not touch retry budget, must not back off, and must not emit a
        terminal. It *does* advance the output epoch: the expired leg's
        authority ends at the lease boundary, so any frame it still has in
        flight is fenced rather than trusted. Exhausting ``max_rounds`` is
        detected before any state moves and enters a fenced terminal state:
        the epoch still advances so the expired leg cannot keep delivering,
        and the only remaining transition is one ``ERROR`` terminal.
        """
        if self._finished:
            raise ChainProtocolError("lease expired after terminal")
        self._require_budget("expire_lease")
        if self._active_leg is None:
            raise ChainProtocolError("lease expiry requires an active decode leg")
        retry_before = self._retry_used
        exhausted = self._attempt + 1 > self.config.max_rounds
        self._active_leg = None
        self._attempt += 1
        self._authority.advance_epoch(self.epoch + 1, resume_output_seq=self._delivered)
        if self.config.durable_output_ack:
            self._durable.epoch = self.epoch
        if exhausted:
            self._rounds_exhausted = True
            self._counters["lease_rounds_exhausted_total"] += 1
            raise ChainProtocolError(
                "lease rounds exceeded max_rounds; the expired leg is fenced "
                "and the only remaining transition is a single ERROR terminal"
            )
        self._counters["lease_expiry_total"] += 1
        self._record(
            ChainEventKind.LEASE_EXPIRED,
            ChainRole.CLIENT,
            f"continuation={self.continuation_input_tokens}",
        )
        if self._retry_used != retry_before:
            raise ChainProtocolError("lease expiry must not consume retry budget")
        return self.continuation_input_tokens

    # -- cooperative preemption (D2, gated) ------------------------------

    def report_checkpoint(
        self,
        economics: KeepMoveInput,
        *,
        leg: DecodeLeg | None,
        served_quantum: int | None = None,
        planner_available: bool = True,
    ) -> tuple[Decision, ActionGrant]:
        """The only decode-side escape hatch, and it is advisory.

        A checkpoint report is a statement by one decode leg about its own
        progress, so the reporting ``leg`` is part of the message identity: a
        report with no leg, from a superseded leg, or from any leg other than
        the active one is rejected before it reaches the controller, and the
        checkpoint is stamped with the leg's own epoch and local sequence
        rather than any global counter.

        With the D2 gate closed the controller is not consulted at all, so the
        answer can only be ``REPORT_CHECKPOINT``. The same fail-open answer is
        returned when the instance never negotiated ``COOPERATIVE_PREEMPT``,
        when the controller's authorized epoch has fallen behind a
        client-driven advance (lease expiry or restart), or when the planner
        is unavailable. None of these paths raises outside ``required`` mode.
        """
        if leg is None:
            raise ChainProtocolError(
                "a checkpoint report must name the reporting decode leg"
            )
        if self._active_leg is None:
            raise ChainProtocolError("checkpoint report requires an active decode leg")
        if leg is not self._active_leg:
            raise ChainProtocolError(
                f"checkpoint report from {leg.leg_id} is rejected: only the "
                f"active leg {self._active_leg.leg_id} may report; superseded "
                "and sibling legs have no checkpoint authority"
            )
        if leg.epoch != self.epoch:
            raise ChainProtocolError(
                "checkpoint report from a superseded epoch is rejected"
            )
        if leg.next_seq != self._delivered:
            raise ChainProtocolError(
                "checkpoint sequence must match the leg's delivered progress"
            )
        if served_quantum is None:
            quantum = self.config.preemption.min_quantum
        else:
            # A zero quantum is invalid progress, not a request for the
            # default; DecodeCheckpoint rejects it below.
            quantum = served_quantum
        checkpoint = DecodeCheckpoint(
            self.logical_request_id,
            leg.epoch,
            leg.next_seq,
            leg.next_seq,
            self.kv_handle,
            1.0,
            quantum,
        )
        self._counters["checkpoint_report_total"] += 1
        self._record(
            ChainEventKind.CHECKPOINT_REPORT,
            ChainRole.DECODE,
            f"leg={leg.leg_id} seq={checkpoint.output_seq} quantum={quantum}",
        )
        if not self.config.d2_gate_open:
            decision = Decision(
                DecodeAction.REPORT_CHECKPOINT, 0.0, 0.0, D2_GATE_CLOSED
            )
            grant = self._local_grant(decision)
        elif self._handshake is None or not self._handshake.d2_available:
            decision = self._d2_fail_open(
                "cooperative_preempt_not_negotiated",
                "preemption_capability_fail_open_total",
            )
            grant = self._local_grant(decision)
        elif self.epoch != self._controller_epoch:
            decision = self._d2_fail_open(
                "controller_epoch_desync",
                "preemption_epoch_desync_fail_open_total",
            )
            grant = self._local_grant(decision)
        else:
            if not planner_available:
                reason = FailOpenReason.PLANNER_UNAVAILABLE
                self._counters[FAIL_OPEN_COUNTERS[reason]] += 1
                self._record(ChainEventKind.FAIL_OPEN, ChainRole.DECODE, reason.value)
                if self.config.mode is EnforcementMode.REQUIRED:
                    raise RequiredModeFailure(
                        f"required mode refuses to fail open on {reason.value}"
                    )
            decision, grant = self._controller.report_checkpoint(
                checkpoint, economics, planner_available=planner_available
            )
            self._grant_seq += 1
            if decision.action is DecodeAction.ABORT_SELF:
                self._counters["abort_self_total"] += 1
                self._attempt += 1
                self._active_leg = None
                self._controller_epoch = grant.target_epoch
                if self.config.durable_output_ack:
                    self._durable.epoch = grant.target_epoch
        if grant.target_epoch < grant.source_epoch:
            raise ChainProtocolError("a grant may never rewind the epoch")
        self._record(
            ChainEventKind.GRANT,
            ChainRole.CLIENT,
            f"{decision.action.value} reason={decision.reason} "
            f"{grant.source_epoch}->{grant.target_epoch}",
        )
        return decision, grant

    def _d2_fail_open(self, reason: str, counter: str) -> Decision:
        self._counters[counter] += 1
        self._record(ChainEventKind.FAIL_OPEN, ChainRole.DECODE, reason)
        if self.config.mode is EnforcementMode.REQUIRED:
            raise RequiredModeFailure(f"required mode refuses to fail open on {reason}")
        return Decision(DecodeAction.REPORT_CHECKPOINT, 0.0, 0.0, reason)

    def _local_grant(self, decision: Decision) -> ActionGrant:
        self._grant_seq += 1
        return ActionGrant(
            f"gate-grant-{self._grant_seq}",
            self.logical_request_id,
            decision.action,
            self.epoch,
            self.epoch,
        )

    # -- crash and recovery ----------------------------------------------

    def crash_and_restart(self) -> int:
        """Restart the client from durable state only.

        The restarted process shares nothing volatile with the crashed one:
        the in-memory fence and delivered counter are rebuilt from the durable
        output-ack ledger, and the R2 checkpoint (when present) is only KV
        recovery material — the user-visible resume position is the durable
        ack alone, so a stale checkpoint cannot rewind the stream and an
        eager checkpoint cannot skip un-acked output. Without a durable ack
        the restart fails closed: the mock refuses to invent what the user
        saw. Attempt index, retry budget, and the failed-host ledger survive
        because they are the client's durable *request ledger* (RFC §3), not
        process memory. Transcript lists and counters are the test's
        God's-eye record, not client memory, and are deliberately kept.
        """
        if self._finished:
            raise ChainProtocolError("client restarted after terminal")
        if not self.config.durable_output_ack:
            self._counters["crash_recovery_fail_closed_total"] += 1
            self._record(
                ChainEventKind.CLIENT_RESTART,
                ChainRole.CLIENT,
                "fail_closed=no_durable_output_ack",
            )
            raise ChainProtocolError(
                "restart without a durable output ack cannot prove what the "
                "user saw; failing closed instead of inventing memory survival"
            )
        stored = self.store.get(self.kv_handle)
        resume = self._durable.acked_seq
        if self._active_leg is not None:
            self._orphans.append(self._active_leg.host)
        # Volatile client state is gone; rebuild it from the durable ledger.
        self._delivered = resume
        self._frames_this_attempt = 0
        self._prefill_committed = False
        self._current_host = None
        self._active_leg = None
        self._authority = OutputAuthority(self._durable.epoch, resume)
        self._authority.advance_epoch(self._durable.epoch + 1, resume_output_seq=resume)
        self._durable.epoch = self._authority.epoch
        self._attempt += 1
        self._counters["client_restart_total"] += 1
        self._counters["r2_recovery_total"] += 1 if stored else 0
        self._record(
            ChainEventKind.CLIENT_RESTART,
            ChainRole.CLIENT,
            f"resume_seq={resume} r2={'hit' if stored else 'miss'}",
        )
        return resume

    def settle_orphan_leg(self) -> str:
        """The leg the crashed client left behind ends on its own lease."""
        if not self._orphans:
            raise ChainProtocolError("no orphan leg to settle")
        host = self._orphans[-1]
        self._counters["orphan_leg_lease_expiry_total"] += 1
        self._record(
            ChainEventKind.ORPHAN_LEG_SETTLED,
            ChainRole.DECODE,
            f"{host} natural_lease_expiry",
        )
        return "natural_lease_expiry"

    def hard_abort(self, reason: str = "unspecified") -> None:
        """Always refused. Kept so the prohibition is executable, not prose."""
        raise ChainProtocolError(
            f"arbitrary hard abort is not part of this protocol ({reason}); "
            "use cooperative checkpointing or let the lease expire"
        )

    # -- terminal --------------------------------------------------------

    def finish(self, reason: str) -> None:
        if self._finished:
            raise ChainProtocolError("a request has exactly one visible terminal")
        if (self._budget_exhausted or self._rounds_exhausted) and str(
            reason
        ) != "ERROR":
            raise ChainProtocolError(
                "budget or lease rounds exhausted; the only permitted terminal is ERROR"
            )
        self._finished = True
        self._terminals.append(str(reason))
        self._counters["terminal_total"] += 1
        self._record(ChainEventKind.TERMINAL, ChainRole.CLIENT, str(reason))

    # -- internal --------------------------------------------------------

    def _require_budget(self, action: str) -> None:
        if self._budget_exhausted:
            raise ChainProtocolError(
                f"retry budget exhausted; {action} is refused — the only "
                "remaining transition is a single ERROR terminal"
            )
        if self._rounds_exhausted:
            raise ChainProtocolError(
                f"lease rounds exhausted; {action} is refused — the only "
                "remaining transition is a single ERROR terminal"
            )

    def _record(self, kind: ChainEventKind, role: ChainRole, detail: str) -> None:
        self._events.append(
            ChainEvent(len(self._events), kind, role, self._attempt, self.epoch, detail)
        )
