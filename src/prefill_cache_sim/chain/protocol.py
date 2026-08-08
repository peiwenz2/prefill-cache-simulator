"""Message and configuration vocabulary for the M11 chain mocks.

This module encodes the contract described in ``docs/m11-production-rfc.md``:
who selects an instance, who owns the request lifecycle, and what an
implementation is allowed to do when a dependency is unavailable. It is a
protocol description, not a performance model — nothing here estimates latency
or throughput.

Honesty note: every value produced through this package is constructed by the
mock. It is labelled :data:`CHAIN_TRUTH_BASIS` and must never be reported as a
production end-to-end measurement.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from ..preemption import PreemptionConfig
from ..replay.sources import TruthBasis

CHAIN_SCHEMA_VERSION = "m11-chain-v1"
SELECTOR_PROTOCOL_VERSION = 1
SELECTOR_SCORE_VERSION = "m11-score-v1"
CHAIN_TRUTH_BASIS = TruthBasis.SYNTHETIC_FIXTURE
CHAIN_EVIDENCE_NOTE = (
    "Chain mocks assert cross-component protocol invariants only. Every number "
    "they produce is constructed; none of it is a production end-to-end "
    "measurement, a hardware result, or a canary observation."
)
D2_GATE_CLOSED = "d2_gate_closed"


class ChainRole(StrEnum):
    """The four participants the mocks model."""

    CLIENT = "CLIENT"
    SELECTOR_OWNER = "SELECTOR_OWNER"
    PREFILL = "PREFILL"
    DECODE = "DECODE"


class SelectorOwner(StrEnum):
    """Which existing component holds the global map.

    M11 never introduces a fourth owner; it contributes a scoring function to
    one of these.
    """

    CENTRALIZED_MASTER = "CENTRALIZED_MASTER"
    TURBO_CACHE_AWARE = "TURBO_CACHE_AWARE"
    NONE = "NONE"


class EnforcementMode(StrEnum):
    """Rollout stage of the selector. ``OFF`` is the production default."""

    OFF = "off"
    SHADOW = "shadow"
    ENFORCE = "enforce"
    REQUIRED = "required"


class Capability(StrEnum):
    """Negotiated features. A missing capability degrades, it does not fail."""

    PREFIX_CACHE_QUERY = "PREFIX_CACHE_QUERY"
    DECODE_LEASE_V1 = "DECODE_LEASE_V1"
    R2_CHECKPOINT_STORE = "R2_CHECKPOINT_STORE"
    COOPERATIVE_PREEMPT = "COOPERATIVE_PREEMPT"


class FailOpenReason(StrEnum):
    """Why a selection fell back to the pre-existing baseline."""

    SELECTOR_TIMEOUT = "SELECTOR_TIMEOUT"
    SELECTOR_ERROR = "SELECTOR_ERROR"
    STALE_VIEW = "STALE_VIEW"
    CAPABILITY_MISMATCH = "CAPABILITY_MISMATCH"
    PLANNER_UNAVAILABLE = "PLANNER_UNAVAILABLE"


class SelectionOutcome(StrEnum):
    """What the owner actually did with the selector score."""

    DISABLED = "DISABLED"
    SHADOW_RECORDED = "SHADOW_RECORDED"
    SELECTOR_APPLIED = "SELECTOR_APPLIED"
    BASELINE_FAIL_OPEN = "BASELINE_FAIL_OPEN"


class TerminalReason(StrEnum):
    """User-visible terminal. A lease expiry is never one of these."""

    STOP = "STOP"
    LENGTH = "LENGTH"
    ERROR = "ERROR"


class ChainProtocolError(Exception):
    """Raised when a participant attempts a forbidden transition."""


class RequiredModeRejected(Exception):
    """Raised when ``required`` mode is configured outside a test workspace."""


class RequiredModeFailure(Exception):
    """Raised when ``required`` mode would have had to fail open."""


@dataclass(frozen=True, slots=True)
class Capabilities:
    """One side of the capability and version handshake."""

    protocol_version: int = SELECTOR_PROTOCOL_VERSION
    score_version: str = SELECTOR_SCORE_VERSION
    supported: frozenset[Capability] = frozenset(Capability)

    def __post_init__(self) -> None:
        if self.protocol_version < 0:
            raise ValueError("protocol version must be non-negative")
        if not self.score_version:
            raise ValueError("score version must be named")


@dataclass(frozen=True, slots=True)
class HandshakeResult:
    """Outcome of :func:`negotiate`.

    ``accepted`` false means the instance stays reachable through the baseline
    path; it is only removed from the *selector* candidate set.
    """

    accepted: bool
    degraded: tuple[Capability, ...]
    owner_score_version: str
    instance_score_version: str
    reason: str | None = None

    @property
    def d2_available(self) -> bool:
        return self.accepted and Capability.COOPERATIVE_PREEMPT not in self.degraded

    @property
    def comparable_scores(self) -> bool:
        return self.owner_score_version == self.instance_score_version


def negotiate(owner: Capabilities, instance: Capabilities) -> HandshakeResult:
    """Compare an owner's expectations against one instance's report.

    Only ``protocol_version`` is a compatibility gate. ``score_version`` affects
    comparability of recorded scores, never acceptance, and capabilities the
    instance did not report are treated as absent rather than inferred.
    """
    degraded = tuple(sorted(owner.supported - instance.supported))
    if owner.protocol_version != instance.protocol_version:
        return HandshakeResult(
            False,
            tuple(sorted(owner.supported)),
            owner.score_version,
            instance.score_version,
            "selector_protocol_version mismatch",
        )
    reason = f"missing {','.join(degraded)}" if degraded else None
    return HandshakeResult(
        True, degraded, owner.score_version, instance.score_version, reason
    )


@dataclass(frozen=True, slots=True)
class ViewSnapshot:
    """The owner's map as observed at one instant.

    ``epoch`` is monotonic per owner and ``age_ms`` is sample-to-use delay; the
    pair is what makes a stale view detectable instead of silently trusted.
    """

    epoch: int
    age_ms: int
    candidates: tuple[str, ...]
    cached_tokens: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.epoch < 0 or self.age_ms < 0:
            raise ValueError("snapshot epoch and age must be non-negative")
        if not self.candidates:
            raise ValueError("snapshot must carry at least one candidate")
        if len(self.candidates) != len(self.cached_tokens):
            raise ValueError("candidate and cached-token arity must match")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("candidates must be distinct")
        if any(value < 0 for value in self.cached_tokens):
            raise ValueError("cached tokens must be non-negative")

    def hit_tokens(self, host: str) -> int:
        return self.cached_tokens[self.candidates.index(host)]


@dataclass(frozen=True, slots=True)
class LegRoute:
    """What the client is allowed to say about placement.

    ``prefer_host`` is advisory and the owner may discard it. ``exclude_hosts``
    is the only hard constraint, because it encodes a leg that already failed.
    """

    prefer_host: str | None = None
    exclude_hosts: frozenset[str] = frozenset()
    hint_input_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.hint_input_tokens is not None and self.hint_input_tokens < 0:
            raise ValueError("input-token hint must be non-negative")
        if self.prefer_host is not None and self.prefer_host in self.exclude_hosts:
            raise ValueError("preferred host cannot also be excluded")


@dataclass(frozen=True, slots=True)
class Selection:
    """One placement decision, with enough detail to audit fail-open."""

    host: str
    outcome: SelectionOutcome
    mode: EnforcementMode
    baseline_host: str
    selector_host: str | None
    fail_open: FailOpenReason | None
    preference_ignored: bool
    snapshot_epoch: int


@dataclass(frozen=True, slots=True)
class ChainConfig:
    """Deployment shape of one chain mock run.

    ``durable_output_ack`` models whether the client has a durable medium for
    acknowledging delivered frames. Without it, crash recovery cannot prove
    what the user saw and must fail closed rather than resume mid-stream.
    """

    mode: EnforcementMode = EnforcementMode.OFF
    owner: SelectorOwner = SelectorOwner.CENTRALIZED_MASTER
    test_workspace: bool = False
    max_view_age_ms: int = 200
    lease_schedule: tuple[int, ...] = (8, 8, 14)
    max_rounds: int = 3
    retry_budget: int = 2
    d2_gate_open: bool = False
    durable_output_ack: bool = True
    preemption: PreemptionConfig = field(default_factory=PreemptionConfig)

    def __post_init__(self) -> None:
        if self.mode is EnforcementMode.REQUIRED and not self.test_workspace:
            raise RequiredModeRejected(
                "required mode is test-workspace only; production must not "
                "silently downgrade it to enforce"
            )
        if self.owner is SelectorOwner.NONE and self.mode is not EnforcementMode.OFF:
            raise ValueError(
                "with no map owner there is no selector attach point; "
                "mode must stay off"
            )
        if self.max_view_age_ms < 0 or self.retry_budget < 0:
            raise ValueError("view age and retry budget must be non-negative")
        if not self.lease_schedule or any(v <= 0 for v in self.lease_schedule):
            raise ValueError("lease schedule must be positive")
        if self.max_rounds <= 0:
            raise ValueError("max rounds must be positive")

    def lease_tokens(self, attempt: int) -> int:
        index = min(attempt, len(self.lease_schedule) - 1)
        return self.lease_schedule[index]


def baseline_host(snapshot: ViewSnapshot, route: LegRoute) -> str:
    """The choice the chain already makes today, ignoring the selector."""
    for host in snapshot.candidates:
        if host not in route.exclude_hosts:
            return host
    raise ChainProtocolError("every candidate is excluded")


def selector_host(snapshot: ViewSnapshot, route: LegRoute) -> str:
    """The cache-aware choice: most reusable prefix, ties broken by map order."""
    allowed = [h for h in snapshot.candidates if h not in route.exclude_hosts]
    if not allowed:
        raise ChainProtocolError("every candidate is excluded")
    ranked = sorted(
        allowed, key=lambda host: (-snapshot.hit_tokens(host), allowed.index(host))
    )
    return ranked[0]


def score_report(snapshot: ViewSnapshot, route: LegRoute) -> Mapping[str, int]:
    """Per-host scores recorded in shadow mode, so both arms stay comparable."""
    return MappingProxyType(
        {
            host: snapshot.hit_tokens(host)
            for host in snapshot.candidates
            if host not in route.exclude_hosts
        }
    )


FAIL_OPEN_COUNTERS: Mapping[FailOpenReason, str] = MappingProxyType(
    {
        FailOpenReason.SELECTOR_TIMEOUT: "selector_timeout_total",
        FailOpenReason.SELECTOR_ERROR: "selector_error_total",
        FailOpenReason.STALE_VIEW: "selector_stale_view_total",
        FailOpenReason.CAPABILITY_MISMATCH: "selector_capability_mismatch_total",
        FailOpenReason.PLANNER_UNAVAILABLE: "preemption_fail_open_total",
    }
)
