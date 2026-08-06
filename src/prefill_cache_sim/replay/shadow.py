"""Record what the harness would have chosen, and refuse to act on it.

A shadow decision is evidence, not a command. :data:`ENFORCEMENT_ENABLED` is a
module-level constant rather than a runtime flag so that enabling enforcement
requires editing this file — there is no configuration path, environment
variable, or keyword argument that turns it on. A decision that claims to be
enforced cannot even be constructed: :class:`ShadowDecision` raises in
``__post_init__``, so an enforced decision cannot exist long enough to be acted
upon.

The gate records *every* reason it withheld, not just the first, because a run
that fails three checks and a run that fails one are different situations and
collapsing them would hide that.

Every threshold comparison here is a ``<`` or ``>`` against a float, and every
such comparison with ``NaN`` is false -- so a ``NaN`` input would clear *all*
the gates at once and arrive at ``SHADOW_RECOMMENDED``. The single input the
gate is least able to judge would produce its most confident output. The gate
therefore rejects a non-finite input before comparing anything.

A decision carries the :class:`ShadowGate` it was judged against, and re-derives
its own verdict from that policy on construction. An earlier version left the
thresholds on the gate and out of the record, reasoning that a decision holding
its own thresholds could be made to justify itself. That was backwards: with no
thresholds stored, *nothing* could be re-derived, and a hand-written record
claiming ``SHADOW_RECOMMENDED`` over measurements the default gate would have
rejected passed every check. Storing the policy does let a forger state a
permissive one -- but it has to state it, in the artifact, where a reader
comparing it against :data:`DEFAULT_SHADOW_GATE` can see the substitution. A
claim that is visible and checkable beats a claim that is merely unstated.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .payload import (
    check,
    check_finite,
    check_ordered_subset,
    check_unit_interval,
    optional_float,
    require_bool,
    require_float,
    require_mapping,
    require_object,
    require_str,
    require_str_sequence,
)

SHADOW_SCHEMA_VERSION = "m10-shadow-v1"

#: Schema of the gate policy embedded in every decision. Separate from
#: :data:`SHADOW_SCHEMA_VERSION` because the thresholds and the record around
#: them can change independently, and a reader has to know which one moved.
SHADOW_GATE_SCHEMA_VERSION = "m10-shadow-gate-v1"

#: Enforcement is off by construction. Flipping this constant is not enough:
#: :class:`ShadowDecision` independently refuses to hold ``enforced=True``.
ENFORCEMENT_ENABLED = False


class ShadowEnforcementError(RuntimeError):
    """Raised on any attempt to act on a shadow decision."""


class ShadowOutcome(StrEnum):
    """What the harness would have said, had anyone been asking."""

    SHADOW_RECOMMENDED = "SHADOW_RECOMMENDED"
    SHADOW_WITHHELD = "SHADOW_WITHHELD"


#: Reason codes, emitted in this order whenever more than one applies.
RECONCILED_FRACTION_BELOW_GATE = "RECONCILED_FRACTION_BELOW_GATE"
DISAGREEMENT_FRACTION_ABOVE_GATE = "DISAGREEMENT_FRACTION_ABOVE_GATE"
RANKING_CONSISTENCY_UNAVAILABLE = "RANKING_CONSISTENCY_UNAVAILABLE"
RANKING_CONSISTENCY_BELOW_GATE = "RANKING_CONSISTENCY_BELOW_GATE"
CANDIDATE_DOES_NOT_BEAT_BASELINE = "CANDIDATE_DOES_NOT_BEAT_BASELINE"

#: The reason codes that exist, in the order :func:`gate_reasons` emits them. A
#: decision listing anything else, or listing these in another order, did not
#: come from the gate.
SHADOW_REASON_ORDER = (
    RECONCILED_FRACTION_BELOW_GATE,
    DISAGREEMENT_FRACTION_ABOVE_GATE,
    RANKING_CONSISTENCY_UNAVAILABLE,
    RANKING_CONSISTENCY_BELOW_GATE,
    CANDIDATE_DOES_NOT_BEAT_BASELINE,
)


@dataclass(frozen=True, slots=True)
class ShadowGate:
    """Thresholds a candidate must clear before it is even *recommended*.

    An unavailable ranking statistic withholds the recommendation rather than
    being treated as a pass: not knowing whether two rankings agree is not the
    same as knowing that they do.

    The gate is frozen and validates itself on construction, so a gate that
    exists is a gate whose thresholds are finite numbers in range. Every
    :class:`ShadowDecision` embeds the gate that judged it, which is what makes
    the decision re-derivable rather than merely self-consistent.
    """

    policy_version: str = SHADOW_GATE_SCHEMA_VERSION
    minimum_reconciled_fraction: float = 0.99
    maximum_disagreement_fraction: float = 0.01
    minimum_tau_b: float = 0.8

    def __post_init__(self) -> None:
        context = "ShadowGate"
        check(
            self.policy_version == SHADOW_GATE_SCHEMA_VERSION,
            context,
            f"policy_version must be {SHADOW_GATE_SCHEMA_VERSION!r}, "
            f"got {self.policy_version!r}",
        )
        # Types before ranges: `check_unit_interval` would raise TypeError, not
        # MalformedPayloadError, on a threshold that arrived as a string, and a
        # gate assembled in-process does not pass through the payload readers.
        # `bool` is an `int`, so `minimum_tau_b=True` would otherwise become 1.0
        # and silently reject every candidate.
        for name, value in (
            ("minimum_reconciled_fraction", self.minimum_reconciled_fraction),
            ("maximum_disagreement_fraction", self.maximum_disagreement_fraction),
            ("minimum_tau_b", self.minimum_tau_b),
        ):
            check(
                isinstance(value, int | float) and not isinstance(value, bool),
                context,
                f"{name} must be a number, got {type(value).__name__} ({value!r})",
            )

        check_unit_interval(
            self.minimum_reconciled_fraction, "minimum_reconciled_fraction", context
        )
        check_unit_interval(
            self.maximum_disagreement_fraction, "maximum_disagreement_fraction", context
        )
        check_finite(self.minimum_tau_b, "minimum_tau_b", context)
        check(
            -1.0 <= self.minimum_tau_b <= 1.0,
            context,
            f"minimum_tau_b must be within [-1, 1], got {self.minimum_tau_b!r}",
        )

    def decide(
        self,
        *,
        baseline_arm_id: str,
        candidate_arm_id: str,
        reconciled_fraction: float,
        disagreement_fraction: float,
        tau_b: float | None,
        baseline_score: float,
        candidate_score: float,
    ) -> ShadowDecision:
        """Judge one candidate and record the policy that judged it.

        The decision embeds ``self``, not a copy of its thresholds: the gate is
        frozen, so the policy in the record is the policy that was applied, with
        no room for the two to drift apart.
        """
        context = "ShadowGate.decide"
        check_unit_interval(reconciled_fraction, "reconciled_fraction", context)
        check_unit_interval(disagreement_fraction, "disagreement_fraction", context)
        check_finite(tau_b, "tau_b", context)
        check_finite(baseline_score, "baseline_score", context)
        check_finite(candidate_score, "candidate_score", context)

        reasons = gate_reasons(
            self,
            reconciled_fraction=reconciled_fraction,
            disagreement_fraction=disagreement_fraction,
            tau_b=tau_b,
            baseline_score=baseline_score,
            candidate_score=candidate_score,
        )
        return ShadowDecision(
            SHADOW_SCHEMA_VERSION,
            self,
            baseline_arm_id,
            candidate_arm_id,
            outcome_for(reasons),
            False,
            reasons,
            reconciled_fraction,
            disagreement_fraction,
            tau_b,
            baseline_score,
            candidate_score,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "minimum_reconciled_fraction": self.minimum_reconciled_fraction,
            "maximum_disagreement_fraction": self.maximum_disagreement_fraction,
            "minimum_tau_b": self.minimum_tau_b,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> ShadowGate:
        context = "ShadowGate"
        data = require_mapping(payload, context)
        return cls(
            require_str(data, "policy_version", context),
            require_float(data, "minimum_reconciled_fraction", context),
            require_float(data, "maximum_disagreement_fraction", context),
            require_float(data, "minimum_tau_b", context),
        )


#: The gate the published M10 artifacts are judged against. A decision that
#: states a different policy is not wrong -- the gate is deliberately
#: configurable -- but it is not this one, and the runner refuses to publish it.
DEFAULT_SHADOW_GATE = ShadowGate()


def gate_reasons(
    policy: ShadowGate,
    *,
    reconciled_fraction: float,
    disagreement_fraction: float,
    tau_b: float | None,
    baseline_score: float,
    candidate_score: float,
) -> tuple[str, ...]:
    """Return every reason ``policy`` withholds, in :data:`SHADOW_REASON_ORDER`.

    A module-level function rather than a method, called with the same
    arguments by :meth:`ShadowGate.decide` and by
    :meth:`ShadowDecision.__post_init__`. One derivation, two callers: there is
    no second implementation for a record to be checked against, so the check
    cannot drift away from the thing it checks.
    """
    reasons: list[str] = []
    if reconciled_fraction < policy.minimum_reconciled_fraction:
        reasons.append(RECONCILED_FRACTION_BELOW_GATE)
    if disagreement_fraction > policy.maximum_disagreement_fraction:
        reasons.append(DISAGREEMENT_FRACTION_ABOVE_GATE)
    if tau_b is None:
        reasons.append(RANKING_CONSISTENCY_UNAVAILABLE)
    elif tau_b < policy.minimum_tau_b:
        reasons.append(RANKING_CONSISTENCY_BELOW_GATE)
    if candidate_score <= baseline_score:
        reasons.append(CANDIDATE_DOES_NOT_BEAT_BASELINE)
    return tuple(reasons)


def outcome_for(reasons: tuple[str, ...]) -> ShadowOutcome:
    """A recommendation is the absence of any reason to withhold, nothing more."""
    return (
        ShadowOutcome.SHADOW_WITHHELD if reasons else ShadowOutcome.SHADOW_RECOMMENDED
    )


@dataclass(frozen=True, slots=True)
class ShadowDecision:
    """One recorded comparison of a candidate arm against the baseline.

    ``enforced`` exists only so that the field is present in the serialized
    record and can be asserted on; it is always ``False``.

    The decision carries the measurements the gate judged, the gate that judged
    them, and the verdict it reached -- and the verdict is re-derived from the
    other two on construction rather than trusted. Every reason and the outcome
    itself are recomputed by :func:`gate_reasons` and :func:`outcome_for`, the
    same functions :meth:`ShadowGate.decide` uses, and a mismatch is refused. A
    record whose verdict does not follow from its own stated inputs and policy
    cannot be built, whether it came from the gate or from a text editor.

    The narrower checks are kept alongside the re-derivation even though it
    subsumes them, because they name the specific contradiction rather than
    reporting that some reason list differs from some other reason list.
    """

    schema_version: str
    policy: ShadowGate
    baseline_arm_id: str
    candidate_arm_id: str
    outcome: ShadowOutcome
    enforced: bool
    reasons: tuple[str, ...]
    reconciled_fraction: float
    disagreement_fraction: float
    tau_b: float | None
    baseline_score: float
    candidate_score: float

    def __post_init__(self) -> None:
        if self.enforced:
            raise ShadowEnforcementError(
                "a shadow decision is never enforced: M10 records decisions only"
            )

        context = "ShadowDecision"
        check(
            self.schema_version == SHADOW_SCHEMA_VERSION,
            context,
            f"schema_version must be {SHADOW_SCHEMA_VERSION!r}, "
            f"got {self.schema_version!r}",
        )
        # Exact type, not isinstance: a subclass could reintroduce the
        # thresholds as properties and answer the re-derivation below
        # differently from the way it answered `decide`, which is precisely the
        # self-justifying record this check exists to prevent.
        check(
            type(self.policy) is ShadowGate,
            context,
            "policy must be a ShadowGate stating the thresholds this decision "
            f"was judged against, got {type(self.policy).__name__}",
        )
        check(
            isinstance(self.outcome, ShadowOutcome),
            context,
            f"outcome must be a ShadowOutcome, got {type(self.outcome).__name__} "
            f"({self.outcome!r})",
        )
        for name, arm_id in (
            ("baseline_arm_id", self.baseline_arm_id),
            ("candidate_arm_id", self.candidate_arm_id),
        ):
            check(bool(arm_id), context, f"{name} must not be empty")
        check(
            self.baseline_arm_id != self.candidate_arm_id,
            context,
            f"an arm cannot be compared with itself: {self.baseline_arm_id!r}",
        )

        check_ordered_subset(self.reasons, SHADOW_REASON_ORDER, "reasons", context)
        withheld = self.outcome is ShadowOutcome.SHADOW_WITHHELD
        check(
            withheld == bool(self.reasons),
            context,
            f"outcome {self.outcome.value} does not match "
            f"{len(self.reasons)} reason(s): a recommendation must state no "
            "reason to withhold, and a withholding must state at least one",
        )

        check_unit_interval(self.reconciled_fraction, "reconciled_fraction", context)
        check_unit_interval(
            self.disagreement_fraction, "disagreement_fraction", context
        )
        check_finite(self.baseline_score, "baseline_score", context)
        check_finite(self.candidate_score, "candidate_score", context)
        check_finite(self.tau_b, "tau_b", context)
        if self.tau_b is not None:
            check(
                -1.0 <= self.tau_b <= 1.0,
                context,
                f"tau_b must be within [-1, 1], got {self.tau_b!r}",
            )

        check(
            (CANDIDATE_DOES_NOT_BEAT_BASELINE in self.reasons)
            == (self.candidate_score <= self.baseline_score),
            context,
            f"{CANDIDATE_DOES_NOT_BEAT_BASELINE} is stated for exactly the "
            f"decisions where candidate_score <= baseline_score, but the scores "
            f"are {self.candidate_score!r} and {self.baseline_score!r}",
        )
        check(
            (RANKING_CONSISTENCY_UNAVAILABLE in self.reasons) == (self.tau_b is None),
            context,
            f"{RANKING_CONSISTENCY_UNAVAILABLE} is stated for exactly the "
            f"decisions with no tau_b, but tau_b is {self.tau_b!r}",
        )
        check(
            RANKING_CONSISTENCY_BELOW_GATE not in self.reasons
            or self.tau_b is not None,
            context,
            f"{RANKING_CONSISTENCY_BELOW_GATE} requires a tau_b to have been "
            "below the gate, but none was recorded",
        )
        self._check_derived_from_policy(context)

    def _check_derived_from_policy(self, context: str) -> None:
        """Refuse a verdict the stored measurements and policy do not produce."""
        expected = gate_reasons(
            self.policy,
            reconciled_fraction=self.reconciled_fraction,
            disagreement_fraction=self.disagreement_fraction,
            tau_b=self.tau_b,
            baseline_score=self.baseline_score,
            candidate_score=self.candidate_score,
        )
        check(
            self.reasons == expected,
            context,
            f"reasons {list(self.reasons)!r} are not what this decision's own "
            f"policy {self.policy.to_dict()!r} produces from its own "
            f"measurements, which is {list(expected)!r}",
        )
        check(
            self.outcome is outcome_for(expected),
            context,
            f"outcome {self.outcome.value} is not what this decision's own "
            f"policy and measurements produce, which is "
            f"{outcome_for(expected).value}",
        )

    def enforce(self) -> None:
        """Always raise. Present so that calling it is a loud error, not a no-op."""
        raise ShadowEnforcementError(
            "a shadow decision is never enforced: M10 records decisions only"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy": self.policy.to_dict(),
            "baseline_arm_id": self.baseline_arm_id,
            "candidate_arm_id": self.candidate_arm_id,
            "outcome": self.outcome.value,
            "enforced": self.enforced,
            "reasons": list(self.reasons),
            "reconciled_fraction": self.reconciled_fraction,
            "disagreement_fraction": self.disagreement_fraction,
            "tau_b": self.tau_b,
            "baseline_score": self.baseline_score,
            "candidate_score": self.candidate_score,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ShadowDecision:
        """Rebuild a decision, and re-judge it while doing so.

        ``enforced`` is required to be a real boolean rather than anything
        truthy: a payload carrying ``0`` or ``"false"`` must not be able to
        decide by coercion whether :meth:`__post_init__` refuses it.

        Nothing here re-checks the verdict; the constructor does that. A payload
        whose reasons or outcome do not follow from its own policy and
        measurements is rejected by exactly the rule that governs a freshly
        computed decision, because it is the same rule.
        """
        context = "ShadowDecision"
        data = require_mapping(payload, context)
        return cls(
            require_str(data, "schema_version", context),
            ShadowGate.from_dict(require_object(data, "policy", context)),
            require_str(data, "baseline_arm_id", context),
            require_str(data, "candidate_arm_id", context),
            ShadowOutcome(require_str(data, "outcome", context)),
            require_bool(data, "enforced", context),
            require_str_sequence(data, "reasons", context),
            require_float(data, "reconciled_fraction", context),
            require_float(data, "disagreement_fraction", context),
            optional_float(data, "tau_b", context),
            require_float(data, "baseline_score", context),
            require_float(data, "candidate_score", context),
        )
