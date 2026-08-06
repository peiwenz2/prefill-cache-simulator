"""What must be true before a replay may call itself ``HW_VALIDATED``.

:mod:`.shadow` already gates a decision, but it gates a different question. Its
:class:`~.shadow.ShadowGate` asks *did the candidate beat the baseline*; this
module asks *is this run entitled to say its answer came from hardware*. The two
are orthogonal by construction: a replay driven by a real engine can be
``HW_VALIDATED`` and still ``SHADOW_WITHHELD`` because the candidate lost, and a
replay that produced a clean win on a mock is neither. Collapsing them would let
a strong result on a simulator borrow the authority of a measurement.

The prerequisites are members of :class:`ReplayBlocker`, a string enum that
survives serialization, so an environment with no accelerator publishes
:attr:`ReplayBlocker.BLOCKED_NO_ENGINE_ACCESS` as data rather than leaving a
comment. That artifact is the deliverable.

Two of the blockers are worth naming separately from the rest, because they
guard against a failure that no threshold catches:

:attr:`ReplayBlocker.BLOCKED_PLAN_NOT_FROZEN`
    The plan is hashed, not described. A run that widened the arrival scales or
    dropped the stop-gated arm after seeing a first result would otherwise
    report the same schema version as one that did not.
:attr:`ReplayBlocker.BLOCKED_FAULT_INJECTION_NOT_DETECTED`
    A reconciler that finds no disagreements because it cannot see any looks
    exactly like one that finds none because there are none. The fault harness
    in :mod:`.faults` is what tells those apart, so a hardware claim requires
    that it was run and that it caught what it planted.

Unlike :mod:`..calibration.hardware`, this module validates with the
:mod:`.payload` helpers. That is the local convention on this side of the
dependency edge -- :mod:`..replay` imports :mod:`..calibration`, so the
calibration package cannot use them, but every module here already does.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ..calibration import (
    CalibrationStatus,
    DishonestLabelError,
    EvidenceTier,
    MachineProvenance,
    TimeUnit,
    assert_labels_supported,
)
from .orchestrator import ReplayPlan
from .payload import (
    check,
    check_finite,
    check_ordered_subset,
    check_unit_interval,
    invalid,
    optional_float,
    optional_str,
    require_bool,
    require_float,
    require_mapping,
    require_str,
    require_str_sequence,
)
from .ranking import FROZEN_RANKING_STATISTIC

__all__ = [
    "DEFAULT_REPLAY_HARDWARE_GATE",
    "FROZEN_PLAN_DIGEST",
    "REPLAY_BLOCKER_ORDER",
    "REPLAY_HARDWARE_GATE_SCHEMA_VERSION",
    "REPLAY_HARDWARE_SCHEMA_VERSION",
    "REQUIRED_CALIBRATION_TIER",
    "ReplayBlocker",
    "ReplayHardwareEvidence",
    "ReplayHardwareGate",
    "ReplayHardwareReport",
    "plan_digest",
    "replay_blockers",
]

REPLAY_HARDWARE_SCHEMA_VERSION = "m10-hardware-v1"
REPLAY_HARDWARE_GATE_SCHEMA_VERSION = "m10-hardware-gate-v1"

#: The calibration label an M9-HW artifact must carry before this replay may
#: build on it. Written as the enum's value rather than the member so that a
#: payload read from disk is compared against it without first being coerced
#: into an enum that would raise on an unrecognized string.
REQUIRED_CALIBRATION_TIER = CalibrationStatus.HW_CALIBRATED.value


class ReplayBlocker(StrEnum):
    """A prerequisite for ``HW_VALIDATED`` that this run has not met."""

    BLOCKED_NO_ENGINE_ACCESS = "BLOCKED_NO_ENGINE_ACCESS"
    BLOCKED_SYNTHETIC_CALIBRATION = "BLOCKED_SYNTHETIC_CALIBRATION"
    BLOCKED_CALIBRATION_NOT_ACCEPTED = "BLOCKED_CALIBRATION_NOT_ACCEPTED"
    BLOCKED_INCOMPLETE_PROVENANCE = "BLOCKED_INCOMPLETE_PROVENANCE"
    BLOCKED_PLAN_NOT_FROZEN = "BLOCKED_PLAN_NOT_FROZEN"
    BLOCKED_STATISTIC_NOT_FROZEN = "BLOCKED_STATISTIC_NOT_FROZEN"
    BLOCKED_NON_PRODUCTION_POLICY = "BLOCKED_NON_PRODUCTION_POLICY"
    BLOCKED_RANKING_UNAVAILABLE = "BLOCKED_RANKING_UNAVAILABLE"
    BLOCKED_RANKING_BELOW_GATE = "BLOCKED_RANKING_BELOW_GATE"
    BLOCKED_RECONCILED_FRACTION_BELOW_GATE = "BLOCKED_RECONCILED_FRACTION_BELOW_GATE"
    BLOCKED_DISAGREEMENT_ABOVE_GATE = "BLOCKED_DISAGREEMENT_ABOVE_GATE"
    BLOCKED_FAULT_INJECTION_NOT_DETECTED = "BLOCKED_FAULT_INJECTION_NOT_DETECTED"
    BLOCKED_DRY_RUN_ONLY = "BLOCKED_DRY_RUN_ONLY"


#: Emission order, cheapest and most upstream first. A run with no engine should
#: read ``BLOCKED_NO_ENGINE_ACCESS`` before it reads anything about tau-b, since
#: the latter is a statement about numbers that were never measured.
REPLAY_BLOCKER_ORDER: tuple[ReplayBlocker, ...] = (
    ReplayBlocker.BLOCKED_NO_ENGINE_ACCESS,
    ReplayBlocker.BLOCKED_SYNTHETIC_CALIBRATION,
    ReplayBlocker.BLOCKED_CALIBRATION_NOT_ACCEPTED,
    ReplayBlocker.BLOCKED_INCOMPLETE_PROVENANCE,
    ReplayBlocker.BLOCKED_PLAN_NOT_FROZEN,
    ReplayBlocker.BLOCKED_STATISTIC_NOT_FROZEN,
    ReplayBlocker.BLOCKED_NON_PRODUCTION_POLICY,
    ReplayBlocker.BLOCKED_RANKING_UNAVAILABLE,
    ReplayBlocker.BLOCKED_RANKING_BELOW_GATE,
    ReplayBlocker.BLOCKED_RECONCILED_FRACTION_BELOW_GATE,
    ReplayBlocker.BLOCKED_DISAGREEMENT_ABOVE_GATE,
    ReplayBlocker.BLOCKED_FAULT_INJECTION_NOT_DETECTED,
    ReplayBlocker.BLOCKED_DRY_RUN_ONLY,
)


def plan_digest(plan: ReplayPlan) -> str:
    """SHA-256 over the plan's serialized form.

    ``ReplayPlan.to_dict`` covers every field, and ``sort_keys`` removes the one
    source of ordering nondeterminism, so two processes that hold equal plans
    produce equal digests. Hashing the serialized form rather than a chosen
    subset means a field added to the plan later is covered without anyone
    remembering to add it here.
    """
    if type(plan) is not ReplayPlan:
        raise invalid(
            "plan_digest", f"expected a ReplayPlan, got {type(plan).__name__}"
        )
    encoded = json.dumps(plan.to_dict(), sort_keys=True, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


#: The digest of the default plan, pinned as a reviewed literal. A change to
#: ``ReplayPlan`` that alters the serialized form must update this literal and
#: be reviewed; the assertion below catches silent drift at import time so a
#: stale digest cannot pass through undetected.
FROZEN_PLAN_DIGEST = "e3f13a2e150ee6563d433b7de2e431d55e5effda7b9254c114cacf94d9c3bbd4"
assert plan_digest(ReplayPlan()) == FROZEN_PLAN_DIGEST, (
    "FROZEN_PLAN_DIGEST has drifted: ReplayPlan defaults changed but the "
    "literal was not updated. Update the literal after review."
)


@dataclass(frozen=True, slots=True)
class ReplayHardwareEvidence:
    """What a replay run observed about its own inputs and outputs.

    Every field defaults to the answer an environment with no accelerator would
    give, so :class:`ReplayHardwareEvidence` constructed with no arguments is a
    truthful description of this repository rather than a blank to be filled in.
    """

    engine_reachable: bool = False
    calibration_status: str | None = None
    calibration_accepted: bool = False
    calibration_endpoint_synthetic: bool = True
    provenance_complete: bool = False
    observed_plan_digest: str | None = None
    ranking_statistic: str | None = None
    tau_b: float | None = None
    reconciled_fraction: float | None = None
    disagreement_fraction: float | None = None
    fault_injection_detected: bool = False
    dry_run: bool = False

    def __post_init__(self) -> None:
        context = "ReplayHardwareEvidence"
        for name in (
            "engine_reachable",
            "calibration_accepted",
            "calibration_endpoint_synthetic",
            "provenance_complete",
            "fault_injection_detected",
            "dry_run",
        ):
            value = getattr(self, name)
            check(
                isinstance(value, bool),
                context,
                f"{name} must be a bool, got {type(value).__name__}",
            )
        for name in (
            "calibration_status",
            "observed_plan_digest",
            "ranking_statistic",
        ):
            value = getattr(self, name)
            check(
                value is None or isinstance(value, str),
                context,
                f"{name} must be a string or None, got {type(value).__name__}",
            )
        # `tau_b` is checked for finiteness but not for range: the gate compares
        # it against a threshold, and a value outside [-1, 1] is a defect in
        # whoever computed it rather than a verdict this module should launder
        # into "below gate".
        check_finite(self.tau_b, "tau_b", context)
        for name in ("reconciled_fraction", "disagreement_fraction"):
            value = getattr(self, name)
            if value is not None:
                check_unit_interval(value, name, context)

    @classmethod
    def blocked(cls) -> ReplayHardwareEvidence:
        """The evidence available in an environment with no engine."""
        return cls()

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_reachable": self.engine_reachable,
            "calibration_status": self.calibration_status,
            "calibration_accepted": self.calibration_accepted,
            "calibration_endpoint_synthetic": self.calibration_endpoint_synthetic,
            "provenance_complete": self.provenance_complete,
            "observed_plan_digest": self.observed_plan_digest,
            "ranking_statistic": self.ranking_statistic,
            "tau_b": self.tau_b,
            "reconciled_fraction": self.reconciled_fraction,
            "disagreement_fraction": self.disagreement_fraction,
            "fault_injection_detected": self.fault_injection_detected,
            "dry_run": self.dry_run,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ReplayHardwareEvidence:
        context = "ReplayHardwareEvidence"
        data = require_mapping(payload, context)
        return cls(
            require_bool(data, "engine_reachable", context),
            optional_str(data, "calibration_status", context),
            require_bool(data, "calibration_accepted", context),
            require_bool(data, "calibration_endpoint_synthetic", context),
            require_bool(data, "provenance_complete", context),
            optional_str(data, "observed_plan_digest", context),
            optional_str(data, "ranking_statistic", context),
            optional_float(data, "tau_b", context),
            optional_float(data, "reconciled_fraction", context),
            optional_float(data, "disagreement_fraction", context),
            require_bool(data, "fault_injection_detected", context),
            require_bool(data, "dry_run", context),
        )


@dataclass(frozen=True, slots=True)
class ReplayHardwareGate:
    """The thresholds an M10-HW run is judged against.

    ``minimum_tau_b`` is stricter than the shadow gate's, and deliberately so.
    The shadow gate is deciding whether to *recommend*, which is reversible;
    this one is deciding whether a number may be described as measured, which is
    not. ``required_plan_digest`` defaults to the digest of the default plan
    rather than to ``None``, so "any plan is acceptable" has to be stated.
    """

    policy_version: str = REPLAY_HARDWARE_GATE_SCHEMA_VERSION
    required_calibration_status: str = REQUIRED_CALIBRATION_TIER
    required_statistic: str = FROZEN_RANKING_STATISTIC
    required_plan_digest: str = FROZEN_PLAN_DIGEST
    minimum_tau_b: float = 0.9
    minimum_reconciled_fraction: float = 0.99
    maximum_disagreement_fraction: float = 0.01
    production: bool = True

    def __post_init__(self) -> None:
        context = "ReplayHardwareGate"
        check(
            self.policy_version == REPLAY_HARDWARE_GATE_SCHEMA_VERSION,
            context,
            f"policy_version must be {REPLAY_HARDWARE_GATE_SCHEMA_VERSION!r}, "
            f"got {self.policy_version!r}",
        )
        for name in (
            "required_calibration_status",
            "required_statistic",
            "required_plan_digest",
        ):
            value = getattr(self, name)
            check(
                isinstance(value, str) and bool(value),
                context,
                f"{name} must be a non-empty string, got {value!r}",
            )
        # Types before ranges: `True < 0.9` is False in Python, so a bool would
        # otherwise pass the threshold check and become a policy.
        for name in (
            "minimum_tau_b",
            "minimum_reconciled_fraction",
            "maximum_disagreement_fraction",
        ):
            value = getattr(self, name)
            check(
                not isinstance(value, bool) and isinstance(value, int | float),
                context,
                f"{name} must be a number, got {type(value).__name__}",
            )
            check_finite(float(value), name, context)
        check(
            -1.0 <= self.minimum_tau_b <= 1.0,
            context,
            f"minimum_tau_b must be within [-1, 1], got {self.minimum_tau_b!r}",
        )
        check_unit_interval(
            self.minimum_reconciled_fraction, "minimum_reconciled_fraction", context
        )
        check_unit_interval(
            self.maximum_disagreement_fraction,
            "maximum_disagreement_fraction",
            context,
        )
        check(
            isinstance(self.production, bool),
            context,
            f"production must be a bool, got {type(self.production).__name__}",
        )

    def evaluate(
        self,
        evidence: ReplayHardwareEvidence,
        machine: MachineProvenance,
    ) -> ReplayHardwareReport:
        """Judge one run and return the verdict as a record."""
        blockers = replay_blockers(self, evidence)
        accepted = not blockers
        status, unit, tier = _labels_for(blockers)
        return ReplayHardwareReport(
            REPLAY_HARDWARE_SCHEMA_VERSION,
            self,
            evidence,
            machine,
            blockers,
            accepted,
            status,
            unit,
            tier,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "required_calibration_status": self.required_calibration_status,
            "required_statistic": self.required_statistic,
            "required_plan_digest": self.required_plan_digest,
            "minimum_tau_b": self.minimum_tau_b,
            "minimum_reconciled_fraction": self.minimum_reconciled_fraction,
            "maximum_disagreement_fraction": self.maximum_disagreement_fraction,
            "production": self.production,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ReplayHardwareGate:
        context = "ReplayHardwareGate"
        data = require_mapping(payload, context)
        return cls(
            require_str(data, "policy_version", context),
            require_str(data, "required_calibration_status", context),
            require_str(data, "required_statistic", context),
            require_str(data, "required_plan_digest", context),
            require_float(data, "minimum_tau_b", context),
            require_float(data, "minimum_reconciled_fraction", context),
            require_float(data, "maximum_disagreement_fraction", context),
            require_bool(data, "production", context),
        )


DEFAULT_REPLAY_HARDWARE_GATE = ReplayHardwareGate()


def replay_blockers(
    policy: ReplayHardwareGate, evidence: ReplayHardwareEvidence
) -> tuple[ReplayBlocker, ...]:
    """Every reason ``evidence`` may not be labelled ``HW_VALIDATED``.

    One derivation, two callers: :meth:`ReplayHardwareGate.evaluate` uses it to
    reach a verdict and :class:`ReplayHardwareReport` uses it to check one. There
    is no second implementation for a record to be compared against, so the
    check cannot drift away from the thing it checks.

    All blockers are reported, not the first. "No engine" and "no engine, and
    the fault harness never ran either" describe different amounts of remaining
    work, and a runner that fixed one at a time would learn that one at a time.
    """
    if type(policy) is not ReplayHardwareGate:
        raise invalid(
            "replay_blockers",
            f"policy must be a ReplayHardwareGate, got {type(policy).__name__}",
        )
    if type(evidence) is not ReplayHardwareEvidence:
        raise invalid(
            "replay_blockers",
            f"evidence must be a ReplayHardwareEvidence, got {type(evidence).__name__}",
        )

    found: list[ReplayBlocker] = []
    if not evidence.engine_reachable:
        found.append(ReplayBlocker.BLOCKED_NO_ENGINE_ACCESS)
    if evidence.calibration_endpoint_synthetic:
        found.append(ReplayBlocker.BLOCKED_SYNTHETIC_CALIBRATION)
    # Two conditions, one blocker: a calibration that was rejected and one that
    # was accepted under the wrong label are the same defect from here, namely
    # that there is no accepted hardware calibration to build on.
    if (
        not evidence.calibration_accepted
        or evidence.calibration_status != policy.required_calibration_status
    ):
        found.append(ReplayBlocker.BLOCKED_CALIBRATION_NOT_ACCEPTED)
    if not evidence.provenance_complete:
        found.append(ReplayBlocker.BLOCKED_INCOMPLETE_PROVENANCE)
    if evidence.observed_plan_digest != policy.required_plan_digest:
        found.append(ReplayBlocker.BLOCKED_PLAN_NOT_FROZEN)
    if evidence.ranking_statistic != policy.required_statistic:
        found.append(ReplayBlocker.BLOCKED_STATISTIC_NOT_FROZEN)

    if not policy.production:
        found.append(ReplayBlocker.BLOCKED_NON_PRODUCTION_POLICY)

    # Unavailable and below-gate are separate members, following the
    # `RANKING_CONSISTENCY_UNAVAILABLE` / `RANKING_CONSISTENCY_BELOW_GATE`
    # split in `.shadow`: a run whose arms could not be ranked has produced no
    # ranking evidence, which is not the same finding as a ranking that
    # disagreed.
    if evidence.tau_b is None:
        found.append(ReplayBlocker.BLOCKED_RANKING_UNAVAILABLE)
    elif evidence.tau_b < policy.minimum_tau_b:
        found.append(ReplayBlocker.BLOCKED_RANKING_BELOW_GATE)

    # A missing fraction is charged as below/above gate rather than as its own
    # member. Unlike tau-b, which is genuinely undefined when a side is fully
    # tied, reconciliation always has an answer; absence here means the
    # reconciler was never run, and that is a failure to meet the gate.
    if (
        evidence.reconciled_fraction is None
        or evidence.reconciled_fraction < policy.minimum_reconciled_fraction
    ):
        found.append(ReplayBlocker.BLOCKED_RECONCILED_FRACTION_BELOW_GATE)
    if (
        evidence.disagreement_fraction is None
        or evidence.disagreement_fraction > policy.maximum_disagreement_fraction
    ):
        found.append(ReplayBlocker.BLOCKED_DISAGREEMENT_ABOVE_GATE)

    if not evidence.fault_injection_detected:
        found.append(ReplayBlocker.BLOCKED_FAULT_INJECTION_NOT_DETECTED)
    if evidence.dry_run:
        found.append(ReplayBlocker.BLOCKED_DRY_RUN_ONLY)

    order = {blocker: index for index, blocker in enumerate(REPLAY_BLOCKER_ORDER)}
    return tuple(sorted(found, key=lambda blocker: order[blocker]))


def _labels_for(
    blockers: tuple[ReplayBlocker, ...],
) -> tuple[CalibrationStatus, TimeUnit, EvidenceTier]:
    """The only label tuple a report with this verdict may carry.

    Three outcomes, not two:

    * No blockers: the replay is ``HW_VALIDATED``.
    * ``BLOCKED_NON_PRODUCTION_POLICY`` present: the thresholds were overridden
      on the CLI, so the run is a non-production experiment. It may have measured
      a real engine, but the policy was not the reviewed one, so it publishes as
      ``NON_PRODUCTION_EXPERIMENT`` and may never claim ``HW_VALIDATED``.
    * Anything else: the arms really were run and ranked, just not against
      hardware, so ``SYNTHETIC_REPLAY`` rather than ``HARNESS_ONLY``.

    Downgrading a non-production experiment further would understate the
    evidence as badly as ``HW_VALIDATED`` overstates it.
    """
    if not blockers:
        return (
            CalibrationStatus.HW_CALIBRATED,
            TimeUnit.MILLISECONDS,
            EvidenceTier.HW_VALIDATED,
        )
    if ReplayBlocker.BLOCKED_NON_PRODUCTION_POLICY in blockers:
        return (
            CalibrationStatus.NON_PRODUCTION_EXPERIMENT,
            TimeUnit.NORMALIZED_WORK,
            EvidenceTier.SYNTHETIC_REPLAY,
        )
    return (
        CalibrationStatus.SYNTHETIC_UNCALIBRATED,
        TimeUnit.NORMALIZED_WORK,
        EvidenceTier.SYNTHETIC_REPLAY,
    )


@dataclass(frozen=True, slots=True)
class ReplayHardwareReport:
    """One M10-HW verdict, carrying the evidence and policy that produced it.

    The policy is stored rather than assumed. That does let a forger state a
    permissive one -- but it has to state it, in the artifact, where a reader
    comparing it against :data:`DEFAULT_REPLAY_HARDWARE_GATE` can see the
    substitution. A report that named no policy would make the same forgery
    invisible.
    """

    schema_version: str
    policy: ReplayHardwareGate
    evidence: ReplayHardwareEvidence
    machine: MachineProvenance
    blockers: tuple[ReplayBlocker, ...]
    accepted: bool
    calibration_status: CalibrationStatus
    time_unit: TimeUnit
    evidence_tier: EvidenceTier

    def __post_init__(self) -> None:
        context = "ReplayHardwareReport"
        check(
            self.schema_version == REPLAY_HARDWARE_SCHEMA_VERSION,
            context,
            f"schema_version must be {REPLAY_HARDWARE_SCHEMA_VERSION!r}, "
            f"got {self.schema_version!r}",
        )
        # Exact type, not isinstance: a subclass could reintroduce the
        # thresholds as properties and answer the re-derivation below
        # differently from the way it answered `evaluate`.
        check(
            type(self.policy) is ReplayHardwareGate,
            context,
            "policy must be a ReplayHardwareGate stating the thresholds this "
            f"report was judged against, got {type(self.policy).__name__}",
        )
        check(
            type(self.evidence) is ReplayHardwareEvidence,
            context,
            f"evidence must be a ReplayHardwareEvidence, got "
            f"{type(self.evidence).__name__}",
        )
        check(
            type(self.machine) is MachineProvenance,
            context,
            f"machine must be a MachineProvenance, got {type(self.machine).__name__}",
        )
        check(
            isinstance(self.accepted, bool),
            context,
            f"accepted must be a bool, got {type(self.accepted).__name__}",
        )
        check_ordered_subset(self.blockers, REPLAY_BLOCKER_ORDER, "blockers", context)

        # `provenance_complete` is evidence, and `machine` is the thing it is
        # evidence about. Checking them against each other closes the gap where
        # a report claims complete provenance while carrying an empty record.
        check(
            self.evidence.provenance_complete == self.machine.complete,
            context,
            f"evidence says provenance_complete="
            f"{self.evidence.provenance_complete!r} but the attached machine "
            f"record is {'complete' if self.machine.complete else 'incomplete'}",
        )

        expected = replay_blockers(self.policy, self.evidence)
        if tuple(self.blockers) != expected:
            raise DishonestLabelError(
                f"blockers {[item.value for item in self.blockers]!r} are not "
                "what this report's own policy and evidence produce, which is "
                f"{[item.value for item in expected]!r}"
            )
        if self.accepted != (not expected):
            raise DishonestLabelError(
                f"accepted={self.accepted!r} contradicts {len(expected)} "
                "blocker(s): a replay is accepted exactly when nothing blocks it"
            )

        labels = (self.calibration_status, self.time_unit, self.evidence_tier)
        allowed = _labels_for(expected)
        if labels != allowed:
            raise DishonestLabelError(
                f"a report with accepted={self.accepted!r} must be labelled "
                f"{tuple(item.value for item in allowed)!r}, got "
                f"{tuple(item.value for item in labels)!r}"
            )
        # Not redundant with the check above: this is the same call every other
        # artifact in the repository passes through, so this record cannot
        # become the one path into the artifact set that skips it.
        assert_labels_supported(
            self.machine,
            self.calibration_status,
            self.time_unit,
            self.evidence_tier,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy": self.policy.to_dict(),
            "evidence": self.evidence.to_dict(),
            "machine": self.machine.to_dict(),
            "blockers": [item.value for item in self.blockers],
            "accepted": self.accepted,
            "calibration_status": self.calibration_status.value,
            "time_unit": self.time_unit.value,
            "evidence_tier": self.evidence_tier.value,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ReplayHardwareReport:
        """Rebuild a report, and re-judge it while doing so.

        Nothing here re-checks the verdict; the constructor does that, using the
        same rule that governs a freshly evaluated report because it is the same
        rule. A hand-edited ``accepted`` therefore fails on the way in.
        """
        context = "ReplayHardwareReport"
        data = require_mapping(payload, context)
        blockers = require_str_sequence(data, "blockers", context)
        try:
            parsed = tuple(ReplayBlocker(item) for item in blockers)
        except ValueError as error:
            raise invalid(context, f"unknown blocker: {error}") from error
        return cls(
            require_str(data, "schema_version", context),
            ReplayHardwareGate.from_dict(data["policy"]),
            ReplayHardwareEvidence.from_dict(data["evidence"]),
            MachineProvenance.from_dict(data["machine"]),
            parsed,
            require_bool(data, "accepted", context),
            CalibrationStatus(require_str(data, "calibration_status", context)),
            TimeUnit(require_str(data, "time_unit", context)),
            EvidenceTier(require_str(data, "evidence_tier", context)),
        )
