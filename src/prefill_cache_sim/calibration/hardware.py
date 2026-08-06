"""What must be true before a calibration may call itself ``HW_CALIBRATED``.

:mod:`.provenance` answers "does this artifact's label match its provenance?".
This module answers the earlier question: "did a hardware run actually happen,
and did it produce a fit worth publishing?". The two are separate because a
complete :class:`~.provenance.MachineProvenance` is necessary but nowhere near
sufficient -- a hand-written provenance block attached to three points measured
against a mock would satisfy it exactly.

Every prerequisite is therefore a member of :class:`HardwareBlocker`, a string
enum that survives serialization. A run that cannot proceed publishes its
blockers as data; it does not print a warning and exit zero, and it does not
leave a TODO in a document. In this repository, with no accelerator reachable,
the honest output is an artifact stating
:attr:`HardwareBlocker.BLOCKED_NO_ENGINE_ACCESS`, and that artifact is a
deliverable rather than a failure.

The gate is deliberately ordered from cheapest to most specific, and reports
*all* blockers rather than the first, because "no engine, and also the model
version was never recorded" and "no engine" describe different amounts of
remaining work.

Validation here uses plain :class:`ValueError` and
:class:`~.provenance.DishonestLabelError` rather than the richer
:mod:`..replay.payload` helpers. That is not duplication by oversight:
:mod:`..replay` imports :mod:`..calibration`, so the reverse import would be a
cycle, and the surrounding modules (:mod:`.fit`, :mod:`.provenance`,
:mod:`.endpoint`) already validate this way.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .endpoint import SweepObservation
from .fit import LinearFit
from .provenance import (
    CalibrationStatus,
    DishonestLabelError,
    EvidenceTier,
    MachineProvenance,
    TimeUnit,
    assert_labels_supported,
)

__all__ = [
    "DEFAULT_HARDWARE_GATE",
    "ENDPOINT_CONTRACT_VERSION",
    "HARDWARE_BLOCKER_ORDER",
    "HARDWARE_GATE_SCHEMA_VERSION",
    "HARDWARE_SCHEMA_VERSION",
    "HardwareBlocker",
    "HardwareContext",
    "HardwareEvidence",
    "HardwareGate",
    "HardwareGateReport",
    "TrustPolicy",
    "hardware_blockers",
    "residual_ratio",
]

#: The request/response contract :mod:`.http_endpoint` speaks. It lives here
#: rather than beside the HTTP adapter because it is a property of the
#: *agreement*, not of the wire: a future gRPC adapter honouring the same
#: request and response shapes reports the same version, and the gate compares
#: against this constant without knowing which transport answered.
ENDPOINT_CONTRACT_VERSION = "engine-endpoint-v1"

HARDWARE_SCHEMA_VERSION = "m9-hardware-v1"

#: Versioned separately from the record around it: the thresholds and the shape
#: of the report can move independently, and a reader has to know which did.
HARDWARE_GATE_SCHEMA_VERSION = "m9-hardware-gate-v1"

#: Fields that describe the run configuration rather than the machine. They sit
#: alongside :class:`~.provenance.MachineProvenance` instead of inside it
#: because that class is already serialized into existing artifacts and drives
#: existing label checks; extending it would silently change the meaning of
#: ``complete`` for every artifact ever written.
_CONTEXT_FIELDS: tuple[str, ...] = (
    "endpoint_contract_version",
    "model_version",
    "topology",
    "clock_settings",
)


class HardwareBlocker(StrEnum):
    """A machine-readable reason a hardware calibration may not be claimed.

    These are *not* error messages. They are published in artifacts and asserted
    on in tests, so a consumer can distinguish "this environment has no
    accelerator" from "the fit was too noisy" without parsing prose.
    """

    #: No engine was reachable. This is the state of the present repository.
    BLOCKED_NO_ENGINE_ACCESS = "BLOCKED_NO_ENGINE_ACCESS"
    #: The endpoint identified itself as a mock, stub, or simulator.
    BLOCKED_SYNTHETIC_ENDPOINT = "BLOCKED_SYNTHETIC_ENDPOINT"
    #: The endpoint is real HTTP but not production-trusted (plain HTTP, or host
    #: not in the allowlist). It may still be measured, but cannot earn
    #: HW_CALIBRATED: it publishes as REMOTE_REPORTED instead.
    BLOCKED_NOT_PRODUCTION_TRUSTED = "BLOCKED_NOT_PRODUCTION_TRUSTED"
    #: The gate thresholds were overridden on the CLI. The run is a non-production
    #: experiment whose labels must never say HW_CALIBRATED or HW_VALIDATED.
    BLOCKED_NON_PRODUCTION_POLICY = "BLOCKED_NON_PRODUCTION_POLICY"
    #: Some part of the machine or run configuration was never recorded.
    BLOCKED_INCOMPLETE_PROVENANCE = "BLOCKED_INCOMPLETE_PROVENANCE"
    #: The endpoint speaks a contract this harness has not agreed to.
    BLOCKED_CONTRACT_VERSION_MISMATCH = "BLOCKED_CONTRACT_VERSION_MISMATCH"
    #: Too few points survived to support a fit at the configured confidence.
    BLOCKED_INSUFFICIENT_OBSERVATIONS = "BLOCKED_INSUFFICIENT_OBSERVATIONS"
    #: No fit was produced at all, so its quality is unknown rather than poor.
    BLOCKED_FIT_UNAVAILABLE = "BLOCKED_FIT_UNAVAILABLE"
    #: A fit exists but its residuals are too large a fraction of the signal.
    BLOCKED_RESIDUAL_ABOVE_GATE = "BLOCKED_RESIDUAL_ABOVE_GATE"
    #: The operator asked for a handshake only, so nothing was measured.
    BLOCKED_DRY_RUN_ONLY = "BLOCKED_DRY_RUN_ONLY"


#: The order :func:`hardware_blockers` emits in. A report listing anything else,
#: or listing these out of order, did not come from the gate.
HARDWARE_BLOCKER_ORDER: tuple[HardwareBlocker, ...] = (
    HardwareBlocker.BLOCKED_NO_ENGINE_ACCESS,
    HardwareBlocker.BLOCKED_SYNTHETIC_ENDPOINT,
    HardwareBlocker.BLOCKED_NOT_PRODUCTION_TRUSTED,
    HardwareBlocker.BLOCKED_NON_PRODUCTION_POLICY,
    HardwareBlocker.BLOCKED_INCOMPLETE_PROVENANCE,
    HardwareBlocker.BLOCKED_CONTRACT_VERSION_MISMATCH,
    HardwareBlocker.BLOCKED_INSUFFICIENT_OBSERVATIONS,
    HardwareBlocker.BLOCKED_FIT_UNAVAILABLE,
    HardwareBlocker.BLOCKED_RESIDUAL_ABOVE_GATE,
    HardwareBlocker.BLOCKED_DRY_RUN_ONLY,
)


@dataclass(frozen=True, slots=True)
class TrustPolicy:
    """Which endpoints may issue hardware-calibration labels.

    An endpoint is production-trusted only when it speaks HTTPS *and* its host
    appears in the allowlist. Plain HTTP, or an HTTPS host not in the list, may
    still be measured for validation; the run publishes as
    :attr:`~.provenance.EvidenceTier.REMOTE_REPORTED`, a machine-readable
    non-production tier that cannot issue ``HW_CALIBRATED`` or ``HW_VALIDATED``.

    Local and test HTTP is validate-only: it can exercise the plumbing
    (handshake, gate logic, artifact write) but never earn a hardware label.
    """

    trusted_hosts: frozenset[str] = frozenset()

    def is_production_trusted(self, base_url: str) -> bool:
        """True only for HTTPS URLs whose host is in the allowlist."""
        if "://" not in base_url:
            return False
        scheme, rest = base_url.split("://", 1)
        host = rest.split("/", 1)[0].split(":", 1)[0].lower()
        return scheme.lower() == "https" and host in self.trusted_hosts


def _require_identifier(name: str, value: Any) -> None:
    """Accept ``None`` or a non-blank string, and nothing else.

    The same rule :class:`~.provenance.MachineProvenance` applies to its own
    fields, for the same reason: ``""`` and ``"   "`` are not identifiers, but
    they are not ``None`` either, so an unchecked blank would satisfy
    :attr:`HardwareContext.complete` while naming nothing.
    """
    if value is None:
        return
    if not isinstance(value, str):
        raise DishonestLabelError(
            f"hardware context {name} must be None or a string, "
            f"got {type(value).__name__}"
        )
    if not value.strip():
        raise DishonestLabelError(
            f"hardware context {name} must be None or a non-blank identifier"
        )


@dataclass(frozen=True, slots=True)
class HardwareContext:
    """Everything a reader needs to know a calibration is reproducible.

    :class:`~.provenance.MachineProvenance` names the host, the accelerator, the
    engine build, and the instant. That is enough to reject a mock, but not
    enough to reproduce a number: the same engine on the same card produces
    different milliseconds at a different tensor-parallel width, a different
    model, or a different clock cap. Those four facts live here.

    ``complete`` is the conjunction of both halves, and it is the only thing
    that licenses a hardware label. Every field is nullable so that an honest
    partial record can be published and its gaps read off, rather than the run
    having to invent values in order to serialize at all.
    """

    machine: MachineProvenance
    endpoint_contract_version: str | None
    model_version: str | None
    topology: str | None
    clock_settings: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.machine, MachineProvenance):
            raise DishonestLabelError(
                "hardware context machine must be a MachineProvenance, "
                f"got {type(self.machine).__name__}"
            )
        for name in _CONTEXT_FIELDS:
            _require_identifier(name, getattr(self, name))

    @classmethod
    def unknown(cls) -> HardwareContext:
        """A context that names nothing, for runs with no engine to ask."""
        return cls(MachineProvenance.unknown(), None, None, None, None)

    @property
    def complete(self) -> bool:
        return self.machine.complete and all(
            getattr(self, name) is not None for name in _CONTEXT_FIELDS
        )

    def missing_fields(self) -> tuple[str, ...]:
        """Dotted names of every unrecorded field, for the blocker message.

        Returned rather than formatted so a caller can assert on the set. A
        report that says "provenance incomplete" and nothing else tells an
        operator to go looking; this tells them what to go and record.
        """
        missing = [
            f"machine.{name}"
            for name in ("host_id", "accelerator_model", "engine_version")
            if getattr(self.machine, name) is None
        ]
        if self.machine.captured_at_utc is None:
            missing.append("machine.captured_at_utc")
        missing.extend(name for name in _CONTEXT_FIELDS if getattr(self, name) is None)
        return tuple(missing)

    def to_dict(self) -> dict[str, Any]:
        return {
            "machine": self.machine.to_dict(),
            "endpoint_contract_version": self.endpoint_contract_version,
            "model_version": self.model_version,
            "topology": self.topology,
            "clock_settings": self.clock_settings,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HardwareContext:
        if not isinstance(payload, dict):
            raise DishonestLabelError(
                f"hardware context must be an object, got {type(payload).__name__}"
            )
        machine = payload.get("machine")
        if not isinstance(machine, dict):
            raise DishonestLabelError(
                "hardware context machine must be an object, "
                f"got {type(machine).__name__}"
            )
        return cls(
            MachineProvenance.from_dict(machine),
            payload.get("endpoint_contract_version"),
            payload.get("model_version"),
            payload.get("topology"),
            payload.get("clock_settings"),
        )


def residual_ratio(fit: LinearFit, observations: Sequence[SweepObservation]) -> float:
    """Mean absolute residual as a fraction of the mean measured magnitude.

    A bare residual is uninterpretable across sweeps: ``0.4`` is excellent for a
    prefill curve measured in hundreds of milliseconds and catastrophic for a
    per-token curve measured in single digits. Dividing by the scale of what was
    actually measured makes one threshold apply to both curves.

    The denominator is the mean of ``abs(value)`` rather than of ``value`` so
    that a curve straddling zero cannot produce a near-zero denominator and an
    enormous ratio from a perfectly good fit.
    """
    if not observations:
        raise ValueError("residual ratio needs at least one observation")
    scale = sum(abs(item.value) for item in observations) / len(observations)
    if not math.isfinite(scale):
        raise ValueError("residual ratio scale must be finite")
    if scale <= 0.0:
        # Every measured value was exactly zero. That is not a well-scaled
        # curve; it is an endpoint returning a constant, and dividing by it
        # would report either 0.0 (a perfect fit) or inf depending on rounding.
        raise ValueError(
            "residual ratio is undefined when every measured value is zero"
        )
    return fit.residuals.mean_absolute / scale


@dataclass(frozen=True, slots=True)
class HardwareGate:
    """Thresholds a run must clear before ``HW_CALIBRATED`` is permitted.

    ``minimum_observations`` is far above :data:`~.fit.MINIMUM_OBSERVATIONS`.
    Three points are the algebraic minimum for a two-coefficient fit and leave
    zero residual by construction, which is exactly the configuration that
    produces a flawless-looking artifact from nothing.

    ``production`` defaults to ``True``, meaning the reviewed default thresholds
    are in force. A CLI override sets it to ``False``, which adds
    :attr:`HardwareBlocker.BLOCKED_NON_PRODUCTION_POLICY` and caps the labels at
    :attr:`~.provenance.CalibrationStatus.NON_PRODUCTION_EXPERIMENT`: a run judged
    under non-default thresholds may never claim ``HW_CALIBRATED``.
    """

    policy_version: str = HARDWARE_GATE_SCHEMA_VERSION
    required_contract_version: str = ENDPOINT_CONTRACT_VERSION
    minimum_observations: int = 12
    maximum_residual_ratio: float = 0.10
    production: bool = True

    def __post_init__(self) -> None:
        if self.policy_version != HARDWARE_GATE_SCHEMA_VERSION:
            raise ValueError(
                f"policy_version must be {HARDWARE_GATE_SCHEMA_VERSION!r}, "
                f"got {self.policy_version!r}"
            )
        if (
            not isinstance(self.required_contract_version, str)
            or not self.required_contract_version.strip()
        ):
            raise ValueError("required_contract_version must be a non-blank string")
        # `bool` is an `int`, so `minimum_observations=True` would otherwise
        # become 1 and admit a fit with no residual freedom at all.
        if isinstance(self.minimum_observations, bool) or not isinstance(
            self.minimum_observations, int
        ):
            raise ValueError("minimum_observations must be an int")
        if self.minimum_observations < 3:
            raise ValueError(
                "minimum_observations must be at least 3, the algebraic minimum "
                f"for a two-coefficient fit, got {self.minimum_observations}"
            )
        if isinstance(self.maximum_residual_ratio, bool) or not isinstance(
            self.maximum_residual_ratio, int | float
        ):
            raise ValueError("maximum_residual_ratio must be a number")
        if (
            not math.isfinite(self.maximum_residual_ratio)
            or not 0.0 < self.maximum_residual_ratio <= 1.0
        ):
            raise ValueError(
                "maximum_residual_ratio must be within (0, 1], got "
                f"{self.maximum_residual_ratio!r}"
            )
        if not isinstance(self.production, bool):
            raise ValueError(
                f"production must be a bool, got {type(self.production).__name__}"
            )

    def evaluate(
        self, context: HardwareContext, evidence: HardwareEvidence
    ) -> HardwareGateReport:
        """Judge one run and record the policy that judged it.

        The return type is a forward reference resolved by
        ``from __future__ import annotations``, and the body resolves the name
        at call time, so the gate can be defined before the report it produces
        without either class having to be split or reordered.
        """
        blockers = hardware_blockers(self, context, evidence)
        accepted = not blockers
        status, unit, tier = _labels_for(blockers, evidence)
        return HardwareGateReport(
            HARDWARE_SCHEMA_VERSION,
            self,
            context,
            evidence,
            blockers,
            accepted,
            status,
            unit,
            tier,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "required_contract_version": self.required_contract_version,
            "minimum_observations": self.minimum_observations,
            "maximum_residual_ratio": self.maximum_residual_ratio,
            "production": self.production,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HardwareGate:
        if not isinstance(payload, dict):
            raise ValueError(
                f"hardware gate must be an object, got {type(payload).__name__}"
            )
        return cls(
            payload["policy_version"],
            payload["required_contract_version"],
            payload["minimum_observations"],
            payload["maximum_residual_ratio"],
            payload.get("production", True),
        )


#: The policy the published M9-HW artifacts are judged against.
DEFAULT_HARDWARE_GATE = HardwareGate()


@dataclass(frozen=True, slots=True)
class HardwareEvidence:
    """The observable facts of one run, separated from the judgement of them.

    Keeping these in a record of their own is what lets a report be re-derived:
    the report stores the evidence and the policy, so a reader can recompute the
    verdict instead of taking it on trust.

    ``endpoint_is_synthetic`` is reported by the *driver*, not inferred from the
    endpoint's own claims, because a mock asked whether it is a mock is exactly
    the wrong witness.

    ``production_trust`` is False unless the driver verified the endpoint is
    HTTPS and on the allowlist. An untrusted endpoint that was nonetheless
    reached and measured publishes as
    :attr:`~.provenance.EvidenceTier.REMOTE_REPORTED`, a tier that cannot issue
    ``HW_CALIBRATED`` or ``HW_VALIDATED``.
    """

    engine_reachable: bool = False
    endpoint_is_synthetic: bool = True
    production_trust: bool = False
    observed_contract_version: str | None = None
    observation_count: int = 0
    fit_residual_ratio: float | None = None
    dry_run: bool = False

    def __post_init__(self) -> None:
        for name in (
            "engine_reachable",
            "endpoint_is_synthetic",
            "production_trust",
            "dry_run",
        ):
            value = getattr(self, name)
            # A truthy string or a 1 must not be able to decide reachability by
            # coercion; these booleans gate the strongest label available.
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a bool, got {type(value).__name__}")
        if self.observed_contract_version is not None and (
            not isinstance(self.observed_contract_version, str)
            or not self.observed_contract_version.strip()
        ):
            raise ValueError(
                "observed_contract_version must be None or a non-blank string"
            )
        if isinstance(self.observation_count, bool) or not isinstance(
            self.observation_count, int
        ):
            raise ValueError("observation_count must be an int")
        if self.observation_count < 0:
            raise ValueError("observation_count must be non-negative")
        if self.fit_residual_ratio is not None:
            if isinstance(self.fit_residual_ratio, bool) or not isinstance(
                self.fit_residual_ratio, int | float
            ):
                raise ValueError("fit_residual_ratio must be None or a number")
            # NaN fails every `>` comparison, so an unchecked NaN would clear
            # the residual gate rather than trip it.
            if not math.isfinite(self.fit_residual_ratio):
                raise ValueError(
                    f"fit_residual_ratio must be finite, got "
                    f"{self.fit_residual_ratio!r}"
                )
            if self.fit_residual_ratio < 0.0:
                raise ValueError("fit_residual_ratio must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_reachable": self.engine_reachable,
            "endpoint_is_synthetic": self.endpoint_is_synthetic,
            "production_trust": self.production_trust,
            "observed_contract_version": self.observed_contract_version,
            "observation_count": self.observation_count,
            "fit_residual_ratio": self.fit_residual_ratio,
            "dry_run": self.dry_run,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HardwareEvidence:
        if not isinstance(payload, dict):
            raise ValueError(
                f"hardware evidence must be an object, got {type(payload).__name__}"
            )
        return cls(
            payload["engine_reachable"],
            payload["endpoint_is_synthetic"],
            payload.get("production_trust", False),
            payload["observed_contract_version"],
            payload["observation_count"],
            payload["fit_residual_ratio"],
            payload["dry_run"],
        )


def hardware_blockers(
    policy: HardwareGate,
    context: HardwareContext,
    evidence: HardwareEvidence,
) -> tuple[HardwareBlocker, ...]:
    """Every reason ``policy`` refuses, in :data:`HARDWARE_BLOCKER_ORDER`.

    A module-level function rather than a method, called with the same arguments
    by :meth:`HardwareGate.evaluate` and by
    :meth:`HardwareGateReport.__post_init__`. One derivation, two callers, so
    the check cannot drift away from the thing it checks.

    Nothing short-circuits. An unreachable engine already settles the outcome,
    but continuing to evaluate turns the report into a checklist of what still
    has to be arranged before the next attempt.
    """
    blockers: list[HardwareBlocker] = []
    if not evidence.engine_reachable:
        blockers.append(HardwareBlocker.BLOCKED_NO_ENGINE_ACCESS)
    if evidence.endpoint_is_synthetic:
        blockers.append(HardwareBlocker.BLOCKED_SYNTHETIC_ENDPOINT)
    if not evidence.production_trust:
        blockers.append(HardwareBlocker.BLOCKED_NOT_PRODUCTION_TRUSTED)
    if not policy.production:
        blockers.append(HardwareBlocker.BLOCKED_NON_PRODUCTION_POLICY)
    if not context.complete:
        blockers.append(HardwareBlocker.BLOCKED_INCOMPLETE_PROVENANCE)
    if evidence.observed_contract_version != policy.required_contract_version:
        # `None` lands here too: an endpoint that never stated a contract
        # version has not agreed to one.
        blockers.append(HardwareBlocker.BLOCKED_CONTRACT_VERSION_MISMATCH)
    if evidence.observation_count < policy.minimum_observations:
        blockers.append(HardwareBlocker.BLOCKED_INSUFFICIENT_OBSERVATIONS)
    if evidence.fit_residual_ratio is None:
        # Distinct from the ratio being too large: not knowing how well the fit
        # describes the data is not the same as knowing that it describes it
        # badly, and the two call for different next steps.
        blockers.append(HardwareBlocker.BLOCKED_FIT_UNAVAILABLE)
    elif evidence.fit_residual_ratio > policy.maximum_residual_ratio:
        blockers.append(HardwareBlocker.BLOCKED_RESIDUAL_ABOVE_GATE)
    if evidence.dry_run:
        blockers.append(HardwareBlocker.BLOCKED_DRY_RUN_ONLY)
    return tuple(blockers)


def _labels_for(
    blockers: tuple[HardwareBlocker, ...],
    evidence: HardwareEvidence,
) -> tuple[CalibrationStatus, TimeUnit, EvidenceTier]:
    """The label tuple a verdict licenses, as one indivisible choice.

    Three outcomes, not two:

    * No blockers: the run measured a production-trusted endpoint under the
      reviewed default policy, so it earns ``HW_CALIBRATED`` / ``MILLISECONDS``
      / ``HW_VALIDATED``.
    * Only ``BLOCKED_NON_PRODUCTION_POLICY``: the thresholds were overridden on
      the CLI. The run may have measured a real engine, but the policy was not
      the reviewed one, so it publishes as ``NON_PRODUCTION_EXPERIMENT`` and
      may never claim a hardware label.
    * Only ``BLOCKED_NOT_PRODUCTION_TRUSTED`` (with a real, non-synthetic
      endpoint): the engine was reached and measured, but the endpoint was not
      HTTPS+allowlist-trusted. It publishes as ``REMOTE_REPORTED``, a tier
      above harness-only synthetic work but below hardware-validated.
    * Anything else: ``SYNTHETIC_UNCALIBRATED`` / ``HARNESS_ONLY``.

    ``NON_PRODUCTION_POLICY`` takes priority over ``NOT_PRODUCTION_TRUSTED``
    because a run under overridden thresholds is an experiment regardless of
    whether the endpoint was also untrusted.
    """
    if not blockers:
        return (
            CalibrationStatus.HW_CALIBRATED,
            TimeUnit.MILLISECONDS,
            EvidenceTier.HW_VALIDATED,
        )
    if HardwareBlocker.BLOCKED_NON_PRODUCTION_POLICY in blockers:
        return (
            CalibrationStatus.NON_PRODUCTION_EXPERIMENT,
            TimeUnit.NORMALIZED_WORK,
            EvidenceTier.SYNTHETIC_REPLAY,
        )
    trust_only = blockers == (HardwareBlocker.BLOCKED_NOT_PRODUCTION_TRUSTED,)
    if (
        trust_only
        and evidence.engine_reachable
        and not evidence.endpoint_is_synthetic
    ):
        return (
            CalibrationStatus.SYNTHETIC_UNCALIBRATED,
            TimeUnit.NORMALIZED_WORK,
            EvidenceTier.REMOTE_REPORTED,
        )
    return (
        CalibrationStatus.SYNTHETIC_UNCALIBRATED,
        TimeUnit.NORMALIZED_WORK,
        EvidenceTier.HARNESS_ONLY,
    )


@dataclass(frozen=True, slots=True)
class HardwareGateReport:
    """One M9-HW verdict, carrying the evidence and policy that produced it.

    The labels are stored so that a consumer reading only this record knows what
    the run is entitled to claim, and they are re-derived on construction so
    that a hand-edited ``HW_CALIBRATED`` over a report whose own evidence says
    the engine was unreachable cannot be built. The final
    :func:`~.provenance.assert_labels_supported` call is not redundant with that
    re-derivation: it is the same check every other calibration artifact passes
    through, so this record cannot become the one path into the artifact set
    that skips it.
    """

    schema_version: str
    policy: HardwareGate
    context: HardwareContext
    evidence: HardwareEvidence
    blockers: tuple[HardwareBlocker, ...]
    accepted: bool
    calibration_status: CalibrationStatus
    time_unit: TimeUnit
    evidence_tier: EvidenceTier

    def __post_init__(self) -> None:
        if self.schema_version != HARDWARE_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {HARDWARE_SCHEMA_VERSION!r}, "
                f"got {self.schema_version!r}"
            )
        # Exact type, not isinstance: a subclass could reintroduce the
        # thresholds as properties and answer the re-derivation below
        # differently from the way it answered `evaluate`.
        if type(self.policy) is not HardwareGate:
            raise ValueError(
                "policy must be a HardwareGate stating the thresholds this "
                f"report was judged against, got {type(self.policy).__name__}"
            )
        if type(self.context) is not HardwareContext:
            raise ValueError(
                f"context must be a HardwareContext, got {type(self.context).__name__}"
            )
        if type(self.evidence) is not HardwareEvidence:
            raise ValueError(
                "evidence must be a HardwareEvidence, got "
                f"{type(self.evidence).__name__}"
            )
        if not isinstance(self.accepted, bool):
            raise ValueError(
                f"accepted must be a bool, got {type(self.accepted).__name__}"
            )

        expected = hardware_blockers(self.policy, self.context, self.evidence)
        if tuple(self.blockers) != expected:
            raise DishonestLabelError(
                f"blockers {[item.value for item in self.blockers]!r} are not "
                "what this report's own policy and evidence produce, which is "
                f"{[item.value for item in expected]!r}"
            )
        if self.accepted != (not expected):
            raise DishonestLabelError(
                f"accepted={self.accepted!r} contradicts {len(expected)} "
                "blocker(s): a calibration is accepted exactly when nothing "
                "blocks it"
            )

        labels = (self.calibration_status, self.time_unit, self.evidence_tier)
        allowed = _labels_for(expected, self.evidence)
        if labels != allowed:
            raise DishonestLabelError(
                f"a report with accepted={self.accepted!r} must be labelled "
                f"{tuple(item.value for item in allowed)!r}, got "
                f"{tuple(item.value for item in labels)!r}"
            )
        assert_labels_supported(
            self.context.machine,
            self.calibration_status,
            self.time_unit,
            self.evidence_tier,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy": self.policy.to_dict(),
            "context": self.context.to_dict(),
            "evidence": self.evidence.to_dict(),
            "blockers": [item.value for item in self.blockers],
            "accepted": self.accepted,
            "calibration_status": self.calibration_status.value,
            "time_unit": self.time_unit.value,
            "evidence_tier": self.evidence_tier.value,
            "missing_context_fields": list(self.context.missing_fields()),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HardwareGateReport:
        """Rebuild a report, and re-judge it while doing so.

        Nothing here re-checks the verdict; the constructor does that, using the
        same rule that governs a freshly evaluated report because it is the same
        rule. ``missing_context_fields`` is deliberately not read back: it is
        derived from ``context`` on the way out, so accepting it on the way in
        would let a payload state one thing and its own context another.
        """
        if not isinstance(payload, dict):
            raise ValueError(
                f"hardware gate report must be an object, got {type(payload).__name__}"
            )
        blockers = payload["blockers"]
        if not isinstance(blockers, list):
            raise ValueError("blockers must be a list")
        return cls(
            payload["schema_version"],
            HardwareGate.from_dict(payload["policy"]),
            HardwareContext.from_dict(payload["context"]),
            HardwareEvidence.from_dict(payload["evidence"]),
            tuple(HardwareBlocker(item) for item in blockers),
            payload["accepted"],
            CalibrationStatus(payload["calibration_status"]),
            TimeUnit(payload["time_unit"]),
            EvidenceTier(payload["evidence_tier"]),
        )
