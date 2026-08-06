"""Honesty labels and nullable machine provenance for calibration artifacts.

The M9 harness can be driven by a mock engine or by a real accelerator. Only the
latter may produce millisecond-labelled parameters, so provenance and label
selection live in one dependency-free module that every other calibration module
imports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any


class CalibrationStatus(StrEnum):
    """Whether the fitted parameters came from real hardware measurements."""

    SYNTHETIC_UNCALIBRATED = "SYNTHETIC_UNCALIBRATED"
    HW_CALIBRATED = "HW_CALIBRATED"


class TimeUnit(StrEnum):
    """Unit of the fitted response variable."""

    NORMALIZED_WORK = "NORMALIZED_WORK"
    MILLISECONDS = "MILLISECONDS"


class EvidenceTier(StrEnum):
    """Strength of the evidence backing a published artifact."""

    HARNESS_ONLY = "HARNESS_ONLY"
    SYNTHETIC_REPLAY = "SYNTHETIC_REPLAY"
    HW_VALIDATED = "HW_VALIDATED"


class DishonestLabelError(ValueError):
    """Raised when a label claims more evidence than the provenance supports."""


#: Every field that must name something before a machine counts as identified.
_PROVENANCE_FIELDS: tuple[str, ...] = (
    "host_id",
    "accelerator_model",
    "engine_version",
    "captured_at_utc",
)

#: RFC 3339 ``date-time``. The offset is required so a bare wall-clock reading
#: can never be mistaken for UTC; the offset value itself is checked separately.
_RFC3339 = re.compile(
    r"^\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(\.\d+)?([Zz]|[+-]\d{2}:\d{2})$"
)


def _validate_captured_at_utc(value: str) -> None:
    """Reject anything that is not an RFC 3339 timestamp at UTC offset zero.

    ``captured_at_utc`` is the only provenance field a reader can use to decide
    whether a measurement is stale, and the field name promises UTC. An
    unparseable string, or one carrying a local offset, would still satisfy
    :attr:`MachineProvenance.complete` and license ``HW_CALIBRATED`` labels while
    saying nothing verifiable about when the run happened.
    """
    if not _RFC3339.match(value):
        raise DishonestLabelError(
            "machine provenance captured_at_utc must be an RFC3339 UTC timestamp "
            f"such as 2026-08-05T00:00:00Z, got {value!r}"
        )
    # RFC 3339 permits a lowercase zone designator, and only the final character
    # can be one, so replace it positionally: a blanket ``str.replace`` would
    # strip the offset instead of converting it and turn a valid UTC stamp into
    # a naive one that then fails the offset check below.
    normalized = value[:-1] + "+00:00" if value[-1] in "Zz" else value
    try:
        moment = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise DishonestLabelError(
            f"machine provenance captured_at_utc is not a real instant: {value!r}"
        ) from error
    if moment.utcoffset() != timedelta(0):
        raise DishonestLabelError(
            "machine provenance captured_at_utc must be at UTC offset zero, "
            f"got {value!r}"
        )


@dataclass(frozen=True, slots=True)
class MachineProvenance:
    """Identity of the measured machine; every field is nullable by design.

    A field is either ``None`` (unknown) or a non-blank identifier string. Blank,
    whitespace-only and non-string values are rejected at construction because
    they would otherwise satisfy :attr:`complete` and silently license
    ``HW_CALIBRATED`` millisecond labels without naming any machine.
    """

    host_id: str | None
    accelerator_model: str | None
    engine_version: str | None
    captured_at_utc: str | None

    def __post_init__(self) -> None:
        for name in _PROVENANCE_FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            # Deserialized payloads are untrusted: a JSON number or list would
            # otherwise flow straight into `complete` and into the artifact.
            if not isinstance(value, str):
                raise DishonestLabelError(
                    f"machine provenance {name} must be None or a string, "
                    f"got {type(value).__name__}"
                )
            if not value.strip():
                raise DishonestLabelError(
                    f"machine provenance {name} must be None or a non-blank identifier"
                )
        if self.captured_at_utc is not None:
            _validate_captured_at_utc(self.captured_at_utc)

    @classmethod
    def unknown(cls) -> MachineProvenance:
        return cls(None, None, None, None)

    @property
    def complete(self) -> bool:
        """True only when every provenance field names a machine."""
        return all(getattr(self, name) is not None for name in _PROVENANCE_FIELDS)

    def to_dict(self) -> dict[str, Any]:
        return {
            "host_id": self.host_id,
            "accelerator_model": self.accelerator_model,
            "engine_version": self.engine_version,
            "captured_at_utc": self.captured_at_utc,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MachineProvenance:
        return cls(
            payload["host_id"],
            payload["accelerator_model"],
            payload["engine_version"],
            payload["captured_at_utc"],
        )


#: The complete set of ``(calibration_status, time_unit, evidence_tier)`` tuples
#: this project is willing to publish. Enumerating the honest combinations, and
#: rejecting everything else, is the only way to keep mutually contradictory
#: labels such as ``HW_CALIBRATED`` + ``HARNESS_ONLY`` out of an artifact: a
#: per-field rule can only ever see one field at a time.
#:
#: Policy, in words:
#:
#: * ``SYNTHETIC_UNCALIBRATED`` measurements are unitless work, and their
#:   evidence is either the harness alone or the harness replayed over a real
#:   trace. They can never be ``MILLISECONDS`` and never ``HW_VALIDATED``.
#: * ``HW_CALIBRATED`` is only meaningful in ``MILLISECONDS`` on identified
#:   hardware, so it comes as one indivisible tuple with ``HW_VALIDATED``.
#:
#: Conservative underclaiming is deliberately permitted, but only along the whole
#: tuple: a run on fully identified hardware may be published as
#: ``SYNTHETIC_UNCALIBRATED`` / ``NORMALIZED_WORK`` / ``HARNESS_ONLY`` (that
#: tuple is internally coherent and claims strictly less than the evidence
#: supports). Underclaiming one field while overclaiming another is not
#: underclaiming; it is a contradiction, and is rejected.
HONEST_LABEL_TUPLES: frozenset[tuple[CalibrationStatus, TimeUnit, EvidenceTier]] = (
    frozenset(
        {
            (
                CalibrationStatus.SYNTHETIC_UNCALIBRATED,
                TimeUnit.NORMALIZED_WORK,
                EvidenceTier.HARNESS_ONLY,
            ),
            (
                CalibrationStatus.SYNTHETIC_UNCALIBRATED,
                TimeUnit.NORMALIZED_WORK,
                EvidenceTier.SYNTHETIC_REPLAY,
            ),
            (
                CalibrationStatus.HW_CALIBRATED,
                TimeUnit.MILLISECONDS,
                EvidenceTier.HW_VALIDATED,
            ),
        }
    )
)

#: Honest tuples that additionally require a fully identified machine.
_PROVENANCE_BACKED_TUPLES: frozenset[
    tuple[CalibrationStatus, TimeUnit, EvidenceTier]
] = frozenset(
    {
        (
            CalibrationStatus.HW_CALIBRATED,
            TimeUnit.MILLISECONDS,
            EvidenceTier.HW_VALIDATED,
        )
    }
)


def _format_tuple(status: CalibrationStatus, unit: TimeUnit, tier: EvidenceTier) -> str:
    return (
        f"calibration_status={status.value}, "
        f"time_unit={unit.value}, "
        f"evidence_tier={tier.value}"
    )


def honest_labels(
    machine: MachineProvenance,
) -> tuple[CalibrationStatus, TimeUnit, EvidenceTier]:
    """Return the strongest labels the supplied provenance can justify."""
    if machine.complete:
        return (
            CalibrationStatus.HW_CALIBRATED,
            TimeUnit.MILLISECONDS,
            EvidenceTier.HW_VALIDATED,
        )
    return (
        CalibrationStatus.SYNTHETIC_UNCALIBRATED,
        TimeUnit.NORMALIZED_WORK,
        EvidenceTier.HARNESS_ONLY,
    )


def assert_labels_supported(
    machine: MachineProvenance,
    status: CalibrationStatus,
    unit: TimeUnit,
    tier: EvidenceTier,
) -> None:
    """Fail closed on label tuples that are contradictory or unsupported.

    Two independent checks, in order: the tuple must be one this project is
    willing to publish at all, and a hardware-claiming tuple must additionally
    name the machine it was measured on.
    """
    labels = (status, unit, tier)
    if labels not in HONEST_LABEL_TUPLES:
        allowed = ", ".join(
            sorted(f"({_format_tuple(*item)})" for item in HONEST_LABEL_TUPLES)
        )
        raise DishonestLabelError(
            f"({_format_tuple(status, unit, tier)}) is not an honest label tuple; "
            f"allowed tuples are {allowed}"
        )
    if labels in _PROVENANCE_BACKED_TUPLES and not machine.complete:
        missing = ", ".join(
            name for name in _PROVENANCE_FIELDS if getattr(machine, name) is None
        )
        raise DishonestLabelError(
            f"({_format_tuple(status, unit, tier)}) requires complete machine "
            f"provenance; unknown fields: {missing}"
        )
