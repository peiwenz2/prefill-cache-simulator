"""Versioned calibration parameter artifact with fail-closed honesty labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .endpoint import SweepKind
from .fit import LinearFit
from .provenance import (
    CalibrationStatus,
    EvidenceTier,
    MachineProvenance,
    TimeUnit,
    assert_labels_supported,
)

CALIBRATION_SCHEMA_VERSION = "calibration-params-v1"


@dataclass(frozen=True, slots=True)
class CalibrationParams:
    """Simulator cost parameters plus the evidence that backs them."""

    endpoint_id: str
    machine: MachineProvenance
    calibration_status: CalibrationStatus
    time_unit: TimeUnit
    evidence_tier: EvidenceTier
    prefill: LinearFit
    tpot: LinearFit
    code_sha: str | None = None
    code_dirty: bool | None = None
    schema_version: str = CALIBRATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CALIBRATION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported calibration schema version {self.schema_version}"
            )
        if not isinstance(self.endpoint_id, str) or not self.endpoint_id.strip():
            raise ValueError("endpoint_id must be a non-blank string")
        if not isinstance(self.machine, MachineProvenance):
            raise ValueError("machine must be a MachineProvenance")
        # The two slots are not interchangeable: `prefill` is a prefill-cost
        # curve and `tpot` a per-output-token curve. Swapping them survives every
        # other check in this module, round-trips cleanly, and silently makes the
        # simulator charge decode cost for prefill work.
        for name, expected in (
            ("prefill", SweepKind.PREFILL),
            ("tpot", SweepKind.DECODE),
        ):
            fit = getattr(self, name)
            if not isinstance(fit, LinearFit):
                raise ValueError(f"{name} must be a LinearFit")
            if fit.kind is not expected:
                raise ValueError(
                    f"{name} must hold a {expected.value} fit, got {fit.kind.value}"
                )
        if self.code_sha is not None and (
            not isinstance(self.code_sha, str) or not self.code_sha.strip()
        ):
            raise ValueError("code_sha must be None or a non-blank string")
        if self.code_dirty is not None and not isinstance(self.code_dirty, bool):
            raise ValueError("code_dirty must be None or a bool")
        # StrEnum members compare equal to their string values but hash by name,
        # so a raw string would miss the honest-tuple set with a confusing error.
        for name, enum_type in (
            ("calibration_status", CalibrationStatus),
            ("time_unit", TimeUnit),
            ("evidence_tier", EvidenceTier),
        ):
            if not isinstance(getattr(self, name), enum_type):
                raise ValueError(f"{name} must be a {enum_type.__name__}")
        assert_labels_supported(
            self.machine,
            self.calibration_status,
            self.time_unit,
            self.evidence_tier,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "endpoint_id": self.endpoint_id,
            "machine": self.machine.to_dict(),
            "calibration_status": self.calibration_status.value,
            "time_unit": self.time_unit.value,
            "evidence_tier": self.evidence_tier.value,
            "prefill": self.prefill.to_dict(),
            "tpot": self.tpot.to_dict(),
            "code_sha": self.code_sha,
            "code_dirty": self.code_dirty,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CalibrationParams:
        version = payload["schema_version"]
        if version != CALIBRATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported calibration schema version {version}")
        return cls(
            payload["endpoint_id"],
            MachineProvenance.from_dict(payload["machine"]),
            CalibrationStatus(payload["calibration_status"]),
            TimeUnit(payload["time_unit"]),
            EvidenceTier(payload["evidence_tier"]),
            LinearFit.from_dict(payload["prefill"]),
            LinearFit.from_dict(payload["tpot"]),
            payload["code_sha"],
            payload["code_dirty"],
            version,
        )
