"""Engine endpoint abstraction and a deterministic mock implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from ..identity import NamedSeedManager
from .provenance import MachineProvenance

_NOISE_RESOLUTION = 1_000_000


class SweepKind(StrEnum):
    """Which cost curve a measurement belongs to."""

    PREFILL = "PREFILL"
    DECODE = "DECODE"


@dataclass(frozen=True, slots=True)
class SweepObservation:
    """One measured point of a cost curve.

    ``tokens`` is uncached input tokens for :data:`SweepKind.PREFILL` and output
    tokens for :data:`SweepKind.DECODE`. ``value`` is prefill cost for the former
    and per-output-token cost for the latter, expressed in the endpoint's own
    unit (normalized work for mock endpoints, milliseconds for real hardware).
    """

    kind: SweepKind
    tokens: int
    batch_size: int
    repeat_index: int
    value: float

    def __post_init__(self) -> None:
        if self.tokens <= 0 or self.batch_size <= 0:
            raise ValueError("tokens and batch_size must be positive")
        if self.repeat_index < 0:
            raise ValueError("repeat_index must be non-negative")
        if not math.isfinite(self.value):
            # A failed or timed-out engine call reads back as NaN/inf. Admitting
            # one poisons every fitted coefficient and serializes as a bare
            # ``NaN`` literal, which is not valid JSON, so reject it here.
            raise ValueError("value must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "tokens": self.tokens,
            "batch_size": self.batch_size,
            "repeat_index": self.repeat_index,
            "value": self.value,
        }


@runtime_checkable
class EngineEndpoint(Protocol):
    """Minimal surface the M9 sweep driver needs from an inference engine."""

    @property
    def endpoint_id(self) -> str:
        """Stable identifier for the measured endpoint.

        Declared read-only so that immutable implementations satisfy the
        protocol: a settable protocol variable additionally demands a setter,
        which a frozen dataclass such as :class:`MockEngine` cannot provide.
        The driver only ever reads this value.
        """

    def machine_provenance(self) -> MachineProvenance:
        """Identify the measured machine, or return nulls when unknown."""

    def measure(
        self,
        *,
        kind: SweepKind,
        tokens: int,
        batch_size: int,
        repeat_index: int,
    ) -> SweepObservation:
        """Measure one grid point of the requested cost curve."""


@dataclass(frozen=True, slots=True)
class MockCurve:
    """Ground-truth linear cost curve used by :class:`MockEngine`."""

    intercept: float
    token_coefficient: float
    batch_coefficient: float

    def value(self, tokens: int, batch_size: int) -> float:
        return (
            self.intercept
            + self.token_coefficient * tokens
            + self.batch_coefficient * batch_size
        )


@dataclass(frozen=True, slots=True)
class MockEngine:
    """Deterministic stand-in for a real engine endpoint.

    A mock can never identify a machine, so :meth:`machine_provenance` always
    returns nulls and every artifact derived from it stays uncalibrated.
    """

    endpoint_id: str = "mock-engine"
    prefill_curve: MockCurve = field(default=MockCurve(3.0, 0.02, 1.5))
    tpot_curve: MockCurve = field(default=MockCurve(8.0, 0.0004, 0.9))
    noise_amplitude: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.noise_amplitude < 0:
            raise ValueError("noise_amplitude must be non-negative")

    def machine_provenance(self) -> MachineProvenance:
        return MachineProvenance.unknown()

    def measure(
        self,
        *,
        kind: SweepKind,
        tokens: int,
        batch_size: int,
        repeat_index: int,
    ) -> SweepObservation:
        curve = self.prefill_curve if kind is SweepKind.PREFILL else self.tpot_curve
        value = curve.value(tokens, batch_size)
        if self.noise_amplitude:
            value += self._noise(kind, tokens, batch_size, repeat_index)
        return SweepObservation(kind, tokens, batch_size, repeat_index, value)

    def _noise(
        self, kind: SweepKind, tokens: int, batch_size: int, repeat_index: int
    ) -> float:
        name = f"calibration/{kind.value}/{tokens}/{batch_size}/{repeat_index}"
        stream = NamedSeedManager(self.seed).stream(name)
        draw = stream.randbelow(2 * _NOISE_RESOLUTION + 1) - _NOISE_RESOLUTION
        return self.noise_amplitude * draw / _NOISE_RESOLUTION
