"""Sweep specification and driver for the M9 calibration harness."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from .endpoint import EngineEndpoint, SweepKind, SweepObservation


def _validate_axis(points: tuple[int, ...], label: str, minimum: int) -> None:
    if len(points) < minimum:
        raise ValueError(f"sweep needs at least two distinct {label} points")
    if any(point <= 0 for point in points):
        raise ValueError(f"{label} points must be positive")
    pairs = zip(points, points[1:], strict=False)
    if any(later <= earlier for earlier, later in pairs):
        raise ValueError(f"{label} points must be strictly increasing")


@dataclass(frozen=True, slots=True)
class SweepSpec:
    """Grid of measurement points for one cost curve.

    ``token_points`` sweeps uncached input tokens for prefill and output length
    for decode; ``batch_points`` sweeps the concurrent batch state in both cases.
    """

    kind: SweepKind
    token_points: tuple[int, ...]
    batch_points: tuple[int, ...]
    repeats: int = 1

    def __post_init__(self) -> None:
        _validate_axis(self.token_points, "token", 2)
        _validate_axis(self.batch_points, "batch", 2)
        if self.repeats <= 0:
            raise ValueError("repeats must be positive")

    def observation_count(self) -> int:
        return len(self.token_points) * len(self.batch_points) * self.repeats

    def points(self) -> Iterator[tuple[int, int, int]]:
        for tokens in self.token_points:
            for batch_size in self.batch_points:
                for repeat_index in range(self.repeats):
                    yield tokens, batch_size, repeat_index

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "token_points": list(self.token_points),
            "batch_points": list(self.batch_points),
            "repeats": self.repeats,
        }


def run_sweep(
    endpoint: EngineEndpoint, spec: SweepSpec
) -> tuple[SweepObservation, ...]:
    """Drive ``endpoint`` across ``spec`` in a fixed, reproducible order.

    Every observation must echo the grid point that was requested. Published
    artifacts state the spec, so an endpoint that silently relabels or clamps a
    point would make ``observations.csv`` contradict ``params.json``.
    """
    observations: list[SweepObservation] = []
    for tokens, batch_size, repeat_index in spec.points():
        observation = endpoint.measure(
            kind=spec.kind,
            tokens=tokens,
            batch_size=batch_size,
            repeat_index=repeat_index,
        )
        if observation.kind is not spec.kind:
            raise ValueError("endpoint returned an observation for a different sweep")
        requested = (tokens, batch_size, repeat_index)
        returned = (
            observation.tokens,
            observation.batch_size,
            observation.repeat_index,
        )
        if returned != requested:
            raise ValueError(
                "endpoint returned an observation for a different grid point: "
                f"requested {requested}, got {returned}"
            )
        observations.append(observation)
    return tuple(observations)
