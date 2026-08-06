"""Deterministic two-factor cost fits with empirical (non-Gaussian) residuals."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .endpoint import SweepKind, SweepObservation

#: Residual quantiles are reported directly from the sorted sample; no
#: distributional family is fitted and none is assumed.
DISTRIBUTION_ASSUMPTION = "NONE_EMPIRICAL"

_RANK_DEFICIENT_TOLERANCE = 1e-12


#: A fit needs at least as many observations as free parameters plus one; the
#: two-factor model has three, so three points is the floor.
MINIMUM_OBSERVATIONS = 3


def _nearest_rank(ordered: Sequence[float], quantile: float) -> float:
    index = max(1, math.ceil(quantile * len(ordered))) - 1
    return ordered[index]


def _require_finite(value: object, label: str) -> float:
    """Reject anything that is not a finite real number.

    ``from_dict`` is fed untrusted JSON, and Python's ``json`` module happily
    parses the non-standard ``NaN`` / ``Infinity`` literals into floats. One of
    them anywhere in a fit silently poisons every prediction derived from it and
    re-serializes into a file no strict JSON reader can load, so it has to die at
    construction rather than at some later comparison that quietly returns False.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be a real number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {number}")
    return number


def _require_count(value: object, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}, got {value}")
    return value


@dataclass(frozen=True, slots=True)
class ResidualSummary:
    """Empirical spread of ``observed - predicted`` values.

    Quantiles use the nearest-rank rule on the sorted signed residuals, so a
    heavy tail stays visible instead of being smoothed into a normal sigma.
    """

    count: int
    distribution_assumption: str
    p50: float
    p90: float
    p95: float
    p99: float
    minimum: float
    maximum: float
    maximum_absolute: float
    mean_absolute: float
    raw: tuple[float, ...]

    def __post_init__(self) -> None:
        raw = tuple(_require_finite(value, "residual") for value in self.raw)
        count = _require_count(self.count, "residual count", 1)
        if count != len(raw):
            raise ValueError(
                f"residual count {count} does not match {len(raw)} raw residuals"
            )
        if self.distribution_assumption != DISTRIBUTION_ASSUMPTION:
            # The quantiles below are order statistics of `raw`. A payload that
            # relabels them as coming from a fitted family would misdescribe
            # numbers this module never produced that way.
            raise ValueError(
                "distribution_assumption must be "
                f"{DISTRIBUTION_ASSUMPTION}, got {self.distribution_assumption!r}"
            )
        # Every summary field is a function of `raw`, so it is recomputed here
        # rather than merely bounds-checked. Bounds admit fabrications: a p95 of
        # zero and a mean_absolute of zero both sit inside [minimum, maximum]
        # for a residual vector whose real values are nothing of the sort, and
        # `from_dict` is fed untrusted JSON. Recomputation makes the published
        # spread of a fit checkable against the vector it claims to summarise.
        ordered = sorted(raw)
        magnitudes = [abs(value) for value in raw]
        # Order statistics are exact selections from `raw`, so they compare
        # exactly: a JSON round trip preserves float values, and sorting and
        # indexing introduce no arithmetic.
        expected: tuple[tuple[str, float], ...] = (
            ("p50", _nearest_rank(ordered, 0.50)),
            ("p90", _nearest_rank(ordered, 0.90)),
            ("p95", _nearest_rank(ordered, 0.95)),
            ("p99", _nearest_rank(ordered, 0.99)),
            ("minimum", ordered[0]),
            ("maximum", ordered[-1]),
            ("maximum_absolute", max(magnitudes)),
        )
        for name, wanted in expected:
            value = _require_finite(getattr(self, name), f"residual {name}")
            if value != wanted:
                raise ValueError(
                    f"residual {name} does not match the value recomputed from "
                    f"the raw residuals: {value!r} != {wanted!r}"
                )
        mean_absolute = _require_finite(self.mean_absolute, "residual mean_absolute")
        # `mean_absolute` is the one derived-by-arithmetic field. `fmean` sums
        # exactly and rounds once, so the same runtime recomputes it bit for bit
        # and a stricter check would pass here today. The tolerance is for a
        # payload written by a different Python whose `statistics` may round the
        # last bit elsewhere; at 1e-12 relative it still rejects every
        # fabrication worth catching, which differ in the leading digits.
        wanted_mean = statistics.fmean(magnitudes)
        if not math.isclose(mean_absolute, wanted_mean, rel_tol=1e-12, abs_tol=0.0):
            raise ValueError(
                "residual mean_absolute does not match the value recomputed "
                f"from the raw residuals: {mean_absolute!r} != {wanted_mean!r}"
            )

    @classmethod
    def from_residuals(cls, residuals: Sequence[float]) -> ResidualSummary:
        if not residuals:
            raise ValueError("residual summary needs at least one residual")
        ordered = sorted(residuals)
        magnitudes = [abs(value) for value in residuals]
        return cls(
            len(residuals),
            DISTRIBUTION_ASSUMPTION,
            _nearest_rank(ordered, 0.50),
            _nearest_rank(ordered, 0.90),
            _nearest_rank(ordered, 0.95),
            _nearest_rank(ordered, 0.99),
            ordered[0],
            ordered[-1],
            max(magnitudes),
            statistics.fmean(magnitudes),
            tuple(residuals),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "distribution_assumption": self.distribution_assumption,
            "p50": self.p50,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "maximum_absolute": self.maximum_absolute,
            "mean_absolute": self.mean_absolute,
            "raw": list(self.raw),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ResidualSummary:
        return cls(
            payload["count"],
            payload["distribution_assumption"],
            payload["p50"],
            payload["p90"],
            payload["p95"],
            payload["p99"],
            payload["minimum"],
            payload["maximum"],
            payload["maximum_absolute"],
            payload["mean_absolute"],
            tuple(payload["raw"]),
        )


@dataclass(frozen=True, slots=True)
class LinearFit:
    """``value = intercept + token_coefficient * tokens + batch_coefficient * batch``.

    The response is prefill cost for :data:`SweepKind.PREFILL` and per-output-token
    cost for :data:`SweepKind.DECODE`.
    """

    kind: SweepKind
    intercept: float
    token_coefficient: float
    batch_coefficient: float
    observation_count: int
    residuals: ResidualSummary

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SweepKind):
            raise ValueError(f"fit kind must be a SweepKind, got {self.kind!r}")
        for name in ("intercept", "token_coefficient", "batch_coefficient"):
            _require_finite(getattr(self, name), f"fit {name}")
        count = _require_count(
            self.observation_count, "fit observation_count", MINIMUM_OBSERVATIONS
        )
        if count != self.residuals.count:
            # One residual is produced per observation. A mismatch means the two
            # halves of the payload describe different runs, and every residual
            # quantile published alongside the coefficients would be misleading.
            raise ValueError(
                f"fit observation_count {count} does not match "
                f"{self.residuals.count} residuals"
            )

    def predict(self, *, tokens: int, batch_size: int) -> float:
        return (
            self.intercept
            + self.token_coefficient * tokens
            + self.batch_coefficient * batch_size
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "intercept": self.intercept,
            "token_coefficient": self.token_coefficient,
            "batch_coefficient": self.batch_coefficient,
            "observation_count": self.observation_count,
            "residuals": self.residuals.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LinearFit:
        return cls(
            SweepKind(payload["kind"]),
            payload["intercept"],
            payload["token_coefficient"],
            payload["batch_coefficient"],
            payload["observation_count"],
            ResidualSummary.from_dict(payload["residuals"]),
        )


def fit_sweep(observations: Sequence[SweepObservation]) -> LinearFit:
    """Fit the two-factor cost model by mean-centred ordinary least squares.

    Centring keeps the normal equations well conditioned for the large token
    magnitudes in the Mooncake trace and makes the solution a closed form, so the
    result is bit-for-bit reproducible without a linear algebra dependency.
    """
    if len(observations) < 3:
        raise ValueError("fit needs at least three observations")
    kinds = {observation.kind for observation in observations}
    if len(kinds) != 1:
        raise ValueError("fit needs a single sweep kind")

    tokens = [float(observation.tokens) for observation in observations]
    batches = [float(observation.batch_size) for observation in observations]
    values = [observation.value for observation in observations]
    token_mean = statistics.fmean(tokens)
    batch_mean = statistics.fmean(batches)
    value_mean = statistics.fmean(values)

    token_dev = [value - token_mean for value in tokens]
    batch_dev = [value - batch_mean for value in batches]
    value_dev = [value - value_mean for value in values]

    token_var = sum(value * value for value in token_dev)
    batch_var = sum(value * value for value in batch_dev)
    cross = sum(a * b for a, b in zip(token_dev, batch_dev, strict=True))
    token_cov = sum(a * b for a, b in zip(token_dev, value_dev, strict=True))
    batch_cov = sum(a * b for a, b in zip(batch_dev, value_dev, strict=True))

    determinant = token_var * batch_var - cross * cross
    if determinant <= _RANK_DEFICIENT_TOLERANCE * max(1.0, token_var * batch_var):
        raise ValueError("sweep observations are rank deficient in (tokens, batch)")

    token_coefficient = (batch_var * token_cov - cross * batch_cov) / determinant
    batch_coefficient = (token_var * batch_cov - cross * token_cov) / determinant
    intercept = (
        value_mean - token_coefficient * token_mean - batch_coefficient * batch_mean
    )

    residuals = [
        observation.value
        - (
            intercept
            + token_coefficient * observation.tokens
            + batch_coefficient * observation.batch_size
        )
        for observation in observations
    ]
    return LinearFit(
        kinds.pop(),
        intercept,
        token_coefficient,
        batch_coefficient,
        len(observations),
        ResidualSummary.from_residuals(residuals),
    )
