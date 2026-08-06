"""The ranking-consistency statistic, frozen before any number is reported.

Two statistics are provided and both are computed for every comparison, but
:data:`FROZEN_RANKING_STATISTIC` names the one that gates decisions. Freezing it
here — in code, before the first replay is run — is what stops the choice from
being made later against whichever number happened to look better.

Both statistics refuse to return a value they cannot support: a side that is
entirely tied has no ranking to agree with, so the answer is an error rather
than a comfortable zero.

That refusal is raised as :class:`UndefinedRankingError`, which is deliberately
*not* the error raised for unusable input. A caller that cannot rank two arms
should record "no ranking evidence"; a caller handed a malformed scoring should
fail. Collapsing the two -- catching plain :class:`ValueError` -- would let a
corrupt artifact be reported as an honest absence of evidence, which is the one
misreading this package exists to prevent.

``NaN`` is rejected before either statistic is computed. Every comparison
against ``NaN`` is false, so :func:`_sign` would return ``0`` for it and the
pair would be counted as a *tie* -- and a scoring that is entirely ``NaN`` would
be indistinguishable from one that is entirely tied except that the tie
correction cannot see it. A single ``NaN`` arm would therefore be silently
dropped from the pair counts and the surviving arms would report perfect
agreement.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import sqrt
from typing import Any

from .payload import (
    check,
    check_finite,
    check_unit_interval,
    invalid,
    require_float,
    require_float_sequence,
    require_int,
    require_mapping,
    require_str,
    require_str_sequence,
)

RANKING_SCHEMA_VERSION = "m10-ranking-v1"

#: The gating statistic. Chosen before any replay result existed.
FROZEN_RANKING_STATISTIC = "KENDALL_TAU_B"

#: Tolerance for re-deriving a stored statistic from its stored inputs.
#: Tight enough that a hand-edited number is caught, loose enough that the same
#: computation on the same floats in a different order still agrees.
_RECOMPUTE_TOLERANCE = 1e-12


class UndefinedRankingError(ValueError):
    """The inputs were well-formed but admit no ranking statistic.

    Raised only when the arms genuinely cannot be ranked -- fewer than two of
    them, or a side with no strictly ordered pair. Unusable input raises
    :class:`~.payload.MalformedPayloadError` instead, so that a caller may treat
    "no ranking evidence" as a normal outcome without also swallowing corruption.
    """


def _checked(left: Sequence[float], right: Sequence[float], context: str) -> None:
    check(
        len(left) == len(right),
        context,
        f"rankings must have equal length, got {len(left)} and {len(right)}",
    )
    for label, scores in (("left", left), ("right", right)):
        for position, score in enumerate(scores):
            check_finite(score, f"{label} score at position {position}", context)


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def concordance(
    left: Sequence[float], right: Sequence[float]
) -> tuple[int, int, int, int]:
    """Return ``(concordant, discordant, left_ties, right_ties)`` pair counts."""
    _checked(left, right, "concordance")
    concordant = discordant = left_ties = right_ties = 0
    for i in range(len(left)):
        for j in range(i + 1, len(left)):
            a = _sign(left[j] - left[i])
            b = _sign(right[j] - right[i])
            if a == 0:
                left_ties += 1
            if b == 0:
                right_ties += 1
            if a == 0 or b == 0:
                continue
            if a == b:
                concordant += 1
            else:
                discordant += 1
    return concordant, discordant, left_ties, right_ties


def kendall_tau_b(left: Sequence[float], right: Sequence[float]) -> float:
    """Kendall's tau-b: rank correlation with an explicit tie correction."""
    _checked(left, right, "kendall_tau_b")
    if len(left) < 2:
        raise UndefinedRankingError("tau-b needs at least two arms")

    concordant, discordant, left_ties, right_ties = concordance(left, right)
    n0 = len(left) * (len(left) - 1) // 2
    denominator = (n0 - left_ties) * (n0 - right_ties)
    if denominator <= 0:
        raise UndefinedRankingError("tau-b is undefined when one side is fully tied")
    return (concordant - discordant) / sqrt(denominator)


def pairwise_winner_agreement(left: Sequence[float], right: Sequence[float]) -> float:
    """Fraction of strictly ordered pairs that pick the same winner.

    A pair tied on either side is dropped rather than counted as agreement: two
    arms that scored identically did not pick a winner to agree about.
    """
    _checked(left, right, "pairwise_winner_agreement")
    concordant, discordant, _, _ = concordance(left, right)
    comparable = concordant + discordant
    if comparable == 0:
        raise UndefinedRankingError("agreement is undefined: no strictly ordered pair")
    return concordant / comparable


@dataclass(frozen=True, slots=True)
class RankingComparison:
    """One frozen-statistic comparison between two scorings of the same arms.

    The record carries both its inputs and its conclusions, so the conclusions
    are checked against the inputs on construction rather than trusted. A
    comparison whose ``tau_b`` does not follow from its own scores is refused:
    the scores are the evidence, and a number that the evidence does not support
    is exactly the claim this package must not publish. Because ``from_dict``
    builds the same record, a hand-edited artifact is rejected by the same rule
    that governs a freshly computed one.
    """

    schema_version: str
    statistic: str
    left_label: str
    right_label: str
    arm_ids: tuple[str, ...]
    left_scores: tuple[float, ...]
    right_scores: tuple[float, ...]
    tau_b: float
    pairwise_agreement: float
    concordant_pairs: int
    discordant_pairs: int

    def __post_init__(self) -> None:
        context = "RankingComparison"
        check(
            self.schema_version == RANKING_SCHEMA_VERSION,
            context,
            f"schema_version must be {RANKING_SCHEMA_VERSION!r}, "
            f"got {self.schema_version!r}",
        )
        check(
            self.statistic == FROZEN_RANKING_STATISTIC,
            context,
            f"statistic must be the frozen {FROZEN_RANKING_STATISTIC!r}, "
            f"got {self.statistic!r}",
        )
        for name, label in (
            ("left_label", self.left_label),
            ("right_label", self.right_label),
        ):
            check(bool(label), context, f"{name} must not be empty")
        check(
            len(self.arm_ids) >= 2,
            context,
            f"a comparison needs at least two arms, got {len(self.arm_ids)}",
        )
        check(
            all(self.arm_ids),
            context,
            f"arm_ids must not contain an empty id: {list(self.arm_ids)!r}",
        )
        check(
            len(set(self.arm_ids)) == len(self.arm_ids),
            context,
            f"arm_ids must be distinct, got {list(self.arm_ids)!r}",
        )
        check(
            list(self.arm_ids) == sorted(self.arm_ids),
            context,
            f"arm_ids must be in sorted order, got {list(self.arm_ids)!r}",
        )
        check(
            len(self.left_scores) == len(self.arm_ids)
            and len(self.right_scores) == len(self.arm_ids),
            context,
            f"each side must score every arm: {len(self.arm_ids)} arms but "
            f"{len(self.left_scores)} left and {len(self.right_scores)} right scores",
        )
        check_finite(self.tau_b, "tau_b", context)
        check(
            -1.0 <= self.tau_b <= 1.0,
            context,
            f"tau_b must be within [-1, 1], got {self.tau_b!r}",
        )
        check_unit_interval(self.pairwise_agreement, "pairwise_agreement", context)
        self._check_derived_from_scores(context)

    def _check_derived_from_scores(self, context: str) -> None:
        """Refuse a conclusion the stored scores do not produce."""
        try:
            concordant, discordant, _, _ = concordance(
                self.left_scores, self.right_scores
            )
            tau_b = kendall_tau_b(self.left_scores, self.right_scores)
            agreement = pairwise_winner_agreement(self.left_scores, self.right_scores)
        except UndefinedRankingError as error:
            raise invalid(
                context,
                f"the stored scores admit no ranking statistic ({error}), so "
                f"tau_b={self.tau_b!r} cannot have been derived from them",
            ) from error

        check(
            self.concordant_pairs == concordant,
            context,
            f"concordant_pairs is {self.concordant_pairs}, but the stored scores "
            f"have {concordant}",
        )
        check(
            self.discordant_pairs == discordant,
            context,
            f"discordant_pairs is {self.discordant_pairs}, but the stored scores "
            f"have {discordant}",
        )
        check(
            math.isclose(
                self.tau_b,
                tau_b,
                rel_tol=_RECOMPUTE_TOLERANCE,
                abs_tol=_RECOMPUTE_TOLERANCE,
            ),
            context,
            f"tau_b is {self.tau_b!r}, but the stored scores give {tau_b!r}",
        )
        check(
            math.isclose(
                self.pairwise_agreement,
                agreement,
                rel_tol=_RECOMPUTE_TOLERANCE,
                abs_tol=_RECOMPUTE_TOLERANCE,
            ),
            context,
            f"pairwise_agreement is {self.pairwise_agreement!r}, but the stored "
            f"scores give {agreement!r}",
        )

    @classmethod
    def from_scores(
        cls,
        left_label: str,
        right_label: str,
        left_scores: Mapping[str, float],
        right_scores: Mapping[str, float],
    ) -> RankingComparison:
        check(
            set(left_scores) == set(right_scores),
            "RankingComparison",
            "both scorings must cover the same arm ids, got "
            f"{sorted(left_scores)!r} and {sorted(right_scores)!r}",
        )
        arm_ids = tuple(sorted(left_scores))
        left = tuple(float(left_scores[arm_id]) for arm_id in arm_ids)
        right = tuple(float(right_scores[arm_id]) for arm_id in arm_ids)
        concordant, discordant, _, _ = concordance(left, right)
        return cls(
            RANKING_SCHEMA_VERSION,
            FROZEN_RANKING_STATISTIC,
            left_label,
            right_label,
            arm_ids,
            left,
            right,
            kendall_tau_b(left, right),
            pairwise_winner_agreement(left, right),
            concordant,
            discordant,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "statistic": self.statistic,
            "left_label": self.left_label,
            "right_label": self.right_label,
            "arm_ids": list(self.arm_ids),
            "left_scores": list(self.left_scores),
            "right_scores": list(self.right_scores),
            "tau_b": self.tau_b,
            "pairwise_agreement": self.pairwise_agreement,
            "concordant_pairs": self.concordant_pairs,
            "discordant_pairs": self.discordant_pairs,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RankingComparison:
        context = "RankingComparison"
        data = require_mapping(payload, context)
        return cls(
            require_str(data, "schema_version", context),
            require_str(data, "statistic", context),
            require_str(data, "left_label", context),
            require_str(data, "right_label", context),
            require_str_sequence(data, "arm_ids", context),
            require_float_sequence(data, "left_scores", context),
            require_float_sequence(data, "right_scores", context),
            require_float(data, "tau_b", context),
            require_float(data, "pairwise_agreement", context),
            require_int(data, "concordant_pairs", context),
            require_int(data, "discordant_pairs", context),
        )
