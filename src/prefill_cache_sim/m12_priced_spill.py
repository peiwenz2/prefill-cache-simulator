"""M12.2 causal-home placement with an explicit marginal-cost ledger."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

HIT_RATE_EPSILON_ABSOLUTE = 0.001
LOAD_SKEW_EPSILON_ABSOLUTE = 0.05
P_QUEUE_P95_EPSILON_RELATIVE = 0.05


@dataclass(frozen=True, slots=True)
class PlacementCandidate:
    node_id: str
    local_hit_tokens: int
    remote_reusable_tokens: int
    p_queue_work: float
    kvs_contention_multiplier: float = 1.0
    model_id: str = "default"
    adapter_id: str = "default"
    work_shape: str = "default"
    slo_eligible: bool = True

    def __post_init__(self) -> None:
        if not self.node_id or not self.model_id or not self.adapter_id:
            raise ValueError("candidate identity and cohort must be non-empty")
        if self.local_hit_tokens < 0 or self.remote_reusable_tokens < 0:
            raise ValueError("reusable tokens must be non-negative")
        if self.p_queue_work < 0 or self.kvs_contention_multiplier < 1:
            raise ValueError("queue work must be non-negative and contention >= 1")


@dataclass(frozen=True, slots=True)
class PricedSpillDecision:
    node_id: str
    causal_home: str
    spilled: bool
    effective_prefill_work: float
    p_queue_work: float
    kvs_transfer_work: float
    total_marginal_work: float


@dataclass(frozen=True, slots=True)
class PricedSpillSelector:
    """Choose minimum causal cost; home is an anchor, never a hard sticky."""

    prefill_token_work: float
    kvs_token_work: float
    home_bias_work: float = 0.0

    def __post_init__(self) -> None:
        if self.prefill_token_work <= 0 or self.kvs_token_work < 0:
            raise ValueError("work coefficients are invalid")
        if self.home_bias_work < 0:
            raise ValueError("home bias must be non-negative")

    def select(
        self,
        *,
        prefix_family: str,
        input_tokens: int,
        candidates: Sequence[PlacementCandidate],
        model_id: str = "default",
        adapter_id: str = "default",
        work_shape: str = "default",
    ) -> PricedSpillDecision:
        if not prefix_family or input_tokens <= 0 or not candidates:
            raise ValueError("request and candidates must be non-empty")
        ordered = tuple(sorted(candidates, key=lambda item: item.node_id))
        home = ordered[_stable_index(prefix_family, len(ordered))].node_id
        eligible = tuple(
            candidate
            for candidate in ordered
            if candidate.slo_eligible
            and candidate.model_id == model_id
            and candidate.adapter_id == adapter_id
            and candidate.work_shape == work_shape
        )
        if not eligible:
            raise ValueError("no cohort-compatible SLO-eligible candidate")
        scored: list[tuple[float, bool, str, float, float, float]] = []
        for candidate in eligible:
            local = min(input_tokens, candidate.local_hit_tokens)
            remote = min(
                input_tokens - local,
                candidate.remote_reusable_tokens,
            )
            uncached = input_tokens - local - remote
            prefill = uncached * self.prefill_token_work
            transfer = (
                remote * self.kvs_token_work * candidate.kvs_contention_multiplier
            )
            home_penalty = 0.0 if candidate.node_id == home else self.home_bias_work
            total = prefill + candidate.p_queue_work + transfer + home_penalty
            if not math.isfinite(total):
                raise ValueError("candidate marginal work must be finite")
            scored.append(
                (
                    total,
                    candidate.node_id != home,
                    candidate.node_id,
                    prefill,
                    transfer,
                    candidate.p_queue_work,
                )
            )
        total, _, node_id, prefill, transfer, queue = min(scored)
        return PricedSpillDecision(
            node_id=node_id,
            causal_home=home,
            spilled=node_id != home,
            effective_prefill_work=prefill,
            p_queue_work=queue,
            kvs_transfer_work=transfer,
            total_marginal_work=total,
        )


@dataclass(frozen=True, slots=True)
class PlacementStrategyMetrics:
    strategy_id: str
    token_hit_rate: float
    request_load_max_mean: float
    p_queue_p95: float
    strict_useful_token_goodput: float
    strict_useful_output_token_goodput: float
    minimum_tier_slo_attainment: float
    jain_fairness: float
    per_tier_slo_attainment: Mapping[str, float]

    def __post_init__(self) -> None:
        tiers = dict(self.per_tier_slo_attainment)
        object.__setattr__(self, "per_tier_slo_attainment", MappingProxyType(tiers))
        values = (
            self.token_hit_rate,
            self.request_load_max_mean,
            self.p_queue_p95,
            self.strict_useful_token_goodput,
            self.strict_useful_output_token_goodput,
            self.minimum_tier_slo_attainment,
            self.jain_fairness,
        )
        if not self.strategy_id or any(
            value < 0 or not math.isfinite(value) for value in values
        ):
            raise ValueError("strategy metrics must be named, finite and non-negative")
        if any(
            value > 1
            for value in (
                self.token_hit_rate,
                self.minimum_tier_slo_attainment,
                self.jain_fairness,
            )
        ):
            raise ValueError("ratio metrics must be in [0, 1]")
        if not tiers or any(
            not tier or value < 0 or value > 1 or not math.isfinite(value)
            for tier, value in tiers.items()
        ):
            raise ValueError("per-tier attainment must be a non-empty ratio mapping")
        if self.minimum_tier_slo_attainment != min(tiers.values()):
            raise ValueError("minimum tier attainment must match the per-tier mapping")


@dataclass(frozen=True, slots=True)
class G12TwoVerdict:
    verdict: str
    strictly_improved_axes: tuple[str, ...]
    violated_axes: tuple[str, ...]


def evaluate_g12_2(
    baseline: PlacementStrategyMetrics,
    candidate: PlacementStrategyMetrics,
) -> G12TwoVerdict:
    """Apply the frozen G12-2 Pareto gate without display-value rounding."""
    if (
        baseline.strategy_id != "HYBRID"
        or candidate.strategy_id == baseline.strategy_id
    ):
        raise ValueError("G12-2 requires HYBRID baseline and a distinct candidate")
    if set(candidate.per_tier_slo_attainment) != set(baseline.per_tier_slo_attainment):
        raise ValueError("G12-2 requires the same frozen tier set")
    improved = tuple(
        name
        for name, better in (
            ("token_hit_rate", candidate.token_hit_rate > baseline.token_hit_rate),
            (
                "request_load_max_mean",
                candidate.request_load_max_mean < baseline.request_load_max_mean,
            ),
            ("p_queue_p95", candidate.p_queue_p95 < baseline.p_queue_p95),
        )
        if better
    )
    violated = tuple(
        name
        for name, violation in (
            (
                "token_hit_rate",
                candidate.token_hit_rate
                < baseline.token_hit_rate - HIT_RATE_EPSILON_ABSOLUTE,
            ),
            (
                "request_load_max_mean",
                candidate.request_load_max_mean
                > baseline.request_load_max_mean + LOAD_SKEW_EPSILON_ABSOLUTE,
            ),
            (
                "p_queue_p95",
                candidate.p_queue_p95
                > baseline.p_queue_p95 * (1 + P_QUEUE_P95_EPSILON_RELATIVE),
            ),
            (
                "strict_useful_token_goodput",
                candidate.strict_useful_token_goodput
                < baseline.strict_useful_token_goodput,
            ),
            (
                "strict_useful_output_token_goodput",
                candidate.strict_useful_output_token_goodput
                < baseline.strict_useful_output_token_goodput * 0.999,
            ),
            (
                "minimum_tier_slo_attainment",
                candidate.minimum_tier_slo_attainment < 0.80,
            ),
            ("jain_fairness", candidate.jain_fairness < 0.90),
            (
                "relative_tier_slo_attainment",
                any(
                    candidate.per_tier_slo_attainment[tier]
                    < baseline.per_tier_slo_attainment[tier] - 0.02
                    for tier in baseline.per_tier_slo_attainment
                ),
            ),
        )
        if violation
    )
    return G12TwoVerdict(
        "PASS_PROVISIONAL"
        if len(improved) >= 2 and not violated
        else "KILL_ENFORCEMENT",
        improved,
        violated,
    )


def _stable_index(key: str, size: int) -> int:
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") % size
