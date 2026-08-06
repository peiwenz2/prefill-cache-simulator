"""M12.2 causal-home placement with an explicit marginal-cost ledger."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass


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


def _stable_index(key: str, size: int) -> int:
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") % size
