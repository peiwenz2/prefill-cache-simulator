"""Mooncake Algorithm-1 compatible KVS-aware selector anchor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .domain import Request, Selection
from .kvs import PrefixObservation


@dataclass(frozen=True, slots=True)
class MooncakeAlgo1Selector:
    kvcache_balancing_threshold: float = 2.0
    transfer_token_cost: float = 0.25
    recompute_token_cost: float = 1.0

    def __post_init__(self) -> None:
        if self.kvcache_balancing_threshold <= 0:
            raise ValueError("balancing threshold must be positive")

    def select_from_observations(
        self,
        request: Request,
        observations: Mapping[str, PrefixObservation],
        queue_work: Mapping[str, int | float],
    ) -> Selection:
        if not observations:
            raise ValueError("Mooncake selector requires candidate observations")
        best_prefix = max(
            observation.local_hit_tokens
            + observation.local_dram_hit_tokens
            + observation.remote_hit_tokens
            for observation in observations.values()
        )
        scored: list[tuple[float, str, bool, int]] = []
        for node_id, observation in observations.items():
            local_prefix = (
                observation.local_hit_tokens + observation.local_dram_hit_tokens
            )
            remote_available = observation.remote_hit_tokens > 0
            use_remote = remote_available and (
                local_prefix == 0
                or best_prefix / local_prefix >= self.kvcache_balancing_threshold
            )
            if use_remote:
                transfer_tokens = min(
                    best_prefix - local_prefix, observation.remote_hit_tokens
                )
                planned_prefix = local_prefix + transfer_tokens
                work = (
                    transfer_tokens * self.transfer_token_cost
                    + (request.input_tokens - planned_prefix)
                    * self.recompute_token_cost
                )
            else:
                planned_prefix = local_prefix
                work = (
                    request.input_tokens - planned_prefix
                ) * self.recompute_token_cost
            score = queue_work.get(node_id, 0) + work
            scored.append((score, node_id, use_remote, planned_prefix))
        score, node_id, use_remote, planned_prefix = min(
            scored, key=lambda item: (item[0], item[1])
        )
        return Selection(
            node_id,
            score,
            {
                "remote_plan": int(use_remote),
                "planned_prefix_tokens": planned_prefix,
                "threshold": self.kvcache_balancing_threshold,
            },
            None,
        )
