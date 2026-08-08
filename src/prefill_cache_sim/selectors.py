"""Phase-A prefill node selectors."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

from .affinity import AffinityKeyProvider, OnlineConversationLinker, PrefixAnchor
from .domain import ClusterView, NodeSnapshot, Request, Selection
from .identity import DeterministicStream


def _alive(view: ClusterView) -> tuple[NodeSnapshot, ...]:
    nodes = view.alive_nodes()
    if not nodes:
        raise ValueError("no alive prefill nodes")
    return nodes


def _load(node: NodeSnapshot) -> int:
    return node.running_tokens + node.queued_uncached_tokens


def _least_work(nodes: tuple[NodeSnapshot, ...]) -> NodeSnapshot:
    return min(
        nodes, key=lambda node: (_load(node), node.last_selected_ms, node.node_id)
    )


@dataclass(slots=True)
class RandomSelector:
    stream: DeterministicStream

    def select(self, request: Request, view: ClusterView) -> Selection:
        del request
        nodes = _alive(view)
        index = self.stream.randbelow(len(nodes))
        return Selection(
            nodes[index].node_id, float(index), {"random_index": index}, None
        )


@dataclass(slots=True)
class RoundRobinSelector:
    index: int = 0

    def select(self, request: Request, view: ClusterView) -> Selection:
        del request
        nodes = _alive(view)
        index = self.index % len(nodes)
        self.index += 1
        return Selection(nodes[index].node_id, float(index), {"rr_index": index}, None)


@dataclass(frozen=True, slots=True)
class LeastWorkSelector:
    def select(self, request: Request, view: ClusterView) -> Selection:
        del request
        node = _least_work(_alive(view))
        load = _load(node)
        return Selection(node.node_id, float(load), {"load_tokens": load}, None)


@dataclass(frozen=True, slots=True)
class CapacityGate:
    soft_alpha: float = 1.25
    max_running_requests: int | None = None
    max_prefilling_requests: int | None = None
    max_running_tokens: int | None = None

    def eligible(self, node: NodeSnapshot) -> bool:
        return not (
            (
                self.max_running_requests is not None
                and node.running_requests >= self.max_running_requests
            )
            or (
                self.max_prefilling_requests is not None
                and node.prefilling_requests >= self.max_prefilling_requests
            )
            or (
                self.max_running_tokens is not None
                and node.running_tokens >= self.max_running_tokens
            )
        )

    def overloaded(
        self, primary: NodeSnapshot, nodes: tuple[NodeSnapshot, ...]
    ) -> bool:
        mean = sum(_load(node) for node in nodes) / len(nodes)
        return mean > 0 and _load(primary) > self.soft_alpha * mean


@dataclass(frozen=True, slots=True)
class BoundedFallback:
    max_secondary_candidates: int = 2

    def choose(
        self,
        primary_index: int,
        nodes: tuple[NodeSnapshot, ...],
        gate: CapacityGate,
    ) -> tuple[NodeSnapshot, bool]:
        candidates = tuple(
            nodes[(primary_index + offset) % len(nodes)]
            for offset in range(
                1, min(self.max_secondary_candidates, len(nodes) - 1) + 1
            )
        )
        eligible = tuple(
            candidate for candidate in candidates if gate.eligible(candidate)
        )
        if not eligible or all(
            gate.overloaded(candidate, nodes) for candidate in eligible
        ):
            cluster_eligible = tuple(node for node in nodes if gate.eligible(node))
            return _least_work(cluster_eligible or nodes), True
        return _least_work(eligible), False


def _stable_owner(key: str, nodes: tuple[NodeSnapshot, ...]) -> int:
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") % len(nodes)


@dataclass(slots=True)
class AffinitySelector:
    provider: AffinityKeyProvider
    gate: CapacityGate = field(default_factory=CapacityGate)
    fallback: BoundedFallback = field(default_factory=BoundedFallback)
    sticky: bool = False
    _last_node: dict[str, str] = field(default_factory=dict)

    def select(self, request: Request, view: ClusterView) -> Selection:
        nodes = _alive(view)
        key = self.provider.key_for(request)
        if key is None:
            node = _least_work(nodes)
            return Selection(
                node.node_id, float(_load(node)), {"affinity": 0}, "no-key"
            )
        by_id = {node.node_id: index for index, node in enumerate(nodes)}
        if self.sticky and key in self._last_node and self._last_node[key] in by_id:
            primary_index = by_id[self._last_node[key]]
        else:
            primary_index = _stable_owner(key, nodes)
        primary = nodes[primary_index]
        hard_broken = not self.gate.eligible(primary)
        soft_broken = self.gate.overloaded(primary, nodes)
        broken = hard_broken or soft_broken
        fail_open = False
        if broken:
            chosen, fail_open = self.fallback.choose(primary_index, nodes, self.gate)
        else:
            chosen = primary
        if self.sticky and not broken:
            self._last_node[key] = chosen.node_id
        components = {
            "affinity": 1,
            "affinity_broken": int(broken),
            "fail_open": int(fail_open),
            "primary_load": _load(primary),
            "chosen_load": _load(chosen),
        }
        return Selection(
            chosen.node_id,
            float(_load(chosen)),
            components,
            (
                "fail-open"
                if fail_open
                else "hard-capacity"
                if hard_broken
                else "soft-overload"
            )
            if broken
            else None,
        )


def gb_prefix_bucket_selector(
    anchor_k: int,
    *,
    soft_alpha: float = 1.25,
    max_secondary_candidates: int = 2,
    max_running_requests: int | None = None,
) -> AffinitySelector:
    return AffinitySelector(
        PrefixAnchor(anchor_k),
        CapacityGate(soft_alpha, max_running_requests=max_running_requests),
        BoundedFallback(max_secondary_candidates),
        sticky=False,
    )


def session_affinity_selector(
    provider: OnlineConversationLinker,
    *,
    soft_alpha: float = 1.25,
    max_secondary_candidates: int = 2,
    max_running_requests: int | None = None,
) -> AffinitySelector:
    return AffinitySelector(
        provider,
        CapacityGate(soft_alpha, max_running_requests=max_running_requests),
        BoundedFallback(max_secondary_candidates),
        sticky=True,
    )


@dataclass(frozen=True, slots=True)
class CentralizedMasterTtftSelector:
    cache_discount: float = 0.7
    top_fraction: float = 0.3
    similar_band_ratio: float = 0.1

    def select(self, request: Request, view: ClusterView) -> Selection:
        nodes = _alive(view)
        cache_hits = view.cache_hits or {}
        scored = []
        for node in nodes:
            hit = cache_hits.get(node.node_id, (0, 0))[1]
            score = _load(node) + request.input_tokens - self.cache_discount * hit
            scored.append((score, node, hit))
        scored.sort(key=lambda item: (item[0], item[1].node_id))
        top_count = max(1, math.ceil(len(scored) * self.top_fraction))
        best = scored[0][0]
        band = max(1.0, abs(best) * self.similar_band_ratio)
        candidates = [item for item in scored[:top_count] if item[0] <= best + band]
        score, node, hit = min(
            candidates, key=lambda item: (item[1].last_selected_ms, item[1].node_id)
        )
        return Selection(
            node.node_id,
            score,
            {"queue_tokens": _load(node), "hit_tokens": hit, "ttft_score": score},
            None if node.node_id in cache_hits else "no-stat",
        )


@dataclass(frozen=True, slots=True)
class CalibratedTtftSelector:
    prefill_uncached_token_ms: float = 1.0
    similar_band_ratio: float = 0.1

    def select(self, request: Request, view: ClusterView) -> Selection:
        nodes = _alive(view)
        cache_hits = view.cache_hits or {}
        scored = []
        for node in nodes:
            hit = cache_hits.get(node.node_id, (0, 0))[1]
            uncached = request.input_tokens - hit
            score = (_load(node) + uncached) * self.prefill_uncached_token_ms
            scored.append((score, node, hit, uncached))
        scored.sort(key=lambda item: (item[0], item[1].node_id))
        best = scored[0][0]
        band = max(1.0, abs(best) * self.similar_band_ratio)
        candidates = [item for item in scored if item[0] <= best + band]
        score, node, hit, uncached = min(
            candidates, key=lambda item: (item[1].last_selected_ms, item[1].node_id)
        )
        return Selection(
            node.node_id,
            score,
            {"hit_tokens": hit, "uncached_tokens": uncached, "ttft_ms": score},
            None if node.node_id in cache_hits else "no-stat",
        )
