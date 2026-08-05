"""Deterministic LOCAL_ONLY replay engine."""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from .cache import LocalNodeCache
from .domain import (
    BlockRef,
    CacheVisibility,
    ClusterView,
    NodeSnapshot,
    ReplayMode,
    Request,
    Selection,
    ViewMode,
)
from .identity import DeterministicStream, NamedSeedManager
from .interfaces import BlockEvictionPolicy, PrefillNodeSelector


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    replay_mode: ReplayMode
    visibility: CacheVisibility
    view_mode: ViewMode
    node_count: int
    capacity_blocks_per_node: tuple[int, ...]
    prefill_uncached_token_ms: float
    view_delay_ms: float = 0.0
    prefill_concurrency_per_node: int = 1
    view_loss_probability: float = 0.0
    root_seed: int = 0

    @classmethod
    def cache_only(cls, node_count: int, capacity_blocks: int) -> SimulationConfig:
        return cls(
            ReplayMode.CACHE_ONLY,
            CacheVisibility.INSERT_AT_COMPLETION,
            ViewMode.ORACLE,
            node_count,
            (capacity_blocks,) * node_count,
            0.0,
        )

    @classmethod
    def queued(
        cls,
        node_count: int,
        capacity_blocks: int,
        prefill_uncached_token_ms: float,
        *,
        visibility: CacheVisibility = CacheVisibility.INSERT_AT_COMPLETION,
        view_mode: ViewMode = ViewMode.ORACLE,
        view_delay_ms: float = 0.0,
        prefill_concurrency_per_node: int = 1,
        view_loss_probability: float = 0.0,
        root_seed: int = 0,
    ) -> SimulationConfig:
        return cls(
            ReplayMode.QUEUED,
            visibility,
            view_mode,
            node_count,
            (capacity_blocks,) * node_count,
            prefill_uncached_token_ms,
            view_delay_ms,
            prefill_concurrency_per_node,
            view_loss_probability,
            root_seed,
        )

    def __post_init__(self) -> None:
        if (
            self.node_count <= 0
            or len(self.capacity_blocks_per_node) != self.node_count
        ):
            raise ValueError("capacity vector must match positive node_count")
        if any(capacity <= 0 for capacity in self.capacity_blocks_per_node):
            raise ValueError("node capacities must be positive")
        if (
            self.replay_mode is ReplayMode.QUEUED
            and self.prefill_uncached_token_ms <= 0
        ):
            raise ValueError("queued replay requires positive token service time")
        if self.prefill_concurrency_per_node <= 0:
            raise ValueError("prefill concurrency must be positive")
        if not 0 <= self.view_loss_probability <= 1:
            raise ValueError("view loss probability must be in [0, 1]")
        if self.replay_mode is ReplayMode.CACHE_ONLY and (
            self.visibility is not CacheVisibility.INSERT_AT_COMPLETION
            or self.view_mode is not ViewMode.ORACLE
            or self.view_delay_ms != 0
            or self.view_loss_probability != 0
            or self.prefill_concurrency_per_node != 1
        ):
            raise ValueError(
                "CACHE_ONLY requires completion visibility and oracle view"
            )


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    request_id: str
    logical_request_id: str
    attempt_index: int
    node_id: str
    arrival_ms: float
    start_ms: float
    finish_ms: float
    hit_blocks: int
    hit_tokens: int
    input_tokens: int
    stranded_resident_blocks: int
    selection_score: float
    selection_components: Mapping[str, float | int | str]
    fallback_reason: str | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selection_components",
            MappingProxyType(dict(self.selection_components)),
        )


@dataclass(frozen=True, slots=True)
class SimulationReport:
    completed_requests: int
    logical_input_tokens: int
    attempt_input_tokens: int
    hit_blocks: int
    block_references: int
    hit_tokens: int
    input_tokens: int
    block_ref_hit_rate: float
    token_weighted_hit_rate: float
    logical_block_ref_hit_rate: float
    logical_token_weighted_hit_rate: float
    eviction_count: int
    rejected_placements: int
    queue_wait_ms_max: float
    per_node_requests: tuple[int, ...]
    per_node_input_tokens: tuple[int, ...]
    per_node_uncached_tokens: tuple[int, ...]
    unique_resident_blocks: int
    replica_factor: float
    inflight_wait_tokens: int
    inflight_wait_ms: float
    eviction_regret: int
    one_hit_pollution: int
    decisions: tuple[DecisionRecord, ...]


@dataclass(slots=True)
class _NodeRuntime:
    node_id: str
    cache: LocalNodeCache
    available_at_ms: float = 0.0
    last_selected_ms: float = 0.0
    requests: int = 0
    input_tokens: int = 0
    uncached_tokens: int = 0
    queued_uncached_tokens: int = 0
    active_requests: int = 0
    queue: list[_QueueItem] = field(default_factory=list)
    eviction_count: int = 0
    rejected_placements: int = 0
    active_uncached_tokens: int = 0
    inflight: dict[BlockRef, float] = field(default_factory=dict)
    inflight_wait_tokens: int = 0
    inflight_wait_ms: float = 0.0
    active_pins: dict[BlockRef, int] = field(default_factory=dict)
    waiter_pins: set[BlockRef] = field(default_factory=set)


@dataclass(slots=True)
class _QueueItem:
    request: Request
    selection: Selection
    queued_uncached_tokens: int
    scheduled: bool = False


@dataclass(order=True, slots=True)
class _Event:
    time_ms: float
    priority: int
    sequence: int
    kind: str = field(compare=False)
    request: Request = field(compare=False)
    selection: Selection | None = field(compare=False, default=None)
    hit_blocks: int = field(compare=False, default=0)
    hit_tokens: int = field(compare=False, default=0)
    start_ms: float = field(compare=False, default=0.0)
    wait_started_ms: float | None = field(compare=False, default=None)
    wait_tokens: int = field(compare=False, default=0)
    stranded_resident_blocks: int = field(compare=False, default=0)


class LocalSimulator:
    def __init__(
        self,
        selector: PrefillNodeSelector,
        eviction_factory: Callable[[], BlockEvictionPolicy],
        config: SimulationConfig,
    ) -> None:
        self._selector = selector
        self._eviction_factory = eviction_factory
        self._config = config
        self._cached_view: ClusterView | None = None
        self._loss_stream: DeterministicStream | None = None

    def _nodes(self) -> list[_NodeRuntime]:
        return [
            _NodeRuntime(
                f"p{index}",
                LocalNodeCache(f"p{index}", capacity, self._eviction_factory()),
            )
            for index, capacity in enumerate(self._config.capacity_blocks_per_node)
        ]

    def _view(
        self, nodes: list[_NodeRuntime], now_ms: float, request: Request
    ) -> ClusterView:
        if (
            self._config.view_mode is ViewMode.DELAYED
            and self._cached_view is not None
            and now_ms - self._cached_view.observed_at_ms < self._config.view_delay_ms
        ):
            snapshots = self._cached_view.nodes
            observed_at = self._cached_view.observed_at_ms
        else:
            observed_at = now_ms
            snapshots = tuple(
                NodeSnapshot(
                    node.node_id,
                    True,
                    node.cache.capacity_blocks,
                    node.cache.resident_count,
                    node.active_requests,
                    node.active_uncached_tokens,
                    node.active_requests,
                    node.queued_uncached_tokens,
                    node.available_at_ms,
                    node.last_selected_ms,
                )
                for node in nodes
            )
        cache_hits = {
            node.node_id: (
                node.cache.preview(request)
                if self._config.view_mode is ViewMode.ORACLE
                else node.cache.preview_at(request, observed_at)
            )
            for node in nodes
            if not self._drop_cache_stats()
        }
        view = ClusterView(now_ms, observed_at, snapshots, None, cache_hits)
        if self._config.view_mode is ViewMode.DELAYED:
            self._cached_view = view
        return view

    def run(self, requests: Iterable[Request]) -> SimulationReport:
        ordered = tuple(requests)
        self._cached_view = None
        self._loss_stream = NamedSeedManager(self._config.root_seed).stream(
            "cluster-view/loss"
        )
        if any(
            left.arrival_ms > right.arrival_ms
            for left, right in zip(ordered, ordered[1:], strict=False)
        ):
            raise ValueError("requests must have non-decreasing arrival_ms")
        nodes = self._nodes()
        if self._config.replay_mode is ReplayMode.CACHE_ONLY:
            decisions = self._run_cache_only(ordered, nodes)
        else:
            decisions = self._run_queued(ordered, nodes)
        return self._report(ordered, nodes, decisions)

    def _drop_cache_stats(self) -> bool:
        probability = self._config.view_loss_probability
        if probability <= 0:
            return False
        if probability >= 1:
            return True
        assert self._loss_stream is not None
        return self._loss_stream.randbelow(1_000_000) < int(probability * 1_000_000)

    def _run_cache_only(
        self, requests: tuple[Request, ...], nodes: list[_NodeRuntime]
    ) -> tuple[DecisionRecord, ...]:
        decisions: list[DecisionRecord] = []
        for request in requests:
            now = float(request.arrival_ms)
            selection = self._selector.select(request, self._view(nodes, now, request))
            node = self._node(nodes, selection.node_id)
            lookup = node.cache.lookup(request, now)
            placement = node.cache.place(
                request, now, frozenset(request.prefix_blocks[: lookup.hit_blocks])
            )
            node.eviction_count += placement.evicted_blocks
            node.rejected_placements += placement.rejected_blocks
            self._account_node(node, request, request.input_tokens - lookup.hit_tokens)
            decisions.append(
                DecisionRecord(
                    request.request_id,
                    request.logical_request_id,
                    request.attempt_index,
                    node.node_id,
                    now,
                    now,
                    now,
                    lookup.hit_blocks,
                    lookup.hit_tokens,
                    request.input_tokens,
                    lookup.stranded_resident_blocks,
                    selection.score,
                    selection.components,
                    selection.fallback_reason,
                )
            )
            node.last_selected_ms = now
        return tuple(decisions)

    def _run_queued(
        self, requests: tuple[Request, ...], nodes: list[_NodeRuntime]
    ) -> tuple[DecisionRecord, ...]:
        events = [
            _Event(float(request.arrival_ms), 2, index, "arrival", request)
            for index, request in enumerate(requests)
        ]
        heapq.heapify(events)
        decisions: list[DecisionRecord] = []
        sequence = len(events)
        while events:
            event = heapq.heappop(events)
            if event.kind == "finish":
                sequence = self._finish(event, nodes, events, decisions, sequence)
            elif event.kind == "start":
                sequence = self._start(event, nodes, events, sequence)
            else:
                view = self._view(nodes, event.time_ms, event.request)
                selection = self._selector.select(event.request, view)
                node = self._node(nodes, selection.node_id)
                node.last_selected_ms = event.time_ms
                visible_hit = 0
                if view.cache_hits is not None:
                    visible_hit = view.cache_hits.get(node.node_id, (0, 0))[1]
                queued_uncached = event.request.input_tokens - visible_hit
                node.queue.append(_QueueItem(event.request, selection, queued_uncached))
                node.queued_uncached_tokens += queued_uncached
                sequence = self._schedule_starts(events, node, event.time_ms, sequence)
        decisions.sort(key=lambda decision: decision.request_id)
        return tuple(decisions)

    def _start(
        self,
        event: _Event,
        nodes: list[_NodeRuntime],
        events: list[_Event],
        sequence: int,
    ) -> int:
        assert event.selection is not None
        node = self._node(nodes, event.selection.node_id)
        item = next(
            queued
            for queued in node.queue
            if queued.request.request_id == event.request.request_id
        )
        item.scheduled = False
        preview_hit_blocks, _ = node.cache.preview(event.request)
        preview_misses = event.request.prefix_blocks[preview_hit_blocks:]
        producers = [
            node.inflight[block] for block in preview_misses if block in node.inflight
        ]
        if self._config.visibility is CacheVisibility.INFLIGHT_DEDUP_WAIT and producers:
            wait_until = max(producers)
            wait_started = event.wait_started_ms or event.time_ms
            waited_blocks = set(preview_misses) & set(node.inflight)
            wait_tokens = sum(
                size
                for block, size in zip(
                    event.request.prefix_blocks,
                    event.request.block_token_sizes,
                    strict=True,
                )
                if block in waited_blocks
            )
            node.waiter_pins.update(waited_blocks)
            item.scheduled = True
            heapq.heappush(
                events,
                _Event(
                    wait_until,
                    1,
                    sequence,
                    "start",
                    event.request,
                    event.selection,
                    wait_started_ms=wait_started,
                    wait_tokens=event.wait_tokens + wait_tokens,
                ),
            )
            return sequence + 1
        lookup = node.cache.lookup(event.request, event.time_ms)
        node.queue.remove(item)
        uncached = event.request.input_tokens - lookup.hit_tokens
        node.queued_uncached_tokens -= item.queued_uncached_tokens
        node.active_requests += 1
        node.active_uncached_tokens += uncached
        finish = event.time_ms + uncached * self._config.prefill_uncached_token_ms
        node.available_at_ms = max(node.available_at_ms, finish)
        if event.wait_started_ms is not None:
            node.inflight_wait_ms += event.time_ms - event.wait_started_ms
            node.inflight_wait_tokens += event.wait_tokens
            node.waiter_pins.difference_update(event.request.prefix_blocks)
        for block in event.request.prefix_blocks[: lookup.hit_blocks]:
            node.active_pins[block] = node.active_pins.get(block, 0) + 1
        for block in lookup.missed_blocks:
            node.inflight.setdefault(block, finish)
        self._account_node(node, event.request, uncached)
        heapq.heappush(
            events,
            _Event(
                finish,
                0,
                sequence,
                "finish",
                event.request,
                item.selection,
                lookup.hit_blocks,
                lookup.hit_tokens,
                event.time_ms,
                stranded_resident_blocks=lookup.stranded_resident_blocks,
            ),
        )
        return sequence + 1

    def _finish(
        self,
        event: _Event,
        nodes: list[_NodeRuntime],
        events: list[_Event],
        decisions: list[DecisionRecord],
        sequence: int,
    ) -> int:
        assert event.selection is not None
        node = self._node(nodes, event.selection.node_id)
        pinned = frozenset(node.active_pins) | frozenset(node.waiter_pins)
        placement = node.cache.place(event.request, event.time_ms, pinned)
        node.eviction_count += placement.evicted_blocks
        node.rejected_placements += placement.rejected_blocks
        uncached = event.request.input_tokens - event.hit_tokens
        node.active_requests -= 1
        node.active_uncached_tokens -= uncached
        for block in event.request.prefix_blocks[: event.hit_blocks]:
            remaining = node.active_pins[block] - 1
            if remaining:
                node.active_pins[block] = remaining
            else:
                node.active_pins.pop(block)
        for block in event.request.prefix_blocks[event.hit_blocks :]:
            if node.inflight.get(block) == event.time_ms:
                node.inflight.pop(block)
        decisions.append(
            DecisionRecord(
                event.request.request_id,
                event.request.logical_request_id,
                event.request.attempt_index,
                node.node_id,
                float(event.request.arrival_ms),
                event.start_ms,
                event.time_ms,
                event.hit_blocks,
                event.hit_tokens,
                event.request.input_tokens,
                event.stranded_resident_blocks,
                event.selection.score,
                event.selection.components,
                event.selection.fallback_reason,
            )
        )
        return self._schedule_starts(events, node, event.time_ms, sequence)

    def _schedule_starts(
        self,
        events: list[_Event],
        node: _NodeRuntime,
        now_ms: float,
        sequence: int,
    ) -> int:
        capacity = self._config.prefill_concurrency_per_node
        scheduled = sum(item.scheduled for item in node.queue)
        for item in node.queue:
            if node.active_requests + scheduled >= capacity:
                break
            if item.scheduled:
                continue
            heapq.heappush(
                events,
                _Event(
                    now_ms,
                    1,
                    sequence,
                    "start",
                    item.request,
                    item.selection,
                ),
            )
            item.scheduled = True
            scheduled += 1
            sequence += 1
        return sequence

    @staticmethod
    def _node(nodes: list[_NodeRuntime], node_id: str) -> _NodeRuntime:
        for node in nodes:
            if node.node_id == node_id:
                return node
        raise ValueError(f"selector returned unknown node {node_id}")

    @staticmethod
    def _account_node(node: _NodeRuntime, request: Request, uncached: int) -> None:
        node.requests += 1
        node.input_tokens += request.input_tokens
        node.uncached_tokens += uncached

    @staticmethod
    def _report(
        requests: tuple[Request, ...],
        nodes: list[_NodeRuntime],
        decisions: tuple[DecisionRecord, ...],
    ) -> SimulationReport:
        hit_blocks = sum(decision.hit_blocks for decision in decisions)
        hit_tokens = sum(decision.hit_tokens for decision in decisions)
        block_refs = sum(len(request.prefix_blocks) for request in requests)
        input_tokens = sum(request.input_tokens for request in requests)
        logical_latest: dict[str, Request] = {}
        for request in requests:
            current = logical_latest.get(request.logical_request_id)
            if current is None or request.attempt_index > current.attempt_index:
                logical_latest[request.logical_request_id] = request
        logical_tokens = sum(
            request.input_tokens for request in logical_latest.values()
        )
        decisions_by_request = {decision.request_id: decision for decision in decisions}
        logical_block_refs = sum(
            len(request.prefix_blocks) for request in logical_latest.values()
        )
        logical_hit_blocks = sum(
            decisions_by_request[request.request_id].hit_blocks
            for request in logical_latest.values()
        )
        logical_hit_tokens = sum(
            decisions_by_request[request.request_id].hit_tokens
            for request in logical_latest.values()
        )
        all_resident = [block for node in nodes for block in node.cache.resident]
        unique_resident = len(set(all_resident))
        replica_factor = len(all_resident) / unique_resident if unique_resident else 0.0
        waits = [decision.start_ms - decision.arrival_ms for decision in decisions]
        return SimulationReport(
            len(decisions),
            logical_tokens,
            input_tokens,
            hit_blocks,
            block_refs,
            hit_tokens,
            input_tokens,
            hit_blocks / block_refs if block_refs else math.nan,
            hit_tokens / input_tokens if input_tokens else math.nan,
            logical_hit_blocks / logical_block_refs if logical_block_refs else math.nan,
            logical_hit_tokens / logical_tokens if logical_tokens else math.nan,
            sum(node.eviction_count for node in nodes),
            sum(node.rejected_placements for node in nodes),
            max(waits, default=0.0),
            tuple(node.requests for node in nodes),
            tuple(node.input_tokens for node in nodes),
            tuple(node.uncached_tokens for node in nodes),
            unique_resident,
            replica_factor,
            sum(node.inflight_wait_tokens for node in nodes),
            sum(node.inflight_wait_ms for node in nodes),
            sum(node.cache.eviction_regret for node in nodes),
            sum(node.cache.one_hit_pollution for node in nodes),
            decisions,
        )
