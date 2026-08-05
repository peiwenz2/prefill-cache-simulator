"""Pure KVS observations and tiered prefix-cache topology for Phase B+."""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
from enum import StrEnum

from .cache import LocalNodeCache
from .domain import BlockRef, Request
from .eviction import LruPolicy


class TopologyMode(StrEnum):
    LOCAL_ONLY = "LOCAL_ONLY"
    SHARED_KVS = "SHARED_KVS"
    HYBRID = "HYBRID"


class PrefixSource(StrEnum):
    LOCAL_HBM = "LOCAL_HBM"
    LOCAL_DRAM = "LOCAL_DRAM"
    REMOTE_KVS = "REMOTE_KVS"
    RECOMPUTE = "RECOMPUTE"


@dataclass(frozen=True, slots=True)
class SourceSegment:
    source: PrefixSource
    start_block: int
    end_block: int
    tokens: int
    holder_id: str | None = None

    def __post_init__(self) -> None:
        if self.start_block < 0 or self.end_block <= self.start_block:
            raise ValueError("source segment must be a non-empty forward range")
        if self.tokens <= 0:
            raise ValueError("source segment tokens must be positive")


@dataclass(frozen=True, slots=True)
class PrefixObservation:
    node_id: str
    input_tokens: int
    local_hit_blocks: int
    local_hit_tokens: int
    local_dram_hit_blocks: int
    local_dram_hit_tokens: int
    remote_hit_blocks: int
    remote_hit_tokens: int
    recompute_tokens: int
    best_remote_holder: str | None
    source_segments: tuple[SourceSegment, ...]
    lookup_unknown: bool = False

    def __post_init__(self) -> None:
        accounted = (
            self.local_hit_tokens
            + self.local_dram_hit_tokens
            + self.remote_hit_tokens
            + self.recompute_tokens
        )
        if accounted != self.input_tokens:
            raise ValueError("prefix observation does not conserve input tokens")
        if sum(segment.tokens for segment in self.source_segments) != self.input_tokens:
            raise ValueError("source segments do not conserve input tokens")
        cursor = 0
        for segment in self.source_segments:
            if segment.start_block != cursor:
                raise ValueError("source segments must be contiguous")
            cursor = segment.end_block


@dataclass(frozen=True, slots=True)
class KvtCostModel:
    fixed_latency_ms: float
    bandwidth_gbps: float
    overlap_ratio: float
    bytes_per_token: int

    def __post_init__(self) -> None:
        if self.fixed_latency_ms < 0 or self.bandwidth_gbps <= 0:
            raise ValueError("KVT latency and bandwidth are invalid")
        if not 0 <= self.overlap_ratio <= 1 or self.bytes_per_token <= 0:
            raise ValueError("KVT overlap and bytes_per_token are invalid")

    def transfer_ms(self, tokens: int) -> float:
        if tokens < 0:
            raise ValueError("transfer tokens must be non-negative")
        if tokens == 0:
            return 0.0
        transfer_bits = tokens * self.bytes_per_token * 8
        return self.fixed_latency_ms + transfer_bits / (self.bandwidth_gbps * 1_000_000)

    def non_overlapped_ms(self, tokens: int) -> float:
        return self.transfer_ms(tokens) * (1 - self.overlap_ratio)

    def effective_ms(self, tokens: int, recompute_ms: float) -> float:
        return max(self.non_overlapped_ms(tokens), recompute_ms)


class RemotePrefixStore:
    """Bounded shared prefix store with per-holder objects and pure observation."""

    def __init__(self, capacity_blocks: int) -> None:
        if capacity_blocks <= 0:
            raise ValueError("remote capacity must be positive")
        self.capacity_blocks = capacity_blocks
        self._objects: OrderedDict[tuple[str, BlockRef], None] = OrderedDict()
        self._by_holder: dict[str, set[BlockRef]] = {}
        self._copy_counts: Counter[BlockRef] = Counter()
        self._transfer_counts: Counter[BlockRef] = Counter()
        self.evictions = 0

    def place(self, holder_id: str, request: Request, now_ms: float) -> int:
        del now_ms
        inserted = 0
        inserted_this_request: set[tuple[str, BlockRef]] = set()
        for block in request.prefix_blocks:
            key = (holder_id, block)
            if key in self._objects:
                self._objects.move_to_end(key)
                continue
            while len(self._objects) >= self.capacity_blocks:
                victim = next(
                    (
                        candidate
                        for candidate in self._objects
                        if candidate not in inserted_this_request
                    ),
                    None,
                )
                if victim is None:
                    break
                evicted_holder, evicted_block = victim
                self._objects.pop(victim)
                holder_blocks = self._by_holder[evicted_holder]
                holder_blocks.remove(evicted_block)
                if not holder_blocks:
                    self._by_holder.pop(evicted_holder)
                self._copy_counts[evicted_block] -= 1
                if self._copy_counts[evicted_block] == 0:
                    self._copy_counts.pop(evicted_block)
                    self._transfer_counts.pop(evicted_block, None)
                self.evictions += 1
            if len(self._objects) >= self.capacity_blocks:
                continue
            self._objects[key] = None
            self._by_holder.setdefault(holder_id, set()).add(block)
            self._copy_counts[block] += 1
            inserted_this_request.add(key)
            inserted += 1
        return inserted

    def observe(
        self, request: Request, *, exclude_holder: str | None = None
    ) -> tuple[int, int, str | None]:
        holders = tuple(
            holder for holder in sorted(self._by_holder) if holder != exclude_holder
        )
        best_blocks = 0
        best_holder: str | None = None
        for holder in holders:
            hit = 0
            for block in request.prefix_blocks:
                if block not in self._by_holder[holder]:
                    break
                hit += 1
            if hit > best_blocks:
                best_blocks = hit
                best_holder = holder
        hit_tokens = sum(request.block_token_sizes[:best_blocks])
        return best_blocks, hit_tokens, best_holder

    def observe_holder(self, holder_id: str, request: Request) -> tuple[int, int]:
        resident = self._by_holder.get(holder_id, set())
        hit = 0
        for block in request.prefix_blocks:
            if block not in resident:
                break
            hit += 1
        return hit, sum(request.block_token_sizes[:hit])

    def record_transfer(self, request: Request, hit_blocks: int) -> None:
        if not 0 <= hit_blocks <= len(request.prefix_blocks):
            raise ValueError("remote transfer block count is invalid")
        for block in request.prefix_blocks[:hit_blocks]:
            self._transfer_counts[block] += 1

    def snapshot(self) -> tuple[tuple[str, BlockRef], ...]:
        return tuple(self._objects)

    @property
    def resident_copies(self) -> int:
        return len(self._objects)

    @property
    def replica_factor(self) -> float:
        unique = len({block for _, block in self._objects})
        return len(self._objects) / unique if unique else 0.0

    @property
    def hotspot_max_mean(self) -> float:
        if not self._transfer_counts:
            return 0.0
        counts = tuple(count for count in self._transfer_counts.values() if count > 0)
        if not counts:
            return 0.0
        return max(counts) / (sum(counts) / len(counts))


class TieredCacheTopology:
    """Observe continuous prefix coverage without mutating cache recency."""

    def __init__(
        self,
        mode: TopologyMode,
        local_caches: dict[str, LocalNodeCache],
        *,
        local_dram_caches: dict[str, LocalNodeCache] | None = None,
        remote_store: RemotePrefixStore | None = None,
    ) -> None:
        self.mode = mode
        self.local_caches = local_caches
        self.local_dram_caches = local_dram_caches or {}
        self.remote_store = remote_store

    @classmethod
    def local_only(
        cls, node_ids: tuple[str, ...], local_capacity_blocks: int
    ) -> TieredCacheTopology:
        return cls(
            TopologyMode.LOCAL_ONLY,
            _caches(node_ids, local_capacity_blocks),
        )

    @classmethod
    def shared_kvs(
        cls,
        node_ids: tuple[str, ...],
        local_capacity_blocks: int,
        remote_store: RemotePrefixStore,
    ) -> TieredCacheTopology:
        return cls(
            TopologyMode.SHARED_KVS,
            _caches(node_ids, local_capacity_blocks),
            remote_store=remote_store,
        )

    @classmethod
    def hybrid(
        cls,
        node_ids: tuple[str, ...],
        local_capacity_blocks: int,
        local_dram_capacity_blocks: int,
        remote_store: RemotePrefixStore,
    ) -> TieredCacheTopology:
        return cls(
            TopologyMode.HYBRID,
            _caches(node_ids, local_capacity_blocks),
            local_dram_caches=_caches(node_ids, local_dram_capacity_blocks),
            remote_store=remote_store,
        )

    def seed_local(self, node_id: str, request: Request, now_ms: float) -> None:
        self.local_caches[node_id].place(request, now_ms, frozenset())

    def seed_local_dram(self, node_id: str, request: Request, now_ms: float) -> None:
        self.local_dram_caches[node_id].place(request, now_ms, frozenset())

    def observe(
        self,
        node_id: str,
        request: Request,
        now_ms: float,
        *,
        lookup_available: bool = True,
    ) -> PrefixObservation:
        del now_ms
        local_blocks, _ = self.local_caches[node_id].preview(request)
        dram_blocks = local_blocks
        if self.mode is TopologyMode.HYBRID:
            dram_blocks = max(
                local_blocks, self.local_dram_caches[node_id].preview(request)[0]
            )
        if self.remote_store is not None:
            own_store_blocks, _ = self.remote_store.observe_holder(node_id, request)
            dram_blocks = max(dram_blocks, own_store_blocks)
        remote_blocks = dram_blocks
        remote_holder: str | None = None
        lookup_unknown = not lookup_available and self.remote_store is not None
        if self.remote_store is not None and lookup_available:
            observed_blocks, _, remote_holder = self.remote_store.observe(
                request, exclude_holder=node_id
            )
            remote_blocks = max(dram_blocks, observed_blocks)
        ends = (local_blocks, dram_blocks, remote_blocks, len(request.prefix_blocks))
        sources = (
            PrefixSource.LOCAL_HBM,
            PrefixSource.LOCAL_DRAM,
            PrefixSource.REMOTE_KVS,
            PrefixSource.RECOMPUTE,
        )
        segments: list[SourceSegment] = []
        start = 0
        for end, source in zip(ends, sources, strict=True):
            if end <= start:
                continue
            segments.append(
                SourceSegment(
                    source,
                    start,
                    end,
                    sum(request.block_token_sizes[start:end]),
                    remote_holder if source is PrefixSource.REMOTE_KVS else node_id,
                )
            )
            start = end
        local_tokens = sum(request.block_token_sizes[:local_blocks])
        dram_tokens = sum(request.block_token_sizes[local_blocks:dram_blocks])
        remote_tokens = sum(request.block_token_sizes[dram_blocks:remote_blocks])
        recompute_tokens = sum(request.block_token_sizes[remote_blocks:])
        return PrefixObservation(
            node_id,
            request.input_tokens,
            local_blocks,
            local_tokens,
            dram_blocks - local_blocks,
            dram_tokens,
            remote_blocks - dram_blocks,
            remote_tokens,
            recompute_tokens,
            remote_holder,
            tuple(segments),
            lookup_unknown,
        )


def _caches(node_ids: tuple[str, ...], capacity: int) -> dict[str, LocalNodeCache]:
    return {
        node_id: LocalNodeCache(node_id, capacity, LruPolicy()) for node_id in node_ids
    }
