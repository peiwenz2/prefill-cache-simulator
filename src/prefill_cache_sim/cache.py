"""LOCAL_ONLY node cache and topology."""

from __future__ import annotations

from dataclasses import dataclass

from .domain import BlockRef, CacheLookup, PlacementResult, Request
from .interfaces import BlockEvictionPolicy


class LocalNodeCache:
    def __init__(
        self,
        node_id: str,
        capacity_blocks: int,
        policy: BlockEvictionPolicy,
    ) -> None:
        if capacity_blocks <= 0:
            raise ValueError("capacity_blocks must be positive")
        self.node_id = node_id
        self.capacity_blocks = capacity_blocks
        self._policy = policy
        self._resident: set[BlockRef] = set()
        self._resident_snapshot: frozenset[BlockRef] = frozenset()
        self._snapshot_dirty = False
        self._history: dict[BlockRef, list[list[float | None]]] = {}
        self._accesses_since_insert: dict[BlockRef, int] = {}
        self.eviction_regret = 0
        self.one_hit_pollution = 0

    @property
    def resident_count(self) -> int:
        return len(self._resident)

    @property
    def resident(self) -> frozenset[BlockRef]:
        if self._snapshot_dirty:
            self._resident_snapshot = frozenset(self._resident)
            self._snapshot_dirty = False
        return self._resident_snapshot

    def contains(self, block: BlockRef) -> bool:
        return block in self._resident

    def lookup(self, request: Request, now_ms: float) -> CacheLookup:
        hit_blocks = 0
        hit_tokens = 0
        for block, size in zip(
            request.prefix_blocks, request.block_token_sizes, strict=True
        ):
            if block not in self._resident:
                intervals = self._history.get(block)
                if intervals and intervals[-1][1] is not None:
                    self.eviction_regret += 1
                break
            self._policy.on_lookup(block, now_ms)
            self._accesses_since_insert[block] = (
                self._accesses_since_insert.get(block, 0) + 1
            )
            hit_blocks += 1
            hit_tokens += size
        stranded = sum(
            block in self._resident for block in request.prefix_blocks[hit_blocks + 1 :]
        )
        return CacheLookup(
            self.node_id,
            hit_blocks,
            hit_tokens,
            request.prefix_blocks[hit_blocks:],
            stranded,
        )

    def preview(self, request: Request) -> tuple[int, int]:
        """Preview continuous-prefix hits without refreshing policy state."""
        hit_blocks = 0
        hit_tokens = 0
        for block, size in zip(
            request.prefix_blocks, request.block_token_sizes, strict=True
        ):
            if block not in self._resident:
                break
            hit_blocks += 1
            hit_tokens += size
        return hit_blocks, hit_tokens

    def preview_at(self, request: Request, observed_at_ms: float) -> tuple[int, int]:
        """Preview hits from historical residency at the observation timestamp."""
        hit_blocks = 0
        hit_tokens = 0
        for block, size in zip(
            request.prefix_blocks, request.block_token_sizes, strict=True
        ):
            intervals = self._history.get(block, ())
            resident_then = any(
                start is not None
                and start <= observed_at_ms
                and (end is None or observed_at_ms < end)
                for start, end in reversed(intervals)
            )
            if not resident_then:
                break
            hit_blocks += 1
            hit_tokens += size
        return hit_blocks, hit_tokens

    def place(
        self,
        request: Request,
        now_ms: float,
        pinned: frozenset[BlockRef],
    ) -> PlacementResult:
        inserted = 0
        evicted = 0
        admission_rejected = 0
        capacity_rejected = 0
        inserted_this_request: set[BlockRef] = set()
        for block in request.prefix_blocks:
            if block in self._resident:
                self._policy.on_lookup(block, now_ms)
                continue
            if not self._policy.admit(block, now_ms):
                admission_rejected += 1
                continue
            needed = max(0, len(self._resident) + 1 - self.capacity_blocks)
            victims = (
                self._policy.victims(
                    needed,
                    self._resident,
                    pinned | frozenset(inserted_this_request),
                )
                if needed
                else ()
            )
            if len(victims) < needed:
                capacity_rejected += 1
                continue
            for victim in victims:
                if self._accesses_since_insert.get(victim, 0) <= 1:
                    self.one_hit_pollution += 1
                self._resident.remove(victim)
                self._snapshot_dirty = True
                self._close_interval(victim, now_ms)
                self._policy.on_remove(victim)
                self._accesses_since_insert.pop(victim, None)
                evicted += 1
            self._resident.add(block)
            self._snapshot_dirty = True
            self._history.setdefault(block, []).append([now_ms, None])
            self._policy.on_insert(block, now_ms)
            self._accesses_since_insert[block] = 0
            inserted += 1
            inserted_this_request.add(block)
        return PlacementResult(
            self.node_id,
            inserted,
            evicted,
            admission_rejected,
            capacity_rejected,
            len(self._resident),
        )

    def remove_for_test(self, block: BlockRef) -> None:
        self._resident.remove(block)
        self._snapshot_dirty = True
        self._close_interval(block, float("inf"))
        self._policy.on_remove(block)

    def _close_interval(self, block: BlockRef, now_ms: float) -> None:
        intervals = self._history.get(block)
        if intervals and intervals[-1][1] is None:
            intervals[-1][1] = now_ms


@dataclass(slots=True)
class LocalOnlyTopology:
    caches: dict[str, LocalNodeCache]

    def lookup(self, node_id: str, request: Request, now_ms: float) -> CacheLookup:
        return self.caches[node_id].lookup(request, now_ms)

    def place(
        self,
        node_id: str,
        request: Request,
        now_ms: float,
        pinned: frozenset[BlockRef],
    ) -> PlacementResult:
        return self.caches[node_id].place(request, now_ms, pinned)
