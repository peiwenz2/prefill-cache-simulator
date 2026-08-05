"""Eviction policy primitives."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Set

from .domain import BlockRef


class FifoPolicy:
    """Evict in first-in order; hits do not refresh recency."""

    def __init__(self) -> None:
        self._order: OrderedDict[BlockRef, None] = OrderedDict()

    def on_lookup(self, block_id: BlockRef, now_ms: float) -> None:
        del block_id, now_ms

    def on_insert(self, block_id: BlockRef, now_ms: float) -> None:
        del now_ms
        self._order.setdefault(block_id, None)

    def on_remove(self, block_id: BlockRef) -> None:
        self._order.pop(block_id, None)

    def admit(self, block_id: BlockRef, now_ms: float) -> bool:
        del block_id, now_ms
        return True

    def victims(
        self,
        required_blocks: int,
        resident: Set[BlockRef],
        pinned: frozenset[BlockRef],
    ) -> tuple[BlockRef, ...]:
        if required_blocks <= 0:
            return ()
        eligible = (
            block for block in self._order if block in resident and block not in pinned
        )
        victims: list[BlockRef] = []
        for block in eligible:
            victims.append(block)
            if len(victims) == required_blocks:
                break
        return tuple(victims)


class LruPolicy(FifoPolicy):
    """Refresh insertion order on every cache hit."""

    def on_lookup(self, block_id: BlockRef, now_ms: float) -> None:
        del now_ms
        if block_id in self._order:
            self._order.move_to_end(block_id)


class SlruPolicy:
    """Two-segment LRU with probation admission and hit promotion."""

    def __init__(self, protected_capacity: int) -> None:
        if protected_capacity <= 0:
            raise ValueError("protected_capacity must be positive")
        self._protected_capacity = protected_capacity
        self._probation: OrderedDict[BlockRef, None] = OrderedDict()
        self._protected: OrderedDict[BlockRef, None] = OrderedDict()

    def on_lookup(self, block_id: BlockRef, now_ms: float) -> None:
        del now_ms
        if block_id in self._protected:
            self._protected.move_to_end(block_id)
        elif block_id in self._probation:
            self._probation.pop(block_id)
            self._protected[block_id] = None
            if len(self._protected) > self._protected_capacity:
                demoted, _ = self._protected.popitem(last=False)
                self._probation[demoted] = None

    def on_insert(self, block_id: BlockRef, now_ms: float) -> None:
        del now_ms
        self._probation.setdefault(block_id, None)

    def on_remove(self, block_id: BlockRef) -> None:
        self._probation.pop(block_id, None)
        self._protected.pop(block_id, None)

    def admit(self, block_id: BlockRef, now_ms: float) -> bool:
        del block_id, now_ms
        return True

    def victims(
        self,
        required_blocks: int,
        resident: Set[BlockRef],
        pinned: frozenset[BlockRef],
    ) -> tuple[BlockRef, ...]:
        if required_blocks <= 0:
            return ()
        victims: list[BlockRef] = []
        for segment in (self._probation, self._protected):
            for block in segment:
                if block in resident and block not in pinned:
                    victims.append(block)
                    if len(victims) == required_blocks:
                        return tuple(victims)
        return tuple(victims)


class SecondHitAdmissionPolicy(LruPolicy):
    """Admit only after a second reference, with bounded FIFO ghost metadata."""

    def __init__(self, ghost_capacity_blocks: int) -> None:
        super().__init__()
        if ghost_capacity_blocks <= 0:
            raise ValueError("ghost capacity must be positive")
        self._ghost_capacity = ghost_capacity_blocks
        self._ghost: OrderedDict[BlockRef, None] = OrderedDict()

    def admit(self, block_id: BlockRef, now_ms: float) -> bool:
        del now_ms
        if block_id in self._ghost:
            self._ghost.pop(block_id)
            return True
        self._ghost[block_id] = None
        if len(self._ghost) > self._ghost_capacity:
            self._ghost.popitem(last=False)
        return False
