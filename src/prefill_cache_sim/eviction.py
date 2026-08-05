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
