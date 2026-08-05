"""Replaceable simulator interfaces."""

from __future__ import annotations

from collections.abc import Set
from typing import Protocol

from .domain import (
    BlockRef,
    CacheLookup,
    ClusterView,
    PlacementResult,
    PrefillResult,
    Request,
    Selection,
)


class PrefillNodeSelector(Protocol):
    def select(self, request: Request, view: ClusterView) -> Selection: ...


class BlockEvictionPolicy(Protocol):
    def on_lookup(self, block_id: BlockRef, now_ms: float) -> None: ...

    def on_insert(self, block_id: BlockRef, now_ms: float) -> None: ...

    def on_remove(self, block_id: BlockRef) -> None: ...

    def victims(
        self,
        required_blocks: int,
        resident: Set[BlockRef],
        pinned: frozenset[BlockRef],
    ) -> tuple[BlockRef, ...]: ...

    def admit(self, block_id: BlockRef, now_ms: float) -> bool: ...


class CacheTopology(Protocol):
    def lookup(self, node_id: str, request: Request, now_ms: float) -> CacheLookup: ...

    def place(
        self,
        node_id: str,
        request: Request,
        now_ms: float,
        pinned: frozenset[BlockRef],
    ) -> PlacementResult: ...


class PrefillRequestProcessor(Protocol):
    def process(
        self, request: Request, view: ClusterView, now_ms: float
    ) -> PrefillResult: ...
