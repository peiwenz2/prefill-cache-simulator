"""Core immutable simulation domain types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


class BlockKind(StrEnum):
    TRACE = "TRACE"
    GENERATED = "GENERATED"


@dataclass(frozen=True, slots=True, order=True)
class BlockRef:
    kind: BlockKind
    value: int | str

    @classmethod
    def trace(cls, value: int) -> BlockRef:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("trace block ID must be a non-negative integer")
        return cls(BlockKind.TRACE, value)

    @classmethod
    def generated(cls, value: str) -> BlockRef:
        if len(value) != 64 or value.lower() != value:
            raise ValueError("generated block ID must be lowercase SHA-256 hex")
        try:
            bytes.fromhex(value)
        except ValueError as exc:
            raise ValueError("generated block ID must be hexadecimal") from exc
        return cls(BlockKind.GENERATED, value)


@dataclass(frozen=True, slots=True)
class Request:
    request_id: str
    logical_request_id: str
    attempt_index: int
    arrival_ms: float
    input_tokens: int
    output_tokens: int
    prefix_blocks: tuple[BlockRef, ...]
    block_token_sizes: tuple[int, ...]
    affinity_key: str | None

    def __post_init__(self) -> None:
        if not self.request_id or not self.logical_request_id:
            raise ValueError("request identities must be non-empty")
        if self.attempt_index < 0 or self.arrival_ms < 0:
            raise ValueError("attempt_index and arrival_ms must be non-negative")
        if len(self.prefix_blocks) != len(self.block_token_sizes):
            raise ValueError("prefix blocks and block token sizes differ")
        if not self.prefix_blocks or any(size <= 0 for size in self.block_token_sizes):
            raise ValueError("block token sizes must be non-empty and positive")
        if sum(self.block_token_sizes) != self.input_tokens:
            raise ValueError("block token sizes do not sum to input_tokens")
        if self.output_tokens < 0:
            raise ValueError("output_tokens must be non-negative")


@dataclass(frozen=True, slots=True)
class NodeSnapshot:
    node_id: str
    alive: bool
    cache_capacity_blocks: int
    resident_blocks: int
    running_requests: int
    running_tokens: int
    prefilling_requests: int
    queued_uncached_tokens: int
    available_at_ms: float
    last_selected_ms: float

    @property
    def load_tokens(self) -> int:
        return self.running_tokens + self.queued_uncached_tokens


@dataclass(frozen=True, slots=True)
class ClusterView:
    now_ms: float
    observed_at_ms: float
    nodes: tuple[NodeSnapshot, ...]
    cache_blocks: Mapping[str, frozenset[BlockRef]] | None = None
    cache_hits: Mapping[str, tuple[int, int]] | None = None

    def alive_nodes(self) -> tuple[NodeSnapshot, ...]:
        return tuple(node for node in self.nodes if node.alive)


@dataclass(frozen=True, slots=True)
class Selection:
    node_id: str
    score: float
    components: Mapping[str, float | int | str]
    fallback_reason: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))


@dataclass(frozen=True, slots=True)
class CacheLookup:
    node_id: str
    hit_blocks: int
    hit_tokens: int
    missed_blocks: tuple[BlockRef, ...]
    stranded_resident_blocks: int


@dataclass(frozen=True, slots=True)
class PlacementResult:
    node_id: str
    inserted_blocks: int
    evicted_blocks: int
    admission_rejected_blocks: int
    capacity_rejected_blocks: int
    resident_blocks: int

    @property
    def rejected_blocks(self) -> int:
        return self.admission_rejected_blocks + self.capacity_rejected_blocks


@dataclass(frozen=True, slots=True)
class PrefillResult:
    request_id: str
    selection: Selection
    lookup: CacheLookup
    placement: PlacementResult


class ReplayMode(StrEnum):
    CACHE_ONLY = "CACHE_ONLY"
    QUEUED = "QUEUED"


class CacheVisibility(StrEnum):
    INSERT_AT_COMPLETION = "INSERT_AT_COMPLETION"
    INFLIGHT_DEDUP_WAIT = "INFLIGHT_DEDUP_WAIT"


class ViewMode(StrEnum):
    ORACLE = "ORACLE"
    DELAYED = "DELAYED"
