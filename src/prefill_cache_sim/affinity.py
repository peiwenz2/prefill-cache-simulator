"""Online-only affinity key providers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

from .domain import BlockRef, Request


class AffinityKeyProvider(Protocol):
    def key_for(self, request: Request) -> str | None: ...


@dataclass(frozen=True, slots=True)
class NoneKey:
    def key_for(self, request: Request) -> None:
        del request
        return None


@dataclass(frozen=True, slots=True)
class PrefixAnchor:
    anchor_k: int

    def __post_init__(self) -> None:
        if self.anchor_k <= 0:
            raise ValueError("anchor_k must be positive")

    def key_for(self, request: Request) -> str:
        index = min(self.anchor_k, len(request.prefix_blocks)) - 1
        block = request.prefix_blocks[index]
        return f"{block.kind.value}:{block.value}"


@dataclass(slots=True)
class OnlineConversationLinker:
    """Link requests using only prefixes observed before the current request."""

    minimum_shared_blocks: int
    family_size_cap: int
    hot_block_percentile: float = 99.0
    max_prefix_depth: int = 32
    _prefix_to_family: dict[tuple[BlockRef, ...], str] = field(default_factory=dict)
    _family_sizes: Counter[str] = field(default_factory=Counter)
    _block_counts: Counter[BlockRef] = field(default_factory=Counter)
    _frequency_histogram: Counter[int] = field(default_factory=Counter)
    _next_family: int = 0
    _requests_seen: int = 0
    _hot_threshold: int = 8

    def __post_init__(self) -> None:
        if self.minimum_shared_blocks <= 0 or self.family_size_cap <= 0:
            raise ValueError("shared-block and family caps must be positive")
        if not 0 < self.hot_block_percentile <= 100:
            raise ValueError("hot_block_percentile must be in (0, 100]")

    def key_for(self, request: Request) -> str:
        blocks = request.prefix_blocks[: self.max_prefix_depth]
        family = self._find_family(blocks)
        if family is None or self._family_sizes[family] >= self.family_size_cap:
            family = f"family:{self._next_family}"
            self._next_family += 1
        self._family_sizes[family] += 1
        self._register(blocks, family)
        self._update_counts(blocks)
        self._requests_seen += 1
        if self._requests_seen % 256 == 0:
            self._refresh_hot_threshold()
        return family

    def _find_family(self, blocks: tuple[BlockRef, ...]) -> str | None:
        if len(blocks) < self.minimum_shared_blocks:
            return None
        for depth in range(len(blocks), self.minimum_shared_blocks - 1, -1):
            prefix = blocks[:depth]
            family = self._prefix_to_family.get(prefix)
            if family is not None and not self._is_hot_only(prefix):
                return family
        return None

    def _register(self, blocks: tuple[BlockRef, ...], family: str) -> None:
        for depth in range(self.minimum_shared_blocks, len(blocks) + 1):
            prefix = blocks[:depth]
            current = self._prefix_to_family.get(prefix)
            if current is None or self._family_sizes[current] >= self.family_size_cap:
                self._prefix_to_family[prefix] = family

    def _is_hot_only(self, prefix: tuple[BlockRef, ...]) -> bool:
        return all(self._block_counts[block] >= self._hot_threshold for block in prefix)

    def _update_counts(self, blocks: tuple[BlockRef, ...]) -> None:
        for block in blocks:
            previous = self._block_counts[block]
            if previous:
                self._frequency_histogram[previous] -= 1
                if self._frequency_histogram[previous] == 0:
                    del self._frequency_histogram[previous]
            current = previous + 1
            self._block_counts[block] = current
            self._frequency_histogram[current] += 1

    def _refresh_hot_threshold(self) -> None:
        population = len(self._block_counts)
        if not population:
            return
        target = int((self.hot_block_percentile / 100) * (population - 1))
        cumulative = 0
        for frequency in sorted(self._frequency_histogram):
            cumulative += self._frequency_histogram[frequency]
            if cumulative > target:
                self._hot_threshold = max(8, frequency)
                return

    @property
    def family_sizes(self) -> dict[str, int]:
        return dict(self._family_sizes)
