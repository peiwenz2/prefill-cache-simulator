from __future__ import annotations

from prefill_cache_sim.cache import LocalNodeCache
from prefill_cache_sim.domain import BlockRef, Request
from prefill_cache_sim.eviction import (
    FifoPolicy,
    LruPolicy,
    SecondHitAdmissionPolicy,
    SlruPolicy,
)


def request(blocks: list[int]) -> Request:
    return Request(
        request_id=str(blocks),
        logical_request_id=str(blocks),
        attempt_index=0,
        arrival_ms=0,
        input_tokens=512 * len(blocks),
        output_tokens=0,
        prefix_blocks=tuple(BlockRef.trace(block) for block in blocks),
        block_token_sizes=(512,) * len(blocks),
        affinity_key=None,
    )


def test_lru_hit_rescues_block_but_fifo_hit_does_not() -> None:
    fifo = LocalNodeCache("fifo", 2, FifoPolicy())
    lru = LocalNodeCache("lru", 2, LruPolicy())
    for cache in (fifo, lru):
        cache.place(request([1, 2]), 0, frozenset())
        cache.lookup(request([1]), 1)
        cache.place(request([3]), 2, frozenset())
    assert not fifo.contains(BlockRef.trace(1))
    assert lru.contains(BlockRef.trace(1))
    assert not lru.contains(BlockRef.trace(2))


def test_slru_promotes_hit_to_protected_segment() -> None:
    cache = LocalNodeCache("p0", 3, SlruPolicy(protected_capacity=1))
    cache.place(request([1, 2, 3]), 0, frozenset())
    cache.lookup(request([1]), 1)
    cache.place(request([4]), 2, frozenset())
    assert cache.contains(BlockRef.trace(1))


def test_second_hit_admission_rejects_first_and_admits_second_reference() -> None:
    cache = LocalNodeCache("p0", 2, SecondHitAdmissionPolicy(ghost_capacity_blocks=8))
    first = cache.place(request([1]), 0, frozenset())
    second = cache.place(request([1]), 1, frozenset())
    assert first.admission_rejected_blocks == 1
    assert second.inserted_blocks == 1
    assert cache.contains(BlockRef.trace(1))
