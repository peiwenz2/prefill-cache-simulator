from __future__ import annotations

from dataclasses import dataclass

import pytest

from prefill_cache_sim.cache import LocalNodeCache
from prefill_cache_sim.continuation import ContinuationPrefixBuilder
from prefill_cache_sim.domain import (
    BlockRef,
    CacheVisibility,
    ClusterView,
    NodeSnapshot,
    ReplayMode,
    Request,
    Selection,
)
from prefill_cache_sim.eviction import FifoPolicy
from prefill_cache_sim.processor import LocalPrefillRequestProcessor


def make_request(blocks: list[int], sizes: list[int] | None = None) -> Request:
    sizes = sizes or [512] * len(blocks)
    return Request(
        request_id=f"r-{blocks}",
        logical_request_id=f"logical-{blocks}",
        attempt_index=0,
        arrival_ms=0,
        input_tokens=sum(sizes),
        output_tokens=10,
        prefix_blocks=tuple(BlockRef.trace(value) for value in blocks),
        block_token_sizes=tuple(sizes),
        affinity_key=None,
    )


@dataclass
class FirstNodeSelector:
    def select(self, request: Request, view: ClusterView) -> Selection:
        del request
        return Selection(view.nodes[0].node_id, 0.0, {}, None)


def snapshot(node_id: str = "p0") -> NodeSnapshot:
    return NodeSnapshot(
        node_id=node_id,
        alive=True,
        cache_capacity_blocks=4,
        resident_blocks=0,
        running_requests=0,
        running_tokens=0,
        prefilling_requests=0,
        queued_uncached_tokens=0,
        available_at_ms=0.0,
        last_selected_ms=0.0,
    )


def test_continuous_prefix_stops_at_first_miss() -> None:
    cache = LocalNodeCache("p0", 4, FifoPolicy())
    cache.place(make_request([1, 2, 3]), now_ms=1, pinned=frozenset())
    cache.remove_for_test(BlockRef.trace(2))
    lookup = cache.lookup(make_request([1, 2, 3]), now_ms=2)
    assert lookup.hit_blocks == 1
    assert lookup.hit_tokens == 512
    assert lookup.stranded_resident_blocks == 1


def test_capacity_never_exceeded_and_pinned_blocks_survive() -> None:
    cache = LocalNodeCache("p0", 2, FifoPolicy())
    first = make_request([1, 2])
    cache.place(first, now_ms=1, pinned=frozenset())
    cache.place(
        make_request([3]),
        now_ms=2,
        pinned=frozenset({BlockRef.trace(1)}),
    )
    assert cache.resident_count <= cache.capacity_blocks
    assert cache.contains(BlockRef.trace(1))


def test_fifo_lookup_does_not_refresh_oldest_block() -> None:
    cache = LocalNodeCache("p0", 2, FifoPolicy())
    cache.place(make_request([1, 2]), now_ms=1, pinned=frozenset())
    cache.lookup(make_request([1]), now_ms=2)
    cache.place(make_request([3]), now_ms=3, pinned=frozenset())
    assert not cache.contains(BlockRef.trace(1))
    assert cache.contains(BlockRef.trace(2))
    assert cache.contains(BlockRef.trace(3))


def test_oversized_request_records_rejected_placements() -> None:
    cache = LocalNodeCache("p0", 2, FifoPolicy())
    result = cache.place(make_request([1, 2, 3]), now_ms=1, pinned=frozenset())
    assert result.resident_blocks == 2
    assert result.rejected_blocks == 1
    assert result.evicted_blocks == 0
    assert result.capacity_rejected_blocks == 1


def test_processor_returns_selection_lookup_and_placement() -> None:
    cache = LocalNodeCache("p0", 4, FifoPolicy())
    processor = LocalPrefillRequestProcessor(
        selector=FirstNodeSelector(),
        caches={"p0": cache},
    )
    view = ClusterView(now_ms=0, observed_at_ms=0, nodes=(snapshot(),))
    result = processor.process(make_request([1, 2]), view=view, now_ms=0)
    assert result.selection.node_id == "p0"
    assert result.lookup.hit_tokens == 0
    assert result.placement.inserted_blocks == 2


def test_domain_enums_pin_replay_and_visibility_vocabulary() -> None:
    assert ReplayMode.CACHE_ONLY.value == "CACHE_ONLY"
    assert ReplayMode.QUEUED.value == "QUEUED"
    assert CacheVisibility.INSERT_AT_COMPLETION.value == "INSERT_AT_COMPLETION"
    assert CacheVisibility.INFLIGHT_DEDUP_WAIT.value == "INFLIGHT_DEDUP_WAIT"


def test_request_rejects_token_accounting_mismatch() -> None:
    with pytest.raises(ValueError, match="block token sizes"):
        make_request([1, 2], [512, 1, 1])


def test_continuation_replaces_partial_tail_and_chains_generated_blocks() -> None:
    original = Request(
        request_id="attempt-0",
        logical_request_id="logical-0",
        attempt_index=0,
        arrival_ms=0,
        input_tokens=600,
        output_tokens=1000,
        prefix_blocks=(BlockRef.trace(1), BlockRef.trace(2)),
        block_token_sizes=(512, 88),
        affinity_key="family-a",
    )
    builder = ContinuationPrefixBuilder(
        trace_sha256="ab" * 32,
        block_size_tokens=512,
        trace_line_index_by_logical_id={"logical-0": 7},
    )
    continuation = builder.build(original, sent_tokens=600, arrival_ms=10)
    assert continuation.attempt_index == 1
    assert continuation.input_tokens == 1200
    assert continuation.prefix_blocks[0] == BlockRef.trace(1)
    assert all(
        block.kind.value == "GENERATED" for block in continuation.prefix_blocks[1:]
    )
    assert continuation.block_token_sizes == (512, 512, 176)
    assert continuation.affinity_key == "family-a"


def test_continuation_with_zero_sent_tokens_is_rejected() -> None:
    builder = ContinuationPrefixBuilder(
        trace_sha256="ab" * 32,
        block_size_tokens=512,
        trace_line_index_by_logical_id={"logical-[1]": 0},
    )
    with pytest.raises(ValueError, match="sent_tokens"):
        builder.build(make_request([1]), sent_tokens=0, arrival_ms=1)


def test_multi_round_continuation_extends_generated_chain() -> None:
    original = Request(
        request_id="a0",
        logical_request_id="logical-0",
        attempt_index=0,
        arrival_ms=0,
        input_tokens=512,
        output_tokens=1000,
        prefix_blocks=(BlockRef.trace(1),),
        block_token_sizes=(512,),
        affinity_key=None,
    )
    builder = ContinuationPrefixBuilder(
        trace_sha256="ab" * 32,
        block_size_tokens=512,
        trace_line_index_by_logical_id={"logical-0": 7},
    )
    first = builder.build(original, sent_tokens=512, arrival_ms=1)
    second = builder.build(first, sent_tokens=512, arrival_ms=2)
    assert second.attempt_index == 2
    assert second.prefix_blocks[:2] == first.prefix_blocks
    assert second.prefix_blocks[-1].kind.value == "GENERATED"
