from __future__ import annotations

import math

from prefill_cache_sim.domain import BlockRef, Request
from prefill_cache_sim.kvs import (
    KvtCostModel,
    PrefixSource,
    RemotePrefixStore,
    TieredCacheTopology,
    TopologyMode,
)
from prefill_cache_sim.mooncake_selector import MooncakeAlgo1Selector


def request(blocks: list[int], sizes: tuple[int, ...] | None = None) -> Request:
    token_sizes = sizes or (512,) * len(blocks)
    return Request(
        "r",
        "r",
        0,
        0,
        sum(token_sizes),
        0,
        tuple(BlockRef.trace(block) for block in blocks),
        token_sizes,
        None,
    )


def test_shared_kvs_observation_conserves_tokens_and_does_not_fake_local_hit() -> None:
    store = RemotePrefixStore(capacity_blocks=8)
    topology = TieredCacheTopology.shared_kvs(
        node_ids=("p0", "p1"),
        local_capacity_blocks=2,
        remote_store=store,
    )
    req = request([1, 2, 3])
    store.place("p1", req, now_ms=0)
    observation = topology.observe("p0", req, now_ms=1)
    assert observation.local_hit_tokens == 0
    assert observation.remote_hit_tokens == req.input_tokens
    assert observation.recompute_tokens == 0
    assert (
        sum(segment.tokens for segment in observation.source_segments)
        == req.input_tokens
    )
    assert [segment.source for segment in observation.source_segments] == [
        PrefixSource.REMOTE_KVS
    ]


def test_observe_is_pure_and_timeout_fails_open_to_recompute() -> None:
    store = RemotePrefixStore(capacity_blocks=8)
    topology = TieredCacheTopology.shared_kvs(
        node_ids=("p0",), local_capacity_blocks=2, remote_store=store
    )
    req = request([1, 2])
    store.place("source", req, now_ms=0)
    before = store.snapshot()
    first = topology.observe("p0", req, now_ms=1, lookup_available=False)
    second = topology.observe("p0", req, now_ms=2, lookup_available=False)
    assert first == second
    assert first.lookup_unknown
    assert first.recompute_tokens == req.input_tokens
    assert store.snapshot() == before


def test_hybrid_uses_contiguous_local_hbm_dram_remote_then_recompute() -> None:
    store = RemotePrefixStore(capacity_blocks=8)
    topology = TieredCacheTopology.hybrid(
        node_ids=("p0",),
        local_capacity_blocks=2,
        local_dram_capacity_blocks=2,
        remote_store=store,
    )
    req = request([1, 2, 3, 4])
    topology.seed_local("p0", request([1]), now_ms=0)
    topology.seed_local_dram("p0", request([1, 2]), now_ms=0)
    store.place("p1", request([1, 2, 3]), now_ms=0)
    observation = topology.observe("p0", req, now_ms=1)
    assert observation.local_hit_tokens == 512
    assert observation.local_dram_hit_tokens == 512
    assert observation.remote_hit_tokens == 512
    assert observation.recompute_tokens == 512
    assert [segment.source for segment in observation.source_segments] == [
        PrefixSource.LOCAL_HBM,
        PrefixSource.LOCAL_DRAM,
        PrefixSource.REMOTE_KVS,
        PrefixSource.RECOMPUTE,
    ]


def test_kvt_cost_accounts_fixed_bandwidth_and_overlap_once() -> None:
    model = KvtCostModel(
        fixed_latency_ms=0.2,
        bandwidth_gbps=100,
        overlap_ratio=0.5,
        bytes_per_token=256,
    )
    raw = 0.2 + (1024 * 256 * 8) / (100 * 1_000_000)
    assert math.isclose(model.transfer_ms(1024), raw)
    assert math.isclose(model.non_overlapped_ms(1024), raw * 0.5)
    assert math.isclose(model.effective_ms(1024, recompute_ms=2), 2)


def test_remote_store_capacity_replica_and_hotspot_accounting() -> None:
    store = RemotePrefixStore(capacity_blocks=3)
    req = request([1, 2, 3])
    store.place("p0", req, now_ms=0)
    store.place("p1", req, now_ms=1)
    assert store.resident_copies == 3
    assert store.replica_factor == 1
    for _ in range(5):
        store.record_transfer(request([1]), 1)
    store.record_transfer(request([1, 2]), 2)
    assert store.hotspot_max_mean > 1


def test_mooncake_algo1_handles_zero_prefix_and_threshold_branch() -> None:
    store = RemotePrefixStore(capacity_blocks=8)
    topology = TieredCacheTopology.shared_kvs(
        node_ids=("p0", "p1"), local_capacity_blocks=4, remote_store=store
    )
    req = request([1, 2, 3, 4])
    topology.seed_local("p1", request([1, 2, 3]), now_ms=0)
    store.place("p1", request([1, 2, 3]), now_ms=0)
    observations = {
        node: topology.observe(node, req, now_ms=1) for node in ("p0", "p1")
    }
    selector = MooncakeAlgo1Selector(kvcache_balancing_threshold=2.0)
    selection = selector.select_from_observations(
        req, observations, queue_work={"p0": 0, "p1": 4096}
    )
    assert selection.node_id == "p0"
    assert selection.components["remote_plan"] == 1


def test_local_only_mode_has_no_remote_accounting() -> None:
    topology = TieredCacheTopology.local_only(("p0",), local_capacity_blocks=4)
    assert topology.mode is TopologyMode.LOCAL_ONLY
    observation = topology.observe("p0", request([1]), now_ms=0)
    assert observation.remote_hit_tokens == 0


def test_remote_store_does_not_evict_blocks_inserted_by_same_request() -> None:
    store = RemotePrefixStore(capacity_blocks=2)
    req = request([1, 2, 3])
    assert store.place("p0", req, now_ms=0) == 2
    assert store.observe_holder("p0", req)[0] == 2
