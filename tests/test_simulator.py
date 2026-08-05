from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from prefill_cache_sim.domain import (
    BlockRef,
    CacheVisibility,
    ClusterView,
    ReplayMode,
    Request,
    Selection,
    ViewMode,
)
from prefill_cache_sim.eviction import FifoPolicy
from prefill_cache_sim.simulator import LocalSimulator, SimulationConfig
from prefill_cache_sim.trace import load_trace, to_simulation_requests


def request(
    request_id: str,
    arrival_ms: int,
    blocks: list[int],
    sizes: list[int] | None = None,
) -> Request:
    sizes = sizes or [512] * len(blocks)
    return Request(
        request_id=request_id,
        logical_request_id=request_id,
        attempt_index=0,
        arrival_ms=arrival_ms,
        input_tokens=sum(sizes),
        output_tokens=0,
        prefix_blocks=tuple(BlockRef.trace(value) for value in blocks),
        block_token_sizes=tuple(sizes),
        affinity_key=None,
    )


@dataclass
class FirstNode:
    def select(self, request: Request, view: ClusterView) -> Selection:
        del request
        return Selection(view.nodes[0].node_id, 0.0, {}, None)


@dataclass
class CaptureView:
    missing_cache_stats: int = 0

    def select(self, request: Request, view: ClusterView) -> Selection:
        del request
        if view.cache_hits is not None and not view.cache_hits:
            self.missing_cache_stats += 1
        return Selection(view.nodes[0].node_id, 0.0, {}, "no-stat")


@dataclass
class CaptureHits:
    hits: list[int]
    queued_loads: list[int]

    def select(self, request: Request, view: ClusterView) -> Selection:
        del request
        self.hits.append(view.cache_hits.get("p0", (0, 0))[1])
        self.queued_loads.append(view.nodes[0].queued_uncached_tokens)
        return Selection("p0", 0.0, {}, None)


@dataclass
class RoundRobin:
    index: int = 0

    def select(self, request: Request, view: ClusterView) -> Selection:
        del request
        node = view.alive_nodes()[self.index % len(view.alive_nodes())]
        self.index += 1
        return Selection(node.node_id, float(self.index), {}, None)


def test_cache_only_reaches_expected_continuous_prefix_hit() -> None:
    simulator = LocalSimulator(
        selector=FirstNode(),
        eviction_factory=FifoPolicy,
        config=SimulationConfig.cache_only(node_count=1, capacity_blocks=10),
    )
    report = simulator.run([request("r0", 0, [1, 2]), request("r1", 1, [1, 3])])
    assert report.completed_requests == 2
    assert report.block_ref_hit_rate == 1 / 4
    assert report.token_weighted_hit_rate == 1 / 4
    assert report.logical_input_tokens == report.attempt_input_tokens


def test_queued_request_observes_insert_when_it_reaches_service() -> None:
    simulator = LocalSimulator(
        selector=FirstNode(),
        eviction_factory=FifoPolicy,
        config=SimulationConfig.queued(
            node_count=1,
            capacity_blocks=10,
            prefill_uncached_token_ms=1.0,
        ),
    )
    report = simulator.run([request("r0", 0, [1]), request("r1", 1, [1])])
    assert report.completed_requests == 2
    assert report.block_ref_hit_rate == 0.5
    assert report.queue_wait_ms_max > 0


def test_queued_completion_order_is_deterministic() -> None:
    requests = [request("a", 0, [1]), request("b", 0, [2])]
    config = SimulationConfig.queued(
        node_count=1,
        capacity_blocks=10,
        prefill_uncached_token_ms=0.1,
    )
    first = LocalSimulator(FirstNode(), FifoPolicy, config).run(requests)
    second = LocalSimulator(FirstNode(), FifoPolicy, config).run(requests)
    assert first == second


def test_same_simulator_instance_has_no_residual_replay_state() -> None:
    requests = [request("a", 0, [1]), request("b", 1, [2])]
    simulator = LocalSimulator(
        FirstNode(), FifoPolicy, SimulationConfig.queued(1, 10, 1.0)
    )
    assert simulator.run(requests) == simulator.run(requests)


def test_inflight_dedup_waits_for_producer_instead_of_duplicate_compute() -> None:
    requests = [request("a", 0, [1]), request("b", 0, [1])]
    insert = SimulationConfig.queued(
        node_count=1,
        capacity_blocks=10,
        prefill_uncached_token_ms=1.0,
        prefill_concurrency_per_node=2,
    )
    dedup = SimulationConfig.queued(
        node_count=1,
        capacity_blocks=10,
        prefill_uncached_token_ms=1.0,
        visibility=CacheVisibility.INFLIGHT_DEDUP_WAIT,
        prefill_concurrency_per_node=2,
    )
    insert_report = LocalSimulator(FirstNode(), FifoPolicy, insert).run(requests)
    dedup_report = LocalSimulator(FirstNode(), FifoPolicy, dedup).run(requests)
    assert insert_report.block_ref_hit_rate == 0
    assert dedup_report.block_ref_hit_rate == 0.5
    assert dedup_report.inflight_wait_tokens == 512
    assert dedup_report.inflight_wait_ms == 512


def test_lossy_view_removes_cache_stats_but_remains_fail_open() -> None:
    selector = CaptureView()
    config = SimulationConfig.queued(
        node_count=1,
        capacity_blocks=10,
        prefill_uncached_token_ms=1.0,
        view_mode=ViewMode.DELAYED,
        view_loss_probability=1.0,
    )
    report = LocalSimulator(selector, FifoPolicy, config).run([request("a", 0, [1])])
    assert report.completed_requests == 1
    assert selector.missing_cache_stats == 1


def test_delayed_view_uses_historical_cache_residency() -> None:
    selector = CaptureHits([], [])
    config = SimulationConfig.queued(
        node_count=1,
        capacity_blocks=10,
        prefill_uncached_token_ms=1.0,
        view_mode=ViewMode.DELAYED,
        view_delay_ms=10_000,
    )
    report = LocalSimulator(selector, FifoPolicy, config).run(
        [request("a", 0, [1]), request("b", 600, [1])]
    )
    assert selector.hits == [0, 0]
    assert report.block_ref_hit_rate == 0.5


def test_warm_queued_request_contributes_zero_uncached_load() -> None:
    selector = CaptureHits([], [])
    config = SimulationConfig.queued(1, 10, 1.0)
    LocalSimulator(selector, FifoPolicy, config).run(
        [
            request("a", 0, [1]),
            request("b", 600, [1]),
            request("c", 600, [1]),
        ]
    )
    assert selector.queued_loads[-1] == 0


def test_concurrency_three_dedup_waiter_does_not_corrupt_queue_order() -> None:
    config = SimulationConfig.queued(
        1,
        10,
        1.0,
        visibility=CacheVisibility.INFLIGHT_DEDUP_WAIT,
        prefill_concurrency_per_node=3,
    )
    report = LocalSimulator(FirstNode(), FifoPolicy, config).run(
        [
            request("r1", 0, [1]),
            request("r2", 0, [2], [10]),
            request("r3", 0, [1]),
            request("r4", 0, [3]),
        ]
    )
    assert report.completed_requests == 4
    assert {decision.request_id for decision in report.decisions} == {
        "r1",
        "r2",
        "r3",
        "r4",
    }


def test_multi_node_accounting_conservation() -> None:
    inputs = [request(f"r{i}", i, [i]) for i in range(8)]
    report = LocalSimulator(
        RoundRobin(), FifoPolicy, SimulationConfig.cache_only(4, 4)
    ).run(inputs)
    assert sum(report.per_node_requests) == report.completed_requests == len(inputs)
    assert (
        sum(report.per_node_uncached_tokens) + report.hit_tokens == report.input_tokens
    )
    assert report.per_node_requests == (2, 2, 2, 2)
    assert report.replica_factor >= 1


def test_logical_and_attempt_weighted_accounting_are_separate() -> None:
    attempt0 = request("a0", 0, [1])
    attempt1 = Request(
        request_id="a1",
        logical_request_id="a0",
        attempt_index=1,
        arrival_ms=1,
        input_tokens=1024,
        output_tokens=0,
        prefix_blocks=(BlockRef.trace(1), BlockRef.trace(2)),
        block_token_sizes=(512, 512),
        affinity_key=None,
    )
    report = LocalSimulator(
        FirstNode(), FifoPolicy, SimulationConfig.cache_only(1, 10)
    ).run([attempt0, attempt1])
    assert report.logical_input_tokens == 1024
    assert report.attempt_input_tokens == 1536
    assert report.token_weighted_hit_rate == 512 / 1536
    assert report.logical_token_weighted_hit_rate == 0.5


def test_cache_only_rejects_queue_only_semantics() -> None:
    with pytest.raises(ValueError, match="CACHE_ONLY"):
        SimulationConfig(
            replay_mode=ReplayMode.CACHE_ONLY,
            visibility=CacheVisibility.INFLIGHT_DEDUP_WAIT,
            view_mode=ViewMode.ORACLE,
            node_count=1,
            capacity_blocks_per_node=(10,),
            prefill_uncached_token_ms=0,
        )


@pytest.mark.slow
def test_capacity_constrained_eviction_path_completes() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "mooncake_trace.jsonl"
    if not path.exists():
        pytest.skip("Mooncake trace fixture is not present")
    loaded = load_trace(path, block_size_tokens=512)
    requests = to_simulation_requests(loaded)[:3000]
    report = LocalSimulator(
        FirstNode(), FifoPolicy, SimulationConfig.cache_only(1, 64)
    ).run(requests)
    assert report.completed_requests == 3000
    assert report.eviction_count > 0
    assert report.rejected_placements > 0


@pytest.mark.slow
def test_one_node_large_cache_matches_independent_analyzer_ceiling() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "mooncake_trace.jsonl"
    if not path.exists():
        pytest.skip("Mooncake trace fixture is not present")
    loaded = load_trace(path, block_size_tokens=512)
    simulator = LocalSimulator(
        FirstNode(),
        FifoPolicy,
        SimulationConfig.cache_only(node_count=1, capacity_blocks=200_000),
    )
    report = simulator.run(to_simulation_requests(loaded))
    assert report.block_ref_hit_rate == pytest.approx(0.5525508359471951)
    assert report.token_weighted_hit_rate == pytest.approx(0.5707002329449369)
