from __future__ import annotations

from prefill_cache_sim.domain import BlockRef, Request
from prefill_cache_sim.m5_experiment import M5Case, run_m5_case


def request(request_id: str, blocks: list[int]) -> Request:
    return Request(
        request_id,
        request_id,
        0,
        0,
        512 * len(blocks),
        0,
        tuple(BlockRef.trace(block) for block in blocks),
        (512,) * len(blocks),
        None,
    )


def test_m5_replay_conserves_local_remote_recompute_tokens() -> None:
    result = run_m5_case(
        M5Case("x", remote_capacity_blocks=8),
        (request("a", [1, 2]), request("b", [1, 2, 3])),
    )
    assert (
        result.local_hit_tokens
        + result.local_dram_hit_tokens
        + result.remote_hit_tokens
        + result.recompute_tokens
        == result.input_tokens
    )
    assert result.remote_hit_tokens > 0


def test_m5_timeout_fails_open_without_fabricating_remote_hit() -> None:
    result = run_m5_case(
        M5Case("timeout", remote_capacity_blocks=8, lookup_timeout_probability=1),
        (request("a", [1, 2]), request("b", [1, 2, 3])),
    )
    assert result.remote_hit_tokens == 0
    assert result.lookup_unknown_count == 2


def test_m5_result_records_cost_and_hotspot() -> None:
    result = run_m5_case(
        M5Case("hot", remote_capacity_blocks=8),
        tuple(request(str(index), [1, 2, index + 3]) for index in range(5)),
    )
    assert result.kvt_normalized_ms >= 0
    assert result.hotspot_max_mean >= 1


def test_local_only_never_books_remote_transfer() -> None:
    from prefill_cache_sim.kvs import TopologyMode

    result = run_m5_case(
        M5Case("local", topology_mode=TopologyMode.LOCAL_ONLY),
        (request("a", [1, 2]), request("b", [1, 2, 3])),
    )
    assert result.remote_hit_tokens == 0
    assert result.kvt_normalized_ms == 0


def test_self_held_store_prefix_is_local_dram_not_remote() -> None:
    result = run_m5_case(
        M5Case("self", node_count=1, local_capacity_blocks=1),
        (request("a", [1, 2]), request("b", [1, 2, 3])),
    )
    assert result.local_dram_hit_tokens > 0
    assert result.remote_hit_tokens == 0


def test_kvt_fixed_and_wire_split_reconciles() -> None:
    result = run_m5_case(
        M5Case("split"),
        (request("a", [1, 2]), request("b", [1, 2, 3])),
    )
    assert result.kvt_normalized_ms == (
        result.kvt_fixed_normalized_ms + result.kvt_wire_normalized_ms
    )
