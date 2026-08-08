from __future__ import annotations

from prefill_cache_sim.affinity import OnlineConversationLinker, PrefixAnchor
from prefill_cache_sim.domain import (
    BlockRef,
    ClusterView,
    NodeSnapshot,
    Request,
)
from prefill_cache_sim.identity import NamedSeedManager
from prefill_cache_sim.selectors import (
    CalibratedTtftSelector,
    CentralizedMasterTtftSelector,
    LeastWorkSelector,
    RandomSelector,
    RoundRobinSelector,
    gb_prefix_bucket_selector,
    session_affinity_selector,
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


def node(node_id: str, load: int = 0, last_selected: float = 0) -> NodeSnapshot:
    return NodeSnapshot(
        node_id,
        True,
        100,
        0,
        int(load > 0),
        load,
        int(load > 0),
        0,
        float(load),
        last_selected,
    )


def view(
    loads: list[int], cache_hits: dict[str, tuple[int, int]] | None = None
) -> ClusterView:
    return ClusterView(
        0,
        0,
        tuple(node(f"p{i}", load) for i, load in enumerate(loads)),
        None,
        cache_hits or {},
    )


def test_random_is_reproducible_and_round_robin_balances() -> None:
    first = RandomSelector(NamedSeedManager(713).stream("selector/random"))
    second = RandomSelector(NamedSeedManager(713).stream("selector/random"))
    req = request([1])
    selections1 = [first.select(req, view([0, 0, 0])).node_id for _ in range(20)]
    selections2 = [second.select(req, view([0, 0, 0])).node_id for _ in range(20)]
    assert selections1 == selections2
    rr = RoundRobinSelector()
    assert [rr.select(req, view([0, 0])).node_id for _ in range(4)] == [
        "p0",
        "p1",
        "p0",
        "p1",
    ]


def test_least_work_uses_uncached_load_units() -> None:
    selected = LeastWorkSelector().select(request([1]), view([100, 20, 50]))
    assert selected.node_id == "p1"
    assert selected.components["load_tokens"] == 20


def test_prefix_bucket_is_stable_but_breaks_on_soft_overload() -> None:
    selector = gb_prefix_bucket_selector(2, soft_alpha=1.1)
    req = request([1, 2, 3])
    first = selector.select(req, view([0, 0, 0, 0]))
    again = selector.select(req, view([0, 0, 0, 0]))
    assert first.node_id == again.node_id
    loads = [0, 0, 0, 0]
    loads[int(first.node_id[1:])] = 100
    broken = selector.select(req, view(loads))
    assert broken.node_id != first.node_id
    assert broken.fallback_reason == "soft-overload"


def test_online_linker_is_online_and_family_cap_is_hard() -> None:
    linker = OnlineConversationLinker(
        minimum_shared_blocks=2,
        family_size_cap=2,
        hot_block_percentile=99,
    )
    first = linker.key_for(request([1, 2, 3]))
    second = linker.key_for(request([1, 2, 4]))
    third = linker.key_for(request([1, 2, 5]))
    fourth = linker.key_for(request([1, 2, 6]))
    assert first == second
    assert third != first
    assert fourth == third
    assert max(linker.family_sizes.values()) <= 2


def test_sticky_fallback_does_not_permanently_move_home() -> None:
    provider = OnlineConversationLinker(2, 8)
    selector = session_affinity_selector(provider, soft_alpha=1.1)
    req = request([1, 2, 3])
    home = selector.select(req, view([0, 0, 0]))
    loads = [0, 0, 0]
    loads[int(home.node_id[1:])] = 100
    assert selector.select(request([1, 2, 4]), view(loads)).node_id != home.node_id
    assert selector.select(request([1, 2, 5]), view([0, 0, 0])).node_id == home.node_id


def test_session_selector_keeps_linked_family_on_same_node() -> None:
    provider = OnlineConversationLinker(2, 8)
    selector = session_affinity_selector(provider, soft_alpha=10)
    first = selector.select(request([1, 2, 3]), view([0, 0, 0]))
    second = selector.select(request([1, 2, 4]), view([0, 0, 0]))
    assert first.node_id == second.node_id


def test_centralized_master_and_calibrated_prefer_warm_node_when_load_is_similar() -> (
    None
):
    req = request([1, 2])
    cluster = view([10, 10], {"p0": (0, 0), "p1": (2, 1024)})
    centralized = CentralizedMasterTtftSelector().select(req, cluster)
    calibrated = CalibratedTtftSelector().select(req, cluster)
    assert centralized.node_id == "p1"
    assert calibrated.node_id == "p1"
    assert calibrated.components["uncached_tokens"] == 0


def test_prefix_anchor_uses_last_block_when_request_is_short() -> None:
    assert PrefixAnchor(8).key_for(request([7, 8])) == "TRACE:8"
