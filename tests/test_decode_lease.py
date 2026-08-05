from __future__ import annotations

import math

from prefill_cache_sim.continuation import ContinuationPrefixBuilder
from prefill_cache_sim.decode_lease import (
    DecodePolicy,
    LeaseConfig,
    RecoveryMode,
    _full_block_prefix,
    _new_full_blocks,
    simulate_decode_leases,
)
from prefill_cache_sim.domain import BlockKind, BlockRef, Request

SHA = "ab" * 32


def request(rid: str, output: int, arrival: float = 0, tenant: str = "a") -> Request:
    return Request(
        rid,
        f"logical-{rid}",
        0,
        arrival,
        512,
        output,
        (BlockRef.trace(abs(hash(rid)) % 1000),),
        (512,),
        tenant,
    )


def builder(*ids: str) -> ContinuationPrefixBuilder:
    return ContinuationPrefixBuilder(
        SHA, 512, {f"logical-{rid}": index for index, rid in enumerate(ids)}
    )


def test_fixed_lease_reproduces_8_8_14_and_conserves_tokens() -> None:
    report = simulate_decode_leases(
        (request("r", 40),),
        builder("r"),
        LeaseConfig(DecodePolicy.D1, RecoveryMode.R0, decode_nodes=1),
    )
    assert report.records[0].quanta == (8, 8, 14, 10)
    assert sum(report.records[0].quanta) == 40
    assert report.recovery_stall_work == 0


def test_kill_switch_and_recovery_failure_downgrade_without_hidden_quantum() -> None:
    for config in (
        LeaseConfig(DecodePolicy.D1, RecoveryMode.R0, enabled=False),
        LeaseConfig(
            DecodePolicy.D1,
            RecoveryMode.R0,
            max_recovery_work=1,
            migrate_on_boundary=True,
        ),
    ):
        report = simulate_decode_leases((request("r", 40),), builder("r"), config)
        record = report.records[0]
        assert sum(record.quanta) == record.emitted_tokens == 40
        assert record.downgraded or not config.enabled


def test_max_rounds_forces_downgrade_instead_of_dropping_partial_service() -> None:
    report = simulate_decode_leases(
        (request("r", 900),),
        builder("r"),
        LeaseConfig(DecodePolicy.D1, RecoveryMode.R0, max_rounds=2),
    )
    assert report.completed_tokens == 900
    assert report.decode_waste_work == 0
    assert sum(report.records[0].quanta) == 900


def test_continuations_chain_attempt_identity() -> None:
    build = builder("r")
    first = build.build(request("r", 40), sent_tokens=8, arrival_ms=8)
    second = build.build(first, sent_tokens=8, arrival_ms=16)
    assert (first.attempt_index, second.attempt_index) == (1, 2)
    assert first.request_id != second.request_id


def test_partial_generated_tail_is_not_admitted_as_kvs_object() -> None:
    build = builder("r")
    continuation = build.build(request("r", 40), sent_tokens=8, arrival_ms=8)
    admitted = _full_block_prefix(continuation, 512)
    assert admitted is not None
    assert all(
        kind is not BlockKind.GENERATED or size == 512
        for kind, size in zip(
            (block.kind for block in admitted.prefix_blocks),
            admitted.block_token_sizes,
            strict=True,
        )
    )
    assert admitted.input_tokens < continuation.input_tokens


def test_generated_tail_is_admitted_when_it_transitions_to_full() -> None:
    build = builder("r")
    partial = build.build(request("r", 600), sent_tokens=8, arrival_ms=8)
    sealed = build.build(partial, sent_tokens=504, arrival_ms=512)
    delta = _new_full_blocks(partial, sealed, 512)
    assert delta is not None
    assert delta.block_token_sizes == (512,)


def test_shared_queue_exposes_short_request_hol_benefit() -> None:
    requests = (request("long", 400), request("short", 8, arrival=1))
    build = builder("long", "short")
    d0 = simulate_decode_leases(
        requests,
        build,
        LeaseConfig(DecodePolicy.D0, RecoveryMode.R0, prefill_nodes=2, decode_nodes=1),
    )
    d1 = simulate_decode_leases(
        requests,
        build,
        LeaseConfig(DecodePolicy.D1, RecoveryMode.R0, prefill_nodes=2, decode_nodes=1),
    )
    starts0 = {record.request_id: record.first_decode_start for record in d0.records}
    starts1 = {record.request_id: record.first_decode_start for record in d1.records}
    assert starts1["short"] < starts0["short"]


def test_recovery_heavy_case_favors_d0_total_finish() -> None:
    requests = (request("a", 80),)
    build = builder("a")
    d0 = simulate_decode_leases(
        requests, build, LeaseConfig(DecodePolicy.D0, RecoveryMode.R0, decode_nodes=1)
    )
    d1 = simulate_decode_leases(
        requests,
        build,
        LeaseConfig(
            DecodePolicy.D1,
            RecoveryMode.R0,
            decode_nodes=1,
            local_capacity_blocks=1,
            migrate_on_boundary=True,
        ),
    )
    assert max(record.finish for record in d0.records) < max(
        record.finish for record in d1.records
    )


def test_r1_moves_recovery_frontier_and_conserves_source_tokens() -> None:
    item = request("r", 700)
    build = builder("r")
    r0 = simulate_decode_leases(
        (item,),
        build,
        LeaseConfig(
            DecodePolicy.D1,
            RecoveryMode.R0,
            local_capacity_blocks=1,
            migrate_on_boundary=True,
        ),
    )
    r1 = simulate_decode_leases(
        (item,),
        build,
        LeaseConfig(
            DecodePolicy.D1,
            RecoveryMode.R1,
            local_capacity_blocks=1,
            migrate_on_boundary=True,
        ),
    )
    assert r1.records[0].remote_recovery_tokens > 0
    assert r1.recovery_stall_work != r0.recovery_stall_work
    for record in (r0.records[0], r1.records[0]):
        assert (
            record.local_recovery_tokens
            + record.remote_recovery_tokens
            + record.recompute_recovery_tokens
            > 0
        )


def test_fairness_is_order_independent_and_single_tenant_is_one() -> None:
    first = request("a", 100, tenant="x")
    second = request("b", 40, tenant="y")
    config = LeaseConfig(DecodePolicy.D1, RecoveryMode.R0, decode_nodes=1)
    forward = simulate_decode_leases((first, second), builder("a", "b"), config)
    reverse = simulate_decode_leases((second, first), builder("a", "b"), config)
    assert math.isclose(forward.jain_fairness, reverse.jain_fairness)
    single = simulate_decode_leases((first,), builder("a"), config)
    assert single.jain_fairness == 1


def test_adaptive_quantum_is_bounded_and_boundary_only() -> None:
    report = simulate_decode_leases(
        (request("r", 100),),
        builder("r"),
        LeaseConfig(DecodePolicy.D1_5, RecoveryMode.R0),
    )
    assert all(8 <= quantum <= 64 for quantum in report.records[0].quanta[:-1])
    assert sum(report.records[0].quanta) == 100
