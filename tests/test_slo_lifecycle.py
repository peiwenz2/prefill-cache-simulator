from __future__ import annotations

import math

from prefill_cache_sim.domain import BlockRef, Request
from prefill_cache_sim.lifecycle import (
    LifecycleAction,
    LifecycleConfig,
    ProtocolId,
    simulate_lifecycle,
)
from prefill_cache_sim.slo import RelativeSloOverlay, Tier, TieredSloOverlay


def request(request_id: str, arrival: float = 0, output_tokens: int = 16) -> Request:
    return Request(
        request_id,
        request_id,
        0,
        arrival,
        512,
        output_tokens,
        (BlockRef.trace(1),),
        (512,),
        None,
    )


def test_relative_slo_overlay_scales_isolated_cost() -> None:
    profile = RelativeSloOverlay(ttft_multiplier=2, tpot_multiplier=3).profile(
        request("r"), isolated_prefill=10, isolated_tpot=2
    )
    assert profile.ttft_deadline == 20
    assert profile.tpot_deadline == 6
    assert profile.completion_deadline == 20 + 16 * 6


def test_tier_and_tenant_rng_streams_are_independent() -> None:
    overlay = TieredSloOverlay(seed=713, tenant_count=4)
    first = overlay.profile(request("a"), isolated_prefill=10, isolated_tpot=2)
    second = TieredSloOverlay(seed=713, tenant_count=4).profile(
        request("a"), isolated_prefill=10, isolated_tpot=2
    )
    assert first == second
    assert first.tier in Tier
    assert first.tenant_id.startswith("tenant-")


def test_tier_overlay_is_request_keyed_not_order_keyed() -> None:
    overlay = TieredSloOverlay(seed=713, tenant_count=4)
    expected = overlay.profile(request("b"), isolated_prefill=10, isolated_tpot=2)
    overlay.profile(request("a"), isolated_prefill=10, isolated_tpot=2)
    actual = overlay.profile(request("b"), isolated_prefill=10, isolated_tpot=2)
    assert actual == expected


def test_all_protocols_emit_auditable_lifecycle_actions() -> None:
    requests = (request("a"), request("b", arrival=1))
    for protocol in ProtocolId:
        report = simulate_lifecycle(
            requests,
            RelativeSloOverlay(4, 4),
            LifecycleConfig(protocol=protocol, prefill_nodes=1, decode_nodes=1),
        )
        actions = {action for record in report.records for action in record.actions}
        assert LifecycleAction.COMMIT_P in actions
        assert LifecycleAction.CONTINUE_D in actions
        assert report.completed_requests == 2


def test_gated_protocol_rejects_before_prefill_and_avoids_waste() -> None:
    requests = tuple(request(str(i), arrival=0, output_tokens=128) for i in range(5))
    loose = RelativeSloOverlay(1.01, 1.01)
    pd = simulate_lifecycle(
        requests,
        loose,
        LifecycleConfig(protocol=ProtocolId.P0_PD, prefill_nodes=1, decode_nodes=1),
    )
    gated = simulate_lifecycle(
        requests,
        loose,
        LifecycleConfig(
            protocol=ProtocolId.P2_GATED_PD, prefill_nodes=1, decode_nodes=1
        ),
    )
    assert gated.rejected_requests > 0
    assert gated.prefill_waste_work <= pd.prefill_waste_work


def test_goodput_partial_output_and_fairness_are_separate() -> None:
    report = simulate_lifecycle(
        (request("a", output_tokens=8), request("b", output_tokens=128)),
        TieredSloOverlay(seed=1, tenant_count=2),
        LifecycleConfig(
            protocol=ProtocolId.P3_GATED_DP,
            prefill_nodes=1,
            decode_nodes=1,
            tier_weights={Tier.STRICT: 2.0, Tier.STANDARD: 1.0, Tier.RELAXED: 0.5},
        ),
    )
    assert report.strict_completed_goodput <= report.lenient_completed_goodput
    assert report.partial_output_tokens >= 0
    assert 0 <= report.jain_fairness <= 1
    assert report.scalar_weights["STRICT"] == 2


def test_hold_reserve_commit_rollback_reject_accounting_conserves_requests() -> None:
    requests = tuple(request(str(i), arrival=0, output_tokens=64) for i in range(8))
    report = simulate_lifecycle(
        requests,
        RelativeSloOverlay(1.1, 1.1),
        LifecycleConfig(
            protocol=ProtocolId.P2_GATED_PD, prefill_nodes=1, decode_nodes=1
        ),
    )
    assert report.completed_requests + report.rejected_requests == len(requests)
    assert report.commit_count == report.completed_requests
    assert report.rollback_count <= report.reserve_count


def test_protocols_have_distinct_resource_ledgers() -> None:
    requests = (request("a", output_tokens=64), request("b", output_tokens=64))
    p0 = simulate_lifecycle(
        requests,
        RelativeSloOverlay(4, 4),
        LifecycleConfig(ProtocolId.P0_PD, 1, 1),
    )
    p1 = simulate_lifecycle(
        requests,
        RelativeSloOverlay(4, 4),
        LifecycleConfig(ProtocolId.P1_DP, 1, 1),
    )
    p4 = simulate_lifecycle(
        requests,
        RelativeSloOverlay(4, 4),
        LifecycleConfig(ProtocolId.P4_KVS_DECOUPLED, 1, 1),
    )
    assert p1.decode_hold_work > p0.decode_hold_work
    assert p4.store_occupancy_work > p0.store_occupancy_work


def test_true_waste_and_slo_missed_work_are_not_mixed() -> None:
    requests = tuple(request(str(i), output_tokens=64) for i in range(3))
    report = simulate_lifecycle(
        requests,
        RelativeSloOverlay(1.01, 1.01),
        LifecycleConfig(ProtocolId.P0_PD, 1, 1, max_post_prefill_wait=1),
    )
    assert report.prefill_waste_work > 0
    assert report.prefill_waste_work != report.slo_missed_prefill_work


def test_mid_decode_deadline_cutoff_preserves_partial_tokens() -> None:
    report = simulate_lifecycle(
        (request("a", output_tokens=128),),
        RelativeSloOverlay(1.1, 0.5),
        LifecycleConfig(ProtocolId.P0_PD, 1, 1),
    )
    assert 0 < report.partial_output_tokens < 128
    assert report.completed_requests == 0


def test_stale_gated_dp_can_rollback_after_prefill() -> None:
    requests = tuple(request(str(i), output_tokens=64) for i in range(8))
    report = simulate_lifecycle(
        requests,
        RelativeSloOverlay(1.5, 1.5),
        LifecycleConfig(
            ProtocolId.P3_GATED_DP,
            1,
            1,
            probe_staleness_work=10_000,
            tier_weights={Tier.STRICT: 1.5, Tier.STANDARD: 1.5, Tier.RELAXED: 1.5},
        ),
    )
    assert report.rollback_count > 0
    assert report.prefill_waste_work > 0
    fresh = simulate_lifecycle(
        requests,
        RelativeSloOverlay(1.5, 1.5),
        LifecycleConfig(
            ProtocolId.P3_GATED_DP,
            1,
            1,
            probe_staleness_work=0,
            tier_weights={Tier.STRICT: 1.5, Tier.STANDARD: 1.5, Tier.RELAXED: 1.5},
        ),
    )
    assert report.rollback_count != fresh.rollback_count


def test_pd_timeout_and_hold_ledgers_apply_to_every_pd_variant() -> None:
    requests = tuple(request(str(i), output_tokens=128) for i in range(5))
    for protocol in (
        ProtocolId.P0_PD,
        ProtocolId.P2_GATED_PD,
        ProtocolId.P4_KVS_DECOUPLED,
    ):
        report = simulate_lifecycle(
            requests,
            RelativeSloOverlay(10, 10),
            LifecycleConfig(
                protocol,
                1,
                1,
                max_post_prefill_wait=1,
                tier_weights={Tier.STRICT: 100, Tier.STANDARD: 100, Tier.RELAXED: 100},
            ),
        )
        assert report.prefill_waste_work > 0
        assert report.hold_count > 0
        assert report.decode_hold_work > 0


def test_rollbacks_only_count_executed_prefills_and_charge_dp_hold() -> None:
    requests = tuple(request(str(i), output_tokens=64) for i in range(4))
    report = simulate_lifecycle(
        requests,
        RelativeSloOverlay(1.5, 1.5),
        LifecycleConfig(
            ProtocolId.P3_GATED_DP,
            1,
            1,
            probe_staleness_work=10_000,
            tier_weights={Tier.STRICT: 10, Tier.STANDARD: 10, Tier.RELAXED: 10},
        ),
    )
    real_rollbacks = sum(
        LifecycleAction.ROLLBACK_P in record.actions
        and record.prefill_start is not None
        for record in report.records
    )
    assert report.rollback_count == real_rollbacks
    assert report.decode_hold_work > 0


def test_store_and_probe_costs_change_outcomes() -> None:
    requests = tuple(request(str(i), output_tokens=64) for i in range(8))
    cheap_store = simulate_lifecycle(
        requests,
        RelativeSloOverlay(1.2, 1.2),
        LifecycleConfig(ProtocolId.P4_KVS_DECOUPLED, 1, 1, store_put_work=0.1),
    )
    costly_store = simulate_lifecycle(
        requests,
        RelativeSloOverlay(1.2, 1.2),
        LifecycleConfig(ProtocolId.P4_KVS_DECOUPLED, 1, 1, store_put_work=64),
    )
    assert cheap_store.records != costly_store.records
    cheap_probe = simulate_lifecycle(
        requests,
        RelativeSloOverlay(1.2, 1.2),
        LifecycleConfig(
            ProtocolId.P2_GATED_PD,
            1,
            1,
            probe_cost=0.1,
            tier_weights={Tier.STRICT: 1.5, Tier.STANDARD: 1.5, Tier.RELAXED: 1.5},
        ),
    )
    costly_probe = simulate_lifecycle(
        requests,
        RelativeSloOverlay(1.2, 1.2),
        LifecycleConfig(
            ProtocolId.P2_GATED_PD,
            1,
            1,
            probe_cost=64,
            tier_weights={Tier.STRICT: 1.5, Tier.STANDARD: 1.5, Tier.RELAXED: 1.5},
        ),
    )
    assert cheap_probe.rejected_requests != costly_probe.rejected_requests


def test_prefill_work_conserves_for_every_protocol() -> None:
    requests = tuple(request(str(i), output_tokens=64) for i in range(6))
    prefill = 512 * 0.0568
    for protocol in ProtocolId:
        report = simulate_lifecycle(
            requests,
            RelativeSloOverlay(1.2, 1.2),
            LifecycleConfig(protocol, 1, 1),
        )
        issued = sum(record.prefill_start is not None for record in report.records)
        strict = sum(record.strict_slo_met for record in report.records)
        accounted = (
            strict * prefill
            + report.prefill_waste_work
            + report.slo_missed_prefill_work
        )
        assert math.isclose(accounted, issued * prefill)


def test_expired_before_decode_is_never_credited_complete() -> None:
    requests = tuple(request(str(i), output_tokens=64) for i in range(8))
    overlay = RelativeSloOverlay(1.5, 1.5)
    report = simulate_lifecycle(
        requests,
        overlay,
        LifecycleConfig(ProtocolId.P0_PD, 1, 1, max_post_prefill_wait=1e9),
    )
    for record, item in zip(report.records, requests, strict=True):
        if record.finish is not None:
            profile = overlay.profile(item, 29.0816, 1)
            deadline = item.arrival_ms + profile.completion_deadline
            assert record.decode_start is not None and record.decode_start < deadline


def test_store_ledger_conserves_on_terminal_paths() -> None:
    requests = tuple(request(str(i), output_tokens=128) for i in range(6))
    report = simulate_lifecycle(
        requests,
        RelativeSloOverlay(10, 10),
        LifecycleConfig(
            ProtocolId.P4_KVS_DECOUPLED,
            1,
            1,
            max_post_prefill_wait=1,
            store_put_work=4,
        ),
    )
    puts = sum(LifecycleAction.STORE_PUT in r.actions for r in report.records)
    assert math.isclose(report.store_occupancy_work, puts * 4)


def test_burst_goodput_denominator_is_protocol_independent() -> None:
    requests = tuple(request(str(i), output_tokens=32) for i in range(4))
    overlay = RelativeSloOverlay(1e6, 1e6)
    reports = [
        simulate_lifecycle(requests, overlay, LifecycleConfig(protocol, 1, 1))
        for protocol in (ProtocolId.P0_PD, ProtocolId.P4_KVS_DECOUPLED)
    ]
    assert reports[0].lenient_completed_goodput == reports[1].lenient_completed_goodput


def test_probe_cost_is_paid_on_realized_timeline() -> None:
    requests = tuple(request(str(i), output_tokens=32) for i in range(4))
    weights = {Tier.STRICT: 1e9, Tier.STANDARD: 1e9, Tier.RELAXED: 1e9}
    schedules = [
        tuple(
            record.finish
            for record in simulate_lifecycle(
                requests,
                RelativeSloOverlay(1e6, 1e6),
                LifecycleConfig(
                    ProtocolId.P2_GATED_PD,
                    1,
                    1,
                    probe_cost=cost,
                    tier_weights=weights,
                ),
            ).records
        )
        for cost in (0.1, 1000)
    ]
    assert schedules[0] != schedules[1]
