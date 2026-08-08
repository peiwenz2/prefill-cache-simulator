"""Phase-C protocol state machines with explicit resource and waste ledgers."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from .domain import Request
from .slo import RelativeSloOverlay, SloProfile, Tier, TieredSloOverlay


class ProtocolId(StrEnum):
    P0_PD = "P0_PD"
    P1_DP = "P1_DP"
    P2_GATED_PD = "P2_GATED_PD"
    P3_GATED_DP = "P3_GATED_DP"
    P4_KVS_DECOUPLED = "P4_KVS_DECOUPLED"


class LifecycleAction(StrEnum):
    HOLD = "HOLD"
    SELECT_P = "SELECT_P"
    SELECT_D = "SELECT_D"
    RESERVE_D = "RESERVE_D"
    COMMIT_P = "COMMIT_P"
    ROLLBACK_P = "ROLLBACK_P"
    CONTINUE_D = "CONTINUE_D"
    RESCHEDULE_D = "RESCHEDULE_D"
    STORE_PUT = "STORE_PUT"
    REJECT = "REJECT"
    TERMINAL_FAIL = "TERMINAL_FAIL"


@dataclass(frozen=True, slots=True)
class LifecycleConfig:
    protocol: ProtocolId
    prefill_nodes: int
    decode_nodes: int
    prefill_token_cost: float = 0.06
    decode_token_cost: float = 1.0
    max_post_prefill_wait: float = 64.0
    probe_cost: float = 2.0
    probe_staleness_work: float = 32.0
    store_put_work: float = 4.0
    store_pull_work: float = 8.0
    tier_weights: Mapping[Tier, float] = field(
        default_factory=lambda: {
            Tier.STRICT: 2.0,
            Tier.STANDARD: 1.0,
            Tier.RELAXED: 0.5,
        }
    )
    lambda_gpu: float = 1.0
    lambda_stall: float = 1.0
    lambda_risk: float = 1.0

    def __post_init__(self) -> None:
        if self.prefill_nodes <= 0 or self.decode_nodes <= 0:
            raise ValueError("lifecycle pools must be positive")
        if self.prefill_token_cost <= 0 or self.decode_token_cost <= 0:
            raise ValueError("service costs must be positive")
        object.__setattr__(
            self, "tier_weights", MappingProxyType(dict(self.tier_weights))
        )


@dataclass(frozen=True, slots=True)
class LifecycleRecord:
    request_id: str
    tenant_id: str
    tier: Tier
    actions: tuple[LifecycleAction, ...]
    arrival: float
    prefill_start: float | None
    decode_start: float | None
    finish: float | None
    strict_slo_met: bool
    emitted_tokens: int
    prefill_waste_work: float
    slo_missed_prefill_work: float
    decode_hold_work: float
    store_occupancy_work: float


@dataclass(frozen=True, slots=True)
class LifecycleReport:
    protocol: ProtocolId
    completed_requests: int
    rejected_requests: int
    strict_completed_goodput: float
    lenient_completed_goodput: float
    partial_output_tokens: int
    prefill_waste_work: float
    slo_missed_prefill_work: float
    decode_hold_work: float
    store_occupancy_work: float
    hold_count: int
    reserve_count: int
    commit_count: int
    rollback_count: int
    reject_count: int
    jain_fairness: float | None
    scalar_weights: Mapping[str, float]
    records: tuple[LifecycleRecord, ...]


Overlay = RelativeSloOverlay | TieredSloOverlay


def simulate_lifecycle(
    requests: Sequence[Request], overlay: Overlay, config: LifecycleConfig
) -> LifecycleReport:
    p_available = [0.0] * config.prefill_nodes
    d_available = [0.0] * config.decode_nodes
    records: list[LifecycleRecord] = []
    demand: defaultdict[str, float] = defaultdict(float)
    service: defaultdict[str, float] = defaultdict(float)
    for request in requests:
        prefill = request.input_tokens * config.prefill_token_cost
        decode = request.output_tokens * config.decode_token_cost
        profile = overlay.profile(request, prefill, config.decode_token_cost)
        demand[profile.tenant_id] += request.output_tokens
        p_index = min(range(len(p_available)), key=lambda i: (p_available[i], i))
        d_index = min(range(len(d_available)), key=lambda i: (d_available[i], i))
        actions: list[LifecycleAction] = [LifecycleAction.SELECT_P]
        probe_overhead = (
            config.probe_cost
            if config.protocol in (ProtocolId.P2_GATED_PD, ProtocolId.P3_GATED_DP)
            else 0
        )
        probe_p = (
            max(request.arrival_ms, p_available[p_index]) + probe_overhead + prefill
        )
        probe_d_available = d_available[d_index]
        if config.protocol is ProtocolId.P3_GATED_DP:
            probe_d_available = max(
                request.arrival_ms,
                probe_d_available - config.probe_staleness_work,
            )
        probe_finish = (
            max(probe_p, probe_d_available)
            + decode
            + (
                config.store_pull_work
                if config.protocol is ProtocolId.P4_KVS_DECOUPLED
                else 0
            )
        )
        deadline = profile.completion_deadline
        completion_risk = (probe_finish - request.arrival_ms) / deadline
        gpu_fraction = prefill / deadline
        stall_fraction = max(0.0, probe_d_available - probe_p) / deadline
        risk = (
            completion_risk * config.lambda_risk
            + gpu_fraction * config.lambda_gpu
            + stall_fraction * config.lambda_stall
        )
        gated = config.protocol in (ProtocolId.P2_GATED_PD, ProtocolId.P3_GATED_DP)
        if gated and risk > config.tier_weights[profile.tier]:
            actions.append(LifecycleAction.REJECT)
            records.append(_terminal(request, profile, actions))
            continue

        p_start, p_finish, d_start, finish, hold = _timeline(
            request.arrival_ms,
            prefill,
            decode,
            p_available[p_index],
            d_available[d_index],
            config,
            actions,
        )
        deadline_finish = request.arrival_ms + profile.completion_deadline
        if config.protocol is ProtocolId.P3_GATED_DP and finish > deadline_finish:
            actions.extend((LifecycleAction.ROLLBACK_P, LifecycleAction.REJECT))
            p_available[p_index] = p_finish
            d_available[d_index] = p_finish
            records.append(
                _terminal(
                    request,
                    profile,
                    actions,
                    prefill_start=p_start,
                    waste=prefill,
                    hold=hold,
                )
            )
            continue
        if (
            config.protocol
            in (ProtocolId.P0_PD, ProtocolId.P2_GATED_PD, ProtocolId.P4_KVS_DECOUPLED)
            and d_start - p_finish > config.max_post_prefill_wait
        ):
            actions.extend((LifecycleAction.TERMINAL_FAIL,))
            p_available[p_index] = p_finish
            records.append(
                _terminal(
                    request,
                    profile,
                    actions,
                    prefill_start=p_start,
                    waste=prefill,
                    hold=hold,
                    store=config.store_put_work
                    if config.protocol is ProtocolId.P4_KVS_DECOUPLED
                    else 0,
                )
            )
            continue

        p_available[p_index] = p_finish
        emitted = request.output_tokens
        completed = True
        actual_finish = finish
        if deadline_finish < finish:
            emitted = min(
                request.output_tokens,
                max(
                    0,
                    math.floor((deadline_finish - d_start) / config.decode_token_cost),
                ),
            )
            completed = False
            actual_finish = max(d_start, deadline_finish)
            actions.append(LifecycleAction.TERMINAL_FAIL)
        d_available[d_index] = actual_finish
        if completed:
            actions.append(LifecycleAction.CONTINUE_D)
            service[profile.tenant_id] += emitted
        ttft = d_start - request.arrival_ms
        tpot = config.decode_token_cost
        elapsed = finish - request.arrival_ms
        strict = (
            completed
            and ttft <= profile.ttft_deadline
            and tpot <= profile.tpot_deadline
            and elapsed <= profile.completion_deadline
        )
        records.append(
            LifecycleRecord(
                request.request_id,
                profile.tenant_id,
                profile.tier,
                tuple(actions),
                request.arrival_ms,
                p_start,
                d_start,
                finish if completed else None,
                strict,
                emitted,
                prefill if not completed else 0,
                prefill if completed and not strict else 0,
                hold,
                config.store_put_work
                if config.protocol is ProtocolId.P4_KVS_DECOUPLED
                else 0,
            )
        )
    return _report(
        requests,
        records,
        demand,
        service,
        config,
        tiered=isinstance(overlay, TieredSloOverlay),
    )


def _timeline(
    arrival: float,
    prefill: float,
    decode: float,
    p_free: float,
    d_free: float,
    config: LifecycleConfig,
    actions: list[LifecycleAction],
) -> tuple[float, float, float, float, float]:
    protocol = config.protocol
    hold = 0.0
    if protocol in (ProtocolId.P1_DP, ProtocolId.P3_GATED_DP):
        actions.append(LifecycleAction.RESERVE_D)
        p_start = max(arrival, p_free, d_free)
        if protocol is ProtocolId.P3_GATED_DP:
            p_start += config.probe_cost
        p_finish = p_start + prefill
        d_start = p_finish
        hold = prefill
        actions.append(LifecycleAction.HOLD)
    else:
        p_start = max(arrival, p_free)
        p_finish = p_start + prefill
        if protocol is ProtocolId.P2_GATED_PD:
            actions.append(LifecycleAction.RESERVE_D)
            p_start += config.probe_cost
            p_finish += config.probe_cost
        if protocol is ProtocolId.P4_KVS_DECOUPLED:
            actions.append(LifecycleAction.STORE_PUT)
            p_finish += config.store_put_work
        d_start = max(p_finish, d_free)
        if d_start > p_finish:
            actions.append(LifecycleAction.HOLD)
            hold = d_start - p_finish
    actions.extend((LifecycleAction.COMMIT_P, LifecycleAction.SELECT_D))
    pull = config.store_pull_work if protocol is ProtocolId.P4_KVS_DECOUPLED else 0
    finish = d_start + pull + decode
    return p_start, p_finish, d_start, finish, hold


def _terminal(
    request: Request,
    profile: SloProfile,
    actions: list[LifecycleAction],
    *,
    prefill_start: float | None = None,
    waste: float = 0,
    hold: float = 0,
    store: float = 0,
) -> LifecycleRecord:
    return LifecycleRecord(
        request_id=request.request_id,
        tenant_id=profile.tenant_id,
        tier=profile.tier,
        actions=tuple(actions),
        arrival=request.arrival_ms,
        prefill_start=prefill_start,
        decode_start=None,
        finish=None,
        strict_slo_met=False,
        emitted_tokens=0,
        prefill_waste_work=waste,
        slo_missed_prefill_work=0,
        decode_hold_work=hold,
        store_occupancy_work=store,
    )


def _report(
    requests: Sequence[Request],
    records: list[LifecycleRecord],
    demand: Mapping[str, float],
    service: Mapping[str, float],
    config: LifecycleConfig,
    *,
    tiered: bool,
) -> LifecycleReport:
    completed = [record for record in records if record.finish is not None]
    horizon = max(
        1.0,
        max((request.arrival_ms for request in requests), default=0)
        - min((request.arrival_ms for request in requests), default=0),
    )
    strict_tokens = sum(r.emitted_tokens for r in completed if r.strict_slo_met)
    lenient_tokens = sum(r.emitted_tokens for r in completed)
    all_actions = [action for record in records for action in record.actions]
    fairness_values = tuple(
        service.get(tenant, 0) / tenant_demand
        for tenant, tenant_demand in demand.items()
        if tenant_demand > 0
    )
    return LifecycleReport(
        config.protocol,
        len(completed),
        len(records) - len(completed),
        strict_tokens / horizon,
        lenient_tokens / horizon,
        sum(record.emitted_tokens for record in records if record.finish is None),
        sum(record.prefill_waste_work for record in records),
        sum(record.slo_missed_prefill_work for record in records),
        sum(record.decode_hold_work for record in records),
        sum(record.store_occupancy_work for record in records),
        all_actions.count(LifecycleAction.HOLD),
        all_actions.count(LifecycleAction.RESERVE_D),
        all_actions.count(LifecycleAction.COMMIT_P),
        all_actions.count(LifecycleAction.ROLLBACK_P),
        all_actions.count(LifecycleAction.REJECT),
        _jain(fairness_values) if tiered else None,
        MappingProxyType(
            {
                **{tier.value: value for tier, value in config.tier_weights.items()},
                "lambda_gpu": config.lambda_gpu,
                "lambda_stall": config.lambda_stall,
                "lambda_risk": config.lambda_risk,
                "prefill_token_cost": config.prefill_token_cost,
                "decode_token_cost": config.decode_token_cost,
                "max_post_prefill_wait": config.max_post_prefill_wait,
                "probe_cost": config.probe_cost,
                "probe_staleness_work": config.probe_staleness_work,
                "store_put_work": config.store_put_work,
                "store_pull_work": config.store_pull_work,
            }
        ),
        tuple(records),
    )


def _jain(values: tuple[float, ...]) -> float:
    denominator = len(values) * sum(value * value for value in values)
    return sum(values) ** 2 / denominator if denominator else 0.0
