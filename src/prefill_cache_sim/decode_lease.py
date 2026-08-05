"""Shared-clock K-server decode lease simulation for Phase D."""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum

from .continuation import ContinuationPrefixBuilder
from .domain import BlockKind, Request
from .kvs import KvtCostModel, RemotePrefixStore, TieredCacheTopology


class DecodePolicy(StrEnum):
    D0 = "D0_NO_LEASE"
    D1 = "D1_FIXED_LEASE"
    D1_5 = "D1_5_ADAPTIVE_LEASE"


class RecoveryMode(StrEnum):
    R0 = "R0_RECOMPUTE"
    R1 = "R1_PREFIX_KVS"


@dataclass(frozen=True, slots=True)
class LeaseConfig:
    policy: DecodePolicy
    recovery: RecoveryMode
    prefill_nodes: int = 2
    decode_nodes: int = 2
    fixed_schedule: tuple[int, ...] = (8, 8, 14)
    max_rounds: int = 64
    enabled: bool = True
    min_quantum: int = 8
    max_quantum: int = 64
    base_quantum: int = 14
    decode_token_work: float = 1.0
    prefill_token_work: float = 0.0568
    max_recovery_work: float = 1e9
    local_capacity_blocks: int = 256
    remote_capacity_blocks: int = 4096
    migrate_on_boundary: bool = False
    completion_slo_multiplier: float = 2.5
    kvt: KvtCostModel = field(
        default_factory=lambda: KvtCostModel(2.0, 100.0, 0.5, 327_680)
    )

    def __post_init__(self) -> None:
        if self.prefill_nodes <= 0 or self.decode_nodes <= 0:
            raise ValueError("resource pools must be positive")
        if not self.fixed_schedule or any(value <= 0 for value in self.fixed_schedule):
            raise ValueError("fixed lease schedule must be positive")
        if not 0 < self.min_quantum <= self.base_quantum <= self.max_quantum:
            raise ValueError("adaptive lease bounds are invalid")


@dataclass(frozen=True, slots=True)
class LeaseRecord:
    request_id: str
    tenant_id: str
    attempts: int
    emitted_tokens: int
    completed: bool
    first_decode_start: float
    finish: float
    recovery_stall_work: float
    local_recovery_tokens: int
    remote_recovery_tokens: int
    recompute_recovery_tokens: int
    fairness_debt: float
    downgraded: bool
    quanta: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LeaseReport:
    policy: DecodePolicy
    recovery: RecoveryMode
    completed_requests: int
    completed_tokens: int
    partial_tokens: int
    strict_completed_goodput: float
    completed_goodput: float
    recovery_stall_work: float
    recovery_p95: float
    decode_waste_work: float
    ttft_p95: float
    short_ttft_p95: float
    jain_fairness: float
    boundaries_per_completed_request: float
    records: tuple[LeaseRecord, ...]


@dataclass(slots=True)
class _State:
    original: Request
    current: Request
    remaining: int
    emitted: int = 0
    attempts: int = 0
    first_start: float | None = None
    finish: float = 0
    stall: float = 0
    local_tokens: int = 0
    remote_tokens: int = 0
    recompute_tokens: int = 0
    downgraded: bool = False
    quanta: list[int] = field(default_factory=list)
    node: int = 0


def simulate_decode_leases(
    requests: Sequence[Request],
    builder: ContinuationPrefixBuilder,
    config: LeaseConfig,
) -> LeaseReport:
    node_ids = tuple(f"d{i}" for i in range(config.decode_nodes))
    remote = RemotePrefixStore(config.remote_capacity_blocks)
    topology = TieredCacheTopology.shared_kvs(
        node_ids, config.local_capacity_blocks, remote
    )
    p_available = [0.0] * config.prefill_nodes
    d_available = [0.0] * config.decode_nodes
    ready: list[tuple[float, int, _State]] = []
    sequence = 0
    for item in sorted(
        requests, key=lambda value: (value.arrival_ms, value.request_id)
    ):
        p_node = min(range(config.prefill_nodes), key=lambda i: (p_available[i], i))
        p_start = max(item.arrival_ms, p_available[p_node])
        p_finish = p_start + item.input_tokens * config.prefill_token_work
        p_available[p_node] = p_finish
        state = _State(item, item, item.output_tokens)
        heapq.heappush(ready, (p_finish, sequence, state))
        sequence += 1
    finished: list[_State] = []
    while ready:
        ready_at, _, state = heapq.heappop(ready)
        node = (
            state.node
            if state.attempts
            else min(range(config.decode_nodes), key=lambda i: (d_available[i], i))
        )
        start = max(ready_at, d_available[node])
        if state.first_start is None:
            state.first_start = start
            topology.seed_local(node_ids[node], state.original, start)
            if config.recovery is RecoveryMode.R1:
                remote.place(node_ids[node], state.original, start)
        quantum = _quantum(config, state)
        sent = min(state.remaining, quantum)
        end = start + sent * config.decode_token_work
        d_available[node] = end
        state.attempts += 1
        state.emitted += sent
        state.remaining -= sent
        state.quanta.append(sent)
        state.finish = end
        if state.remaining == 0:
            finished.append(state)
            continue
        if state.downgraded or config.policy is DecodePolicy.D0 or not config.enabled:
            drain = state.remaining
            state.quanta.append(drain)
            state.emitted += drain
            state.remaining = 0
            state.finish = end + drain * config.decode_token_work
            d_available[node] = state.finish
            finished.append(state)
            continue
        if state.attempts >= config.max_rounds:
            state.downgraded = True
            heapq.heappush(ready, (end, sequence, state))
            sequence += 1
            continue
        continuation = builder.build(
            state.current, sent_tokens=sent, arrival_ms=math.ceil(end)
        )
        local_materialized = _local_materialized_delta(state.current, continuation)
        if local_materialized is not None:
            topology.seed_local(node_ids[node], local_materialized, end)
        materialized = _new_full_blocks(
            state.current, continuation, builder.block_size_tokens
        )
        if materialized is not None and config.recovery is RecoveryMode.R1:
            remote.place(node_ids[node], materialized, end)
        next_node = (
            (node + 1) % config.decode_nodes if config.migrate_on_boundary else node
        )
        observation = topology.observe(
            node_ids[next_node],
            continuation,
            end,
            lookup_available=config.recovery is RecoveryMode.R1,
        )
        recompute_work = observation.recompute_tokens * config.prefill_token_work
        recovery = recompute_work
        if config.recovery is RecoveryMode.R1:
            recovery = config.kvt.effective_ms(
                observation.remote_hit_tokens, recompute_work
            )
        state.local_tokens += (
            observation.local_hit_tokens + observation.local_dram_hit_tokens
        )
        state.remote_tokens += observation.remote_hit_tokens
        state.recompute_tokens += observation.recompute_tokens
        if recovery > config.max_recovery_work:
            state.downgraded = True
            recovery = 0
        elif recovery > 0:
            p_node = min(range(config.prefill_nodes), key=lambda i: (p_available[i], i))
            recovery_start = max(end, p_available[p_node])
            p_available[p_node] = recovery_start + recovery
            state.stall += p_available[p_node] - end
            end = p_available[p_node]
        state.current = continuation
        state.node = next_node
        heapq.heappush(ready, (end, sequence, state))
        sequence += 1
    return _report(requests, finished, config)


def _quantum(config: LeaseConfig, state: _State) -> int:
    if config.policy is DecodePolicy.D0 or not config.enabled or state.downgraded:
        return state.remaining
    if config.policy is DecodePolicy.D1:
        return config.fixed_schedule[
            min(state.attempts, len(config.fixed_schedule) - 1)
        ]
    pressure = min(config.max_quantum - config.base_quantum, state.attempts * 4)
    return max(
        config.min_quantum, min(config.max_quantum, config.base_quantum + pressure)
    )


def _full_block_prefix(request: Request, block_size: int) -> Request | None:
    keep = len(request.prefix_blocks)
    while keep and (
        request.prefix_blocks[keep - 1].kind is BlockKind.GENERATED
        and request.block_token_sizes[keep - 1] < block_size
    ):
        keep -= 1
    if keep == 0:
        return None
    sizes = request.block_token_sizes[:keep]
    return replace(
        request,
        input_tokens=sum(sizes),
        prefix_blocks=request.prefix_blocks[:keep],
        block_token_sizes=sizes,
    )


def _new_full_blocks(
    previous: Request, current: Request, block_size: int
) -> Request | None:
    prior_sizes = dict(
        zip(previous.prefix_blocks, previous.block_token_sizes, strict=True)
    )
    indexes = [
        index
        for index, (block, size) in enumerate(
            zip(current.prefix_blocks, current.block_token_sizes, strict=True)
        )
        if (
            (block not in prior_sizes or prior_sizes[block] < block_size)
            and block.kind is BlockKind.GENERATED
            and size == block_size
        )
    ]
    if not indexes:
        return None
    blocks = tuple(current.prefix_blocks[index] for index in indexes)
    sizes = tuple(current.block_token_sizes[index] for index in indexes)
    return replace(
        current,
        input_tokens=sum(sizes),
        prefix_blocks=blocks,
        block_token_sizes=sizes,
    )


def _local_materialized_delta(previous: Request, current: Request) -> Request | None:
    prior = set(previous.prefix_blocks)
    indexes = [
        index
        for index, block in enumerate(current.prefix_blocks)
        if block not in prior
        or (
            index == len(current.prefix_blocks) - 1
            and block.kind is BlockKind.GENERATED
        )
    ]
    if not indexes:
        return None
    blocks = tuple(current.prefix_blocks[index] for index in indexes)
    sizes = tuple(current.block_token_sizes[index] for index in indexes)
    return replace(
        current,
        input_tokens=sum(sizes),
        prefix_blocks=blocks,
        block_token_sizes=sizes,
    )


def _report(
    requests: Sequence[Request], states: list[_State], config: LeaseConfig
) -> LeaseReport:
    by_tenant_demand: defaultdict[str, int] = defaultdict(int)
    by_tenant_service: defaultdict[str, int] = defaultdict(int)
    for request in requests:
        by_tenant_demand[request.affinity_key or "tenant-0"] += request.output_tokens
    for state in states:
        by_tenant_service[state.original.affinity_key or "tenant-0"] += state.emitted
    ratios = [
        by_tenant_service[tenant] / demand
        for tenant, demand in by_tenant_demand.items()
        if demand
    ]
    fairness = (
        1.0
        if len(ratios) <= 1
        else sum(ratios) ** 2 / (len(ratios) * sum(value * value for value in ratios))
    )
    records = tuple(
        LeaseRecord(
            state.original.request_id,
            state.original.affinity_key or "tenant-0",
            state.attempts,
            state.emitted,
            state.remaining == 0,
            state.first_start or 0,
            state.finish,
            state.stall,
            state.local_tokens,
            state.remote_tokens,
            state.recompute_tokens,
            1
            - by_tenant_service[state.original.affinity_key or "tenant-0"]
            / by_tenant_demand[state.original.affinity_key or "tenant-0"],
            state.downgraded,
            tuple(state.quanta),
        )
        for state in states
    )
    horizon = max(
        1.0,
        max((request.arrival_ms for request in requests), default=0)
        - min((request.arrival_ms for request in requests), default=0),
    )
    strict_tokens = sum(
        state.emitted
        for state in states
        if state.finish - state.original.arrival_ms
        <= (
            state.original.input_tokens * config.prefill_token_work
            + state.original.output_tokens * config.decode_token_work
        )
        * config.completion_slo_multiplier
    )
    ttfts = [(state.first_start or 0) - state.original.arrival_ms for state in states]
    short_ttfts = [
        (state.first_start or 0) - state.original.arrival_ms
        for state in states
        if state.original.output_tokens <= 64
    ]
    stalls = [state.stall for state in states]
    boundaries = sum(max(0, state.attempts - 1) for state in states)
    completed = sum(state.remaining == 0 for state in states)
    return LeaseReport(
        config.policy,
        config.recovery,
        completed,
        sum(state.emitted for state in states if state.remaining == 0),
        sum(state.emitted for state in states if state.remaining > 0),
        strict_tokens / horizon,
        sum(state.emitted for state in states if state.remaining == 0) / horizon,
        sum(stalls),
        _percentile(stalls, 0.95),
        sum(state.emitted for state in states if state.remaining > 0),
        _percentile(ttfts, 0.95),
        _percentile(short_ttfts, 0.95),
        fairness,
        boundaries / completed if completed else 0,
        records,
    )


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[math.ceil(quantile * len(ordered)) - 1]
