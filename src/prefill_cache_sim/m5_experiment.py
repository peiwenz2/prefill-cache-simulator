"""Bounded Phase-B KVS replay and stable result accounting."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .domain import Request
from .identity import DeterministicStream, NamedSeedManager
from .kvs import KvtCostModel, RemotePrefixStore, TieredCacheTopology, TopologyMode
from .mooncake_selector import MooncakeAlgo1Selector


@dataclass(frozen=True, slots=True)
class M5Case:
    case_id: str
    node_count: int = 4
    local_capacity_blocks: int = 12_500
    remote_capacity_blocks: int = 50_000
    bandwidth_gbps: float = 100
    fixed_latency_ms: float = 0.2
    overlap_ratio: float = 0.5
    bytes_per_token: int = 65_536
    prefill_token_ms: float = 0.0568
    kvcache_balancing_threshold: float = 2.0
    lookup_timeout_probability: float = 0
    seed: int = 713
    topology_mode: TopologyMode = TopologyMode.SHARED_KVS
    local_dram_capacity_blocks: int = 25_000

    def __post_init__(self) -> None:
        if self.node_count <= 0:
            raise ValueError("M5 node count must be positive")
        if not 0 <= self.lookup_timeout_probability <= 1:
            raise ValueError("lookup timeout probability must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class M5Result:
    case_id: str
    input_tokens: int
    local_hit_tokens: int
    local_dram_hit_tokens: int
    remote_hit_tokens: int
    recompute_tokens: int
    local_hit_rate: float
    remote_hit_rate: float
    effective_prefix_rate: float
    kvt_normalized_ms: float
    recompute_normalized_ms: float
    effective_plan_normalized_ms: float
    lookup_unknown_count: int
    remote_evictions: int
    remote_replica_factor: float
    hotspot_max_mean: float
    remote_bound_request_count: int
    kvt_fixed_normalized_ms: float
    kvt_wire_normalized_ms: float
    hotspot_migrations: int
    bandwidth_gbps: float
    fixed_latency_ms: float
    bytes_per_token: int
    remote_capacity_blocks: int
    kvcache_balancing_threshold: float
    topology_mode: str
    trace_sha256: str = ""
    config_sha256: str = ""
    git_sha: str | None = None
    git_dirty: bool | None = None
    metric_unit: str = "NORMALIZED_MS"
    replay_mode: str = "SEQUENTIAL_KVS"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_m5_case(
    case: M5Case,
    requests: tuple[Request, ...],
    *,
    trace_sha256: str = "",
    config_sha256: str = "",
    git_sha: str | None = None,
    git_dirty: bool | None = None,
) -> M5Result:
    node_ids = tuple(f"p{index}" for index in range(case.node_count))
    store = RemotePrefixStore(case.remote_capacity_blocks)
    if case.topology_mode is TopologyMode.LOCAL_ONLY:
        topology = TieredCacheTopology.local_only(node_ids, case.local_capacity_blocks)
    elif case.topology_mode is TopologyMode.HYBRID:
        topology = TieredCacheTopology.hybrid(
            node_ids,
            case.local_capacity_blocks,
            case.local_dram_capacity_blocks,
            store,
        )
    else:
        topology = TieredCacheTopology.shared_kvs(
            node_ids, case.local_capacity_blocks, store
        )
    kvt = KvtCostModel(
        case.fixed_latency_ms,
        case.bandwidth_gbps,
        case.overlap_ratio,
        case.bytes_per_token,
    )
    selector = MooncakeAlgo1Selector(
        case.kvcache_balancing_threshold,
        transfer_token_cost=kvt.non_overlapped_ms(512) / 512,
        recompute_token_cost=case.prefill_token_ms,
    )
    timeout_stream = NamedSeedManager(case.seed).stream("kvs/lookup-timeout")
    queue_work = {node_id: 0.0 for node_id in node_ids}
    input_tokens = 0
    local_tokens = 0
    dram_tokens = 0
    remote_tokens = 0
    recompute_tokens = 0
    kvt_ms = 0.0
    recompute_ms = 0.0
    effective_ms = 0.0
    unknown = 0
    remote_bound = 0
    fixed_ms = 0.0
    wire_ms = 0.0
    hotspot_migrations = 0
    previous_arrival = requests[0].arrival_ms if requests else 0.0
    for request in requests:
        elapsed = request.arrival_ms - previous_arrival
        queue_work = {
            node_id: max(0.0, work - elapsed) for node_id, work in queue_work.items()
        }
        previous_arrival = request.arrival_ms
        lookup_available = not _draw_probability(
            timeout_stream,
            case.lookup_timeout_probability,
        )
        observations = {
            node_id: topology.observe(
                node_id,
                request,
                request.arrival_ms,
                lookup_available=lookup_available,
            )
            for node_id in node_ids
        }
        selection = selector.select_from_observations(request, observations, queue_work)
        observation = observations[selection.node_id]
        local_prefix_tokens = (
            observation.local_hit_tokens + observation.local_dram_hit_tokens
        )
        remote_plan = bool(selection.components["remote_plan"])
        planned_prefix_tokens = int(selection.components["planned_prefix_tokens"])
        executed_remote_tokens = (
            max(0, planned_prefix_tokens - local_prefix_tokens) if remote_plan else 0
        )
        executed_recompute_tokens = (
            request.input_tokens - local_prefix_tokens - executed_remote_tokens
        )
        input_tokens += request.input_tokens
        local_tokens += observation.local_hit_tokens
        dram_tokens += observation.local_dram_hit_tokens
        remote_tokens += executed_remote_tokens
        recompute_tokens += executed_recompute_tokens
        transfer = kvt.non_overlapped_ms(executed_remote_tokens)
        raw_wire = (executed_remote_tokens * case.bytes_per_token * 8) / (
            case.bandwidth_gbps * 1_000_000
        )
        raw_fixed = case.fixed_latency_ms if executed_remote_tokens else 0.0
        compute = executed_recompute_tokens * case.prefill_token_ms
        kvt_ms += transfer
        recompute_ms += compute
        effective_ms += max(transfer, compute)
        remote_bound += int(transfer > compute)
        fixed_ms += raw_fixed * (1 - case.overlap_ratio)
        wire_ms += raw_wire * (1 - case.overlap_ratio)
        unknown += int(observation.lookup_unknown)
        queue_work[selection.node_id] += max(transfer, compute)
        executed_remote_blocks = (
            sum(
                segment.end_block - segment.start_block
                for segment in observation.source_segments
                if segment.source.value == "REMOTE_KVS"
            )
            if remote_plan
            else 0
        )
        if executed_remote_blocks:
            store.record_transfer(request, executed_remote_blocks)
            hotspot_migrations += 1
        topology.seed_local(selection.node_id, request, request.arrival_ms)
        if case.topology_mode is TopologyMode.HYBRID:
            topology.seed_local_dram(selection.node_id, request, request.arrival_ms)
        if case.topology_mode is not TopologyMode.LOCAL_ONLY:
            store.place(selection.node_id, request, request.arrival_ms)
    effective_hit = local_tokens + dram_tokens + remote_tokens
    return M5Result(
        case.case_id,
        input_tokens,
        local_tokens,
        dram_tokens,
        remote_tokens,
        recompute_tokens,
        local_tokens / input_tokens if input_tokens else 0.0,
        remote_tokens / input_tokens if input_tokens else 0.0,
        effective_hit / input_tokens if input_tokens else 0.0,
        kvt_ms,
        recompute_ms,
        effective_ms,
        unknown,
        store.evictions,
        store.replica_factor,
        store.hotspot_max_mean,
        remote_bound,
        fixed_ms,
        wire_ms,
        hotspot_migrations,
        case.bandwidth_gbps,
        case.fixed_latency_ms,
        case.bytes_per_token,
        case.remote_capacity_blocks,
        case.kvcache_balancing_threshold,
        case.topology_mode.value,
        trace_sha256,
        config_sha256,
        git_sha,
        git_dirty,
    )


def _draw_probability(stream: DeterministicStream, probability: float) -> bool:
    if probability <= 0:
        return False
    if probability >= 1:
        return True
    return stream.randbelow(1_000_000) < int(probability * 1_000_000)
