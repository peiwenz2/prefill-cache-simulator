"""Stable Phase-A experiment case and flat result schema."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace

from .affinity import OnlineConversationLinker
from .domain import CacheVisibility, ReplayMode, Request, ViewMode
from .eviction import (
    FifoPolicy,
    LruPolicy,
    SecondHitAdmissionPolicy,
    SlruPolicy,
)
from .identity import NamedSeedManager
from .interfaces import BlockEvictionPolicy, PrefillNodeSelector
from .selectors import (
    CalibratedTtftSelector,
    FlexLbTtftSelector,
    LeastWorkSelector,
    RandomSelector,
    RoundRobinSelector,
    gb_prefix_bucket_selector,
    session_affinity_selector,
)
from .simulator import LocalSimulator, SimulationConfig, SimulationReport

RESULT_SCHEMA_VERSION = "phase-a-result-v3"


@dataclass(frozen=True, slots=True)
class ExperimentCase:
    case_id: str
    selector_id: str
    eviction_id: str
    node_count: int
    capacity_mode: str
    capacity_blocks: int
    replay_mode: ReplayMode
    visibility: CacheVisibility
    view_mode: ViewMode
    view_delay_ms: float
    view_loss_probability: float
    replay_speed: float
    seed: int
    prefill_uncached_token_ms: float = 1.0
    prefill_concurrency_per_node: int = 1
    prefix_anchor_k: int = 2
    soft_alpha: float = 1.25
    minimum_shared_blocks: int = 2
    family_size_cap: int = 8
    hot_block_percentile: float = 99.0
    cache_discount: float = 0.7
    similar_band_ratio: float = 0.1

    def __post_init__(self) -> None:
        if self.replay_speed <= 0:
            raise ValueError("replay_speed must be positive")
        if self.node_count <= 0 or self.capacity_blocks <= 0:
            raise ValueError("node count and capacity must be positive")
        if self.family_size_cap <= 0:
            raise ValueError("family_size_cap must be positive")

    def capacities(self) -> tuple[int, ...]:
        if self.capacity_mode == "FIXED_PER_NODE":
            return (self.capacity_blocks,) * self.node_count
        if self.capacity_mode != "FIXED_TOTAL":
            raise ValueError(f"unknown capacity mode {self.capacity_mode}")
        base, remainder = divmod(self.capacity_blocks, self.node_count)
        if base == 0:
            raise ValueError("fixed-total capacity is smaller than node_count")
        return tuple(base + (index < remainder) for index in range(self.node_count))


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    result_schema_version: str
    case_id: str
    selector_id: str
    eviction_id: str
    node_count: int
    capacity_mode: str
    capacity_blocks: int
    replay_mode: str
    visibility: str
    view_mode: str
    view_delay_ms: float
    view_loss_probability: float
    replay_speed: float
    seed: int
    prefill_uncached_token_ms: float
    prefill_concurrency_per_node: int
    prefix_anchor_k: int
    soft_alpha: float
    minimum_shared_blocks: int
    family_size_cap: int
    hot_block_percentile: float
    cache_discount: float
    similar_band_ratio: float
    completed_requests: int
    block_ref_hit_rate: float
    token_weighted_hit_rate: float
    logical_block_ref_hit_rate: float
    logical_token_weighted_hit_rate: float
    request_load_max_mean: float
    input_token_load_max_mean: float
    uncached_token_load_max_mean: float
    request_load_cv: float
    request_load_gini: float
    queue_wait_normalized_ms_p50: float
    queue_wait_normalized_ms_p95: float
    queue_wait_normalized_ms_p99: float
    queue_wait_normalized_ms_max: float
    offered_load_ratio: float
    eviction_count: int
    rejected_placements: int
    unique_resident_blocks: int
    replica_factor: float
    fallback_count: int
    no_stat_fallback_count: int
    inflight_wait_tokens: int
    inflight_wait_ms: float
    stranded_block_references: int
    eviction_regret: int
    one_hit_pollution: int
    affinity_kept: int
    affinity_broken: int
    trace_sha256: str
    config_sha256: str
    git_sha: str | None
    git_dirty: bool | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _ratio_max_mean(values: tuple[int, ...]) -> float:
    mean = statistics.fmean(values)
    return max(values) / mean if mean else 0.0


def _cv(values: tuple[int, ...]) -> float:
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else 0.0


def _gini(values: tuple[int, ...]) -> float:
    total = sum(values)
    if total == 0:
        return 0.0
    ordered = sorted(values)
    weighted = sum((index + 1) * value for index, value in enumerate(ordered))
    return (2 * weighted) / (len(values) * total) - (len(values) + 1) / len(values)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def selector_for(case: ExperimentCase) -> PrefillNodeSelector:
    if case.selector_id == "S0_RANDOM":
        stream = NamedSeedManager(case.seed).stream("selector/random")
        return RandomSelector(stream)
    if case.selector_id == "S1_ROUND_ROBIN":
        return RoundRobinSelector()
    if case.selector_id == "S2_LEAST_WORK":
        return LeastWorkSelector()
    if case.selector_id == "S3_GB_PREFIX_BUCKET":
        return gb_prefix_bucket_selector(
            case.prefix_anchor_k, soft_alpha=case.soft_alpha
        )
    if case.selector_id == "S4_SESSION_AFFINITY":
        provider = OnlineConversationLinker(
            case.minimum_shared_blocks,
            case.family_size_cap,
            case.hot_block_percentile,
        )
        return session_affinity_selector(provider, soft_alpha=case.soft_alpha)
    if case.selector_id == "S5_FLEXLB_TTFT":
        return FlexLbTtftSelector(case.cache_discount, 0.3, case.similar_band_ratio)
    if case.selector_id == "S6_CALIBRATED_TTFT":
        return CalibratedTtftSelector(
            case.prefill_uncached_token_ms, case.similar_band_ratio
        )
    raise ValueError(f"unknown selector {case.selector_id}")


def eviction_factory_for(
    case: ExperimentCase,
) -> Callable[[], BlockEvictionPolicy]:
    per_node_capacity = min(case.capacities())
    if case.eviction_id == "E0_FIFO":
        return FifoPolicy
    if case.eviction_id == "E1_LRU":
        return LruPolicy
    if case.eviction_id == "E2_SLRU":
        protected = max(1, math.floor(per_node_capacity * 0.8))
        return lambda: SlruPolicy(protected)
    if case.eviction_id == "E3_SECOND_HIT_ADMISSION":
        ghost = max(1, per_node_capacity)
        return lambda: SecondHitAdmissionPolicy(ghost)
    raise ValueError(f"unknown eviction {case.eviction_id}")


def run_case(
    case: ExperimentCase,
    requests: tuple[Request, ...],
    *,
    trace_sha256: str,
    config_sha256: str,
    git_sha: str | None,
    git_dirty: bool | None,
) -> ExperimentResult:
    config = SimulationConfig(
        replay_mode=case.replay_mode,
        visibility=case.visibility,
        view_mode=case.view_mode,
        node_count=case.node_count,
        capacity_blocks_per_node=case.capacities(),
        prefill_uncached_token_ms=case.prefill_uncached_token_ms,
        view_delay_ms=case.view_delay_ms,
        prefill_concurrency_per_node=case.prefill_concurrency_per_node,
        view_loss_probability=case.view_loss_probability,
        root_seed=case.seed,
    )
    replay_requests = tuple(
        replace(request, arrival_ms=request.arrival_ms / case.replay_speed)
        for request in requests
    )
    report = LocalSimulator(selector_for(case), eviction_factory_for(case), config).run(
        replay_requests
    )
    return result_from_report(
        case,
        report,
        trace_sha256=trace_sha256,
        config_sha256=config_sha256,
        git_sha=git_sha,
        git_dirty=git_dirty,
    )


def result_from_report(
    case: ExperimentCase,
    report: SimulationReport,
    *,
    trace_sha256: str,
    config_sha256: str,
    git_sha: str | None,
    git_dirty: bool | None,
) -> ExperimentResult:
    waits = [decision.start_ms - decision.arrival_ms for decision in report.decisions]
    fallbacks = [decision.fallback_reason for decision in report.decisions]
    return ExperimentResult(
        RESULT_SCHEMA_VERSION,
        case.case_id,
        case.selector_id,
        case.eviction_id,
        case.node_count,
        case.capacity_mode,
        case.capacity_blocks,
        case.replay_mode.value,
        case.visibility.value,
        case.view_mode.value,
        case.view_delay_ms,
        case.view_loss_probability,
        case.replay_speed,
        case.seed,
        case.prefill_uncached_token_ms,
        case.prefill_concurrency_per_node,
        case.prefix_anchor_k,
        case.soft_alpha,
        case.minimum_shared_blocks,
        case.family_size_cap,
        case.hot_block_percentile,
        case.cache_discount,
        case.similar_band_ratio,
        report.completed_requests,
        report.block_ref_hit_rate,
        report.token_weighted_hit_rate,
        report.logical_block_ref_hit_rate,
        report.logical_token_weighted_hit_rate,
        _ratio_max_mean(report.per_node_requests),
        _ratio_max_mean(report.per_node_input_tokens),
        _ratio_max_mean(report.per_node_uncached_tokens),
        _cv(report.per_node_requests),
        _gini(report.per_node_requests),
        _percentile(waits, 0.5),
        _percentile(waits, 0.95),
        _percentile(waits, 0.99),
        report.queue_wait_ms_max,
        (
            sum(report.per_node_uncached_tokens)
            * case.prefill_uncached_token_ms
            / (
                (
                    max(decision.arrival_ms for decision in report.decisions)
                    - min(decision.arrival_ms for decision in report.decisions)
                )
                * case.node_count
                * case.prefill_concurrency_per_node
            )
            if len(report.decisions) > 1
            else 0.0
        ),
        report.eviction_count,
        report.rejected_placements,
        report.unique_resident_blocks,
        report.replica_factor,
        sum(reason is not None for reason in fallbacks),
        sum(reason == "no-stat" for reason in fallbacks),
        report.inflight_wait_tokens,
        report.inflight_wait_ms,
        sum(decision.stranded_resident_blocks for decision in report.decisions),
        report.eviction_regret,
        report.one_hit_pollution,
        sum(
            int(decision.selection_components.get("affinity", 0))
            and not int(decision.selection_components.get("affinity_broken", 0))
            for decision in report.decisions
        ),
        sum(
            int(decision.selection_components.get("affinity_broken", 0))
            for decision in report.decisions
        ),
        trace_sha256,
        config_sha256,
        git_sha,
        git_dirty,
    )
