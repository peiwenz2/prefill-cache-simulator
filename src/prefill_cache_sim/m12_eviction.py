"""M12.4 bounded stale census and replication-aware eviction."""

from __future__ import annotations

import math
from collections import OrderedDict, defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from .m12_kernel import (
    AttemptExecutionSpec,
    CacheMutation,
    CausalView,
    KernelPolicy,
    KernelRequestSpec,
    KernelResult,
)


class EvictionMode(StrEnum):
    LRU = "LRU"
    CENSUS_REGRET = "CENSUS_REGRET"


@dataclass(frozen=True, slots=True)
class CensusConfig:
    max_entries: int
    max_staleness_work: float

    def __post_init__(self) -> None:
        if type(self.max_entries) is not int or self.max_entries <= 0:
            raise ValueError("census max entries must be a positive integer")
        if self.max_staleness_work < 0 or not math.isfinite(
            self.max_staleness_work
        ):
            raise ValueError("census staleness must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class CensusEntry:
    cache_key: str
    cohort_id: str
    holders: frozenset[str]
    observed_at_work: float
    recovery_work_by_holder: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "recovery_work_by_holder",
            MappingProxyType(dict(self.recovery_work_by_holder)),
        )


class ClusterCacheCensus:
    """Bounded observation cache; callers explicitly refresh stale snapshots."""

    def __init__(self, config: CensusConfig) -> None:
        self.config = config
        self._entries: OrderedDict[tuple[str, str], CensusEntry] = OrderedDict()
        self.replica_updates = 0
        self.first_holder_observations = 0
        self.refresh_observations = 0
        self._latest_observed_work = -math.inf

    @property
    def duplicate_replica_updates(self) -> int:
        """Compatibility metric: refreshes are observations, never replicas."""
        return self.refresh_observations

    def observe(
        self,
        cache_key: str,
        cohort_id: str,
        holder_id: str,
        *,
        at_work: float,
        recovery_work: float,
    ) -> bool:
        if not cache_key or not cohort_id or not holder_id:
            raise ValueError("census identities must be explicit")
        if not math.isfinite(at_work):
            raise ValueError("census observation time must be finite")
        if recovery_work < 0 or not math.isfinite(recovery_work):
            raise ValueError("recovery work must be finite and non-negative")
        if at_work < self._latest_observed_work:
            raise ValueError("census observations must be monotonic")
        self._latest_observed_work = at_work
        key = (cohort_id, cache_key)
        current = self._entries.get(key)
        duplicate = current is not None and holder_id in current.holders
        holders = set(current.holders if current else ())
        recoveries = dict(current.recovery_work_by_holder if current else {})
        holders.add(holder_id)
        recoveries[holder_id] = recovery_work
        self._entries[key] = CensusEntry(
            cache_key, cohort_id, frozenset(holders), at_work, recoveries
        )
        self._entries.move_to_end(key)
        while len(self._entries) > self.config.max_entries:
            self._entries.popitem(last=False)
        if current is None:
            self.first_holder_observations += 1
            return True
        if duplicate:
            self.refresh_observations += 1
            return False
        self.replica_updates += 1
        return True

    def remove(self, cache_key: str, cohort_id: str, holder_id: str) -> None:
        key = (cohort_id, cache_key)
        current = self._entries.get(key)
        if current is None or holder_id not in current.holders:
            return
        holders = set(current.holders)
        recoveries = dict(current.recovery_work_by_holder)
        holders.remove(holder_id)
        recoveries.pop(holder_id, None)
        if not holders:
            self._entries.pop(key)
            return
        self._entries[key] = CensusEntry(
            cache_key,
            cohort_id,
            frozenset(holders),
            current.observed_at_work,
            recoveries,
        )

    def lookup(
        self, cache_key: str, cohort_id: str, *, now_work: float
    ) -> CensusEntry | None:
        if not math.isfinite(now_work):
            raise ValueError("census lookup time must be finite")
        entry = self._entries.get((cohort_id, cache_key))
        if entry is None:
            return None
        if entry.observed_at_work > now_work:
            return None
        if now_work - entry.observed_at_work > self.config.max_staleness_work:
            return None
        return entry


class PastOnlyReuseEstimator:
    """Bounded causal reference-frequency estimate scoped by cohort."""

    def __init__(
        self, history_limit: int = 128, decay_window_work: float = 100.0
    ) -> None:
        if type(history_limit) is not int or history_limit <= 0:
            raise ValueError("history limit must be a positive integer")
        self.history_limit = history_limit
        if decay_window_work <= 0 or not math.isfinite(decay_window_work):
            raise ValueError("reuse decay window must be finite and positive")
        self.decay_window_work = decay_window_work
        self._history: dict[tuple[str, str], deque[float]] = defaultdict(
            lambda: deque(maxlen=history_limit)
        )
        self._latest_work = -math.inf

    def observe(self, cache_key: str, cohort_id: str, *, at_work: float) -> None:
        self._check_time(at_work)
        self._history[(cohort_id, cache_key)].append(at_work)

    def estimate(self, cache_key: str, cohort_id: str, *, at_work: float) -> float:
        self._check_time(at_work)
        count = sum(
            observed >= at_work - self.decay_window_work
            for observed in self._history[(cohort_id, cache_key)]
        )
        if count < 2:
            return 0.0
        return min(1.0, (count - 1) / max(1, self.history_limit - 1))

    def _check_time(self, at_work: float) -> None:
        if not math.isfinite(at_work) or at_work < self._latest_work:
            raise ValueError("future observations cannot be read in the past")
        self._latest_work = at_work


@dataclass(frozen=True, slots=True)
class EvictionCandidate:
    cache_key: str
    reuse_estimate: float
    recompute_work: float
    cheapest_recovery_work: float | None
    unique_copy: bool

    def __post_init__(self) -> None:
        if not self.cache_key or not 0 <= self.reuse_estimate <= 1:
            raise ValueError("eviction candidate identity/reuse is invalid")
        if self.recompute_work < 0:
            raise ValueError("recompute work must be non-negative")
        if self.cheapest_recovery_work is not None and self.cheapest_recovery_work < 0:
            raise ValueError("recovery work must be non-negative")

    @property
    def regret_work(self) -> float:
        recovery = self.cheapest_recovery_work or 0.0
        return self.reuse_estimate * self.recompute_work - recovery

    @property
    def unique_hot(self) -> bool:
        return self.unique_copy and self.reuse_estimate > 0


def choose_eviction_victims(
    candidates: Sequence[EvictionCandidate], *, required: int
) -> tuple[str, ...]:
    if required < 0:
        raise ValueError("required victims must be non-negative")
    eligible = sorted(
        (item for item in candidates if not item.unique_hot),
        key=lambda item: (item.regret_work, item.cache_key),
    )
    if len(eligible) < required:
        raise ValueError("unique-hot entries are protected; insufficient victims")
    return tuple(item.cache_key for item in eligible[:required])


@dataclass(frozen=True, slots=True)
class M12EvictionConfig:
    mode: EvictionMode
    cache_capacity_entries: int
    prefill_work_per_token: float
    tier_recovery_work: float
    sender_congestion: Mapping[str, float]
    census_refresh_work: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.mode, EvictionMode):
            raise ValueError("eviction mode must be explicit")
        if (
            type(self.cache_capacity_entries) is not int
            or self.cache_capacity_entries <= 0
        ):
            raise ValueError("cache capacity must be a positive integer")
        values = (
            self.prefill_work_per_token,
            self.tier_recovery_work,
            self.census_refresh_work,
            *self.sender_congestion.values(),
        )
        if any(value < 0 or not math.isfinite(value) for value in values):
            raise ValueError("eviction prices must be finite and non-negative")
        if self.census_refresh_work == 0:
            raise ValueError("census refresh interval must be positive")


class M12EvictionPolicy(KernelPolicy):
    """LRU baseline or census-regret cache-mutation wrapper."""

    def __init__(
        self,
        placement: KernelPolicy,
        config: M12EvictionConfig,
        census: ClusterCacheCensus,
        *,
        cache_key_cohorts: Mapping[str, str],
        reuse: PastOnlyReuseEstimator | None = None,
    ) -> None:
        self.placement = placement
        self.config = config
        self.census = census
        self.cache_key_cohorts = dict(cache_key_cohorts)
        self.reuse = reuse or PastOnlyReuseEstimator()
        self._lru: dict[str, OrderedDict[str, None]] = defaultdict(OrderedDict)
        self._sizes: dict[str, int] = {}
        self.hit_tokens = 0
        self.input_tokens = 0
        self._last_census_refresh = -math.inf

    def plan_attempts(
        self, request: KernelRequestSpec, view: CausalView
    ) -> Sequence[AttemptExecutionSpec]:
        now = view.now_work
        if now - self._last_census_refresh >= self.config.census_refresh_work:
            for node, resident in view.cache_by_node.items():
                order = self._lru[node]
                for key in sorted(resident):
                    order.setdefault(key, None)
                    cohort = self._cohort(key)
                    self.census.observe(
                        key,
                        cohort,
                        node,
                        at_work=now,
                        recovery_work=self.config.tier_recovery_work,
                    )
            self._last_census_refresh = now
        attempts = tuple(self.placement.plan_attempts(request, view))
        if attempts:
            node = attempts[0].p_node_id
            resident = view.cache_by_node[node]
            hit = 0
            for key, size in zip(
                request.prefix_cache_keys,
                request.prefix_token_sizes,
                strict=True,
            ):
                self._sizes[key] = size
                self._cohort(key)
            for key, size in zip(
                request.prefix_cache_keys, request.prefix_token_sizes, strict=True
            ):
                cohort = self._cohort(key)
                if key in resident:
                    hit += size
                    if key in self._lru[node]:
                        self._lru[node].move_to_end(key)
                else:
                    self.reuse.observe(key, cohort, at_work=now)
                    break
                self.reuse.observe(key, cohort, at_work=now)
            self.hit_tokens += hit
            self.input_tokens += request.logical.input_tokens
        return attempts

    def cache_mutation(
        self,
        request: KernelRequestSpec,
        attempt: AttemptExecutionSpec,
        view: CausalView,
    ) -> CacheMutation:
        node = attempt.p_node_id
        resident = set(view.cache_by_node[node])
        additions = set(request.prefix_cache_keys) - resident
        required = max(
            0, len(resident | additions) - self.config.cache_capacity_entries
        )
        if self.config.mode is EvictionMode.LRU:
            victims = tuple(
                key for key in self._lru[node] if key in resident
            )[:required]
        else:
            candidates = tuple(
                self._candidate(key, node, view.now_work)
                for key in sorted(resident)
            )
            try:
                victims = choose_eviction_victims(candidates, required=required)
            except ValueError:
                return CacheMutation(admit=False)
        if len(victims) != required or not set(victims) <= resident:
            raise ValueError("eviction policy selected invalid victims")
        for key in victims:
            self._lru[node].pop(key, None)
            self.census.remove(key, self._cohort(key), node)
        for key in sorted(additions):
            self._lru[node][key] = None
            self.census.observe(
                key,
                self._cohort(key),
                node,
                at_work=view.now_work,
                recovery_work=self.config.tier_recovery_work,
            )
        return CacheMutation(admit=True, evict_keys=victims)

    def summarize(self, result: KernelResult) -> EvictionRunReport:
        metrics = result.metrics
        accounted = (
            metrics.winner_attributable_gpu_work
            + metrics.slo_missed_gpu_work
            + metrics.wasted_gpu_work
            + metrics.unclassified_attempt_gpu_work
        )
        return EvictionRunReport(
            mode=self.config.mode,
            offered_requests=metrics.offered_logical_requests,
            offered_tokens=metrics.offered_input_tokens + metrics.offered_output_tokens,
            token_hit_rate=(
                self.hit_tokens / self.input_tokens if self.input_tokens else 0.0
            ),
            strict_goodput=metrics.strict_useful_token_goodput,
            strict_output_goodput=metrics.strict_useful_output_token_goodput,
            jain_fairness=metrics.jain_fairness,
            minimum_tier_slo_attainment=metrics.minimum_tier_slo_attainment,
            per_tier_slo_attainment=metrics.per_tier_slo_attainment,
            total_gpu_work=metrics.total_gpu_work,
            accounted_gpu_work=accounted,
            wasted_gpu_work=metrics.wasted_gpu_work,
            kvs_work=metrics.kvs_normalized_work,
            replica_updates=self.census.replica_updates,
            duplicate_replica_updates=self.census.refresh_observations,
            winner_gpu_work=metrics.winner_attributable_gpu_work,
            slo_missed_gpu_work=metrics.slo_missed_gpu_work,
            unclassified_gpu_work=metrics.unclassified_attempt_gpu_work,
        )

    def _candidate(
        self, key: str, node: str, now_work: float
    ) -> EvictionCandidate:
        cohort = self._cohort(key)
        entry = self.census.lookup(key, cohort, now_work=now_work)
        recoveries: list[float] = [self.config.tier_recovery_work]
        holders: set[str] = set()
        if entry is not None:
            holders = set(entry.holders) - {node}
            recoveries.extend(
                entry.recovery_work_by_holder[holder]
                + self.config.sender_congestion.get(holder, 0.0)
                for holder in holders
            )
        return EvictionCandidate(
            key,
            self.reuse.estimate(key, cohort, at_work=now_work),
            self._sizes.get(key, 1) * self.config.prefill_work_per_token,
            min(recoveries),
            not holders,
        )

    def _cohort(self, key: str) -> str:
        try:
            return self.cache_key_cohorts[key]
        except KeyError as error:
            raise ValueError(f"missing cohort identity for cache key: {key}") from error


@dataclass(frozen=True, slots=True)
class EvictionRunReport:
    mode: EvictionMode
    offered_requests: int
    offered_tokens: int
    token_hit_rate: float
    strict_goodput: float
    strict_output_goodput: float
    jain_fairness: float
    minimum_tier_slo_attainment: float
    per_tier_slo_attainment: Mapping[str, float]
    total_gpu_work: float
    accounted_gpu_work: float
    wasted_gpu_work: float
    kvs_work: float
    replica_updates: int
    duplicate_replica_updates: int
    winner_gpu_work: float
    slo_missed_gpu_work: float
    unclassified_gpu_work: float


@dataclass(frozen=True, slots=True)
class G12FourVerdict:
    capacity_binding: bool
    hit_gain_pp: float
    strict_goodput_gain: float
    stop_enforcement: bool
    constraints_preserved: bool
    passed: bool


def evaluate_g12_4(
    baseline: EvictionRunReport,
    candidate: EvictionRunReport,
    *,
    hit_ceiling: float,
) -> G12FourVerdict:
    if baseline.mode is not EvictionMode.LRU:
        raise ValueError("G12-4 baseline must be LRU")
    if candidate.mode is not EvictionMode.CENSUS_REGRET:
        raise ValueError("G12-4 candidate must be census regret")
    if not all(
        0 <= value <= 1
        for value in (
            baseline.token_hit_rate,
            candidate.token_hit_rate,
            hit_ceiling,
        )
    ):
        raise ValueError("hit rates and ceiling must be in [0, 1]")
    binding = hit_ceiling - baseline.token_hit_rate >= 0.03
    if not binding:
        raise ValueError("G12-4 requires a capacity-binding case")
    hit_gain = (candidate.token_hit_rate - baseline.token_hit_rate) * 100
    goodput_gain = _relative_gain(
        baseline.strict_goodput, candidate.strict_goodput
    )
    stop = hit_gain < 0.5 and goodput_gain < 0.02
    constraints = _constraints_preserved(baseline, candidate)
    return G12FourVerdict(
        binding,
        hit_gain,
        goodput_gain,
        stop,
        constraints,
        not stop and constraints,
    )


def _constraints_preserved(
    baseline: EvictionRunReport, candidate: EvictionRunReport
) -> bool:
    same_load = (
        baseline.offered_requests == candidate.offered_requests
        and baseline.offered_tokens == candidate.offered_tokens
    )
    fairness = (
        candidate.jain_fairness >= 0.90
        and candidate.minimum_tier_slo_attainment >= 0.80
        and set(baseline.per_tier_slo_attainment)
        == set(candidate.per_tier_slo_attainment)
        and all(
            candidate.per_tier_slo_attainment[tier] >= 0.80
            for tier in baseline.per_tier_slo_attainment
        )
        and all(
            baseline.per_tier_slo_attainment[tier] == 0
            or (
                baseline.per_tier_slo_attainment[tier]
                - candidate.per_tier_slo_attainment[tier]
            )
            / baseline.per_tier_slo_attainment[tier]
            <= 0.02
            for tier in baseline.per_tier_slo_attainment
        )
    )
    work = all(
        math.isclose(
            report.accounted_gpu_work,
            report.winner_gpu_work
            + report.slo_missed_gpu_work
            + report.wasted_gpu_work
            + report.unclassified_gpu_work,
        )
        and math.isclose(report.total_gpu_work, report.accounted_gpu_work)
        and 0 <= report.wasted_gpu_work <= report.total_gpu_work
        and report.kvs_work >= 0
        and report.replica_updates >= 0
        and report.duplicate_replica_updates >= 0
        for report in (baseline, candidate)
    )
    output = candidate.strict_output_goodput >= baseline.strict_output_goodput
    return same_load and fairness and work and output


def _relative_gain(baseline: float, candidate: float) -> float:
    if baseline:
        return (candidate - baseline) / baseline
    return math.inf if candidate > 0 else 0.0
