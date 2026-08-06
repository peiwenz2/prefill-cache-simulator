"""M12.2 placement-policy adapters and fixed-grid comparison runner."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import overload

from .m12_kernel import (
    AttemptExecutionSpec,
    CacheMutation,
    CausalKernel,
    CausalView,
    FrozenKernelCostModel,
    KernelConfig,
    KernelPolicy,
    KernelRequestSpec,
    KernelResult,
)
from .m12_metrics import (
    SERVICE_REGIMES,
    LogicalRequestSpec,
    M12MetricReport,
    SyntheticServiceRegime,
)
from .m12_priced_spill import (
    G12TwoVerdict,
    PlacementCandidate,
    PlacementStrategyMetrics,
    PricedSpillSelector,
    _stable_index,
    evaluate_g12_2,
)


class PlacementMode(StrEnum):
    HYBRID = "HYBRID"
    PRICED_SPILL = "PRICED_SPILL"
    S3 = "S3_GB_PREFIX_BUCKET"
    S4 = "S4_SESSION_AFFINITY"
    S5 = "S5_FLEXLB_TTFT"
    S6 = "S6_CALIBRATED_TTFT"


class KvsPriceMode(StrEnum):
    DISABLED = "DISABLED"
    NORMAL = "NORMAL"
    EXPENSIVE = "EXPENSIVE"


@dataclass(frozen=True, slots=True)
class TraceRequestInput:
    logical_request_id: str
    tenant_id: str
    tier: str
    arrival_work: float
    prefix_cache_keys: tuple[str, ...]
    prefix_token_sizes: tuple[int, ...]
    true_output_tokens: int
    model_id: str
    adapter_id: str
    work_shape: str
    slo_slack_work: float
    eligible_node_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.logical_request_id or not self.tenant_id or not self.tier:
            raise ValueError("identity, tenant, and tier must be explicit")
        if not self.model_id or not self.adapter_id or not self.work_shape:
            raise ValueError("model, adapter, and work shape must be explicit")
        if self.slo_slack_work < 0 or not math.isfinite(self.slo_slack_work):
            raise ValueError("SLO slack must be finite and non-negative")
        if not self.eligible_node_ids:
            raise ValueError("cohort eligibility must name at least one node")


@dataclass(frozen=True, slots=True)
class CohortTruth:
    model_id: str
    adapter_id: str
    work_shape: str
    slo_slack_work: float
    eligible_node_ids: frozenset[str]
    raw_prefix_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PlacementWorkload(Sequence[KernelRequestSpec]):
    requests: tuple[KernelRequestSpec, ...]
    request_truth: Mapping[str, CohortTruth]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "request_truth", MappingProxyType(dict(self.request_truth))
        )

    def __len__(self) -> int:
        return len(self.requests)

    @overload
    def __getitem__(self, index: int) -> KernelRequestSpec: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[KernelRequestSpec]: ...

    def __getitem__(
        self, index: int | slice
    ) -> KernelRequestSpec | Sequence[KernelRequestSpec]:
        return self.requests[index]

    def __iter__(self) -> Iterator[KernelRequestSpec]:
        return iter(self.requests)


def build_kernel_requests(
    rows: Sequence[TraceRequestInput],
) -> PlacementWorkload:
    """Freeze trace rows without synthesizing identity or fairness metadata."""
    seen: set[str] = set()
    result: list[KernelRequestSpec] = []
    truth: dict[str, CohortTruth] = {}
    for row in rows:
        if row.logical_request_id in seen:
            raise ValueError(f"duplicate trace identity: {row.logical_request_id}")
        seen.add(row.logical_request_id)
        logical = LogicalRequestSpec(
            row.logical_request_id,
            row.tenant_id,
            row.tier,
            row.arrival_work,
            sum(row.prefix_token_sizes),
            row.true_output_tokens,
        )
        result.append(
            KernelRequestSpec(
                logical,
                tuple(
                    _namespace_cache_key(
                        row.model_id,
                        row.adapter_id,
                        row.work_shape,
                        raw_key,
                    )
                    for raw_key in row.prefix_cache_keys
                ),
                row.prefix_token_sizes,
            )
        )
        truth[row.logical_request_id] = CohortTruth(
            row.model_id,
            row.adapter_id,
            row.work_shape,
            row.slo_slack_work,
            frozenset(row.eligible_node_ids),
            row.prefix_cache_keys,
        )
    return PlacementWorkload(tuple(result), truth)


@dataclass(frozen=True, slots=True)
class PlacementDecisionRecord:
    logical_request_id: str
    node_id: str
    queue_work: float
    spilled: bool
    decode_queue_work: float


@dataclass(frozen=True, slots=True)
class PlacementRunReport:
    strategy_id: str
    kernel_metrics: M12MetricReport
    local_hit_tokens: int
    remote_hit_tokens: int
    uncached_tokens: int
    token_hit_rate: float
    request_load_max_mean: float
    p_queue_p95: float
    kvs_normalized_work: float
    spill_count: int
    decode_normalized_utilization: float
    decode_queue_p95: float
    completion_max_work: float

    def gate_metrics(self) -> PlacementStrategyMetrics:
        return PlacementStrategyMetrics(
            self.strategy_id,
            self.token_hit_rate,
            self.request_load_max_mean,
            self.p_queue_p95,
            self.kernel_metrics.strict_useful_token_goodput,
            self.kernel_metrics.strict_useful_output_token_goodput,
            self.kernel_metrics.minimum_tier_slo_attainment,
            self.kernel_metrics.jain_fairness,
            self.kernel_metrics.per_tier_slo_attainment,
        )


class M12PlacementPolicy(KernelPolicy):
    """Causal adapter for the Hybrid anchor and M12.2 Priced Spill."""

    eviction_regret_work = 0.0
    decode_debt_work = 0.0
    flexlb_cache_discount = 0.7
    flexlb_top_fraction = 0.3
    similar_band_ratio = 0.1

    def __init__(
        self,
        mode: PlacementMode,
        cost_model: FrozenKernelCostModel,
        *,
        kvs_enabled: bool,
        mooncake_balancing_threshold: float = 2.0,
        request_truth: Mapping[str, CohortTruth] | None = None,
        kvs_contention_multiplier: float = 1.0,
    ) -> None:
        if not isinstance(mode, PlacementMode):
            raise ValueError("mode must be a PlacementMode")
        if mooncake_balancing_threshold <= 0:
            raise ValueError("Mooncake balancing threshold must be positive")
        if kvs_contention_multiplier < 1 or not math.isfinite(
            kvs_contention_multiplier
        ):
            raise ValueError("KVS contention multiplier must be finite and >= 1")
        self.mode = mode
        self.cost_model = cost_model
        self.kvs_enabled = kvs_enabled
        self.mooncake_balancing_threshold = mooncake_balancing_threshold
        self.request_truth = dict(request_truth or {})
        self.kvs_contention_multiplier = kvs_contention_multiplier
        self.decisions: list[PlacementDecisionRecord] = []
        self._node_ids: tuple[str, ...] = ()
        self._last_selected: dict[str, int] = {}
        self._selection_clock = 0
        self._prefix_to_family: dict[tuple[str, ...], str] = {}
        self._family_sizes: Counter[str] = Counter()
        self._family_node: dict[str, str] = {}
        self._block_counts: Counter[str] = Counter()
        self._next_family = 0
        self.session_families: tuple[str, ...] = ()

    def plan_attempts(
        self, request: KernelRequestSpec, view: CausalView
    ) -> tuple[AttemptExecutionSpec, ...]:
        if not self._node_ids:
            self._node_ids = tuple(sorted(view.cache_by_node))
        if request.logical.input_tokens == 0:
            node_id = min(view.cache_by_node)
            remote_keys: tuple[str, ...] = ()
            spilled = False
            queue = self._queue(node_id, view)
        else:
            node_id, remote_keys, spilled, queue = self._select(request, view)
        d_node = min(
            view.decode_available_at,
            key=lambda node: (view.decode_available_at[node], node),
        )
        d_queue = max(0.0, view.decode_available_at[d_node] - view.now_work)
        self.decisions.append(
            PlacementDecisionRecord(
                request.logical.logical_request_id,
                node_id,
                queue,
                spilled,
                d_queue,
            )
        )
        self._selection_clock += 1
        self._last_selected[node_id] = self._selection_clock
        return (
            AttemptExecutionSpec(
                request.logical.logical_request_id,
                0,
                request.logical.arrival_work,
                node_id,
                d_node,
                request.logical.true_output_tokens,
                remote_keys,
            ),
        )

    def cache_mutation(
        self,
        request: KernelRequestSpec,
        attempt: AttemptExecutionSpec,
        view: CausalView,
    ) -> CacheMutation:
        return CacheMutation(admit=True)

    def summarize(self, result: KernelResult) -> PlacementRunReport:
        local = remote = uncached = 0
        for executed in result.attempts:
            outcome = executed.outcome
            uncached_tokens = _exact_tokens(
                outcome.prefill_gpu_work,
                self.cost_model.prefill_work_per_token,
            )
            remote_tokens = (
                outcome.kvs_bytes // self.cost_model.kvs_bytes_per_token
                if self.cost_model.kvs_bytes_per_token
                else 0
            )
            uncached += uncached_tokens
            remote += remote_tokens
            local += max(0, outcome.input_tokens - uncached_tokens - remote_tokens)
        total_input = sum(value.outcome.input_tokens for value in result.attempts)
        loads = Counter(record.node_id for record in self.decisions)
        counts = tuple(loads.get(node, 0) for node in self._node_ids)
        max_mean = max(counts) / (sum(counts) / len(counts)) if counts else 0.0
        queues = sorted(record.queue_work for record in self.decisions)
        d_queues = sorted(record.decode_queue_work for record in self.decisions)
        return PlacementRunReport(
            self.mode.value,
            result.metrics,
            local,
            remote,
            uncached,
            (local + remote) / total_input if total_input else 0.0,
            max_mean,
            _percentile95(queues),
            result.metrics.kvs_normalized_work,
            sum(record.spilled for record in self.decisions),
            result.metrics.decode_normalized_utilization,
            _percentile95(d_queues),
            max(
                (
                    executed.outcome.finish_work or result.metrics.observation_end_work
                    for executed in result.attempts
                ),
                default=0.0,
            ),
        )

    def _select(
        self, request: KernelRequestSpec, view: CausalView
    ) -> tuple[str, tuple[str, ...], bool, float]:
        all_observations = {
            node: _prefix_tokens(request, resident)
            for node, resident in view.cache_by_node.items()
        }
        observations = dict(all_observations)
        truth = self.request_truth.get(request.logical.logical_request_id)
        if truth is not None:
            observations = {
                node: hit
                for node, hit in observations.items()
                if node in truth.eligible_node_ids
            }
        best = max(all_observations.values(), default=0)
        if truth is not None:
            observations = {
                node: local
                for node, local in observations.items()
                if self._estimated_marginal(request, view, node, local, best)
                <= truth.slo_slack_work
            }
        if not observations:
            raise ValueError("no cohort-compatible SLO-eligible candidate")
        family = request.prefix_cache_keys[0]
        if self.mode is PlacementMode.PRICED_SPILL:
            candidates = tuple(
                PlacementCandidate(
                    node,
                    local,
                    max(0, best - local) if self.kvs_enabled else 0,
                    self._queue(node, view),
                    self.kvs_contention_multiplier,
                    truth.model_id if truth else "default",
                    truth.adapter_id if truth else "default",
                    truth.work_shape if truth else "default",
                    True,
                )
                for node, local in observations.items()
            )
            decision = PricedSpillSelector(
                self.cost_model.prefill_work_per_token,
                self.cost_model.kvs_work_per_token,
            ).select(
                prefix_family=family,
                input_tokens=request.logical.input_tokens,
                candidates=candidates,
                model_id=truth.model_id if truth else "default",
                adapter_id=truth.adapter_id if truth else "default",
                work_shape=truth.work_shape if truth else "default",
            )
            remote = best > observations[decision.node_id] and self.kvs_enabled
            return (
                decision.node_id,
                request.prefix_cache_keys if remote else (),
                decision.spilled,
                decision.p_queue_work,
            )
        if self.mode in (PlacementMode.S3, PlacementMode.S4):
            ordered = tuple(sorted(observations))
            if self.mode is PlacementMode.S4:
                family_id = self._causal_session_family(request)
                node = self._family_node.get(family_id, "")
                if node not in observations:
                    node = ordered[_stable_index(family_id, len(ordered))]
                self._family_node.setdefault(family_id, node)
            else:
                primary_index = _stable_index(family, len(ordered))
                node = ordered[primary_index]
                loads = tuple(self._queue(candidate, view) for candidate in ordered)
                mean = sum(loads) / len(loads)
                if mean > 0 and loads[primary_index] > 1.25 * mean:
                    secondary = tuple(
                        ordered[(primary_index + offset) % len(ordered)]
                        for offset in range(1, min(2, len(ordered) - 1) + 1)
                    )
                    node = min(
                        secondary or ordered,
                        key=lambda item: (self._queue(item, view), item),
                    )
            return node, (), False, self._queue(node, view)
        scored: list[tuple[float, str, bool]] = []
        for node, local in observations.items():
            queue = self._queue(node, view)
            remote = self.kvs_enabled and best > local
            if self.mode is PlacementMode.S5:
                score = (
                    queue
                    + request.logical.input_tokens
                    - self.flexlb_cache_discount * local
                )
            elif self.mode is PlacementMode.S6:
                score = (
                    queue
                    + (request.logical.input_tokens - local)
                    * self.cost_model.prefill_work_per_token
                )
            else:  # HYBRID: Mooncake Algorithm 1 threshold branch.
                use_remote = remote and (
                    local == 0 or best / local >= self.mooncake_balancing_threshold
                )
                planned = best if use_remote else local
                transfer = best - local if use_remote else 0
                score = (
                    queue
                    + (request.logical.input_tokens - planned)
                    * self.cost_model.prefill_work_per_token
                    + transfer
                    * self.cost_model.kvs_work_per_token
                    * self.kvs_contention_multiplier
                )
                remote = use_remote
            scored.append((score, node, remote))
        scored.sort(key=lambda value: (value[0], value[1]))
        best_score = scored[0][0]
        band = max(1.0, abs(best_score) * self.similar_band_ratio)
        if self.mode is PlacementMode.S5:
            top_count = max(1, math.ceil(len(scored) * self.flexlb_top_fraction))
            eligible = [
                value for value in scored[:top_count] if value[0] <= best_score + band
            ]
        elif self.mode is PlacementMode.S6:
            eligible = [value for value in scored if value[0] <= best_score + band]
        else:
            eligible = [scored[0]]
        _, node, remote = min(
            eligible,
            key=lambda value: (self._last_selected.get(value[1], -1), value[1]),
        )
        home = sorted(observations)[_stable_index(family, len(observations))]
        return (
            node,
            request.prefix_cache_keys if remote else (),
            node != home,
            self._queue(node, view),
        )

    @staticmethod
    def _queue(node: str, view: CausalView) -> float:
        return view.queued_prefill_work[node] + max(
            0.0, view.prefill_available_at[node] - view.now_work
        )

    def _estimated_marginal(
        self,
        request: KernelRequestSpec,
        view: CausalView,
        node: str,
        local: int,
        best: int,
    ) -> float:
        remote = max(0, best - local) if self.kvs_enabled else 0
        return (
            self._queue(node, view)
            + (request.logical.input_tokens - local - remote)
            * self.cost_model.prefill_work_per_token
            + remote
            * self.cost_model.kvs_work_per_token
            * self.kvs_contention_multiplier
        )

    def _causal_session_family(self, request: KernelRequestSpec) -> str:
        truth = self.request_truth.get(request.logical.logical_request_id)
        raw_blocks = (
            truth.raw_prefix_keys if truth is not None else request.prefix_cache_keys
        )[:32]
        cohort = (
            _canonical_identity(
                b"m12-cohort-v1\x00",
                (truth.model_id, truth.adapter_id, truth.work_shape),
            )
            if truth is not None
            else "default"
        )
        count_keys = request.prefix_cache_keys[: len(raw_blocks)]
        family: str | None = None
        if len(raw_blocks) >= 2:
            for depth in range(len(raw_blocks), 1, -1):
                prefix = (cohort, *raw_blocks[:depth])
                candidate = self._prefix_to_family.get(prefix)
                hot_only = all(
                    self._block_counts[block] >= 8 for block in count_keys[:depth]
                )
                if (
                    candidate is not None
                    and self._family_sizes[candidate] < 64
                    and not hot_only
                ):
                    family = candidate
                    break
        if family is None:
            family = f"family:{self._next_family}"
            self._next_family += 1
        self._family_sizes[family] += 1
        for depth in range(2, len(raw_blocks) + 1):
            prefix = (cohort, *raw_blocks[:depth])
            current = self._prefix_to_family.get(prefix)
            if current is None or self._family_sizes[current] >= 64:
                self._prefix_to_family[prefix] = family
        self._block_counts.update(count_keys)
        self.session_families = (*self.session_families, family)
        return family


@dataclass(frozen=True, slots=True)
class M12PlacementCase:
    case_id: str
    regime: SyntheticServiceRegime
    kvs_mode: KvsPriceMode
    horizon: float
    tier_slo_work: Mapping[str, float]
    decode_binding: bool = False
    kvs_contention_multiplier: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tier_slo_work", MappingProxyType(dict(self.tier_slo_work))
        )
        if not self.tier_slo_work:
            raise ValueError("case requires explicit tier SLO mapping")


@dataclass(frozen=True, slots=True)
class PlacementPairResult:
    case: M12PlacementCase
    hybrid: PlacementRunReport
    priced_spill: PlacementRunReport
    verdict: G12TwoVerdict
    result_table: tuple[PlacementRunReport, ...]


def build_m12_2_cases(
    *, horizon: float, tier_slo_work: Mapping[str, float]
) -> tuple[M12PlacementCase, ...]:
    cases = [
        M12PlacementCase(
            f"{regime.regime_id.value}-{mode.value}",
            regime,
            mode,
            horizon,
            tier_slo_work,
        )
        for regime in SERVICE_REGIMES
        for mode in KvsPriceMode
    ]
    cases.extend(
        M12PlacementCase(
            f"{regime.regime_id.value}-DECODE_BINDING",
            regime,
            KvsPriceMode.NORMAL,
            horizon,
            tier_slo_work,
            True,
        )
        for regime in SERVICE_REGIMES
    )
    cases.extend(
        M12PlacementCase(
            f"{regime.regime_id.value}-KVS_CONTENDED",
            regime,
            KvsPriceMode.NORMAL,
            horizon,
            tier_slo_work,
            False,
            3.0,
        )
        for regime in SERVICE_REGIMES
    )
    return tuple(cases)


def run_placement_case(
    workload: Sequence[KernelRequestSpec], case: M12PlacementCase
) -> PlacementPairResult:
    reports: list[PlacementRunReport] = []
    modes = (
        PlacementMode.HYBRID,
        PlacementMode.PRICED_SPILL,
        PlacementMode.S3,
        PlacementMode.S4,
        PlacementMode.S5,
        PlacementMode.S6,
    )
    truth = workload.request_truth if isinstance(workload, PlacementWorkload) else {}
    for mode in modes:
        cost = _case_cost(case)
        config = KernelConfig(
            0,
            case.horizon,
            ("p0", "p1"),
            ("d0",) if case.decode_binding else ("d0", "d1"),
            case.tier_slo_work,
            max(1, sum(len(item.prefix_cache_keys) for item in workload)),
            cost,
        )
        policy = M12PlacementPolicy(
            mode,
            cost,
            kvs_enabled=case.kvs_mode is not KvsPriceMode.DISABLED,
            request_truth=truth,
            kvs_contention_multiplier=1.0,
        )
        reports.append(policy.summarize(CausalKernel(config).run(workload, policy)))
    hybrid, priced = reports[:2]
    if case.decode_binding and not any(
        report.decode_normalized_utilization >= 0.8 or report.decode_queue_p95 > 0
        for report in reports
    ):
        raise ValueError(
            "case is marked decode-binding but executed ledger is not decode-binding"
        )
    return PlacementPairResult(
        case,
        hybrid,
        priced,
        evaluate_g12_2(hybrid.gate_metrics(), priced.gate_metrics()),
        tuple(reports),
    )


def _case_cost(case: M12PlacementCase) -> FrozenKernelCostModel:
    multiplier = {
        KvsPriceMode.DISABLED: 0.0,
        KvsPriceMode.NORMAL: 1.0,
        KvsPriceMode.EXPENSIVE: 1_000_000.0,
    }[case.kvs_mode]
    return FrozenKernelCostModel(
        case.regime.prefill_token_work,
        case.regime.kvs_byte_work * multiplier * case.kvs_contention_multiplier,
        1,
        case.regime.decode_token_work * (4 if case.decode_binding else 1),
    )


def _prefix_tokens(request: KernelRequestSpec, resident: frozenset[str]) -> int:
    tokens = 0
    for key, size in zip(
        request.prefix_cache_keys, request.prefix_token_sizes, strict=True
    ):
        if key not in resident:
            break
        tokens += size
    return tokens


def _exact_tokens(work: float, coefficient: float) -> int:
    if coefficient == 0:
        return 0
    value = work / coefficient
    rounded = round(value)
    if not math.isclose(value, rounded, abs_tol=1e-8):
        raise ValueError("executed work does not map to an integral token count")
    return rounded


def _percentile95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    index = math.ceil(0.95 * len(values)) - 1
    return values[index]


def _namespace_cache_key(
    model_id: str, adapter_id: str, work_shape: str, raw_key: str
) -> str:
    """Create a versioned length-framed SHA-256 cache identity."""
    digest = _canonical_identity(
        b"m12-cache-key-v1\x00", (model_id, adapter_id, work_shape, raw_key)
    )
    return f"m12:v1:{digest}"


def _canonical_identity(domain: bytes, values: tuple[str, ...]) -> str:
    digest = hashlib.sha256(domain)
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()
