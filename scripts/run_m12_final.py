#!/usr/bin/env python3
"""M12.3–M12.5 fixed-grid final experiment runner.

The trace is parsed and frozen once.  Oracle variants are isolated in sensitivity
artifacts and eviction variants are created only for predeclared binding cells.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from prefill_cache_sim.config import git_provenance  # noqa: E402
from prefill_cache_sim.m12_decode import (  # noqa: E402
    AbortFence,
    AdmissionAction,
    DecodeAdmissionConfig,
    DecodeAdmissionMode,
    DecodeCapacityPolicy,
    DecodeRunReport,
    PrefixFamilyPredictor,
    evaluate_g12_3,
)
from prefill_cache_sim.m12_eviction import (  # noqa: E402
    CensusConfig,
    ClusterCacheCensus,
    EvictionMode,
    EvictionRunReport,
    M12EvictionConfig,
    M12EvictionPolicy,
    evaluate_g12_4,
)
from prefill_cache_sim.m12_kernel import (  # noqa: E402
    AttemptExecutionSpec,
    AttemptTerminal,
    CacheMutation,
    CausalKernel,
    CausalView,
    FrozenKernelCostModel,
    KernelConfig,
    KernelPolicy,
    KernelRequestSpec,
)
from prefill_cache_sim.m12_metrics import SERVICE_REGIMES  # noqa: E402
from prefill_cache_sim.m12_placement import (  # noqa: E402
    M12PlacementPolicy,
    PlacementMode,
    PlacementWorkload,
)
from prefill_cache_sim.replay.fingerprint import source_manifest  # noqa: E402
from scripts.run_m12_placement import (  # noqa: E402
    OBSERVATION_END_WORK,
    TIER_SLO_WORK,
    TRUTH_BASIS,
    build_trace_workload,
)

ARRIVAL_SCALES = (0.8, 1.0, 1.2, 1.5, 2.0)
PRIMARY_STRATEGIES = ("BASELINE", "PRICED_SPILL", "DECODE_CAUSAL")
SENSITIVITY_STRATEGIES = (
    "DECODE_NO_GATE",
    "DECODE_ORACLE",
    "DECODE_ORACLE_NOISED",
)
MANIFEST_SCHEMA = "m12-final-manifest-v1"


@dataclass(frozen=True, slots=True)
class FinalCell:
    regime: str
    arrival_scale: float
    strategy: str
    category: str

    @property
    def cell_id(self) -> str:
        return f"{self.regime}-{self.arrival_scale:.1f}x-{self.strategy}"


@dataclass(frozen=True, slots=True)
class CellResult:
    cell: FinalCell
    offered_requests: int
    offered_tokens: int
    strict_goodput: float
    strict_output_goodput: float
    request_goodput: float
    minimum_tier: float
    jain: float
    per_tier: Mapping[str, float]
    queue_p95_normalized_work: float
    token_hit_rate: float
    waste_fraction: float
    load_skew: float
    kvs_work: float
    p_utilization: float
    d_utilization: float
    p_to_d_debt: float
    total_work: float
    accounted_work: float
    capacity_binding: bool
    cache_capacity_entries: int = 0
    decode_report: DecodeRunReport | None = None
    eviction_report: EvictionRunReport | None = None
    hit_ceiling: float = 1.0
    decision_log: tuple[str, ...] = ()
    decision_fingerprint: str = ""
    census_age_work: float | None = None
    visibility_delay_work: float = 0.0
    attempt_count: int = -1
    retry_count: int = -1
    congestion_action: str | None = None
    gated_retry_count: int = 0

    def __post_init__(self) -> None:
        if self.attempt_count == -1:
            object.__setattr__(self, "attempt_count", self.offered_requests)
        if self.retry_count == -1:
            object.__setattr__(
                self, "retry_count", self.attempt_count - self.offered_requests
            )
        numeric = (
            self.strict_goodput,
            self.strict_output_goodput,
            self.request_goodput,
            self.minimum_tier,
            self.jain,
            self.queue_p95_normalized_work,
            self.token_hit_rate,
            self.waste_fraction,
            self.load_skew,
            self.kvs_work,
            self.p_utilization,
            self.d_utilization,
            self.p_to_d_debt,
            self.total_work,
            self.accounted_work,
            *self.per_tier.values(),
        )
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("cell results must be finite")
        if self.offered_requests < 0 or self.offered_tokens < 0:
            raise ValueError("offered workload must be non-negative")
        if self.attempt_count < self.offered_requests or self.retry_count < 0:
            raise ValueError("attempt and retry counts must cover offered requests")
        if self.retry_count != self.attempt_count - self.offered_requests:
            raise ValueError("retry count must equal attempts minus offered requests")
        if not 0 <= self.gated_retry_count <= self.retry_count:
            raise ValueError("gated retry count must be covered by retries")


def build_cell_plan(
    binding_cells: set[tuple[str, float]],
) -> tuple[FinalCell, ...]:
    regimes = tuple(regime.regime_id.value for regime in SERVICE_REGIMES)
    cells = [
        FinalCell(regime, scale, strategy, "PRIMARY")
        for regime in regimes
        for scale in ARRIVAL_SCALES
        for strategy in PRIMARY_STRATEGIES
    ]
    cells.extend(
        FinalCell(regime, scale, strategy, "PRIMARY")
        for regime, scale in sorted(binding_cells)
        for strategy in ("EVICTION_LRU", "CENSUS_EVICTION")
    )
    cells.extend(
        FinalCell(regime, 1.5, strategy, "SENSITIVITY")
        for regime in regimes
        for strategy in SENSITIVITY_STRATEGIES
    )
    return tuple(cells)


def placement_run_active(process_lines: Iterable[str] | None = None) -> bool:
    if process_lines is None:
        output = subprocess.run(
            ("ps", "-axo", "pid=,command="),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        process_lines = output.splitlines()
    return any(
        "scripts/run_m12_placement.py" in line and "grep" not in line
        for line in process_lines
    )


def run_artifacts(
    trace_path: Path,
    output_dir: Path,
    *,
    executor: Callable[[object, FinalCell], CellResult] | None = None,
    workload_loader: Callable[[Path], tuple[object, Mapping[str, object]]] = (
        build_trace_workload
    ),
    binding_cells: set[tuple[str, float]] | None = None,
) -> dict[str, bytes]:
    workload, trace_metadata = workload_loader(trace_path)
    execute = executor or execute_cell
    cached_results: dict[str, CellResult] = {}
    discovered_hit_ceiling: float | None = None
    if binding_cells is None:
        if not isinstance(workload, Sequence):
            raise ValueError("binding discovery requires a frozen workload")
        hit_ceiling = _causal_hit_ceiling(workload)
        discovered_hit_ceiling = hit_ceiling
        binding = set()
        probes = tuple(
            (regime, scale)
            for regime in (item.regime_id.value for item in SERVICE_REGIMES)
            for scale in ARRIVAL_SCALES
        )
        for probe_index, (regime, scale) in enumerate(probes, start=1):
            print(
                f"[probe {probe_index}/{len(probes)}] {regime}-{scale:.1f}x",
                file=sys.stderr,
                flush=True,
            )
            probe = FinalCell(regime, scale, "EVICTION_LRU", "PRIMARY")
            result = execute(workload, probe)
            is_binding = hit_ceiling - result.token_hit_rate >= 0.03
            if is_binding:
                binding.add((regime, scale))
                cached_results[probe.cell_id] = replace(
                    result,
                    capacity_binding=True,
                    hit_ceiling=hit_ceiling,
                )
            else:
                del result
    else:
        binding = set(binding_cells)
    plan = build_cell_plan(binding)
    results: list[CellResult] = []
    failures: list[dict[str, object]] = []
    for index, cell in enumerate(plan, start=1):
        print(
            f"[{index:03d}/{len(plan):03d}] {cell.cell_id}",
            file=sys.stderr,
            flush=True,
        )
        try:
            result = cached_results.get(cell.cell_id) or execute(workload, cell)
            if result.cell != cell:
                raise ValueError("executor returned a mismatched cell identity")
            if (
                cell.strategy in ("EVICTION_LRU", "CENSUS_EVICTION")
                and discovered_hit_ceiling is not None
            ):
                result = replace(result, hit_ceiling=discovered_hit_ceiling)
            results.append(result)
        except (ValueError, RuntimeError) as error:
            failures.append(
                {
                    "cell_id": cell.cell_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
    if not results:
        raise ValueError("final experiment produced no finite results")
    artifacts = _build_artifacts(
        results,
        failures,
        trace_metadata,
        binding,
        workload=workload,
        executor=execute,
    )
    _write_atomic(output_dir, artifacts)
    return artifacts


def with_visibility_delay(
    executor: Callable[[object, FinalCell], CellResult], delta_work: float
) -> Callable[[object, FinalCell], CellResult]:
    """Independently rerun with completion-derived policy knowledge delayed."""
    if delta_work < 0 or not math.isfinite(delta_work):
        raise ValueError("visibility delay must be finite and non-negative")

    def wrapped(workload: object, cell: FinalCell) -> CellResult:
        if executor is execute_cell:
            result = execute_cell(workload, cell, visibility_delay_work=delta_work)
        else:
            result = executor(workload, cell)
        if not math.isclose(result.total_work, result.accounted_work):
            raise ValueError("visibility audit requires conserved work")
        return replace(
            result,
            visibility_delay_work=delta_work,
        )

    return wrapped


class _FinalEvictionPolicy(M12EvictionPolicy):
    """Eviction owns mutation; all decode lifecycle hooks stay composed."""

    def plan_attempts(
        self, request: KernelRequestSpec, view: CausalView
    ) -> Sequence[AttemptExecutionSpec]:
        return super().plan_attempts(request, view)

    def admission_event(
        self, request: KernelRequestSpec, attempt: AttemptExecutionSpec
    ) -> str | None:
        return self.placement.admission_event(request, attempt)

    def decode_not_before(
        self,
        request: KernelRequestSpec,
        attempt: AttemptExecutionSpec,
        view: CausalView,
    ) -> float:
        return self.placement.decode_not_before(request, attempt, view)

    def decode_lease_tokens(
        self,
        request: KernelRequestSpec,
        attempt: AttemptExecutionSpec,
        view: CausalView,
    ) -> int | None:
        return self.placement.decode_lease_tokens(request, attempt, view)

    def reprice_decode(
        self,
        request: KernelRequestSpec,
        attempt: AttemptExecutionSpec,
        view: CausalView,
    ) -> AttemptExecutionSpec:
        return self.placement.reprice_decode(request, attempt, view)

    def decode_started(
        self,
        request: KernelRequestSpec,
        attempt: AttemptExecutionSpec,
        view: CausalView,
        *,
        finish_work: float,
    ) -> None:
        self.placement.decode_started(request, attempt, view, finish_work=finish_work)

    def attempt_finished(
        self,
        request: KernelRequestSpec,
        attempt: AttemptExecutionSpec,
        view: CausalView,
        *,
        actual_decode_work: float,
    ) -> None:
        self.placement.attempt_finished(
            request,
            attempt,
            view,
            actual_decode_work=actual_decode_work,
        )

    def next_attempt(
        self,
        request: KernelRequestSpec,
        previous: AttemptExecutionSpec,
        view: CausalView,
    ) -> AttemptExecutionSpec | None:
        return self.placement.next_attempt(request, previous, view)

    def cache_mutation(
        self,
        request: KernelRequestSpec,
        attempt: AttemptExecutionSpec,
        view: CausalView,
    ) -> CacheMutation:
        self.placement.cache_mutation(request, attempt, view)
        return super().cache_mutation(request, attempt, view)


class _CacheDigestLedger:
    """Order-independent O(1)-per-key digest without retaining cache identities."""

    _DOMAIN = b"m12-census-input-v1\x00"

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._digests: dict[str, int] = {}

    @classmethod
    def _key_digest(cls, key: str) -> int:
        return int.from_bytes(
            hashlib.sha256(cls._DOMAIN + key.encode("utf-8")).digest(), "big"
        )

    def snapshot(self, node: str) -> tuple[int, str]:
        return self._counts.get(node, 0), f"{self._digests.get(node, 0):064x}"

    def apply(
        self,
        node: str,
        request_keys: Sequence[str],
        resident: frozenset[str],
        mutation: CacheMutation,
    ) -> None:
        count, digest = self.snapshot(node)
        count_value = count
        digest_value = int(digest, 16)
        victims = frozenset(mutation.evict_keys)
        for key in victims:
            count_value -= 1
            digest_value ^= self._key_digest(key)
        if mutation.admit:
            for key in dict.fromkeys(request_keys):
                if key not in resident or key in victims:
                    count_value += 1
                    digest_value ^= self._key_digest(key)
        self._counts[node] = count_value
        self._digests[node] = digest_value


def _causal_hit_ceiling(workload: Sequence[KernelRequestSpec]) -> float:
    seen: set[str] = set()
    hit_tokens = 0
    input_tokens = 0
    for request in workload:
        input_tokens += request.logical.input_tokens
        for key, size in zip(
            request.prefix_cache_keys, request.prefix_token_sizes, strict=True
        ):
            if key not in seen:
                break
            hit_tokens += size
        seen.update(request.prefix_cache_keys)
    return hit_tokens / input_tokens if input_tokens else 0.0


class _DelayedPredictor(PrefixFamilyPredictor):
    def __init__(self, inner: PrefixFamilyPredictor, delay_work: float) -> None:
        super().__init__(default_output_tokens=inner.default_output_tokens)
        self.inner = inner
        self.delay_work = delay_work
        self._pending: list[tuple[float, int, KernelRequestSpec]] = []
        self._sequence = 0

    def predict(self, request: KernelRequestSpec, *, at_work: float) -> int:
        while self._pending and self._pending[0][0] <= at_work:
            visible_at, _, observed = heapq.heappop(self._pending)
            self.inner.observe(observed, completed_at_work=visible_at)
        return self.inner.predict(request, at_work=at_work)

    def observe(self, request: KernelRequestSpec, *, completed_at_work: float) -> None:
        self._sequence += 1
        heapq.heappush(
            self._pending,
            (completed_at_work + self.delay_work, self._sequence, request),
        )


class _DelayedCensus(ClusterCacheCensus):
    def __init__(self, config: CensusConfig, delay_work: float) -> None:
        super().__init__(config)
        self.delay_work = delay_work
        self._pending_observations: list[
            tuple[float, int, float, str, str, str, float]
        ] = []
        self._sequence = 0

    def observe(
        self,
        cache_key: str,
        cohort_id: str,
        holder_id: str,
        *,
        at_work: float,
        recovery_work: float,
    ) -> bool:
        self._sequence += 1
        heapq.heappush(
            self._pending_observations,
            (
                at_work + self.delay_work,
                self._sequence,
                at_work,
                cache_key,
                cohort_id,
                holder_id,
                recovery_work,
            ),
        )
        return True

    def lookup(self, cache_key: str, cohort_id: str, *, now_work: float):
        self._flush(now_work)
        return super().lookup(cache_key, cohort_id, now_work=now_work)

    def remove(self, cache_key: str, cohort_id: str, holder_id: str) -> None:
        self._pending_observations = [
            item
            for item in self._pending_observations
            if (item[3], item[4], item[5]) != (cache_key, cohort_id, holder_id)
        ]
        heapq.heapify(self._pending_observations)
        super().remove(cache_key, cohort_id, holder_id)

    def _flush(self, now_work: float) -> None:
        while (
            self._pending_observations and self._pending_observations[0][0] <= now_work
        ):
            _, _, captured_at, key, cohort, holder, recovery = heapq.heappop(
                self._pending_observations
            )
            super().observe(
                key,
                cohort,
                holder,
                at_work=captured_at,
                recovery_work=recovery,
            )


class _DecisionLedgerPolicy(KernelPolicy):
    def __init__(
        self,
        inner: KernelPolicy,
        placement: M12PlacementPolicy,
        decode: DecodeCapacityPolicy | None,
        eviction: M12EvictionPolicy | None,
    ) -> None:
        self.inner = inner
        self.placement = placement
        self.decode = decode
        self.eviction = eviction
        self.records: list[dict[str, object]] = []
        self._sequence = 0
        self._cache_digest = _CacheDigestLedger()

    def _append(self, record: dict[str, object], *, at_work: float) -> None:
        record["decision_time_work"] = at_work
        record["sequence"] = self._sequence
        self._sequence += 1
        self.records.append(record)

    def plan_attempts(
        self, request: KernelRequestSpec, view: CausalView
    ) -> Sequence[AttemptExecutionSpec]:
        attempts = tuple(self.inner.plan_attempts(request, view))
        placement_record = self.placement.decisions[-1]
        for attempt in attempts:
            self._append(
                {
                    "logical_id": attempt.logical_request_id,
                    "attempt_index": attempt.attempt_index,
                    "decision_kind": "PLACEMENT",
                    "node": placement_record.node_id,
                    "spill": placement_record.spilled,
                    "components": {
                        "prefill_queue_work": placement_record.queue_work,
                        "decode_queue_work": placement_record.decode_queue_work,
                    },
                },
                at_work=view.now_work,
            )
            if self.decode is not None:
                decision = self.decode.decisions[-1]
                self._append(
                    {
                        "logical_id": attempt.logical_request_id,
                        "attempt_index": attempt.attempt_index,
                        "decision_kind": "DECODE",
                        "admission_action": decision.action.value,
                        "predicted_output_tokens": decision.predicted_output_tokens,
                        "reserved_credits": decision.reserved_decode_credits,
                        "debt_credits": self.decode.ledger.p_to_d_debt_credits,
                    },
                    at_work=view.now_work,
                )
        return attempts

    def admission_event(
        self, request: KernelRequestSpec, attempt: AttemptExecutionSpec
    ) -> str | None:
        return self.inner.admission_event(request, attempt)

    def decode_not_before(
        self,
        request: KernelRequestSpec,
        attempt: AttemptExecutionSpec,
        view: CausalView,
    ) -> float:
        return self.inner.decode_not_before(request, attempt, view)

    def decode_lease_tokens(
        self,
        request: KernelRequestSpec,
        attempt: AttemptExecutionSpec,
        view: CausalView,
    ) -> int | None:
        return self.inner.decode_lease_tokens(request, attempt, view)

    def reprice_decode(
        self,
        request: KernelRequestSpec,
        attempt: AttemptExecutionSpec,
        view: CausalView,
    ) -> AttemptExecutionSpec:
        repriced = self.inner.reprice_decode(request, attempt, view)
        if repriced.terminal is AttemptTerminal.ABORTED and self.decode is not None:
            fence = self.decode.config.abort_fences.get(
                request.logical.logical_request_id
            )
            self._append(
                {
                    "logical_id": repriced.logical_request_id,
                    "attempt_index": repriced.attempt_index,
                    "decision_kind": "ABORT_FENCE",
                    "fence_valid": fence is not None and fence.allows_abort,
                    "terminal": repriced.terminal.value,
                },
                at_work=view.now_work,
            )
        return repriced

    def decode_started(
        self,
        request: KernelRequestSpec,
        attempt: AttemptExecutionSpec,
        view: CausalView,
        *,
        finish_work: float,
    ) -> None:
        self.inner.decode_started(request, attempt, view, finish_work=finish_work)

    def attempt_finished(
        self,
        request: KernelRequestSpec,
        attempt: AttemptExecutionSpec,
        view: CausalView,
        *,
        actual_decode_work: float,
    ) -> None:
        self.inner.attempt_finished(
            request, attempt, view, actual_decode_work=actual_decode_work
        )

    def next_attempt(
        self,
        request: KernelRequestSpec,
        previous: AttemptExecutionSpec,
        view: CausalView,
    ) -> AttemptExecutionSpec | None:
        retry = self.inner.next_attempt(request, previous, view)
        if retry is not None and self.decode is not None:
            key = (retry.logical_request_id, retry.attempt_index)
            credits = self.decode._credits.get(key, 0.0)
            action = (
                self.decode.config.congestion_action.value
                if retry.arrival_work > view.now_work
                else "ADMIT"
            )
            self._append(
                {
                    "logical_id": retry.logical_request_id,
                    "attempt_index": retry.attempt_index,
                    "decision_kind": "DECODE",
                    "admission_action": action,
                    "predicted_output_tokens": (
                        credits / self.placement.cost_model.decode_work_per_token
                        if self.placement.cost_model.decode_work_per_token
                        else 0.0
                    ),
                    "reserved_credits": credits,
                    "debt_credits": self.decode.ledger.p_to_d_debt_credits,
                },
                at_work=view.now_work,
            )
        return retry

    def cache_mutation(
        self,
        request: KernelRequestSpec,
        attempt: AttemptExecutionSpec,
        view: CausalView,
    ) -> CacheMutation:
        mutation = self.inner.cache_mutation(request, attempt, view)
        if self.eviction is not None:
            resident = view.cache_by_node[attempt.p_node_id]
            census_input_count, census_input_digest = self._cache_digest.snapshot(
                attempt.p_node_id
            )
            if census_input_count != len(resident):
                raise ValueError("cache digest ledger diverged from causal cache size")
            refreshed_at = self.eviction.last_decision_census_refresh_work
            self._append(
                {
                    "logical_id": attempt.logical_request_id,
                    "attempt_index": attempt.attempt_index,
                    "decision_kind": "EVICTION",
                    "victims": list(mutation.evict_keys),
                    "census_age_work": (
                        None
                        if refreshed_at is None
                        else view.now_work - refreshed_at
                    ),
                    "census_refreshed_at_work": refreshed_at,
                    "census_input_count": census_input_count,
                    "census_input_digest": census_input_digest,
                },
                at_work=view.now_work,
            )
            self._cache_digest.apply(
                attempt.p_node_id,
                request.prefix_cache_keys,
                resident,
                mutation,
            )
        return mutation


def execute_cell(
    workload: object,
    cell: FinalCell,
    *,
    visibility_delay_work: float = 0.0,
) -> CellResult:
    if not isinstance(workload, Sequence):
        raise ValueError("executor requires a frozen kernel workload")
    scaled = _scaled_workload(workload, cell.arrival_scale)
    regime = next(
        item for item in SERVICE_REGIMES if item.regime_id.value == cell.regime
    )
    cost = FrozenKernelCostModel(
        regime.prefill_token_work,
        regime.kvs_byte_work,
        1,
        regime.decode_token_work,
    )
    truth = scaled.request_truth if isinstance(scaled, PlacementWorkload) else {}
    placement_mode = (
        PlacementMode.HYBRID
        if cell.strategy == "BASELINE"
        else PlacementMode.PRICED_SPILL
    )
    placement = M12PlacementPolicy(
        placement_mode,
        cost,
        kvs_enabled=True,
        request_truth=truth,
    )
    policy: object = placement
    decode: DecodeCapacityPolicy | None = None
    eviction: M12EvictionPolicy | None = None
    decode_mode = {
        "DECODE_NO_GATE": DecodeAdmissionMode.NO_GATE,
        "DECODE_ORACLE": DecodeAdmissionMode.ORACLE,
        "DECODE_ORACLE_NOISED": DecodeAdmissionMode.ORACLE_NOISED,
    }.get(cell.strategy, DecodeAdmissionMode.CAUSAL)
    if cell.strategy not in ("BASELINE", "PRICED_SPILL"):
        retry_pressure_cell = (
            cell.regime == "MIXED"
            and cell.arrival_scale == 1.5
            and cell.strategy == "DECODE_CAUSAL"
        )
        abort_fences = _retry_pressure_abort_fences(
            scaled, enabled=retry_pressure_cell
        )
        decode = DecodeCapacityPolicy(
            placement,
            DecodeAdmissionConfig(
                decode_mode,
                capacity_credits=max(
                    1.0,
                    (128 if retry_pressure_cell else 4096)
                    * cost.decode_work_per_token,
                ),
                congestion_action=(
                        AdmissionAction.GATED_DP
                    if retry_pressure_cell
                    else AdmissionAction.DEFER
                ),
                oracle_noise_multiplier=1.25,
                abort_fences=abort_fences,
            ),
            predictor=_DelayedPredictor(
                PrefixFamilyPredictor(default_output_tokens=128),
                visibility_delay_work,
            ),
        )
        policy = decode
    unique_keys = {key for item in scaled for key in item.prefix_cache_keys}
    bounded_capacity = max(1, len(unique_keys) // 4)
    if cell.strategy in ("EVICTION_LRU", "CENSUS_EVICTION"):
        census_config = CensusConfig(
            max(1, len(unique_keys)), OBSERVATION_END_WORK / 100
        )
        census = (
            _DelayedCensus(census_config, visibility_delay_work)
            if visibility_delay_work
            else ClusterCacheCensus(census_config)
        )
        eviction = _FinalEvictionPolicy(
            policy,  # type: ignore[arg-type]
            M12EvictionConfig(
                EvictionMode.LRU
                if cell.strategy == "EVICTION_LRU"
                else EvictionMode.CENSUS_REGRET,
                bounded_capacity,
                cost.prefill_work_per_token,
                cost.kvs_work_per_token,
                {},
            ),
            census,
            cache_key_cohorts={key: key for key in sorted(unique_keys)},
        )
        policy = eviction
    config = KernelConfig(
        0,
        OBSERVATION_END_WORK,
        ("p0", "p1"),
        ("d0", "d1"),
        TIER_SLO_WORK,
        bounded_capacity if eviction is not None else max(1, len(unique_keys)),
        cost,
    )
    ledger_policy = _DecisionLedgerPolicy(
        policy,  # type: ignore[arg-type]
        placement,
        decode,
        eviction,
    )
    result = CausalKernel(config).run(scaled, ledger_policy)
    placement_report = placement.summarize(result)
    decode_report = decode.summarize(result) if decode is not None else None
    eviction_report = eviction.summarize(result) if eviction is not None else None
    metrics = result.metrics
    accounted = (
        metrics.winner_attributable_gpu_work
        + metrics.slo_missed_gpu_work
        + metrics.wasted_gpu_work
        + metrics.unclassified_attempt_gpu_work
    )
    debt = decode.ledger.p_to_d_debt_credits if decode is not None else 0.0
    decision_log = tuple(
        json.dumps(item, sort_keys=True, separators=(",", ":"))
        for item in ledger_policy.records
    )
    decision_fingerprint = hashlib.sha256("\n".join(decision_log).encode()).hexdigest()
    measured_census_ages = [
        age
        for record in ledger_policy.records
        if record.get("decision_kind") == "EVICTION"
        and isinstance(age := record.get("census_age_work"), (int, float))
    ]
    gated_retry_count = 0
    if decode is not None:
        gated_ids = {
            decision.logical_request_id
            for decision in decode.decisions
            if decision.action in (AdmissionAction.GATED_PD, AdmissionAction.GATED_DP)
        }
        valid_fence_ids = {
            logical_id
            for logical_id, fence in decode.config.abort_fences.items()
            if fence.allows_abort
        }
        aborted_ids = {
            str(record["logical_id"])
            for record in ledger_policy.records
            if record.get("decision_kind") == "ABORT_FENCE"
            and record.get("fence_valid") is True
            and record.get("terminal") == AttemptTerminal.ABORTED.value
        }
        retried_ids = {
            attempt.logical_request_id
            for attempt in result.attempts
            if attempt.attempt_index > 0
        }
        gated_retry_count = len(
            gated_ids & valid_fence_ids & aborted_ids & retried_ids
        )
    return CellResult(
        cell,
        metrics.offered_logical_requests,
        metrics.offered_input_tokens + metrics.offered_output_tokens,
        metrics.strict_useful_token_goodput,
        metrics.strict_useful_output_token_goodput,
        metrics.request_goodput,
        metrics.minimum_tier_slo_attainment,
        metrics.jain_fairness,
        metrics.per_tier_slo_attainment,
        max(placement_report.p_queue_p95, placement_report.decode_queue_p95),
        placement_report.token_hit_rate,
        metrics.waste_fraction,
        placement_report.request_load_max_mean,
        metrics.kvs_normalized_work,
        metrics.prefill_normalized_utilization,
        metrics.decode_normalized_utilization,
        debt,
        metrics.total_gpu_work,
        accounted,
        eviction is not None,
        config.cache_capacity_entries,
        decode_report,
        eviction_report,
        1.0,
        decision_log,
        decision_fingerprint,
        (
            max(measured_census_ages, default=0.0)
            if cell.strategy == "CENSUS_EVICTION"
            else None
        ),
        visibility_delay_work,
        metrics.attempt_count,
        metrics.attempt_count - metrics.offered_logical_requests,
        decode.config.congestion_action.value if decode is not None else None,
        gated_retry_count,
    )


def _retry_pressure_abort_fences(
    workload: Sequence[KernelRequestSpec], *, enabled: bool
) -> dict[str, AbortFence]:
    if not enabled:
        return {}
    return {
        item.logical.logical_request_id: AbortFence(True, -1.0, True, 1)
        for item in workload
    }


def _scaled_workload(
    workload: Sequence[KernelRequestSpec], scale: float
) -> Sequence[KernelRequestSpec]:
    requests = tuple(
        replace(
            item,
            logical=replace(
                item.logical, arrival_work=item.logical.arrival_work / scale
            ),
        )
        for item in workload
    )
    if isinstance(workload, PlacementWorkload):
        return PlacementWorkload(requests, workload.request_truth)
    return requests


def _build_artifacts(
    results: Sequence[CellResult],
    failures: Sequence[Mapping[str, object]],
    trace_metadata: Mapping[str, object],
    binding_cells: set[tuple[str, float]],
    *,
    workload: object,
    executor: Callable[[object, FinalCell], CellResult],
) -> dict[str, bytes]:
    plan = build_cell_plan(binding_cells)
    primary = [result for result in results if result.cell.category == "PRIMARY"]
    sensitivity = [
        result for result in results if result.cell.category == "SENSITIVITY"
    ]
    provenance = git_provenance(ROOT)
    artifacts = {
        "primary.csv": _csv_bytes(primary, _primary_row),
        "constraints.csv": _csv_bytes(primary, _constraint_row),
        "explanation.csv": _csv_bytes(primary, _explanation_row),
        "sensitivities.csv": _csv_bytes(sensitivity, _primary_row),
        "pareto.json": _json_bytes(_pareto(primary, expected_cells=plan)),
        "attribution.json": _json_bytes(_attribution(primary, expected_cells=plan)),
        "falsification/visibility-delay.json": _json_bytes(
            _visibility_delay_audit(
                primary,
                workload=workload,
                executor=executor,
                delta_work=1.0,
            )
        ),
        "crossovers.csv": _csv_mapping_bytes(_crossovers(primary, expected_cells=plan)),
        "failures.csv": _csv_mapping_bytes(failures),
        "gates/g12-3.json": _json_bytes(_g12_3(results, expected_cells=plan)),
        "gates/g12-4.json": _json_bytes(_g12_4(results, binding_cells)),
        "contract.json": _json_bytes(
            {
                "schema_version": "m12-final-contract-v1",
                "truth_basis": TRUTH_BASIS,
                "fixed_horizon": OBSERVATION_END_WORK,
                "arrival_scales": ARRIVAL_SCALES,
                "regimes": [item.regime_id.value for item in SERVICE_REGIMES],
                "primary_metric": "strict_useful_token_goodput",
                "oracle_deployable": False,
            }
        ),
        "config.json": _json_bytes(
            {
                **trace_metadata,
                "workload_parse_count": 1,
                "planned_cell_count": len(build_cell_plan(binding_cells)),
                "completed_cell_count": len(results),
                "failed_cell_count": len(failures),
                "binding_cells": sorted(
                    f"{regime}:{scale:.1f}" for regime, scale in binding_cells
                ),
            }
        ),
        "provenance.json": _json_bytes(
            {
                "git_sha": provenance.sha,
                "git_dirty": provenance.dirty,
                "source_manifest": source_manifest(ROOT, "scripts/run_m12_final.py"),
                "imported_script_sha256": {
                    name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                    for name in (
                        "scripts/run_m12_final.py",
                        "scripts/run_m12_placement.py",
                    )
                },
            }
        ),
    }
    artifacts["MANIFEST.json"] = _manifest_bytes(artifacts)
    return artifacts


def _base_row(result: CellResult) -> dict[str, object]:
    return {
        "cell_id": result.cell.cell_id,
        "regime": result.cell.regime,
        "arrival_scale": result.cell.arrival_scale,
        "strategy": result.cell.strategy,
        "cache_capacity_entries": result.cache_capacity_entries,
        "capacity_binding": result.capacity_binding,
        "hit_ceiling": result.hit_ceiling,
    }


def _primary_row(result: CellResult) -> dict[str, object]:
    return {
        **_base_row(result),
        "offered_requests": result.offered_requests,
        "offered_tokens": result.offered_tokens,
        "attempt_count": result.attempt_count,
        "retry_count": result.retry_count,
        "congestion_action": result.congestion_action,
        "gated_retry_count": result.gated_retry_count,
        "strict_goodput": result.strict_goodput,
        "strict_output_goodput": result.strict_output_goodput,
        "request_goodput": result.request_goodput,
    }


def _constraint_row(result: CellResult) -> dict[str, object]:
    return {
        **_base_row(result),
        "queue_p95_normalized_work": result.queue_p95_normalized_work,
        "minimum_tier": result.minimum_tier,
        "jain": result.jain,
        "per_tier": json.dumps(result.per_tier, sort_keys=True),
        "strict_output_goodput": result.strict_output_goodput,
        "accounting_conserved": math.isclose(result.total_work, result.accounted_work),
    }


def _explanation_row(result: CellResult) -> dict[str, object]:
    return {
        **_base_row(result),
        "token_hit_rate": result.token_hit_rate,
        "waste_fraction": result.waste_fraction,
        "load_skew": result.load_skew,
        "kvs_work": result.kvs_work,
        "p_utilization": result.p_utilization,
        "d_utilization": result.d_utilization,
        "p_to_d_debt": result.p_to_d_debt,
        "decision_fingerprint": result.decision_fingerprint,
        "census_age_work": result.census_age_work,
        "visibility_delay_work": result.visibility_delay_work,
    }


def _pareto(
    results: Sequence[CellResult],
    *,
    expected_cells: Sequence[FinalCell] | None = None,
) -> dict[str, object]:
    groups: list[dict[str, object]] = []
    incomplete = False
    expected_primary = tuple(
        cell
        for cell in (expected_cells or tuple(item.cell for item in results))
        if cell.category == "PRIMARY"
    )
    keys = sorted(
        {
            (
                cell.regime,
                cell.arrival_scale,
                "EVICTION" in cell.strategy,
            )
            for cell in expected_primary
        }
    )
    for regime, scale, bounded in keys:
        expected = {
            cell.strategy
            for cell in expected_primary
            if cell.regime == regime
            and cell.arrival_scale == scale
            and ("EVICTION" in cell.strategy) is bounded
        }
        peers = [
            item
            for item in results
            if item.cell.regime == regime
            and item.cell.arrival_scale == scale
            and ("EVICTION" in item.cell.strategy) is bounded
        ]
        capacities = {item.cache_capacity_entries for item in peers}
        capacity = next(iter(capacities)) if len(capacities) == 1 else None
        present = {item.cell.strategy for item in peers}
        missing = sorted(expected - present)
        if len(capacities) > 1:
            missing.append("CACHE_CAPACITY_MISMATCH")
        incomplete |= bool(missing)
        baseline_name = "EVICTION_LRU" if bounded else "BASELINE"
        baseline = next(
            (item for item in peers if item.cell.strategy == baseline_name), None
        )
        eligible = [
            item
            for item in peers
            if item.cell.strategy in expected and _pareto_eligible(item, baseline)
        ]
        frontier = [
            candidate.cell.cell_id
            for candidate in eligible
            if not any(_dominates(other, candidate) for other in eligible)
        ]
        groups.append(
            {
                "regime": regime,
                "arrival_scale": scale,
                "cache_capacity_entries": capacity,
                "candidate_set": sorted(expected),
                "missing": missing,
                "excluded": sorted(
                    item.cell.cell_id for item in peers if item not in eligible
                ),
                "frontier_cell_ids": sorted(frontier),
                "status": "INCOMPLETE" if missing else "COMPLETE",
            }
        )
    return {
        "overall_verdict": "INCOMPLETE" if incomplete else "COMPLETE",
        "groups": groups,
    }


def _pareto_eligible(candidate: CellResult, baseline: CellResult | None) -> bool:
    return (
        baseline is not None
        and candidate.offered_requests == baseline.offered_requests
        and candidate.offered_tokens == baseline.offered_tokens
        and candidate.minimum_tier >= 0.80
        and candidate.jain >= 0.90
        and candidate.strict_output_goodput >= baseline.strict_output_goodput
        and candidate.load_skew <= baseline.load_skew
        and set(candidate.per_tier) == set(baseline.per_tier)
        and all(
            candidate.per_tier[tier] >= value - 0.02
            for tier, value in baseline.per_tier.items()
        )
        and math.isclose(candidate.total_work, candidate.accounted_work)
    )


def _dominates(left: CellResult, right: CellResult) -> bool:
    weak = (
        left.strict_goodput >= right.strict_goodput
        and left.token_hit_rate >= right.token_hit_rate
        and left.waste_fraction <= right.waste_fraction
    )
    strict = (
        left.strict_goodput > right.strict_goodput
        or left.token_hit_rate > right.token_hit_rate
        or left.waste_fraction < right.waste_fraction
    )
    return weak and strict


def _attribution(
    results: Sequence[CellResult],
    *,
    expected_cells: Sequence[FinalCell] | None = None,
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    incomplete = False
    expected = expected_cells or tuple(item.cell for item in results)
    groups = sorted(
        {
            (cell.regime, cell.arrival_scale)
            for cell in expected
            if cell.category == "PRIMARY"
        }
    )
    for regime, scale in groups:
        lookup = {
            item.cell.strategy: item
            for item in results
            if item.cell.regime == regime and item.cell.arrival_scale == scale
        }
        switches = (
            ("BASELINE", "PRICED_SPILL", "PLACEMENT"),
            ("PRICED_SPILL", "DECODE_CAUSAL", "DECODE_CREDITS"),
            ("EVICTION_LRU", "CENSUS_EVICTION", "CENSUS_EVICTION"),
        )
        for before_name, after_name, switch in switches:
            before, after = lookup.get(before_name), lookup.get(after_name)
            if before is None or after is None:
                eviction_expected = any(
                    cell.regime == regime
                    and cell.arrival_scale == scale
                    and "EVICTION" in cell.strategy
                    for cell in expected
                )
                if switch != "CENSUS_EVICTION" or eviction_expected:
                    incomplete = True
                continue
            first_diff = _first_decision_diff(before.decision_log, after.decision_log)
            delta = after.strict_goodput - before.strict_goodput
            traceable = (
                before.offered_requests == after.offered_requests
                and before.offered_tokens == after.offered_tokens
                and first_diff is not None
            )
            classification = (
                "NO_CHANGE"
                if math.isclose(delta, 0.0)
                else "ATTRIBUTED_FIRST_DIVERGENCE"
                if traceable
                else "INTERACTION"
            )
            records.append(
                {
                    "regime": regime,
                    "arrival_scale": scale,
                    "single_switch": switch,
                    "before": before.cell.cell_id,
                    "after": after.cell.cell_id,
                    "metric": "strict_goodput",
                    "delta": delta,
                    "classification": classification,
                    "first_divergent_decision": first_diff,
                    "before_decision_fingerprint": before.decision_fingerprint,
                    "after_decision_fingerprint": after.decision_fingerprint,
                    "gate_rerun": {
                        "before_eligible": _pareto_eligible(before, before),
                        "after_eligible": _pareto_eligible(after, before),
                    },
                }
            )
    return {
        "overall_verdict": "INCOMPLETE" if incomplete else "COMPLETE",
        "allocation_rule": (
            "first divergent decision only; untraceable/non-additive is INTERACTION"
        ),
        "records": records,
    }


def _visibility_delay_audit(
    results: Sequence[CellResult],
    *,
    workload: object,
    executor: Callable[[object, FinalCell], CellResult],
    delta_work: float,
) -> dict[str, object]:
    delayed = with_visibility_delay(executor, delta_work)
    rows = []
    for baseline in results:
        candidate = delayed(workload, baseline.cell)
        rows.append(
            {
                "cell_id": baseline.cell.cell_id,
                "delta_work": delta_work,
                "decision_fingerprint_before": baseline.decision_fingerprint,
                "decision_fingerprint_after": candidate.decision_fingerprint,
                "decision_sequence_unchanged": (
                    baseline.decision_log == candidate.decision_log
                ),
                "census_not_newer": (
                    None
                    if baseline.census_age_work is None
                    else candidate.census_age_work is not None
                    and candidate.census_age_work >= baseline.census_age_work
                ),
                "census_age_before_work": baseline.census_age_work,
                "census_age_after_work": candidate.census_age_work,
                "offered_requests_conserved": (
                    candidate.offered_requests == baseline.offered_requests
                ),
                "offered_tokens_conserved": (
                    candidate.offered_tokens == baseline.offered_tokens
                ),
                "work_conserved": math.isclose(
                    candidate.total_work, candidate.accounted_work
                ),
            }
        )
    return {
        "independent_rerun": True,
        "truth_visibility_only": True,
        "decision_points_moved": False,
        "rows": rows,
    }


def _first_decision_diff(
    before: Sequence[str], after: Sequence[str]
) -> dict[str, object] | None:
    before_records = _keyed_decision_records(before)
    after_records = _keyed_decision_records(after)
    if before_records is not None and after_records is not None:
        divergent = [
            key
            for key in before_records.keys() | after_records.keys()
            if before_records.get(key) != after_records.get(key)
        ]
        if not divergent:
            return None

        def causal_order(key: tuple[str, int, str]) -> tuple[float, int, str, int, str]:
            record = after_records.get(key) or before_records[key]
            decision_time = record["decision_time_work"]
            sequence = record["sequence"]
            if not isinstance(decision_time, (int, float)) or not isinstance(
                sequence, int
            ):
                raise TypeError("validated decision ledger record changed type")
            return (
                float(decision_time),
                sequence,
                key[0],
                key[1],
                key[2],
            )

        key = min(divergent, key=causal_order)
        left, right = before_records.get(key), after_records.get(key)
        change_type = "MODIFIED"
        if left is None:
            change_type = "ADDED"
        elif right is None:
            change_type = "REMOVED"
        return {
            "index": causal_order(key)[1],
            "key": list(key),
            "decision_layer": key[2],
            "change_type": change_type,
            "before": left,
            "after": right,
        }

    for index in range(max(len(before), len(after))):
        legacy_left = before[index] if index < len(before) else None
        legacy_right = after[index] if index < len(after) else None
        if legacy_left != legacy_right:
            layer = None
            for value in (legacy_left, legacy_right):
                if value is None:
                    continue
                layer = _decision_layer(value)
                if layer is not None:
                    break
            return {
                "index": index,
                "decision_layer": layer,
                "before": legacy_left,
                "after": legacy_right,
            }
    return None


def _keyed_decision_records(
    values: Sequence[str],
) -> dict[tuple[str, int, str], dict[str, object]] | None:
    records: dict[tuple[str, int, str], dict[str, object]] = {}
    for value in values:
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return None
        if not isinstance(decoded, dict):
            return None
        logical_id = decoded.get("logical_id")
        attempt_index = decoded.get("attempt_index")
        kind = decoded.get("decision_kind")
        decision_time = decoded.get("decision_time_work")
        sequence = decoded.get("sequence")
        if (
            not isinstance(logical_id, str)
            or not isinstance(attempt_index, int)
            or not isinstance(kind, str)
            or not isinstance(decision_time, (int, float))
            or not isinstance(sequence, int)
        ):
            return None
        key = (logical_id, attempt_index, kind)
        if key in records:
            return None
        records[key] = decoded
    return records


def _decision_layer(value: str) -> object | None:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return None
    return decoded.get("decision_kind") if isinstance(decoded, dict) else None


def _crossovers(
    results: Sequence[CellResult],
    *,
    expected_cells: Sequence[FinalCell] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    expected = expected_cells or tuple(item.cell for item in results)
    regimes = sorted({cell.regime for cell in expected if cell.category == "PRIMARY"})
    for regime in regimes:
        previous: str | None = None
        for scale in ARRIVAL_SCALES:
            cells = [
                item
                for item in results
                if item.cell.regime == regime
                and item.cell.arrival_scale == scale
                and item.cell.strategy in PRIMARY_STRATEGIES
            ]
            expected_strategies = {
                cell.strategy
                for cell in expected
                if cell.category == "PRIMARY"
                and cell.regime == regime
                and cell.arrival_scale == scale
                and cell.strategy in PRIMARY_STRATEGIES
            }
            missing = sorted(
                expected_strategies - {item.cell.strategy for item in cells}
            )
            if missing:
                rows.append(
                    {
                        "regime": regime,
                        "arrival_scale": scale,
                        "status": "INCOMPLETE",
                        "missing": ";".join(missing),
                    }
                )
                previous = None
                continue
            baseline = next(
                (item for item in cells if item.cell.strategy == "BASELINE"), None
            )
            cells = [item for item in cells if _pareto_eligible(item, baseline)]
            if not cells:
                previous = None
                continue
            winner = max(
                cells, key=lambda item: (item.strict_goodput, item.cell.strategy)
            )
            if previous is not None and winner.cell.strategy != previous:
                rows.append(
                    {
                        "regime": regime,
                        "arrival_scale": scale,
                        "from_strategy": previous,
                        "to_strategy": winner.cell.strategy,
                        "status": "COMPLETE",
                    }
                )
            previous = winner.cell.strategy
    return rows


def _g12_3(
    results: Sequence[CellResult],
    *,
    expected_cells: Sequence[FinalCell] | None = None,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    incomplete = False
    expected = expected_cells or tuple(item.cell for item in results)
    regimes = sorted(
        {
            cell.regime
            for cell in expected
            if cell.arrival_scale == 1.5
            and cell.strategy
            in {
                "DECODE_NO_GATE",
                "DECODE_CAUSAL",
                "DECODE_ORACLE",
                "DECODE_ORACLE_NOISED",
            }
        }
    )
    for regime in regimes:
        lookup = {
            item.cell.strategy: item
            for item in results
            if item.cell.regime == regime and item.cell.arrival_scale == 1.5
        }
        baseline = lookup.get("DECODE_NO_GATE")
        if baseline is None:
            incomplete = True
            continue
        for strategy in (
            "DECODE_CAUSAL",
            "DECODE_ORACLE",
            "DECODE_ORACLE_NOISED",
        ):
            candidate = lookup.get(strategy)
            if (
                candidate is None
                or baseline.decode_report is None
                or candidate.decode_report is None
            ):
                incomplete = True
                rows.append(
                    {
                        "regime": regime,
                        "arrival_scale": 1.5,
                        "strategy": strategy,
                        "status": "INCOMPLETE",
                        "passed": False,
                        "deployable": False,
                    }
                )
                continue
            verdict = evaluate_g12_3(
                no_gate=baseline.decode_report,
                candidate=candidate.decode_report,
                arrival_scale=1.5,
            )
            oracle = strategy in ("DECODE_ORACLE", "DECODE_ORACLE_NOISED")
            rows.append(
                {
                    "regime": regime,
                    "arrival_scale": candidate.cell.arrival_scale,
                    "strategy": strategy,
                    "status": "SENSITIVITY_ONLY" if oracle else verdict.conclusion,
                    "passed": verdict.passed and not oracle,
                    "sensitivity_passed": verdict.passed,
                    "deployable": verdict.deployable_conclusion and not oracle,
                    "canonical_verdict": asdict(verdict),
                    "attempt_count": candidate.attempt_count,
                    "retry_count": candidate.retry_count,
                    "congestion_action": candidate.congestion_action,
                    "gated_retry_count": candidate.gated_retry_count,
                }
            )
    causal = [row for row in rows if row["strategy"] == "DECODE_CAUSAL"]
    overall = (
        "INCOMPLETE"
        if incomplete
        else "PASS"
        if causal and all(bool(row["passed"]) for row in causal)
        else "NARROW_OVERLOAD_ONLY"
    )
    retry_pressure_covered = any(
        row["regime"] == "MIXED"
        and row["arrival_scale"] == 1.5
        and row["strategy"] == "DECODE_CAUSAL"
        and isinstance(gated_retry_count := row.get("gated_retry_count"), int)
        and gated_retry_count > 0
        and row.get("congestion_action") in {"GATED_PD", "GATED_DP"}
        for row in rows
    )
    if not retry_pressure_covered:
        overall = "INCOMPLETE_RETRY_PRESSURE"
    return {
        "overall_verdict": overall,
        "retry_pressure_covered": retry_pressure_covered,
        "cells": rows,
    }


def _g12_4(
    results: Sequence[CellResult], binding: set[tuple[str, float]]
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    incomplete = False
    for regime, scale in sorted(binding):
        lookup = {
            item.cell.strategy: item
            for item in results
            if item.cell.regime == regime and item.cell.arrival_scale == scale
        }
        lru, census = lookup.get("EVICTION_LRU"), lookup.get("CENSUS_EVICTION")
        if (
            lru is None
            or census is None
            or lru.eviction_report is None
            or census.eviction_report is None
        ):
            incomplete = True
            rows.append(
                {
                    "regime": regime,
                    "arrival_scale": scale,
                    "status": "INCOMPLETE",
                    "passed": False,
                }
            )
            continue
        verdict = evaluate_g12_4(
            lru.eviction_report,
            census.eviction_report,
            hit_ceiling=lru.hit_ceiling,
        )
        rows.append(
            {
                "regime": regime,
                "arrival_scale": scale,
                "status": "PASS" if verdict.passed else "KILL_OR_NARROW",
                "passed": verdict.passed,
                "canonical_verdict": asdict(verdict),
            }
        )
    overall = (
        "INCOMPLETE"
        if incomplete
        else "PASS"
        if rows and all(bool(row["passed"]) for row in rows)
        else "KILL_OR_NARROW"
    )
    return {"overall_verdict": overall, "cells": rows}


def _relative_gain(baseline: float, candidate: float) -> float:
    return (candidate - baseline) / baseline if baseline else 0.0


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _csv_bytes(
    values: Sequence[CellResult], mapper: Callable[[CellResult], Mapping[str, object]]
) -> bytes:
    return _csv_mapping_bytes([mapper(value) for value in values])


def _csv_mapping_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    fields = sorted({key for row in rows for key in row})
    buffer = io.StringIO(newline="")
    if fields:
        writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return buffer.getvalue().encode()


def _manifest_bytes(artifacts: Mapping[str, bytes]) -> bytes:
    return _json_bytes(
        {
            "schema_version": MANIFEST_SCHEMA,
            "algorithm": "sha256",
            "files": {
                name: hashlib.sha256(payload).hexdigest()
                for name, payload in sorted(artifacts.items())
            },
        }
    )


def _write_atomic(output_dir: Path, artifacts: Mapping[str, bytes]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=output_dir.parent, prefix=".m12-final-"))
    try:
        ordered = [name for name in sorted(artifacts) if name != "MANIFEST.json"]
        ordered.append("MANIFEST.json")
        for name in ordered:
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(artifacts[name])
        for name in ordered:
            destination = output_dir / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / name, destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main() -> int:
    if placement_run_active():
        print(
            "run_m12_placement.py is active; final heavy run deferred", file=sys.stderr
        )
        return 2
    run_artifacts(
        ROOT / "mooncake_trace.jsonl",
        ROOT / "results" / "m12-final",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
