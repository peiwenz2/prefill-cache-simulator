"""Deterministic minimum-prefill-capacity evaluation for the M12 kernel.

All quantities in this module are normalized simulation units.  The zero-transfer
price topology is a control and is mechanically excluded from deployable verdicts.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType
from typing import overload

from .m12_kernel import CausalKernel, FrozenKernelCostModel, KernelConfig
from .m12_metrics import SyntheticServiceRegime
from .m12_placement import (
    M12PlacementPolicy,
    PlacementMode,
    PlacementRunReport,
    PlacementWorkload,
)


class SizingTopology(StrEnum):
    LOCAL_ONLY = "LOCAL_ONLY"
    SHARED_KVS = "SHARED_KVS"
    ZERO_TRANSFER_PRICE_CONTROL = "ZERO_TRANSFER_PRICE_CONTROL"

    @property
    def deployable(self) -> bool:
        return self is not SizingTopology.ZERO_TRANSFER_PRICE_CONTROL


@dataclass(frozen=True, slots=True)
class SizingGates:
    minimum_completion_ratio: float
    minimum_tier_slo_attainment: float
    minimum_jain_fairness: float
    maximum_p_queue_p95_work: float
    maximum_kvs_bytes_per_work: float

    def __post_init__(self) -> None:
        ratios = (
            self.minimum_completion_ratio,
            self.minimum_tier_slo_attainment,
            self.minimum_jain_fairness,
        )
        if any(not _finite(value) or not 0 <= value <= 1 for value in ratios):
            raise ValueError("ratio gates must be finite and within [0, 1]")
        ceilings = (
            self.maximum_p_queue_p95_work,
            self.maximum_kvs_bytes_per_work,
        )
        if any(not _finite(value) or value < 0 for value in ceilings):
            raise ValueError("resource ceilings must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class GateObservation:
    completion_ratio: float
    minimum_tier_slo_attainment: float
    jain_fairness: float
    p_queue_p95_work: float
    kvs_bytes_per_work: float

    def __post_init__(self) -> None:
        values = (
            self.completion_ratio,
            self.minimum_tier_slo_attainment,
            self.jain_fairness,
            self.p_queue_p95_work,
            self.kvs_bytes_per_work,
        )
        if any(not _finite(value) or value < 0 for value in values):
            raise ValueError("gate observations must be finite and non-negative")
        if any(value > 1 for value in values[:3]):
            raise ValueError("observed ratios must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class SizingCell:
    p_count: int
    topology: SizingTopology
    observation: GateObservation
    failed_gates: tuple[str, ...]

    def __post_init__(self) -> None:
        if not _plain_positive_int(self.p_count):
            raise ValueError("P count must be a plain positive integer")
        if not isinstance(self.topology, SizingTopology):
            raise ValueError("topology must be a SizingTopology")

    @property
    def feasible(self) -> bool:
        return not self.failed_gates


@dataclass(frozen=True, slots=True)
class InfeasibilityCertificate:
    p_count: int
    topology: SizingTopology
    failed_gates: tuple[str, ...]
    observed: Mapping[str, float]
    required: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "observed", MappingProxyType(dict(self.observed)))
        object.__setattr__(self, "required", MappingProxyType(dict(self.required)))


@dataclass(frozen=True, slots=True)
class SizingVerdict:
    minimum_feasible_p: int | None
    selected_topology: SizingTopology | None
    predecessor_certificate: InfeasibilityCertificate | None
    predecessor_certificates: tuple[InfeasibilityCertificate, ...]
    non_monotonic_p_counts: tuple[int, ...]
    grid_exhausted: bool


@dataclass(frozen=True, slots=True)
class SizingRunRecord:
    cell: SizingCell
    strict_useful_token_goodput: float
    strict_useful_output_token_goodput: float
    request_goodput: float
    token_hit_rate: float
    local_hit_tokens: int
    remote_hit_tokens: int
    recompute_tokens: int
    p_normalized_utilization: float
    normalized_kvs_work: float
    decision_fingerprint: str


def run_sizing_cell(
    workload: PlacementWorkload,
    *,
    p_count: int,
    topology: SizingTopology,
    gates: SizingGates,
    regime: SyntheticServiceRegime,
    observation_end_work: float,
    tier_slo_work: Mapping[str, float],
    cache_capacity_entries_per_p: int,
    decode_node_count: int = 8,
    cost_scale: float = 1.0,
    kvs_bytes_per_token: int = 65_536,
) -> SizingRunRecord:
    """Execute one sizing point through the causal kernel.

    Cache capacity is per P node.  The initial full-trace contract intentionally
    sets it above the unique-key count, so this measures resource sizing without
    conflating an unimplemented sizing-specific eviction policy.
    """
    if not _plain_positive_int(p_count):
        raise ValueError("P count must be a plain positive integer")
    if not _plain_positive_int(decode_node_count):
        raise ValueError("decode node count must be a plain positive integer")
    if not _plain_positive_int(cache_capacity_entries_per_p):
        raise ValueError("cache capacity must be a plain positive integer")
    if not _plain_positive_int(kvs_bytes_per_token):
        raise ValueError("KVS bytes per token must be a plain positive integer")
    if not _finite(cost_scale) or cost_scale <= 0:
        raise ValueError("cost scale must be finite and positive")
    p_nodes = tuple(f"p{index}" for index in range(p_count))
    d_nodes = tuple(f"d{index}" for index in range(decode_node_count))
    resized = _resize_eligibility(workload, p_nodes)
    cost = _sizing_cost(regime, topology, cost_scale, kvs_bytes_per_token)
    config = KernelConfig(
        0,
        observation_end_work,
        p_nodes,
        d_nodes,
        tier_slo_work,
        cache_capacity_entries_per_p,
        cost,
    )
    policy = _sizing_policy(topology, cost, resized)
    result = CausalKernel(config).run(resized, policy)
    report = policy.summarize(result)
    completed = {
        attempt.logical_request_id for attempt in result.attempts if attempt.completed
    }
    completion_ratio = (
        len(completed) / result.metrics.offered_logical_requests
        if result.metrics.offered_logical_requests
        else 0.0
    )
    observation = GateObservation(
        completion_ratio,
        result.metrics.minimum_tier_slo_attainment,
        result.metrics.jain_fairness,
        report.p_queue_p95,
        result.metrics.kvs_bytes_per_horizon,
    )
    cell = evaluate_gates(
        observation,
        gates,
        p_count=p_count,
        topology=topology,
    )
    assert isinstance(cell, SizingCell)
    _assert_conservation(report)
    return SizingRunRecord(
        cell,
        result.metrics.strict_useful_token_goodput,
        result.metrics.strict_useful_output_token_goodput,
        result.metrics.request_goodput,
        report.token_hit_rate,
        report.local_hit_tokens,
        report.remote_hit_tokens,
        report.uncached_tokens,
        result.metrics.prefill_normalized_utilization,
        result.metrics.kvs_normalized_work,
        _decision_fingerprint(policy),
    )


@overload
def evaluate_gates(
    observation: GateObservation,
    gates: SizingGates,
) -> tuple[str, ...]: ...


@overload
def evaluate_gates(
    observation: GateObservation,
    gates: SizingGates,
    *,
    p_count: int,
    topology: SizingTopology,
) -> SizingCell: ...


def evaluate_gates(
    observation: GateObservation,
    gates: SizingGates,
    *,
    p_count: int | None = None,
    topology: SizingTopology | None = None,
) -> tuple[str, ...] | SizingCell:
    """Evaluate every frozen gate; equality passes for all thresholds."""
    failed: list[str] = []
    if observation.completion_ratio < gates.minimum_completion_ratio:
        failed.append("COMPLETION_FLOOR")
    if (
        observation.minimum_tier_slo_attainment
        < gates.minimum_tier_slo_attainment
    ):
        failed.append("TIER_SLO_FLOOR")
    if observation.jain_fairness < gates.minimum_jain_fairness:
        failed.append("FAIRNESS_FLOOR")
    if observation.p_queue_p95_work > gates.maximum_p_queue_p95_work:
        failed.append("P_QUEUE_P95_CEILING")
    if (
        observation.kvs_bytes_per_work
        > gates.maximum_kvs_bytes_per_work
    ):
        failed.append("KVS_UTILIZATION_CEILING")
    result = tuple(failed)
    if p_count is None and topology is None:
        return result
    if p_count is None or topology is None:
        raise ValueError("P count and topology must be supplied together")
    return SizingCell(p_count, topology, observation, result)


def select_minimum(
    cells: Sequence[SizingCell],
    gates: SizingGates,
    *,
    include_control: bool = False,
    required_topologies: Sequence[SizingTopology] | None = None,
) -> SizingVerdict:
    """Select a deployable minimum and retain its literal P-1 evidence."""
    ordered = tuple(sorted(cells, key=lambda cell: (cell.p_count, cell.topology.value)))
    required = tuple(
        required_topologies
        if required_topologies is not None
        else (topology for topology in SizingTopology if topology.deployable)
    )
    if not required or len(set(required)) != len(required):
        raise ValueError("required topologies must be non-empty and unique")
    if any(not isinstance(topology, SizingTopology) for topology in required):
        raise ValueError("required topologies must use SizingTopology values")
    candidates = tuple(
        cell
        for cell in ordered
        if cell.topology in required
        and (include_control or cell.topology.deployable)
    )
    present = {cell.topology for cell in candidates}
    if present != set(required):
        return SizingVerdict(None, None, None, (), (), True)
    feasible = tuple(cell for cell in candidates if cell.feasible)
    if not feasible:
        return SizingVerdict(None, None, None, (), (), True)
    winner = feasible[0]
    considered_topologies = tuple(sorted(required, key=lambda value: value.value))
    topology_cells = tuple(
        cell for cell in candidates if cell.topology is winner.topology
    )
    seen_pass = False
    reversals: list[int] = []
    for cell in topology_cells:
        if cell.feasible:
            seen_pass = True
        elif seen_pass:
            reversals.append(cell.p_count)
    predecessor_cells = tuple(
        cell
        for topology in considered_topologies
        for cell in candidates
        if cell.topology is topology and cell.p_count == winner.p_count - 1
    )
    if winner.p_count > 1 and (
        len(predecessor_cells) != len(considered_topologies)
        or any(cell.feasible for cell in predecessor_cells)
    ):
        return SizingVerdict(None, None, None, (), tuple(reversals), True)
    certificates = tuple(_certificate(cell, gates) for cell in predecessor_cells)
    certificate = next(
        (
            value
            for value in certificates
            if value.topology is winner.topology
        ),
        None,
    )
    return SizingVerdict(
        winner.p_count,
        winner.topology,
        certificate,
        certificates,
        tuple(reversals),
        False,
    )


def _certificate(
    cell: SizingCell, gates: SizingGates
) -> InfeasibilityCertificate:
    observation = cell.observation
    observed = {
        "COMPLETION_FLOOR": observation.completion_ratio,
        "TIER_SLO_FLOOR": observation.minimum_tier_slo_attainment,
        "FAIRNESS_FLOOR": observation.jain_fairness,
        "P_QUEUE_P95_CEILING": observation.p_queue_p95_work,
        "KVS_UTILIZATION_CEILING": observation.kvs_bytes_per_work,
    }
    required = {
        "COMPLETION_FLOOR": gates.minimum_completion_ratio,
        "TIER_SLO_FLOOR": gates.minimum_tier_slo_attainment,
        "FAIRNESS_FLOOR": gates.minimum_jain_fairness,
        "P_QUEUE_P95_CEILING": gates.maximum_p_queue_p95_work,
        "KVS_UTILIZATION_CEILING": (
            gates.maximum_kvs_bytes_per_work
        ),
    }
    return InfeasibilityCertificate(
        cell.p_count,
        cell.topology,
        cell.failed_gates,
        {name: observed[name] for name in cell.failed_gates},
        {name: required[name] for name in cell.failed_gates},
    )


def _finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _plain_positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _resize_eligibility(
    workload: PlacementWorkload, p_nodes: tuple[str, ...]
) -> PlacementWorkload:
    truth = {
        identity: replace(value, eligible_node_ids=frozenset(p_nodes))
        for identity, value in workload.request_truth.items()
    }
    return PlacementWorkload(workload.requests, truth)


def _sizing_cost(
    regime: SyntheticServiceRegime,
    topology: SizingTopology,
    cost_scale: float,
    kvs_bytes_per_token: int,
) -> FrozenKernelCostModel:
    kvs_work = (
        0.0
        if topology is SizingTopology.ZERO_TRANSFER_PRICE_CONTROL
        else regime.kvs_byte_work * kvs_bytes_per_token * cost_scale
    )
    return FrozenKernelCostModel(
        regime.prefill_token_work * cost_scale,
        kvs_work,
        kvs_bytes_per_token,
        regime.decode_token_work * cost_scale,
    )


def _sizing_policy(
    topology: SizingTopology,
    cost: FrozenKernelCostModel,
    workload: PlacementWorkload,
) -> M12PlacementPolicy:
    if topology is SizingTopology.LOCAL_ONLY:
        return M12PlacementPolicy(
            # Hold routing policy constant across topology controls.  Only
            # remote-cache visibility and its transfer price may differ.
            PlacementMode.HYBRID,
            cost,
            kvs_enabled=False,
            request_truth=workload.request_truth,
        )
    return M12PlacementPolicy(
        PlacementMode.HYBRID,
        cost,
        kvs_enabled=True,
        # The zero-transfer-price control changes only transfer price. Keeping the same
        # routing threshold as SHARED_KVS makes it a price-only control.
        mooncake_balancing_threshold=2.0,
        request_truth=workload.request_truth,
    )


def _assert_conservation(report: PlacementRunReport) -> None:
    total = report.local_hit_tokens + report.remote_hit_tokens + report.uncached_tokens
    if total < 0:
        raise ValueError("cache accounting produced negative token work")
    metrics = report.kernel_metrics
    if not math.isclose(
        metrics.total_gpu_work,
        metrics.prefill_gpu_work + metrics.decode_gpu_work,
        rel_tol=1e-12,
        abs_tol=1e-9,
    ):
        raise ValueError("GPU work conservation failed")


def _decision_fingerprint(policy: M12PlacementPolicy) -> str:
    digest = hashlib.sha256()
    for decision in policy.decisions:
        digest.update(decision.logical_request_id.encode())
        digest.update(b"\0")
        digest.update(decision.node_id.encode())
        digest.update(b"\0")
        digest.update(format(decision.queue_work, ".17g").encode())
        digest.update(b"\n")
    return digest.hexdigest()
