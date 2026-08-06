"""M12.1 metric contract and synthetic service regimes."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum


class ServiceRegimeId(StrEnum):
    COMPUTE_BOUND = "COMPUTE_BOUND"
    MEMORY_BOUND = "MEMORY_BOUND"
    MIXED = "MIXED"


@dataclass(frozen=True, slots=True)
class SyntheticServiceRegime:
    regime_id: ServiceRegimeId
    prefill_token_work: float
    decode_token_work: float
    prefill_batch_max_saving: float
    prefill_batch_saturation_tokens: int
    decode_batch_max_saving: float
    decode_batch_saturation_sequences: int
    kvs_byte_work: float

    def __post_init__(self) -> None:
        positive = (
            self.prefill_token_work,
            self.decode_token_work,
            self.prefill_batch_saturation_tokens,
            self.decode_batch_saturation_sequences,
            self.kvs_byte_work,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("service-regime costs and saturations must be positive")
        if not 0 <= self.prefill_batch_max_saving < 1:
            raise ValueError("prefill batch saving must be in [0, 1)")
        if not 0 <= self.decode_batch_max_saving < 1:
            raise ValueError("decode batch saving must be in [0, 1)")

    def prefill_work(self, tokens: int, *, batch_tokens: int) -> float:
        if tokens < 0 or batch_tokens < tokens:
            raise ValueError("prefill batch tokens must cover non-negative work")
        fill = min(1.0, batch_tokens / self.prefill_batch_saturation_tokens)
        return (
            tokens
            * self.prefill_token_work
            * (1 - self.prefill_batch_max_saving * fill)
        )

    def decode_work(self, tokens: int, *, concurrent_sequences: int) -> float:
        if tokens < 0 or concurrent_sequences <= 0:
            raise ValueError("decode work inputs are invalid")
        fill = min(
            1.0,
            (concurrent_sequences - 1)
            / max(1, self.decode_batch_saturation_sequences - 1),
        )
        return (
            tokens * self.decode_token_work * (1 - self.decode_batch_max_saving * fill)
        )

    def kvs_work(self, transferred_bytes: int) -> float:
        if transferred_bytes < 0:
            raise ValueError("transferred bytes must be non-negative")
        return transferred_bytes * self.kvs_byte_work


SERVICE_REGIMES: tuple[SyntheticServiceRegime, ...] = (
    SyntheticServiceRegime(
        ServiceRegimeId.COMPUTE_BOUND,
        prefill_token_work=0.0800,
        decode_token_work=1.0000,
        prefill_batch_max_saving=0.45,
        prefill_batch_saturation_tokens=65_536,
        decode_batch_max_saving=0.20,
        decode_batch_saturation_sequences=64,
        kvs_byte_work=1.0e-7,
    ),
    SyntheticServiceRegime(
        ServiceRegimeId.MEMORY_BOUND,
        prefill_token_work=0.0400,
        decode_token_work=1.4000,
        prefill_batch_max_saving=0.15,
        prefill_batch_saturation_tokens=32_768,
        decode_batch_max_saving=0.35,
        decode_batch_saturation_sequences=32,
        kvs_byte_work=2.5e-7,
    ),
    SyntheticServiceRegime(
        ServiceRegimeId.MIXED,
        prefill_token_work=0.0568,
        decode_token_work=1.0000,
        prefill_batch_max_saving=0.30,
        prefill_batch_saturation_tokens=49_152,
        decode_batch_max_saving=0.25,
        decode_batch_saturation_sequences=48,
        kvs_byte_work=1.5e-7,
    ),
)

MINIMUM_TIER_SLO_ATTAINMENT = 0.80
MINIMUM_JAIN_FAIRNESS = 0.90
MAX_TIER_REGRESSION_ABSOLUTE = 0.02


@dataclass(frozen=True, slots=True)
class AttemptOutcome:
    logical_request_id: str
    attempt_index: int
    tenant_id: str
    tier: str
    arrival_work: float
    input_tokens: int
    requested_output_tokens: int
    emitted_output_tokens: int
    completed: bool
    strict_slo_met: bool
    finish_work: float | None
    prefill_gpu_work: float
    decode_gpu_work: float
    wasted_prefill_work: float
    wasted_decode_work: float
    kvs_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.logical_request_id or not self.tenant_id or not self.tier:
            raise ValueError("attempt identity, tenant and tier must be non-empty")
        if self.attempt_index < 0 or self.arrival_work < 0:
            raise ValueError("attempt index and arrival must be non-negative")
        token_values = (
            self.input_tokens,
            self.requested_output_tokens,
            self.emitted_output_tokens,
            self.kvs_bytes,
        )
        if any(value < 0 for value in token_values):
            raise ValueError("token and byte counts must be non-negative")
        if self.emitted_output_tokens > self.requested_output_tokens:
            raise ValueError("emitted output exceeds requested output")
        work_values = (
            self.prefill_gpu_work,
            self.decode_gpu_work,
            self.wasted_prefill_work,
            self.wasted_decode_work,
        )
        if any(value < 0 or not math.isfinite(value) for value in work_values):
            raise ValueError("GPU work must be finite and non-negative")
        if self.wasted_prefill_work > self.prefill_gpu_work:
            raise ValueError("wasted prefill work exceeds issued prefill work")
        if self.wasted_decode_work > self.decode_gpu_work:
            raise ValueError("wasted decode work exceeds issued decode work")
        if self.completed != (self.finish_work is not None):
            raise ValueError("completed and finish_work must agree")
        if self.strict_slo_met and not self.completed:
            raise ValueError("an incomplete attempt cannot meet strict SLO")
        if self.finish_work is not None and self.finish_work < self.arrival_work:
            raise ValueError("finish precedes arrival")


@dataclass(frozen=True, slots=True)
class M12MetricReport:
    schema_version: str
    truth_basis: str
    observation_start_work: float
    observation_end_work: float
    offered_logical_requests: int
    offered_input_tokens: int
    offered_output_tokens: int
    attempt_count: int
    retry_amplification: float
    strict_completed_requests: int
    strict_useful_tokens: int
    request_goodput: float
    strict_useful_token_goodput: float
    strict_useful_tokens_per_gpu_work: float
    raw_output_token_throughput: float
    prefill_gpu_work: float
    decode_gpu_work: float
    total_gpu_work: float
    wasted_gpu_work: float
    waste_fraction: float
    slo_missed_gpu_work: float
    prefill_normalized_utilization: float
    decode_normalized_utilization: float
    cluster_normalized_utilization: float
    kvs_bytes_per_work: float
    jain_fairness: float
    per_tier_slo_attainment: dict[str, float]
    minimum_tier_slo_attainment: float
    fairness_floor_pass: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def aggregate_m12_metrics(
    attempts: Sequence[AttemptOutcome],
    *,
    observation_start_work: float,
    observation_end_work: float,
    prefill_nodes: int,
    decode_nodes: int,
) -> M12MetricReport:
    """Aggregate attempt work while crediting each logical request at most once."""
    if observation_start_work < 0 or observation_end_work <= observation_start_work:
        raise ValueError("observation window must be positive")
    if prefill_nodes <= 0 or decode_nodes <= 0:
        raise ValueError("resource pool sizes must be positive")
    identities: set[tuple[str, int]] = set()
    logical_shape: dict[str, tuple[str, str, float, int, int]] = {}
    by_logical: defaultdict[str, list[AttemptOutcome]] = defaultdict(list)
    for attempt in attempts:
        if not observation_start_work <= attempt.arrival_work < observation_end_work:
            raise ValueError("attempt arrival is outside the observation window")
        if (
            attempt.finish_work is not None
            and attempt.finish_work > observation_end_work
        ):
            raise ValueError("attempt finish is outside the observation window")
        identity = (attempt.logical_request_id, attempt.attempt_index)
        if identity in identities:
            raise ValueError(f"duplicate attempt identity: {identity}")
        identities.add(identity)
        shape = (
            attempt.tenant_id,
            attempt.tier,
            attempt.arrival_work,
            attempt.input_tokens,
            attempt.requested_output_tokens,
        )
        previous = logical_shape.setdefault(attempt.logical_request_id, shape)
        if previous != shape:
            raise ValueError(
                f"logical request shape changed across attempts: "
                f"{attempt.logical_request_id}"
            )
        by_logical[attempt.logical_request_id].append(attempt)

    horizon = observation_end_work - observation_start_work
    strict_winners: dict[str, AttemptOutcome] = {}
    for logical_id, values in by_logical.items():
        eligible = [value for value in values if value.strict_slo_met]
        if eligible:
            strict_winners[logical_id] = min(
                eligible,
                key=lambda value: (
                    value.finish_work if value.finish_work is not None else math.inf,
                    value.attempt_index,
                ),
            )

    offered_input = sum(shape[3] for shape in logical_shape.values())
    offered_output = sum(shape[4] for shape in logical_shape.values())
    strict_useful = sum(
        winner.input_tokens + winner.emitted_output_tokens
        for winner in strict_winners.values()
    )
    prefill_work = sum(value.prefill_gpu_work for value in attempts)
    decode_work = sum(value.decode_gpu_work for value in attempts)
    total_work = prefill_work + decode_work
    waste = sum(
        value.wasted_prefill_work + value.wasted_decode_work for value in attempts
    )
    slo_missed_work = sum(
        value.prefill_gpu_work
        + value.decode_gpu_work
        - value.wasted_prefill_work
        - value.wasted_decode_work
        for value in attempts
        if value.completed and not value.strict_slo_met
    )
    raw_output = sum(value.emitted_output_tokens for value in attempts)

    tenant_demand: defaultdict[str, int] = defaultdict(int)
    tenant_service: defaultdict[str, int] = defaultdict(int)
    tier_demand: defaultdict[str, int] = defaultdict(int)
    tier_service: defaultdict[str, int] = defaultdict(int)
    for logical_id, shape in logical_shape.items():
        tenant, tier, _, input_tokens, output_tokens = shape
        demand = input_tokens + output_tokens
        tenant_demand[tenant] += demand
        tier_demand[tier] += 1
        winner = strict_winners.get(logical_id)
        if winner is not None:
            tenant_service[tenant] += winner.input_tokens + winner.emitted_output_tokens
            tier_service[tier] += 1
    service_ratios = tuple(
        tenant_service[tenant] / demand
        for tenant, demand in sorted(tenant_demand.items())
        if demand > 0
    )
    jain = _jain(service_ratios)
    tier_attainment = {
        tier: tier_service[tier] / demand
        for tier, demand in sorted(tier_demand.items())
        if demand > 0
    }
    minimum_tier = min(tier_attainment.values(), default=0.0)
    offered_count = len(logical_shape)
    p_capacity = prefill_nodes * horizon
    d_capacity = decode_nodes * horizon
    if prefill_work > p_capacity + 1e-9 or decode_work > d_capacity + 1e-9:
        raise ValueError("issued GPU work exceeds normalized pool capacity")
    return M12MetricReport(
        schema_version="m12-metric-contract-v1",
        truth_basis="SYNTHETIC_SERVICE_REGIME",
        observation_start_work=observation_start_work,
        observation_end_work=observation_end_work,
        offered_logical_requests=offered_count,
        offered_input_tokens=offered_input,
        offered_output_tokens=offered_output,
        attempt_count=len(attempts),
        retry_amplification=len(attempts) / offered_count if offered_count else 0.0,
        strict_completed_requests=len(strict_winners),
        strict_useful_tokens=strict_useful,
        request_goodput=len(strict_winners) / horizon,
        strict_useful_token_goodput=strict_useful / horizon,
        strict_useful_tokens_per_gpu_work=(
            strict_useful / total_work if total_work else 0.0
        ),
        raw_output_token_throughput=raw_output / horizon,
        prefill_gpu_work=prefill_work,
        decode_gpu_work=decode_work,
        total_gpu_work=total_work,
        wasted_gpu_work=waste,
        waste_fraction=waste / total_work if total_work else 0.0,
        slo_missed_gpu_work=slo_missed_work,
        prefill_normalized_utilization=prefill_work / p_capacity,
        decode_normalized_utilization=decode_work / d_capacity,
        cluster_normalized_utilization=total_work / (p_capacity + d_capacity),
        kvs_bytes_per_work=sum(value.kvs_bytes for value in attempts) / horizon,
        jain_fairness=jain,
        per_tier_slo_attainment=tier_attainment,
        minimum_tier_slo_attainment=minimum_tier,
        fairness_floor_pass=(
            minimum_tier >= MINIMUM_TIER_SLO_ATTAINMENT
            and jain >= MINIMUM_JAIN_FAIRNESS
        ),
    )


def _jain(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    denominator = len(values) * sum(value * value for value in values)
    return sum(values) ** 2 / denominator if denominator else 0.0
