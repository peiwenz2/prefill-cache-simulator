"""Shared causal discrete-event kernel for M12 policy evaluation.

The kernel owns truth, time, resource pools, and metric attribution.  Policies only
return immutable attempt execution plans from a past-only view.
"""

from __future__ import annotations

import heapq
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType
from typing import TypeVar

from .m12_metrics import (
    AttemptOutcome,
    LogicalRequestSpec,
    M12MetricReport,
    aggregate_m12_metrics,
)

_T = TypeVar("_T")


def _remaining_queue_work(
    remaining_components: Iterable[float] | None,
) -> float:
    """Update the queue proxy using the waiting heap as structural truth."""
    if remaining_components is None:
        return 0.0
    return math.fsum(remaining_components)


@dataclass(frozen=True, slots=True)
class FrozenKernelCostModel:
    prefill_work_per_token: float
    kvs_work_per_token: float
    kvs_bytes_per_token: int
    decode_work_per_token: float

    def __post_init__(self) -> None:
        values = (
            self.prefill_work_per_token,
            self.kvs_work_per_token,
            self.decode_work_per_token,
        )
        if any(not _is_finite_number(value) or value < 0 for value in values):
            raise ValueError("kernel costs must be finite and non-negative")
        if not _is_plain_int(self.kvs_bytes_per_token) or self.kvs_bytes_per_token < 0:
            raise ValueError("KVS bytes per token must be a plain non-negative integer")


@dataclass(frozen=True, slots=True)
class KernelConfig:
    observation_start_work: float
    observation_end_work: float
    prefill_node_ids: tuple[str, ...]
    decode_node_ids: tuple[str, ...]
    tier_slo_work: Mapping[str, float]
    cache_capacity_entries: int
    cost_model: FrozenKernelCostModel
    retry_budget: int = 2

    def __post_init__(self) -> None:
        if not _is_finite_number(self.observation_start_work) or not _is_finite_number(
            self.observation_end_work
        ):
            raise ValueError("observation window boundaries must be finite")
        if (
            self.observation_start_work < 0
            or self.observation_end_work <= self.observation_start_work
        ):
            raise ValueError("observation window must be positive")
        if not self.prefill_node_ids or not self.decode_node_ids:
            raise ValueError("P and D pools must be non-empty")
        if (
            not _is_plain_int(self.cache_capacity_entries)
            or self.cache_capacity_entries <= 0
        ):
            raise ValueError("cache capacity must be a plain positive integer")
        if not _is_plain_int(self.retry_budget) or self.retry_budget < 0:
            raise ValueError("retry budget must be a plain non-negative integer")
        if len(set(self.prefill_node_ids)) != len(self.prefill_node_ids):
            raise ValueError("duplicate P node identity")
        if len(set(self.decode_node_ids)) != len(self.decode_node_ids):
            raise ValueError("duplicate D node identity")
        slos = dict(self.tier_slo_work)
        if not slos or any(
            not tier or value < 0 or not math.isfinite(value)
            for tier, value in slos.items()
        ):
            raise ValueError("tier SLOs must be named, finite, and non-negative")
        object.__setattr__(self, "tier_slo_work", MappingProxyType(slos))


@dataclass(frozen=True, slots=True)
class KernelRequestSpec:
    logical: LogicalRequestSpec
    prefix_cache_keys: tuple[str, ...]
    prefix_token_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not _is_plain_int(self.logical.input_tokens) or not _is_plain_int(
            self.logical.true_output_tokens
        ):
            raise ValueError("logical token fields must be plain integers")
        if len(self.prefix_cache_keys) != len(self.prefix_token_sizes):
            raise ValueError("prefix keys and token sizes differ")
        if any(not key for key in self.prefix_cache_keys):
            raise ValueError("prefix cache keys must be non-empty")
        if any(
            not _is_plain_int(size) or size <= 0 for size in self.prefix_token_sizes
        ):
            raise ValueError("prefix token sizes must be plain positive integers")
        if sum(self.prefix_token_sizes) != self.logical.input_tokens:
            raise ValueError("prefix sizes differ from frozen logical input")


class AttemptTerminal(StrEnum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


@dataclass(frozen=True, slots=True)
class AttemptExecutionSpec:
    """Policy-issued physical work; it contains no workload truth fields."""

    logical_request_id: str
    attempt_index: int
    arrival_work: float
    p_node_id: str
    d_node_id: str
    emitted_output_tokens: int
    remote_cache_keys: tuple[str, ...] = ()
    terminal: AttemptTerminal = AttemptTerminal.COMPLETED
    prefill_reusable: bool = True

    def __post_init__(self) -> None:
        if not self.logical_request_id or not self.p_node_id or not self.d_node_id:
            raise ValueError("attempt and pool-node identities must be non-empty")
        if not _is_plain_int(self.attempt_index) or self.attempt_index < 0:
            raise ValueError("attempt index must be a plain non-negative integer")
        if not _is_finite_number(self.arrival_work) or self.arrival_work < 0:
            raise ValueError("attempt arrival must be finite and non-negative")
        if (
            not _is_plain_int(self.emitted_output_tokens)
            or self.emitted_output_tokens < 0
        ):
            raise ValueError("emitted tokens must be a plain non-negative integer")
        if not isinstance(self.terminal, AttemptTerminal):
            raise ValueError("terminal must be an AttemptTerminal enum")
        if type(self.prefill_reusable) is not bool:
            raise ValueError("prefill_reusable must be a boolean")


@dataclass(frozen=True, slots=True)
class CausalView:
    now_work: float
    completed_cache_keys: AbstractSet[str]
    prefill_available_at: Mapping[str, float]
    decode_available_at: Mapping[str, float]
    cache_by_node: Mapping[str, frozenset[str]]
    # Ready but not started, conservative P-only work. In-flight reservation is
    # represented separately by ``prefill_available_at``; KVS is priced at P-start.
    queued_prefill_work: Mapping[str, float]
    retry_budget_remaining: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "completed_cache_keys", frozenset(self.completed_cache_keys)
        )
        object.__setattr__(
            self,
            "prefill_available_at",
            MappingProxyType(dict(self.prefill_available_at)),
        )
        object.__setattr__(
            self,
            "decode_available_at",
            MappingProxyType(dict(self.decode_available_at)),
        )
        object.__setattr__(
            self, "cache_by_node", MappingProxyType(dict(self.cache_by_node))
        )
        object.__setattr__(
            self,
            "queued_prefill_work",
            MappingProxyType(dict(self.queued_prefill_work)),
        )

    @classmethod
    def _from_kernel(
        cls,
        now_work: float,
        completed_cache_keys: AbstractSet[str],
        prefill_available_at: Mapping[str, float],
        decode_available_at: Mapping[str, float],
        cache_by_node: Mapping[str, frozenset[str]],
        queued_prefill_work: Mapping[str, float],
    ) -> CausalView:
        """Build from kernel-owned immutable snapshots without copying them again."""
        view = object.__new__(cls)
        object.__setattr__(view, "now_work", now_work)
        object.__setattr__(view, "completed_cache_keys", completed_cache_keys)
        object.__setattr__(view, "prefill_available_at", prefill_available_at)
        object.__setattr__(view, "decode_available_at", decode_available_at)
        object.__setattr__(view, "cache_by_node", cache_by_node)
        object.__setattr__(view, "queued_prefill_work", queued_prefill_work)
        object.__setattr__(view, "retry_budget_remaining", 0)
        return view


class _CompletedCacheKeysView(AbstractSet[str]):
    """Immutable lazy union over one frozen per-node cache snapshot."""

    __slots__ = ("_mapping", "_record_visits")

    def __init__(
        self,
        mapping: Mapping[str, frozenset[str]],
        record_visits: Callable[[int], None],
    ) -> None:
        self._mapping = mapping
        self._record_visits = record_visits

    def __contains__(self, key: object) -> bool:
        return any(key in values for values in self._mapping.values())

    def __iter__(self) -> Iterator[str]:
        seen: set[str] = set()
        visits = 0
        for values in self._mapping.values():
            for key in values:
                visits += 1
                if key in seen:
                    continue
                seen.add(key)
                yield key
        self._record_visits(visits)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    @classmethod
    def _from_iterable(cls, values: Iterable[_T]) -> AbstractSet[_T]:
        return frozenset(values)


class _CacheSnapshotStore:
    """Versioned immutable views over kernel-owned mutable node caches."""

    __slots__ = (
        "_cache",
        "_completed",
        "_dirty_nodes",
        "_mapping",
        "_node_order",
        "snapshot_entry_copies",
        "union_entry_visits",
    )

    def __init__(self, cache: Mapping[str, set[str]]) -> None:
        self._cache = cache
        self._node_order = tuple(cache)
        self._mapping: Mapping[str, frozenset[str]] | None = None
        self._completed: AbstractSet[str] | None = None
        self._dirty_nodes = set(cache)
        self.snapshot_entry_copies = 0
        self.union_entry_visits = 0

    def invalidate(self, node: str) -> None:
        self._dirty_nodes.add(node)
        self._completed = None

    def _record_union_visits(self, visits: int) -> None:
        self.union_entry_visits += visits

    def view(self) -> tuple[Mapping[str, frozenset[str]], AbstractSet[str]]:
        if self._mapping is None or self._dirty_nodes:
            frozen = dict(self._mapping or {})
            for node in self._node_order:
                if node not in self._dirty_nodes:
                    continue
                values = frozenset(self._cache[node])
                self.snapshot_entry_copies += len(values)
                frozen[node] = values
            self._mapping = MappingProxyType(frozen)
            self._dirty_nodes.clear()
        if self._completed is None:
            self._completed = _CompletedCacheKeysView(
                self._mapping, self._record_union_visits
            )
        return self._mapping, self._completed


@dataclass(frozen=True, slots=True)
class CacheMutation:
    admit: bool
    evict_keys: tuple[str, ...] = ()


class KernelPolicy(ABC):
    """Hook boundary for selector/admission/lease/eviction policy composition."""

    @abstractmethod
    def plan_attempts(
        self, request: KernelRequestSpec, view: CausalView
    ) -> Sequence[AttemptExecutionSpec]:
        """Return attempts using only frozen request data and the causal view."""

    def cache_mutation(
        self,
        request: KernelRequestSpec,
        attempt: AttemptExecutionSpec,
        view: CausalView,
    ) -> CacheMutation:
        """Choose completion-time admission/eviction; the kernel validates it."""
        return CacheMutation(admit=True)

    def next_attempt(
        self,
        request: KernelRequestSpec,
        previous: AttemptExecutionSpec,
        view: CausalView,
    ) -> AttemptExecutionSpec | None:
        """Optionally issue one retry after a failed/aborted physical completion."""
        return None

    def decode_not_before(
        self,
        request: KernelRequestSpec,
        attempt: AttemptExecutionSpec,
        view: CausalView,
    ) -> float:
        """Return an idempotent D-start fence; it may be re-read after wake-up."""
        return view.now_work

    def admission_event(
        self, request: KernelRequestSpec, attempt: AttemptExecutionSpec
    ) -> str | None:
        """Optionally expose a policy admission lifecycle marker."""
        return None

    def reprice_decode(
        self,
        request: KernelRequestSpec,
        attempt: AttemptExecutionSpec,
        view: CausalView,
    ) -> AttemptExecutionSpec:
        """Reprice only at the decode/lease boundary."""
        return attempt

    def decode_lease_tokens(
        self,
        request: KernelRequestSpec,
        attempt: AttemptExecutionSpec,
        view: CausalView,
    ) -> int | None:
        """Return a positive boundary strictly before full output, or no lease."""
        return None

    def decode_started(
        self,
        request: KernelRequestSpec,
        attempt: AttemptExecutionSpec,
        view: CausalView,
        *,
        finish_work: float,
    ) -> None:
        """Publish kernel-owned actual D residency to the policy."""
        return None

    def attempt_finished(
        self,
        request: KernelRequestSpec,
        attempt: AttemptExecutionSpec,
        view: CausalView,
        *,
        actual_decode_work: float,
    ) -> None:
        """Settle policy reservations against kernel-owned execution truth."""
        return None


@dataclass(frozen=True, slots=True)
class KernelEvent:
    at_work: float
    kind: str
    logical_request_id: str
    attempt_index: int | None = None


@dataclass(frozen=True, slots=True)
class ExecutedAttempt:
    logical_request_id: str
    attempt_index: int
    prefill_finish_work: float
    completed: bool
    finish_work: float | None
    strict_slo_met: bool
    outcome: AttemptOutcome


@dataclass(frozen=True, slots=True)
class KernelResult:
    attempts: tuple[ExecutedAttempt, ...]
    metrics: M12MetricReport
    events: tuple[KernelEvent, ...]
    completed_cache_keys: frozenset[str]


@dataclass(slots=True)
class _Pending:
    request: KernelRequestSpec
    spec: AttemptExecutionSpec
    prefill_finish: float | None = None
    finish: float | None = None
    prefill_work: float = 0
    kvs_work: float = 0
    kvs_bytes: int = 0
    decode_start: float | None = None


class CausalKernel:
    """Deterministic event heap with completion-before-arrival tie semantics."""

    _COMPLETION = 0
    _LEASE_BOUNDARY = 1
    _PREFILL_FINISH = 2
    _DECODE_START = 3
    _ATTEMPT_READY = 4
    _PREFILL_START = 5
    _ARRIVAL = 6

    def __init__(self, config: KernelConfig) -> None:
        self.config = config

    def run(
        self,
        workload: Sequence[LogicalRequestSpec | KernelRequestSpec],
        policy: KernelPolicy,
    ) -> KernelResult:
        requests = tuple(self._freeze_request(value) for value in workload)
        logicals = tuple(value.logical for value in requests)
        self._validate_workload(requests)
        p_available = {
            node: self.config.observation_start_work
            for node in self.config.prefill_node_ids
        }
        d_available = {
            node: self.config.observation_start_work
            for node in self.config.decode_node_ids
        }
        cache: dict[str, set[str]] = {
            node: set() for node in self.config.prefill_node_ids
        }
        cache_snapshots = _CacheSnapshotStore(cache)
        p_busy = {node: False for node in self.config.prefill_node_ids}
        p_waiting: dict[str, list[tuple[float, int, tuple[str, int]]]] = {
            node: [] for node in self.config.prefill_node_ids
        }
        queued_work = {node: 0.0 for node in self.config.prefill_node_ids}
        planned: dict[tuple[str, int], _Pending] = {}
        pending: dict[tuple[str, int], _Pending] = {}
        attempt_keys: set[tuple[str, int]] = set()
        event_log: list[KernelEvent] = []
        queue: list[tuple[float, int, int, str, object]] = []
        serial = 0
        for request in requests:
            heapq.heappush(
                queue,
                (
                    request.logical.arrival_work,
                    self._ARRIVAL,
                    serial,
                    "ARRIVAL",
                    request,
                ),
            )
            serial += 1

        while queue:
            at, priority, _, kind, payload = heapq.heappop(queue)
            if at > self.config.observation_end_work:
                break
            if priority == self._ARRIVAL:
                arrival_request = payload
                assert isinstance(arrival_request, KernelRequestSpec)
                logical = arrival_request.logical
                event_log.append(KernelEvent(at, kind, logical.logical_request_id))
                view = self._view(
                    at, cache_snapshots, p_available, d_available, queued_work
                )
                plans = tuple(policy.plan_attempts(arrival_request, view))
                if len(plans) > 1:
                    raise ValueError("initial policy may issue at most one attempt")
                self._validate_plans(logical, plans, attempt_keys)
                for spec in plans:
                    admission_kind = policy.admission_event(arrival_request, spec)
                    if admission_kind is not None:
                        event_log.append(
                            KernelEvent(
                                at,
                                admission_kind,
                                spec.logical_request_id,
                                spec.attempt_index,
                            )
                        )
                    key = (spec.logical_request_id, spec.attempt_index)
                    attempt_keys.add(key)
                    planned[key] = _Pending(arrival_request, spec)
                    heapq.heappush(
                        queue,
                        (
                            spec.arrival_work,
                            self._ATTEMPT_READY,
                            serial,
                            "ATTEMPT_READY",
                            key,
                        ),
                    )
                    serial += 1
            elif priority == self._ATTEMPT_READY:
                ready_key = payload
                assert isinstance(ready_key, tuple)
                item = planned.pop(ready_key)
                pending[ready_key] = item
                spec = item.spec
                queued_work[spec.p_node_id] += self._maximum_prefill_work(item.request)
                heapq.heappush(p_waiting[spec.p_node_id], (at, serial, ready_key))
                if not p_busy[spec.p_node_id]:
                    p_busy[spec.p_node_id] = True
                    _, _, next_key = heapq.heappop(p_waiting[spec.p_node_id])
                    heapq.heappush(
                        queue,
                        (at, self._PREFILL_START, serial, "PREFILL_START", next_key),
                    )
                serial += 1
            elif priority == self._PREFILL_START:
                start_key = payload
                assert isinstance(start_key, tuple)
                item = pending[start_key]
                self._price_prefill(item, cache)
                node_id = item.spec.p_node_id
                queued_work[node_id] = _remaining_queue_work(
                    (
                        (
                            self._maximum_prefill_work(pending[key].request)
                            for _, _, key in p_waiting[node_id]
                        )
                        if p_waiting[node_id]
                        else None
                    ),
                )
                finish = at + item.prefill_work + item.kvs_work
                item.prefill_finish = finish
                p_available[item.spec.p_node_id] = finish
                heapq.heappush(
                    queue,
                    (
                        finish,
                        self._PREFILL_FINISH,
                        serial,
                        "PREFILL_FINISH",
                        start_key,
                    ),
                )
                serial += 1
            elif priority == self._PREFILL_FINISH:
                prefill_key = payload
                assert isinstance(prefill_key, tuple)
                item = pending[prefill_key]
                spec = item.spec
                p_busy[spec.p_node_id] = False
                if p_waiting[spec.p_node_id]:
                    next_at, _, next_key = heapq.heappop(p_waiting[spec.p_node_id])
                    p_busy[spec.p_node_id] = True
                    heapq.heappush(
                        queue,
                        (
                            max(at, next_at),
                            self._PREFILL_START,
                            serial,
                            "PREFILL_START",
                            next_key,
                        ),
                    )
                    serial += 1
                view = self._view(
                    at, cache_snapshots, p_available, d_available, queued_work
                )
                policy_fence = policy.decode_not_before(item.request, spec, view)
                if not _is_finite_number(policy_fence) or policy_fence < at:
                    raise ValueError("decode fence must be finite and causal")
                d_start = max(at, d_available[spec.d_node_id], policy_fence)
                if d_start > at:
                    event_log.append(
                        KernelEvent(
                            at,
                            "DECODE_GATED",
                            spec.logical_request_id,
                            spec.attempt_index,
                        )
                    )
                    heapq.heappush(
                        queue,
                        (
                            d_start,
                            self._DECODE_START,
                            serial,
                            "DECODE_START",
                            prefill_key,
                        ),
                    )
                else:
                    finish = (
                        at
                        + spec.emitted_output_tokens
                        * self.config.cost_model.decode_work_per_token
                    )
                    d_available[spec.d_node_id] = finish
                    item.decode_start = at
                    item.finish = finish
                    policy.decode_started(
                        item.request, spec, view, finish_work=finish
                    )
                    lease_tokens = policy.decode_lease_tokens(
                        item.request, spec, view
                    )
                    event_at = finish
                    event_priority = self._COMPLETION
                    event_kind = "COMPLETION"
                    if lease_tokens is not None:
                        if not 0 < lease_tokens < spec.emitted_output_tokens:
                            raise ValueError("lease must precede fully emitted output")
                        event_at = (
                            at
                            + lease_tokens
                            * self.config.cost_model.decode_work_per_token
                        )
                        event_priority = self._LEASE_BOUNDARY
                        event_kind = "LEASE_BOUNDARY"
                    heapq.heappush(
                        queue,
                        (event_at, event_priority, serial, event_kind, prefill_key),
                    )
                serial += 1
            elif priority == self._DECODE_START:
                decode_key = payload
                assert isinstance(decode_key, tuple)
                item = pending[decode_key]
                if item.decode_start is not None:
                    continue
                live_view = self._view(
                    at, cache_snapshots, p_available, d_available, queued_work
                )
                policy_fence = policy.decode_not_before(
                    item.request, item.spec, live_view
                )
                if not _is_finite_number(policy_fence):
                    raise ValueError("decode fence must be finite")
                next_start = max(
                    at, d_available[item.spec.d_node_id], policy_fence
                )
                if next_start > at:
                    heapq.heappush(
                        queue,
                        (
                            next_start,
                            self._DECODE_START,
                            serial,
                            "DECODE_START",
                            decode_key,
                        ),
                    )
                    serial += 1
                    continue
                event_log.append(KernelEvent(at, kind, decode_key[0], decode_key[1]))
                lease_view = replace(
                    live_view,
                    retry_budget_remaining=max(
                        0, self.config.retry_budget - item.spec.attempt_index
                    ),
                )
                finish = (
                    at
                    + item.spec.emitted_output_tokens
                    * self.config.cost_model.decode_work_per_token
                )
                d_available[item.spec.d_node_id] = finish
                item.decode_start = at
                item.finish = finish
                policy.decode_started(
                    item.request, item.spec, lease_view, finish_work=finish
                )
                lease_tokens = policy.decode_lease_tokens(
                    item.request, item.spec, lease_view
                )
                event_at = finish
                event_priority = self._COMPLETION
                event_kind = "COMPLETION"
                if lease_tokens is not None:
                    if not 0 < lease_tokens < item.spec.emitted_output_tokens:
                        raise ValueError("lease must precede fully emitted output")
                    event_at = (
                        at
                        + lease_tokens
                        * self.config.cost_model.decode_work_per_token
                    )
                    event_priority = self._LEASE_BOUNDARY
                    event_kind = "LEASE_BOUNDARY"
                heapq.heappush(
                    queue,
                    (event_at, event_priority, serial, event_kind, decode_key),
                )
                serial += 1
            elif priority == self._LEASE_BOUNDARY:
                lease_key = payload
                assert isinstance(lease_key, tuple)
                item = pending[lease_key]
                event_log.append(KernelEvent(at, kind, lease_key[0], lease_key[1]))
                lease_view = replace(
                    self._view(
                        at, cache_snapshots, p_available, d_available, queued_work
                    ),
                    retry_budget_remaining=max(
                        0, self.config.retry_budget - item.spec.attempt_index
                    ),
                )
                revised = policy.reprice_decode(item.request, item.spec, lease_view)
                if revised.terminal is AttemptTerminal.ABORTED:
                    if item.decode_start is None:
                        raise ValueError("lease boundary is missing D start truth")
                    emitted = _exact_elapsed_tokens(
                        at - item.decode_start,
                        self.config.cost_model.decode_work_per_token,
                    )
                    if emitted <= 0 or emitted >= item.spec.emitted_output_tokens:
                        raise ValueError("abort must follow partial lease progress")
                    item.spec = replace(revised, emitted_output_tokens=emitted)
                    item.finish = at
                    d_available[item.spec.d_node_id] = at
                    waiting_keys = sorted(
                        {
                            queued_payload
                            for (
                                queued_at,
                                queued_priority,
                                _,
                                _,
                                queued_payload,
                            ) in queue
                            if queued_priority == self._DECODE_START
                            and queued_at > at
                            and isinstance(queued_payload, tuple)
                            and pending[queued_payload].spec.d_node_id
                            == item.spec.d_node_id
                            and pending[queued_payload].decode_start is None
                        }
                    )
                    for waiting_key in waiting_keys:
                        heapq.heappush(
                            queue,
                            (
                                at,
                                self._DECODE_START,
                                serial,
                                "DECODE_START",
                                waiting_key,
                            ),
                        )
                        serial += 1
                    heapq.heappush(
                        queue,
                        (at, self._COMPLETION, serial, "COMPLETION", lease_key),
                    )
                else:
                    assert item.finish is not None
                    heapq.heappush(
                        queue,
                        (
                            item.finish,
                            self._COMPLETION,
                            serial,
                            "COMPLETION",
                            lease_key,
                        ),
                    )
                serial += 1
            else:
                completion_key = payload
                assert isinstance(completion_key, tuple)
                item = pending[completion_key]
                event_log.append(
                    KernelEvent(at, kind, completion_key[0], completion_key[1])
                )
                completion_view = self._view(
                    at, cache_snapshots, p_available, d_available, queued_work
                )
                policy.attempt_finished(
                    item.request,
                    item.spec,
                    completion_view,
                    actual_decode_work=(
                        item.spec.emitted_output_tokens
                        * self.config.cost_model.decode_work_per_token
                    ),
                )
                mutation = policy.cache_mutation(
                    item.request,
                    item.spec,
                    completion_view,
                )
                if (
                    item.spec.terminal is AttemptTerminal.COMPLETED
                    or item.spec.prefill_reusable
                ) and self._apply_cache_mutation(item, mutation, cache):
                    cache_snapshots.invalidate(item.spec.p_node_id)
                if item.spec.terminal is not AttemptTerminal.COMPLETED:
                    retry = policy.next_attempt(
                        item.request,
                        item.spec,
                            self._view(
                                at,
                                cache_snapshots,
                                p_available,
                                d_available,
                                queued_work,
                            ),
                    )
                    used_retries = item.spec.attempt_index
                    if retry is not None:
                        if used_retries >= self.config.retry_budget:
                            raise ValueError("retry budget exhausted")
                        self._validate_plans(
                            item.request.logical, (retry,), attempt_keys
                        )
                        retry_key = (retry.logical_request_id, retry.attempt_index)
                        attempt_keys.add(retry_key)
                        admission_kind = policy.admission_event(item.request, retry)
                        if admission_kind is not None:
                            event_log.append(
                                KernelEvent(
                                    at,
                                    admission_kind,
                                    retry.logical_request_id,
                                    retry.attempt_index,
                                )
                            )
                        planned[retry_key] = _Pending(item.request, retry)
                        heapq.heappush(
                            queue,
                            (
                                max(at, retry.arrival_work),
                                self._ATTEMPT_READY,
                                serial,
                                "ATTEMPT_READY",
                                retry_key,
                            ),
                        )
                        serial += 1

        executed = tuple(
            self._materialize(item)
            for _, item in sorted(pending.items(), key=lambda value: value[0])
        )
        metrics = aggregate_m12_metrics(
            logicals,
            tuple(value.outcome for value in executed),
            observation_start_work=self.config.observation_start_work,
            observation_end_work=self.config.observation_end_work,
            prefill_nodes=len(self.config.prefill_node_ids),
            decode_nodes=len(self.config.decode_node_ids),
        )
        return KernelResult(
            executed,
            metrics,
            tuple(event_log),
            frozenset(cache_snapshots.view()[1]),
        )

    def _materialize(self, item: _Pending) -> ExecutedAttempt:
        completed = (
            item.spec.terminal is AttemptTerminal.COMPLETED
            and item.finish is not None
            and item.finish <= self.config.observation_end_work
            and item.spec.emitted_output_tokens
            == item.request.logical.true_output_tokens
        )
        if (
            item.spec.terminal is AttemptTerminal.COMPLETED
            and item.finish is not None
            and item.finish <= self.config.observation_end_work
            and item.spec.emitted_output_tokens
            != item.request.logical.true_output_tokens
        ):
            raise ValueError("completed attempt differs from frozen true output")
        finish = item.finish if completed else None
        finish_for_slo = finish if finish is not None else math.inf
        strict = completed and (
            finish_for_slo - item.request.logical.arrival_work
            <= self.config.tier_slo_work[item.request.logical.tier]
        )
        outcome = AttemptOutcome(
            item.request.logical.logical_request_id,
            item.spec.attempt_index,
            item.request.logical.tenant_id,
            item.request.logical.tier,
            item.request.logical.arrival_work,
            item.request.logical.input_tokens,
            item.request.logical.true_output_tokens,
            item.spec.emitted_output_tokens,
            completed,
            strict,
            finish,
            item.prefill_work,
            item.spec.emitted_output_tokens
            * self.config.cost_model.decode_work_per_token,
            item.prefill_work
            if item.spec.terminal is not AttemptTerminal.COMPLETED
            and not item.spec.prefill_reusable
            else 0,
            item.spec.emitted_output_tokens
            * self.config.cost_model.decode_work_per_token
            if item.spec.terminal is not AttemptTerminal.COMPLETED
            else 0,
            item.kvs_bytes,
            item.kvs_work,
        )
        return ExecutedAttempt(
            item.request.logical.logical_request_id,
            item.spec.attempt_index,
            item.prefill_finish if item.prefill_finish is not None else math.inf,
            completed,
            finish,
            strict,
            outcome,
        )

    def _validate_workload(self, workload: tuple[KernelRequestSpec, ...]) -> None:
        identities: set[str] = set()
        previous = self.config.observation_start_work
        for request in workload:
            logical = request.logical
            if not _is_finite_number(logical.arrival_work):
                raise ValueError("logical request arrival must be finite")
            if logical.logical_request_id in identities:
                raise ValueError(
                    f"duplicate workload identity: {logical.logical_request_id}"
                )
            identities.add(logical.logical_request_id)
            if logical.arrival_work < previous:
                raise ValueError("workload must be in non-decreasing arrival order")
            if not (
                self.config.observation_start_work
                <= logical.arrival_work
                < self.config.observation_end_work
            ):
                raise ValueError("workload arrival is outside the observation window")
            if logical.tier not in self.config.tier_slo_work:
                raise ValueError(f"missing evaluator SLO for tier: {logical.tier}")
            previous = logical.arrival_work

    def _validate_plans(
        self,
        logical: LogicalRequestSpec,
        plans: tuple[AttemptExecutionSpec, ...],
        attempt_keys: AbstractSet[tuple[str, int]],
    ) -> None:
        previous = -1
        seen: set[int] = set()
        for spec in plans:
            if spec.logical_request_id != logical.logical_request_id:
                raise ValueError("attempt references unknown logical workload")
            if (
                spec.attempt_index in seen
                or (spec.logical_request_id, spec.attempt_index) in attempt_keys
            ):
                raise ValueError("duplicate attempt identity")
            if spec.attempt_index <= previous:
                raise ValueError("attempt order must be strictly increasing")
            if spec.arrival_work < logical.arrival_work:
                raise ValueError("attempt arrival precedes logical arrival")
            if spec.p_node_id not in self.config.prefill_node_ids:
                raise ValueError("attempt references unknown P node")
            if spec.d_node_id not in self.config.decode_node_ids:
                raise ValueError("attempt references unknown D node")
            seen.add(spec.attempt_index)
            previous = spec.attempt_index

    def _price_prefill(self, item: _Pending, cache: Mapping[str, set[str]]) -> None:
        local = cache[item.spec.p_node_id]
        remote_claim = set(item.spec.remote_cache_keys)
        remote_tokens = 0
        covered_tokens = 0
        for key, tokens in zip(
            item.request.prefix_cache_keys,
            item.request.prefix_token_sizes,
            strict=True,
        ):
            if key in local:
                covered_tokens += tokens
            elif key in remote_claim and any(
                key in values
                for node, values in cache.items()
                if node != item.spec.p_node_id
            ):
                covered_tokens += tokens
                remote_tokens += tokens
            else:
                break
        uncached = item.request.logical.input_tokens - covered_tokens
        model = self.config.cost_model
        item.prefill_work = uncached * model.prefill_work_per_token
        item.kvs_work = remote_tokens * model.kvs_work_per_token
        item.kvs_bytes = remote_tokens * model.kvs_bytes_per_token

    def _apply_cache_mutation(self, item, mutation, cache) -> bool:
        node_cache = cache[item.spec.p_node_id]
        victims = set(mutation.evict_keys)
        if not victims <= node_cache:
            raise ValueError("eviction references non-resident cache key")
        node_cache.difference_update(victims)
        changed = bool(victims)
        if mutation.admit:
            additions = set(item.request.prefix_cache_keys)
            missing = additions - node_cache
            if len(node_cache) + len(missing) > self.config.cache_capacity_entries:
                raise ValueError("cache admission exceeds bounded capacity")
            node_cache.update(missing)
            changed = changed or bool(missing)
        return changed

    @staticmethod
    def _freeze_request(
        value: LogicalRequestSpec | KernelRequestSpec,
    ) -> KernelRequestSpec:
        if isinstance(value, KernelRequestSpec):
            return value
        if not _is_plain_int(value.input_tokens) or not _is_plain_int(
            value.true_output_tokens
        ):
            raise ValueError("logical token fields must be plain integers")
        key = f"logical:{value.logical_request_id}"
        if value.input_tokens == 0:
            return KernelRequestSpec(value, (), ())
        return KernelRequestSpec(value, (key,), (value.input_tokens,))

    def _maximum_prefill_work(self, request: KernelRequestSpec) -> float:
        return (
            request.logical.input_tokens * self.config.cost_model.prefill_work_per_token
        )

    @staticmethod
    def _view(
        now, cache_snapshots, p_available, d_available, queued_work
    ) -> CausalView:
        frozen, completed = cache_snapshots.view()
        return CausalView._from_kernel(
            now,
            completed,
            MappingProxyType(dict(p_available)),
            MappingProxyType(dict(d_available)),
            frozen,
            MappingProxyType(dict(queued_work)),
        )


def _is_plain_int(value: object) -> bool:
    return type(value) is int


def _exact_elapsed_tokens(work: float, work_per_token: float) -> int:
    if work_per_token <= 0:
        raise ValueError("lease boundary requires positive decode cost")
    tokens = round(work / work_per_token)
    if not math.isclose(tokens * work_per_token, work):
        raise ValueError("lease boundary is not token aligned")
    return tokens


def _is_finite_number(value: object) -> bool:
    if type(value) is int:
        return True
    if type(value) is float:
        return math.isfinite(value)
    return False
