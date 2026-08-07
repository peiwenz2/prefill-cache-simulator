"""M12.3 decode-capacity futures composed over M12.2 Priced Spill."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum

from .m12_kernel import (
    AttemptExecutionSpec,
    AttemptTerminal,
    CacheMutation,
    CausalView,
    KernelPolicy,
    KernelRequestSpec,
    KernelResult,
)
from .m12_placement import M12PlacementPolicy


class DecodeAdmissionMode(StrEnum):
    NO_GATE = "NO_GATE"
    ORACLE = "ORACLE_NON_DEPLOYABLE"
    ORACLE_NOISED = "ORACLE_NOISED_NON_DEPLOYABLE"
    CAUSAL = "CAUSAL_PAST_ONLY"

    @property
    def deployable(self) -> bool:
        return self in (DecodeAdmissionMode.NO_GATE, DecodeAdmissionMode.CAUSAL)

    @property
    def uses_output_label(self) -> bool:
        return self in (
            DecodeAdmissionMode.ORACLE,
            DecodeAdmissionMode.ORACLE_NOISED,
        )


class AdmissionAction(StrEnum):
    ADMIT = "ADMIT"
    DEFER = "DEFER"
    GATED_PD = "GATED_PD"
    GATED_DP = "GATED_DP"


@dataclass(frozen=True, slots=True)
class AbortFence:
    at_lease_boundary: bool
    continuation_value: float
    output_authority_held: bool
    retry_budget_remaining: int

    @property
    def allows_abort(self) -> bool:
        return (
            self.at_lease_boundary
            and self.continuation_value < 0
            and self.output_authority_held
            and self.retry_budget_remaining > 0
        )


@dataclass(frozen=True, slots=True)
class DecodeAdmissionConfig:
    mode: DecodeAdmissionMode
    capacity_credits: float
    congestion_action: AdmissionAction = AdmissionAction.DEFER
    oracle_noise_multiplier: float = 1.0
    lease_boundary_tokens: int = 8
    abort_fences: Mapping[str, AbortFence] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.mode, DecodeAdmissionMode):
            raise ValueError("mode must be a DecodeAdmissionMode")
        if not isinstance(self.congestion_action, AdmissionAction):
            raise ValueError("congestion action must be explicit")
        if self.congestion_action is AdmissionAction.ADMIT:
            raise ValueError("configured congestion action cannot be ADMIT")
        if self.capacity_credits < 0 or not math.isfinite(self.capacity_credits):
            raise ValueError("capacity credits must be finite and non-negative")
        if self.oracle_noise_multiplier <= 0 or not math.isfinite(
            self.oracle_noise_multiplier
        ):
            raise ValueError("oracle noise must be finite and positive")
        if (
            type(self.lease_boundary_tokens) is not int
            or self.lease_boundary_tokens <= 0
        ):
            raise ValueError("lease boundary tokens must be a positive integer")
        object.__setattr__(self, "abort_fences", dict(self.abort_fences))


class PrefixFamilyPredictor:
    """Online median predictor containing completed past observations only."""

    def __init__(self, *, default_output_tokens: int, history_limit: int = 32) -> None:
        if type(default_output_tokens) is not int or default_output_tokens < 0:
            raise ValueError("default output tokens must be a non-negative integer")
        if type(history_limit) is not int or history_limit <= 0:
            raise ValueError("history limit must be a positive integer")
        self.default_output_tokens = default_output_tokens
        self.history_limit = history_limit
        self._history: dict[str, deque[tuple[float, int]]] = defaultdict(
            lambda: deque(maxlen=history_limit)
        )
        self._latest_time = -math.inf

    def predict(self, request: KernelRequestSpec, *, at_work: float) -> int:
        self._check_time(at_work)
        values = [
            output
            for completed, output in self._history[self._family(request)]
            if completed <= at_work
        ]
        if not values:
            return self.default_output_tokens
        ordered = sorted(values)
        return ordered[(len(ordered) - 1) // 2]

    def observe(self, request: KernelRequestSpec, *, completed_at_work: float) -> None:
        self._check_time(completed_at_work)
        self._history[self._family(request)].append(
            (completed_at_work, request.logical.true_output_tokens)
        )

    def _check_time(self, at_work: float) -> None:
        if not math.isfinite(at_work) or at_work < self._latest_time:
            raise ValueError("future observations cannot be read by a past prediction")
        self._latest_time = at_work

    @staticmethod
    def _family(request: KernelRequestSpec) -> str:
        return request.prefix_cache_keys[0] if request.prefix_cache_keys else "<empty>"


@dataclass(frozen=True, slots=True)
class DecodeReservation:
    logical_request_id: str
    attempt_index: int
    credits: float
    starts_at_work: float
    releases_at_work: float


class DecodeCreditLedger:
    """Shared temporal D-capacity reservations and P→D debt."""

    def __init__(self, capacity_credits: float) -> None:
        if capacity_credits < 0 or not math.isfinite(capacity_credits):
            raise ValueError("capacity credits must be finite and non-negative")
        self.capacity_credits = capacity_credits
        self._active: dict[tuple[str, int], tuple[DecodeReservation, ...]] = {}
        self._executing: dict[tuple[str, int], DecodeReservation] = {}
        self._reserved_totals: dict[tuple[str, int], float] = {}
        self.actual_decode_work = 0.0
        self.reconciled_credit_error = 0.0
        self.peak_reserved_decode_credits = 0.0

    @property
    def reserved_decode_credits(self) -> float:
        return sum(self._reserved_totals.values())

    @property
    def p_to_d_debt_credits(self) -> float:
        return self.reserved_decode_credits

    @property
    def decode_residency(self) -> Mapping[str, float]:
        residency: dict[str, float] = {}
        for item in self._temporal_reservations():
            residency[item.logical_request_id] = max(
                residency.get(item.logical_request_id, -math.inf),
                item.releases_at_work,
            )
        return residency

    def _temporal_reservations(self) -> tuple[DecodeReservation, ...]:
        return tuple(
            item
            for reservations in self._active.values()
            for item in reservations
        ) + tuple(self._executing.values())

    def reserve(
        self,
        logical_request_id: str,
        attempt_index: int,
        credits: float,
        *,
        now_work: float,
    ) -> DecodeReservation:
        self._validate_request(credits, now_work)
        key = (logical_request_id, attempt_index)
        if key in self._active or key in self._executing:
            raise ValueError("duplicate decode reservation")
        if credits > 0 and self.capacity_credits == 0:
            raise ValueError("positive decode reservation requires positive capacity")
        remaining = credits
        cursor = now_work
        chunks: list[DecodeReservation] = []
        while remaining > 0:
            chunk_credits = min(remaining, self.capacity_credits)
            starts = self.available_at(chunk_credits, now_work=cursor)
            releases = starts + max(chunk_credits, 1e-9)
            existing = self._temporal_reservations() + tuple(chunks)
            breakpoints = (starts,) + tuple(
                item.starts_at_work
                for item in existing
                if starts < item.starts_at_work < releases
            )
            occupied = max(
                sum(
                    item.credits
                    for item in existing
                    if item.starts_at_work <= point < item.releases_at_work
                )
                for point in breakpoints
            )
            self.peak_reserved_decode_credits = max(
                self.peak_reserved_decode_credits, occupied + chunk_credits
            )
            reservation = DecodeReservation(
                logical_request_id,
                attempt_index,
                chunk_credits,
                starts,
                releases,
            )
            chunks.append(reservation)
            remaining -= chunk_credits
            cursor = reservation.releases_at_work
        if not chunks:
            chunks.append(
                DecodeReservation(
                    logical_request_id,
                    attempt_index,
                    0.0,
                    now_work,
                    now_work + 1e-9,
                )
            )
        self._active[key] = tuple(chunks)
        self._reserved_totals[key] = credits
        return chunks[0]

    def available_at(
        self,
        credits: float,
        *,
        now_work: float,
        exclude: tuple[str, int] | None = None,
    ) -> float:
        self._validate_request(credits, now_work)
        if credits > 0 and self.capacity_credits == 0:
            raise ValueError("positive decode reservation requires positive capacity")
        width = min(credits, self.capacity_credits)
        existing = tuple(
            item
            for key, reservations in self._active.items()
            if key != exclude
            for item in reservations
        ) + tuple(
            item
            for key, item in self._executing.items()
            if key != exclude
        )
        candidates = (now_work,) + tuple(
            sorted(
                {
                    item.releases_at_work
                    for item in existing
                    if item.releases_at_work > now_work
                }
            )
        )
        duration = max(width, 1e-9)
        for candidate in candidates:
            releases = candidate + duration
            breakpoints = (candidate,) + tuple(
                item.starts_at_work
                for item in existing
                if candidate < item.starts_at_work < releases
            )
            if all(
                sum(
                    item.credits
                    for item in existing
                    if item.starts_at_work <= point < item.releases_at_work
                )
                + width
                <= self.capacity_credits
                for point in breakpoints
            ):
                return candidate
        raise AssertionError("finite reservations must eventually release")

    @staticmethod
    def _validate_request(credits: float, now_work: float) -> None:
        if credits < 0 or not math.isfinite(credits):
            raise ValueError("decode credits must be finite and non-negative")
        if not math.isfinite(now_work):
            raise ValueError("reservation time must be finite")

    def activate(
        self,
        logical_request_id: str,
        attempt_index: int,
        credits: float,
        *,
        starts_at_work: float,
        finishes_at_work: float,
    ) -> None:
        self._validate_request(credits, starts_at_work)
        if not math.isfinite(finishes_at_work) or finishes_at_work < starts_at_work:
            raise ValueError("decode finish must be finite and not precede start")
        key = (logical_request_id, attempt_index)
        if key in self._executing:
            raise ValueError("duplicate decode activation")
        execution = DecodeReservation(
            logical_request_id,
            attempt_index,
            min(credits, self.capacity_credits),
            starts_at_work,
            finishes_at_work,
        )
        executing = tuple(self._executing.values())
        breakpoints = (starts_at_work,) + tuple(
            item.starts_at_work
            for item in executing
            if starts_at_work < item.starts_at_work < finishes_at_work
        )
        occupied = max(
            sum(
                item.credits
                for item in executing
                if item.starts_at_work <= point < item.releases_at_work
            )
            for point in breakpoints
        )
        if occupied + execution.credits > self.capacity_credits:
            raise ValueError("decode activation exceeds shared capacity")

        current = self._active.get(key)
        if current is None:
            self.reserve(
                logical_request_id,
                attempt_index,
                credits,
                now_work=starts_at_work,
            )
        self._active.pop(key)
        invalidated = tuple(
            other_key
            for other_key, reservations in self._active.items()
            if any(
                item.starts_at_work < finishes_at_work
                and starts_at_work < item.releases_at_work
                for item in reservations
            )
        )
        for other_key in invalidated:
            # Keep its full debt in _reserved_totals.  The kernel rechecks the
            # causal fence before Decode starts; activate() will then rebuild a
            # non-overlapping temporal reservation at the delayed start.
            self._active.pop(other_key)
        self._executing[key] = execution
        self.peak_reserved_decode_credits = max(
            self.peak_reserved_decode_credits,
            occupied + execution.credits,
        )

    def settle(
        self,
        logical_request_id: str,
        attempt_index: int,
        *,
        actual_decode_work: float,
    ) -> None:
        key = (logical_request_id, attempt_index)
        reservations = self._active.pop(key, ())
        execution = self._executing.pop(key, None)
        if not reservations and execution is None:
            raise KeyError(key)
        reserved_total = self._reserved_totals.pop(
            key,
            sum(item.credits for item in reservations)
            + (execution.credits if execution is not None else 0.0),
        )
        self.actual_decode_work += actual_decode_work
        self.reconciled_credit_error += actual_decode_work - reserved_total


@dataclass(frozen=True, slots=True)
class DecodeAdmissionDecision:
    logical_request_id: str
    mode: DecodeAdmissionMode
    predicted_output_tokens: int
    reserved_decode_credits: float
    action: AdmissionAction
    deferred_until_work: float


@dataclass(frozen=True, slots=True)
class DecodeRunReport:
    mode: DecodeAdmissionMode
    offered_logical_requests: int
    useful_output_tokens: int
    decode_work: float
    wasted_decode_work: float
    preemptions: int
    strict_output_goodput: float
    minimum_tier_slo_attainment: float
    jain_fairness: float
    p_to_d_debt_credits: float
    actual_decode_work: float
    offered_tokens: int
    strict_combined_goodput: float
    per_tier_slo_attainment: Mapping[str, float]
    issued_decode_work: float
    aborted_attempts: int
    credit_reconciliation_credits: float


class DecodeCapacityPolicy(KernelPolicy):
    """Admission and lease layer; the kernel retains cost and token truth."""

    def __init__(
        self,
        placement: M12PlacementPolicy,
        config: DecodeAdmissionConfig,
        *,
        predictor: PrefixFamilyPredictor | None = None,
    ) -> None:
        self.placement = placement
        self.config = config
        self.predictor = predictor or PrefixFamilyPredictor(default_output_tokens=1)
        self.decisions: list[DecodeAdmissionDecision] = []
        self.ledger = DecodeCreditLedger(config.capacity_credits)
        self._decode_fences: dict[tuple[str, int], float] = {}
        self._credits: dict[tuple[str, int], float] = {}
        self._aborted: set[str] = set()
        self._preemptions = 0

    def plan_attempts(
        self, request: KernelRequestSpec, view: CausalView
    ) -> tuple[AttemptExecutionSpec, ...]:
        attempt = tuple(self.placement.plan_attempts(request, view))[0]
        predicted = self._predict(request, view.now_work)
        credit = predicted * self.placement.cost_model.decode_work_per_token
        key = (request.logical.logical_request_id, attempt.attempt_index)
        self._credits[key] = credit
        reservation_start = view.now_work
        if self.config.mode is not DecodeAdmissionMode.NO_GATE:
            reservation_start = self.ledger.available_at(
                credit, now_work=view.now_work
            )
            if self.config.congestion_action is AdmissionAction.GATED_DP:
                reservation_start = self.ledger.reserve(
                    request.logical.logical_request_id,
                    attempt.attempt_index,
                    credit,
                    now_work=view.now_work,
                ).starts_at_work
        d_ready = view.decode_available_at[attempt.d_node_id]
        congested = reservation_start > view.now_work or d_ready > view.now_work
        action = (
            self.config.congestion_action
            if congested and self.config.mode is not DecodeAdmissionMode.NO_GATE
            else AdmissionAction.ADMIT
        )
        deferred_until = max(attempt.arrival_work, d_ready, reservation_start)
        if action in (AdmissionAction.DEFER, AdmissionAction.GATED_DP):
            attempt = replace(attempt, arrival_work=deferred_until)
        elif action is AdmissionAction.GATED_PD:
            self._decode_fences[
                (attempt.logical_request_id, attempt.attempt_index)
            ] = reservation_start
        self.decisions.append(
            DecodeAdmissionDecision(
                request.logical.logical_request_id,
                self.config.mode,
                predicted,
                credit,
                action,
                deferred_until,
            )
        )
        return (attempt,)

    def admission_event(
        self, request: KernelRequestSpec, attempt: AttemptExecutionSpec
    ) -> str | None:
        del request
        decision = next(
            (
                item
                for item in reversed(self.decisions)
                if item.logical_request_id == attempt.logical_request_id
            ),
            None,
        )
        if decision is None:
            return "ADMISSION_RETRY"
        return f"ADMISSION_{decision.action.value}"

    def decode_not_before(
        self,
        request: KernelRequestSpec,
        attempt: AttemptExecutionSpec,
        view: CausalView,
    ) -> float:
        key = (attempt.logical_request_id, attempt.attempt_index)
        fence = max(
            view.now_work,
            self._decode_fences.get(key, view.now_work),
        )
        if self.config.mode is not DecodeAdmissionMode.NO_GATE:
            fence = max(
                fence,
                self.ledger.available_at(
                    self._credits[key],
                    now_work=view.now_work,
                    exclude=key,
                ),
            )
        if (
            self.config.mode is not DecodeAdmissionMode.NO_GATE
            and self.config.congestion_action is AdmissionAction.GATED_PD
            and key not in self.ledger._active
        ):
            reservation = self.ledger.reserve(
                request.logical.logical_request_id,
                attempt.attempt_index,
                self._credits[key],
                now_work=view.now_work,
            )
            fence = max(fence, reservation.starts_at_work)
        return fence

    def decode_started(
        self,
        request: KernelRequestSpec,
        attempt: AttemptExecutionSpec,
        view: CausalView,
        *,
        finish_work: float,
    ) -> None:
        if self.config.mode is DecodeAdmissionMode.NO_GATE:
            return
        key = (attempt.logical_request_id, attempt.attempt_index)
        self.ledger.activate(
            request.logical.logical_request_id,
            attempt.attempt_index,
            self._credits[key],
            starts_at_work=view.now_work,
            finishes_at_work=finish_work,
        )

    def decode_lease_tokens(
        self,
        request: KernelRequestSpec,
        attempt: AttemptExecutionSpec,
        view: CausalView,
    ) -> int | None:
        del request, view
        if (
            attempt.attempt_index == 0
            and self.config.lease_boundary_tokens < attempt.emitted_output_tokens
        ):
            return self.config.lease_boundary_tokens
        return None

    def reprice_decode(
        self,
        request: KernelRequestSpec,
        attempt: AttemptExecutionSpec,
        view: CausalView,
    ) -> AttemptExecutionSpec:
        fence = self.config.abort_fences.get(request.logical.logical_request_id)
        if (
            attempt.attempt_index == 0
            and request.logical.logical_request_id not in self._aborted
            and fence is not None
            and fence.allows_abort
            and view.retry_budget_remaining > 0
        ):
            self._aborted.add(request.logical.logical_request_id)
            self._preemptions += 1
            return replace(
                attempt,
                emitted_output_tokens=min(
                    attempt.emitted_output_tokens, self.config.lease_boundary_tokens
                ),
                terminal=AttemptTerminal.ABORTED,
            )
        return attempt

    def attempt_finished(
        self,
        request: KernelRequestSpec,
        attempt: AttemptExecutionSpec,
        view: CausalView,
        *,
        actual_decode_work: float,
    ) -> None:
        del request, view
        if self.config.mode is DecodeAdmissionMode.NO_GATE:
            self.ledger.actual_decode_work += actual_decode_work
        else:
            self.ledger.settle(
                attempt.logical_request_id,
                attempt.attempt_index,
                actual_decode_work=actual_decode_work,
            )

    def cache_mutation(
        self,
        request: KernelRequestSpec,
        attempt: AttemptExecutionSpec,
        view: CausalView,
    ) -> CacheMutation:
        if (
            self.config.mode is DecodeAdmissionMode.CAUSAL
            and attempt.terminal is AttemptTerminal.COMPLETED
        ):
            self.predictor.observe(request, completed_at_work=view.now_work)
        return self.placement.cache_mutation(request, attempt, view)

    def next_attempt(
        self,
        request: KernelRequestSpec,
        previous: AttemptExecutionSpec,
        view: CausalView,
    ) -> AttemptExecutionSpec | None:
        delegated = self.placement.next_attempt(request, previous, view)
        if delegated is not None:
            return delegated
        if previous.terminal is not AttemptTerminal.ABORTED:
            return None
        retry = replace(
            previous,
            attempt_index=previous.attempt_index + 1,
            arrival_work=view.now_work,
            emitted_output_tokens=request.logical.true_output_tokens,
            terminal=AttemptTerminal.COMPLETED,
        )
        if self.config.mode is not DecodeAdmissionMode.NO_GATE:
            predicted = self._predict(request, view.now_work)
            credit = predicted * self.placement.cost_model.decode_work_per_token
            key = (retry.logical_request_id, retry.attempt_index)
            self._credits[key] = credit
            available = self.ledger.available_at(credit, now_work=view.now_work)
            if self.config.congestion_action is AdmissionAction.GATED_DP:
                available = self.ledger.reserve(
                    request.logical.logical_request_id,
                    retry.attempt_index,
                    credit,
                    now_work=view.now_work,
                ).starts_at_work
            if self.config.congestion_action in (
                AdmissionAction.DEFER,
                AdmissionAction.GATED_DP,
            ):
                retry = replace(retry, arrival_work=available)
        return retry

    def summarize(self, result: KernelResult) -> DecodeRunReport:
        metrics = result.metrics
        outcomes = tuple(attempt.outcome for attempt in result.attempts)
        return DecodeRunReport(
            self.config.mode,
            metrics.offered_logical_requests,
            metrics.strict_useful_output_tokens,
            metrics.decode_gpu_work,
            sum(outcome.wasted_decode_work for outcome in outcomes),
            self._preemptions,
            metrics.strict_useful_output_token_goodput,
            metrics.minimum_tier_slo_attainment,
            metrics.jain_fairness,
            self.ledger.p_to_d_debt_credits,
            self.ledger.actual_decode_work,
            metrics.offered_input_tokens + metrics.offered_output_tokens,
            metrics.strict_useful_token_goodput,
            metrics.per_tier_slo_attainment,
            sum(outcome.decode_gpu_work for outcome in outcomes),
            sum(outcome.wasted_decode_work > 0 for outcome in outcomes),
            self.ledger.p_to_d_debt_credits,
        )

    def _predict(self, request: KernelRequestSpec, at_work: float) -> int:
        if self.config.mode is DecodeAdmissionMode.NO_GATE:
            return 0
        if self.config.mode is DecodeAdmissionMode.ORACLE:
            return request.logical.true_output_tokens
        if self.config.mode is DecodeAdmissionMode.ORACLE_NOISED:
            return max(
                0,
                round(
                    request.logical.true_output_tokens
                    * self.config.oracle_noise_multiplier
                ),
            )
        return self.predictor.predict(request, at_work=at_work)


@dataclass(frozen=True, slots=True)
class G12ThreeVerdict:
    passed: bool
    relative_goodput_gain: float
    offered_load_unchanged: bool
    fairness_preserved: bool
    deployable_conclusion: bool
    conclusion: str
    output_goodput_pass: bool
    combined_goodput_pass: bool
    work_accounted: bool


def evaluate_g12_3(
    *,
    no_gate: DecodeRunReport,
    candidate: DecodeRunReport,
    arrival_scale: float,
) -> G12ThreeVerdict:
    """Evaluate the fixed 1.5x overload gate; only CAUSAL can deploy."""
    if arrival_scale != 1.5:
        raise ValueError("G12-3 is defined only for the 1.5x overload workload")
    if min(no_gate.strict_output_goodput, candidate.strict_output_goodput) < 0:
        raise ValueError("goodput must be non-negative")
    gain = _relative_gain(
        no_gate.strict_output_goodput, candidate.strict_output_goodput
    )
    combined_gain = _relative_gain(
        no_gate.strict_combined_goodput, candidate.strict_combined_goodput
    )
    offered_same = (
        no_gate.offered_logical_requests == candidate.offered_logical_requests
        and no_gate.offered_tokens == candidate.offered_tokens
    )
    same_tiers = set(no_gate.per_tier_slo_attainment) == set(
        candidate.per_tier_slo_attainment
    )
    tier_floor = same_tiers and all(
        value >= 0.80 for value in candidate.per_tier_slo_attainment.values()
    )
    tier_regression = same_tiers and all(
        baseline == 0
        or (baseline - candidate.per_tier_slo_attainment[tier]) / baseline <= 0.02
        for tier, baseline in no_gate.per_tier_slo_attainment.items()
    )
    fairness = (
        candidate.minimum_tier_slo_attainment >= 0.80
        and candidate.jain_fairness >= 0.90
        and tier_floor
        and tier_regression
    )
    labels_valid = (
        no_gate.mode is DecodeAdmissionMode.NO_GATE
        and candidate.mode is not DecodeAdmissionMode.NO_GATE
    )
    work_accounted = (
        _has_complete_work_evidence(no_gate)
        and _has_complete_work_evidence(candidate)
    )
    passed = (
        gain >= 0.05
        and combined_gain >= 0.05
        and offered_same
        and fairness
        and work_accounted
        and labels_valid
    )
    deployable = passed and candidate.mode is DecodeAdmissionMode.CAUSAL
    conclusion = (
        "DEPLOYABLE_CAUSAL"
        if deployable
        else "SENSITIVITY_ONLY"
        if passed
        else "OVERLOAD_PROTECTION_ONLY"
    )
    return G12ThreeVerdict(
        passed,
        gain,
        offered_same,
        fairness,
        deployable,
        conclusion,
        gain >= 0.05,
        combined_gain >= 0.05,
        work_accounted,
    )


def _relative_gain(baseline: float, candidate: float) -> float:
    if baseline:
        return (candidate - baseline) / baseline
    return math.inf if candidate > 0 else 0.0


def _has_complete_work_evidence(report: DecodeRunReport) -> bool:
    return (
        math.isclose(report.decode_work, report.issued_decode_work)
        and math.isclose(report.decode_work, report.actual_decode_work)
        and 0 <= report.wasted_decode_work <= report.decode_work
        and report.credit_reconciliation_credits == 0
        and report.preemptions <= report.aborted_attempts
        and (report.preemptions == 0 or report.wasted_decode_work > 0)
    )
