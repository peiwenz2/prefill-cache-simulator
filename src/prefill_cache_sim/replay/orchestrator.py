"""Replay the modeled 4 P+D pool across arms and arrival scales.

The orchestrator does not implement a simulator. It builds
:class:`~..experiment.ExperimentCase` values, hands them to the existing
selector, eviction and simulator APIs, and then re-reads the resulting decision
log through the three-source reconciler so that every published cell has to
survive the same join a real deployment would.

Honesty note: the bundle produced here is labelled
:data:`~.sources.TruthBasis.MODELED_SIMULATOR` and its ``tpot_work`` is ``None``
with :data:`~.sources.TPOT_UNMODELED_REASON`. The simulator models prefill
placement only; inventing a decode cost to fill that column would turn a missing
observation into a fabricated one. Timings stay in normalized work units
throughout, which is why the outcome may never carry
:data:`~..calibration.TimeUnit.MILLISECONDS`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, replace
from enum import StrEnum
from typing import Any

from ..calibration import (
    CalibrationStatus,
    EvidenceTier,
    MachineProvenance,
    TimeUnit,
    assert_labels_supported,
)
from ..domain import CacheVisibility, ReplayMode, Request, ViewMode
from ..experiment import (
    ExperimentCase,
    ExperimentResult,
    eviction_factory_for,
    result_from_report,
    selector_for,
)
from ..simulator import LocalSimulator, SimulationConfig, SimulationReport
from .ranking import RankingComparison, UndefinedRankingError
from .reconcile import ReconciliationReport, reconcile
from .shadow import ShadowDecision, ShadowGate
from .sources import (
    SOURCE_SCHEMA_VERSION,
    TPOT_UNMODELED_REASON,
    AttemptKey,
    AttemptTraceRecord,
    ClientLatencyRecord,
    EngineHitRecord,
    SourceBundle,
    TruthBasis,
)

REPLAY_SCHEMA_VERSION = "m10-replay-v1"

#: The two arrival scalings M10 replays: the trace as recorded, and twice as
#: fast. Nothing here claims those correspond to any measured request rate.
ARRIVAL_SCALES: tuple[float, ...] = (1.0, 2.0)

BASELINE_ARM_ID = "S0_RANDOM"
M4_WINNER_ARM_ID = "S4_SESSION_AFFINITY"

#: Metrics a caller may rank arms by. The list is closed so that a typo becomes
#: an error rather than a silently missing column.
SCORE_METRICS: tuple[str, ...] = (
    "token_weighted_hit_rate",
    "block_ref_hit_rate",
    "logical_token_weighted_hit_rate",
    "logical_block_ref_hit_rate",
)


class ArmRole(StrEnum):
    """Why an arm is in the matrix.

    ``STOP_GATED`` records that an arm is replayed for comparison even though a
    prior milestone recommended against shipping it; dropping it would quietly
    remove the evidence for that recommendation.
    """

    BASELINE = "BASELINE"
    CANDIDATE = "CANDIDATE"
    STOP_GATED = "STOP_GATED"
    M4_WINNER = "M4_WINNER"


@dataclass(frozen=True, slots=True)
class ReplayArm:
    """One policy configuration under comparison."""

    arm_id: str
    selector_id: str
    eviction_id: str
    role: ArmRole

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "selector_id": self.selector_id,
            "eviction_id": self.eviction_id,
            "role": self.role.value,
        }


#: Baseline, the two candidates M10 was asked to compare, and the M4 winner.
DEFAULT_ARMS: tuple[ReplayArm, ...] = (
    ReplayArm(BASELINE_ARM_ID, "S0_RANDOM", "E1_LRU", ArmRole.BASELINE),
    ReplayArm(
        "S3_GB_PREFIX_BUCKET", "S3_GB_PREFIX_BUCKET", "E1_LRU", ArmRole.CANDIDATE
    ),
    ReplayArm(
        "S5_CENTRALIZED_MASTER_TTFT",
        "S5_CENTRALIZED_MASTER_TTFT",
        "E1_LRU",
        ArmRole.STOP_GATED,
    ),
    ReplayArm(M4_WINNER_ARM_ID, "S4_SESSION_AFFINITY", "E1_LRU", ArmRole.M4_WINNER),
)


def case_fingerprint(case: ExperimentCase) -> str:
    """Hash every field of a case so a changed knob changes the identity."""
    payload = json.dumps(
        {field.name: str(getattr(case, field.name)) for field in fields(case)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ReplayPlan:
    """The arm x arrival-scale matrix, plus the shared topology it runs on."""

    node_count: int = 4
    capacity_blocks: int = 256
    arms: tuple[ReplayArm, ...] = DEFAULT_ARMS
    arrival_scales: tuple[float, ...] = ARRIVAL_SCALES
    seed: int = 20261010
    prefill_uncached_token_ms: float = 1.0
    prefill_concurrency_per_node: int = 1

    def __post_init__(self) -> None:
        if not self.arms:
            raise ValueError("a replay plan needs at least one arm")
        if len({arm.arm_id for arm in self.arms}) != len(self.arms):
            raise ValueError("replay arm ids must be unique")
        if not self.arrival_scales:
            raise ValueError("a replay plan needs at least one arrival scale")
        if any(scale <= 0 for scale in self.arrival_scales):
            raise ValueError("arrival scales must be positive")

    def cases(self) -> tuple[ExperimentCase, ...]:
        """Expand to one case per (arm, arrival scale), validating each arm.

        Selector and eviction ids are resolved here so that an unknown arm fails
        at plan expansion rather than midway through a replay.
        """
        cases: list[ExperimentCase] = []
        for arm in self.arms:
            for scale in self.arrival_scales:
                case = ExperimentCase(
                    f"M10-{arm.arm_id}-x{scale:g}",
                    arm.selector_id,
                    arm.eviction_id,
                    self.node_count,
                    "FIXED_PER_NODE",
                    self.capacity_blocks,
                    ReplayMode.QUEUED,
                    CacheVisibility.INSERT_AT_COMPLETION,
                    ViewMode.ORACLE,
                    0.0,
                    0.0,
                    scale,
                    self.seed,
                    prefill_uncached_token_ms=self.prefill_uncached_token_ms,
                    prefill_concurrency_per_node=self.prefill_concurrency_per_node,
                )
                selector_for(case)
                eviction_factory_for(case)
                cases.append(case)
        return tuple(cases)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_count": self.node_count,
            "capacity_blocks": self.capacity_blocks,
            "arms": [arm.to_dict() for arm in self.arms],
            "arrival_scales": list(self.arrival_scales),
            "seed": self.seed,
            "prefill_uncached_token_ms": self.prefill_uncached_token_ms,
            "prefill_concurrency_per_node": self.prefill_concurrency_per_node,
        }


def bundle_from_report(
    report: SimulationReport, requests: tuple[Request, ...]
) -> SourceBundle:
    """Split one simulation into the three observations M10 reconciles.

    Every source is derived from the same decision log, so a clean ledger here
    proves the join keys are stable rather than proving the sources agree about
    the world. ``output_tokens`` is joined from the request set because the
    decision log does not carry it.

    The client's ``ttft_work`` spans ``arrival`` to ``finish``. The scheduler's
    ``start_ms`` is not something a client can observe, so subtracting it would
    quietly hand the client the scheduler's view and hide every queueing delay
    the arrival-rate sweep exists to measure.
    """
    output_tokens = {request.request_id: request.output_tokens for request in requests}
    engine_hits: list[EngineHitRecord] = []
    client_latencies: list[ClientLatencyRecord] = []
    attempt_traces: list[AttemptTraceRecord] = []
    for decision in report.decisions:
        key = AttemptKey(decision.logical_request_id, decision.attempt_index)
        engine_hits.append(
            EngineHitRecord(
                key, decision.node_id, decision.input_tokens, decision.hit_tokens
            )
        )
        client_latencies.append(
            ClientLatencyRecord(
                key,
                decision.input_tokens,
                output_tokens.get(decision.request_id, 0),
                decision.finish_ms - decision.arrival_ms,
                None,
            )
        )
        attempt_traces.append(
            AttemptTraceRecord(
                key,
                decision.node_id,
                decision.input_tokens,
                decision.arrival_ms,
                decision.start_ms,
                decision.finish_ms,
            )
        )
    return SourceBundle(
        SOURCE_SCHEMA_VERSION,
        TruthBasis.MODELED_SIMULATOR,
        MachineProvenance.unknown(),
        TPOT_UNMODELED_REASON,
        tuple(engine_hits),
        tuple(client_latencies),
        tuple(attempt_traces),
    )


@dataclass(frozen=True, slots=True)
class ReplayCell:
    """One arm replayed at one arrival scale, with its reconciliation.

    ``bundle`` is the three-source input the reconciliation was computed from.
    It is kept in memory so a caller can inspect what was joined, but it is left
    out of :meth:`to_dict` along with the per-attempt reconciled rows: neither
    is evidence, both scale with the trace, and a full-trace replay would
    otherwise serialize hundreds of thousands of rows the CSV artifacts already
    carry. The ledger and the join rates survive via
    :meth:`ReconciliationReport.summary`.
    """

    arm_id: str
    arm_role: ArmRole
    arrival_scale: float
    case_fingerprint: str
    result: ExperimentResult
    bundle: SourceBundle
    reconciliation: ReconciliationReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "arm_role": self.arm_role.value,
            "arrival_scale": self.arrival_scale,
            "case_fingerprint": self.case_fingerprint,
            "result": self.result.to_dict(),
            "reconciliation": self.reconciliation.summary(),
        }


@dataclass(frozen=True, slots=True)
class ReplayOutcome:
    """Every cell of one replay, plus the labels it is allowed to carry."""

    schema_version: str
    calibration_status: CalibrationStatus
    time_unit: TimeUnit
    evidence_tier: EvidenceTier
    machine: MachineProvenance
    plan: ReplayPlan
    cells: tuple[ReplayCell, ...]
    trace_sha256: str
    git_sha: str | None
    git_dirty: bool | None

    def cell(self, arm_id: str, arrival_scale: float) -> ReplayCell:
        for cell in self.cells:
            if cell.arm_id == arm_id and cell.arrival_scale == arrival_scale:
                return cell
        raise KeyError(f"no cell for arm {arm_id} at arrival scale {arrival_scale}")

    def scores(self, metric: str, arrival_scale: float) -> dict[str, float]:
        """Return ``{arm_id: metric}`` for one arrival scale."""
        if metric not in SCORE_METRICS:
            raise ValueError(f"unknown metric {metric!r}")
        return {
            cell.arm_id: float(getattr(cell.result, metric))
            for cell in self.cells
            if cell.arrival_scale == arrival_scale
        }

    def ranking_comparison(self, metric: str) -> RankingComparison | None:
        """Compare the arm ranking across the first two arrival scales.

        Returns ``None`` when the plan has a single scale, or when one side is
        fully tied: an undefined statistic is reported as absent rather than as
        agreement.

        Only :class:`~.ranking.UndefinedRankingError` is caught. A corrupt score
        raises :class:`~.payload.MalformedPayloadError`, which is also a
        ``ValueError``; swallowing that as well would publish "no ranking
        evidence" for what is really unreadable input, turning a defect into an
        honest-looking absence.
        """
        if len(self.plan.arrival_scales) < 2:
            return None
        left, right = self.plan.arrival_scales[0], self.plan.arrival_scales[1]
        try:
            return RankingComparison.from_scores(
                f"arrival_x{left:g}",
                f"arrival_x{right:g}",
                self.scores(metric, left),
                self.scores(metric, right),
            )
        except UndefinedRankingError:
            return None

    def _tau_b(self, metric: str) -> float | None:
        comparison = self.ranking_comparison(metric)
        return None if comparison is None else comparison.tau_b

    def shadow_decisions(
        self, metric: str, gate: ShadowGate | None = None
    ) -> tuple[ShadowDecision, ...]:
        """Record, for each non-baseline arm, what the gate would have said.

        Nothing acts on the result: see :mod:`.shadow`.
        """
        gate = gate or ShadowGate()
        primary = self.plan.arrival_scales[0]
        scores = self.scores(metric, primary)
        tau_b = self._tau_b(metric)
        decisions: list[ShadowDecision] = []
        for arm in self.plan.arms:
            if arm.arm_id == BASELINE_ARM_ID:
                continue
            cells = [cell for cell in self.cells if cell.arm_id == arm.arm_id]
            decisions.append(
                gate.decide(
                    baseline_arm_id=BASELINE_ARM_ID,
                    candidate_arm_id=arm.arm_id,
                    reconciled_fraction=min(
                        cell.reconciliation.reconciled_fraction for cell in cells
                    ),
                    disagreement_fraction=max(
                        cell.reconciliation.disagreement_fraction for cell in cells
                    ),
                    tau_b=tau_b,
                    baseline_score=scores[BASELINE_ARM_ID],
                    candidate_score=scores[arm.arm_id],
                )
            )
        return tuple(decisions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "calibration_status": self.calibration_status.value,
            "time_unit": self.time_unit.value,
            "evidence_tier": self.evidence_tier.value,
            "machine": self.machine.to_dict(),
            "plan": self.plan.to_dict(),
            "cells": [cell.to_dict() for cell in self.cells],
            "trace_sha256": self.trace_sha256,
            "git_sha": self.git_sha,
            "git_dirty": self.git_dirty,
        }


def _run_one(
    case: ExperimentCase,
    requests: tuple[Request, ...],
    *,
    trace_sha256: str,
    git_sha: str | None,
    git_dirty: bool | None,
) -> tuple[ExperimentResult, SourceBundle, ReconciliationReport]:
    """Run one case and reconcile its own decision log.

    This mirrors :func:`~..experiment.run_case` rather than calling it because
    the reconciler needs the :class:`~..simulator.SimulationReport`, which
    ``run_case`` discards.
    """
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
    result = result_from_report(
        case,
        report,
        trace_sha256=trace_sha256,
        config_sha256=case_fingerprint(case),
        git_sha=git_sha,
        git_dirty=git_dirty,
    )
    bundle = bundle_from_report(report, replay_requests)
    return result, bundle, reconcile(bundle)


def run_replay(
    plan: ReplayPlan,
    requests: tuple[Request, ...],
    *,
    trace_sha256: str,
    git_sha: str | None,
    git_dirty: bool | None,
    evidence_tier: EvidenceTier = EvidenceTier.SYNTHETIC_REPLAY,
    machine: MachineProvenance | None = None,
) -> ReplayOutcome:
    """Replay every plan cell and label the outcome no harder than it earns."""
    machine = machine or MachineProvenance.unknown()
    calibration_status = CalibrationStatus.SYNTHETIC_UNCALIBRATED
    time_unit = TimeUnit.NORMALIZED_WORK
    assert_labels_supported(machine, calibration_status, time_unit, evidence_tier)

    # ``cases()`` nests arrival scales inside arms, so repeating each arm once
    # per scale reproduces its order without parsing case ids back apart.
    arms = [arm for arm in plan.arms for _ in plan.arrival_scales]
    cells: list[ReplayCell] = []
    for arm, case in zip(arms, plan.cases(), strict=True):
        result, bundle, reconciliation = _run_one(
            case,
            requests,
            trace_sha256=trace_sha256,
            git_sha=git_sha,
            git_dirty=git_dirty,
        )
        cells.append(
            ReplayCell(
                arm.arm_id,
                arm.role,
                case.replay_speed,
                case_fingerprint(case),
                result,
                bundle,
                reconciliation,
            )
        )
    return ReplayOutcome(
        REPLAY_SCHEMA_VERSION,
        calibration_status,
        time_unit,
        evidence_tier,
        machine,
        plan,
        tuple(cells),
        trace_sha256,
        git_sha,
        git_dirty,
    )
