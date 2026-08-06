"""M10 replay harness: replay arms, reconcile sources, record shadow decisions.

The package is a *harness*, not a measurement. Everything it produces is derived
from the modeled simulator or from a synthetic fixture, so nothing it exports may
be labelled :data:`~..calibration.EvidenceTier.HW_VALIDATED` or timed in
:data:`~..calibration.TimeUnit.MILLISECONDS`. The modules divide the work:

``payload``
    Typed accessors that reject a malformed payload at the deserialization
    boundary instead of letting a wrong type become a fabricated finding.
``sources``
    The three observation streams and the join key that ties them together.
``reconcile``
    The join itself, plus the ledger of every way the sources fail to agree.
``faults``
    A synthetic bundle and known defects, so the reconciler can be falsified.
``ranking``
    The ranking-consistency statistic, frozen before any result was produced.
``shadow``
    Decisions that are recorded and never enforced.
``orchestrator``
    The arm x arrival-scale matrix that drives the existing simulator.
``fingerprint``
    SHA-256 digests of the sources that produced an artifact, because the
    recorded commit does not identify untracked code.
``hardware``
    The M10-HW gate. Everything above is reachable on a laptop; this module
    decides whether a particular run earned the right to say otherwise, and
    names the reasons it did not.
"""

from .faults import (
    FaultPlan,
    FieldPerturbation,
    InjectionResult,
    apply_faults,
    synthetic_bundle,
)
from .fingerprint import (
    REPRODUCIBILITY_CLAIM,
    REPRODUCIBILITY_NOTE,
    combined_digest,
    m10_source_paths,
    source_fingerprints,
    source_manifest,
)
from .hardware import (
    DEFAULT_REPLAY_HARDWARE_GATE,
    FROZEN_PLAN_DIGEST,
    REPLAY_BLOCKER_ORDER,
    REPLAY_HARDWARE_GATE_SCHEMA_VERSION,
    REPLAY_HARDWARE_SCHEMA_VERSION,
    REQUIRED_CALIBRATION_TIER,
    ReplayBlocker,
    ReplayHardwareEvidence,
    ReplayHardwareGate,
    ReplayHardwareReport,
    plan_digest,
    replay_blockers,
)
from .orchestrator import (
    ARRIVAL_SCALES,
    BASELINE_ARM_ID,
    DEFAULT_ARMS,
    M4_WINNER_ARM_ID,
    REPLAY_SCHEMA_VERSION,
    SCORE_METRICS,
    ArmRole,
    ReplayArm,
    ReplayCell,
    ReplayOutcome,
    ReplayPlan,
    bundle_from_report,
    case_fingerprint,
    run_replay,
)
from .payload import MalformedPayloadError
from .ranking import (
    FROZEN_RANKING_STATISTIC,
    RANKING_SCHEMA_VERSION,
    RankingComparison,
    UndefinedRankingError,
    concordance,
    kendall_tau_b,
    pairwise_winner_agreement,
)
from .reconcile import (
    ABSENT_VALUE,
    RECONCILE_SCHEMA_VERSION,
    LedgerEntry,
    LedgerKind,
    ReconciledRow,
    ReconciliationReport,
    disagreement_entries,
    reconcile,
)
from .shadow import (
    CANDIDATE_DOES_NOT_BEAT_BASELINE,
    DEFAULT_SHADOW_GATE,
    DISAGREEMENT_FRACTION_ABOVE_GATE,
    ENFORCEMENT_ENABLED,
    RANKING_CONSISTENCY_BELOW_GATE,
    RANKING_CONSISTENCY_UNAVAILABLE,
    RECONCILED_FRACTION_BELOW_GATE,
    SHADOW_GATE_SCHEMA_VERSION,
    SHADOW_REASON_ORDER,
    SHADOW_SCHEMA_VERSION,
    ShadowDecision,
    ShadowEnforcementError,
    ShadowGate,
    ShadowOutcome,
    gate_reasons,
    outcome_for,
)
from .sources import (
    SHARED_FIELDS,
    SOURCE_ORDER,
    SOURCE_SCHEMA_VERSION,
    TPOT_UNMODELED_REASON,
    AttemptKey,
    AttemptTraceRecord,
    ClientLatencyRecord,
    EngineHitRecord,
    SharedField,
    SourceBundle,
    SourceName,
    TruthBasis,
)

__all__ = [
    "ABSENT_VALUE",
    "ARRIVAL_SCALES",
    "BASELINE_ARM_ID",
    "CANDIDATE_DOES_NOT_BEAT_BASELINE",
    "DEFAULT_ARMS",
    "DEFAULT_REPLAY_HARDWARE_GATE",
    "DEFAULT_SHADOW_GATE",
    "DISAGREEMENT_FRACTION_ABOVE_GATE",
    "ENFORCEMENT_ENABLED",
    "FROZEN_PLAN_DIGEST",
    "FROZEN_RANKING_STATISTIC",
    "M4_WINNER_ARM_ID",
    "RANKING_CONSISTENCY_BELOW_GATE",
    "RANKING_CONSISTENCY_UNAVAILABLE",
    "RANKING_SCHEMA_VERSION",
    "RECONCILED_FRACTION_BELOW_GATE",
    "RECONCILE_SCHEMA_VERSION",
    "REPLAY_BLOCKER_ORDER",
    "REPLAY_HARDWARE_GATE_SCHEMA_VERSION",
    "REPLAY_HARDWARE_SCHEMA_VERSION",
    "REPLAY_SCHEMA_VERSION",
    "REPRODUCIBILITY_CLAIM",
    "REPRODUCIBILITY_NOTE",
    "REQUIRED_CALIBRATION_TIER",
    "SCORE_METRICS",
    "SHADOW_GATE_SCHEMA_VERSION",
    "SHADOW_REASON_ORDER",
    "SHADOW_SCHEMA_VERSION",
    "SHARED_FIELDS",
    "SOURCE_ORDER",
    "SOURCE_SCHEMA_VERSION",
    "TPOT_UNMODELED_REASON",
    "ArmRole",
    "AttemptKey",
    "AttemptTraceRecord",
    "ClientLatencyRecord",
    "EngineHitRecord",
    "FaultPlan",
    "FieldPerturbation",
    "InjectionResult",
    "LedgerEntry",
    "LedgerKind",
    "MalformedPayloadError",
    "RankingComparison",
    "ReconciledRow",
    "ReconciliationReport",
    "ReplayArm",
    "ReplayBlocker",
    "ReplayCell",
    "ReplayHardwareEvidence",
    "ReplayHardwareGate",
    "ReplayHardwareReport",
    "ReplayOutcome",
    "ReplayPlan",
    "ShadowDecision",
    "ShadowEnforcementError",
    "ShadowGate",
    "ShadowOutcome",
    "SharedField",
    "SourceBundle",
    "SourceName",
    "TruthBasis",
    "UndefinedRankingError",
    "apply_faults",
    "bundle_from_report",
    "case_fingerprint",
    "combined_digest",
    "concordance",
    "disagreement_entries",
    "gate_reasons",
    "kendall_tau_b",
    "m10_source_paths",
    "outcome_for",
    "pairwise_winner_agreement",
    "plan_digest",
    "reconcile",
    "replay_blockers",
    "run_replay",
    "source_fingerprints",
    "source_manifest",
    "synthetic_bundle",
]
