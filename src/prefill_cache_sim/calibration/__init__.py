"""M9 calibration harness.

Scope note: this package is HARNESS_ONLY. It defines the engine abstraction,
sweep grid, deterministic fit and versioned artifact needed to calibrate the
simulator, and it ships a mock endpoint so the whole path is testable without an
accelerator. It does not contain any hardware measurement. Real KVT
benchmarking stays deferred per the M5 *modelled* finding that, in normalized
work under the modelled capacity, the shared tier moves prefill cost by roughly
4.7% and the worst modelled KVT corner by roughly 1.03% -- neither of which is
enough to change the M6-M8 ranking. M5 measured no hardware either.
"""

from __future__ import annotations

from .endpoint import (
    EngineEndpoint,
    MockCurve,
    MockEngine,
    SweepKind,
    SweepObservation,
)
from .fingerprint import (
    DEPENDENCY_PATHS,
    REPRODUCIBILITY_CLAIM,
    REPRODUCIBILITY_NOTE,
    combined_digest,
    m9_source_paths,
    runtime_identity,
    source_fingerprints,
    source_manifest,
)
from .fit import (
    DISTRIBUTION_ASSUMPTION,
    MINIMUM_OBSERVATIONS,
    LinearFit,
    ResidualSummary,
    fit_sweep,
)
from .params import CALIBRATION_SCHEMA_VERSION, CalibrationParams
from .provenance import (
    HONEST_LABEL_TUPLES,
    CalibrationStatus,
    DishonestLabelError,
    EvidenceTier,
    MachineProvenance,
    TimeUnit,
    assert_labels_supported,
    honest_labels,
)
from .sweep import SweepSpec, run_sweep

__all__ = [
    "CALIBRATION_SCHEMA_VERSION",
    "DEPENDENCY_PATHS",
    "DISTRIBUTION_ASSUMPTION",
    "HONEST_LABEL_TUPLES",
    "MINIMUM_OBSERVATIONS",
    "REPRODUCIBILITY_CLAIM",
    "REPRODUCIBILITY_NOTE",
    "CalibrationParams",
    "CalibrationStatus",
    "DishonestLabelError",
    "EngineEndpoint",
    "EvidenceTier",
    "LinearFit",
    "MachineProvenance",
    "MockCurve",
    "MockEngine",
    "ResidualSummary",
    "SweepKind",
    "SweepObservation",
    "SweepSpec",
    "TimeUnit",
    "assert_labels_supported",
    "combined_digest",
    "fit_sweep",
    "honest_labels",
    "m9_source_paths",
    "run_sweep",
    "runtime_identity",
    "source_fingerprints",
    "source_manifest",
]
