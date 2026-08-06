"""M9 calibration harness.

Scope note: this package is HARNESS_ONLY. It defines the engine abstraction,
sweep grid, deterministic fit and versioned artifact needed to calibrate the
simulator, and it ships a mock endpoint so the whole path is testable without an
accelerator. It does not contain any hardware measurement. Real KVT
benchmarking stays deferred per the M5 *modelled* finding that, in normalized
work under the modelled capacity, the shared tier moves prefill cost by roughly
4.7% and the worst modelled KVT corner by roughly 1.03% -- neither of which is
enough to change the M6-M8 ranking. M5 measured no hardware either.

The M9-HW modules (:mod:`.transport`, :mod:`.http_endpoint`, :mod:`.hardware`)
change what is *possible*, not what has been *done*. They make a real engine
reachable and make the absence of one a machine-readable blocker
(:class:`~.hardware.HardwareBlocker`) rather than a comment. Running them in an
environment with no accelerator produces a report that says
:attr:`~.hardware.HardwareBlocker.BLOCKED_NO_ENGINE_ACCESS`, which is the honest
result -- not a failure to produce one.
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
from .hardware import (
    DEFAULT_HARDWARE_GATE,
    ENDPOINT_CONTRACT_VERSION,
    HARDWARE_BLOCKER_ORDER,
    HARDWARE_GATE_SCHEMA_VERSION,
    HARDWARE_SCHEMA_VERSION,
    HardwareBlocker,
    HardwareContext,
    HardwareEvidence,
    HardwareGate,
    HardwareGateReport,
    TrustPolicy,
    hardware_blockers,
    residual_ratio,
)
from .http_endpoint import (
    DESCRIBE_FIELDS,
    DESCRIBE_PATH,
    MEASURE_PATH,
    HttpJsonEngineEndpoint,
    endpoint_is_synthetic,
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
from .transport import (
    ALLOWED_URL_SCHEMES,
    MAX_RESPONSE_BYTES,
    EndpointError,
    EndpointProtocolError,
    EndpointTimeout,
    EndpointTransport,
    EndpointTransportError,
    HttpJsonTransport,
    StubTransport,
)

__all__ = [
    "ALLOWED_URL_SCHEMES",
    "CALIBRATION_SCHEMA_VERSION",
    "DEFAULT_HARDWARE_GATE",
    "DEPENDENCY_PATHS",
    "DESCRIBE_FIELDS",
    "DESCRIBE_PATH",
    "DISTRIBUTION_ASSUMPTION",
    "ENDPOINT_CONTRACT_VERSION",
    "HARDWARE_BLOCKER_ORDER",
    "HARDWARE_GATE_SCHEMA_VERSION",
    "HARDWARE_SCHEMA_VERSION",
    "HONEST_LABEL_TUPLES",
    "MAX_RESPONSE_BYTES",
    "MEASURE_PATH",
    "MINIMUM_OBSERVATIONS",
    "REPRODUCIBILITY_CLAIM",
    "REPRODUCIBILITY_NOTE",
    "CalibrationParams",
    "CalibrationStatus",
    "DishonestLabelError",
    "EndpointError",
    "EndpointProtocolError",
    "EndpointTimeout",
    "EndpointTransport",
    "EndpointTransportError",
    "EngineEndpoint",
    "EvidenceTier",
    "HardwareBlocker",
    "HardwareContext",
    "HardwareEvidence",
    "HardwareGate",
    "HardwareGateReport",
    "HttpJsonEngineEndpoint",
    "HttpJsonTransport",
    "LinearFit",
    "MachineProvenance",
    "MockCurve",
    "MockEngine",
    "ResidualSummary",
    "StubTransport",
    "SweepKind",
    "SweepObservation",
    "SweepSpec",
    "TimeUnit",
    "TrustPolicy",
    "assert_labels_supported",
    "combined_digest",
    "endpoint_is_synthetic",
    "fit_sweep",
    "hardware_blockers",
    "honest_labels",
    "m9_source_paths",
    "residual_ratio",
    "run_sweep",
    "runtime_identity",
    "source_fingerprints",
    "source_manifest",
]
