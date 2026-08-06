"""M11 chain mocks: cross-component protocol invariants, not a performance model.

The package encodes the contract in ``docs/m11-production-rfc.md`` and makes it
executable. It answers "can this message order happen", never "how fast is it".
The modules divide the work:

``protocol``
    The message and configuration vocabulary: roles, enforcement modes,
    capabilities, fail-open reasons, and the two placement functions.
``harness``
    A single-threaded state machine that plays client, selector owner, prefill
    and decode at once, and raises on a forbidden transition.
``scenarios``
    The seven fixtures the RFC names, each one message order with no assertions.

Honesty note: every value this package produces is constructed. It is labelled
:data:`CHAIN_TRUTH_BASIS` and must never be reported as a production end-to-end
measurement, a hardware result, or a canary observation.
"""

from .harness import (
    ChainEvent,
    ChainEventKind,
    ChainHarness,
    ChainTranscript,
    DecodeLeg,
    DurableClientLedger,
)
from .protocol import (
    CHAIN_EVIDENCE_NOTE,
    CHAIN_SCHEMA_VERSION,
    CHAIN_TRUTH_BASIS,
    D2_GATE_CLOSED,
    FAIL_OPEN_COUNTERS,
    SELECTOR_PROTOCOL_VERSION,
    SELECTOR_SCORE_VERSION,
    Capabilities,
    Capability,
    ChainConfig,
    ChainProtocolError,
    ChainRole,
    EnforcementMode,
    FailOpenReason,
    HandshakeResult,
    LegRoute,
    RequiredModeFailure,
    RequiredModeRejected,
    Selection,
    SelectionOutcome,
    SelectorOwner,
    TerminalReason,
    ViewSnapshot,
    baseline_host,
    negotiate,
    score_report,
    selector_host,
)
from .scenarios import (
    SCENARIO_ORDER,
    SCENARIOS,
    enforce_config,
    fresh_view,
    run_all,
    stale_view,
)

__all__ = [
    "CHAIN_EVIDENCE_NOTE",
    "CHAIN_SCHEMA_VERSION",
    "CHAIN_TRUTH_BASIS",
    "D2_GATE_CLOSED",
    "FAIL_OPEN_COUNTERS",
    "SCENARIOS",
    "SCENARIO_ORDER",
    "SELECTOR_PROTOCOL_VERSION",
    "SELECTOR_SCORE_VERSION",
    "Capabilities",
    "Capability",
    "ChainConfig",
    "ChainEvent",
    "ChainEventKind",
    "ChainHarness",
    "ChainProtocolError",
    "ChainRole",
    "ChainTranscript",
    "DecodeLeg",
    "DurableClientLedger",
    "EnforcementMode",
    "FailOpenReason",
    "HandshakeResult",
    "LegRoute",
    "RequiredModeFailure",
    "RequiredModeRejected",
    "Selection",
    "SelectionOutcome",
    "SelectorOwner",
    "TerminalReason",
    "ViewSnapshot",
    "baseline_host",
    "enforce_config",
    "fresh_view",
    "negotiate",
    "run_all",
    "score_report",
    "selector_host",
    "stale_view",
]
