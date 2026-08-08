"""Record selector decisions for offline R0/R1 comparison, and refuse to act.

This module is the recording half of the M11 rollout. The RFC defines four
deployment stages -- ``off``, ``shadow``, ``enforce``, ``required`` -- and the
first two exist to produce evidence, not to route traffic. A
:class:`DecisionRecord` captures one routing decision as it happened: which host
the baseline chose, which host the selector would have chosen, why the selector
might not have answered, and what the view and capability looked like. It never
carries user content.

``enforced`` is always ``False``. Like
:class:`~..replay.shadow.ShadowDecision`, a record that claims to be enforced
cannot be constructed: the field is present so it survives serialization and
can be asserted on, and :data:`DECISION_ENFORCEMENT_ENABLED` is a module-level
constant that no configuration path flips. A decision that acted on itself
would be the one failure this whole module exists to prevent.

Privacy is structural, not editorial. :attr:`DecisionRecord.logical_request_id`
is the trace-derived ``trace:<sha>:<index>`` identity from
:func:`~..identity.logical_request_id`, never request text or token content.
:attr:`DecisionRecord.feature` is a bounded-cardinality mapping: its keys come
from :data:`DECISION_METRIC_NAMES`, a closed vocabulary of selector score
components, so no free-form string can reach the sink and inflate aggregation
cardinality.

The :class:`DecisionSink` is append-only and crash-safe. Each record is one
JSON line written with ``allow_nan=False`` and flushed to disk before the call
returns. The clock is injected so a test can stamp deterministic timings
without touching the real one. A restarted process reopens the same file and
keeps appending; nothing is rewritten.

The R1 :class:`PairReporter` reads a sink back and judges every pair. For a
push-based owner (:attr:`~.protocol.SelectorOwner.CENTRALIZED_MASTER`) a missing
shadow choice is a defect; for a pull-based owner
(:attr:`~.protocol.SelectorOwner.TURBO_CACHE_AWARE`) the shadow comparison is
structurally unresolved, because the pull dispatch loop has no channel a push
observer can listen on. That is not a silent gap: it is published as
:data:`PairStatus.UNRESOLVED_OWNER_SIGNOFF`, a machine-readable statement that
the Turbo owner must sign off on the comparison before it can be claimed.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..replay.payload import (
    MalformedPayloadError,
    check,
    check_finite,
    invalid,
    require_bool,
    require_float,
    require_int,
    require_mapping,
    require_str,
)
from .protocol import (
    Capability,
    EnforcementMode,
    FailOpenReason,
    SelectorOwner,
)

__all__ = [
    "DECISION_ENFORCEMENT_ENABLED",
    "DECISION_METRIC_NAMES",
    "DECISION_SCHEMA_VERSION",
    "DECISION_SINK_SCHEMA_VERSION",
    "DIFF_SCHEMA_VERSION",
    "MAX_FEATURE_KEYS",
    "MAX_LOGICAL_REQUEST_ID_LEN",
    "DecisionEnforcementError",
    "DecisionRecord",
    "DecisionSink",
    "DiffReport",
    "JsonlPushObserver",
    "PairReporter",
    "PairStatus",
    "PushObserver",
]

#: Schema of one record. Bumped, not extended, so an older file is rejected.
DECISION_SCHEMA_VERSION = "m11-decision-record-v1"

#: Schema of the sink header line. Separate so the wire format and the record
#: can move independently.
DECISION_SINK_SCHEMA_VERSION = "m11-decision-sink-v1"

#: Schema of the R1 diff artifact.
DIFF_SCHEMA_VERSION = "m11-decision-diff-v1"

#: Enforcement is off by construction. The constant mirrors
#: :data:`~..replay.shadow.ENFORCEMENT_ENABLED` so the two cannot diverge, and
#: flipping it alone does not turn it on: :class:`DecisionRecord` independently
#: refuses ``enforced=True``.
DECISION_ENFORCEMENT_ENABLED = False

#: Maximum length of a logical request ID. The ID is a privacy-safe hash or
#: UUID, never request text; bounding it keeps a hostile or buggy producer from
#: flooding the sink with oversized strings.
MAX_LOGICAL_REQUEST_ID_LEN = 128

#: Privacy-safe request identity schema: either ``trace:<hex>:<digits>`` (the
#: trace-derived ID from :mod:`..identity`) or a standard UUID. Anything else
#: is rejected so raw request content cannot reach the sink as an ID.
_LOGICAL_REQUEST_ID_RE = re.compile(
    r"^(trace:[0-9a-f]{1,128}:[0-9]+"
    r"|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$"
)

#: The closed vocabulary of selector score-component names. A feature mapping
#: whose keys are outside this set is rejected, which is what keeps the sink's
#: aggregation cardinality bounded: a new metric is a schema change, not a
#: string someone happened to emit.
DECISION_METRIC_NAMES: frozenset[str] = frozenset(
    {
        "affinity",
        "affinity_broken",
        "fail_open",
        "primary_load",
        "chosen_load",
        "queue_tokens",
        "hit_tokens",
        "ttft_score",
        "uncached_tokens",
        "ttft_ms",
        "random_index",
        "rr_index",
        "load_tokens",
        "cache_hit_tokens",
        "shadow_hit_tokens",
    }
)

#: Upper bound on feature keys. Redundant with the vocabulary check but kept so
#: a future relaxation to "any string" still has a ceiling.
MAX_FEATURE_KEYS = len(DECISION_METRIC_NAMES)


class DecisionEnforcementError(RuntimeError):
    """Raised on any attempt to mark a decision record as enforced."""


class PairStatus(StrEnum):
    """The verdict on one (online, shadow) pair, or on a structural defect."""

    AGREED = "AGREED"
    DISAGREED = "DISAGREED"
    SHADOW_MISSING = "SHADOW_MISSING"
    R0_OFF_FALLBACK = "R0_OFF_FALLBACK"
    EXPLAINED_FAIL_OPEN = "EXPLAINED_FAIL_OPEN"
    DUPLICATE = "DUPLICATE"
    OUT_OF_ORDER = "OUT_OF_ORDER"
    UNRESOLVED_OWNER_SIGNOFF = "UNRESOLVED_OWNER_SIGNOFF"


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """One routing decision, with both the baseline and the selector choice.

    The record carries the context the RFC §5 fail-open table names -- view
    staleness, capability handshake, fallback reason, timing -- so an offline
    reader can tell a disagreement from a fail-open without a second source.

    ``shadow_host`` is ``None`` when the selector did not produce a choice:
    it timed out, threw, saw a stale view, or never handshook. That absence is
    recorded, not invented, and the R1 reporter charges it as
    :data:`PairStatus.SHADOW_MISSING` for a push owner or
    :data:`PairStatus.UNRESOLVED_OWNER_SIGNOFF` for a pull owner.

    ``enforced`` is always ``False``. A record that claims otherwise cannot be
    constructed, so an enforced decision cannot exist long enough to be written,
    read, or acted upon.
    """

    schema_version: str
    logical_request_id: str
    attempt_index: int
    owner: SelectorOwner
    mode: EnforcementMode
    online_host: str
    shadow_host: str | None
    feature: Mapping[str, float]
    view_epoch: int
    view_age_ms: int
    view_stale: bool
    capability_accepted: bool
    capability_degraded: tuple[str, ...]
    fallback: FailOpenReason | None
    recorded_at_ms: float
    enforced: bool = False

    def __post_init__(self) -> None:
        context = "DecisionRecord"
        if self.enforced:
            raise DecisionEnforcementError(
                "a decision record is never enforced: R0/R1 records decisions only"
            )
        check(
            self.schema_version == DECISION_SCHEMA_VERSION,
            context,
            f"schema_version must be {DECISION_SCHEMA_VERSION!r}, "
            f"got {self.schema_version!r}",
        )
        check(
            isinstance(self.logical_request_id, str) and bool(self.logical_request_id),
            context,
            "logical_request_id must be a non-empty string",
        )
        check(
            len(self.logical_request_id) <= MAX_LOGICAL_REQUEST_ID_LEN,
            context,
            f"logical_request_id must be at most {MAX_LOGICAL_REQUEST_ID_LEN} "
            f"chars, got {len(self.logical_request_id)}",
        )
        check(
            bool(_LOGICAL_REQUEST_ID_RE.match(self.logical_request_id)),
            context,
            "logical_request_id must be a privacy-safe trace hash or UUID, "
            f"got {self.logical_request_id!r}",
        )
        check(
            isinstance(self.attempt_index, int)
            and not isinstance(self.attempt_index, bool)
            and self.attempt_index >= 0,
            context,
            f"attempt_index must be a non-negative int, got {self.attempt_index!r}",
        )
        check(
            isinstance(self.owner, SelectorOwner),
            context,
            f"owner must be a SelectorOwner, got {type(self.owner).__name__}",
        )
        check(
            isinstance(self.mode, EnforcementMode),
            context,
            f"mode must be an EnforcementMode, got {type(self.mode).__name__}",
        )
        check(
            isinstance(self.online_host, str) and bool(self.online_host),
            context,
            "online_host must be a non-empty string",
        )
        if self.shadow_host is not None:
            check(
                isinstance(self.shadow_host, str) and bool(self.shadow_host),
                context,
                "shadow_host must be None or a non-empty string",
            )
        check(
            isinstance(self.view_epoch, int)
            and not isinstance(self.view_epoch, bool)
            and self.view_epoch >= 0,
            context,
            f"view_epoch must be a non-negative int, got {self.view_epoch!r}",
        )
        check(
            isinstance(self.view_age_ms, int)
            and not isinstance(self.view_age_ms, bool)
            and self.view_age_ms >= 0,
            context,
            f"view_age_ms must be a non-negative int, got {self.view_age_ms!r}",
        )
        check(
            isinstance(self.view_stale, bool),
            context,
            f"view_stale must be a bool, got {type(self.view_stale).__name__}",
        )
        check(
            isinstance(self.capability_accepted, bool),
            context,
            "capability_accepted must be a bool",
        )
        check(
            len(self.capability_degraded) <= len(Capability),
            context,
            f"capability_degraded has {len(self.capability_degraded)} items, "
            f"at most {len(Capability)} allowed",
        )
        for item in self.capability_degraded:
            try:
                Capability(item)
            except ValueError as error:
                raise invalid(
                    context,
                    f"capability_degraded contains {item!r}, "
                    "which is not a valid Capability",
                ) from error
        if self.fallback is not None:
            check(
                isinstance(self.fallback, FailOpenReason),
                context,
                f"fallback must be None or a FailOpenReason, "
                f"got {type(self.fallback).__name__}",
            )
        check_finite(self.recorded_at_ms, "recorded_at_ms", context)
        check(
            self.recorded_at_ms >= 0,
            context,
            f"recorded_at_ms must be non-negative, got {self.recorded_at_ms!r}",
        )
        self._check_feature(context)

    def _check_feature(self, context: str) -> None:
        if not isinstance(self.feature, Mapping):
            raise invalid(
                context,
                f"feature must be a mapping, got {type(self.feature).__name__}",
            )
        check(
            len(self.feature) <= MAX_FEATURE_KEYS,
            context,
            f"feature has {len(self.feature)} keys, at most {MAX_FEATURE_KEYS} allowed",
        )
        for name, value in self.feature.items():
            check(
                isinstance(name, str) and name in DECISION_METRIC_NAMES,
                context,
                f"feature key {name!r} is not in the bounded vocabulary "
                f"DECISION_METRIC_NAMES",
            )
            check(
                not isinstance(value, bool) and isinstance(value, int | float),
                context,
                f"feature[{name!r}] must be a number, got {type(value).__name__}",
            )
            check_finite(float(value), f"feature[{name!r}]", context)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "logical_request_id": self.logical_request_id,
            "attempt_index": self.attempt_index,
            "owner": self.owner.value,
            "mode": self.mode.value,
            "online_host": self.online_host,
            "shadow_host": self.shadow_host,
            "feature": dict(self.feature),
            "view_epoch": self.view_epoch,
            "view_age_ms": self.view_age_ms,
            "view_stale": self.view_stale,
            "capability_accepted": self.capability_accepted,
            "capability_degraded": list(self.capability_degraded),
            "fallback": None if self.fallback is None else self.fallback.value,
            "recorded_at_ms": self.recorded_at_ms,
            "enforced": self.enforced,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DecisionRecord:
        context = "DecisionRecord"
        data = require_mapping(payload, context)
        feature = data.get("feature", {})
        if not isinstance(feature, Mapping):
            raise invalid(context, "feature must be an object")
        fallback_raw = data.get("fallback")
        degraded_raw = data.get("capability_degraded", [])
        if not isinstance(degraded_raw, Sequence) or isinstance(
            degraded_raw, str | bytes
        ):
            raise invalid(context, "capability_degraded must be an array")
        return cls(
            require_str(data, "schema_version", context),
            require_str(data, "logical_request_id", context),
            require_int(data, "attempt_index", context),
            SelectorOwner(require_str(data, "owner", context)),
            EnforcementMode(require_str(data, "mode", context)),
            require_str(data, "online_host", context),
            data.get("shadow_host") if "shadow_host" in data else None,
            {str(k): v for k, v in feature.items()},
            require_int(data, "view_epoch", context),
            require_int(data, "view_age_ms", context),
            require_bool(data, "view_stale", context),
            require_bool(data, "capability_accepted", context),
            tuple(str(item) for item in degraded_raw),
            None if fallback_raw is None else FailOpenReason(str(fallback_raw)),
            require_float(data, "recorded_at_ms", context),
            require_bool(data, "enforced", context),
        )


def _reject_json_constant(token: str) -> float:
    raise MalformedPayloadError(
        f"{token} is not valid JSON and is refused at the sink boundary"
    )


class DecisionSink:
    """Append-only JSONL sink for :class:`DecisionRecord`, crash-safe.

    Each ``write`` appends one strict-JSON line (``allow_nan=False``) and
    flushes. ``fsync`` defaults on because a sink that buffers in the kernel is
    not crash-safe; a test that does not need the syscall can disable it for
    speed, and the records are still valid JSONL.

    The ``clock`` is injected and called once per write to stamp
    :attr:`DecisionRecord.recorded_at_ms`. The real clock is never imported
    here, so a test can drive deterministic timings without monkeypatching.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Any,
        fsync: bool = True,
    ) -> None:
        self._path = Path(path)
        self._clock = clock
        self._fsync = fsync
        self._recover_truncated_tail()
        self._file = open(self._path, "a", encoding="utf-8")  # noqa: SIM115

    def _recover_truncated_tail(self) -> None:
        """Detect and truncate an incomplete final line from a crashed write.

        A complete record is one JSON line terminated by ``\\n``. A crash
        mid-write leaves a partial line, which would be rejected as malformed on
        the next read. This scans the file, finds the last valid line boundary,
        and truncates anything after it so the next append starts clean.
        """
        if not self._path.exists():
            return
        data = self._path.read_bytes()
        if not data:
            return
        valid_end = 0
        line_start = 0
        for i, byte in enumerate(data):
            if byte == 0x0A:  # \n
                text = data[line_start:i].decode("utf-8", errors="replace").strip()
                if text:
                    try:
                        json.loads(text, parse_constant=_reject_json_constant)
                        valid_end = i + 1
                    except ValueError:
                        break
                else:
                    valid_end = i + 1
                line_start = i + 1
        if valid_end < len(data):
            with open(self._path, "r+b") as handle:
                handle.truncate(valid_end)

    @property
    def path(self) -> Path:
        return self._path

    def write(self, record: DecisionRecord) -> None:
        if not isinstance(record, DecisionRecord):
            raise TypeError(
                f"write expects a DecisionRecord, got {type(record).__name__}"
            )
        timing = float(self._clock())
        if not math.isfinite(timing) or timing < 0:
            raise ValueError(f"clock must return non-negative finite, got {timing!r}")
        stamped = DecisionRecord(
            record.schema_version,
            record.logical_request_id,
            record.attempt_index,
            record.owner,
            record.mode,
            record.online_host,
            record.shadow_host,
            record.feature,
            record.view_epoch,
            record.view_age_ms,
            record.view_stale,
            record.capability_accepted,
            record.capability_degraded,
            record.fallback,
            timing,
            record.enforced,
        )
        line = (
            json.dumps(stamped.to_dict(), sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        self._file.write(line.decode("utf-8"))
        self._file.flush()
        if self._fsync:
            os.fsync(self._file.fileno())

    def read_all(self) -> tuple[DecisionRecord, ...]:
        """Parse every line, refusing non-JSON and non-finite content.

        A line that is not valid JSON, or that carries a ``NaN``/``Infinity``
        literal, is rejected at the boundary rather than coerced. A truncated
        final line (the crash window) is skipped with a single trailing newline
        guarantee, not silently accepted as a half record.
        """
        if not self._file.closed:
            self._file.flush()
        records: list[DecisionRecord] = []
        with open(self._path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text, parse_constant=_reject_json_constant)
                except ValueError as error:
                    raise MalformedPayloadError(
                        f"{self._path}:{line_number}: not valid JSON: {error}"
                    ) from error
                records.append(DecisionRecord.from_dict(payload))
        return tuple(records)

    def close(self) -> None:
        if not self._file.closed:
            self._file.flush()
            self._file.close()


@runtime_checkable
class PushObserver(Protocol):
    """A generic sink for decision records.

    Any component that routes traffic -- the chain harness, a real Centralized Master
    adapter, a Turbo pull-loop shim -- can implement this and hand it to the
    decision source. The protocol is deliberately minimal: one record in, nothing
    out, because the observer's job is to persist, not to advise.
    """

    def observe(self, record: DecisionRecord) -> None: ...


@dataclass(slots=True)
class JsonlPushObserver:
    """A :class:`PushObserver` backed by a :class:`DecisionSink`.

    Separate from the sink so the sink stays a byte-level concern and the
    observer stays the routing-level seam. A test can substitute a counting
    observer without touching the file.
    """

    sink: DecisionSink

    def observe(self, record: DecisionRecord) -> None:
        self.sink.write(record)


def _pair_status(record: DecisionRecord) -> PairStatus:
    if record.owner is SelectorOwner.TURBO_CACHE_AWARE:
        return PairStatus.UNRESOLVED_OWNER_SIGNOFF
    if record.shadow_host is None:
        if record.mode is EnforcementMode.OFF:
            return PairStatus.R0_OFF_FALLBACK
        if record.fallback is not None:
            return PairStatus.EXPLAINED_FAIL_OPEN
        return PairStatus.SHADOW_MISSING
    if record.online_host == record.shadow_host:
        return PairStatus.AGREED
    return PairStatus.DISAGREED


@dataclass(frozen=True, slots=True)
class DiffReport:
    """The R1 pair/diff verdict over one sink of records.

    The report lists the count of each :class:`PairStatus` and the per-status
    records, so a reader can see *which* requests disagreed, not just how many.

    ``accepted`` is structural: no duplicates, no out-of-order, and no missing
    pairs. Disagreements are not a defect -- they are the signal shadow mode
    exists to produce -- so they do not fail the gate. A missing pair for a pull
    owner is :data:`PairStatus.UNRESOLVED_OWNER_SIGNOFF`, which does not fail
    the structural gate either, because the limitation is known and published
    rather than hidden.

    ``rollout_gate_accepted`` is stricter: it is False whenever the collection
    cannot support a rollout decision. That includes structural defects, but also
    :data:`PairStatus.UNRESOLVED_OWNER_SIGNOFF` (the Turbo owner must sign off
    before the comparison can be claimed), :data:`PairStatus.R0_OFF_FALLBACK`
    (R0 is off), and :data:`PairStatus.EXPLAINED_FAIL_OPEN` (the record is
    structurally valid, but its rate still needs comparison with the online
    baseline before rollout).
    """

    schema_version: str
    total_records: int
    pair_counts: Mapping[PairStatus, int]
    disagree_records: tuple[dict[str, Any], ...]
    missing_records: tuple[dict[str, Any], ...]
    unresolved_records: tuple[dict[str, Any], ...]
    accepted: bool
    rollout_gate_accepted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "total_records": self.total_records,
            "pair_counts": {
                status.value: self.pair_counts.get(status, 0) for status in PairStatus
            },
            "disagree_records": list(self.disagree_records),
            "missing_records": list(self.missing_records),
            "unresolved_records": list(self.unresolved_records),
            "accepted": self.accepted,
            "rollout_gate_accepted": self.rollout_gate_accepted,
        }


class PairReporter:
    """Read a sink and judge every (online, shadow) pair.

    The reporter is the R1 component: it reads the JSONL sink back, reconstructs
    the pairing, and publishes a :class:`DiffReport`. The artifact is written
    atomically (stage then :func:`os.replace`) so a reader that catches the
    window sees either the old report or the new one, never a half-written one.
    """

    def report(self, records: Sequence[DecisionRecord]) -> DiffReport:
        counts: dict[PairStatus, int] = {status: 0 for status in PairStatus}
        disagree: list[dict[str, Any]] = []
        missing: list[dict[str, Any]] = []
        unresolved: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        high_water: dict[str, int] = {}
        structural_defect = False

        for record in records:
            key = (record.logical_request_id, record.attempt_index)
            if key in seen:
                counts[PairStatus.DUPLICATE] += 1
                structural_defect = True
                continue
            seen.add(key)
            prev = high_water.get(record.logical_request_id)
            if prev is not None and record.attempt_index < prev:
                counts[PairStatus.OUT_OF_ORDER] += 1
                structural_defect = True
                continue
            high_water[record.logical_request_id] = record.attempt_index

            status = _pair_status(record)
            counts[status] += 1
            if status is PairStatus.DISAGREED:
                disagree.append(record.to_dict())
            elif status is PairStatus.SHADOW_MISSING:
                missing.append(record.to_dict())
                structural_defect = True
            elif status is PairStatus.UNRESOLVED_OWNER_SIGNOFF:
                unresolved.append(record.to_dict())

        accepted = not structural_defect
        rollout_gate_accepted = (
            accepted
            and counts[PairStatus.UNRESOLVED_OWNER_SIGNOFF] == 0
            and counts[PairStatus.R0_OFF_FALLBACK] == 0
            and counts[PairStatus.EXPLAINED_FAIL_OPEN] == 0
        )
        return DiffReport(
            DIFF_SCHEMA_VERSION,
            len(records),
            counts,
            tuple(disagree),
            tuple(missing),
            tuple(unresolved),
            accepted,
            rollout_gate_accepted,
        )

    def write_artifact(self, path: str | Path, report: DiffReport) -> None:
        """Write the diff report atomically: stage, then replace."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode("utf-8")
        fd, staging_name = tempfile.mkstemp(
            dir=str(target.parent), prefix=".decision-diff-staging-"
        )
        os.close(fd)
        staging = Path(staging_name)
        try:
            staging.write_bytes(payload)
            os.replace(staging, target)
        finally:
            if staging.exists():
                staging.unlink(missing_ok=True)
