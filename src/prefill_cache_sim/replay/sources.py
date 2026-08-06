"""Three observation sources and the stable join key that links them.

M10 reconciles what three independent observers claim about the same attempt:
the engine's cache-hit accounting, the client's latency view, and the scheduler
attempt trace. Each source is a separate record type, so a field only one
observer can see stays on that observer and only the fields listed in
:data:`SHARED_FIELDS` are ever cross-checked.

Honesty note: nothing here measures hardware. A bundle carries an explicit
:class:`TruthBasis`, and the strongest basis (``MEASURED_ENGINE``) is refused
unless complete :class:`MachineProvenance` is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ..calibration import DishonestLabelError, MachineProvenance
from .payload import (
    check,
    check_finite,
    optional_float,
    optional_str,
    require_float,
    require_int,
    require_mapping,
    require_object,
    require_sequence,
    require_str,
)

SOURCE_SCHEMA_VERSION = "m10-source-v1"

#: The simulator models prefill placement only; it has no decode loop, so a
#: per-output-token cost cannot be derived from it without inventing one.
TPOT_UNMODELED_REASON = "SIMULATOR_DOES_NOT_MODEL_DECODE"


class SourceName(StrEnum):
    """The three observers M10 reconciles."""

    ENGINE_HIT = "ENGINE_HIT"
    CLIENT_LATENCY = "CLIENT_LATENCY"
    ATTEMPT_TRACE = "ATTEMPT_TRACE"


#: Canonical ordering used whenever a ledger entry lists more than one source.
SOURCE_ORDER: tuple[SourceName, ...] = (
    SourceName.ENGINE_HIT,
    SourceName.CLIENT_LATENCY,
    SourceName.ATTEMPT_TRACE,
)


class TruthBasis(StrEnum):
    """Where a bundle's observations came from.

    ``SYNTHETIC_FIXTURE`` is generated data, ``MODELED_SIMULATOR`` is derived
    from the local simulator, and ``MEASURED_ENGINE`` is the only basis that
    claims a real engine produced the numbers.
    """

    SYNTHETIC_FIXTURE = "SYNTHETIC_FIXTURE"
    MODELED_SIMULATOR = "MODELED_SIMULATOR"
    MEASURED_ENGINE = "MEASURED_ENGINE"


@dataclass(frozen=True, slots=True, order=True)
class AttemptKey:
    """Join key: a logical request plus the attempt index within it.

    The logical request id survives retries, so ``(logical_request_id,
    attempt_index)`` identifies exactly one placement decision across all
    three sources.
    """

    logical_request_id: str
    attempt_index: int

    def __post_init__(self) -> None:
        context = "AttemptKey"
        check(
            bool(self.logical_request_id),
            context,
            "logical_request_id must not be empty",
        )
        check(
            self.attempt_index >= 0,
            context,
            f"attempt_index must not be negative, got {self.attempt_index!r}",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_request_id": self.logical_request_id,
            "attempt_index": self.attempt_index,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AttemptKey:
        """Rebuild a key, refusing a payload that would forge a second identity.

        A string ``attempt_index`` hashes and sorts differently from its integer
        twin, so accepting one would split a single attempt into two and charge
        both as MISSING. The type is checked rather than coerced.
        """
        data = require_mapping(payload, "AttemptKey")
        return cls(
            require_str(data, "logical_request_id", "AttemptKey"),
            require_int(data, "attempt_index", "AttemptKey"),
        )


def _field(record: object, names: tuple[str, ...], name: str) -> Any:
    if name not in names:
        raise ValueError(f"unknown field {name!r} on {type(record).__name__}")
    return getattr(record, name)


def _check_counts(context: str, counts: tuple[tuple[str, int], ...]) -> None:
    """Refuse a negative count: no observer can see fewer than none of a thing.

    This is as far as a record checks itself. Relations *between* observations
    -- that a cache hit is no larger than the prompt, or that a request arrived
    before it started -- are not asserted anywhere in this module. A source that
    reported such a thing would be making a real and reportable claim about the
    system it watches, and a record that refused to hold it would delete the
    finding rather than surface it. Cross-source contradictions are the ledger's
    subject; a record's job is only to be readable.
    """
    for name, count in counts:
        check(count >= 0, context, f"{name} must not be negative, got {count!r}")


@dataclass(frozen=True, slots=True)
class EngineHitRecord:
    """What the engine says it reused for one attempt."""

    key: AttemptKey
    node_id: str
    input_tokens: int
    hit_tokens: int

    _FIELDS = ("node_id", "input_tokens", "hit_tokens")

    def __post_init__(self) -> None:
        context = "EngineHitRecord"
        check(bool(self.node_id), context, "node_id must not be empty")
        _check_counts(
            context,
            (
                ("input_tokens", self.input_tokens),
                ("hit_tokens", self.hit_tokens),
            ),
        )

    def field(self, name: str) -> Any:
        return _field(self, EngineHitRecord._FIELDS, name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key.to_dict(),
            "node_id": self.node_id,
            "input_tokens": self.input_tokens,
            "hit_tokens": self.hit_tokens,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EngineHitRecord:
        context = "EngineHitRecord"
        data = require_mapping(payload, context)
        return cls(
            AttemptKey.from_dict(require_object(data, "key", context)),
            require_str(data, "node_id", context),
            require_int(data, "input_tokens", context),
            require_int(data, "hit_tokens", context),
        )


@dataclass(frozen=True, slots=True)
class ClientLatencyRecord:
    """What the client saw for one attempt.

    ``ttft_work`` is measured from the moment the client handed the request over
    to the moment the first token came back -- ``arrival`` to ``finish``, not
    ``start`` to ``finish``. The client cannot see the scheduler, so it cannot
    subtract the time its request spent queued; a request that waited behind a
    hundred others and then answered instantly was slow, and a record that said
    otherwise would report the server's convenience as the user's experience.
    The queueing delay is exactly what an arrival-rate sweep is meant to expose,
    so excluding it would blind the sweep to its own subject.

    ``ttft_work`` and ``tpot_work`` are in normalized work units, never
    milliseconds. ``tpot_work`` is ``None`` whenever the producing system does
    not model decode; see :data:`TPOT_UNMODELED_REASON`.
    """

    key: AttemptKey
    input_tokens: int
    output_tokens: int
    ttft_work: float
    tpot_work: float | None

    _FIELDS = ("input_tokens", "output_tokens", "ttft_work", "tpot_work")

    def __post_init__(self) -> None:
        context = "ClientLatencyRecord"
        _check_counts(
            context,
            (
                ("input_tokens", self.input_tokens),
                ("output_tokens", self.output_tokens),
            ),
        )
        check_finite(self.ttft_work, "ttft_work", context)
        check_finite(self.tpot_work, "tpot_work", context)

    def field(self, name: str) -> Any:
        return _field(self, ClientLatencyRecord._FIELDS, name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key.to_dict(),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "ttft_work": self.ttft_work,
            "tpot_work": self.tpot_work,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ClientLatencyRecord:
        context = "ClientLatencyRecord"
        data = require_mapping(payload, context)
        return cls(
            AttemptKey.from_dict(require_object(data, "key", context)),
            require_int(data, "input_tokens", context),
            require_int(data, "output_tokens", context),
            require_float(data, "ttft_work", context),
            optional_float(data, "tpot_work", context),
        )


@dataclass(frozen=True, slots=True)
class AttemptTraceRecord:
    """Where and when the scheduler says the attempt ran, in work units."""

    key: AttemptKey
    node_id: str
    input_tokens: int
    arrival_work: float
    start_work: float
    finish_work: float

    _FIELDS = (
        "node_id",
        "input_tokens",
        "arrival_work",
        "start_work",
        "finish_work",
    )

    def __post_init__(self) -> None:
        context = "AttemptTraceRecord"
        check(bool(self.node_id), context, "node_id must not be empty")
        _check_counts(context, (("input_tokens", self.input_tokens),))
        for name, work in (
            ("arrival_work", self.arrival_work),
            ("start_work", self.start_work),
            ("finish_work", self.finish_work),
        ):
            check_finite(work, name, context)

    def field(self, name: str) -> Any:
        return _field(self, AttemptTraceRecord._FIELDS, name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key.to_dict(),
            "node_id": self.node_id,
            "input_tokens": self.input_tokens,
            "arrival_work": self.arrival_work,
            "start_work": self.start_work,
            "finish_work": self.finish_work,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AttemptTraceRecord:
        context = "AttemptTraceRecord"
        data = require_mapping(payload, context)
        return cls(
            AttemptKey.from_dict(require_object(data, "key", context)),
            require_str(data, "node_id", context),
            require_int(data, "input_tokens", context),
            require_float(data, "arrival_work", context),
            require_float(data, "start_work", context),
            require_float(data, "finish_work", context),
        )


@dataclass(frozen=True, slots=True)
class SharedField:
    """A field more than one source claims to observe."""

    field_name: str
    sources: tuple[SourceName, ...]


#: The complete cross-source observable surface. A field that only one source
#: can see is deliberately absent: disagreeing about it is not observable, and
#: pretending otherwise would manufacture evidence.
SHARED_FIELDS: tuple[SharedField, ...] = (
    SharedField(
        "input_tokens",
        (
            SourceName.ENGINE_HIT,
            SourceName.CLIENT_LATENCY,
            SourceName.ATTEMPT_TRACE,
        ),
    ),
    SharedField("node_id", (SourceName.ENGINE_HIT, SourceName.ATTEMPT_TRACE)),
)


@dataclass(frozen=True, slots=True)
class SourceBundle:
    """One replay's worth of observations from all three sources."""

    schema_version: str
    truth_basis: TruthBasis
    machine: MachineProvenance
    tpot_reason: str | None
    engine_hits: tuple[EngineHitRecord, ...]
    client_latencies: tuple[ClientLatencyRecord, ...]
    attempt_traces: tuple[AttemptTraceRecord, ...]

    def __post_init__(self) -> None:
        if self.truth_basis is TruthBasis.MEASURED_ENGINE and not self.machine.complete:
            raise DishonestLabelError(
                "truth_basis MEASURED_ENGINE requires complete machine provenance"
            )
        missing_tpot = any(record.tpot_work is None for record in self.client_latencies)
        if missing_tpot and not self.tpot_reason:
            raise ValueError("a null tpot_work requires an explicit tpot_reason")

    def records_for(self, source: SourceName) -> tuple[Any, ...]:
        if source is SourceName.ENGINE_HIT:
            return self.engine_hits
        if source is SourceName.CLIENT_LATENCY:
            return self.client_latencies
        return self.attempt_traces

    def with_records(
        self, source: SourceName, records: tuple[Any, ...]
    ) -> SourceBundle:
        """Return a copy where one source's records have been replaced."""
        if source is SourceName.ENGINE_HIT:
            return SourceBundle(
                self.schema_version,
                self.truth_basis,
                self.machine,
                self.tpot_reason,
                records,
                self.client_latencies,
                self.attempt_traces,
            )
        if source is SourceName.CLIENT_LATENCY:
            return SourceBundle(
                self.schema_version,
                self.truth_basis,
                self.machine,
                self.tpot_reason,
                self.engine_hits,
                records,
                self.attempt_traces,
            )
        return SourceBundle(
            self.schema_version,
            self.truth_basis,
            self.machine,
            self.tpot_reason,
            self.engine_hits,
            self.client_latencies,
            records,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "truth_basis": self.truth_basis.value,
            "machine": self.machine.to_dict(),
            "tpot_reason": self.tpot_reason,
            "engine_hits": [record.to_dict() for record in self.engine_hits],
            "client_latencies": [record.to_dict() for record in self.client_latencies],
            "attempt_traces": [record.to_dict() for record in self.attempt_traces],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SourceBundle:
        context = "SourceBundle"
        data = require_mapping(payload, context)
        return cls(
            require_str(data, "schema_version", context),
            TruthBasis(require_str(data, "truth_basis", context)),
            MachineProvenance.from_dict(require_object(data, "machine", context)),
            optional_str(data, "tpot_reason", context),
            tuple(
                EngineHitRecord.from_dict(item)
                for item in require_sequence(data, "engine_hits", context)
            ),
            tuple(
                ClientLatencyRecord.from_dict(item)
                for item in require_sequence(data, "client_latencies", context)
            ),
            tuple(
                AttemptTraceRecord.from_dict(item)
                for item in require_sequence(data, "attempt_traces", context)
            ),
        )
