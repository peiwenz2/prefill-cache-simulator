"""Join three observation sources and record every way they fail to agree.

The reconciler never repairs a disagreement and never picks a winner. It emits a
ledger of exactly three defects — a source is :data:`LedgerKind.MISSING` for an
attempt, a source reports the attempt more than once
(:data:`LedgerKind.DUPLICATE`), or the sources that both claim to observe a
shared field report different values (:data:`LedgerKind.DISAGREEMENT`).

Only one defect kind is charged per attempt, in the fixed precedence
``DUPLICATE > MISSING > DISAGREEMENT``: once a source has reported an attempt
twice we cannot say which copy the other sources should be compared against, and
once a source is absent the fields it would have contributed are unobservable.
Reporting a downstream disagreement in either case would be inventing evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .payload import (
    check,
    check_finite,
    check_ordered_subset,
    invalid,
    optional_float,
    require_float,
    require_int,
    require_mapping,
    require_sequence,
    require_str,
    require_str_sequence,
)
from .sources import (
    SHARED_FIELDS,
    SOURCE_ORDER,
    AttemptKey,
    SourceBundle,
    SourceName,
)

RECONCILE_SCHEMA_VERSION = "m10-reconcile-v1"

#: Marker written into :attr:`LedgerEntry.values` when a source has no record.
ABSENT_VALUE = "ABSENT"

#: ``field_name -> the sources that observe it``, so an entry can check that it
#: names a disagreement between the observers that field actually has.
_SHARED_FIELD_SOURCES: dict[str, tuple[SourceName, ...]] = {
    shared.field_name: shared.sources for shared in SHARED_FIELDS
}


class LedgerKind(StrEnum):
    """The three observable ways the sources fail to agree."""

    DISAGREEMENT = "DISAGREEMENT"
    DUPLICATE = "DUPLICATE"
    MISSING = "MISSING"


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """One defect charged against one attempt.

    ``field_name`` is empty for :data:`LedgerKind.MISSING` and
    :data:`LedgerKind.DUPLICATE` because those defects are about the record as a
    whole rather than about any single field.
    """

    kind: LedgerKind
    logical_request_id: str
    attempt_index: int
    field_name: str
    sources: tuple[SourceName, ...]
    values: tuple[str, ...]

    def __post_init__(self) -> None:
        context = "LedgerEntry"
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
        check(bool(self.sources), context, "an entry must name at least one source")
        check_ordered_subset(self.sources, SOURCE_ORDER, "sources", context)
        check(
            len(self.values) == len(self.sources),
            context,
            f"{len(self.values)} value(s) were recorded for "
            f"{len(self.sources)} source(s): each named source reports one value",
        )

        if self.kind is LedgerKind.DISAGREEMENT:
            observers = _SHARED_FIELD_SOURCES.get(self.field_name)
            check(
                observers is not None,
                context,
                f"{self.field_name!r} is not a shared field, so no two sources "
                "can be said to disagree about it",
            )
            check(
                self.sources == observers,
                context,
                f"a disagreement about {self.field_name!r} is between its "
                f"observers {observers!r}, not {self.sources!r}",
            )
            check(
                len(set(self.values)) > 1,
                context,
                f"a DISAGREEMENT entry records values that differ, but every "
                f"source reported {self.values[0]!r}",
            )
            return

        check(
            not self.field_name,
            context,
            f"{self.kind.value} is charged against the record as a whole, so it "
            f"names no field, but field_name is {self.field_name!r}",
        )
        if self.kind is LedgerKind.MISSING:
            check(
                set(self.values) == {ABSENT_VALUE},
                context,
                f"a MISSING entry records {ABSENT_VALUE!r} for each absent "
                f"source, got {self.values!r}",
            )

    @property
    def sort_key(self) -> tuple[str, int, str, str]:
        """Total order used to make the ledger byte-stable across runs."""
        return (
            self.logical_request_id,
            self.attempt_index,
            self.kind.value,
            self.field_name,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "logical_request_id": self.logical_request_id,
            "attempt_index": self.attempt_index,
            "field_name": self.field_name,
            "sources": [source.value for source in self.sources],
            "values": list(self.values),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LedgerEntry:
        context = "LedgerEntry"
        data = require_mapping(payload, context)
        return cls(
            LedgerKind(require_str(data, "kind", context)),
            require_str(data, "logical_request_id", context),
            require_int(data, "attempt_index", context),
            require_str(data, "field_name", context),
            tuple(
                SourceName(item)
                for item in require_str_sequence(data, "sources", context)
            ),
            require_str_sequence(data, "values", context),
        )


@dataclass(frozen=True, slots=True)
class ReconciledRow:
    """One attempt on which all three sources agreed.

    ``tpot_work`` stays ``None`` whenever the producing system does not model
    decode; the row copies the observation rather than filling a gap.

    The invariants below are the ones a row cannot violate and still be a
    reading: an identity is present, a count is not negative, a duration is a
    number. Relations *between* observations -- that arrival precedes finish, or
    that a hit is no larger than the prompt -- are deliberately not asserted. A
    row that broke one would be a real and interesting finding about the
    producing system, and a record that refused to hold it would destroy the
    evidence instead of reporting it.
    """

    logical_request_id: str
    attempt_index: int
    node_id: str
    input_tokens: int
    hit_tokens: int
    output_tokens: int
    ttft_work: float
    tpot_work: float | None
    arrival_work: float
    start_work: float
    finish_work: float

    def __post_init__(self) -> None:
        context = "ReconciledRow"
        for name, identifier in (
            ("logical_request_id", self.logical_request_id),
            ("node_id", self.node_id),
        ):
            check(bool(identifier), context, f"{name} must not be empty")
        for name, count in (
            ("attempt_index", self.attempt_index),
            ("input_tokens", self.input_tokens),
            ("hit_tokens", self.hit_tokens),
            ("output_tokens", self.output_tokens),
        ):
            check(count >= 0, context, f"{name} must not be negative, got {count!r}")
        for name, work in (
            ("ttft_work", self.ttft_work),
            ("tpot_work", self.tpot_work),
            ("arrival_work", self.arrival_work),
            ("start_work", self.start_work),
            ("finish_work", self.finish_work),
        ):
            check_finite(work, name, context)

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_request_id": self.logical_request_id,
            "attempt_index": self.attempt_index,
            "node_id": self.node_id,
            "input_tokens": self.input_tokens,
            "hit_tokens": self.hit_tokens,
            "output_tokens": self.output_tokens,
            "ttft_work": self.ttft_work,
            "tpot_work": self.tpot_work,
            "arrival_work": self.arrival_work,
            "start_work": self.start_work,
            "finish_work": self.finish_work,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ReconciledRow:
        context = "ReconciledRow"
        data = require_mapping(payload, context)
        return cls(
            require_str(data, "logical_request_id", context),
            require_int(data, "attempt_index", context),
            require_str(data, "node_id", context),
            require_int(data, "input_tokens", context),
            require_int(data, "hit_tokens", context),
            require_int(data, "output_tokens", context),
            require_float(data, "ttft_work", context),
            optional_float(data, "tpot_work", context),
            require_float(data, "arrival_work", context),
            require_float(data, "start_work", context),
            require_float(data, "finish_work", context),
        )


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    """Ledger plus the attempts that survived it.

    Every attempt is accounted for exactly once: it is either reconciled or
    charged with defects, never both and never neither. That is what makes
    :attr:`reconciled_fraction` a fraction of something rather than a ratio of
    two numbers that were counted separately, so it is checked here instead of
    being trusted.
    """

    schema_version: str
    attempt_count: int
    ledger: tuple[LedgerEntry, ...]
    reconciled: tuple[ReconciledRow, ...]

    def __post_init__(self) -> None:
        context = "ReconciliationReport"
        check(
            self.schema_version == RECONCILE_SCHEMA_VERSION,
            context,
            f"schema_version must be {RECONCILE_SCHEMA_VERSION!r}, "
            f"got {self.schema_version!r}",
        )
        check(
            self.attempt_count >= 0,
            context,
            f"attempt_count must not be negative, got {self.attempt_count!r}",
        )

        settled = {
            (row.logical_request_id, row.attempt_index) for row in self.reconciled
        }
        check(
            len(settled) == len(self.reconciled),
            context,
            f"{len(self.reconciled)} reconciled row(s) cover only "
            f"{len(settled)} attempt(s): an attempt is reconciled once",
        )

        charged: dict[tuple[str, int], set[LedgerKind]] = {}
        for entry in self.ledger:
            attempt = (entry.logical_request_id, entry.attempt_index)
            charged.setdefault(attempt, set()).add(entry.kind)
        for attempt, kinds in charged.items():
            check(
                len(kinds) == 1,
                context,
                f"attempt {attempt[0]}#{attempt[1]} is charged with "
                f"{sorted(kind.value for kind in kinds)}: only the most severe "
                "defect kind is charged, because the others are unobservable "
                "once it holds",
            )

        overlap = sorted(settled & charged.keys())
        if overlap:
            raise invalid(
                context,
                f"attempt {overlap[0][0]}#{overlap[0][1]} is both reconciled "
                "and charged with a defect",
            )
        accounted = len(settled | charged.keys())
        check(
            accounted == self.attempt_count,
            context,
            f"{accounted} attempt(s) are accounted for but attempt_count is "
            f"{self.attempt_count}: every joined attempt is either reconciled "
            "or charged",
        )

        ordered = sorted(self.ledger, key=lambda entry: entry.sort_key)
        check(
            list(self.ledger) == ordered,
            context,
            "ledger must be in sort_key order so that two runs that found the "
            "same defects produce the same bytes",
        )

    @property
    def reconciled_count(self) -> int:
        return len(self.reconciled)

    @property
    def reconciled_fraction(self) -> float:
        if self.attempt_count == 0:
            return 0.0
        return self.reconciled_count / self.attempt_count

    @property
    def disagreement_fraction(self) -> float:
        """Share of attempts carrying at least one disagreement entry."""
        if self.attempt_count == 0:
            return 0.0
        affected = {
            (entry.logical_request_id, entry.attempt_index)
            for entry in self.ledger
            if entry.kind is LedgerKind.DISAGREEMENT
        }
        return len(affected) / self.attempt_count

    def to_dict(self) -> dict[str, Any]:
        """Serialize completely, including every reconciled attempt.

        This is the round-trip form. It grows with the trace, so prefer
        :meth:`summary` for anything published.
        """
        return {
            "schema_version": self.schema_version,
            "attempt_count": self.attempt_count,
            "ledger": [entry.to_dict() for entry in self.ledger],
            "reconciled": [row.to_dict() for row in self.reconciled],
        }

    def summary(self) -> dict[str, Any]:
        """Serialize the evidence without the per-attempt row table.

        The ledger is kept in full because it *is* the finding; the reconciled
        rows are dropped because they are bulk, not evidence, and are already
        published per cell as CSV.
        """
        return {
            "schema_version": self.schema_version,
            "attempt_count": self.attempt_count,
            "reconciled_count": self.reconciled_count,
            "reconciled_fraction": self.reconciled_fraction,
            "disagreement_fraction": self.disagreement_fraction,
            "ledger": [entry.to_dict() for entry in self.ledger],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ReconciliationReport:
        """Rebuild from :meth:`to_dict`.

        A payload from :meth:`summary` is rejected for its missing ``reconciled``
        table rather than rebuilt with an empty one, which would understate the
        reconciled count.
        """
        context = "ReconciliationReport"
        data = require_mapping(payload, context)
        return cls(
            require_str(data, "schema_version", context),
            require_int(data, "attempt_count", context),
            tuple(
                LedgerEntry.from_dict(item)
                for item in require_sequence(data, "ledger", context)
            ),
            tuple(
                ReconciledRow.from_dict(item)
                for item in require_sequence(data, "reconciled", context)
            ),
        )


def _index(bundle: SourceBundle) -> dict[SourceName, dict[AttemptKey, list[Any]]]:
    index: dict[SourceName, dict[AttemptKey, list[Any]]] = {}
    for source in SOURCE_ORDER:
        buckets: dict[AttemptKey, list[Any]] = {}
        for record in bundle.records_for(source):
            buckets.setdefault(record.key, []).append(record)
        index[source] = buckets
    return index


def disagreement_entries(
    key: AttemptKey,
    records: dict[SourceName, Any],
) -> tuple[LedgerEntry, ...]:
    """Return one entry per shared field whose observers do not agree.

    ``records`` must hold exactly one record per source. Nothing outside this
    module may call it: the fault injector states its expected ledger from the
    fault plan instead, so that a bug living here cannot be reproduced by the
    oracle that is supposed to catch it.
    """
    entries: list[LedgerEntry] = []
    for shared in SHARED_FIELDS:
        values = tuple(
            records[source].field(shared.field_name) for source in shared.sources
        )
        if len(set(values)) == 1:
            continue
        entries.append(
            LedgerEntry(
                LedgerKind.DISAGREEMENT,
                key.logical_request_id,
                key.attempt_index,
                shared.field_name,
                shared.sources,
                tuple(str(value) for value in values),
            )
        )
    return tuple(entries)


def reconcile(bundle: SourceBundle) -> ReconciliationReport:
    """Join the bundle on ``(logical_request_id, attempt_index)``."""
    index = _index(bundle)
    keys = sorted({key for buckets in index.values() for key in buckets})

    ledger: list[LedgerEntry] = []
    reconciled: list[ReconciledRow] = []
    for key in keys:
        duplicated = tuple(
            source for source in SOURCE_ORDER if len(index[source].get(key, ())) > 1
        )
        if duplicated:
            ledger.append(
                LedgerEntry(
                    LedgerKind.DUPLICATE,
                    key.logical_request_id,
                    key.attempt_index,
                    "",
                    duplicated,
                    tuple(f"count={len(index[source][key])}" for source in duplicated),
                )
            )
            continue

        absent = tuple(source for source in SOURCE_ORDER if key not in index[source])
        if absent:
            ledger.append(
                LedgerEntry(
                    LedgerKind.MISSING,
                    key.logical_request_id,
                    key.attempt_index,
                    "",
                    absent,
                    (ABSENT_VALUE,) * len(absent),
                )
            )
            continue

        records = {source: index[source][key][0] for source in SOURCE_ORDER}
        entries = disagreement_entries(key, records)
        if entries:
            ledger.extend(entries)
            continue
        reconciled.append(_row(key, records))

    ledger.sort(key=lambda entry: entry.sort_key)
    return ReconciliationReport(
        RECONCILE_SCHEMA_VERSION,
        len(keys),
        tuple(ledger),
        tuple(reconciled),
    )


def _row(key: AttemptKey, records: dict[SourceName, Any]) -> ReconciledRow:
    engine = records[SourceName.ENGINE_HIT]
    client = records[SourceName.CLIENT_LATENCY]
    trace = records[SourceName.ATTEMPT_TRACE]
    return ReconciledRow(
        key.logical_request_id,
        key.attempt_index,
        engine.node_id,
        engine.input_tokens,
        engine.hit_tokens,
        client.output_tokens,
        client.ttft_work,
        client.tpot_work,
        trace.arrival_work,
        trace.start_work,
        trace.finish_work,
    )
