"""Generate a clean synthetic bundle and inject defects whose ledger is known.

The point of this module is to make the reconciler falsifiable. A fault plan
names exactly which records to drop, duplicate, or perturb, and
:func:`apply_faults` returns both the damaged bundle and the ledger those faults
*should* produce.

The expected ledger is derived from the fault plan and the *clean* bundle. It
never calls :func:`~.reconcile.reconcile` or any of its helpers, and it never
reads the damaged records back: a plan that says "set ``node_id`` to ``X`` on
the engine record" already determines what the engine will report, so consulting
the mutated bundle would only ask the injector to confirm itself. An injector
that wrote the perturbation to the wrong source, duplicated the wrong record, or
dropped one too many is therefore caught by the same comparison that checks the
reconciler.

The two also disagree structurally on purpose. The reconciler decides precedence
by falling through a chain of early returns; the oracle computes all three
candidate defect sets unconditionally and then selects one from the
:data:`_PRECEDENCE` table. A bug that reordered the reconciler's branches would
not be mirrored here, because there is no branch order here to mirror.

Honesty note: :func:`synthetic_bundle` fabricates every number it returns. The
bundle is therefore labelled :data:`TruthBasis.SYNTHETIC_FIXTURE` and must never
be published as a measurement. It carries a ``tpot_work`` value only because a
fixture is free to invent one; the simulator-derived bundle does not, and says
so via :data:`TPOT_UNMODELED_REASON`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from ..calibration import MachineProvenance
from ..identity import NamedSeedManager
from .reconcile import ABSENT_VALUE, LedgerEntry, LedgerKind
from .sources import (
    SHARED_FIELDS,
    SOURCE_ORDER,
    SOURCE_SCHEMA_VERSION,
    AttemptKey,
    AttemptTraceRecord,
    ClientLatencyRecord,
    EngineHitRecord,
    SourceBundle,
    SourceName,
    TruthBasis,
)

#: Every eighth attempt is a retry of its predecessor, so the join key has to
#: carry the attempt index to stay unique.
_RETRY_STRIDE = 8

#: Node pool the fixture places attempts on.
_NODE_COUNT = 4

#: Fixture token quantum, matching the simulator's block size.
_BLOCK_TOKENS = 512


def synthetic_bundle(*, attempt_count: int, seed: int) -> SourceBundle:
    """Return a fully agreeing bundle of ``attempt_count`` attempts.

    The bundle is a pure function of ``seed``: the same seed reproduces it
    byte-for-byte and a different seed produces a different one.
    """
    if attempt_count < 1:
        raise ValueError("attempt_count must be positive")

    manager = NamedSeedManager(seed)
    engine_hits: list[EngineHitRecord] = []
    client_latencies: list[ClientLatencyRecord] = []
    attempt_traces: list[AttemptTraceRecord] = []

    logical_id = ""
    ordinal = 0
    for index in range(attempt_count):
        stream = manager.stream(f"replay/synthetic/{index}")
        is_retry = index > 0 and index % _RETRY_STRIDE == _RETRY_STRIDE - 1
        if is_retry:
            key = AttemptKey(logical_id, 1)
        else:
            logical_id = f"synthetic:{seed:06d}:{ordinal:012d}"
            ordinal += 1
            key = AttemptKey(logical_id, 0)

        blocks = 1 + stream.randbelow(8)
        input_tokens = _BLOCK_TOKENS * blocks
        hit_tokens = _BLOCK_TOKENS * stream.randbelow(blocks + 1)
        output_tokens = 16 * (1 + stream.randbelow(8))
        node_id = f"node-{stream.randbelow(_NODE_COUNT)}"

        arrival_work = float(index * 40)
        start_work = arrival_work + float(stream.randbelow(20))
        service_work = (input_tokens - hit_tokens) / _BLOCK_TOKENS
        finish_work = start_work + service_work
        # The client cannot see ``start_work``, so its view of the attempt
        # includes the queueing delay. Deriving it from the trace here keeps the
        # fixture's three sources genuinely consistent: a bundle whose client
        # record silently disagreed with its own trace would make a clean ledger
        # meaningless.
        ttft_work = finish_work - arrival_work
        tpot_work = (1 + stream.randbelow(10)) / 10

        engine_hits.append(EngineHitRecord(key, node_id, input_tokens, hit_tokens))
        client_latencies.append(
            ClientLatencyRecord(key, input_tokens, output_tokens, ttft_work, tpot_work)
        )
        attempt_traces.append(
            AttemptTraceRecord(
                key, node_id, input_tokens, arrival_work, start_work, finish_work
            )
        )

    return SourceBundle(
        SOURCE_SCHEMA_VERSION,
        TruthBasis.SYNTHETIC_FIXTURE,
        MachineProvenance.unknown(),
        None,
        tuple(engine_hits),
        tuple(client_latencies),
        tuple(attempt_traces),
    )


@dataclass(frozen=True, slots=True)
class FieldPerturbation:
    """Overwrite one field of one source's record for one attempt."""

    source: SourceName
    key: AttemptKey
    field_name: str
    value: Any


@dataclass(frozen=True, slots=True)
class FaultPlan:
    """Which records to damage, and how.

    ``drop`` and ``duplicate`` are ``(source, key)`` pairs. The unit of damage is
    the ``(source, key)`` pair, not the key: two sources may be damaged
    differently for the same attempt, and that combination is exactly what
    exercises the ledger's ``DUPLICATE > MISSING > DISAGREEMENT`` precedence. A
    plan that forbade it could never falsify the precedence at all.

    What remains contradictory is damaging the *same* ``(source, key)`` twice --
    dropping a record and then perturbing it leaves nothing to perturb -- so a
    repeated target is refused.
    """

    drop: tuple[tuple[SourceName, AttemptKey], ...] = ()
    duplicate: tuple[tuple[SourceName, AttemptKey], ...] = ()
    perturb: tuple[FieldPerturbation, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        seen: set[tuple[SourceName, AttemptKey]] = set()
        for source, key in self.targets():
            if (source, key) in seen:
                raise ValueError(
                    f"fault plan repeats target {source.value}/{key.logical_request_id}"
                    f"#{key.attempt_index}"
                )
            seen.add((source, key))

    def targets(self) -> tuple[tuple[SourceName, AttemptKey], ...]:
        """Every ``(source, key)`` this plan touches, in declaration order."""
        return (
            *self.drop,
            *self.duplicate,
            *((item.source, item.key) for item in self.perturb),
        )

    def affected_keys(self) -> tuple[AttemptKey, ...]:
        return tuple(sorted({key for _, key in self.targets()}))


@dataclass(frozen=True, slots=True)
class InjectionResult:
    """The damaged bundle plus the ledger those damages should produce."""

    bundle: SourceBundle
    expected_ledger: tuple[LedgerEntry, ...]


def _bucket(records: tuple[Any, ...]) -> dict[AttemptKey, list[Any]]:
    buckets: dict[AttemptKey, list[Any]] = {}
    for record in records:
        buckets.setdefault(record.key, []).append(record)
    return buckets


def _rebuild(
    bundle: SourceBundle,
    source: SourceName,
    working: dict[AttemptKey, list[Any]],
) -> tuple[Any, ...]:
    """Flatten a working bucket back into records, preserving original order."""
    emitted: set[AttemptKey] = set()
    out: list[Any] = []
    for record in bundle.records_for(source):
        if record.key in emitted:
            continue
        emitted.add(record.key)
        out.extend(working.get(record.key, ()))
    return tuple(out)


#: Defect precedence, most severe first. Written as data because the oracle must
#: not reproduce the reconciler's control flow; see the module docstring.
_PRECEDENCE: tuple[LedgerKind, ...] = (
    LedgerKind.DUPLICATE,
    LedgerKind.MISSING,
    LedgerKind.DISAGREEMENT,
)

#: A duplicated record is emitted twice, so the count the ledger reports is
#: fixed by the plan rather than counted out of the damaged bundle.
_DUPLICATED_COUNT = "count=2"


def _expected_disagreements(
    key: AttemptKey,
    records: dict[SourceName, Any],
    plan: FaultPlan,
) -> tuple[LedgerEntry, ...]:
    """Entries for every shared field whose observers will not agree.

    The value each observer will report is the plan's value where the plan names
    that ``(source, field)``, and the clean bundle's value otherwise. A field
    only one source can see is not in :data:`SHARED_FIELDS` and so produces no
    entry however far it is perturbed: an unobservable disagreement is not one.
    """
    overrides = {
        (item.source, item.field_name): item.value
        for item in plan.perturb
        if item.key == key
    }
    entries: list[LedgerEntry] = []
    for shared in SHARED_FIELDS:
        values = tuple(
            str(
                overrides.get(
                    (source, shared.field_name),
                    records[source].field(shared.field_name),
                )
            )
            for source in shared.sources
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
                values,
            )
        )
    return tuple(entries)


def _expected_ledger(
    plan: FaultPlan,
    clean: dict[SourceName, dict[AttemptKey, list[Any]]],
) -> tuple[LedgerEntry, ...]:
    """State, from the plan alone, the ledger the damaged bundle should produce."""
    entries: list[LedgerEntry] = []
    for key in plan.affected_keys():
        records = {source: clean[source][key][0] for source in SOURCE_ORDER}
        duplicated = tuple(
            source for source in SOURCE_ORDER if (source, key) in plan.duplicate
        )
        dropped = tuple(source for source in SOURCE_ORDER if (source, key) in plan.drop)

        candidates: dict[LedgerKind, tuple[LedgerEntry, ...]] = {
            LedgerKind.DUPLICATE: (
                (
                    LedgerEntry(
                        LedgerKind.DUPLICATE,
                        key.logical_request_id,
                        key.attempt_index,
                        "",
                        duplicated,
                        (_DUPLICATED_COUNT,) * len(duplicated),
                    ),
                )
                if duplicated
                else ()
            ),
            LedgerKind.MISSING: (
                (
                    LedgerEntry(
                        LedgerKind.MISSING,
                        key.logical_request_id,
                        key.attempt_index,
                        "",
                        dropped,
                        (ABSENT_VALUE,) * len(dropped),
                    ),
                )
                if dropped
                else ()
            ),
            LedgerKind.DISAGREEMENT: _expected_disagreements(key, records, plan),
        }

        for kind in _PRECEDENCE:
            if candidates[kind]:
                entries.extend(candidates[kind])
                break

    entries.sort(key=lambda entry: entry.sort_key)
    return tuple(entries)


def apply_faults(bundle: SourceBundle, plan: FaultPlan) -> InjectionResult:
    """Damage ``bundle`` per ``plan`` and return the ledger it should produce.

    Every attempt the plan touches must be clean to begin with -- present
    exactly once in all three sources. The expected ledger says what the *plan*
    does to a healthy attempt, so a pre-existing duplicate or gap would show up
    in the reconciler's output as a defect the oracle never declared, and the
    resulting mismatch would look like a reconciler bug.
    """
    clean = {
        source: {
            key: list(records)
            for key, records in _bucket(bundle.records_for(source)).items()
        }
        for source in SOURCE_ORDER
    }

    for key in plan.affected_keys():
        for source in SOURCE_ORDER:
            found = len(clean[source].get(key, ()))
            if found != 1:
                raise ValueError(
                    f"fault target {key.logical_request_id}#{key.attempt_index} "
                    f"must appear exactly once in {source.value}, found {found}"
                )
    for item in plan.perturb:
        clean[item.source][item.key][0].field(item.field_name)

    working = {
        source: {key: list(records) for key, records in buckets.items()}
        for source, buckets in clean.items()
    }

    for source, key in plan.drop:
        del working[source][key]
    for source, key in plan.duplicate:
        working[source][key] = working[source][key] * 2
    for item in plan.perturb:
        working[item.source][item.key] = [
            replace(record, **{item.field_name: item.value})
            for record in working[item.source][item.key]
        ]

    injected = bundle
    for source in SOURCE_ORDER:
        injected = injected.with_records(
            source, _rebuild(bundle, source, working[source])
        )
    return InjectionResult(injected, _expected_ledger(plan, clean))
