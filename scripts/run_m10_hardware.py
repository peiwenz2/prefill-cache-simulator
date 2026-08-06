#!/usr/bin/env python3
"""Validate the M10 replay against measured engine data, or say why it could not.

This is the hardware counterpart of ``run_m10_synthetic.py``. That script scores
four arms in the local simulator and compares their ranking across two arrival
scales -- a *stability* statistic that needs no engine. This one asks the harder
question the M10 gate was written for: does the simulator rank the arms the same
way a real engine does?

Answering it needs two things this environment does not have, and the script is
built to be honest about their absence rather than to fail:

``--calibration DIR``
    An *accepted* M9-HW artifact directory. A synthetic or rejected calibration
    is refused (``BLOCKED_SYNTHETIC_CALIBRATION`` /
    ``BLOCKED_CALIBRATION_NOT_ACCEPTED``), because a validated ranking on top of
    an uncalibrated cost model validates nothing.
``--observed FILE``
    Measured per-cell scores and source bundles from a real run of the frozen
    plan. Every bundle must declare ``MEASURED_ENGINE``, which
    :class:`SourceBundle` already couples to complete machine provenance, so a
    hand-written file cannot claim measurement without naming the machine.

With neither, the script still writes a report: a :class:`ReplayHardwareReport`
naming ``BLOCKED_NO_ENGINE_ACCESS`` first among its blockers, in
``results/m10-hardware-blocked``. A reviewer asking "was M10 validated on
hardware?" gets a machine-readable no.

Three properties keep the accepted path from overclaiming.

*The modeled replay stays modeled.* :func:`run_replay` hard-codes
``SYNTHETIC_UNCALIBRATED`` / ``NORMALIZED_WORK`` and would raise
:class:`DishonestLabelError` if asked for a hardware tier. It is therefore
always run at ``SYNTHETIC_REPLAY`` here, even on an accepted run. The simulator
does not become a measurement by being compared to one.

*Only the gate report may say ``HW_VALIDATED``.*
:func:`_assert_no_stronger_claim` permits the hardware labels in ``GATE.json``
and nowhere else, and only when the gate accepted. Every other artifact carries
the synthetic labels because that is what produced its numbers.

*Aggregation takes the worst cell, not the mean.* ``tau_b`` is the worst metric
and the reconciliation fractions are the worst cell, so a weak corner cannot be
diluted by adding strong ones.

``fault_injection.csv`` is written on both paths. It needs no engine, and an
empty measured ledger is only evidence if the reconciler that produced it can be
shown to report defects when they exist.

Exit codes: ``0`` accepted, ``2`` rejected with a report written, ``1`` the
script itself failed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prefill_cache_sim.calibration import (
    CalibrationStatus,
    DishonestLabelError,
    EvidenceTier,
    HardwareContext,
    HardwareGateReport,
    MachineProvenance,
    TimeUnit,
)
from prefill_cache_sim.config import git_provenance
from prefill_cache_sim.replay import (
    DEFAULT_REPLAY_HARDWARE_GATE,
    DEFAULT_SHADOW_GATE,
    ENFORCEMENT_ENABLED,
    FROZEN_PLAN_DIGEST,
    FROZEN_RANKING_STATISTIC,
    SCORE_METRICS,
    AttemptKey,
    FaultPlan,
    FieldPerturbation,
    LedgerEntry,
    RankingComparison,
    ReplayHardwareEvidence,
    ReplayHardwareGate,
    ReplayOutcome,
    ReplayPlan,
    ShadowDecision,
    SourceBundle,
    SourceName,
    TruthBasis,
    UndefinedRankingError,
    apply_faults,
    plan_digest,
    reconcile,
    run_replay,
    source_manifest,
    synthetic_bundle,
)
from prefill_cache_sim.trace import load_trace, to_simulation_requests

#: Recorded in the provenance block so the manifest names the generator that
#: actually ran rather than the synthetic one.
GENERATOR_PATH = "scripts/run_m10_hardware.py"

BLOCK_SIZE_TOKENS = 512

#: The modeled side is always the simulator, whatever the measured side proves.
MODELED_EVIDENCE_TIER = EvidenceTier.SYNTHETIC_REPLAY
MODELED_CALIBRATION_STATUS = CalibrationStatus.SYNTHETIC_UNCALIBRATED
MODELED_TIME_UNIT = TimeUnit.NORMALIZED_WORK

ACCEPTED_DIR = "results/m10-hardware"
BLOCKED_DIR = "results/m10-hardware-blocked"

MANIFEST_NAME = "MANIFEST.json"
MANIFEST_SCHEMA_VERSION = "m10-hardware-manifest-v1"
GATE_NAME = "GATE.json"

#: Version of the measured-input document this script accepts. Bumped rather
#: than extended, so an older file is rejected instead of half-understood.
OBSERVED_SCHEMA_VERSION = "m10-hardware-observed-v2"

#: Names of the M9-HW artifacts consumed from ``--calibration``.
CALIBRATION_GATE_NAME = "GATE.json"
CALIBRATION_PARAMS_NAME = "params.json"

#: Labels only an accepted run may publish, and only in ``GATE.json``.
HARDWARE_LABELS = (
    CalibrationStatus.HW_CALIBRATED.value,
    TimeUnit.MILLISECONDS.value,
    EvidenceTier.HW_VALIDATED.value,
)

#: The four score metrics are cache hit-rate fractions. They carry no time unit,
#: so comparing a modeled ranking to a measured one makes no wall-clock claim.
SCORE_UNIT_NOTE = "SCORES_ARE_UNITLESS_HIT_RATE_FRACTIONS_NOT_LATENCIES"

#: Inherited simulator field names carry an ``_ms`` suffix and are normalized
#: work units, not milliseconds.
MODELED_UNIT_NOTE = "FIELDS_SUFFIXED_MS_ARE_NORMALIZED_WORK_NOT_WALL_CLOCK"

FAULT_ATTEMPTS = 64
FAULT_SEED = 20261010

LEDGER_FIELDS = (
    "scope",
    "kind",
    "logical_request_id",
    "attempt_index",
    "field_name",
    "sources",
    "values",
)


class ObservedError(ValueError):
    """A measured-input or calibration document this script refuses to read.

    Raised during loading and caught in :func:`main`, where it becomes a string
    in the report rather than a traceback. A malformed measurement is a reason
    the gate blocks, and a reason is only useful if it reaches the artifact.
    """


# --------------------------------------------------------------------------
# artifact plumbing
# --------------------------------------------------------------------------


def _json_bytes(payload: object) -> bytes:
    # allow_nan=False: a NaN score that survived validation must fail loudly
    # here rather than ship as a literal no JSON reader accepts.
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _csv_bytes(
    rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]
) -> bytes:
    # newline="" keeps csv's own \r\n terminators intact, exactly as writing
    # through an opened file handle would, so the bytes are the artifact.
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _manifest_bytes(artifacts: Mapping[str, bytes]) -> bytes:
    return _json_bytes(
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "algorithm": "sha256",
            "note": (
                "Digests of the artifacts generated by this run. This file is "
                "replaced last, so a reader that observes a digest mismatch is "
                "looking at a partially replaced set, not at one run's output."
            ),
            "files": {
                name: hashlib.sha256(payload).hexdigest()
                for name, payload in sorted(artifacts.items())
            },
        }
    )


def _claim_text(name: str, payload: bytes) -> str:
    """Return the part of an artifact that makes a claim about this run.

    A gate report names the bar it judges against, so ``HW_CALIBRATED`` appears
    in its policy even when the run was blocked for the want of exactly that.
    Stating a requirement is not claiming to have met it, so the policy block is
    dropped before the scan. Nothing else is: every field the report uses to
    label its own output, and every other artifact in full, is read as written.
    """
    if name != GATE_NAME:
        return payload.decode("utf-8")
    document = json.loads(payload.decode("utf-8"), parse_constant=_reject_constant)
    report = document.get("report")
    if isinstance(report, dict):
        report.pop("policy", None)
    return json.dumps(document, sort_keys=True)


def _assert_no_stronger_claim(
    name: str, payload: bytes, *, allow_hardware_labels: bool
) -> None:
    """Refuse to publish a hardware label this artifact did not earn.

    The permission is per artifact rather than per run. An accepted run does
    earn ``HW_VALIDATED``, but only for the gate report: the replay numbers next
    to it still came out of the simulator, and an artifact that named a hardware
    tier beside them would invite exactly the misreading this whole script
    exists to prevent.
    """
    if allow_hardware_labels:
        return
    text = _claim_text(name, payload)
    for label in HARDWARE_LABELS:
        if label in text:
            raise RuntimeError(
                f"{name} claims {label}, which only an accepted {GATE_NAME} may"
            )


def _write_artifacts(output_dir: Path, artifacts: Mapping[str, bytes]) -> None:
    """Stage every artifact, then move each into place with :func:`os.replace`.

    Per-file replacement rather than swapping the directory, so hand-written
    files that live alongside the generated ones survive. ``MANIFEST.json`` goes
    last and covers the rest, so a reader who catches the window between renames
    can detect the mix instead of trusting it.
    """
    if MANIFEST_NAME not in artifacts:
        raise RuntimeError(f"artifact set must include {MANIFEST_NAME}")
    ordered = [name for name in sorted(artifacts) if name != MANIFEST_NAME]
    ordered.append(MANIFEST_NAME)

    output_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=output_dir.parent, prefix=".m10hw-staging-"))
    try:
        for name in ordered:
            (staging / name).write_bytes(artifacts[name])
        for name in ordered:
            os.replace(staging / name, output_dir / name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------


def _reject_constant(token: str) -> float:
    raise ObservedError(f"{token} is not valid JSON and is refused at the boundary")


def _read_json(path: Path, context: str) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ObservedError(f"{context}: cannot read {path}: {error}") from error
    try:
        return json.loads(text, parse_constant=_reject_constant)
    except ObservedError:
        raise
    except (ValueError, UnicodeDecodeError) as error:
        raise ObservedError(f"{context}: {path} is not valid JSON: {error}") from error


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ObservedError(
            f"{context}: expected an object, got {type(value).__name__}"
        )
    return value


def _require_object(
    data: Mapping[str, Any], key: str, context: str
) -> Mapping[str, Any]:
    if key not in data:
        raise ObservedError(f"{context}: missing {key!r}")
    return _require_mapping(data[key], f"{context}.{key}")


def _require_sequence(data: Mapping[str, Any], key: str, context: str) -> Sequence[Any]:
    if key not in data:
        raise ObservedError(f"{context}: missing {key!r}")
    value = data[key]
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ObservedError(f"{context}.{key}: expected a list")
    return value


def _require_str(data: Mapping[str, Any], key: str, context: str) -> str:
    if key not in data:
        raise ObservedError(f"{context}: missing {key!r}")
    value = data[key]
    if not isinstance(value, str) or not value:
        raise ObservedError(f"{context}.{key}: expected a non-empty string")
    return value


def _require_sha256(data: Mapping[str, Any], key: str, context: str) -> str:
    value = _require_str(data, key, context)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ObservedError(f"{context}.{key}: expected lowercase SHA-256 hex")
    return value


def _require_bool(data: Mapping[str, Any], key: str, context: str) -> bool:
    if key not in data:
        raise ObservedError(f"{context}: missing {key!r}")
    value = data[key]
    if not isinstance(value, bool):
        raise ObservedError(f"{context}.{key}: expected a bool")
    return value


def _require_float(data: Mapping[str, Any], key: str, context: str) -> float:
    if key not in data:
        raise ObservedError(f"{context}: missing {key!r}")
    value = data[key]
    # bool is an int in Python, so it would otherwise pass as 0.0 or 1.0 and a
    # typo would become a score.
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ObservedError(f"{context}.{key}: expected a number")
    number = float(value)
    if not math.isfinite(number):
        raise ObservedError(f"{context}.{key}: expected a finite number")
    return number


@dataclass(frozen=True, slots=True)
class ObservedCell:
    """One measured (arm, arrival scale) cell of the frozen plan."""

    arm_id: str
    arrival_scale: float
    scores: Mapping[str, float]
    bundle: SourceBundle


@dataclass(frozen=True, slots=True)
class ObservedDocument:
    """The whole measured input, after validation."""

    plan_digest: str
    endpoint_id: str
    calibration_manifest_sha256: str
    producer_run_id: str
    cells: tuple[ObservedCell, ...]

    def machines(self) -> tuple[MachineProvenance, ...]:
        return tuple(cell.bundle.machine for cell in self.cells)

    def scores(self, metric: str, arrival_scale: float) -> dict[str, float]:
        return {
            cell.arm_id: cell.scores[metric]
            for cell in self.cells
            if cell.arrival_scale == arrival_scale
        }


def _load_observed(path: Path, plan: ReplayPlan) -> ObservedDocument:
    """Read measured scores and bundles, refusing anything short of the plan.

    Coverage is checked against the frozen plan rather than against whatever the
    file happens to contain, so a document that measured one favourable arm
    cannot be compared to a four-arm modeled ranking. Partial coverage is an
    error here rather than a low ``tau_b`` later, because a low ``tau_b`` would
    be reported as a disagreement between the model and the engine when it was
    really a disagreement about which experiment was run.
    """
    context = "observed"
    data = _require_mapping(_read_json(path, context), context)

    version = _require_str(data, "schema_version", context)
    if version != OBSERVED_SCHEMA_VERSION:
        raise ObservedError(
            f"{context}.schema_version must be {OBSERVED_SCHEMA_VERSION!r}, "
            f"got {version!r}"
        )

    expected: set[tuple[str, float]] = {
        (arm.arm_id, scale) for arm in plan.arms for scale in plan.arrival_scales
    }
    seen: set[tuple[str, float]] = set()
    cells: list[ObservedCell] = []

    for index, item in enumerate(_require_sequence(data, "cells", context)):
        cell_context = f"{context}.cells[{index}]"
        entry = _require_mapping(item, cell_context)
        arm_id = _require_str(entry, "arm_id", cell_context)
        arrival_scale = _require_float(entry, "arrival_scale", cell_context)
        key = (arm_id, arrival_scale)
        if key not in expected:
            raise ObservedError(
                f"{cell_context}: ({arm_id}, x{arrival_scale:g}) is not a cell of "
                "the frozen plan"
            )
        if key in seen:
            raise ObservedError(
                f"{cell_context}: ({arm_id}, x{arrival_scale:g}) appears twice"
            )
        seen.add(key)

        raw_scores = _require_object(entry, "scores", cell_context)
        scores = {
            metric: _require_float(raw_scores, metric, f"{cell_context}.scores")
            for metric in SCORE_METRICS
        }

        try:
            bundle = SourceBundle.from_dict(
                dict(_require_object(entry, "bundle", cell_context))
            )
        except ObservedError:
            raise
        except Exception as error:  # noqa: BLE001 - re-raised as one taxonomy
            raise ObservedError(f"{cell_context}.bundle: {error}") from error
        if bundle.truth_basis is not TruthBasis.MEASURED_ENGINE:
            raise ObservedError(
                f"{cell_context}.bundle.truth_basis must be "
                f"{TruthBasis.MEASURED_ENGINE.value}, got {bundle.truth_basis.value}"
            )
        cells.append(ObservedCell(arm_id, arrival_scale, scores, bundle))

    missing = sorted(f"{arm}@x{scale:g}" for arm, scale in expected - seen)
    if missing:
        raise ObservedError(f"{context}.cells does not cover the plan: {missing}")

    return ObservedDocument(
        _require_str(data, "plan_digest", context),
        _require_str(data, "endpoint_id", context),
        _require_sha256(data, "calibration_manifest_sha256", context),
        _require_str(data, "producer_run_id", context),
        tuple(cells),
    )


@dataclass(frozen=True, slots=True)
class CalibrationArtifact:
    """The parts of an M9-HW artifact directory this gate depends on.

    ``report`` is re-derived from the artifact's own evidence and policy via
    :meth:`HardwareGateReport.from_dict`, which re-judges the blockers. A
    hand-edited ``accepted`` or ``blockers`` therefore fails on the way in.
    """

    report: HardwareGateReport
    context: HardwareContext
    endpoint_id: str | None
    git_dirty: bool | None
    source_combined_digest: str | None
    manifest_sha256: str


#: Files that must exist in an accepted M9-HW directory.
_REQUIRED_CALIBRATION_FILES: tuple[str, ...] = (
    "GATE.json",
    "params.json",
    "results.csv",
    "observations.csv",
    "MANIFEST.json",
)


def _verify_manifest(directory: Path, context: str) -> Mapping[str, str]:
    """Read MANIFEST.json and verify every file digest matches the bytes on disk.

    A manifest that omits a file or whose hash does not match is a tampered or
    partially-replaced artifact, not a calibration. The check covers every file
    the manifest names, so a hand-edit after the manifest was written is caught.
    """
    manifest_path = directory / "MANIFEST.json"
    if not manifest_path.is_file():
        raise ObservedError(f"{context}: missing MANIFEST.json")
    manifest_data = _require_mapping(_read_json(manifest_path, context), context)
    files = _require_object(manifest_data, "files", context)
    for name in sorted(files):
        expected = files[name]
        if not isinstance(expected, str) or not expected:
            raise ObservedError(f"{context}.files.{name}: invalid digest")
        path = directory / name
        if not path.is_file():
            raise ObservedError(f"{context}: manifest lists {name} but file is absent")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ObservedError(
                f"{context}: {name} digest mismatch (manifest={expected[:12]}…, "
                f"actual={actual[:12]}…): artifact is tampered or partially replaced"
            )
    return files


def _load_calibration(directory: Path) -> CalibrationArtifact:
    """Read an M9-HW artifact directory as a complete immutable artifact.

    The directory is treated as immutable: the MANIFEST covers every file and
    every digest is verified, the gate report is re-derived from its own
    evidence and policy so a hand-edited verdict fails, and the git dirty flag
    is checked so a build from uncommitted code is not a trusted calibration.
    """
    context = "calibration"

    for name in _REQUIRED_CALIBRATION_FILES:
        if not (directory / name).is_file():
            raise ObservedError(f"{context}: missing expected file {name}")

    manifest_files = _verify_manifest(directory, context)
    for name in _REQUIRED_CALIBRATION_FILES:
        if name != "MANIFEST.json" and name not in manifest_files:
            raise ObservedError(
                f"{context}: {name} is not covered by MANIFEST.json"
            )

    gate_doc = _require_mapping(
        _read_json(directory / CALIBRATION_GATE_NAME, context), context
    )
    report_dict = _require_object(gate_doc, "report", context)
    try:
        report = HardwareGateReport.from_dict(dict(report_dict))
    except (ValueError, DishonestLabelError) as error:
        raise ObservedError(
            f"{context}.report: re-derivation failed "
            f"(tampered or inconsistent): {error}"
        ) from error

    provenance = _require_object(gate_doc, "provenance", f"{context}.gate")
    git_dirty_raw = provenance.get("git_dirty")
    git_dirty: bool | None = None
    if isinstance(git_dirty_raw, bool):
        git_dirty = git_dirty_raw
    if git_dirty:
        raise ObservedError(
            f"{context}: M9 artifact was built from a dirty git tree; "
            "cannot be trusted as an immutable calibration"
        )
    source_manifest_obj = provenance.get("source_fingerprints")
    source_combined_digest: str | None = None
    if isinstance(source_manifest_obj, Mapping):
        combined = source_manifest_obj.get("combined_digest")
        if isinstance(combined, str):
            source_combined_digest = combined

    endpoint_id: str | None = None
    params_path = directory / CALIBRATION_PARAMS_NAME
    if params_path.exists():
        params_doc = _require_mapping(
            _read_json(params_path, f"{context}.params"), f"{context}.params"
        )
        params = _require_object(params_doc, "params", f"{context}.params")
        endpoint_id = _require_str(params, "endpoint_id", f"{context}.params.params")

    return CalibrationArtifact(
        report,
        report.context,
        endpoint_id,
        git_dirty,
        source_combined_digest,
        hashlib.sha256((directory / "MANIFEST.json").read_bytes()).hexdigest(),
    )


# --------------------------------------------------------------------------
# fault injection (engine-independent)
# --------------------------------------------------------------------------


def _ledger_row(scope: str, entry: LedgerEntry) -> dict[str, Any]:
    return {
        "scope": scope,
        "kind": entry.kind.value,
        "logical_request_id": entry.logical_request_id,
        "attempt_index": entry.attempt_index,
        "field_name": entry.field_name,
        "sources": "|".join(source.value for source in entry.sources),
        "values": "|".join(entry.values),
    }


def _fault_plan(keys: Mapping[str, AttemptKey]) -> FaultPlan:
    """Damage three different attempts in three different ways."""
    return FaultPlan(
        drop=((SourceName.ENGINE_HIT, keys["dropped"]),),
        duplicate=((SourceName.CLIENT_LATENCY, keys["duplicated"]),),
        perturb=(
            FieldPerturbation(
                SourceName.ATTEMPT_TRACE, keys["perturbed"], "node_id", "node-injected"
            ),
        ),
    )


def _fault_injection_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reconcile a bundle whose defects are known in advance.

    Deliberately identical to the synthetic runner's check, and deliberately run
    on both paths. A measured ledger that comes back empty is only evidence of a
    healthy join if the same reconciler can be shown to report defects when they
    are present, and that demonstration needs no engine.
    """
    clean = synthetic_bundle(attempt_count=FAULT_ATTEMPTS, seed=FAULT_SEED)
    retries = [
        record.key for record in clean.engine_hits if record.key.attempt_index > 0
    ]
    keys = {
        "dropped": clean.engine_hits[3].key,
        "duplicated": clean.engine_hits[5].key,
        "perturbed": retries[0],
    }
    injection = apply_faults(clean, _fault_plan(keys))
    recovered = reconcile(injection.bundle)
    if recovered.ledger != injection.expected_ledger:
        raise RuntimeError("reconciler did not recover the injected ledger exactly")

    rows = [_ledger_row("EXPECTED", entry) for entry in injection.expected_ledger]
    rows += [_ledger_row("RECOVERED", entry) for entry in recovered.ledger]
    summary = {
        "attempt_count": recovered.attempt_count,
        "expected_entries": len(injection.expected_ledger),
        "recovered_entries": len(recovered.ledger),
        "exact_match": True,
        "seed": FAULT_SEED,
        "targets": {name: key.to_dict() for name, key in keys.items()},
    }
    return rows, summary


# --------------------------------------------------------------------------
# modeled vs measured
# --------------------------------------------------------------------------


def _model_vs_measured(
    outcome: ReplayOutcome, observed: ObservedDocument, arrival_scale: float
) -> dict[str, RankingComparison | None]:
    """Compare the modeled and measured arm rankings, one metric at a time.

    This is not the comparison :meth:`ReplayOutcome.ranking_comparison` makes.
    That one holds the simulator fixed and varies the arrival scale, which
    measures whether the *model* is stable. This one holds the arrival scale
    fixed and varies the source of the numbers, which is the only comparison
    that can tell anyone whether the model is right.
    """
    comparisons: dict[str, RankingComparison | None] = {}
    for metric in SCORE_METRICS:
        modeled = outcome.scores(metric, arrival_scale)
        measured = observed.scores(metric, arrival_scale)
        try:
            comparisons[metric] = RankingComparison.from_scores(
                f"MODELED@x{arrival_scale:g}",
                f"MEASURED@x{arrival_scale:g}",
                modeled,
                measured,
            )
        except UndefinedRankingError:
            # Every score tied on one side, so no ranking exists to agree with.
            # Recorded as absent rather than as a tau_b of 0, which would read
            # as measured disagreement instead of an undefined comparison.
            comparisons[metric] = None
    return comparisons


def _worst_tau_b(
    comparisons: Mapping[str, RankingComparison | None],
) -> tuple[float | None, str | None]:
    """Return the weakest metric's ``tau_b``, or ``None`` if any is undefined.

    Worst rather than mean: a mean lets three strong metrics carry a fourth that
    disagrees with the engine, and the gate exists to catch exactly that metric.
    An undefined comparison collapses the whole result to ``None`` for the same
    reason -- it cannot be shown to pass, so it must not be averaged away.
    """
    worst: float | None = None
    worst_metric: str | None = None
    for metric in SCORE_METRICS:
        comparison = comparisons.get(metric)
        if comparison is None:
            return None, metric
        if worst is None or comparison.tau_b < worst:
            worst = comparison.tau_b
            worst_metric = metric
    return worst, worst_metric


@dataclass(frozen=True, slots=True)
class MeasuredJoin:
    """Reconciliation health across the measured bundles."""

    rows: tuple[dict[str, Any], ...]
    ledger_rows: tuple[dict[str, Any], ...]
    reconciled_fraction: float | None
    disagreement_fraction: float | None


def _reconcile_observed(observed: ObservedDocument) -> MeasuredJoin:
    """Reconcile every measured bundle and aggregate on the worst cell.

    Minimum reconciled fraction and maximum disagreement fraction, not means.
    Averaging would let a run add well-joined cells until a badly-joined one
    stopped mattering, which is the opposite of what a join-health gate is for.
    """
    rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    reconciled: float | None = None
    disagreement: float | None = None

    for cell in observed.cells:
        result = reconcile(cell.bundle)
        rows.append(
            {
                "arm_id": cell.arm_id,
                "arrival_scale": cell.arrival_scale,
                "attempt_count": result.attempt_count,
                "reconciled_count": result.reconciled_count,
                "reconciled_fraction": result.reconciled_fraction,
                "disagreement_fraction": result.disagreement_fraction,
                "ledger_entries": len(result.ledger),
                "truth_basis": cell.bundle.truth_basis.value,
                "tpot_reason": cell.bundle.tpot_reason,
                "scope": "MEASURED_ENGINE",
            }
        )
        for entry in result.ledger:
            if LedgerEntry.from_dict(entry.to_dict()) != entry:
                raise RuntimeError(f"{cell.arm_id}: ledger entry failed to round-trip")
            ledger_rows.append(
                {
                    "arm_id": cell.arm_id,
                    "arrival_scale": cell.arrival_scale,
                    **_ledger_row("MEASURED_ENGINE", entry),
                }
            )
        reconciled = (
            result.reconciled_fraction
            if reconciled is None
            else min(reconciled, result.reconciled_fraction)
        )
        disagreement = (
            result.disagreement_fraction
            if disagreement is None
            else max(disagreement, result.disagreement_fraction)
        )

    return MeasuredJoin(tuple(rows), tuple(ledger_rows), reconciled, disagreement)


def _one_machine(
    calibration: CalibrationArtifact | None, observed: ObservedDocument | None
) -> tuple[MachineProvenance, bool]:
    """Name the machine this run measured, or refuse to name one.

    The report checks ``evidence.provenance_complete`` against
    ``machine.complete``, so a conflict cannot be recorded as "complete but
    disputed". It is recorded as no machine at all, which blocks, plus a
    ``provenance_conflict`` flag in the artifact so a reader can tell a conflict
    from a plain absence.
    """
    candidates: list[MachineProvenance] = []
    if calibration is not None:
        candidates.append(calibration.context.machine)
    if observed is not None:
        candidates.extend(observed.machines())
    if not candidates:
        return MachineProvenance.unknown(), False
    first = candidates[0]
    if any(other != first for other in candidates[1:]):
        return MachineProvenance.unknown(), True
    return first, False


def _assert_modeled_round_trips(
    comparisons: Mapping[str, RankingComparison | None],
    shadow: Mapping[str, tuple[ShadowDecision, ...]],
) -> None:
    """Check every published record survives its own serialization."""
    for metric, comparison in comparisons.items():
        if comparison is None:
            continue
        if RankingComparison.from_dict(comparison.to_dict()) != comparison:
            raise RuntimeError(f"{metric}: ranking comparison failed to round-trip")
    for metric, decisions in shadow.items():
        for decision in decisions:
            if decision.enforced:
                raise RuntimeError(
                    f"{metric}: shadow decision for {decision.candidate_arm_id} "
                    "reports itself enforced"
                )
            if decision.policy != DEFAULT_SHADOW_GATE:
                raise RuntimeError(
                    f"{metric}: shadow decision for {decision.candidate_arm_id} "
                    f"was judged against {decision.policy.to_dict()}, not the "
                    f"default gate {DEFAULT_SHADOW_GATE.to_dict()}"
                )
            if ShadowDecision.from_dict(decision.to_dict()) != decision:
                raise RuntimeError(f"{metric}: shadow decision failed to round-trip")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate M10 against measured engine data. With no --calibration "
            "and no --observed this writes a BLOCKED_NO_ENGINE_ACCESS report "
            "and exits 2."
        )
    )
    parser.add_argument(
        "--calibration",
        default=None,
        help=(
            "Directory holding an accepted M9-HW artifact "
            f"({CALIBRATION_GATE_NAME} and {CALIBRATION_PARAMS_NAME}). A "
            "rejected or synthetic calibration is refused."
        ),
    )
    parser.add_argument(
        "--observed",
        default=None,
        help=(
            f"JSON document with schema_version {OBSERVED_SCHEMA_VERSION}: "
            "measured scores and MEASURED_ENGINE source bundles for every cell "
            "of the frozen plan."
        ),
    )
    parser.add_argument(
        "--trace",
        default="mooncake_trace.jsonl",
        help="Trace replayed by the modeled side, relative to the repo root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Load and validate the inputs without claiming validation. Always "
            "rejected (BLOCKED_DRY_RUN_ONLY): reading a file is not a replay."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Root the artifact directories are written under. Defaults to the "
            "repository root; a test can point it at a temporary directory "
            "without writing a hardware result into the repo."
        ),
    )
    parser.add_argument(
        "--minimum-tau-b",
        type=float,
        default=DEFAULT_REPLAY_HARDWARE_GATE.minimum_tau_b,
        help="Override the frozen ranking-agreement threshold. Recorded.",
    )
    parser.add_argument(
        "--minimum-reconciled-fraction",
        type=float,
        default=DEFAULT_REPLAY_HARDWARE_GATE.minimum_reconciled_fraction,
        help="Override the frozen join-completeness threshold. Recorded.",
    )
    parser.add_argument(
        "--maximum-disagreement-fraction",
        type=float,
        default=DEFAULT_REPLAY_HARDWARE_GATE.maximum_disagreement_fraction,
        help="Override the frozen cross-source disagreement ceiling. Recorded.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    provenance = git_provenance(root)
    plan = ReplayPlan()

    # The plan digest is derived here and compared to the frozen constant, so a
    # change to ReplayPlan's defaults fails now rather than silently redefining
    # what "the frozen plan" means for every artifact after it.
    observed_plan_digest_local = plan_digest(plan)

    production = (
        args.minimum_tau_b == DEFAULT_REPLAY_HARDWARE_GATE.minimum_tau_b
        and args.minimum_reconciled_fraction
        == DEFAULT_REPLAY_HARDWARE_GATE.minimum_reconciled_fraction
        and args.maximum_disagreement_fraction
        == DEFAULT_REPLAY_HARDWARE_GATE.maximum_disagreement_fraction
    )
    gate = ReplayHardwareGate(
        minimum_tau_b=args.minimum_tau_b,
        minimum_reconciled_fraction=args.minimum_reconciled_fraction,
        maximum_disagreement_fraction=args.maximum_disagreement_fraction,
        production=production,
    )

    calibration: CalibrationArtifact | None = None
    calibration_error: str | None = None
    if args.calibration:
        try:
            calibration = _load_calibration(Path(args.calibration).resolve())
        except ObservedError as error:
            calibration_error = str(error)

    observed: ObservedDocument | None = None
    observed_error: str | None = None
    if args.observed:
        try:
            observed = _load_observed(Path(args.observed).resolve(), plan)
        except ObservedError as error:
            observed_error = str(error)

    # Engine-independent, so it runs on every path. An empty measured ledger is
    # only evidence if this can fail.
    fault_rows, fault_summary = _fault_injection_rows()

    machine, provenance_conflict = _one_machine(calibration, observed)

    outcome: ReplayOutcome | None = None
    comparisons: dict[str, RankingComparison | None] = {}
    shadow: dict[str, tuple[ShadowDecision, ...]] = {}
    join: MeasuredJoin | None = None
    trace_sha256: str | None = None
    trace_requests: int | None = None
    worst_metric: str | None = None
    tau_b: float | None = None

    if observed is not None:
        loaded = load_trace(root / args.trace, block_size_tokens=BLOCK_SIZE_TOKENS)
        requests = to_simulation_requests(loaded)
        trace_sha256 = loaded.sha256
        trace_requests = len(requests)
        # SYNTHETIC_REPLAY, and MachineProvenance.unknown(), on purpose. This
        # side is the simulator whatever the measured side proves; run_replay
        # would raise DishonestLabelError if asked for a hardware tier, and the
        # honest reason is that no hardware produced these numbers.
        outcome = run_replay(
            plan,
            requests,
            trace_sha256=loaded.sha256,
            git_sha=provenance.sha,
            git_dirty=provenance.dirty,
            evidence_tier=MODELED_EVIDENCE_TIER,
            machine=MachineProvenance.unknown(),
        )
        if outcome.calibration_status is not MODELED_CALIBRATION_STATUS:
            raise RuntimeError("the modeled side must stay SYNTHETIC_UNCALIBRATED")
        if outcome.time_unit is not MODELED_TIME_UNIT:
            raise RuntimeError("the modeled side must stay in NORMALIZED_WORK")
        if outcome.evidence_tier is not MODELED_EVIDENCE_TIER:
            raise RuntimeError("the modeled side must stay SYNTHETIC_REPLAY")
        if outcome.machine != MachineProvenance.unknown():
            raise RuntimeError(
                "the modeled side drives the local simulator and must not report "
                f"machine provenance, got {outcome.machine.to_dict()}"
            )

        primary_scale = plan.arrival_scales[0]
        comparisons = _model_vs_measured(outcome, observed, primary_scale)
        tau_b, worst_metric = _worst_tau_b(comparisons)
        shadow = {metric: outcome.shadow_decisions(metric) for metric in SCORE_METRICS}
        _assert_modeled_round_trips(comparisons, shadow)
        join = _reconcile_observed(observed)

    if ENFORCEMENT_ENABLED:
        raise RuntimeError("M10 records shadow decisions and must never enforce them")

    # Bind observed evidence to the M9 calibration: the engine was reachable
    # only if the calibration says it was, not merely because a JSON file parsed.
    # The endpoint_id and machine must also match, so a measured bundle from one
    # engine cannot be grafted onto a calibration from another.
    cal_engine_reachable = False
    cal_accepted = False
    cal_synthetic = True
    cal_status: str | None = None
    calibration_bound = True
    if calibration is not None:
        cal_engine_reachable = calibration.report.evidence.engine_reachable
        cal_accepted = calibration.report.accepted
        cal_synthetic = calibration.report.evidence.endpoint_is_synthetic
        cal_status = calibration.report.calibration_status.value
        if observed is not None:
            if calibration.endpoint_id is not None and (
                observed.endpoint_id != calibration.endpoint_id
            ):
                calibration_bound = False
            if (
                observed.calibration_manifest_sha256
                != calibration.manifest_sha256
            ):
                calibration_bound = False
            for cell_machine in observed.machines():
                if cell_machine != calibration.context.machine:
                    calibration_bound = False

    evidence = ReplayHardwareEvidence(
        engine_reachable=(
            observed is not None
            and cal_engine_reachable
            and cal_accepted
            and calibration_bound
        ),
        calibration_status=cal_status,
        calibration_accepted=cal_accepted and calibration_bound,
        calibration_endpoint_synthetic=cal_synthetic,
        provenance_complete=machine.complete,
        observed_plan_digest=None if observed is None else observed.plan_digest,
        ranking_statistic=None if not comparisons else FROZEN_RANKING_STATISTIC,
        tau_b=tau_b,
        reconciled_fraction=None if join is None else join.reconciled_fraction,
        disagreement_fraction=None if join is None else join.disagreement_fraction,
        fault_injection_detected=bool(fault_summary["exact_match"]),
        dry_run=bool(args.dry_run),
    )
    report = gate.evaluate(evidence, machine)
    accepted = report.accepted

    labels = {
        "evidence_tier": MODELED_EVIDENCE_TIER.value,
        "calibration_status": MODELED_CALIBRATION_STATUS.value,
        "time_unit": MODELED_TIME_UNIT.value,
        "unit_note": MODELED_UNIT_NOTE,
    }

    gate_payload: dict[str, Any] = {
        "report": report.to_dict(),
        "calibration_error": calibration_error,
        "observed_error": observed_error,
        "calibration_dir_provided": bool(args.calibration),
        "observed_file_provided": bool(args.observed),
        "provenance_conflict": provenance_conflict,
        "worst_ranking_metric": worst_metric,
        "local_plan_digest": observed_plan_digest_local,
        "frozen_plan_digest": FROZEN_PLAN_DIGEST,
        "score_unit_note": SCORE_UNIT_NOTE,
        "fault_injection": fault_summary,
        "provenance": {
            "git_sha": provenance.sha,
            "git_dirty": provenance.dirty,
            "trace_sha256": trace_sha256,
            "trace_requests": trace_requests,
            "block_size_tokens": BLOCK_SIZE_TOKENS,
            "modeled_labels": labels,
            "source_fingerprints": source_manifest(root, GENERATOR_PATH),
        },
    }

    artifacts: dict[str, bytes] = {
        GATE_NAME: _json_bytes(gate_payload),
        "fault_injection.csv": _csv_bytes(fault_rows, list(LEDGER_FIELDS)),
    }

    if accepted:
        if outcome is None or join is None or observed is None:
            # Unreachable through the gate, which blocks on
            # BLOCKED_NO_ENGINE_ACCESS before it can accept. A raise rather than
            # an assert because -O would strip the assert.
            raise RuntimeError("gate accepted a run with no measured replay")

        ranking_rows = [
            {
                "metric": metric,
                "frozen_statistic": FROZEN_RANKING_STATISTIC,
                "arrival_scale": plan.arrival_scales[0],
                "left_label": "" if item is None else item.left_label,
                "right_label": "" if item is None else item.right_label,
                "tau_b": "" if item is None else item.tau_b,
                "pairwise_agreement": "" if item is None else item.pairwise_agreement,
                "concordant_pairs": "" if item is None else item.concordant_pairs,
                "discordant_pairs": "" if item is None else item.discordant_pairs,
                "available": item is not None,
                "score_unit_note": SCORE_UNIT_NOTE,
            }
            for metric, item in comparisons.items()
        ]
        result_rows = [
            {
                "arm_id": cell.arm_id,
                "arm_role": cell.arm_role.value,
                "arrival_scale": cell.arrival_scale,
                "case_fingerprint": cell.case_fingerprint,
                **cell.result.to_dict(),
                **labels,
            }
            for cell in outcome.cells
        ]
        measured_rows = [
            {
                "arm_id": cell.arm_id,
                "arrival_scale": cell.arrival_scale,
                "source": "MEASURED_ENGINE",
                **cell.scores,
            }
            for cell in observed.cells
        ] + [
            {
                "arm_id": cell.arm_id,
                "arrival_scale": cell.arrival_scale,
                "source": "MODELED_REPLAY",
                **{
                    metric: outcome.scores(metric, cell.arrival_scale)[cell.arm_id]
                    for metric in SCORE_METRICS
                },
            }
            for cell in observed.cells
        ]
        shadow_rows = [
            {
                "metric": metric,
                **{k: v for k, v in decision.to_dict().items() if k != "policy"},
                "reasons": "|".join(decision.reasons),
                **{f"policy_{k}": v for k, v in decision.policy.to_dict().items()},
            }
            for metric, decisions in shadow.items()
            for decision in decisions
        ]

        artifacts["replay.json"] = _json_bytes(
            {
                "modeled_outcome": outcome.to_dict(),
                "ranking": {
                    "frozen_statistic": FROZEN_RANKING_STATISTIC,
                    "arrival_scale": plan.arrival_scales[0],
                    "worst_metric": worst_metric,
                    "worst_tau_b": tau_b,
                    "score_unit_note": SCORE_UNIT_NOTE,
                    "comparisons": {
                        metric: None if item is None else item.to_dict()
                        for metric, item in comparisons.items()
                    },
                },
                "shadow": {
                    "enforced": False,
                    "decisions": {
                        metric: [decision.to_dict() for decision in decisions]
                        for metric, decisions in shadow.items()
                    },
                },
                "measured": {
                    "endpoint_id": observed.endpoint_id,
                    "plan_digest": observed.plan_digest,
                    "calibration_manifest_sha256": (
                        observed.calibration_manifest_sha256
                    ),
                    "producer_run_id": observed.producer_run_id,
                    "cell_count": len(observed.cells),
                    "reconciled_fraction": join.reconciled_fraction,
                    "disagreement_fraction": join.disagreement_fraction,
                },
                "fault_injection": fault_summary,
                "provenance": gate_payload["provenance"],
            }
        )
        artifacts["results.csv"] = _csv_bytes(result_rows, list(result_rows[0]))
        artifacts["scores.csv"] = _csv_bytes(measured_rows, list(measured_rows[0]))
        artifacts["ranking.csv"] = _csv_bytes(ranking_rows, list(ranking_rows[0]))
        artifacts["shadow.csv"] = _csv_bytes(shadow_rows, list(shadow_rows[0]))
        artifacts["reconciliation.csv"] = _csv_bytes(join.rows, list(join.rows[0]))
        # An empty measured ledger is the healthy outcome, so the field names are
        # declared rather than read off a first row that may not exist.
        artifacts["ledger.csv"] = _csv_bytes(
            join.ledger_rows, ["arm_id", "arrival_scale", *LEDGER_FIELDS]
        )

    artifacts[MANIFEST_NAME] = _manifest_bytes(artifacts)

    for name, blob in sorted(artifacts.items()):
        _assert_no_stronger_claim(
            name, blob, allow_hardware_labels=accepted and name == GATE_NAME
        )

    output_root = Path(args.output_root).resolve() if args.output_root else root
    output_dir = output_root / (ACCEPTED_DIR if accepted else BLOCKED_DIR)
    _write_artifacts(output_dir, artifacts)

    for name in sorted(artifacts):
        print(f"wrote {output_dir / name} ({len(artifacts[name])} bytes)")
    if accepted:
        print(f"M10-HW accepted: {report.evidence_tier.value}")
        return 0
    print("M10-HW rejected: " + ", ".join(item.value for item in report.blockers))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
