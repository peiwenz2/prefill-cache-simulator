from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import platform
import statistics
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from prefill_cache_sim.calibration import (
    CALIBRATION_SCHEMA_VERSION,
    DEPENDENCY_PATHS,
    HONEST_LABEL_TUPLES,
    MINIMUM_OBSERVATIONS,
    REPRODUCIBILITY_CLAIM,
    CalibrationParams,
    CalibrationStatus,
    DishonestLabelError,
    EngineEndpoint,
    EvidenceTier,
    LinearFit,
    MachineProvenance,
    MockCurve,
    MockEngine,
    ResidualSummary,
    SweepKind,
    SweepObservation,
    SweepSpec,
    TimeUnit,
    combined_digest,
    fit_sweep,
    honest_labels,
    m9_source_paths,
    run_sweep,
    runtime_identity,
    source_fingerprints,
    source_manifest,
)
from prefill_cache_sim.calibration import fingerprint as fingerprint_module

REPO_ROOT = Path(__file__).resolve().parents[1]


def prefill_spec(*, repeats: int = 1) -> SweepSpec:
    return SweepSpec(
        SweepKind.PREFILL,
        token_points=(512, 2048, 8192),
        batch_points=(1, 4, 8),
        repeats=repeats,
    )


def decode_spec(*, repeats: int = 1) -> SweepSpec:
    return SweepSpec(
        SweepKind.DECODE,
        token_points=(64, 256, 1024),
        batch_points=(1, 8, 32),
        repeats=repeats,
    )


def noiseless_engine() -> MockEngine:
    return MockEngine(
        endpoint_id="mock-noiseless",
        prefill_curve=MockCurve(3.0, 0.02, 1.5),
        tpot_curve=MockCurve(8.0, 0.0004, 0.9),
        noise_amplitude=0.0,
    )


def complete_machine() -> MachineProvenance:
    return MachineProvenance(
        host_id="bench-host-0",
        accelerator_model="L20",
        engine_version="vllm-0.0.0-test",
        captured_at_utc="2026-08-05T00:00:00Z",
    )


def sample_params(**overrides: object) -> CalibrationParams:
    engine = noiseless_engine()
    prefill = fit_sweep(run_sweep(engine, prefill_spec()))
    tpot = fit_sweep(run_sweep(engine, decode_spec()))
    machine = MachineProvenance.unknown()
    status, unit, tier = honest_labels(machine)
    base = {
        "endpoint_id": engine.endpoint_id,
        "machine": machine,
        "calibration_status": status,
        "time_unit": unit,
        "evidence_tier": tier,
        "prefill": prefill,
        "tpot": tpot,
        "code_sha": None,
        "code_dirty": None,
    }
    base.update(overrides)
    return CalibrationParams(**base)  # type: ignore[arg-type]


def test_mock_engine_reproduces_configured_curve_without_noise() -> None:
    engine = noiseless_engine()
    observation = engine.measure(
        kind=SweepKind.PREFILL, tokens=2048, batch_size=4, repeat_index=0
    )
    assert observation.kind is SweepKind.PREFILL
    assert (observation.tokens, observation.batch_size) == (2048, 4)
    assert observation.value == pytest.approx(3.0 + 0.02 * 2048 + 1.5 * 4)


def test_mock_engine_noise_is_deterministic_for_a_given_seed() -> None:
    noisy = MockEngine(noise_amplitude=5.0, seed=7)
    first = run_sweep(noisy, prefill_spec(repeats=2))
    second = run_sweep(MockEngine(noise_amplitude=5.0, seed=7), prefill_spec(repeats=2))
    assert [item.value for item in first] == [item.value for item in second]

    other = run_sweep(MockEngine(noise_amplitude=5.0, seed=8), prefill_spec(repeats=2))
    assert [item.value for item in first] != [item.value for item in other]


def test_mock_engine_reports_incomplete_machine_provenance() -> None:
    machine = noiseless_engine().machine_provenance()
    assert machine.complete is False
    assert machine.host_id is None
    assert machine.accelerator_model is None


def test_run_sweep_emits_one_observation_per_grid_point_and_repeat() -> None:
    spec = prefill_spec(repeats=3)
    observations = run_sweep(noiseless_engine(), spec)
    assert len(observations) == 3 * 3 * 3 == spec.observation_count()
    keys = [(item.tokens, item.batch_size, item.repeat_index) for item in observations]
    assert len(set(keys)) == len(keys)
    assert keys == sorted(keys)


def test_observation_rejects_non_finite_measurements() -> None:
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            SweepObservation(SweepKind.PREFILL, 512, 1, 0, bad)


class _RelabellingEngine:
    """Endpoint that answers every request with the same fabricated point."""

    endpoint_id = "mock-relabelling"

    def machine_provenance(self) -> MachineProvenance:
        return MachineProvenance.unknown()

    def measure(
        self,
        *,
        kind: SweepKind,
        tokens: int,
        batch_size: int,
        repeat_index: int,
    ) -> SweepObservation:
        return SweepObservation(kind, 999, 7, 0, 1.0)


def test_run_sweep_rejects_observations_that_do_not_echo_the_request() -> None:
    with pytest.raises(ValueError, match="different grid point"):
        run_sweep(_RelabellingEngine(), prefill_spec())


def test_prefill_fit_recovers_known_linear_curve_from_noiseless_sweep() -> None:
    fit = fit_sweep(run_sweep(noiseless_engine(), prefill_spec()))
    assert fit.kind is SweepKind.PREFILL
    assert fit.intercept == pytest.approx(3.0, abs=1e-9)
    assert fit.token_coefficient == pytest.approx(0.02, abs=1e-12)
    assert fit.batch_coefficient == pytest.approx(1.5, abs=1e-9)
    assert fit.observation_count == 9


def test_tpot_fit_recovers_known_linear_curve_from_noiseless_sweep() -> None:
    fit = fit_sweep(run_sweep(noiseless_engine(), decode_spec()))
    assert fit.kind is SweepKind.DECODE
    assert fit.intercept == pytest.approx(8.0, abs=1e-9)
    assert fit.token_coefficient == pytest.approx(0.0004, abs=1e-12)
    assert fit.batch_coefficient == pytest.approx(0.9, abs=1e-9)


def test_predict_matches_fitted_coefficients() -> None:
    fit = fit_sweep(run_sweep(noiseless_engine(), prefill_spec()))
    assert fit.predict(tokens=4096, batch_size=2) == pytest.approx(
        3.0 + 0.02 * 4096 + 1.5 * 2, abs=1e-8
    )


def test_noiseless_fit_leaves_negligible_residuals() -> None:
    fit = fit_sweep(run_sweep(noiseless_engine(), prefill_spec(repeats=2)))
    assert fit.residuals.count == 18
    assert fit.residuals.maximum_absolute < 1e-8


def test_residual_quantiles_use_nearest_rank_without_normal_assumption() -> None:
    summary = ResidualSummary.from_residuals([0.0] * 9 + [100.0])
    assert summary.distribution_assumption == "NONE_EMPIRICAL"
    assert summary.p50 == 0.0
    assert summary.p90 == 0.0
    assert summary.p95 == 100.0
    assert summary.p99 == 100.0


def test_residual_summary_keeps_raw_residuals_for_downstream_audit() -> None:
    residuals = [-2.0, 0.5, 3.25]
    summary = ResidualSummary.from_residuals(residuals)
    assert summary.raw == (-2.0, 0.5, 3.25)
    assert summary.minimum == -2.0
    assert summary.maximum == 3.25
    assert summary.maximum_absolute == 3.25
    assert summary.mean_absolute == pytest.approx((2.0 + 0.5 + 3.25) / 3)


def test_residual_summary_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one residual"):
        ResidualSummary.from_residuals([])


def test_sweep_spec_rejects_degenerate_grids() -> None:
    with pytest.raises(ValueError, match="at least two distinct token points"):
        SweepSpec(SweepKind.PREFILL, token_points=(512,), batch_points=(1, 4))
    with pytest.raises(ValueError, match="at least two distinct batch points"):
        SweepSpec(SweepKind.PREFILL, token_points=(512, 1024), batch_points=(4,))
    with pytest.raises(ValueError, match="strictly increasing"):
        SweepSpec(SweepKind.PREFILL, token_points=(1024, 512), batch_points=(1, 4))
    with pytest.raises(ValueError, match="repeats must be positive"):
        SweepSpec(
            SweepKind.PREFILL, token_points=(512, 1024), batch_points=(1, 4), repeats=0
        )
    with pytest.raises(ValueError, match="must be positive"):
        SweepSpec(SweepKind.PREFILL, token_points=(0, 1024), batch_points=(1, 4))


def test_fit_rejects_rank_deficient_observations() -> None:
    engine = noiseless_engine()
    collinear = tuple(
        engine.measure(
            kind=SweepKind.PREFILL, tokens=512 * step, batch_size=step, repeat_index=0
        )
        for step in (1, 2, 3, 4)
    )
    with pytest.raises(ValueError, match="rank deficient"):
        fit_sweep(collinear)


def test_fit_rejects_mixed_sweep_kinds() -> None:
    engine = noiseless_engine()
    mixed = (
        *run_sweep(engine, prefill_spec()),
        engine.measure(kind=SweepKind.DECODE, tokens=64, batch_size=1, repeat_index=0),
    )
    with pytest.raises(ValueError, match="single sweep kind"):
        fit_sweep(mixed)


def test_fit_requires_minimum_observation_count() -> None:
    engine = noiseless_engine()
    too_few = tuple(
        engine.measure(
            kind=SweepKind.PREFILL, tokens=tokens, batch_size=1, repeat_index=0
        )
        for tokens in (512, 1024)
    )
    with pytest.raises(ValueError, match="at least three observations"):
        fit_sweep(too_few)


def test_honest_labels_default_to_synthetic_when_provenance_missing() -> None:
    status, unit, tier = honest_labels(MachineProvenance.unknown())
    assert status is CalibrationStatus.SYNTHETIC_UNCALIBRATED
    assert unit is TimeUnit.NORMALIZED_WORK
    assert tier is EvidenceTier.HARNESS_ONLY


def test_honest_labels_allow_calibrated_output_only_with_complete_provenance() -> None:
    status, unit, tier = honest_labels(complete_machine())
    assert status is CalibrationStatus.HW_CALIBRATED
    assert unit is TimeUnit.MILLISECONDS
    assert tier is EvidenceTier.HW_VALIDATED


def test_blank_provenance_fields_are_rejected_at_construction() -> None:
    for name in ("host_id", "accelerator_model", "engine_version", "captured_at_utc"):
        with pytest.raises(DishonestLabelError, match=name):
            replace(complete_machine(), **{name: "   "})


def test_blank_provenance_cannot_license_calibrated_labels() -> None:
    with pytest.raises(DishonestLabelError, match="non-blank"):
        MachineProvenance("", "", "", "")


def test_partial_provenance_is_treated_as_missing() -> None:
    partial = replace(complete_machine(), accelerator_model=None)
    assert partial.complete is False
    status, unit, tier = honest_labels(partial)
    assert status is CalibrationStatus.SYNTHETIC_UNCALIBRATED
    assert unit is TimeUnit.NORMALIZED_WORK
    assert tier is EvidenceTier.HARNESS_ONLY


def test_incomplete_provenance_blocks_hw_calibrated_status() -> None:
    with pytest.raises(DishonestLabelError, match="calibration_status"):
        sample_params(calibration_status=CalibrationStatus.HW_CALIBRATED)


def test_incomplete_provenance_blocks_millisecond_time_unit() -> None:
    with pytest.raises(DishonestLabelError, match="time_unit"):
        sample_params(time_unit=TimeUnit.MILLISECONDS)


def test_incomplete_provenance_blocks_hw_validated_tier() -> None:
    with pytest.raises(DishonestLabelError, match="evidence_tier"):
        sample_params(evidence_tier=EvidenceTier.HW_VALIDATED)


def test_uncalibrated_status_cannot_pair_with_millisecond_unit() -> None:
    with pytest.raises(DishonestLabelError, match="time_unit"):
        sample_params(
            machine=complete_machine(),
            calibration_status=CalibrationStatus.SYNTHETIC_UNCALIBRATED,
            time_unit=TimeUnit.MILLISECONDS,
        )


def test_complete_provenance_allows_calibrated_labels() -> None:
    params = sample_params(
        machine=complete_machine(),
        calibration_status=CalibrationStatus.HW_CALIBRATED,
        time_unit=TimeUnit.MILLISECONDS,
        evidence_tier=EvidenceTier.HW_VALIDATED,
    )
    assert params.machine.complete is True
    assert params.time_unit is TimeUnit.MILLISECONDS


def test_params_round_trip_preserves_schema_version_and_fits() -> None:
    params = sample_params()
    restored = CalibrationParams.from_dict(json.loads(json.dumps(params.to_dict())))
    assert restored.schema_version == CALIBRATION_SCHEMA_VERSION
    assert restored == params
    assert restored.prefill.predict(tokens=1024, batch_size=2) == pytest.approx(
        params.prefill.predict(tokens=1024, batch_size=2)
    )


def test_from_dict_rejects_unknown_schema_version() -> None:
    payload = sample_params().to_dict()
    payload["schema_version"] = "calibration-params-v0"
    with pytest.raises(ValueError, match="unsupported calibration schema version"):
        CalibrationParams.from_dict(payload)


def test_serialized_payload_carries_honesty_labels_and_raw_residuals() -> None:
    payload = sample_params().to_dict()
    assert payload["calibration_status"] == "SYNTHETIC_UNCALIBRATED"
    assert payload["time_unit"] == "NORMALIZED_WORK"
    assert payload["evidence_tier"] == "HARNESS_ONLY"
    assert payload["machine"] == {
        "host_id": None,
        "accelerator_model": None,
        "engine_version": None,
        "captured_at_utc": None,
    }
    residuals = payload["prefill"]["residuals"]
    assert residuals["distribution_assumption"] == "NONE_EMPIRICAL"
    assert isinstance(residuals["raw"], list)


def test_linear_fit_round_trip_is_exact() -> None:
    fit = fit_sweep(run_sweep(noiseless_engine(), decode_spec()))
    assert LinearFit.from_dict(json.loads(json.dumps(fit.to_dict()))) == fit


# --- F1: honest label tuples and provenance validation ----------------------


#: Tuples that mix a strong claim on one field with a weak one on another. Each
#: was accepted before the policy was enumerated, because complete provenance
#: satisfied every per-field rule while the tuple as a whole stayed incoherent.
CONTRADICTORY_TUPLES = [
    (CalibrationStatus.HW_CALIBRATED, TimeUnit.MILLISECONDS, EvidenceTier.HARNESS_ONLY),
    (
        CalibrationStatus.HW_CALIBRATED,
        TimeUnit.MILLISECONDS,
        EvidenceTier.SYNTHETIC_REPLAY,
    ),
    (
        CalibrationStatus.HW_CALIBRATED,
        TimeUnit.NORMALIZED_WORK,
        EvidenceTier.HW_VALIDATED,
    ),
    (
        CalibrationStatus.SYNTHETIC_UNCALIBRATED,
        TimeUnit.MILLISECONDS,
        EvidenceTier.HW_VALIDATED,
    ),
    (
        CalibrationStatus.SYNTHETIC_UNCALIBRATED,
        TimeUnit.NORMALIZED_WORK,
        EvidenceTier.HW_VALIDATED,
    ),
    (
        CalibrationStatus.SYNTHETIC_UNCALIBRATED,
        TimeUnit.MILLISECONDS,
        EvidenceTier.SYNTHETIC_REPLAY,
    ),
]


@pytest.mark.parametrize(("status", "unit", "tier"), CONTRADICTORY_TUPLES)
def test_contradictory_labels_are_rejected_even_on_identified_hardware(
    status: CalibrationStatus, unit: TimeUnit, tier: EvidenceTier
) -> None:
    with pytest.raises(DishonestLabelError, match="not an honest label tuple"):
        sample_params(
            machine=complete_machine(),
            calibration_status=status,
            time_unit=unit,
            evidence_tier=tier,
        )


@pytest.mark.parametrize(("status", "unit", "tier"), sorted(HONEST_LABEL_TUPLES))
def test_every_honest_tuple_is_accepted_with_supporting_provenance(
    status: CalibrationStatus, unit: TimeUnit, tier: EvidenceTier
) -> None:
    params = sample_params(
        machine=complete_machine(),
        calibration_status=status,
        time_unit=unit,
        evidence_tier=tier,
    )
    assert (params.calibration_status, params.time_unit, params.evidence_tier) == (
        status,
        unit,
        tier,
    )


def test_underclaiming_the_whole_tuple_on_identified_hardware_is_allowed() -> None:
    """The documented policy: underclaim along the tuple, never one field of it."""
    params = sample_params(
        machine=complete_machine(),
        calibration_status=CalibrationStatus.SYNTHETIC_UNCALIBRATED,
        time_unit=TimeUnit.NORMALIZED_WORK,
        evidence_tier=EvidenceTier.HARNESS_ONLY,
    )
    assert params.machine.complete is True
    assert params.evidence_tier is EvidenceTier.HARNESS_ONLY


def test_synthetic_replay_tier_is_honest_without_any_provenance() -> None:
    params = sample_params(evidence_tier=EvidenceTier.SYNTHETIC_REPLAY)
    assert params.machine.complete is False
    assert params.evidence_tier is EvidenceTier.SYNTHETIC_REPLAY


@pytest.mark.parametrize(
    "stamp",
    [
        "2026-08-05",
        "2026-08-05 00:00:00Z",
        "2026-08-05T00:00:00",
        "20260805T000000Z",
        "yesterday",
        "",
    ],
)
def test_captured_at_utc_must_be_an_rfc3339_timestamp(stamp: str) -> None:
    with pytest.raises(DishonestLabelError):
        replace(complete_machine(), captured_at_utc=stamp)


def test_captured_at_utc_rejects_a_local_offset() -> None:
    with pytest.raises(DishonestLabelError, match="UTC offset zero"):
        replace(complete_machine(), captured_at_utc="2026-08-05T08:00:00+08:00")


def test_captured_at_utc_rejects_a_syntactically_valid_non_instant() -> None:
    with pytest.raises(DishonestLabelError, match="not a real instant"):
        replace(complete_machine(), captured_at_utc="2026-13-45T00:00:00Z")


@pytest.mark.parametrize(
    "stamp",
    [
        "2026-08-05T00:00:00Z",
        "2026-08-05t00:00:00z",
        "2026-08-05T00:00:00.123456Z",
        "2026-08-05T00:00:00+00:00",
    ],
)
def test_captured_at_utc_accepts_every_rfc3339_utc_spelling(stamp: str) -> None:
    machine = replace(complete_machine(), captured_at_utc=stamp)
    assert machine.complete is True


@pytest.mark.parametrize(
    "name", ["host_id", "accelerator_model", "engine_version", "captured_at_utc"]
)
def test_provenance_fields_reject_non_string_values(name: str) -> None:
    # A deserialized payload can carry any JSON type, so the field is deliberately
    # given a value its annotation forbids; ``dict[str, Any]`` is what lets a
    # type checker accept the very call the runtime guard exists to reject.
    override: dict[str, Any] = {name: 1234}
    with pytest.raises(DishonestLabelError, match="must be None or a string"):
        replace(complete_machine(), **override)


def test_provenance_from_dict_rejects_a_json_payload_of_the_wrong_type() -> None:
    payload = complete_machine().to_dict()
    payload["accelerator_model"] = ["L20"]
    with pytest.raises(DishonestLabelError, match="must be None or a string"):
        MachineProvenance.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("calibration_status", "SYNTHETIC_UNCALIBRATED"),
        ("time_unit", "NORMALIZED_WORK"),
        ("evidence_tier", "HARNESS_ONLY"),
    ],
)
def test_raw_label_strings_are_rejected_in_favour_of_enum_members(
    field: str, value: str
) -> None:
    # StrEnum members compare equal to these strings but hash by name, so a raw
    # string would miss the honest-tuple set and fail with a confusing message.
    with pytest.raises(ValueError, match=field):
        sample_params(**{field: value})


def test_params_reject_a_machine_of_the_wrong_type() -> None:
    with pytest.raises(ValueError, match="machine must be a MachineProvenance"):
        sample_params(machine={"host_id": None})


def test_params_reject_a_blank_endpoint_id() -> None:
    with pytest.raises(ValueError, match="endpoint_id must be a non-blank string"):
        sample_params(endpoint_id="   ")


# --- F2: fail-closed fit and parameter invariants ---------------------------


def test_params_reject_a_decode_fit_in_the_prefill_slot() -> None:
    engine = noiseless_engine()
    decode_fit = fit_sweep(run_sweep(engine, decode_spec()))
    with pytest.raises(ValueError, match="prefill must hold a PREFILL fit"):
        sample_params(prefill=decode_fit)


def test_params_reject_a_prefill_fit_in_the_tpot_slot() -> None:
    engine = noiseless_engine()
    prefill_fit = fit_sweep(run_sweep(engine, prefill_spec()))
    with pytest.raises(ValueError, match="tpot must hold a DECODE fit"):
        sample_params(tpot=prefill_fit)


def test_from_dict_rejects_swapped_fit_roles_after_a_json_round_trip() -> None:
    payload = json.loads(json.dumps(sample_params().to_dict()))
    payload["prefill"], payload["tpot"] = payload["tpot"], payload["prefill"]
    with pytest.raises(ValueError, match="must hold a"):
        CalibrationParams.from_dict(payload)


@pytest.mark.parametrize(
    "name", ["intercept", "token_coefficient", "batch_coefficient"]
)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_fit_rejects_non_finite_coefficients(name: str, bad: float) -> None:
    fit = fit_sweep(run_sweep(noiseless_engine(), prefill_spec()))
    override: dict[str, Any] = {name: bad}
    with pytest.raises(ValueError, match=f"fit {name} must be finite"):
        replace(fit, **override)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_from_dict_rejects_non_finite_coefficients_parsed_from_json(
    literal: str,
) -> None:
    # json.loads accepts these non-standard literals, so the guard cannot live
    # in the parser; it has to be a construction-time invariant.
    payload = json.loads(json.dumps(sample_params().to_dict()))
    payload["prefill"]["token_coefficient"] = json.loads(literal)
    assert not math.isfinite(payload["prefill"]["token_coefficient"])
    with pytest.raises(ValueError, match="must be finite"):
        CalibrationParams.from_dict(payload)


def test_residual_summary_rejects_non_finite_residuals() -> None:
    with pytest.raises(ValueError, match="residual must be finite"):
        ResidualSummary.from_residuals([0.0, float("nan"), 1.0])


def test_residual_summary_rejects_a_count_that_contradicts_the_raw_vector() -> None:
    summary = ResidualSummary.from_residuals([-1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="does not match 3 raw residuals"):
        replace(summary, count=9)


def test_residual_summary_rejects_a_relabelled_distribution_assumption() -> None:
    summary = ResidualSummary.from_residuals([-1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="distribution_assumption must be"):
        replace(summary, distribution_assumption="GAUSSIAN")


def test_residual_summary_rejects_order_statistics_that_do_not_match_raw() -> None:
    summary = ResidualSummary.from_residuals([-1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="residual minimum does not match"):
        replace(summary, minimum=-99.0)
    with pytest.raises(ValueError, match="residual maximum_absolute does not match"):
        replace(summary, maximum_absolute=99.0)


def test_residual_summary_rejects_decreasing_quantiles() -> None:
    summary = ResidualSummary.from_residuals([-1.0, 0.0, 1.0])
    # Swapped quantiles are rejected because neither value is the nearest-rank
    # statistic of `raw`, not merely because the pair is out of order.
    with pytest.raises(ValueError, match="residual p50 does not match"):
        replace(summary, p50=1.0, p90=0.0)


def test_residual_summary_rejects_a_mean_absolute_above_the_maximum() -> None:
    summary = ResidualSummary.from_residuals([-1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="residual mean_absolute does not match"):
        replace(summary, mean_absolute=5.0)


# --- F2a: every summary field is recomputed from the raw vector -------------


def _expected_summary_fields(raw: list[float]) -> dict[str, float]:
    """Recompute the summary independently of the module under test."""
    ordered = sorted(raw)
    magnitudes = [abs(value) for value in raw]

    def nearest_rank(quantile: float) -> float:
        return ordered[max(1, math.ceil(quantile * len(ordered))) - 1]

    return {
        "p50": nearest_rank(0.50),
        "p90": nearest_rank(0.90),
        "p95": nearest_rank(0.95),
        "p99": nearest_rank(0.99),
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "maximum_absolute": max(magnitudes),
        "mean_absolute": statistics.fmean(magnitudes),
    }


@pytest.mark.parametrize(
    "raw",
    [
        [0.0],
        [-1.0, 0.0, 1.0],
        [2.0, -5.0, 0.25, 0.25, -0.5],
        # Heavy right tail: p90 and p95/p99 land on different order statistics,
        # which a Gaussian summary of the same vector would smooth away.
        [0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 40.0],
    ],
)
def test_residual_summary_matches_independently_recomputed_statistics(
    raw: list[float],
) -> None:
    summary = ResidualSummary.from_residuals(raw)
    for name, wanted in _expected_summary_fields(raw).items():
        assert getattr(summary, name) == wanted, name


@pytest.mark.parametrize(
    "name",
    ["p50", "p90", "p95", "p99", "minimum", "maximum", "maximum_absolute"],
)
def test_hostile_payload_cannot_fabricate_an_order_statistic(name: str) -> None:
    """A payload may not relabel the spread of the vector it ships.

    Each fabrication below sits inside ``[minimum, maximum]`` and leaves the
    quantiles non-decreasing, so bounds-only validation accepted them: a fit
    could publish a flattering p99 while carrying the raw residuals that
    disprove it. Only recomputation from ``raw`` rejects them.
    """
    raw = [-4.0, -1.0, 0.0, 0.5, 1.0, 9.0]
    payload = ResidualSummary.from_residuals(raw).to_dict()
    assert payload[name] != 0.5
    payload[name] = 0.5
    with pytest.raises(ValueError, match=f"residual {name} does not match"):
        ResidualSummary.from_dict(payload)


def test_hostile_payload_cannot_understate_the_mean_absolute_residual() -> None:
    raw = [-4.0, -1.0, 0.0, 0.5, 1.0, 9.0]
    payload = ResidualSummary.from_residuals(raw).to_dict()
    # Inside [0, maximum_absolute], so only recomputation catches it.
    payload["mean_absolute"] = 0.01
    with pytest.raises(ValueError, match="residual mean_absolute does not match"):
        ResidualSummary.from_dict(payload)


def test_mean_absolute_tolerance_is_tight_enough_to_catch_a_rounded_claim() -> None:
    """The `fmean` tolerance absorbs a last-bit difference and nothing wider."""
    raw = [-4.0, -1.0, 0.0, 0.5, 1.0, 9.0]
    summary = ResidualSummary.from_residuals(raw)
    payload = summary.to_dict()

    payload["mean_absolute"] = math.nextafter(summary.mean_absolute, math.inf)
    assert ResidualSummary.from_dict(payload).mean_absolute != summary.mean_absolute

    # One part in 10^9 is still numerically indistinguishable to a reader and
    # far beyond a rounding difference, so it must be rejected.
    payload["mean_absolute"] = summary.mean_absolute * (1 + 1e-9)
    with pytest.raises(ValueError, match="residual mean_absolute does not match"):
        ResidualSummary.from_dict(payload)


def test_hostile_payload_cannot_swap_the_raw_vector_under_a_valid_summary() -> None:
    """Keeping the summary and replacing `raw` is the same lie from the other side."""
    payload = ResidualSummary.from_residuals([-4.0, -1.0, 0.0, 0.5, 1.0, 9.0]).to_dict()
    payload["raw"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="does not match the value recomputed"):
        ResidualSummary.from_dict(payload)


def test_reordering_raw_preserves_the_summary_it_validates_against() -> None:
    """A permutation of `raw` is the same multiset, so it must still validate."""
    raw = [-4.0, -1.0, 0.0, 0.5, 1.0, 9.0]
    payload = ResidualSummary.from_residuals(raw).to_dict()
    payload["raw"] = list(reversed(raw))
    restored = ResidualSummary.from_dict(payload)
    assert restored.p99 == 9.0
    assert restored.raw == tuple(reversed(raw))


def test_residual_summary_of_a_real_fit_survives_a_json_round_trip() -> None:
    """The generator's own residuals must pass the stricter check unchanged."""
    summary = fit_sweep(run_sweep(noiseless_engine(), prefill_spec())).residuals
    assert ResidualSummary.from_dict(json.loads(json.dumps(summary.to_dict()))) == (
        summary
    )


def test_fit_rejects_an_observation_count_that_contradicts_the_residuals() -> None:
    fit = fit_sweep(run_sweep(noiseless_engine(), prefill_spec()))
    with pytest.raises(ValueError, match="does not match 9 residuals"):
        replace(fit, observation_count=42)


def test_fit_rejects_an_observation_count_below_the_parameter_floor() -> None:
    summary = ResidualSummary.from_residuals([0.0, 0.0])
    with pytest.raises(
        ValueError, match=f"observation_count must be at least {MINIMUM_OBSERVATIONS}"
    ):
        LinearFit(SweepKind.PREFILL, 1.0, 1.0, 1.0, 2, summary)


def test_fit_rejects_a_kind_that_is_not_a_sweep_kind() -> None:
    summary = ResidualSummary.from_residuals([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="fit kind must be a SweepKind"):
        LinearFit("PREFILL", 1.0, 1.0, 1.0, 3, summary)  # type: ignore[arg-type]


def test_params_serialize_under_strict_json_without_nan_literals() -> None:
    encoded = json.dumps(sample_params().to_dict(), allow_nan=False)
    assert "NaN" not in encoded
    assert "Infinity" not in encoded


# --- F3: the protocol is satisfiable by an immutable endpoint ---------------


def test_frozen_mock_engine_satisfies_the_endpoint_protocol() -> None:
    # The protocol declares endpoint_id read-only; a settable protocol variable
    # would additionally demand a setter that a frozen dataclass cannot provide.
    engine = noiseless_engine()
    assert isinstance(engine, EngineEndpoint)
    assert engine.endpoint_id == "mock-noiseless"


# --- F4: source fingerprints -----------------------------------------------


def _plant_m9_tree(root: Path) -> Path:
    """Create the minimum file set :func:`source_fingerprints` insists on."""
    package = root / fingerprint_module.PACKAGE_DIR
    package.mkdir(parents=True)
    (package / "endpoint.py").write_text("original\n", encoding="utf-8")
    generator = root / fingerprint_module.GENERATOR_PATH
    generator.parent.mkdir(parents=True)
    generator.write_text("generator\n", encoding="utf-8")
    for relative in DEPENDENCY_PATHS:
        dependency = root / relative
        dependency.parent.mkdir(parents=True, exist_ok=True)
        dependency.write_text(f"# {relative}\n", encoding="utf-8")
    return package


def test_source_paths_cover_the_generator_and_every_calibration_module() -> None:
    paths = m9_source_paths(REPO_ROOT)
    assert "scripts/run_m9_synthetic.py" in paths
    discovered = {
        f"src/prefill_cache_sim/calibration/{path.name}"
        for path in (REPO_ROOT / "src/prefill_cache_sim/calibration").glob("*.py")
    }
    # Discovery rather than a hand-maintained list: a module added tomorrow must
    # not silently escape the manifest.
    assert discovered <= set(paths)
    assert set(paths) - discovered == {"scripts/run_m9_synthetic.py", *DEPENDENCY_PATHS}


def test_source_paths_cover_the_out_of_package_dependencies() -> None:
    paths = m9_source_paths(REPO_ROOT)
    # The generator's published numbers depend on these modules, so a digest set
    # that omitted them could stay constant across a run that changed the
    # artifact. They cannot be globbed without dragging in the whole simulator,
    # so they are listed, and the list is asserted here rather than assumed.
    assert set(DEPENDENCY_PATHS) <= set(paths)
    assert set(DEPENDENCY_PATHS) == {
        "src/prefill_cache_sim/config.py",
        "src/prefill_cache_sim/identity.py",
        "src/prefill_cache_sim/trace.py",
    }
    for relative in DEPENDENCY_PATHS:
        assert (REPO_ROOT / relative).is_file()


@pytest.mark.parametrize("relative", DEPENDENCY_PATHS)
def test_source_fingerprints_fail_loudly_when_a_dependency_is_absent(
    tmp_path: Path, relative: str
) -> None:
    _plant_m9_tree(tmp_path)
    (tmp_path / relative).unlink()
    with pytest.raises(FileNotFoundError, match=relative):
        source_fingerprints(tmp_path)


def test_source_fingerprints_are_the_sha256_of_the_files_on_disk() -> None:
    fingerprints = source_fingerprints(REPO_ROOT)
    assert set(fingerprints) == set(m9_source_paths(REPO_ROOT))
    for relative, digest in fingerprints.items():
        expected = hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
        assert digest == expected


def test_source_paths_fail_loudly_when_the_package_is_absent(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="calibration package not found"):
        m9_source_paths(tmp_path)


def test_combined_digest_is_order_independent_but_content_sensitive() -> None:
    base = {"a.py": "0" * 64, "b.py": "1" * 64}
    reordered = {"b.py": "1" * 64, "a.py": "0" * 64}
    edited = {"a.py": "0" * 64, "b.py": "2" * 64}
    assert combined_digest(base) == combined_digest(reordered)
    assert combined_digest(base) != combined_digest(edited)


def test_combined_digest_changes_when_a_file_is_renamed_or_dropped() -> None:
    base = {"a.py": "0" * 64, "b.py": "1" * 64}
    renamed = {"a.py": "0" * 64, "c.py": "1" * 64}
    dropped = {"a.py": "0" * 64}
    # Paths are hashed alongside their digests, so the file set is covered too.
    assert combined_digest(base) != combined_digest(renamed)
    assert combined_digest(base) != combined_digest(dropped)


def test_source_manifest_publishes_its_own_limitation() -> None:
    manifest = source_manifest(REPO_ROOT)
    assert manifest["algorithm"] == "sha256"
    assert manifest["reproducibility_claim"] == REPRODUCIBILITY_CLAIM
    assert (
        REPRODUCIBILITY_CLAIM
        == "SOURCE_AND_RUNTIME_IDENTIFIED_REPRODUCTION_CONDITIONAL"
    )
    note = manifest["note"]
    # The artifact must not imply a commit checkout is enough to reproduce it,
    # nor that a matching source tree is: the claim is identification, and the
    # two things reproduction additionally depends on are named in the note.
    assert "git_sha" in note
    assert "runtime" in note
    assert "trace" in note
    assert manifest["files"] == source_fingerprints(REPO_ROOT)
    assert manifest["combined_digest"] == combined_digest(manifest["files"])


def test_source_manifest_records_the_interpreter_that_ran_the_generator() -> None:
    manifest = source_manifest(REPO_ROOT)
    runtime = manifest["runtime"]
    assert runtime == runtime_identity()
    assert set(runtime) == {"python_implementation", "python_version"}
    assert runtime["python_implementation"] == platform.python_implementation()
    assert runtime["python_version"] == platform.python_version()
    # The runtime is recorded beside the digests, not folded into them, so that
    # `combined_digest` stays a checkable function of the source file set alone.
    assert combined_digest(manifest["files"]) == manifest["combined_digest"]


def test_source_manifest_tracks_an_edit_to_any_covered_file(tmp_path: Path) -> None:
    package = _plant_m9_tree(tmp_path)

    before = source_manifest(tmp_path)
    (package / "endpoint.py").write_text("edited\n", encoding="utf-8")
    after = source_manifest(tmp_path)
    assert before["combined_digest"] != after["combined_digest"]

    (package / "extra.py").write_text("added\n", encoding="utf-8")
    with_extra = source_manifest(tmp_path)
    assert set(with_extra["files"]) - set(after["files"]) == {
        f"{fingerprint_module.PACKAGE_DIR}/extra.py"
    }


@pytest.mark.parametrize("relative", DEPENDENCY_PATHS)
def test_source_manifest_tracks_an_edit_to_a_dependency(
    tmp_path: Path, relative: str
) -> None:
    _plant_m9_tree(tmp_path)
    before = source_manifest(tmp_path)
    # An edit here can change the artifact while every calibration module stays
    # byte-identical, which is exactly the change the digests exist to catch.
    (tmp_path / relative).write_text("# edited\n", encoding="utf-8")
    after = source_manifest(tmp_path)
    assert after["files"][relative] != before["files"][relative]
    assert after["combined_digest"] != before["combined_digest"]
    assert set(after["files"]) == set(before["files"])


def test_combined_digest_changes_when_the_dependency_set_changes(
    tmp_path: Path,
) -> None:
    _plant_m9_tree(tmp_path)
    covered = source_manifest(tmp_path)["files"]
    dropped = {
        path: digest for path, digest in covered.items() if path not in DEPENDENCY_PATHS
    }
    assert set(covered) - set(dropped) == set(DEPENDENCY_PATHS)
    # Dropping the dependencies from the set must not be able to masquerade as
    # the same code: paths are hashed alongside their digests.
    assert combined_digest(dropped) != combined_digest(covered)


# --- F5: staged, manifest-indexed artifact writes ---------------------------


def _load_generator() -> ModuleType:
    path = REPO_ROOT / "scripts/run_m9_synthetic.py"
    spec = importlib.util.spec_from_file_location("m9_generator_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_generator()


def _sample_artifacts() -> dict[str, bytes]:
    artifacts = {
        "params.json": GENERATOR._json_bytes({"schema_version": "x"}),
        "results.csv": GENERATOR._csv_bytes([{"a": 1, "b": 2}]),
    }
    artifacts[GENERATOR.MANIFEST_NAME] = GENERATOR._manifest_bytes(artifacts)
    return artifacts


def test_json_bytes_refuses_to_emit_nan_or_infinity() -> None:
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="Out of range float"):
            GENERATOR._json_bytes({"coefficient": bad})


def test_json_and_csv_bytes_are_deterministic() -> None:
    payload: dict[str, Any] = {"b": 2, "a": [1, 2, 3]}
    assert GENERATOR._json_bytes(payload) == GENERATOR._json_bytes(dict(payload))
    rows = [{"x": 1, "y": "two"}, {"x": 3, "y": "four"}]
    assert GENERATOR._csv_bytes(rows) == GENERATOR._csv_bytes(list(rows))


def test_csv_bytes_match_what_a_file_handle_write_would_produce(
    tmp_path: Path,
) -> None:
    import csv

    rows = [{"x": 1, "y": "two"}, {"x": 3, "y": "four"}]
    reference = tmp_path / "reference.csv"
    with reference.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["x", "y"])
        writer.writeheader()
        writer.writerows(rows)
    assert GENERATOR._csv_bytes(rows) == reference.read_bytes()


def test_manifest_digests_every_other_artifact() -> None:
    artifacts = _sample_artifacts()
    manifest = json.loads(artifacts[GENERATOR.MANIFEST_NAME])
    assert manifest["schema_version"] == GENERATOR.MANIFEST_SCHEMA_VERSION
    assert set(manifest["files"]) == {"params.json", "results.csv"}
    for name, digest in manifest["files"].items():
        assert digest == hashlib.sha256(artifacts[name]).hexdigest()


def test_write_artifacts_requires_a_manifest_in_the_set() -> None:
    with pytest.raises(RuntimeError, match="must include MANIFEST.json"):
        GENERATOR._write_artifacts(Path("/nonexistent"), {"params.json": b"{}"})


def test_write_artifacts_replaces_generated_files_and_creates_the_directory(
    tmp_path: Path,
) -> None:
    output = tmp_path / "results" / "m9-synthetic"
    artifacts = _sample_artifacts()
    GENERATOR._write_artifacts(output, artifacts)
    for name, payload in artifacts.items():
        assert (output / name).read_bytes() == payload


def test_write_artifacts_never_deletes_files_it_did_not_generate(
    tmp_path: Path,
) -> None:
    output = tmp_path / "results" / "m9-synthetic"
    output.mkdir(parents=True)
    # PROVENANCE.md is hand-written and lives in the output directory, so a
    # whole-directory swap would destroy it.
    provenance = output / "PROVENANCE.md"
    provenance.write_text("hand written\n", encoding="utf-8")
    stale = output / "results.csv"
    stale.write_text("stale\n", encoding="utf-8")

    GENERATOR._write_artifacts(output, _sample_artifacts())

    assert provenance.read_text(encoding="utf-8") == "hand written\n"
    assert stale.read_bytes() == _sample_artifacts()["results.csv"]


def test_write_artifacts_leaves_no_staging_directory_behind(tmp_path: Path) -> None:
    output = tmp_path / "results" / "m9-synthetic"
    GENERATOR._write_artifacts(output, _sample_artifacts())
    assert list(output.parent.glob(".m9-staging-*")) == []
    assert sorted(item.name for item in output.iterdir()) == [
        "MANIFEST.json",
        "params.json",
        "results.csv",
    ]


def test_write_artifacts_is_byte_identical_across_runs(tmp_path: Path) -> None:
    first = tmp_path / "a" / "m9-synthetic"
    second = tmp_path / "b" / "m9-synthetic"
    GENERATOR._write_artifacts(first, _sample_artifacts())
    GENERATOR._write_artifacts(second, _sample_artifacts())
    for name in _sample_artifacts():
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_published_artifact_manifest_matches_the_committed_files() -> None:
    output = REPO_ROOT / "results/m9-synthetic"
    manifest_path = output / GENERATOR.MANIFEST_NAME
    if not manifest_path.is_file():
        pytest.skip("results/m9-synthetic has not been regenerated yet")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        actual = hashlib.sha256((output / name).read_bytes()).hexdigest()
        assert actual == digest, f"{name} does not match the manifest digest"


def test_published_params_record_source_fingerprints() -> None:
    params_path = REPO_ROOT / "results/m9-synthetic/params.json"
    if not params_path.is_file():
        pytest.skip("results/m9-synthetic has not been generated yet")
    payload = json.loads(params_path.read_text(encoding="utf-8"))
    manifest = payload["provenance"]["source_fingerprints"]
    assert manifest["reproducibility_claim"] == REPRODUCIBILITY_CLAIM
    assert manifest["combined_digest"] == combined_digest(manifest["files"])
    # Internal consistency is not enough: a stale artifact recomputes its own
    # digest happily. The recorded file map has to name the sources on disk now,
    # digest for digest, or the published numbers came from code nobody can read.
    live = source_fingerprints(REPO_ROOT)
    assert manifest["files"] == live, (
        "results/m9-synthetic/params.json fingerprints are stale; "
        "rerun scripts/run_m9_synthetic.py"
    )
    assert manifest["combined_digest"] == combined_digest(live)
    assert set(DEPENDENCY_PATHS) <= set(manifest["files"])


def test_published_params_record_the_runtime_that_produced_them() -> None:
    params_path = REPO_ROOT / "results/m9-synthetic/params.json"
    if not params_path.is_file():
        pytest.skip("results/m9-synthetic has not been generated yet")
    payload = json.loads(params_path.read_text(encoding="utf-8"))
    runtime = payload["provenance"]["source_fingerprints"]["runtime"]
    assert set(runtime) == {"python_implementation", "python_version"}
    # Not asserted equal to the running interpreter: the artifact records the
    # runtime it was produced under, and a reader on a different one must still
    # be able to see that mismatch rather than be told the file is corrupt.
    assert all(isinstance(value, str) and value for value in runtime.values())
