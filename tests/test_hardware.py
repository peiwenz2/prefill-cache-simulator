"""Tests for M9-HW and M10-HW: hardware gates, transport, endpoint, scripts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from prefill_cache_sim.calibration import (
    DEFAULT_HARDWARE_GATE,
    ENDPOINT_CONTRACT_VERSION,
    CalibrationStatus,
    DishonestLabelError,
    EndpointProtocolError,
    EndpointTransportError,
    EvidenceTier,
    HardwareBlocker,
    HardwareContext,
    HardwareEvidence,
    HardwareGate,
    HardwareGateReport,
    HttpJsonEngineEndpoint,
    HttpJsonTransport,
    MachineProvenance,
    StubTransport,
    TrustPolicy,
    endpoint_is_synthetic,
    residual_ratio,
)
from prefill_cache_sim.calibration.endpoint import MockEngine
from prefill_cache_sim.replay import (
    DEFAULT_REPLAY_HARDWARE_GATE,
    FROZEN_PLAN_DIGEST,
    FROZEN_RANKING_STATISTIC,
    REPLAY_HARDWARE_SCHEMA_VERSION,
    REQUIRED_CALIBRATION_TIER,
    ReplayBlocker,
    ReplayHardwareEvidence,
    ReplayHardwareGate,
    ReplayHardwareReport,
    ReplayPlan,
    plan_digest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _complete_machine() -> MachineProvenance:
    return MachineProvenance(
        "host-1",
        "A100-80GB",
        "vllm-0.6.3",
        "2026-08-06T00:00:00Z",
    )


def _complete_context() -> HardwareContext:
    return HardwareContext(
        _complete_machine(),
        ENDPOINT_CONTRACT_VERSION,
        "qwen-32b",
        "tp=2",
        "boost-clock-1410",
    )


# -- M9-HW gate -------------------------------------------------------------


def test_no_engine_blocks_on_everything() -> None:
    context = HardwareContext.unknown()
    evidence = HardwareEvidence()
    report = DEFAULT_HARDWARE_GATE.evaluate(context, evidence)
    assert not report.accepted
    assert HardwareBlocker.BLOCKED_NO_ENGINE_ACCESS in report.blockers
    assert HardwareBlocker.BLOCKED_SYNTHETIC_ENDPOINT in report.blockers
    assert HardwareBlocker.BLOCKED_INCOMPLETE_PROVENANCE in report.blockers


def test_synthetic_endpoint_always_blocks() -> None:
    context = _complete_context()
    evidence = HardwareEvidence(
        engine_reachable=True,
        endpoint_is_synthetic=True,
        observed_contract_version=ENDPOINT_CONTRACT_VERSION,
        observation_count=20,
        fit_residual_ratio=0.01,
    )
    report = DEFAULT_HARDWARE_GATE.evaluate(context, evidence)
    assert not report.accepted
    assert HardwareBlocker.BLOCKED_SYNTHETIC_ENDPOINT in report.blockers
    assert HardwareBlocker.BLOCKED_NO_ENGINE_ACCESS not in report.blockers


def test_dry_run_always_blocks() -> None:
    context = _complete_context()
    evidence = HardwareEvidence(
        engine_reachable=True,
        endpoint_is_synthetic=False,
        observed_contract_version=ENDPOINT_CONTRACT_VERSION,
        observation_count=20,
        fit_residual_ratio=0.01,
        dry_run=True,
    )
    report = DEFAULT_HARDWARE_GATE.evaluate(context, evidence)
    assert not report.accepted
    assert HardwareBlocker.BLOCKED_DRY_RUN_ONLY in report.blockers


def test_accepted_evidence_produces_hw_calibrated() -> None:
    context = _complete_context()
    evidence = HardwareEvidence(
        engine_reachable=True,
        endpoint_is_synthetic=False,
        production_trust=True,
        observed_contract_version=ENDPOINT_CONTRACT_VERSION,
        observation_count=20,
        fit_residual_ratio=0.01,
    )
    report = DEFAULT_HARDWARE_GATE.evaluate(context, evidence)
    assert report.accepted
    assert report.blockers == ()
    assert report.calibration_status.value == "HW_CALIBRATED"
    assert report.time_unit.value == "MILLISECONDS"
    assert report.evidence_tier.value == "HW_VALIDATED"


def test_hand_edited_accepted_is_rejected() -> None:
    context = _complete_context()
    evidence = HardwareEvidence()  # all defaults = blocked
    payload = DEFAULT_HARDWARE_GATE.evaluate(context, evidence).to_dict()
    payload["accepted"] = True
    payload["blockers"] = []
    with pytest.raises(DishonestLabelError):
        HardwareGateReport.from_dict(payload)


def test_hardware_gate_report_round_trips() -> None:
    context = _complete_context()
    evidence = HardwareEvidence(
        engine_reachable=True,
        endpoint_is_synthetic=False,
        production_trust=True,
        observed_contract_version=ENDPOINT_CONTRACT_VERSION,
        observation_count=20,
        fit_residual_ratio=0.05,
    )
    report = DEFAULT_HARDWARE_GATE.evaluate(context, evidence)
    rebuilt = HardwareGateReport.from_dict(report.to_dict())
    assert rebuilt == report


def test_context_missing_fields_named() -> None:
    context = HardwareContext(
        MachineProvenance.unknown(),
        None,
        None,
        None,
        None,
    )
    missing = context.missing_fields()
    assert "machine.host_id" in missing
    assert "machine.accelerator_model" in missing
    assert "model_version" in missing
    assert "topology" in missing
    assert not context.complete


def test_residual_ratio_is_scale_invariant() -> None:
    from prefill_cache_sim.calibration import (
        MockEngine,
        SweepKind,
        SweepSpec,
        fit_sweep,
        run_sweep,
    )

    endpoint = MockEngine()
    spec = SweepSpec(
        SweepKind.PREFILL,
        token_points=(128, 256, 512),
        batch_points=(1, 2, 4),
        repeats=2,
    )
    observations = run_sweep(endpoint, spec)
    fit = fit_sweep(observations)
    ratio = residual_ratio(fit, observations)
    assert 0.0 <= ratio < 1.0


# -- transport --------------------------------------------------------------


def test_stub_transport_records_calls() -> None:
    stub = StubTransport({"/test": {"ok": True}})
    reply = stub.request(path="/test", payload={"q": 1})
    assert reply == {"ok": True}
    assert stub.calls == [("/test", {"q": 1})]


def test_stub_transport_raises_on_missing_path() -> None:
    stub = StubTransport({"/known": {}})
    with pytest.raises(EndpointTransportError):
        stub.request(path="/unknown", payload={})


def test_http_transport_rejects_file_scheme() -> None:
    with pytest.raises(ValueError, match="scheme"):
        HttpJsonTransport(base_url="file:///etc/passwd", timeout_s=1.0)


def test_http_transport_requires_positive_timeout() -> None:
    with pytest.raises(ValueError, match="timeout"):
        HttpJsonTransport(base_url="http://localhost:8080", timeout_s=0.0)


# -- http_endpoint ----------------------------------------------------------


def _describe_reply() -> dict:
    return {
        "contract_version": ENDPOINT_CONTRACT_VERSION,
        "host_id": "host-1",
        "accelerator_model": "A100-80GB",
        "engine_version": "vllm-0.6.3",
        "captured_at_utc": "2026-08-06T00:00:00Z",
        "model_version": "qwen-32b",
        "topology": "tp=2",
        "clock_settings": "boost-1410",
    }


def _measure_reply(kind: str = "PREFILL") -> dict:
    return {
        "kind": kind,
        "tokens": 512,
        "batch_size": 1,
        "repeat_index": 0,
        "value_ms": 12.5,
    }


def test_describe_returns_complete_context() -> None:
    stub = StubTransport({"/v1/describe": _describe_reply()})
    endpoint = HttpJsonEngineEndpoint(endpoint_id="engine-1", transport=stub)
    context = endpoint.describe()
    assert context.complete
    assert context.endpoint_contract_version == ENDPOINT_CONTRACT_VERSION
    assert context.model_version == "qwen-32b"


def test_describe_rejects_missing_field() -> None:
    reply = _describe_reply()
    del reply["accelerator_model"]
    stub = StubTransport({"/v1/describe": reply})
    endpoint = HttpJsonEngineEndpoint(endpoint_id="e", transport=stub)
    with pytest.raises(EndpointProtocolError):
        endpoint.describe()


def test_measure_rejects_echo_mismatch() -> None:
    reply = _measure_reply()
    reply["tokens"] = 999
    stub = StubTransport({"/v1/measure": reply})
    endpoint = HttpJsonEngineEndpoint(endpoint_id="e", transport=stub)
    with pytest.raises(EndpointProtocolError, match="different grid point"):
        endpoint.measure(
            kind="PREFILL", tokens=512, batch_size=1, repeat_index=0
        )


def test_endpoint_is_synthetic_identifies_mock() -> None:
    mock = MockEngine()
    stub = StubTransport({"/v1/describe": _describe_reply()})
    real = HttpJsonEngineEndpoint(endpoint_id="e", transport=stub)
    assert endpoint_is_synthetic(mock) is True
    assert endpoint_is_synthetic(real) is False


# -- M10-HW gate ------------------------------------------------------------


def test_replay_blocked_evidence_has_all_blockers() -> None:
    evidence = ReplayHardwareEvidence()
    report = DEFAULT_REPLAY_HARDWARE_GATE.evaluate(
        evidence, MachineProvenance.unknown()
    )
    assert not report.accepted
    assert ReplayBlocker.BLOCKED_NO_ENGINE_ACCESS in report.blockers
    assert ReplayBlocker.BLOCKED_SYNTHETIC_CALIBRATION in report.blockers
    assert ReplayBlocker.BLOCKED_PLAN_NOT_FROZEN in report.blockers


def test_frozen_plan_digest_matches_default() -> None:
    assert plan_digest(ReplayPlan()) == FROZEN_PLAN_DIGEST


def test_replay_report_round_trips_blocked() -> None:
    evidence = ReplayHardwareEvidence()
    report = DEFAULT_REPLAY_HARDWARE_GATE.evaluate(
        evidence, MachineProvenance.unknown()
    )
    rebuilt = ReplayHardwareReport.from_dict(report.to_dict())
    assert rebuilt == report


def test_replay_gate_requires_hw_calibrated() -> None:
    assert REQUIRED_CALIBRATION_TIER == "HW_CALIBRATED"
    assert DEFAULT_REPLAY_HARDWARE_GATE.required_statistic == FROZEN_RANKING_STATISTIC


def test_replay_hand_edited_accepted_rejected() -> None:
    evidence = ReplayHardwareEvidence()
    report = DEFAULT_REPLAY_HARDWARE_GATE.evaluate(
        evidence, MachineProvenance.unknown()
    )
    payload = report.to_dict()
    payload["accepted"] = True
    payload["blockers"] = []
    with pytest.raises(DishonestLabelError):
        ReplayHardwareReport.from_dict(payload)


def test_replay_schema_version() -> None:
    assert REPLAY_HARDWARE_SCHEMA_VERSION == "m10-hardware-v1"


# -- script integration (blocked path only) --------------------------------


def _run_script(script: str, tmp_path: Path) -> tuple[int, Path]:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / script),
            "--output-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    return result.returncode, tmp_path


def test_m9_hardware_blocked_writes_blocked_dir(tmp_path: Path) -> None:
    code, root = _run_script("run_m9_hardware.py", tmp_path)
    assert code == 2
    blocked = root / "results" / "m9-hardware-blocked"
    accepted = root / "results" / "m9-hardware"
    assert blocked.is_dir()
    assert not accepted.exists()
    gate = json.loads((blocked / "GATE.json").read_text())
    assert gate["report"]["accepted"] is False
    text = (blocked / "GATE.json").read_text()
    assert "HW_CALIBRATED" not in text or gate["report"]["accepted"]


def test_m10_hardware_blocked_writes_blocked_dir(tmp_path: Path) -> None:
    code, root = _run_script("run_m10_hardware.py", tmp_path)
    assert code == 2
    blocked = root / "results" / "m10-hardware-blocked"
    accepted = root / "results" / "m10-hardware"
    assert blocked.is_dir()
    assert not accepted.exists()
    gate = json.loads((blocked / "GATE.json").read_text())
    assert gate["report"]["accepted"] is False


def test_m9_hardware_dry_run_is_blocked(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_m9_hardware.py"),
            "--output-root",
            str(tmp_path),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 2
    assert "BLOCKED_DRY_RUN_ONLY" in result.stdout


# -- trust policy (Fix #2) --------------------------------------------------


def test_trust_policy_https_allowlist_trusted() -> None:
    policy = TrustPolicy(trusted_hosts=frozenset({"engine.prod.internal"}))
    assert policy.is_production_trusted("https://engine.prod.internal:8443/v1")


def test_trust_policy_plain_http_not_trusted() -> None:
    policy = TrustPolicy(trusted_hosts=frozenset({"localhost"}))
    assert not policy.is_production_trusted("http://localhost:8080/v1")


def test_trust_policy_https_unknown_host_not_trusted() -> None:
    policy = TrustPolicy(trusted_hosts=frozenset({"engine.prod.internal"}))
    assert not policy.is_production_trusted("https://engine.dev.internal/v1")


def test_untrusted_endpoint_gets_remote_reported_not_hw() -> None:
    context = _complete_context()
    evidence = HardwareEvidence(
        engine_reachable=True,
        endpoint_is_synthetic=False,
        production_trust=False,
        observed_contract_version=ENDPOINT_CONTRACT_VERSION,
        observation_count=20,
        fit_residual_ratio=0.01,
    )
    report = DEFAULT_HARDWARE_GATE.evaluate(context, evidence)
    assert not report.accepted
    assert HardwareBlocker.BLOCKED_NOT_PRODUCTION_TRUSTED in report.blockers
    assert report.evidence_tier is EvidenceTier.REMOTE_REPORTED
    assert report.calibration_status is not CalibrationStatus.HW_CALIBRATED
    assert report.evidence_tier is not EvidenceTier.HW_VALIDATED


def test_trusted_endpoint_can_be_hw_calibrated() -> None:
    context = _complete_context()
    evidence = HardwareEvidence(
        engine_reachable=True,
        endpoint_is_synthetic=False,
        production_trust=True,
        observed_contract_version=ENDPOINT_CONTRACT_VERSION,
        observation_count=20,
        fit_residual_ratio=0.01,
    )
    report = DEFAULT_HARDWARE_GATE.evaluate(context, evidence)
    assert report.accepted
    assert report.evidence_tier is EvidenceTier.HW_VALIDATED


# -- non-production policy (Fix #3) -----------------------------------------


def test_m9_cli_override_produces_non_production() -> None:
    gate = HardwareGate(minimum_observations=15, production=False)
    context = _complete_context()
    evidence = HardwareEvidence(
        engine_reachable=True,
        endpoint_is_synthetic=False,
        production_trust=True,
        observed_contract_version=ENDPOINT_CONTRACT_VERSION,
        observation_count=20,
        fit_residual_ratio=0.01,
    )
    report = gate.evaluate(context, evidence)
    assert not report.accepted
    assert HardwareBlocker.BLOCKED_NON_PRODUCTION_POLICY in report.blockers
    assert report.calibration_status is CalibrationStatus.NON_PRODUCTION_EXPERIMENT
    assert report.evidence_tier is not EvidenceTier.HW_VALIDATED


def test_m10_cli_override_produces_non_production() -> None:
    gate = ReplayHardwareGate(minimum_tau_b=0.85, production=False)
    evidence = ReplayHardwareEvidence(
        engine_reachable=True,
        calibration_status=REQUIRED_CALIBRATION_TIER,
        calibration_accepted=True,
        calibration_endpoint_synthetic=False,
        provenance_complete=True,
        observed_plan_digest=FROZEN_PLAN_DIGEST,
        ranking_statistic=FROZEN_RANKING_STATISTIC,
        tau_b=0.95,
        reconciled_fraction=0.999,
        disagreement_fraction=0.0,
        fault_injection_detected=True,
    )
    report = gate.evaluate(evidence, _complete_machine())
    assert not report.accepted
    assert ReplayBlocker.BLOCKED_NON_PRODUCTION_POLICY in report.blockers
    assert (
        report.calibration_status is CalibrationStatus.NON_PRODUCTION_EXPERIMENT
    )


def test_frozen_plan_digest_is_pinned_literal() -> None:
    assert isinstance(FROZEN_PLAN_DIGEST, str)
    assert len(FROZEN_PLAN_DIGEST) == 64
    assert plan_digest(ReplayPlan()) == FROZEN_PLAN_DIGEST


def test_m10_relaxed_threshold_refused() -> None:
    gate = DEFAULT_REPLAY_HARDWARE_GATE
    evidence = ReplayHardwareEvidence(
        engine_reachable=True,
        calibration_status=REQUIRED_CALIBRATION_TIER,
        calibration_accepted=True,
        calibration_endpoint_synthetic=False,
        provenance_complete=True,
        observed_plan_digest=FROZEN_PLAN_DIGEST,
        ranking_statistic=FROZEN_RANKING_STATISTIC,
        tau_b=0.5,
        reconciled_fraction=0.999,
        disagreement_fraction=0.0,
        fault_injection_detected=True,
    )
    report = gate.evaluate(evidence, _complete_machine())
    assert not report.accepted
    assert ReplayBlocker.BLOCKED_RANKING_BELOW_GATE in report.blockers


def test_m10_plan_drift_blocked() -> None:
    evidence = ReplayHardwareEvidence(
        engine_reachable=True,
        calibration_status=REQUIRED_CALIBRATION_TIER,
        calibration_accepted=True,
        calibration_endpoint_synthetic=False,
        provenance_complete=True,
        observed_plan_digest="0" * 64,
        ranking_statistic=FROZEN_RANKING_STATISTIC,
        tau_b=0.95,
        reconciled_fraction=0.999,
        disagreement_fraction=0.0,
        fault_injection_detected=True,
    )
    report = DEFAULT_REPLAY_HARDWARE_GATE.evaluate(evidence, _complete_machine())
    assert not report.accepted
    assert ReplayBlocker.BLOCKED_PLAN_NOT_FROZEN in report.blockers


# -- M10 calibration loader (Fix #1) ---------------------------------------


def _write_blocked_m9_artifact(
    directory: Path, *, git_dirty: bool = False
) -> None:
    """Write a minimal blocked M9-HW artifact directory for loader tests."""
    import csv
    import hashlib
    import io
    import json

    directory.mkdir(parents=True, exist_ok=True)
    context = HardwareContext.unknown()
    evidence = HardwareEvidence()
    report = DEFAULT_HARDWARE_GATE.evaluate(context, evidence)
    gate_payload = {
        "report": report.to_dict(),
        "handshake_error": None,
        "endpoint_url_provided": False,
        "provenance": {
            "git_sha": "test123",
            "git_dirty": git_dirty,
            "source_fingerprints": {"combined_digest": "abcd"},
            "schema_version": "m9-calibration-v1",
        },
    }
    gate_bytes = (
        json.dumps(gate_payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    params_bytes = (
        json.dumps({"params": {"endpoint_id": "test"}}, indent=2) + "\n"
    ).encode("utf-8")
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=["case_id"])
    w.writeheader()
    w.writerow({"case_id": "M9HW-PREFILL"})
    results_bytes = buf.getvalue().encode("utf-8")
    obs_buf = io.StringIO(newline="")
    ow = csv.DictWriter(obs_buf, fieldnames=["case_id"])
    ow.writeheader()
    ow.writerow({"case_id": "M9HW-PREFILL"})
    obs_bytes = obs_buf.getvalue().encode("utf-8")

    files = {
        "GATE.json": gate_bytes,
        "params.json": params_bytes,
        "results.csv": results_bytes,
        "observations.csv": obs_bytes,
    }
    for name, data in files.items():
        (directory / name).write_bytes(data)
    manifest = {
        "schema_version": "m9-hardware-manifest-v1",
        "algorithm": "sha256",
        "files": {
            name: hashlib.sha256(data).hexdigest() for name, data in files.items()
        },
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    (directory / "MANIFEST.json").write_bytes(manifest_bytes)


def test_m10_calibration_missing_files_rejected(tmp_path: Path) -> None:
    """An M9 artifact missing required files is rejected."""
    cal_dir = tmp_path / "m9-incomplete"
    _write_blocked_m9_artifact(cal_dir)
    (cal_dir / "params.json").unlink()  # remove a required file
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_m10_hardware.py"),
            "--calibration",
            str(cal_dir),
            "--output-root",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 2
    gate_path = tmp_path / "out" / "results" / "m10-hardware-blocked" / "GATE.json"
    gate = json.loads(gate_path.read_text())
    assert gate["calibration_error"] is not None
    assert "missing" in gate["calibration_error"].lower()


def test_m10_calibration_tampered_manifest_rejected(tmp_path: Path) -> None:
    """A tampered MANIFEST (hash mismatch) is rejected."""
    cal_dir = tmp_path / "m9-tampered"
    _write_blocked_m9_artifact(cal_dir)
    # Tamper: modify GATE.json after MANIFEST was written.
    gate_path = cal_dir / "GATE.json"
    original = json.loads(gate_path.read_text())
    original["extra_field"] = "tampered"
    gate_path.write_text(json.dumps(original, indent=2))
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_m10_hardware.py"),
            "--calibration",
            str(cal_dir),
            "--output-root",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 2
    gate_path = tmp_path / "out" / "results" / "m10-hardware-blocked" / "GATE.json"
    gate = json.loads(gate_path.read_text())
    assert gate["calibration_error"] is not None
    assert "digest mismatch" in gate["calibration_error"].lower()


def test_m10_calibration_dirty_git_rejected(tmp_path: Path) -> None:
    """An M9 artifact built from a dirty git tree is rejected."""
    cal_dir = tmp_path / "m9-dirty"
    _write_blocked_m9_artifact(cal_dir, git_dirty=True)
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_m10_hardware.py"),
            "--calibration",
            str(cal_dir),
            "--output-root",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 2
    gate_path = tmp_path / "out" / "results" / "m10-hardware-blocked" / "GATE.json"
    gate = json.loads(gate_path.read_text())
    assert gate["calibration_error"] is not None
    assert "dirty" in gate["calibration_error"].lower()
