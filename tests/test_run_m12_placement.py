from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts.run_m12_placement import (
    TRUTH_BASIS,
    ArtifactGridFailure,
    ArtifactSetInvalid,
    _manifest_bytes,
    _write_atomic,
    build_trace_workload,
    read_valid_artifact_set,
    run_artifacts,
)


def write_trace(path: Path) -> None:
    rows = [
        {
            "timestamp": 0,
            "input_length": 512,
            "output_length": 100_000,
            "hash_ids": [1],
        },
        {
            "timestamp": 10,
            "input_length": 512,
            "output_length": 100_000,
            "hash_ids": [1],
        },
        {
            "timestamp": 20,
            "input_length": 600,
            "output_length": 100_000,
            "hash_ids": [2, 3],
        },
    ]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_trace_workload_assignment_is_stable_and_not_loop_order(tmp_path: Path) -> None:
    trace = tmp_path / "fixture.jsonl"
    write_trace(trace)
    first, metadata = build_trace_workload(trace)
    second, _ = build_trace_workload(trace)
    assert metadata["truth_basis"] == TRUTH_BASIS
    assert metadata["record_count"] == 3
    assert [item.logical.tenant_id for item in first] == [
        item.logical.tenant_id for item in second
    ]
    assert all(item.logical.tenant_id.startswith("tenant-") for item in first)
    assert {item.logical.tier for item in first} <= {"STRICT", "STANDARD", "RELAXED"}
    assert first[2].prefix_token_sizes == (512, 88)


def test_artifacts_are_deterministic_atomic_and_manifested(tmp_path: Path) -> None:
    trace = tmp_path / "fixture.jsonl"
    write_trace(trace)
    left = tmp_path / "left"
    right = tmp_path / "right"
    run_artifacts(trace, left)
    run_artifacts(trace, right)
    names = {
        "contract.json",
        "config.json",
        "results.csv",
        "comparisons/g12-verdicts.csv",
        "failures.csv",
    }
    left_manifest = json.loads((left / "MANIFEST.json").read_text())
    right_manifest = json.loads((right / "MANIFEST.json").read_text())
    assert set(left_manifest["files"]) == names
    assert left_manifest == right_manifest
    for name, digest in left_manifest["files"].items():
        payload = (left / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == digest
        assert payload == (right / name).read_bytes()
    config = json.loads((left / "config.json").read_text())
    assert config["trace_sha256"] == metadata_sha(trace)
    assert config["trace_record_count"] == 3
    assert config["truth_basis"] == TRUTH_BASIS
    assert config["case_count"] == 15
    assert config["successful_case_count"] == 15
    assert config["result_count"] == 90
    assert config["verdict_count"] == 15
    assert config["source_fingerprints"]["scripts/run_m12_placement.py"]
    assert len(config["source_combined_sha256"]) == 64
    assert len(config["dirty_patch_sha256"]) == 64
    assert read_valid_artifact_set(left) == left_manifest
    verdict_header = (left / "comparisons/g12-verdicts.csv").read_text().splitlines()[0]
    assert "baseline_also_fails_floor" in verdict_header
    assert "cause" in verdict_header


def test_failed_grid_publishes_diagnostics_then_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = tmp_path / "fixture.jsonl"
    write_trace(trace)
    output = tmp_path / "failed"

    def fail(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected placement failure")

    monkeypatch.setattr("scripts.run_m12_placement.run_placement_case", fail)
    with pytest.raises(ArtifactGridFailure, match="15 placement cases failed"):
        run_artifacts(trace, output)
    assert read_valid_artifact_set(output)["files"]
    config = json.loads((output / "config.json").read_text())
    assert config["successful_case_count"] == 0
    assert config["failed_case_count"] == 15


def test_reader_rejects_mixed_set_after_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "published"
    first = {"payload.txt": b"old\n"}
    first["MANIFEST.json"] = _manifest_bytes(first)
    _write_atomic(output, first)
    assert read_valid_artifact_set(output)["files"]

    replacement = {"payload.txt": b"new\n"}
    replacement["MANIFEST.json"] = _manifest_bytes(replacement)
    real_replace = os.replace
    calls = 0

    def fail_manifest(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected manifest replace failure")
        real_replace(source, destination)

    monkeypatch.setattr("scripts.run_m12_placement.os.replace", fail_manifest)
    with pytest.raises(OSError, match="injected manifest"):
        _write_atomic(output, replacement)
    with pytest.raises(ArtifactSetInvalid, match="digest mismatch"):
        read_valid_artifact_set(output)


def metadata_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
