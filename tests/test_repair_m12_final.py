from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path

import pytest

from scripts import repair_m12_final as repair
from scripts import run_m12_final

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "m12-final"
REPAIR = ROOT / "results" / "m12-repair"
REPAIR_MANIFEST = ROOT / "docs" / "evidence" / "m12-repair-manifest.json"


@pytest.fixture
def repair_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build an immutable 44-row input from the published final artifact.

    Production still trusts the original pinned digest.  Tests use the same
    validator and artifact set with a locally generated trust anchor so the
    repair path remains executable after the final directory is published.
    """
    base = tmp_path / "pre-repair-base"
    for name in repair.EXPECTED_BASE_FILES:
        target = base / name
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = (BASE / name).read_bytes()
        if name in {"primary.csv", "constraints.csv", "explanation.csv"}:
            rows = [
                row
                for row in _rows(payload)
                if row["cell_id"] != repair.MISSING_CELL
            ]
            payload = run_m12_final._csv_mapping_bytes(rows)
        elif name == "provenance.json":
            document = json.loads(payload)
            document["git_sha"] = repair.BASE_SHA
            document["git_dirty"] = False
            payload = run_m12_final._json_bytes(document)
        elif name == "config.json":
            document = json.loads(payload)
            document.update(completed_cell_count=53, failed_cell_count=1)
            document.pop("repair_mode", None)
            payload = run_m12_final._json_bytes(document)
        elif name == "gates/g12-3.json":
            document = json.loads(payload)
            document["cells"] = [
                row
                for row in document["cells"]
                if not (
                    row.get("regime") == "MIXED"
                    and row.get("strategy") == "DECODE_CAUSAL"
                )
            ]
            document["retry_pressure_covered"] = False
            document["overall_verdict"] = "INCOMPLETE"
            payload = run_m12_final._json_bytes(document)
        elif name == "attribution.json":
            document = json.loads(payload)
            document["records"] = [
                row
                for row in document["records"]
                if not (
                    row.get("regime") == "MIXED"
                    and row.get("arrival_scale") == 1.5
                    and row.get("single_switch") == "DECODE_CREDITS"
                )
            ]
            document["overall_verdict"] = "INCOMPLETE"
            payload = run_m12_final._json_bytes(document)
        elif name == "falsification/visibility-delay.json":
            document = json.loads(payload)
            document["rows"] = [
                row
                for row in document["rows"]
                if row.get("cell_id") != repair.MISSING_CELL
            ]
            payload = run_m12_final._json_bytes(document)
        elif name == "pareto.json":
            document = json.loads(payload)
            document["overall_verdict"] = "INCOMPLETE"
            payload = run_m12_final._json_bytes(document)
        target.write_bytes(payload)
    files = {
        name: hashlib.sha256((base / name).read_bytes()).hexdigest()
        for name in sorted(repair.EXPECTED_BASE_FILES)
    }
    manifest = run_m12_final._json_bytes(
        {
            "algorithm": "sha256",
            "files": files,
            "schema_version": "m12-final-manifest-v1",
        }
    )
    (base / "MANIFEST.json").write_bytes(manifest)
    monkeypatch.setattr(
        repair, "BASE_MANIFEST_SHA256", hashlib.sha256(manifest).hexdigest()
    )
    return base


def _rows(payload: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(payload.decode())))


@pytest.mark.skipif(not REPAIR.exists(), reason="local repair bundle is unavailable")
def test_repair_is_deterministic_complete_and_manifest_valid(repair_base: Path) -> None:
    first = repair.build_repair(repair_base, REPAIR, REPAIR_MANIFEST)
    second = repair.build_repair(repair_base, REPAIR, REPAIR_MANIFEST)

    assert first == second
    assert len(_rows(first["primary.csv"])) == 45
    assert len(_rows(first["sensitivities.csv"])) == 9
    assert _rows(first["failures.csv"]) == []
    config = json.loads(first["config.json"])
    assert (config["planned_cell_count"], config["completed_cell_count"]) == (54, 54)
    assert config["failed_cell_count"] == 0

    manifest = json.loads(first["MANIFEST.json"])
    assert manifest["files"] == {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in sorted(first.items())
        if name != "MANIFEST.json"
    }


@pytest.mark.skipif(not REPAIR.exists(), reason="local repair bundle is unavailable")
def test_repair_closes_incomplete_derived_artifacts_honestly(repair_base: Path) -> None:
    artifacts = repair.build_repair(repair_base, REPAIR, REPAIR_MANIFEST)

    assert json.loads(artifacts["pareto.json"])["overall_verdict"] == "COMPLETE"
    attribution = json.loads(artifacts["attribution.json"])
    assert attribution["overall_verdict"] == "COMPLETE"
    assert any(
        row.get("regime") == "MIXED"
        and row.get("arrival_scale") == 1.5
        and row.get("single_switch") == "DECODE_CREDITS"
        for row in attribution["records"]
    )
    g12_3 = json.loads(artifacts["gates/g12-3.json"])
    assert g12_3["overall_verdict"] == "NARROW_OVERLOAD_ONLY"
    assert g12_3["retry_pressure_covered"] is True
    g12_4 = json.loads(artifacts["gates/g12-4.json"])
    assert g12_4["overall_verdict"] == "KILL_OR_NARROW"
    assert g12_4["cells"] == []
    visibility = json.loads(artifacts["falsification/visibility-delay.json"])
    assert any(row["cell_id"] == repair.MISSING_CELL for row in visibility["rows"])
    config = json.loads(artifacts["config.json"])
    assert (config["completed_cell_count"], config["failed_cell_count"]) == (54, 0)


@pytest.mark.skipif(not REPAIR.exists(), reason="local repair bundle is unavailable")
def test_repair_refuses_failed_invariance_gate(
    repair_base: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = repair._load_pickle

    def failed(path: Path, digest: str):
        payload = original(path, digest)
        if path.name == "MIXED-1.5x-BASELINE.pkl":
            result = payload["result"]
            payload = {**payload, "result": replace(result, strict_goodput=0.0)}
        return payload

    monkeypatch.setattr(repair, "_load_pickle", failed)
    with pytest.raises(ValueError, match="invariance gate failed"):
        repair.build_repair(repair_base, REPAIR, REPAIR_MANIFEST)


def test_pickle_hash_is_checked_before_deserialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def must_not_run(_stream):
        raise AssertionError("pickle deserialization ran before hash validation")

    monkeypatch.setattr(repair.pickle, "load", must_not_run)
    with pytest.raises(ValueError, match="hash mismatch"):
        repair._load_pickle(REPAIR / "missing-cell.pkl", "0" * 64)


def test_alternate_self_consistent_manifest_is_not_a_trust_anchor(
    tmp_path: Path,
) -> None:
    document = json.loads(REPAIR_MANIFEST.read_text())
    document["payloads"]["missing-cell.pkl"] = "0" * 64
    alternate = tmp_path / "alternate.json"
    alternate.write_text(json.dumps(document, sort_keys=True))

    with pytest.raises(ValueError, match="untrusted repair manifest digest"):
        repair._validate_repair_manifest(alternate, REPAIR)


def test_repair_manifest_is_read_only_once(monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.read_bytes
    manifest_reads = 0

    def counted(path: Path) -> bytes:
        nonlocal manifest_reads
        if path == REPAIR_MANIFEST:
            manifest_reads += 1
            if manifest_reads > 1:
                raise AssertionError("repair manifest was reopened after verification")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", counted)
    repair._validate_repair_manifest(REPAIR_MANIFEST, REPAIR)
    assert manifest_reads == 1


def test_alternate_base_manifest_is_not_a_trust_anchor(
    tmp_path: Path, repair_base: Path
) -> None:
    base = tmp_path / "base"
    base.mkdir()
    document = json.loads((repair_base / "MANIFEST.json").read_text())
    document["files"]["primary.csv"] = "0" * 64
    (base / "MANIFEST.json").write_text(json.dumps(document, sort_keys=True))

    with pytest.raises(ValueError, match="untrusted base manifest digest"):
        repair._validate_manifest(base)


def test_base_snapshot_reads_each_artifact_once(
    repair_base: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = Path.read_bytes
    reads: dict[Path, int] = {}

    def counted(path: Path) -> bytes:
        if path == repair_base / "MANIFEST.json" or repair_base in path.parents:
            reads[path] = reads.get(path, 0) + 1
            if reads[path] > 1:
                raise AssertionError(f"base artifact reopened: {path}")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", counted)
    _manifest, snapshot, _manifest_bytes = repair._validate_manifest(repair_base)
    assert set(snapshot) == repair.EXPECTED_BASE_FILES
    assert all(count == 1 for count in reads.values())


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../b", "a\\b"])
def test_artifact_names_reject_path_escape(name: str) -> None:
    assert not repair._safe_name(name)


def test_atomic_writer_rejects_path_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsafe artifact path"):
        run_m12_final._write_atomic(
            tmp_path / "output",
            {"../escape": b"bad", "MANIFEST.json": b"{}"},
        )
    assert not (tmp_path / "escape").exists()


@pytest.mark.skipif(not REPAIR.exists(), reason="local repair bundle is unavailable")
def test_atomic_publish_is_idempotent(tmp_path: Path, repair_base: Path) -> None:
    artifacts = repair.build_repair(repair_base, REPAIR, REPAIR_MANIFEST)
    output = tmp_path / "published"
    run_m12_final._write_atomic(output, artifacts)
    first = {
        path.relative_to(output): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    run_m12_final._write_atomic(output, artifacts)
    second = {
        path.relative_to(output): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert first == second
