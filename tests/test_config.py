from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from prefill_cache_sim.config import load_scenario, verify_trace_sha256

ROOT = Path(__file__).resolve().parents[1]


def test_baseline_resolves_paths_and_exact_config_sha() -> None:
    path = ROOT / "scenarios/baseline.local-lru.json"
    loaded = load_scenario(path, schema_path=ROOT / "scenario.schema.json")
    assert loaded.trace_path == ROOT / "mooncake_trace.jsonl"
    assert loaded.output_directory == ROOT / "results/phase-a-local-lru-delayed"
    assert loaded.config_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert verify_trace_sha256(loaded) == loaded.value["trace"]["sha256"]


def test_trace_digest_mismatch_fails_closed(tmp_path: Path) -> None:
    scenario = json.loads((ROOT / "scenarios/baseline.local-lru.json").read_text())
    trace = tmp_path / "trace.jsonl"
    trace.write_text("changed")
    scenario["trace"]["path"] = "trace.jsonl"
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(scenario))
    loaded = load_scenario(path, schema_path=ROOT / "scenario.schema.json")
    with pytest.raises(ValueError, match="trace.sha256 mismatch"):
        verify_trace_sha256(loaded)
