"""Strict resolved-scenario loading and provenance."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


@dataclass(frozen=True, slots=True)
class LoadedScenario:
    path: Path
    value: dict[str, Any]
    config_sha256: str
    trace_path: Path
    output_directory: Path


@dataclass(frozen=True, slots=True)
class GitProvenance:
    sha: str | None
    dirty: bool | None


def git_provenance(repository: Path) -> GitProvenance:
    """Return code SHA and dirty state, or null fields when git is unavailable."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if revision.returncode != 0:
            return GitProvenance(None, None)
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except OSError:
        return GitProvenance(None, None)
    dirty = bool(status.stdout) if status.returncode == 0 else None
    return GitProvenance(revision.stdout.strip(), dirty)


def load_scenario(path: Path, *, schema_path: Path) -> LoadedScenario:
    """Validate exact scenario bytes and resolve paths relative to the scenario file."""
    payload = path.read_bytes()
    value = json.loads(payload)
    schema = json.loads(schema_path.read_bytes())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(value)
    base = path.resolve().parent
    if Path(value["trace"]["path"]).is_absolute():
        raise ValueError("trace.path must be relative to the scenario file")
    if Path(value["output"]["directory"]).is_absolute():
        raise ValueError("output.directory must be relative to the scenario file")
    trace_path = (base / value["trace"]["path"]).resolve()
    output_directory = (base / value["output"]["directory"]).resolve()
    cluster = value["cluster"]
    capacity = cluster["cache_capacity"]
    if capacity["mode"] == "FIXED_TOTAL" and (
        capacity["total_blocks"] < cluster["prefill_nodes"]
    ):
        raise ValueError("cluster.cache_capacity.total_blocks < prefill_nodes")
    selector = value["selector"]
    if selector["id"] == "S4_SESSION_AFFINITY" and (
        selector["affinity"]["family_size_cap"] > cluster["prefill_nodes"]
    ):
        raise ValueError("selector.affinity.family_size_cap > prefill_nodes")
    return LoadedScenario(
        path.resolve(),
        value,
        hashlib.sha256(payload).hexdigest(),
        trace_path,
        output_directory,
    )


def verify_trace_sha256(scenario: LoadedScenario) -> str:
    """Fail closed unless the trace bytes match the resolved scenario."""
    hasher = hashlib.sha256()
    with scenario.trace_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    actual = hasher.hexdigest()
    expected = scenario.value["trace"]["sha256"]
    if actual != expected:
        raise ValueError(f"trace.sha256 mismatch: expected {expected}, got {actual}")
    return actual
