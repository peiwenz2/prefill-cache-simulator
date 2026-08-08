from __future__ import annotations

import csv
import hashlib
import io
import json

import pytest

import scripts.run_m12_sizing as sizing_runner
from prefill_cache_sim.m12_placement import TraceRequestInput, build_kernel_requests
from prefill_cache_sim.m12_sizing import (
    GateObservation,
    SizingCell,
    SizingRunRecord,
    SizingTopology,
)
from scripts.run_m12_sizing import (
    BASELINE_GATES,
    build_artifacts,
    build_plan,
    publish,
    restore_raw_trace_timestamps,
)


def record(
    topology: SizingTopology,
    p_count: int,
    *,
    completion: float,
) -> SizingRunRecord:
    observation = GateObservation(completion, 0.9, 0.95, 10, 0.25)
    failed = () if completion == 1 else ("COMPLETION_FLOOR",)
    return SizingRunRecord(
        SizingCell(p_count, topology, observation, failed),
        1.0,
        0.1,
        0.01,
        0.5,
        10,
        5,
        15,
        0.7,
        0.2,
        {"STRICT": 0.9, "STANDARD": 0.95, "RELAXED": 0.99},
        hashlib.sha256(f"{topology}-{p_count}".encode()).hexdigest(),
    )


def test_plan_is_complete_canonical_and_rejects_invalid_counts() -> None:
    plan = build_plan(
        (3, 1, 2),
        (SizingTopology.SHARED_KVS, SizingTopology.LOCAL_ONLY),
    )
    assert plan == (
        (SizingTopology.LOCAL_ONLY, 1),
        (SizingTopology.LOCAL_ONLY, 2),
        (SizingTopology.LOCAL_ONLY, 3),
        (SizingTopology.SHARED_KVS, 1),
        (SizingTopology.SHARED_KVS, 2),
        (SizingTopology.SHARED_KVS, 3),
    )
    with pytest.raises(ValueError):
        build_plan((True, 2))


def test_artifacts_are_deterministic_and_exclude_zero_price_control_from_winner(
    tmp_path, monkeypatch
) -> None:
    records = (
        record(SizingTopology.ZERO_TRANSFER_PRICE_CONTROL, 1, completion=1),
        record(SizingTopology.LOCAL_ONLY, 1, completion=0.9),
        record(SizingTopology.LOCAL_ONLY, 2, completion=1),
        record(SizingTopology.SHARED_KVS, 1, completion=0.9),
        record(SizingTopology.SHARED_KVS, 2, completion=1),
    )
    clean = type("Provenance", (), {"sha": "b" * 40, "dirty": False})()
    monkeypatch.setattr(sizing_runner, "git_provenance", lambda _root: clean)
    first = build_artifacts(
        records,
        {"trace_sha256": "a" * 64, "record_count": 23_608},
    )
    second = build_artifacts(
        tuple(reversed(records)),
        {"trace_sha256": "a" * 64, "record_count": 23_608},
    )
    assert first == second
    manifest = json.loads(first["MANIFEST.json"])
    for name, digest in manifest["files"].items():
        assert hashlib.sha256(first[name]).hexdigest() == digest
    verdict = json.loads(first["verdict.json"])
    assert verdict["baseline"]["DEPLOYABLE_WINNER"] == {
        "grid_exhausted": False,
        "minimum_feasible_p": 2,
        "predecessor_certificates": [
            {
                "failed_gates": ["COMPLETION_FLOOR"],
                "observed": {"COMPLETION_FLOOR": 0.9},
                "p_count": 1,
                "required": {"COMPLETION_FLOOR": 1.0},
                "topology": "LOCAL_ONLY",
            },
            {
                "failed_gates": ["COMPLETION_FLOOR"],
                "observed": {"COMPLETION_FLOOR": 0.9},
                "p_count": 1,
                "required": {"COMPLETION_FLOOR": 1.0},
                "topology": "SHARED_KVS",
            },
        ],
        "selected_topology": SizingTopology.LOCAL_ONLY.value,
    }
    assert (
        verdict["baseline"][SizingTopology.ZERO_TRANSFER_PRICE_CONTROL.value][
            "deployable"
        ]
        is False
    )
    assert (
        verdict["baseline"][SizingTopology.ZERO_TRANSFER_PRICE_CONTROL.value][
            "minimum_feasible_p"
        ]
        == 1
    )
    provenance = json.loads(first["provenance.json"])
    assert provenance["record_count"] == 23_608
    assert provenance["sizing_cell_count"] == len(records)
    contract = json.loads(first["contract.json"])
    assert contract["schema_version"] == "m12-sizing-v2.1"
    assert contract["service_costs"]["prefill_token_work"] == 0.06
    assert contract["service_costs"]["decode_token_work"] == 1.0
    assert contract["service_costs"]["kvs_token_work"] == 0.01
    assert contract["service_costs"]["kvs_bytes_per_token"] == 65_536
    cells = list(csv.DictReader(io.StringIO(first["cells.csv"].decode())))
    assert cells[0]["strict_slo_attainment"] == "0.9"
    assert cells[0]["standard_slo_attainment"] == "0.95"
    assert cells[0]["relaxed_slo_attainment"] == "0.99"
    frontier_text = first["threshold-frontier.csv"].decode()
    frontier = list(csv.DictReader(io.StringIO(frontier_text)))
    assert len(frontier) == 29 * 3
    assert {float(row["tier_slo_floor"]) for row in frontier} == {
        round(0.70 + 0.01 * index, 2) for index in range(29)
    }
    assert {row["topology"] for row in frontier} == {
        topology.value for topology in SizingTopology
    }
    output = tmp_path / "result"
    publish(output, first)
    assert (output / "MANIFEST.json").read_bytes() == first["MANIFEST.json"]


def test_dirty_provenance_is_rejected(monkeypatch) -> None:
    dirty = type("Provenance", (), {"sha": "b" * 40, "dirty": True})()
    monkeypatch.setattr(sizing_runner, "git_provenance", lambda _root: dirty)
    with pytest.raises(RuntimeError, match="dirty tree"):
        build_artifacts(
            (record(SizingTopology.LOCAL_ONLY, 1, completion=1),),
            {"trace_sha256": "a" * 64},
        )


def test_baseline_gate_contract_is_frozen() -> None:
    assert BASELINE_GATES.minimum_tier_slo_attainment == 0.8
    assert BASELINE_GATES.minimum_jain_fairness == 0.9


def test_sizing_restores_raw_trace_timestamps_instead_of_m12_slowdown() -> None:
    workload = build_kernel_requests(
        [
            TraceRequestInput(
                "r",
                "tenant",
                "STANDARD",
                50,
                ("A",),
                (10,),
                1,
                "model",
                "adapter",
                "shape",
                100,
                ("p0", "p1"),
            )
        ]
    )
    restored = restore_raw_trace_timestamps(workload)
    assert restored[0].logical.arrival_work == 10
    assert restored.request_truth == workload.request_truth
