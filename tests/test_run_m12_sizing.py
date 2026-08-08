from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import replace

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
        0.4,
        1.0,
        2.0,
        3.0,
        4.0,
        3,
        5,
        0.2,
        {"STRICT": 0.9, "STANDARD": 0.95, "RELAXED": 0.99},
        {
            "STRICT": (10.0, 20.0),
            "STANDARD": (10.0,),
            "RELAXED": (10.0,),
        },
        (
            ("tenant-a", "STRICT", 10, 10.0),
            ("tenant-b", "STRICT", 10, 20.0),
            ("tenant-a", "STANDARD", 10, 10.0),
            ("tenant-b", "RELAXED", 10, 10.0),
        ),
        hashlib.sha256(f"{topology}-{p_count}".encode()).hexdigest(),
    )


@pytest.mark.parametrize(
    "tiers",
    [
        {"STRICT": 0.9, "STANDARD": 0.95},
        {
            "STRICT": 0.9,
            "STANDARD": 0.95,
            "RELAXED": 0.99,
            "EXTRA": 1.0,
        },
    ],
)
def test_record_rejects_tier_maps_that_serializer_cannot_publish(
    tiers: dict[str, float],
) -> None:
    observation = GateObservation(1.0, min(tiers.values()), 0.95, 10, 0.25)
    invalid = SizingRunRecord(
            SizingCell(1, SizingTopology.LOCAL_ONLY, observation, ()),
            1.0,
            0.1,
            0.01,
            0.5,
            10,
            5,
            15,
            0.7,
            0.4,
            1.0,
            2.0,
            3.0,
            4.0,
            3,
            5,
            0.2,
            tiers,
            {tier: (10.0,) for tier in tiers},
            tuple(("tenant", tier, 10, 10.0) for tier in tiers),
            "f" * 64,
        )
    with pytest.raises(ValueError, match="requires exactly STRICT"):
        build_artifacts((invalid,), {"trace_sha256": "a" * 64})


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


def test_worker_wire_round_trip_preserves_record() -> None:
    original = record(SizingTopology.LOCAL_ONLY, 2, completion=1)
    assert sizing_runner._record_from_wire(
        sizing_runner._record_to_wire(original)
    ) == original


def test_deadline_frontier_recomputes_fairness_from_logical_ledger() -> None:
    original = record(SizingTopology.LOCAL_ONLY, 1, completion=1)
    strict_elapsed = (1000.0,) * 8 + (3000.0,) * 2
    ledger = tuple(
        [("tenant-a", "STRICT", 10, 1000.0)] * 8
        + [("tenant-b", "STRICT", 10, 3000.0)] * 2
        + [("tenant-a", "STANDARD", 10, 1000.0)]
        + [("tenant-b", "RELAXED", 10, 1000.0)]
    )
    changed = replace(
        original,
        per_tier_completion_elapsed_work={
            "STRICT": strict_elapsed,
            "STANDARD": (1000.0,),
            "RELAXED": (1000.0,),
        },
        completion_ledger=ledger,
    )
    rows = sizing_runner._deadline_frontier_rows((changed,))
    local_half = next(
        row
        for row in rows
        if row["deadline_multiplier"] == 0.5
        and row["topology"] == SizingTopology.LOCAL_ONLY.value
    )
    assert local_half["minimum_feasible_p"] == "GRID_EXHAUSTED"


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
    assert contract["schema_version"] == "m12-sizing-v2.3"
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
    deadlines = list(
        csv.DictReader(io.StringIO(first["deadline-frontier.csv"].decode()))
    )
    assert len(deadlines) == 6 * 3
    assert {float(row["deadline_multiplier"]) for row in deadlines} == {
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        2.0,
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
