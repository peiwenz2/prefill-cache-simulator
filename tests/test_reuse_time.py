from __future__ import annotations

import json
from pathlib import Path

import pytest

from prefill_cache_sim.reuse_time import _CausalHotTracker, analyze_reuse_time
from prefill_cache_sim.trace import load_trace


def _trace(tmp_path: Path, records: list[dict[str, object]]):
    path = tmp_path / "trace.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return load_trace(path, block_size_tokens=512)


def test_reuse_windows_include_same_timestamp_and_boundary(tmp_path: Path) -> None:
    trace = _trace(
        tmp_path,
        [
            {"timestamp": 0, "input_length": 512, "output_length": 1, "hash_ids": [1]},
            {"timestamp": 0, "input_length": 512, "output_length": 1, "hash_ids": [1]},
            {
                "timestamp": 250,
                "input_length": 512,
                "output_length": 1,
                "hash_ids": [1],
            },
            {
                "timestamp": 251,
                "input_length": 512,
                "output_length": 1,
                "hash_ids": [1],
            },
        ],
    )
    analysis = analyze_reuse_time(trace)
    block = analysis.rows[0]
    assert block.reused_events == 3
    assert block.within_window_weight["10"] == 2
    assert block.within_window_weight["250"] == 3
    assert block.within_window_ratio["250"] == 1.0


def test_partial_block_uses_token_weight(tmp_path: Path) -> None:
    trace = _trace(
        tmp_path,
        [
            {
                "timestamp": 0,
                "input_length": 600,
                "output_length": 1,
                "hash_ids": [1, 2],
            },
            {
                "timestamp": 100,
                "input_length": 600,
                "output_length": 1,
                "hash_ids": [1, 2],
            },
        ],
    )
    analysis = analyze_reuse_time(trace)
    token = analysis.rows[1]
    assert token.denominator_weight == 600
    assert token.within_window_weight["100"] == 600


def test_single_reference_blocks_do_not_enter_denominator(tmp_path: Path) -> None:
    trace = _trace(
        tmp_path,
        [
            {"timestamp": 0, "input_length": 512, "output_length": 1, "hash_ids": [1]},
            {"timestamp": 1, "input_length": 512, "output_length": 1, "hash_ids": [2]},
        ],
    )
    analysis = analyze_reuse_time(trace)
    assert analysis.rows[0].denominator_weight == 0
    assert analysis.gate_ratio == 0.0
    assert analysis.gate_verdict == "KILL_ROUTER_HOLD"


def test_family_anchor_depth_is_bounded_by_request_length(tmp_path: Path) -> None:
    trace = _trace(
        tmp_path,
        [
            {
                "timestamp": 0,
                "input_length": 1024,
                "output_length": 1,
                "hash_ids": [1, 2],
            },
            {
                "timestamp": 50,
                "input_length": 1024,
                "output_length": 1,
                "hash_ids": [1, 2],
            },
        ],
    )
    family = analyze_reuse_time(trace).rows[2]
    assert family.reused_events == 1
    assert family.within_window_ratio["50"] == pytest.approx(1.0)


def test_gate_excludes_only_references_hot_before_current_observation(
    tmp_path: Path,
) -> None:
    records = [
        {
            "timestamp": timestamp,
            "input_length": 512,
            "output_length": 1,
            "hash_ids": [1],
        }
        for timestamp in range(10)
    ]
    analysis = analyze_reuse_time(_trace(tmp_path, records))
    gate = analysis.rows[3]
    # Counts 1..7 before observation are admitted. Once the historical count
    # reaches the frozen floor of 8, later references are excluded.
    assert gate.reused_events == 7
    assert gate.denominator_weight == 7 * 512


def test_hot_tracker_refresh_can_raise_threshold_above_floor() -> None:
    tracker = _CausalHotTracker()
    for _ in range(12):
        tracker.observe(7)
    tracker.refresh()
    assert tracker.threshold == 12


def test_kill_marks_build_side_decomposition_not_applicable(tmp_path: Path) -> None:
    trace = _trace(
        tmp_path,
        [
            {
                "timestamp": 0,
                "input_length": 512,
                "output_length": 1,
                "hash_ids": [1],
            },
            {
                "timestamp": 10_000,
                "input_length": 512,
                "output_length": 1,
                "hash_ids": [1],
            },
        ],
    )
    analysis = analyze_reuse_time(trace)
    assert analysis.gate_verdict == "KILL_ROUTER_HOLD"
    assert analysis.decomposition_status == "NOT_APPLICABLE_AFTER_HARD_KILL"
    assert analysis.distinct_timestamps == 2
    assert analysis.minimum_positive_timestamp_gap_ms == 10_000
    assert analysis.maximum_positive_timestamp_gap_ms == 10_000
