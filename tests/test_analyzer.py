from __future__ import annotations

import json
from pathlib import Path

import pytest

from prefill_cache_sim.analyzer import analyze_trace
from prefill_cache_sim.trace import load_trace

ROOT = Path(__file__).resolve().parents[1]


def test_small_trace_continuous_prefix_ceiling(tmp_path: Path) -> None:
    records = [
        {"timestamp": 0, "input_length": 600, "output_length": 2, "hash_ids": [1, 2]},
        {"timestamp": 1, "input_length": 600, "output_length": 3, "hash_ids": [1, 3]},
        {"timestamp": 2, "input_length": 600, "output_length": 4, "hash_ids": [4, 2]},
    ]
    path = tmp_path / "trace.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    analysis = analyze_trace(load_trace(path, block_size_tokens=512))
    assert analysis.infinite_cache_prefix_hit_blocks == 1
    assert analysis.infinite_cache_prefix_hit_tokens == 512
    assert analysis.infinite_cache_prefix_hit_blocks <= analysis.reused_block_references
    assert analysis.block_ref_hit_ceiling == pytest.approx(1 / 6)
    assert analysis.token_weighted_hit_ceiling == pytest.approx(512 / 1800)


@pytest.mark.slow
def test_mooncake_trace_matches_independent_ceiling() -> None:
    path = ROOT / "mooncake_trace.jsonl"
    if not path.exists():
        pytest.skip("Mooncake trace fixture is not present")
    loaded = load_trace(path, block_size_tokens=512)
    assert loaded.sha256 == (
        "b434f1816a707f4bac697235588184ebc374c9907cb981bb65fb0643471fe711"
    )
    analysis = analyze_trace(loaded)
    assert analysis.request_count == 23_608
    assert analysis.block_references == 409_356
    assert analysis.unique_blocks == 183_166
    assert analysis.infinite_cache_prefix_hit_blocks == (
        analysis.block_references - analysis.unique_blocks
    )
    assert analysis.block_ref_hit_ceiling == pytest.approx(0.5525508359471951)
    assert analysis.token_weighted_hit_ceiling == pytest.approx(0.5707002329449369)
