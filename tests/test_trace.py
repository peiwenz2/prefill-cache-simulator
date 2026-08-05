from __future__ import annotations

import json
from pathlib import Path

import pytest

from prefill_cache_sim.trace import TraceValidationError, load_trace


def write_trace(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(item) + "\n" for item in records))
    return path


def record(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "timestamp": 0,
        "input_length": 513,
        "output_length": 2,
        "hash_ids": [10, 11],
    }
    value.update(overrides)
    return value


def test_partial_block_sizes_and_duplicate_timestamp(tmp_path: Path) -> None:
    path = write_trace(tmp_path / "trace.jsonl", [record(), record(hash_ids=[10, 12])])
    loaded = load_trace(path, block_size_tokens=512)
    assert list(loaded.records[0].block_token_sizes) == [512, 1]
    assert loaded.records[0].timestamp_ms == loaded.records[1].timestamp_ms


def test_shorter_than_one_block(tmp_path: Path) -> None:
    path = write_trace(
        tmp_path / "trace.jsonl",
        [record(input_length=7, hash_ids=[99])],
    )
    loaded = load_trace(path, block_size_tokens=512)
    assert list(loaded.records[0].block_token_sizes) == [7]


@pytest.mark.parametrize(
    "payload, match",
    [
        (b"", "empty"),
        (b"not-json\n", "invalid JSON"),
        (
            (json.dumps(record(input_length=513, hash_ids=[1])) + "\n").encode(),
            "hash_ids length",
        ),
        (
            (
                json.dumps(record(timestamp=2))
                + "\n"
                + json.dumps(record(timestamp=1))
                + "\n"
            ).encode(),
            "timestamp decreased",
        ),
    ],
)
def test_invalid_trace_fails_closed(tmp_path: Path, payload: bytes, match: str) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_bytes(payload)
    with pytest.raises(TraceValidationError, match=match):
        load_trace(path, block_size_tokens=512)


def test_very_long_request_uses_compact_tuples(tmp_path: Path) -> None:
    block_count = 10_000
    path = write_trace(
        tmp_path / "trace.jsonl",
        [
            record(
                input_length=block_count * 512,
                hash_ids=list(range(block_count)),
            )
        ],
    )
    loaded = load_trace(path, block_size_tokens=512)
    assert len(loaded.records[0].prefix_blocks) == block_count
    assert loaded.records[0].prefix_blocks.typecode == "Q"
