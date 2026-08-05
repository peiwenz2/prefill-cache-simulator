"""Mooncake JSONL trace loading and validation."""

from __future__ import annotations

import hashlib
import json
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .identity import logical_request_id


class TraceValidationError(ValueError):
    """Raised when a trace record violates the Mooncake contract."""


@dataclass(frozen=True, slots=True)
class TraceRecord:
    line_index: int
    request_id: str
    timestamp_ms: int
    input_tokens: int
    output_tokens: int
    prefix_blocks: array[int]
    block_token_sizes: array[int]


@dataclass(frozen=True, slots=True)
class LoadedTrace:
    path: Path
    sha256: str
    block_size_tokens: int
    records: tuple[TraceRecord, ...]


def _plain_int(
    value: Any,
    field: str,
    line_number: int,
    *,
    minimum: int,
    maximum: int = 2**64 - 1,
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not minimum <= value <= maximum
    ):
        raise TraceValidationError(
            f"line {line_number}: {field} must be an integer in [{minimum}, {maximum}]"
        )
    return value


def load_trace(path: Path, *, block_size_tokens: int) -> LoadedTrace:
    """Load a validated Mooncake JSONL trace into a compact immutable structure."""
    if block_size_tokens <= 0:
        raise ValueError("block_size_tokens must be positive")
    hasher = hashlib.sha256()
    raw_records: list[tuple[int, int, int, array[int], array[int]]] = []
    previous_timestamp = -1
    with path.open("rb") as handle:
        for line_index, raw_line in enumerate(handle):
            hasher.update(raw_line)
            line_number = line_index + 1
            if not raw_line.strip():
                message = f"line {line_number}: blank lines are not allowed"
                raise TraceValidationError(message)
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                message = f"line {line_number}: invalid JSON: {exc.msg}"
                raise TraceValidationError(message) from exc
            if not isinstance(value, dict):
                raise TraceValidationError(
                    f"line {line_number}: record must be an object"
                )
            required = {"timestamp", "input_length", "output_length", "hash_ids"}
            if set(value) != required:
                raise TraceValidationError(
                    f"line {line_number}: fields must be exactly {sorted(required)}"
                )
            timestamp = _plain_int(
                value["timestamp"], "timestamp", line_number, minimum=0
            )
            if timestamp < previous_timestamp:
                raise TraceValidationError(f"line {line_number}: timestamp decreased")
            previous_timestamp = timestamp
            input_tokens = _plain_int(
                value["input_length"], "input_length", line_number, minimum=1
            )
            output_tokens = _plain_int(
                value["output_length"], "output_length", line_number, minimum=0
            )
            raw_ids = value["hash_ids"]
            if not isinstance(raw_ids, list) or not raw_ids:
                message = f"line {line_number}: hash_ids must be non-empty"
                raise TraceValidationError(message)
            expected_blocks = (
                input_tokens + block_size_tokens - 1
            ) // block_size_tokens
            if len(raw_ids) != expected_blocks:
                raise TraceValidationError(
                    f"line {line_number}: hash_ids length {len(raw_ids)} != "
                    f"ceil(input_length / block_size_tokens) {expected_blocks}"
                )
            blocks = array(
                "Q",
                (
                    _plain_int(item, "hash_ids[]", line_number, minimum=0)
                    for item in raw_ids
                ),
            )
            tail = input_tokens - block_size_tokens * (len(blocks) - 1)
            sizes = array("Q", [block_size_tokens] * (len(blocks) - 1) + [tail])
            raw_records.append((timestamp, input_tokens, output_tokens, blocks, sizes))
    if not raw_records:
        raise TraceValidationError("trace is empty")
    trace_sha256 = hasher.hexdigest()
    records = tuple(
        TraceRecord(
            line_index=line_index,
            request_id=logical_request_id(trace_sha256, line_index),
            timestamp_ms=timestamp,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            prefix_blocks=blocks,
            block_token_sizes=sizes,
        )
        for line_index, (
            timestamp,
            input_tokens,
            output_tokens,
            blocks,
            sizes,
        ) in enumerate(raw_records)
    )
    return LoadedTrace(path, trace_sha256, block_size_tokens, records)
