from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from prefill_cache_sim.identity import (
    GeneratedBlockId,
    NamedSeedManager,
    TraceBlockId,
    generated_block_id,
    generated_block_message,
    logical_request_id,
)

ROOT = Path(__file__).resolve().parents[1]


def test_generated_identity_matches_all_golden_vectors() -> None:
    fixture = json.loads((ROOT / "tests/fixtures/identity-v1-golden.json").read_text())
    previous: GeneratedBlockId | None = None
    for vector in fixture["vectors"]:
        if vector["parent_kind"] == "TRACE_BLOCK":
            parent = TraceBlockId(vector["parent_trace_block_id"])
        elif vector["parent_kind"] == "GENERATED_BLOCK":
            parent = GeneratedBlockId(vector["parent_generated_digest_hex"])
        else:
            parent = None
        kwargs = {
            "trace_sha256": fixture["trace_sha256_hex"],
            "block_size_tokens": fixture["block_size_tokens"],
            "trace_line_index": vector["trace_line_index"],
            "parent": parent,
            "generated_index": vector["generated_index"],
        }
        assert generated_block_message(**kwargs).hex() == vector["message_hex"]
        actual = generated_block_id(**kwargs)
        assert actual.value == vector["digest_hex"]
        previous = actual
    assert previous is not None


def test_logical_request_id_is_normative() -> None:
    digest = "ab" * 32
    assert logical_request_id(digest, 42) == f"trace:{digest}:00000000000000000042"


@pytest.mark.parametrize("value", [-1, 2**64, True])
def test_typed_ids_reject_values_outside_u64(value: int) -> None:
    with pytest.raises(ValueError):
        TraceBlockId(value)


def test_named_streams_are_repeatable_and_independent() -> None:
    first_manager = NamedSeedManager(713)
    first_stream = first_manager.stream("selector/random")
    assert first_manager.stream("selector/random") is first_stream
    second_stream = NamedSeedManager(713).stream("selector/random")
    loss_stream = NamedSeedManager(713).stream("cluster-view/loss")
    first = [first_stream.randbelow(10_000) for _ in range(3)]
    second = [second_stream.randbelow(10_000) for _ in range(3)]
    loss = [loss_stream.randbelow(10_000) for _ in range(3)]
    assert first == second
    assert len(set(first)) > 1
    assert first != loss


def test_named_stream_matches_golden_vectors() -> None:
    fixture = json.loads(
        (ROOT / "tests/fixtures/randomness-v1-golden.json").read_text()
    )
    manager = NamedSeedManager(fixture["root_seed"])
    key = manager.stream_key(fixture["consumer_name"])
    assert key.hex() == fixture["stream_key_hex"]
    blocks = [
        hashlib.sha256(key + struct.pack(">Q", index)).hexdigest() for index in range(8)
    ]
    assert blocks == fixture["raw_block_hex"]
    for upper_text, expected in fixture["randbelow_sequences"].items():
        stream = NamedSeedManager(fixture["root_seed"]).stream(fixture["consumer_name"])
        actual = [stream.randbelow(int(upper_text)) for _ in expected]
        assert actual == expected


def test_randbelow_rejects_invalid_bound() -> None:
    with pytest.raises(ValueError):
        NamedSeedManager(0).stream("selector/random").randbelow(0)
