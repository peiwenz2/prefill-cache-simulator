"""Stable request, block, and random-stream identities."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Final

_GENERATED_DOMAIN: Final = b"generated-sha256-v1"
_RANDOM_DOMAIN: Final = b"named-sha256-v1"
_U64_MAX: Final = 2**64 - 1


def _u64(value: int, field: str) -> bytes:
    valid = isinstance(value, int) and not isinstance(value, bool)
    if not valid or not 0 <= value <= _U64_MAX:
        raise ValueError(f"{field} must fit unsigned 64-bit")
    return struct.pack(">Q", value)


def _length_prefixed(value: bytes) -> bytes:
    return _u64(len(value), "byte length") + value


def _raw_sha256(value: str, field: str) -> bytes:
    if len(value) != 64 or value.lower() != value:
        raise ValueError(f"{field} must be 64 lowercase hexadecimal characters")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be hexadecimal") from exc
    if len(raw) != 32:
        raise ValueError(f"{field} must decode to 32 bytes")
    return raw


@dataclass(frozen=True, slots=True)
class TraceBlockId:
    """A block identity supplied by the input trace."""

    value: int

    def __post_init__(self) -> None:
        _u64(self.value, "trace block ID")


@dataclass(frozen=True, slots=True)
class GeneratedBlockId:
    """A generated block identity rendered as lowercase SHA-256 hex."""

    value: str

    def __post_init__(self) -> None:
        _raw_sha256(self.value, "generated block ID")


BlockId = TraceBlockId | GeneratedBlockId


def logical_request_id(trace_sha256: str, line_index: int) -> str:
    """Return the normative trace-sha256-line-index-v1 display identity."""
    _raw_sha256(trace_sha256, "trace SHA-256")
    _u64(line_index, "trace line index")
    return f"trace:{trace_sha256}:{line_index:020d}"


def generated_block_message(
    *,
    trace_sha256: str,
    block_size_tokens: int,
    trace_line_index: int,
    parent: BlockId | None,
    generated_index: int,
) -> bytes:
    """Encode the normative generated-sha256-v1 message."""
    message = _length_prefixed(_GENERATED_DOMAIN)
    message += _raw_sha256(trace_sha256, "trace SHA-256")
    message += _u64(block_size_tokens, "block size tokens")
    message += _u64(trace_line_index, "trace line index")
    if isinstance(parent, TraceBlockId):
        message += b"\x00" + _u64(parent.value, "trace block ID")
    elif isinstance(parent, GeneratedBlockId):
        message += b"\x01" + _raw_sha256(parent.value, "parent generated ID")
    elif parent is None:
        message += b"\x02"
    else:  # pragma: no cover - type checkers prevent this path
        raise TypeError("parent must be TraceBlockId, GeneratedBlockId, or None")
    return message + _u64(generated_index, "generated index")


def generated_block_id(
    *,
    trace_sha256: str,
    block_size_tokens: int,
    trace_line_index: int,
    parent: BlockId | None,
    generated_index: int,
) -> GeneratedBlockId:
    """Hash a generated-sha256-v1 message into its typed identity."""
    digest = hashlib.sha256(
        generated_block_message(
            trace_sha256=trace_sha256,
            block_size_tokens=block_size_tokens,
            trace_line_index=trace_line_index,
            parent=parent,
            generated_index=generated_index,
        )
    ).hexdigest()
    return GeneratedBlockId(digest)


class DeterministicStream:
    """Counter-mode SHA-256 stream with unbiased bounded integer sampling."""

    def __init__(self, key: bytes) -> None:
        if len(key) != 32:
            raise ValueError("stream key must be exactly 32 bytes")
        self._key = key
        self._counter = 0

    def _block(self) -> bytes:
        block = hashlib.sha256(self._key + _u64(self._counter, "counter")).digest()
        self._counter += 1
        return block

    def randbelow(self, upper: int) -> int:
        """Return a uniform integer in ``range(upper)``."""
        if not isinstance(upper, int) or isinstance(upper, bool) or upper <= 0:
            raise ValueError("upper must be a positive integer")
        width = max(1, (upper.bit_length() + 7) // 8)
        space = 1 << (width * 8)
        limit = space - (space % upper)
        while True:
            value = int.from_bytes(self._bytes(width), "big")
            if value < limit:
                return value % upper

    def _bytes(self, size: int) -> bytes:
        output = bytearray()
        while len(output) < size:
            output.extend(self._block())
        return bytes(output[:size])


class NamedSeedManager:
    """Derive independent deterministic streams from one root seed."""

    def __init__(self, root_seed: int) -> None:
        self._root_seed = root_seed
        _u64(root_seed, "root seed")
        self._streams: dict[str, DeterministicStream] = {}

    def stream_key(self, consumer_name: str) -> bytes:
        """Return the frozen named-sha256-v1 key for a consumer."""
        if not consumer_name or consumer_name.strip() != consumer_name:
            raise ValueError("consumer name must be non-empty without edge whitespace")
        name = consumer_name.encode("utf-8")
        return hashlib.sha256(
            _length_prefixed(_RANDOM_DOMAIN)
            + _u64(self._root_seed, "root seed")
            + _length_prefixed(name)
        ).digest()

    def stream(self, consumer_name: str) -> DeterministicStream:
        if consumer_name not in self._streams:
            self._streams[consumer_name] = DeterministicStream(
                self.stream_key(consumer_name)
            )
        return self._streams[consumer_name]
