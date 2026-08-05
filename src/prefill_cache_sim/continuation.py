"""Deterministic continuation-prefix construction."""

from __future__ import annotations

from dataclasses import dataclass

from .domain import BlockRef, Request
from .identity import GeneratedBlockId, TraceBlockId, generated_block_id


@dataclass(frozen=True, slots=True)
class ContinuationPrefixBuilder:
    trace_sha256: str
    block_size_tokens: int
    trace_line_index_by_logical_id: dict[str, int]

    def build(
        self,
        original: Request,
        *,
        sent_tokens: int,
        arrival_ms: int,
    ) -> Request:
        if sent_tokens <= 0:
            raise ValueError("sent_tokens must be positive")
        if arrival_ms < original.arrival_ms:
            raise ValueError("continuation arrival precedes original attempt")
        if any(
            size != self.block_size_tokens for size in original.block_token_sizes[:-1]
        ):
            raise ValueError("interior blocks must equal block_size_tokens")
        try:
            line_index = self.trace_line_index_by_logical_id[
                original.logical_request_id
            ]
        except KeyError as exc:
            raise ValueError("logical request has no trace line index") from exc
        full_count = original.input_tokens // self.block_size_tokens
        has_partial = original.input_tokens % self.block_size_tokens != 0
        retained_count = full_count if has_partial else len(original.prefix_blocks)
        retained = original.prefix_blocks[:retained_count]
        sizes = list(original.block_token_sizes[:retained_count])
        pending_tokens = original.input_tokens - sum(sizes) + sent_tokens
        if retained:
            retained_parent = retained[-1]
            if retained_parent.kind.value == "TRACE":
                parent: TraceBlockId | GeneratedBlockId | None = TraceBlockId(
                    int(retained_parent.value)
                )
            else:
                parent = GeneratedBlockId(str(retained_parent.value))
        else:
            parent = None
        generated: list[BlockRef] = []
        generated_index = 0
        while pending_tokens > 0:
            identity = generated_block_id(
                trace_sha256=self.trace_sha256,
                block_size_tokens=self.block_size_tokens,
                trace_line_index=line_index,
                parent=parent,
                generated_index=generated_index,
            )
            generated.append(BlockRef.generated(identity.value))
            size = min(self.block_size_tokens, pending_tokens)
            sizes.append(size)
            pending_tokens -= size
            parent = identity
            generated_index += 1
        next_attempt = original.attempt_index + 1
        return Request(
            request_id=f"{original.logical_request_id}:attempt:{next_attempt}",
            logical_request_id=original.logical_request_id,
            attempt_index=next_attempt,
            arrival_ms=arrival_ms,
            input_tokens=original.input_tokens + sent_tokens,
            output_tokens=max(0, original.output_tokens - sent_tokens),
            prefix_blocks=retained + tuple(generated),
            block_token_sizes=tuple(sizes),
            affinity_key=original.affinity_key,
        )
