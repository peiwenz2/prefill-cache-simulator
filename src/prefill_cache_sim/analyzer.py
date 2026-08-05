"""Independent trace statistics and infinite-cache ceiling analysis."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

from .trace import LoadedTrace


@dataclass(frozen=True, slots=True)
class TraceAnalysis:
    request_count: int
    input_tokens: int
    output_tokens: int
    block_references: int
    unique_blocks: int
    reused_block_references: int
    partial_tail_requests: int
    duplicate_timestamp_adjacent_pairs: int
    prefix_family_count: int
    largest_prefix_family: int
    prefix_family_counts_by_depth: dict[str, int]
    largest_prefix_family_by_depth: dict[str, int]
    hottest_block_references: int
    hottest_block_share: float
    reuse_distance_samples: int
    reuse_distance_p50_requests: float | None
    reuse_distance_p95_requests: float | None
    infinite_cache_prefix_hit_blocks: int
    infinite_cache_prefix_hit_tokens: int
    block_ref_hit_ceiling: float
    token_weighted_hit_ceiling: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def analyze_trace(trace: LoadedTrace) -> TraceAnalysis:
    """Compute workload stats and a one-node infinite-cache ceiling."""
    seen: set[int] = set()
    frequencies: Counter[int] = Counter()
    prefix_families_by_depth: dict[int, Counter[tuple[int, ...]]] = {
        depth: Counter() for depth in range(1, 5)
    }
    last_request: dict[int, int] = {}
    reuse_distances: list[int] = []
    hit_blocks = 0
    hit_tokens = 0
    total_blocks = 0
    total_input_tokens = 0
    total_output_tokens = 0
    partial_tails = 0
    duplicate_timestamps = 0
    previous_timestamp: int | None = None

    for record in trace.records:
        block_values = record.prefix_blocks
        sizes = record.block_token_sizes
        prefix_hit = 0
        for block in block_values:
            if block not in seen:
                break
            prefix_hit += 1
        hit_blocks += prefix_hit
        hit_tokens += sum(sizes[:prefix_hit])
        total_blocks += len(block_values)
        total_input_tokens += record.input_tokens
        total_output_tokens += record.output_tokens
        partial_tails += sizes[-1] != trace.block_size_tokens
        duplicate_timestamps += previous_timestamp == record.timestamp_ms
        previous_timestamp = record.timestamp_ms
        for depth, families in prefix_families_by_depth.items():
            family = tuple(block_values[:depth])
            families[family] += 1
        for block in block_values:
            frequencies[block] += 1
            if block in last_request:
                reuse_distances.append(record.line_index - last_request[block])
            last_request[block] = record.line_index
        seen.update(block_values)

    hottest_count = frequencies.most_common(1)[0][1]
    depth_counts = {
        str(depth): len(families)
        for depth, families in prefix_families_by_depth.items()
    }
    depth_largest = {
        str(depth): max(families.values())
        for depth, families in prefix_families_by_depth.items()
    }
    return TraceAnalysis(
        request_count=len(trace.records),
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        block_references=total_blocks,
        unique_blocks=len(frequencies),
        reused_block_references=total_blocks - len(frequencies),
        partial_tail_requests=partial_tails,
        duplicate_timestamp_adjacent_pairs=duplicate_timestamps,
        prefix_family_count=depth_counts["1"],
        largest_prefix_family=depth_largest["1"],
        prefix_family_counts_by_depth=depth_counts,
        largest_prefix_family_by_depth=depth_largest,
        hottest_block_references=hottest_count,
        hottest_block_share=hottest_count / total_blocks,
        reuse_distance_samples=len(reuse_distances),
        reuse_distance_p50_requests=_percentile(reuse_distances, 0.5),
        reuse_distance_p95_requests=_percentile(reuse_distances, 0.95),
        infinite_cache_prefix_hit_blocks=hit_blocks,
        infinite_cache_prefix_hit_tokens=hit_tokens,
        block_ref_hit_ceiling=hit_blocks / total_blocks,
        token_weighted_hit_ceiling=hit_tokens / total_input_tokens,
    )
