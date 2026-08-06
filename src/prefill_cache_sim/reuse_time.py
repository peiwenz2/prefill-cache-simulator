"""Causal reuse-time analysis for the M12 Goodput x Reuse gate."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from itertools import pairwise

from .trace import LoadedTrace

WINDOWS_MS: tuple[int, ...] = (10, 50, 100, 250, 1_000, 5_000, 60_000, 300_000)
GATE_WINDOW_MS = 250
GATE_THRESHOLD = 0.10
HOT_PERCENTILE = 99.0
HOT_MIN_REFERENCES = 8
HOT_REFRESH_REQUESTS = 256
FAMILY_ANCHOR_DEPTH = 4


@dataclass(frozen=True, slots=True)
class ReuseTimeRow:
    caliber: str
    denominator_weight: int
    reused_events: int
    within_window_weight: dict[str, int]
    within_window_ratio: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReuseTimeAnalysis:
    schema_version: str
    windows_ms: tuple[int, ...]
    hot_definition: str
    family_anchor_depth: int
    rows: tuple[ReuseTimeRow, ...]
    gate_caliber: str
    gate_window_ms: int
    gate_threshold: float
    gate_ratio: float
    gate_verdict: str
    decomposition_status: str
    distinct_timestamps: int
    minimum_positive_timestamp_gap_ms: int | None
    maximum_positive_timestamp_gap_ms: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class _CausalHotTracker:
    """Match S4's history-only p99 rule without reading future counts."""

    def __init__(self) -> None:
        self.counts: Counter[int] = Counter()
        self.histogram: Counter[int] = Counter()
        self.threshold = HOT_MIN_REFERENCES

    def is_hot(self, block: int) -> bool:
        return self.counts[block] >= self.threshold

    def observe(self, block: int) -> None:
        previous = self.counts[block]
        if previous:
            self.histogram[previous] -= 1
            if self.histogram[previous] == 0:
                del self.histogram[previous]
        current = previous + 1
        self.counts[block] = current
        self.histogram[current] += 1

    def refresh(self) -> None:
        population = len(self.counts)
        if not population:
            return
        target = int((HOT_PERCENTILE / 100) * (population - 1))
        cumulative = 0
        for frequency in sorted(self.histogram):
            cumulative += self.histogram[frequency]
            if cumulative > target:
                self.threshold = max(HOT_MIN_REFERENCES, frequency)
                return


class _Accumulator:
    def __init__(self, caliber: str) -> None:
        self.caliber = caliber
        self.denominator = 0
        self.events = 0
        self.within = {window: 0 for window in WINDOWS_MS}

    def add(self, delta_ms: int, weight: int) -> None:
        self.denominator += weight
        self.events += 1
        for window in WINDOWS_MS:
            if delta_ms <= window:
                self.within[window] += weight

    def finish(self) -> ReuseTimeRow:
        weights = {str(window): self.within[window] for window in WINDOWS_MS}
        ratios = {
            str(window): (
                self.within[window] / self.denominator if self.denominator else 0.0
            )
            for window in WINDOWS_MS
        }
        return ReuseTimeRow(
            caliber=self.caliber,
            denominator_weight=self.denominator,
            reused_events=self.events,
            within_window_weight=weights,
            within_window_ratio=ratios,
        )


def analyze_reuse_time(trace: LoadedTrace) -> ReuseTimeAnalysis:
    """Measure history-only inter-reference deltas without future leakage."""
    block_ref = _Accumulator("block_reference")
    token_weighted = _Accumulator("token_weighted")
    family_root = _Accumulator("prefix_family_depth4_reference")
    gate = _Accumulator("token_weighted_causal_hot_excluded")
    last_block_ms: dict[int, int] = {}
    last_family_ms: dict[int, int] = {}
    hot = _CausalHotTracker()
    distinct_timestamps: list[int] = []
    previous_timestamp: int | None = None

    for request_number, record in enumerate(trace.records, start=1):
        if record.timestamp_ms != previous_timestamp:
            distinct_timestamps.append(record.timestamp_ms)
            previous_timestamp = record.timestamp_ms
        anchor_index = min(FAMILY_ANCHOR_DEPTH, len(record.prefix_blocks)) - 1
        anchor = record.prefix_blocks[anchor_index]
        previous_family = last_family_ms.get(anchor)
        if previous_family is not None:
            family_root.add(record.timestamp_ms - previous_family, 1)
        last_family_ms[anchor] = record.timestamp_ms

        for block, tokens in zip(
            record.prefix_blocks, record.block_token_sizes, strict=True
        ):
            previous = last_block_ms.get(block)
            is_hot_before_observe = hot.is_hot(block)
            if previous is not None:
                delta = record.timestamp_ms - previous
                block_ref.add(delta, 1)
                token_weighted.add(delta, tokens)
                if not is_hot_before_observe:
                    gate.add(delta, tokens)
            last_block_ms[block] = record.timestamp_ms
            hot.observe(block)
        if request_number % HOT_REFRESH_REQUESTS == 0:
            hot.refresh()

    rows = (
        block_ref.finish(),
        token_weighted.finish(),
        family_root.finish(),
        gate.finish(),
    )
    gate_row = rows[-1]
    gate_ratio = gate_row.within_window_ratio[str(GATE_WINDOW_MS)]
    verdict = "KILL_ROUTER_HOLD" if gate_ratio < GATE_THRESHOLD else "CONTINUE_TO_M12_1"
    timestamp_gaps = [
        current - previous for previous, current in pairwise(distinct_timestamps)
    ]
    return ReuseTimeAnalysis(
        schema_version="m12-reuse-time-v1",
        windows_ms=WINDOWS_MS,
        hot_definition=(
            "history-only counts; refresh every 256 requests; p99 frequency; "
            "minimum threshold 8"
        ),
        family_anchor_depth=FAMILY_ANCHOR_DEPTH,
        rows=rows,
        gate_caliber=gate_row.caliber,
        gate_window_ms=GATE_WINDOW_MS,
        gate_threshold=GATE_THRESHOLD,
        gate_ratio=gate_ratio,
        gate_verdict=verdict,
        decomposition_status=(
            "NOT_APPLICABLE_AFTER_HARD_KILL"
            if verdict == "KILL_ROUTER_HOLD"
            else "PENDING_M12_1_SYNTHETIC_SERVICE_REGIMES"
        ),
        distinct_timestamps=len(distinct_timestamps),
        minimum_positive_timestamp_gap_ms=min(timestamp_gaps, default=None),
        maximum_positive_timestamp_gap_ms=max(timestamp_gaps, default=None),
    )
