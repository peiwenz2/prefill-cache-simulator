"""Deterministic relative and tiered SLO workload overlays."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .domain import Request
from .identity import NamedSeedManager


class Tier(StrEnum):
    STRICT = "STRICT"
    STANDARD = "STANDARD"
    RELAXED = "RELAXED"


@dataclass(frozen=True, slots=True)
class SloProfile:
    tenant_id: str
    tier: Tier
    ttft_deadline: float
    tpot_deadline: float
    completion_deadline: float


@dataclass(frozen=True, slots=True)
class RelativeSloOverlay:
    ttft_multiplier: float
    tpot_multiplier: float

    def __post_init__(self) -> None:
        if self.ttft_multiplier <= 0 or self.tpot_multiplier <= 0:
            raise ValueError("relative SLO multipliers must be positive")

    def profile(
        self, request: Request, isolated_prefill: float, isolated_tpot: float
    ) -> SloProfile:
        ttft = isolated_prefill * self.ttft_multiplier
        tpot = isolated_tpot * self.tpot_multiplier
        return SloProfile(
            "tenant-0",
            Tier.STANDARD,
            ttft,
            tpot,
            ttft + request.output_tokens * tpot,
        )


@dataclass(frozen=True, slots=True)
class TieredSloOverlay:
    seed: int
    tenant_count: int
    strict_fraction: float = 0.2
    relaxed_fraction: float = 0.2

    def __post_init__(self) -> None:
        if self.tenant_count <= 0:
            raise ValueError("tenant_count must be positive")
        if self.strict_fraction < 0 or self.relaxed_fraction < 0:
            raise ValueError("tier fractions must be non-negative")
        if self.strict_fraction + self.relaxed_fraction > 1:
            raise ValueError("tier fractions exceed one")

    def profile(
        self, request: Request, isolated_prefill: float, isolated_tpot: float
    ) -> SloProfile:
        seeds = NamedSeedManager(self.seed)
        tier_stream = seeds.stream(f"overlay/tier/{request.request_id}")
        tenant_stream = seeds.stream(f"overlay/tenant/{request.request_id}")
        draw = tier_stream.randbelow(1_000_000) / 1_000_000
        if draw < self.strict_fraction:
            tier, multiplier = Tier.STRICT, 1.5
        elif draw >= 1 - self.relaxed_fraction:
            tier, multiplier = Tier.RELAXED, 4.0
        else:
            tier, multiplier = Tier.STANDARD, 2.5
        tenant = tenant_stream.randbelow(self.tenant_count)
        ttft = isolated_prefill * multiplier
        tpot = isolated_tpot * multiplier
        return SloProfile(
            f"tenant-{tenant}",
            tier,
            ttft,
            tpot,
            ttft + request.output_tokens * tpot,
        )
