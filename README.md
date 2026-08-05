# Prefill Cache Simulator

Deterministic CPU simulator for Mooncake-style prefill KV-cache placement，eviction，
and replay experiments. Schema 1.x intentionally covers Phase A only.

## Local setup

```bash
./scripts/fetch_mooncake_trace.sh
uv run --extra test pytest -q
uv run prefill-cache-sim analyze \
  --scenario scenarios/baseline.local-lru.json
```

The trace is not committed. The fetch script verifies the normative SHA-256 before
moving it into place.

## M1 analyzer output

The analyzer reports request／block／reuse-distance／hotness／prefix-family statistics，
both continuous-prefix hit ceilings，and trace／config／git provenance. On the released
trace，the independent ceilings are：

- block-ref：`226190 / 409356 = 55.2550836%`；
- token-weighted：`115733271 / 202791701 = 57.0700233%`。

Mooncake Table 1 reports a cache-policy hit ratio but does not fully specify the
denominator and insertion timing needed to equate it with either simulator unit.
Therefore its `Inf = 0.51` is retained as a paper reference，not used as a direct
assertion for these two analyzer ceilings.
