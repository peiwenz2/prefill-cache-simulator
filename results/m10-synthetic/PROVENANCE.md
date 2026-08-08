# M10 synthetic replay-harness provenance

## Evidence tier

| Label | Value |
| --- | --- |
| `evidence_tier` | `SYNTHETIC_REPLAY` |
| `calibration_status` | `SYNTHETIC_UNCALIBRATED` |
| `time_unit` | `NORMALIZED_WORK` |
| `hardware_validation` | `BLOCKED_NO_ENGINE_ACCESS` |
| `schema_version` | `m10-replay-v1` |

**No engine, GPU, or network was involved.** Every number in this directory is
produced by the local simulator in this repository. The trace is real and its
SHA-256 is recorded; the *outcomes* replayed over it are modeled.
`MachineProvenance` is all-null by construction, so no artifact here may be
quoted as a millisecond cost, a calibrated model, or a measurement.

`scripts/run_m10_synthetic.py` builds every artifact's bytes in memory and scans
each one for `MILLISECONDS`, `HW_CALIBRATED`, and `HW_VALIDATED` **before** any
of them reaches this directory. A run that would have overclaimed therefore
writes no file at all, rather than writing one that then has to be retracted.
Five further fail-closed checks run before the bytes are even built: machine
provenance must be all-null, the three labels above must hold, and
`ENFORCEMENT_ENABLED` must be false. These raise `RuntimeError` rather than using
`assert`, which `python -O` strips. The label discipline is enforced by the
runner, not merely asserted in this document.

### The `_ms` suffix is not milliseconds

`ExperimentResult` is an inherited schema and several of its field names carry an
`_ms` suffix — `view_delay_ms`, `prefill_uncached_token_ms`, `inflight_wait_ms`,
`queue_wait_normalized_ms_p50/p95/p99/max`. **These are normalized work units.**
Renaming the schema was out of scope for M10, so every artifact instead carries
an explicit column:

```
unit_note = FIELDS_SUFFIXED_MS_ARE_NORMALIZED_WORK_NOT_WALL_CLOCK
```

Do not read any `_ms` value as wall-clock time.

## Reproduction

```
.venv/bin/python scripts/run_m10_synthetic.py
```

Deterministic: the plan seed is `20261010`, the fault fixture seed is `20261010`,
and randomness flows through the repository's `NamedSeedManager`
(`named-sha256-v1`). Re-running overwrites this directory with identical bytes
apart from the git provenance fields.

### What the git SHA cannot tell you

`replay.json` records `git_sha` and `git_dirty`, but at the time of this run the
generator and the `replay` package are **untracked**, so `git_sha` names a commit
that does not contain the code that produced these numbers. `git_dirty` flags
that, and no more. `replay.json` therefore also carries a
`source_fingerprints` block:

| Field | Value |
| --- | --- |
| `algorithm` | `sha256` |
| `reproducibility_claim` | `SOURCE_AND_RUNTIME_IDENTIFIED_REPRODUCTION_CONDITIONAL` |
| `combined_digest` | `7b9c1d03ed47b7667330a22cdaeb38701a384b6417d61e700ff9269a141c9e1d` |
| `file_count` | 40 |
| `runtime` | CPython 3.14.2 |

The 40 files are the generator plus every `.py` under `src/prefill_cache_sim`,
discovered by walking the tree rather than read from a hand-maintained list, so a
module added later cannot silently escape the fingerprint. `combined_digest`
hashes the path *and* the content of each file, so renaming a file changes it
even when no line of code does.

The claim label is deliberately not "reproducible". These digests **identify**
the source and the interpreter; they do not pin the dependency versions, the OS,
or the trace file, and they cannot by themselves reproduce anything. They let a
reader determine whether the code in front of them is the code that ran.

## Inputs

| Input | Value |
| --- | --- |
| Trace | `mooncake_trace.jsonl` |
| Trace SHA-256 | `b434f1816a707f4bac697235588184ebc374c9907cb981bb65fb0643471fe711` |
| Trace requests | 23,608 (full trace, no subsampling) |
| Block size | 512 tokens |
| Nodes | 4 P+D, `FIXED_PER_NODE`, 256 blocks each |
| Arrival scales | 1x, 2x (`replay_speed`) |
| Git SHA / dirty | recorded per-run in `replay.json` |

## Arms

| `arm_id` | `arm_role` | Meaning |
| --- | --- | --- |
| `S0_RANDOM` | `BASELINE` | Comparison baseline |
| `S3_GB_PREFIX_BUCKET` | `CANDIDATE` | Prefix-bucket candidate |
| `S5_CENTRALIZED_MASTER_TTFT` | `STOP_GATED` | Carries the M6-M8 stop gate; replayed, never promoted |
| `S4_SESSION_AFFINITY` | `M4_WINNER` | The arm M4 selected |

## Three-source reconciliation

Three observers are reconciled per attempt on the join key
`(logical_request_id, attempt_index)`, which survives retries:

| Source | Observes |
| --- | --- |
| `ENGINE_HIT` | `node_id`, `input_tokens`, `hit_tokens` |
| `CLIENT_LATENCY` | `input_tokens`, `output_tokens`, `ttft_work`, `tpot_work` |
| `ATTEMPT_TRACE` | `node_id`, `input_tokens`, `arrival_work`, `start_work`, `finish_work` |

Only fields more than one source can actually see are cross-checked
(`input_tokens` across all three, `node_id` across engine and trace). A field
only one observer can see is deliberately excluded: disagreement about it is not
observable, and checking it would manufacture evidence.

Ledger precedence is `DUPLICATE > MISSING > DISAGREEMENT`, one defect kind per
attempt.

### `tpot_work` is null, and says why

The simulator models prefill placement and has no decode loop, so a
per-output-token cost cannot be derived from it. `tpot_work` is `None` for every
modeled attempt, with `tpot_reason = SIMULATOR_DOES_NOT_MODEL_DECODE` recorded
per cell in `reconciliation.csv`. It is not imputed, defaulted, or filled with a
plausible-looking number.

### The modeled ledger is empty — and why that alone proves nothing

`ledger.csv` has zero rows: all 8 cells reconcile 23,608/23,608 attempts with
`disagreement_fraction = 0`. This is expected and is **not** evidence of
agreement between independent systems. All three modeled sources are projections
of a single simulator decision log, so they cannot disagree. An empty ledger is
only meaningful if the reconciler can be shown to produce a non-empty one.

`fault_injection.csv` supplies that control. A 64-attempt synthetic fixture is
damaged in three distinct ways and reconciled:

| Fault | Target | Source |
| --- | --- | --- |
| drop | `synthetic:20261010:000000000003#0` | `ENGINE_HIT` |
| duplicate | `synthetic:20261010:000000000005#0` | `CLIENT_LATENCY` |
| perturb `node_id` | `synthetic:20261010:000000000006#1` | `ATTEMPT_TRACE` |

The perturbed attempt is a **retry** (`attempt_index = 1`), so recovering it also
exercises the join key against a repeated logical request id.

The expected ledger is derived by `apply_faults` from the plan and the mutated
records — never by calling `reconcile` — so agreement between the two columns is
a check on the reconciler rather than a restatement of it. The runner raises if
they differ. Result: 3 expected entries, 3 recovered, exact match.

## Frozen ranking statistic

`FROZEN_RANKING_STATISTIC = KENDALL_TAU_B`, frozen in `replay/ranking.py` before
any result in this directory was produced. Tie handling is part of the frozen
definition, and `pairwise_winner_agreement` is reported alongside it. When the
statistic is undefined (fewer than two arms, or every score tied) the comparison
is emitted as `available = False` rather than coerced to a number.

Computed from modeled scores: `tau_b = 1.0`, `pairwise_agreement = 1.0`, 6
concordant / 0 discordant pairs, on all four score metrics. Nothing here was
measured; the statistic ranks simulator output. **See the limitation below
before reading anything into that.**

## Shadow decisions: recorded, never enforced

`ENFORCEMENT_ENABLED = False`, and `shadow_decisions.csv` carries
`enforced = False` on every row. Attempting to enforce raises
`ShadowEnforcementError`. Nothing in M10 gates, promotes, or demotes an arm.

Recorded outcomes (identical across all four metrics): `S3_GB_PREFIX_BUCKET` and
`S4_SESSION_AFFINITY` are `SHADOW_RECOMMENDED`; `S5_CENTRALIZED_MASTER_TTFT` is
`SHADOW_WITHHELD` with reason `CANDIDATE_DOES_NOT_BEAT_BASELINE`. That S5 lands
withheld is consistent with its M6-M8 stop gate but is **not** an independent
confirmation of it — same simulator, same trace.

## Artifact integrity

A directory of result files is only trustworthy if a reader can tell whether all
of it came from one run. Two mechanisms make that checkable.

**`MANIFEST.json`** records the SHA-256 of every artifact this run generated. It
is **written last**, after all the files it describes. A reader who recomputes
the digests and finds a mismatch is therefore looking at a partially replaced
set, not at one run's output, and knows to discard it rather than reconcile it.
It does not cover itself, and it does not cover this file: `PROVENANCE.md` is
hand-written and is not produced by the run.

**Staged writes.** Every artifact's bytes are built in memory, written into a
sibling staging directory, and only then moved into place with `os.replace`. The
staging directory is a sibling so the renames stay on one filesystem, where they
are atomic. A crash or a label violation mid-run leaves the previous artifacts
untouched instead of leaving a half-written file behind.

Renaming the whole directory in one move would be atomic for the set as well, and
is deliberately *not* done: it would delete files this script does not
generate — `PROVENANCE.md` is hand-written and lives here. **Honest limitation:**
per-file renames leave a window in which a concurrent reader could see files from
two runs. `MANIFEST.json` being written last is what makes that window
*detectable*; it does not close it.

## Files

| File | Contents |
| --- | --- |
| `MANIFEST.json` | SHA-256 of every other artifact in this directory; written last |
| `replay.json` | Full `m10-replay-v1` outcome, ranking comparisons, shadow decisions, fault-injection summary, provenance. Per-cell reconciliation is serialized via `ReconciliationReport.summary()`: the ledger in full, the per-attempt row table dropped (it is bulk, not evidence, and is already published as CSV). |
| `results.csv` | One row per (arm x arrival scale) cell: complete `ExperimentResult` plus honesty labels |
| `reconciliation.csv` | Per-cell join health, ledger size, truth basis, TPOT reason |
| `ledger.csv` | Modeled disagreement ledger (header only; empty by construction) |
| `fault_injection.csv` | Expected vs recovered ledger for the known-defect fixture |
| `ranking.csv` | Frozen statistic, 1x-vs-2x comparison per metric |
| `shadow_decisions.csv` | Recorded, unenforced baseline-vs-candidate outcomes |

## Limitations that bound interpretation

- **`tau_b = 1.0` is weak evidence, not a strong result.** It compares arm
  rankings at 1x against 2x arrival. The 2x condition does raise load —
  `queue_wait_normalized_ms_p95` rises roughly 6% across every arm — but cache
  hit rate is nearly insensitive to it. `S0_RANDOM` scores *bit-identically* at
  both scales (0.34403309236012575), and the other three move in the fourth
  decimal place or beyond. The ranking is therefore stable across two conditions
  the arms barely distinguish. This exercises the statistic's plumbing; it does
  not demonstrate ranking robustness under meaningful perturbation.
- **This is not simulator-vs-real ranking consistency.** DESIGN-v3 M10 asks
  whether simulated rankings survive contact with a real engine. Both sides of
  every comparison here are simulated. The real-vs-simulated arm of M10 is
  unstarted and blocked.
- **Reconciliation is untested against genuinely independent observers.** It is
  tested against injected faults in a synthetic fixture (which it recovers
  exactly) and run against three projections of one decision log (which cannot
  disagree). Neither exercises real clock skew, real retry ambiguity, or real
  partial log loss.
- **TPOT is absent, not zero.** Any downstream analysis needing decode cost must
  treat these cells as having no TPOT observation at all.
- **The 1x/2x axis is `replay_speed`**, i.e. compression of inter-arrival gaps
  from the trace. It is not a load-generator model and does not reshape the
  request mix.

## Blocked

**Hardware validation is blocked.** Promoting anything here to `HW_VALIDATED` /
`HW_CALIBRATED` / `MILLISECONDS` requires an engine endpoint backed by a real
accelerator that can return complete `MachineProvenance` (`host_id`,
`accelerator_model`, `engine_version`, `captured_at_utc`). No such endpoint
exists in this repository, none is implied by these artifacts, and
`results/m10` (the tier that would hold measured results) deliberately does not
exist.
