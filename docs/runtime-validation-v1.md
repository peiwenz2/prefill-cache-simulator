# Runtime validation v1

JSON Schema validates shape. The loader and runner must additionally fail closed on
cross-file and filesystem invariants before emitting simulation events.

| Check | Required behavior |
|---|---|
| Trace integrity | Stream the configured trace file，compute SHA-256，and require an exact match with `trace.sha256`. |
| Replay order | Require non-decreasing trace timestamps；equal timestamps preserve original line order. |
| Path resolution | Resolve relative trace and output paths against the scenario file's parent directory，not process CWD. |
| Fixed-total capacity | Require `total_blocks >= prefill_nodes`；node `i` gets `floor(total/nodes)` plus one block iff `i < total mod nodes`. |
| Session family cap | For S4，require `family_size_cap <= prefill_nodes`. |
| Numeric identity range | Require trace line indices，trace block IDs，and generated indices to fit `u64`；reject negative block IDs. |
| Exact config digest | Compute SHA-256 from the original scenario file bytes before parsing；record it in every output artifact. |
| No implicit defaults | Execute only the validated resolved object；the loader must not insert behavior fields. |
| Output ownership | Refuse a non-empty output directory for the same run unless a future explicit overwrite control is added. |
| Artifact identity | Bind output metadata to `schema_version`，`scenario_id`，`seed`，trace SHA-256，and config SHA-256. |

Runtime failures occur before the first request event and include the violated field or
path in the error message.

The event loop uses a virtual millisecond clock. `replay.speed` multiplies trace time：
`virtual_arrival_ms = trace_arrival_ms / speed`，so values above 1 replay faster.
`NORMALIZED_WORK` is deliberately uncalibrated，but remains ms-shaped：
`prefill_ms = uncached_tokens × prefill_uncached_token_ms`. Its values must not be
reported as production latency. `CALIBRATED_MS` uses the same unit with a named fitted
model. Consequently `delay_ms` and `link_ttl_ms` are coherent under both modes.

Eviction capacities are per prefill node. `E2_SLRU.protected_fraction` partitions that
node's assigned blocks；`E3_SECOND_HIT_ADMISSION.ghost_capacity_blocks` is an additional
metadata-entry limit per node and does not consume KV block capacity.
