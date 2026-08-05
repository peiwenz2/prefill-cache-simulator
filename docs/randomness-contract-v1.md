# Randomness contract v1

`named-sha256-v1` turns one scenario root seed into independent deterministic streams.
Adding or removing a random consumer must not perturb any existing stream.

For each consumer name，compute：

```text
stream_seed_bytes = SHA256(
  len64be("named-sha256-v1") || utf8("named-sha256-v1")
  || u64be(root_seed)
  || len64be(consumer_name) || utf8(consumer_name)
)
```

The 32-byte digest is the stream key. Raw block `i` is
`SHA256(stream_key || u64be(i))`，starting at counter zero. A bounded draw for upper
bound `k` consumes `w = max(1，ceil(bit_length(k) / 8))` bytes from a fresh raw block，
interprets them as an unsigned big-endian integer，and accepts values below
`2^(8w) - (2^(8w) mod k)`；accepted values return `value mod k`. Rejected draws consume
another complete raw block. Unused bytes in a raw block are discarded between draws.
This is counter-mode SHA-256 with rejection sampling，not Python `random`.

One `NamedSeedManager` memoizes exactly one advancing stream per consumer name. Calling
`stream(name)` again returns that same stream；a new manager with the same root seed
restarts every named stream at counter zero.

Phase A reserves these consumer names：

- `selector/random`：S0 choices；
- `selector/tie-break`：policy tie-breaks；
- `cluster-view/loss`：delayed-view loss draws；
- `eviction/tie-break`：eviction tie-breaks，if a policy needs randomness.

Selector experiments with the same root seed therefore share arrival and loss draws.
A new consumer gets a new stable name；renaming an existing consumer is a major schema
semantic change.
