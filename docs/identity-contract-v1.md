# Identity contract v1

`generated-sha256-v1` defines stable generated-block identities across processes,
Python versions, and machines. Implementations must hash the byte sequence below;
they must not hash a formatted logical request ID or language-runtime object.

## Byte layout

All integers are unsigned 64-bit big-endian. A length-prefixed byte string is an
unsigned 64-bit big-endian length followed by exactly that many bytes.

| Field | Encoding |
|---|---|
| domain tag | length-prefixed UTF-8 bytes for `generated-sha256-v1` |
| trace digest | 32 raw bytes decoded from the trace SHA-256 hex string |
| block size | configured `block_size_tokens` as `u64-be` |
| trace line index | `u64-be` |
| parent tag | one byte：`0` trace block，`1` generated block，`2` empty prefix |
| parent value | tag `0`：trace block ID as `u64-be`；tag `1`：parent digest as 32 raw bytes；tag `2`：absent |
| generated index | `u64-be`，zero-based within the request continuation |

The generated block ID is the lowercase hexadecimal rendering of
`SHA256(concatenated_fields)`. The human-readable logical request ID is display
metadata only and is never an input to this digest.

`trace-sha256-line-index-v1` is the UTF-8 display string
`trace:<64 lowercase hex trace digest>:<20-digit zero-padded decimal line index>`.
Line indices are zero-based. Parsers reject uppercase digests，signs，extra whitespace，
or indices outside `u64`.

## Boundary rules

- A trace block parent uses the trace's stable non-negative block ID.
- A generated parent uses the preceding generated block's 32 raw digest bytes.
- A request with no input blocks uses parent tag `2`.
- A partial input tail needs no token offset in v1：the trace content and encoded
  `block_size_tokens` fix its boundary. Changing either input changes the generated ID.
- Indices and trace block IDs must fit in unsigned 64-bit integers；the loader fails
  closed on overflow or negative values.

## Golden vectors

The normative vectors live in
`tests/fixtures/identity-v1-golden.json`. They cover a trace parent，a generated
parent，and an empty prefix. Tests recompute every digest from `message_hex`.
