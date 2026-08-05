# Scenario schema versioning

Schema `1.x` is the strict Phase-A resolved-run contract. Later phases are design
material until a new major schema makes them executable.

- Every scenario must declare `schema_version`；the loader selects an exact supported
  major and rejects unknown majors.
- Patch releases may change only editorial text such as descriptions；the accepted
  instance set must remain identical.
- Minor releases may add only optional，behavior-neutral fields. Replaying the same
  resolved scenario must produce identical events and metrics.
- Each supported minor ships its own exact schema artifact；the current file accepts
  `1.0.0` only. The loader first selects a supported major／minor，then validates against
  that exact file.
- A major release is required for a new phase，enum member，required field，validation
  tightening or loosening，or semantic change.
- `additionalProperties: false` applies at every object boundary. The schema and loader
  supply no defaults.
- A scenario generator may provide ergonomic defaults，but it must emit a resolved
  scenario containing every behavior-affecting field before validation and execution.
- `config_sha256` is SHA-256 over the exact scenario file bytes. No JSON
  canonicalization，whitespace normalization，or key reordering is performed.
- Stable vocabulary carries forward across majors unless that major explicitly revises
  its semantics.

`docs/design-examples/advanced.kvs-d2.draft.json` is intentionally invalid under 1.x.
It records design intent，not an executable promise.
