"""Contract tests for the Phase-A resolved-run scenario schema."""

from __future__ import annotations

import copy
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "scenario.schema.json"
BASELINE_PATH = ROOT / "scenarios" / "baseline.local-lru.json"
DRAFT_ADVANCED_PATH = ROOT / "docs" / "design-examples" / "advanced.kvs-d2.draft.json"
GOLDEN_PATH = ROOT / "tests" / "fixtures" / "identity-v1-golden.json"


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    assert isinstance(value, dict)
    return value


@pytest.fixture(scope="module")
def schema() -> dict[str, Any]:
    return load_json(SCHEMA_PATH)


@pytest.fixture(scope="module")
def validator(schema: dict[str, Any]) -> Draft202012Validator:
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@pytest.fixture()
def baseline() -> dict[str, Any]:
    return load_json(BASELINE_PATH)


def assert_valid(
    validator: Draft202012Validator,
    scenario: dict[str, Any],
) -> None:
    validator.validate(scenario)


def assert_invalid(
    validator: Draft202012Validator,
    scenario: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        validator.validate(scenario)


def test_schema_is_valid_draft_2020_12(schema: dict[str, Any]) -> None:
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:  # pragma: no cover - clearer assertion message
        pytest.fail(f"invalid JSON Schema: {exc}")


def test_baseline_scenario_is_valid(
    baseline: dict[str, Any],
    validator: Draft202012Validator,
) -> None:
    assert_valid(validator, baseline)


def test_advanced_design_example_is_not_a_v1_scenario(
    validator: Draft202012Validator,
) -> None:
    assert_invalid(validator, load_json(DRAFT_ADVANCED_PATH))


def test_schema_version_and_phase_are_frozen(
    baseline: dict[str, Any],
    validator: Draft202012Validator,
) -> None:
    for path, value in [
        (("schema_version",), "2.0.0"),
        (("phase",), "B"),
    ]:
        changed = copy.deepcopy(baseline)
        changed[path[0]] = value
        assert_invalid(validator, changed)


def test_identity_contract_is_versioned_and_fixed(
    baseline: dict[str, Any],
    validator: Draft202012Validator,
) -> None:
    assert baseline["identity_contract"] == {
        "logical_request_id_scheme": "trace-sha256-line-index-v1",
        "generated_block_id_scheme": "generated-sha256-v1",
    }
    changed = copy.deepcopy(baseline)
    changed["identity_contract"]["generated_block_id_scheme"] = "python-hash"
    assert_invalid(validator, changed)


def test_golden_identity_vectors_are_byte_exact() -> None:
    fixture = load_json(GOLDEN_PATH)
    assert fixture["contract"] == "generated-sha256-v1"
    assert len(fixture["vectors"]) == 3
    for vector in fixture["vectors"]:
        domain = fixture["contract"].encode()
        message = struct.pack(">Q", len(domain)) + domain
        message += bytes.fromhex(fixture["trace_sha256_hex"])
        message += struct.pack(">Q", fixture["block_size_tokens"])
        message += struct.pack(">Q", vector["trace_line_index"])
        parent_kind = vector["parent_kind"]
        if parent_kind == "TRACE_BLOCK":
            message += b"\x00"
            message += struct.pack(">Q", vector["parent_trace_block_id"])
        elif parent_kind == "GENERATED_BLOCK":
            message += b"\x01"
            message += bytes.fromhex(vector["parent_generated_digest_hex"])
        else:
            assert parent_kind == "EMPTY_PREFIX"
            message += b"\x02"
        message += struct.pack(">Q", vector["generated_index"])
        assert message.hex() == vector["message_hex"]
        assert hashlib.sha256(message).hexdigest() == vector["digest_hex"]


def test_named_random_stream_contract_is_required(
    baseline: dict[str, Any],
    validator: Draft202012Validator,
) -> None:
    assert baseline["randomness_contract"] == {"stream_scheme": "named-sha256-v1"}
    changed = copy.deepcopy(baseline)
    changed.pop("randomness_contract")
    assert_invalid(validator, changed)


def test_cache_metric_units_have_canonical_order(
    baseline: dict[str, Any],
    validator: Draft202012Validator,
) -> None:
    expected = ["BLOCK_REF", "TOKEN_WEIGHTED"]
    assert baseline["metrics"]["cache_hit_units"] == expected
    changed = copy.deepcopy(baseline)
    changed["metrics"]["cache_hit_units"] = list(reversed(expected))
    assert_invalid(validator, changed)


@pytest.mark.parametrize(
    "selector",
    [
        {"id": "S0_RANDOM"},
        {"id": "S1_ROUND_ROBIN"},
        {"id": "S2_LEAST_WORK"},
        {
            "id": "S3_GB_PREFIX_BUCKET",
            "capacity_gate": {"hard_enabled": True, "soft_alpha": 1.25},
            "fallback": {
                "max_secondary_candidates": 2,
                "terminal": "LEAST_WORK",
            },
            "affinity": {"provider": "PREFIX_ANCHOR", "prefix_anchor_k": 2},
        },
        {
            "id": "S4_SESSION_AFFINITY",
            "capacity_gate": {"hard_enabled": True, "soft_alpha": 1.25},
            "fallback": {
                "max_secondary_candidates": 2,
                "terminal": "LEAST_WORK",
            },
            "affinity": {
                "provider": "ONLINE_CONVERSATION_LINKER",
                "link_ttl_ms": 60000,
                "minimum_shared_blocks": 2,
                "hot_block_percentile": 99.0,
                "family_size_cap": 4,
            },
        },
        {
            "id": "S5_FLEXLB_TTFT",
            "flexlb": {
                "cache_discount": 0.7,
                "top_fraction": 0.3,
                "similar_band_ratio": 0.1,
            },
        },
        {
            "id": "S6_CALIBRATED_TTFT",
            "calibrated": {
                "model_kind": "NORMALIZED_WORK",
                "model_version": "normalized-work-v1",
                "similar_band_ratio": 0.1,
            },
        },
    ],
    ids=["s0", "s1", "s2", "s3", "s4", "s5", "s6"],
)
def test_each_phase_a_selector_has_a_valid_explicit_shape(
    selector: dict[str, Any],
    baseline: dict[str, Any],
    validator: Draft202012Validator,
) -> None:
    changed = copy.deepcopy(baseline)
    changed["selector"] = selector
    assert_valid(validator, changed)


@pytest.mark.parametrize(
    ("selector_id", "missing_block"),
    [
        ("S3_GB_PREFIX_BUCKET", "affinity"),
        ("S4_SESSION_AFFINITY", "affinity"),
        ("S5_FLEXLB_TTFT", "flexlb"),
        ("S6_CALIBRATED_TTFT", "calibrated"),
    ],
)
def test_selector_specific_parameters_are_required(
    selector_id: str,
    missing_block: str,
    baseline: dict[str, Any],
    validator: Draft202012Validator,
) -> None:
    changed = copy.deepcopy(baseline)
    changed["selector"] = {"id": selector_id}
    assert missing_block not in changed["selector"]
    assert_invalid(validator, changed)


def test_irrelevant_selector_parameters_are_rejected(
    baseline: dict[str, Any],
    validator: Draft202012Validator,
) -> None:
    changed = copy.deepcopy(baseline)
    changed["selector"]["flexlb"] = {
        "cache_discount": 0.7,
        "top_fraction": 0.3,
        "similar_band_ratio": 0.1,
    }
    assert_invalid(validator, changed)


@pytest.mark.parametrize(
    "eviction",
    [
        {"id": "E0_FIFO"},
        {"id": "E1_LRU"},
        {"id": "E2_SLRU", "protected_fraction": 0.8},
        {"id": "E3_SECOND_HIT_ADMISSION", "ghost_capacity_blocks": 10000},
    ],
    ids=["e0", "e1", "e2", "e3"],
)
def test_each_phase_a_eviction_has_a_valid_explicit_shape(
    eviction: dict[str, Any],
    baseline: dict[str, Any],
    validator: Draft202012Validator,
) -> None:
    changed = copy.deepcopy(baseline)
    changed["eviction"] = eviction
    assert_valid(validator, changed)


def test_eviction_parameters_are_bound_to_policy_id(
    baseline: dict[str, Any],
    validator: Draft202012Validator,
) -> None:
    missing = copy.deepcopy(baseline)
    missing["eviction"] = {"id": "E2_SLRU"}
    assert_invalid(validator, missing)

    irrelevant = copy.deepcopy(baseline)
    irrelevant["eviction"] = {"id": "E0_FIFO", "protected_fraction": 0.8}
    assert_invalid(validator, irrelevant)


def test_cache_only_has_no_queue_or_inflight_semantics(
    baseline: dict[str, Any],
    validator: Draft202012Validator,
) -> None:
    cache_only = copy.deepcopy(baseline)
    cache_only["replay"] = {
        "mode": "CACHE_ONLY",
        "cache_visibility": "INSERT_AT_COMPLETION",
        "cluster_view": {"mode": "ORACLE"},
    }
    cache_only["selector"] = {"id": "S1_ROUND_ROBIN"}
    cache_only["metrics"]["report_inflight_wait"] = False
    assert_valid(validator, cache_only)

    cache_only["replay"]["cache_visibility"] = "INFLIGHT_DEDUP_WAIT"
    assert_invalid(validator, cache_only)

    cache_only["replay"]["cache_visibility"] = "INSERT_AT_COMPLETION"
    cache_only["selector"] = {"id": "S2_LEAST_WORK"}
    assert_invalid(validator, cache_only)


def test_inflight_wait_metric_is_bidirectionally_bound(
    baseline: dict[str, Any],
    validator: Draft202012Validator,
) -> None:
    changed = copy.deepcopy(baseline)
    changed["metrics"]["report_inflight_wait"] = True
    assert_invalid(validator, changed)


def test_inflight_visibility_requires_wait_accounting(
    baseline: dict[str, Any],
    validator: Draft202012Validator,
) -> None:
    changed = copy.deepcopy(baseline)
    changed["replay"]["cache_visibility"] = "INFLIGHT_DEDUP_WAIT"
    changed["metrics"]["report_inflight_wait"] = False
    assert_invalid(validator, changed)


def test_calibrated_time_requires_model_version(
    baseline: dict[str, Any],
    validator: Draft202012Validator,
) -> None:
    changed = copy.deepcopy(baseline)
    changed["replay"]["time_model"]["kind"] = "CALIBRATED_MS"
    changed["replay"]["time_model"].pop("model_version")
    assert_invalid(validator, changed)


def test_phase_a_topology_and_execution_are_fixed(
    baseline: dict[str, Any],
    validator: Draft202012Validator,
) -> None:
    changed = copy.deepcopy(baseline)
    changed["topology"]["mode"] = "SHARED_KVS"
    assert_invalid(validator, changed)

    changed = copy.deepcopy(baseline)
    changed["execution"]["decode_policy"] = "D1_FIXED_LEASE"
    assert_invalid(validator, changed)


def test_unknown_fields_are_rejected(
    baseline: dict[str, Any],
    validator: Draft202012Validator,
) -> None:
    changed = copy.deepcopy(baseline)
    changed["silent_fallback"] = True
    assert_invalid(validator, changed)


def test_conditionals_have_explicit_presence_guards(schema: dict[str, Any]) -> None:
    offenders: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            conditional = node.get("if")
            if (
                isinstance(conditional, dict)
                and "properties" in conditional
                and "required" not in conditional
            ):
                offenders.append(f"{path}/if")
            for key, value in node.items():
                walk(value, f"{path}/{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}/{index}")

    walk(schema, "#")
    assert not offenders, f"vacuity-prone conditionals: {offenders}"
