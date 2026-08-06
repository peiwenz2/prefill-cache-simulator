"""Typed accessors for untrusted payloads crossing back into the harness.

Every ``from_dict`` in this package is a system boundary: the bytes may come from
a hand-edited artifact, a truncated file, or a different schema version. Indexing
``payload[...]`` directly fails loudly on a *missing* key but silently accepts a
*wrong-typed* one, and in this package a wrong type is worse than a crash:

* ``input_tokens`` arriving as ``"1024"`` from one source and ``1024`` from
  another makes :func:`~.reconcile.disagreement_entries` report a DISAGREEMENT
  between two observers that actually agree -- a fabricated finding.
* ``attempt_index`` arriving as a string makes :class:`~.sources.AttemptKey`
  hash and sort differently from its integer twin, so one attempt becomes two
  and each is charged MISSING against all three sources.

Both failures produce a plausible-looking ledger out of a malformed file, which
is precisely the outcome the ledger exists to rule out. These helpers therefore
reject the wrong type at the boundary instead of propagating it.

``bool`` is rejected wherever a number is required. It is a subclass of ``int``
in Python, so ``attempt_index: true`` would otherwise silently become ``1``.

``NaN`` and ``Infinity`` are rejected wherever a number is required. They are not
JSON, but :func:`json.loads` accepts them by default, and once inside they are
worse than a wrong type because they pass every comparison silently: ``NaN`` is
neither above nor below a gate threshold, so a threshold check on it *succeeds*
and a rank comparison against it reports a tie. A single ``NaN`` in a
hand-edited artifact would therefore turn into a clean recommendation rather
than an error.

This module also carries the small predicates (:func:`check`,
:func:`check_finite`, :func:`check_unit_interval`) that every record's
``__post_init__`` uses to state its own invariants. They live here so that the
rule is written once: a record enforces itself on construction, and
``from_dict`` inherits the same enforcement by building the same record rather
than by re-checking the payload a second, subtly different way.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, TypeVar

__all__ = [
    "MalformedPayloadError",
    "check",
    "check_finite",
    "check_ordered_subset",
    "check_unit_interval",
    "invalid",
    "optional_float",
    "optional_str",
    "require_bool",
    "require_float",
    "require_float_sequence",
    "require_int",
    "require_mapping",
    "require_object",
    "require_sequence",
    "require_str",
    "require_str_sequence",
]

_T = TypeVar("_T")


class MalformedPayloadError(ValueError):
    """A payload was structurally unusable, so nothing was built from it."""


def invalid(context: str, problem: str) -> MalformedPayloadError:
    """Return the error for a record that parsed but cannot be true."""
    return MalformedPayloadError(f"{context}: {problem}")


def check(condition: bool, context: str, problem: str) -> None:
    """Raise :class:`MalformedPayloadError` unless ``condition`` holds."""
    if not condition:
        raise invalid(context, problem)


def check_finite(value: float | None, name: str, context: str) -> None:
    """Reject ``NaN`` and infinity on an already-constructed value.

    ``None`` passes: an absent observation is a different thing from an
    unusable one, and the fields that may be ``None`` say so in their type.
    """
    if value is None:
        return
    if not math.isfinite(value):
        raise invalid(context, f"{name} must be a finite number, got {value!r}")


def check_unit_interval(value: float, name: str, context: str) -> None:
    """Require a finite value within ``[0, 1]``, as fractions and rates are."""
    check_finite(value, name, context)
    if not 0.0 <= value <= 1.0:
        raise invalid(context, f"{name} must be within [0, 1], got {value!r}")


def check_ordered_subset(
    values: Sequence[Any], universe: Iterable[Any], name: str, context: str
) -> None:
    """Require ``values`` to be distinct members of ``universe``, in its order.

    Ledger entries list their sources in a canonical order so that two runs
    that found the same defect produce the same bytes. Accepting a reordered or
    repeated list would make two spellings of one finding compare unequal.
    """
    allowed = tuple(universe)
    for item in values:
        if item not in allowed:
            raise invalid(context, f"{name} contains unknown member {item!r}")
    if len(set(values)) != len(values):
        raise invalid(context, f"{name} repeats a member: {list(values)!r}")
    expected = [item for item in allowed if item in set(values)]
    if list(values) != expected:
        raise invalid(
            context, f"{name} must be in canonical order, got {list(values)!r}"
        )


def _missing(context: str, name: str) -> MalformedPayloadError:
    return MalformedPayloadError(f"{context}: missing field {name!r}")


def _wrong(
    context: str, name: str, expected: str, value: object
) -> MalformedPayloadError:
    return MalformedPayloadError(
        f"{context}: field {name!r} must be {expected}, got "
        f"{type(value).__name__} ({value!r})"
    )


def _not_finite(context: str, name: str, value: object) -> MalformedPayloadError:
    return MalformedPayloadError(
        f"{context}: field {name!r} must be a finite number, got {value!r}"
    )


def require_mapping(payload: object, context: str) -> Mapping[str, Any]:
    """Return ``payload`` as a mapping or refuse to interpret it."""
    if not isinstance(payload, Mapping):
        raise MalformedPayloadError(
            f"{context}: expected an object, got {type(payload).__name__}"
        )
    return payload


def _get(payload: Mapping[str, Any], name: str, context: str) -> Any:
    if name not in payload:
        raise _missing(context, name)
    return payload[name]


def require_object(payload: Mapping[str, Any], name: str, context: str) -> Any:
    """Return a nested object, blaming the parent for its absence.

    Delegating a missing ``key`` straight to the child's ``from_dict`` would
    report the child's first missing field instead of the field that is actually
    absent, which sends a reader looking in the wrong record.
    """
    value = _get(payload, name, context)
    if not isinstance(value, Mapping):
        raise _wrong(context, name, "an object", value)
    return value


def require_str(payload: Mapping[str, Any], name: str, context: str) -> str:
    value = _get(payload, name, context)
    if not isinstance(value, str):
        raise _wrong(context, name, "a string", value)
    return value


def optional_str(payload: Mapping[str, Any], name: str, context: str) -> str | None:
    value = _get(payload, name, context)
    if value is None:
        return None
    if not isinstance(value, str):
        raise _wrong(context, name, "a string or null", value)
    return value


def require_int(payload: Mapping[str, Any], name: str, context: str) -> int:
    value = _get(payload, name, context)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _wrong(context, name, "an integer", value)
    return value


def require_bool(payload: Mapping[str, Any], name: str, context: str) -> bool:
    value = _get(payload, name, context)
    if not isinstance(value, bool):
        raise _wrong(context, name, "a boolean", value)
    return value


def require_float(payload: Mapping[str, Any], name: str, context: str) -> float:
    """Accept a JSON number. Integers are numbers; booleans are not."""
    value = _get(payload, name, context)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _wrong(context, name, "a number", value)
    if not math.isfinite(value):
        raise _not_finite(context, name, value)
    return float(value)


def optional_float(payload: Mapping[str, Any], name: str, context: str) -> float | None:
    value = _get(payload, name, context)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _wrong(context, name, "a number or null", value)
    if not math.isfinite(value):
        raise _not_finite(context, name, value)
    return float(value)


def require_sequence(
    payload: Mapping[str, Any], name: str, context: str
) -> Sequence[Any]:
    """Return a JSON array. Strings are rejected: iterating one yields characters."""
    value = _get(payload, name, context)
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise _wrong(context, name, "an array", value)
    return value


def require_str_sequence(
    payload: Mapping[str, Any], name: str, context: str
) -> tuple[str, ...]:
    """Return a JSON array of strings, checking every element.

    The ledger stringifies observed values before recording them, so an element
    that is not a string means the array was not built by the reconciler. The
    element is named here rather than left to fail obscurely downstream.
    """
    values = require_sequence(payload, name, context)
    for item in values:
        if not isinstance(item, str):
            raise _wrong(context, name, "an array of strings", item)
    return tuple(values)


def require_float_sequence(
    payload: Mapping[str, Any], name: str, context: str
) -> tuple[float, ...]:
    """Return a JSON array of numbers, checking every element."""
    values = require_sequence(payload, name, context)
    for item in values:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise _wrong(context, name, "an array of numbers", item)
        if not math.isfinite(item):
            raise _not_finite(context, name, item)
    return tuple(float(item) for item in values)
