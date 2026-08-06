"""A real :class:`~.endpoint.EngineEndpoint` that talks to a serving engine.

:class:`~.endpoint.MockEngine` implements the same protocol from a closed-form
curve, which is what makes every other module in this package testable without
an accelerator. This module is the other implementation: the one whose numbers
can honestly be called measurements.

The contract it speaks is deliberately small and versioned. Two endpoints:

``POST /v1/describe``
    Identify the machine, the engine build, the model, the topology and the
    clock configuration, and state which contract version the server speaks.
``POST /v1/measure``
    Time exactly one grid point and echo back which point was timed.

Nothing here assumes vLLM, SGLang, or any particular internal API. An engine
integrator writes a thin server exposing these two routes; that server is the
seam, and :data:`~.hardware.ENDPOINT_CONTRACT_VERSION` is what both sides agree
on. A future gRPC adapter would reuse :mod:`.hardware` unchanged.

Every response field is checked before it becomes a number. The reason is the
same one that motivates the error taxonomy in :mod:`.transport`: the dangerous
failure is not the server that is down, it is the server that answers with
something almost right. A reply that omits ``value_ms``, or echoes a different
batch size than was asked for, or returns a string where a float belongs, is a
:class:`~.transport.EndpointProtocolError` -- never a coerced value and never a
default.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .endpoint import EngineEndpoint, SweepKind, SweepObservation
from .hardware import ENDPOINT_CONTRACT_VERSION, HardwareContext
from .provenance import DishonestLabelError, MachineProvenance
from .transport import EndpointProtocolError, EndpointTransport

__all__ = [
    "DESCRIBE_FIELDS",
    "DESCRIBE_PATH",
    "MEASURE_PATH",
    "HttpJsonEngineEndpoint",
    "endpoint_is_synthetic",
]

#: Route that answers the handshake. A separate route from ``measure`` so a
#: dry run can establish what is on the other end without timing anything, and
#: so an integrator can be told exactly which half is unimplemented.
DESCRIBE_PATH = "/v1/describe"

#: Route that times one grid point.
MEASURE_PATH = "/v1/measure"

#: Fields ``describe`` must return. Listed as data so a test can assert on the
#: contract rather than on a sequence of ``if`` statements, and so an
#: integrator has one place to read the requirement from.
DESCRIBE_FIELDS: tuple[str, ...] = (
    "contract_version",
    "host_id",
    "accelerator_model",
    "engine_version",
    "captured_at_utc",
    "model_version",
    "topology",
    "clock_settings",
)

#: Fields ``measure`` must return. The four echo fields are what make a
#: response attributable to a request.
_MEASURE_FIELDS: tuple[str, ...] = (
    "kind",
    "tokens",
    "batch_size",
    "repeat_index",
    "value_ms",
)


def _require_mapping(path: str, reply: Any) -> Mapping[str, Any]:
    if not isinstance(reply, Mapping):
        raise EndpointProtocolError(
            f"{path}: response must be a JSON object, got {type(reply).__name__}"
        )
    return reply


def _require_fields(
    path: str, reply: Mapping[str, Any], fields: tuple[str, ...]
) -> None:
    missing = [name for name in fields if name not in reply]
    if missing:
        raise EndpointProtocolError(
            f"{path}: response is missing required field(s) {missing}"
        )


def _require_str(path: str, name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise EndpointProtocolError(
            f"{path}: {name} must be a string, got {type(value).__name__}"
        )
    if not value.strip():
        # A blank identifier is worse than a missing one: it satisfies a
        # presence check and then flows into provenance as though something
        # had been recorded.
        raise EndpointProtocolError(f"{path}: {name} must not be blank")
    return value


def _require_int(path: str, name: str, value: Any) -> int:
    # `bool` is an `int` in Python, so an unguarded check would accept `true`
    # as an echoed batch size of 1 and compare equal to a request for 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise EndpointProtocolError(
            f"{path}: {name} must be an integer, got {type(value).__name__}"
        )
    return value


def _require_float(path: str, name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise EndpointProtocolError(
            f"{path}: {name} must be a number, got {type(value).__name__}"
        )
    # NaN and the infinities cannot arrive through `.transport`, which refuses
    # the JSON constants that spell them. They can still arrive as the result
    # of arithmetic in a hand-rolled fake transport, and a NaN latency that
    # reaches a fit poisons every coefficient derived from it.
    numeric = float(value)
    if numeric != numeric or numeric in (float("inf"), float("-inf")):
        raise EndpointProtocolError(f"{path}: {name} must be finite, got {value!r}")
    return numeric


@dataclass(frozen=True, slots=True)
class HttpJsonEngineEndpoint:
    """Measure a real engine over the versioned JSON contract.

    ``transport`` is injected rather than constructed here. That is what lets
    every protocol violation above be tested exhaustively with a
    :class:`~.transport.StubTransport` and no socket, and it is also what keeps
    the wire format, the deadline, and the schema in three separate places.

    The class is frozen, so ``endpoint_id`` is effectively read-only and
    satisfies the read-only property declared by
    :class:`~.endpoint.EngineEndpoint`.
    """

    endpoint_id: str
    transport: EndpointTransport

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint_id, str) or not self.endpoint_id.strip():
            raise ValueError("endpoint_id must be a non-empty string")
        if not hasattr(self.transport, "request"):
            raise TypeError(
                "transport must implement EndpointTransport.request, got "
                f"{type(self.transport).__name__}"
            )

    def describe(self) -> HardwareContext:
        """Handshake: ask what is on the other end, and record the answer.

        The declared contract version is stored in the returned context rather
        than being asserted here. A mismatch is a *gate* decision
        (:attr:`~.hardware.HardwareBlocker.BLOCKED_CONTRACT_VERSION_MISMATCH`),
        and reporting it as data means a dry run against an incompatible server
        still produces a report naming the version it actually speaks, instead
        of an exception naming only the version it does not.
        """
        reply = _require_mapping(
            DESCRIBE_PATH,
            self.transport.request(
                path=DESCRIBE_PATH,
                payload={"contract_version": ENDPOINT_CONTRACT_VERSION},
            ),
        )
        _require_fields(DESCRIBE_PATH, reply, DESCRIBE_FIELDS)
        values = {
            name: _require_str(DESCRIBE_PATH, name, reply[name])
            for name in DESCRIBE_FIELDS
        }
        try:
            machine = MachineProvenance(
                values["host_id"],
                values["accelerator_model"],
                values["engine_version"],
                values["captured_at_utc"],
            )
        except DishonestLabelError as error:
            # `captured_at_utc` is the field that fails here in practice: the
            # provenance record accepts only RFC3339 at offset zero, and a
            # server reporting local time would otherwise become a timestamp
            # nobody can line up against an engine log.
            raise EndpointProtocolError(
                f"{DESCRIBE_PATH}: response does not describe a usable machine: {error}"
            ) from error
        return HardwareContext(
            machine,
            values["contract_version"],
            values["model_version"],
            values["topology"],
            values["clock_settings"],
        )

    def machine_provenance(self) -> MachineProvenance:
        """The machine half of :meth:`describe`, as the protocol requires.

        Deliberately not cached. A cached handshake would let a sweep that
        outlives a server restart attribute its later points to the machine
        that answered first.
        """
        return self.describe().machine

    def measure(
        self,
        *,
        kind: SweepKind,
        tokens: int,
        batch_size: int,
        repeat_index: int,
    ) -> SweepObservation:
        """Time one grid point, and refuse any reply that is not about it."""
        payload = {
            "contract_version": ENDPOINT_CONTRACT_VERSION,
            "kind": SweepKind(kind).value,
            "tokens": tokens,
            "batch_size": batch_size,
            "repeat_index": repeat_index,
        }
        reply = _require_mapping(
            MEASURE_PATH, self.transport.request(path=MEASURE_PATH, payload=payload)
        )
        _require_fields(MEASURE_PATH, reply, _MEASURE_FIELDS)

        echoed_kind = _require_str(MEASURE_PATH, "kind", reply["kind"])
        try:
            parsed_kind = SweepKind(echoed_kind)
        except ValueError as error:
            raise EndpointProtocolError(
                f"{MEASURE_PATH}: kind {echoed_kind!r} is not a SweepKind"
            ) from error

        echoed = {
            name: _require_int(MEASURE_PATH, name, reply[name])
            for name in ("tokens", "batch_size", "repeat_index")
        }
        requested = {
            "tokens": tokens,
            "batch_size": batch_size,
            "repeat_index": repeat_index,
        }
        # `run_sweep` performs this comparison too. It is repeated here on
        # purpose: that check guards the driver loop and raises ValueError,
        # while this one guards the wire and raises the taxonomy member that
        # says "the server broke the contract". An endpoint used outside
        # `run_sweep` would otherwise have no echo check at all.
        if parsed_kind is not SweepKind(kind) or echoed != requested:
            raise EndpointProtocolError(
                f"{MEASURE_PATH}: response describes a different grid point than "
                f"the request: asked for {SweepKind(kind).value} {requested}, "
                f"got {parsed_kind.value} {echoed}"
            )

        value = _require_float(MEASURE_PATH, "value_ms", reply["value_ms"])
        try:
            return SweepObservation(
                parsed_kind,
                echoed["tokens"],
                echoed["batch_size"],
                echoed["repeat_index"],
                value,
            )
        except ValueError as error:
            # Reached when the echo is self-consistent but the request itself
            # was outside what an observation may hold, e.g. zero tokens.
            raise EndpointProtocolError(
                f"{MEASURE_PATH}: response is not a usable observation: {error}"
            ) from error


def endpoint_is_synthetic(endpoint: EngineEndpoint) -> bool:
    """Whether ``endpoint`` produces constructed values rather than timings.

    An allowlist of one, checked by exact type. Asking the endpoint to declare
    its own honesty would put the answer in the hands of the object with the
    most to gain from lying about it, and ``isinstance`` would let a subclass
    of the HTTP adapter that overrides :meth:`~HttpJsonEngineEndpoint.measure`
    with a table lookup pass as measured.
    """
    return type(endpoint) is not HttpJsonEngineEndpoint
