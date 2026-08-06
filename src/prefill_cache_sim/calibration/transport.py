"""The wire seam beneath :mod:`.http_endpoint`, and its error taxonomy.

:class:`~.endpoint.EngineEndpoint` says *what* a measurement is. This module says
how one request reaches a server and comes back, and nothing else: it does not
know about sweeps, kinds, or calibration labels. Splitting it out is what lets
:mod:`.http_endpoint` be tested exhaustively -- timeouts, refusals, malformed
bodies, contract violations -- with no socket and no network, by substituting a
transport rather than by monkeypatching a module.

The error taxonomy is the point of the split. A hardware calibration run must be
able to tell these three apart, because they call for different responses and
because two of them must never be confused with a *measurement*:

:class:`EndpointTimeout`
    The engine did not answer in time. The measurement did not happen. It is
    emphatically not "a very large latency": recording it as a number would fit
    a curve to the harness's own patience.
:class:`EndpointTransportError`
    The request never completed -- refused, unresolvable, or a non-2xx status.
    Again, no measurement happened.
:class:`EndpointProtocolError`
    Bytes came back, but they are not the agreed contract. This is the dangerous
    one: a response that is *almost* right is the failure most likely to be
    mistaken for data, so it is raised rather than coerced.

None of these ever produce a :class:`~.endpoint.SweepObservation`. That class
rejects non-finite values, but the hole it cannot close is a *finite* number
invented by an error path -- a timeout charged as ``timeout_s``, a 503 body
parsed as ``0.0``. Raising here is what keeps that number from existing.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "ALLOWED_URL_SCHEMES",
    "MAX_RESPONSE_BYTES",
    "EndpointError",
    "EndpointProtocolError",
    "EndpointTimeout",
    "EndpointTransport",
    "EndpointTransportError",
    "HttpJsonTransport",
    "StubTransport",
]

#: Only real network schemes are accepted. ``urllib`` also opens ``file:`` and
#: ``ftp:``, and a ``file:`` URL is the single cheapest way to turn a
#: hand-written JSON document into something that looks like an engine and gets
#: labelled HW_CALIBRATED. Refusing the scheme here removes that path entirely
#: rather than relying on a reviewer noticing the URL in a config.
ALLOWED_URL_SCHEMES = frozenset({"http", "https"})

#: Responses are read up to this bound. An unbounded ``read()`` against a hostile
#: or malfunctioning server is an out-of-memory condition, which would abort a
#: sweep midway and leave a partial observation set with no explanation.
MAX_RESPONSE_BYTES = 1 << 20


class EndpointError(RuntimeError):
    """Base class for every way a measurement request can fail to produce data."""


class EndpointTimeout(EndpointError):
    """The engine did not answer within the configured deadline."""


class EndpointTransportError(EndpointError):
    """The request did not complete: refused, unresolvable, or non-2xx."""


class EndpointProtocolError(EndpointError):
    """A response arrived but does not satisfy the agreed contract."""


@runtime_checkable
class EndpointTransport(Protocol):
    """One request/response exchange, in decoded JSON objects.

    The transport owns the wire format and the deadline; the caller owns the
    schema. That division is deliberate: a fake transport for tests is then a
    dictionary lookup, and every schema rule stays in one place where it is
    exercised by the same tests regardless of how the bytes arrived.

    Implementations must raise a member of the taxonomy above and must never
    return a sentinel value for a failed call.
    """

    def request(
        self, *, path: str, payload: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...


def _decode(body: bytes, path: str) -> Mapping[str, Any]:
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EndpointProtocolError(
            f"{path}: response body is not valid UTF-8"
        ) from error
    try:
        # parse_constant fires for the non-standard NaN/Infinity/-Infinity
        # literals, which json.loads otherwise accepts. One of them reaching a
        # fit poisons every coefficient derived from it, and the resulting
        # artifact re-serializes into a file no strict JSON reader can load.
        decoded = json.loads(
            text,
            parse_constant=lambda name: (_ for _ in ()).throw(
                ValueError(f"non-JSON constant {name!r}")
            ),
        )
    except ValueError as error:
        raise EndpointProtocolError(f"{path}: response is not JSON: {error}") from error
    if not isinstance(decoded, Mapping):
        raise EndpointProtocolError(
            f"{path}: response must be a JSON object, got {type(decoded).__name__}"
        )
    return decoded


@dataclass(frozen=True, slots=True)
class HttpJsonTransport:
    """POST a JSON object to ``base_url + path`` and decode the JSON reply.

    ``opener`` is injected so this class itself is testable without a socket;
    it defaults to the stdlib opener. There is no third-party HTTP dependency
    because the package declares none, and a calibration harness is a poor
    reason to add one.

    ``timeout_s`` is a single deadline applied to the whole exchange. It is a
    required, explicit value rather than a default, because a transport that
    silently waits forever turns a hung engine into a hung sweep with no
    artifact and no error.
    """

    base_url: str
    timeout_s: float
    opener: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        scheme = self.base_url.split(":", 1)[0].lower()
        if scheme not in ALLOWED_URL_SCHEMES:
            raise ValueError(
                f"base_url scheme {scheme!r} is not one of "
                f"{sorted(ALLOWED_URL_SCHEMES)}; a local file cannot be a "
                "measured engine"
            )
        if isinstance(self.timeout_s, bool) or not isinstance(
            self.timeout_s, int | float
        ):
            raise ValueError("timeout_s must be a number")
        if not self.timeout_s > 0:
            raise ValueError(f"timeout_s must be positive, got {self.timeout_s!r}")

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

    def request(self, *, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        url = self._url(path)
        # allow_nan=False on the way out for the same reason it is refused on the
        # way in: a request carrying a bare NaN is not JSON and the server's
        # rejection of it would surface as an opaque 400.
        body = json.dumps(dict(payload), sort_keys=True, allow_nan=False).encode(
            "utf-8"
        )
        request = urllib.request.Request(  # noqa: S310 - scheme checked above
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        opener = self.opener or urllib.request.urlopen
        try:
            with opener(request, timeout=self.timeout_s) as response:
                status = getattr(response, "status", None)
                if status is not None and not 200 <= int(status) < 300:
                    raise EndpointTransportError(f"{url}: HTTP status {status}")
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except TimeoutError as error:
            raise EndpointTimeout(
                f"{url}: no response within {self.timeout_s}s"
            ) from error
        except urllib.error.HTTPError as error:
            raise EndpointTransportError(f"{url}: HTTP status {error.code}") from error
        except urllib.error.URLError as error:
            # A timeout surfaces here rather than as TimeoutError whenever it is
            # raised below the HTTP layer, and the distinction between "did not
            # answer" and "could not be reached" is the whole point of the
            # taxonomy, so the reason is inspected instead of being flattened.
            if isinstance(error.reason, TimeoutError | socket.timeout):
                raise EndpointTimeout(
                    f"{url}: no response within {self.timeout_s}s"
                ) from error
            raise EndpointTransportError(f"{url}: {error.reason}") from error
        except OSError as error:
            raise EndpointTransportError(f"{url}: {error}") from error

        if len(raw) > MAX_RESPONSE_BYTES:
            raise EndpointProtocolError(
                f"{url}: response exceeds {MAX_RESPONSE_BYTES} bytes"
            )
        return _decode(raw, url)


@dataclass(frozen=True, slots=True)
class StubTransport:
    """A scripted transport for tests: exact request in, canned reply out.

    It ships in the package rather than in the test module because the failure
    modes it exists to reproduce -- a timeout, a truncated body, a response that
    echoes the wrong grid point -- are properties of the *contract*, and anything
    that implements the contract should be able to be tested against them.

    A stub can never be mistaken for an engine: it has no URL and produces no
    provenance, and :mod:`.hardware` refuses a calibration whose endpoint
    reports incomplete provenance.
    """

    responses: Mapping[str, Any]
    calls: list[tuple[str, Mapping[str, Any]]] = field(default_factory=list)

    def request(self, *, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        self.calls.append((path, dict(payload)))
        if path not in self.responses:
            raise EndpointTransportError(f"{path}: stub has no scripted response")
        reply = self.responses[path]
        if isinstance(reply, BaseException):
            raise reply
        if callable(reply):
            reply = reply(payload)
        if isinstance(reply, BaseException):
            raise reply
        if not isinstance(reply, Mapping):
            raise EndpointProtocolError(
                f"{path}: scripted reply must be an object, got {type(reply).__name__}"
            )
        return reply
