"""One-request prefill processing."""

from __future__ import annotations

from dataclasses import dataclass

from .cache import LocalNodeCache
from .domain import ClusterView, PrefillResult, Request
from .interfaces import PrefillNodeSelector


@dataclass(slots=True)
class LocalPrefillRequestProcessor:
    selector: PrefillNodeSelector
    caches: dict[str, LocalNodeCache]

    def process(
        self, request: Request, view: ClusterView, now_ms: float
    ) -> PrefillResult:
        selection = self.selector.select(request, view)
        cache = self.caches[selection.node_id]
        lookup = cache.lookup(request, now_ms)
        placement = cache.place(request, now_ms, frozenset())
        return PrefillResult(request.request_id, selection, lookup, placement)
