from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from prefill_cache_sim.m12_kernel import (
    AttemptExecutionSpec,
    AttemptTerminal,
    CacheMutation,
    CausalKernel,
    FrozenKernelCostModel,
    KernelConfig,
    KernelPolicy,
    KernelRequestSpec,
)
from prefill_cache_sim.m12_metrics import LogicalRequestSpec


def request(
    identity: str,
    *,
    arrival: float = 0,
    tenant: str = "tenant-a",
    tier: str = "STANDARD",
    input_tokens: int = 10,
    output_tokens: int = 2,
) -> LogicalRequestSpec:
    return LogicalRequestSpec(
        identity, tenant, tier, arrival, input_tokens, output_tokens
    )


class RecordingPolicy(KernelPolicy):
    def __init__(self, plans: dict[str, tuple[AttemptExecutionSpec, ...]]) -> None:
        self.plans = plans
        self.visible: list[tuple[str, frozenset[str]]] = []

    def plan_attempts(self, request, view):
        identity = request.logical.logical_request_id
        self.visible.append((identity, view.completed_cache_keys))
        return self.plans.get(identity, ())[:1]

    def cache_mutation(self, request, attempt, view):
        return CacheMutation(admit=True)

    def next_attempt(self, request, previous, view):
        plans = self.plans.get(request.logical.logical_request_id, ())
        return next(
            (
                plan
                for plan in plans
                if plan.attempt_index == previous.attempt_index + 1
            ),
            None,
        )


def attempt(
    logical: str,
    index: int = 0,
    *,
    arrival: float = 0,
    p: float = 1,
    d: float = 1,
    kvs: float = 0,
    emitted: int = 2,
    placement: str | None = None,
    p_node: str = "p0",
    d_node: str = "d0",
    completes: bool = True,
    reusable: bool = True,
) -> AttemptExecutionSpec:
    del p, d
    return AttemptExecutionSpec(
        logical_request_id=logical,
        attempt_index=index,
        arrival_work=arrival,
        p_node_id=p_node,
        d_node_id=d_node,
        emitted_output_tokens=emitted,
        remote_cache_keys=(placement,) if kvs and placement else (),
        terminal=AttemptTerminal.COMPLETED if completes else AttemptTerminal.FAILED,
        prefill_reusable=reusable,
    )


def kernel(
    *,
    end: float = 100,
    tiers=None,
    prefill_rate: float = 0.1,
    kvs_rate: float = 0.1,
    retry_budget: int = 2,
) -> CausalKernel:
    return CausalKernel(
        KernelConfig(
            observation_start_work=0,
            observation_end_work=end,
            prefill_node_ids=("p0", "p1"),
            decode_node_ids=("d0", "d1"),
            tier_slo_work=tiers or {"STANDARD": 20},
            cache_capacity_entries=2,
            retry_budget=retry_budget,
            cost_model=FrozenKernelCostModel(
                prefill_work_per_token=prefill_rate,
                kvs_work_per_token=kvs_rate,
                kvs_bytes_per_token=10,
                decode_work_per_token=0.5,
            ),
        )
    )


def cached_request(logical: LogicalRequestSpec, key: str) -> KernelRequestSpec:
    return KernelRequestSpec(logical, (key,), (logical.input_tokens,))


def test_workload_and_attempt_specs_are_frozen_and_independent() -> None:
    logical = request("r")
    execution = attempt("r")
    with pytest.raises(FrozenInstanceError):
        logical.input_tokens = 99  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        execution.emitted_output_tokens = 99  # type: ignore[misc]


def test_completion_is_materialized_before_same_time_arrival() -> None:
    policy = RecordingPolicy(
        {
            "first": (attempt("first", p=1, d=1, placement="shared"),),
            "second": (attempt("second", arrival=2),),
        }
    )
    result = kernel().run(
        [
            cached_request(request("first"), "shared"),
            request("second", arrival=2),
        ],
        policy,
    )
    assert policy.visible == [("first", frozenset()), ("second", frozenset({"shared"}))]
    assert [event.kind for event in result.events if event.at_work == 2] == [
        "COMPLETION",
        "ARRIVAL",
    ]


def test_future_completion_is_not_visible_and_placement_occurs_only_then() -> None:
    policy = RecordingPolicy(
        {
            "slow": (attempt("slow", p=5, d=1, placement="future"),),
            "observer": (),
        }
    )
    result = kernel().run(
        [cached_request(request("slow"), "future"), request("observer", arrival=1)],
        policy,
    )
    assert policy.visible[1] == ("observer", frozenset())
    assert result.completed_cache_keys == frozenset({"future"})


def test_policy_cannot_claim_a_future_hit_or_zero_prefill_work() -> None:
    policy = RecordingPolicy(
        {
            "a": (attempt("a", d=1),),
            "b": (attempt("b", arrival=1, d=1, kvs=1, placement="K", p_node="p1"),),
        }
    )
    result = kernel().run(
        [
            cached_request(request("a", input_tokens=10), "K"),
            cached_request(request("b", arrival=1, input_tokens=10), "K"),
        ],
        policy,
    )
    by_id = {item.logical_request_id: item.outcome for item in result.attempts}
    # A only publishes K at t=2. B starts P at t=1, so the remote claim is false.
    assert by_id["b"].prefill_gpu_work == 1
    assert by_id["b"].kvs_normalized_work == 0
    assert by_id["b"].finish_work == 3


def test_policy_cannot_self_report_zero_decode_work() -> None:
    result = kernel().run(
        [request("r", output_tokens=4)],
        RecordingPolicy({"r": (attempt("r", emitted=4),)}),
    )
    assert result.attempts[0].outcome.decode_gpu_work == 2


def test_completion_time_eviction_is_bounded_and_causal() -> None:
    class EvictingPolicy(RecordingPolicy):
        def cache_mutation(self, request, attempt, view):
            victims = ("old",) if request.logical.logical_request_id == "new" else ()
            return CacheMutation(admit=True, evict_keys=victims)

    policy = EvictingPolicy(
        {"old": (attempt("old"),), "new": (attempt("new", arrival=2),)}
    )
    result = kernel().run(
        [
            cached_request(request("old"), "old"),
            cached_request(request("new", arrival=2), "new"),
        ],
        policy,
    )
    assert result.completed_cache_keys == frozenset({"new"})


def test_admission_cannot_exceed_kernel_owned_cache_capacity() -> None:
    policy = RecordingPolicy({"wide": (attempt("wide"),)})
    wide = KernelRequestSpec(request("wide"), ("a", "b", "c"), (3, 3, 4))
    with pytest.raises(ValueError, match="bounded capacity"):
        kernel().run([wide], policy)


def test_positive_kvs_work_advances_completion_and_can_fail_evaluator_slo() -> None:
    workload = [
        cached_request(request("seed"), "K"),
        cached_request(request("r", arrival=2), "K"),
    ]
    no_kvs = kernel(tiers={"STANDARD": 1.5}, kvs_rate=0).run(
        workload,
        RecordingPolicy(
            {
                "seed": (attempt("seed"),),
                "r": (attempt("r", arrival=2, kvs=1, placement="K", p_node="p1"),),
            }
        ),
    )
    with_kvs = kernel(tiers={"STANDARD": 1.5}, kvs_rate=0.1).run(
        workload,
        RecordingPolicy(
            {
                "seed": (attempt("seed"),),
                "r": (attempt("r", arrival=2, kvs=1, placement="K", p_node="p1"),),
            }
        ),
    )
    by_id_no = {item.logical_request_id: item for item in no_kvs.attempts}
    by_id_yes = {item.logical_request_id: item for item in with_kvs.attempts}
    assert by_id_no["r"].finish_work == 3
    assert by_id_yes["r"].finish_work == 4
    assert by_id_no["r"].strict_slo_met
    assert not by_id_yes["r"].strict_slo_met


def test_p_and_d_pools_queue_independently() -> None:
    plans = {
        "a": (attempt("a", p=4, d=1, p_node="p0", d_node="d0"),),
        "b": (attempt("b", p=1, d=4, emitted=8, p_node="p1", d_node="d0"),),
    }
    result = kernel().run(
        [request("a", input_tokens=40), request("b", output_tokens=8)],
        RecordingPolicy(plans),
    )
    by_id = {value.logical_request_id: value for value in result.attempts}
    assert by_id["b"].prefill_finish_work == 1
    assert by_id["a"].prefill_finish_work == 4
    assert by_id["a"].finish_work == 6  # waits for b's decode on shared d0


def test_future_retry_does_not_reserve_an_idle_p_pool_early() -> None:
    policy = RecordingPolicy(
        {
            "future": (attempt("future", arrival=10),),
            "now": (attempt("now", arrival=1),),
        }
    )
    result = kernel().run([request("future"), request("now", arrival=1)], policy)
    by_id = {item.logical_request_id: item for item in result.attempts}
    assert by_id["now"].prefill_finish_work == 2
    assert by_id["future"].prefill_finish_work == 11


def test_future_plan_is_not_counted_in_queue_proxy_until_ready() -> None:
    class QueueObserver(RecordingPolicy):
        def __init__(self):
            super().__init__({"future": (attempt("future", arrival=10),), "seen": ()})
            self.at_one = None

        def plan_attempts(self, request, view):
            if request.logical.logical_request_id == "seen":
                self.at_one = view.queued_prefill_work["p0"]
            return super().plan_attempts(request, view)

    policy = QueueObserver()
    kernel().run([request("future"), request("seen", arrival=1)], policy)
    assert policy.at_one == 0


def test_attempt_beyond_horizon_is_planned_but_not_issued_or_charged() -> None:
    result = kernel(end=10).run(
        [request("r")], RecordingPolicy({"r": (attempt("r", arrival=99),)})
    )
    assert result.metrics.offered_logical_requests == 1
    assert result.metrics.attempt_count == 0
    assert result.metrics.total_gpu_work == 0


def test_same_time_later_arrival_observes_kernel_owned_p_queue_work() -> None:
    class QueuePolicy(RecordingPolicy):
        def __init__(self):
            super().__init__({"a": (attempt("a"),), "b": ()})
            self.queued = []
            self.available = []

        def plan_attempts(self, request, view):
            self.queued.append(view.queued_prefill_work["p0"])
            self.available.append(view.prefill_available_at["p0"])
            return super().plan_attempts(request, view)

    policy = QueuePolicy()
    kernel().run([request("a"), request("b")], policy)
    assert policy.queued == [0, 0]
    assert policy.available == [0, 1]


def test_zero_attempt_drop_remains_offered_and_jain_uses_tenants() -> None:
    policy = RecordingPolicy({"served": (attempt("served"),), "drop": ()})
    result = kernel().run(
        [request("served", tenant="a"), request("drop", tenant="b")], policy
    )
    assert result.metrics.offered_logical_requests == 2
    assert result.metrics.attempt_count == 1
    assert result.metrics.jain_fairness == pytest.approx(0.5)


def test_retries_charge_all_work_but_get_useful_credit_once() -> None:
    policy = RecordingPolicy(
        {
            "r": (
                attempt("r", 0, emitted=1, completes=False, reusable=False),
                attempt("r", 1, emitted=2, p_node="p1", d_node="d1"),
            )
        }
    )
    result = kernel().run([request("r")], policy)
    assert result.metrics.attempt_count == 2
    assert result.metrics.total_gpu_work == 3.5
    assert result.metrics.strict_completed_requests == 1
    assert result.metrics.strict_useful_tokens == 12


def test_unfinished_at_horizon_is_censored_but_offered_and_work_remain() -> None:
    result = kernel(end=5).run(
        [request("r", output_tokens=10)],
        RecordingPolicy({"r": (attempt("r", emitted=10),)}),
    )
    assert result.metrics.offered_logical_requests == 1
    assert result.metrics.attempt_count == 1
    assert result.metrics.total_gpu_work == 6
    assert not result.attempts[0].completed
    assert result.attempts[0].finish_work is None


def test_failed_nonreusable_work_is_waste_and_does_not_publish_cache() -> None:
    result = kernel().run(
        [cached_request(request("r"), "K")],
        RecordingPolicy(
            {"r": (attempt("r", emitted=1, completes=False, reusable=False),)}
        ),
    )
    assert result.metrics.wasted_gpu_work == 1.5
    assert result.completed_cache_keys == frozenset()


def test_retry_is_completion_driven_and_budget_fenced() -> None:
    policy = RecordingPolicy(
        {
            "r": (
                attempt("r", emitted=1, completes=False, reusable=False),
                attempt("r", 1, emitted=1, completes=False, reusable=False),
            )
        }
    )
    result = kernel().run([request("r")], policy)
    assert [event.kind for event in result.events].count("COMPLETION") == 2
    assert result.attempts[1].prefill_finish_work == 2.5
    with pytest.raises(ValueError, match="retry budget"):
        kernel(retry_budget=0).run([request("r")], policy)


def test_future_retry_does_not_leak_into_queue_before_readiness() -> None:
    class RetryQueueObserver(RecordingPolicy):
        def __init__(self):
            super().__init__(
                {
                    "r": (
                        attempt("r", emitted=0, completes=False, reusable=False),
                        attempt("r", 1, arrival=10),
                    ),
                    "at-two": (),
                    "at-ten": (),
                }
            )
            self.queue_views = {}

        def plan_attempts(self, request, view):
            identity = request.logical.logical_request_id
            if identity.startswith("at-"):
                self.queue_views[identity] = view.queued_prefill_work["p0"]
            return super().plan_attempts(request, view)

    policy = RetryQueueObserver()
    kernel().run(
        [
            request("r"),
            request("at-two", arrival=2),
            request("at-ten", arrival=10),
        ],
        policy,
    )
    assert policy.queue_views == {"at-two": 0, "at-ten": 0}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_attempt_and_workload_arrivals_are_rejected(value) -> None:
    with pytest.raises(ValueError, match="finite"):
        attempt("r", arrival=value)
    malformed = request("r")
    object.__setattr__(malformed, "arrival_work", value)
    with pytest.raises(ValueError, match="finite"):
        kernel().run([malformed], RecordingPolicy({}))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_config_window_and_costs_are_rejected(value) -> None:
    with pytest.raises(ValueError, match="finite"):
        KernelConfig(
            value,
            10,
            ("p0",),
            ("d0",),
            {"STANDARD": 1},
            1,
            FrozenKernelCostModel(0.1, 0.1, 1, 0.5),
        )
    with pytest.raises(ValueError, match="finite"):
        FrozenKernelCostModel(value, 0.1, 1, 0.5)


@pytest.mark.parametrize(
    ("workload", "message"),
    [
        ([request("b", arrival=1), request("a", arrival=0)], "arrival order"),
        ([request("a"), request("a")], "duplicate workload"),
    ],
)
def test_input_order_and_identity_are_validated(workload, message) -> None:
    with pytest.raises(ValueError, match=message):
        kernel().run(workload, RecordingPolicy({}))


def test_policy_cannot_override_frozen_tenant_tier_output_or_slo() -> None:
    bad_output = RecordingPolicy({"r": (attempt("r", emitted=3),)})
    with pytest.raises(ValueError, match="true output"):
        kernel().run([request("r", tenant="owner", tier="STANDARD")], bad_output)


def test_unknown_attempt_and_invalid_attempt_order_are_rejected() -> None:
    with pytest.raises(ValueError, match="references unknown"):
        kernel().run([request("r")], RecordingPolicy({"r": (attempt("x"),)}))

    class BulkPolicy(RecordingPolicy):
        def plan_attempts(self, request, view):
            return self.plans[request.logical.logical_request_id]

    plans = {"r": (attempt("r", 1), attempt("r", 0))}
    with pytest.raises(ValueError, match="at most one"):
        kernel().run([request("r")], BulkPolicy(plans))


@pytest.mark.parametrize(
    "field,value",
    [
        ("attempt_index", True),
        ("attempt_index", 0.0),
        ("emitted_output_tokens", False),
        ("emitted_output_tokens", 2.0),
        ("terminal", "COMPLETED"),
    ],
)
def test_attempt_boundary_rejects_non_exact_scalar_types(field, value) -> None:
    values = {
        "logical_request_id": "r",
        "attempt_index": 0,
        "arrival_work": 0,
        "p_node_id": "p0",
        "d_node_id": "d0",
        "emitted_output_tokens": 2,
        "terminal": AttemptTerminal.COMPLETED,
    }
    values[field] = value
    with pytest.raises(ValueError, match="plain.*integer|terminal"):
        AttemptExecutionSpec(**values)


def test_prefix_and_kvs_byte_fields_require_plain_integers() -> None:
    with pytest.raises(ValueError, match="plain positive integers"):
        KernelRequestSpec(request("r"), ("K",), (10.0,))
    with pytest.raises(ValueError, match="plain non-negative integer"):
        FrozenKernelCostModel(0.1, 0.1, True, 0.5)
