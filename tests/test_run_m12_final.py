from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import pytest

from prefill_cache_sim.m12_decode import (
    DecodeAdmissionConfig,
    DecodeAdmissionMode,
    DecodeCapacityPolicy,
    PrefixFamilyPredictor,
)
from prefill_cache_sim.m12_eviction import (
    CensusConfig,
    ClusterCacheCensus,
    EvictionMode,
    M12EvictionConfig,
    M12EvictionPolicy,
)
from prefill_cache_sim.m12_kernel import (
    CacheMutation,
    CausalKernel,
    CausalView,
    FrozenKernelCostModel,
    KernelConfig,
    KernelRequestSpec,
)
from prefill_cache_sim.m12_metrics import SERVICE_REGIMES, LogicalRequestSpec
from prefill_cache_sim.m12_placement import M12PlacementPolicy, PlacementMode
from scripts.run_m12_final import (
    ARRIVAL_SCALES,
    PRIMARY_STRATEGIES,
    CellResult,
    RssGridAborted,
    _attribution,
    _CacheDigestLedger,
    _causal_hit_ceiling,
    _crossovers,
    _DecisionLedgerPolicy,
    _DelayedCensus,
    _FinalEvictionPolicy,
    _first_decision_diff,
    _g12_3,
    _pareto,
    _retry_pressure_abort_fences,
    _RssWatchdog,
    build_cell_plan,
    execute_cell,
    placement_run_active,
    run_artifacts,
    with_visibility_delay,
)


def fake_result(cell) -> CellResult:
    value = 1 + cell.arrival_scale
    return CellResult(
        cell,
        offered_requests=3,
        offered_tokens=30,
        strict_goodput=value,
        strict_output_goodput=value,
        request_goodput=value,
        minimum_tier=0.9,
        jain=0.95,
        per_tier={"STANDARD": 1.0},
        queue_p95_normalized_work=2,
        token_hit_rate=0.5,
        waste_fraction=0,
        load_skew=1,
        kvs_work=0,
        p_utilization=0.2,
        d_utilization=0.3,
        p_to_d_debt=0,
        total_work=10,
        accounted_work=10,
        capacity_binding=cell.strategy in {"EVICTION_LRU", "CENSUS_EVICTION"},
    )


def test_plan_avoids_cartesian_sensitivity_and_nonbinding_eviction() -> None:
    binding = {("MIXED", 1.5)}
    plan = build_cell_plan(binding)
    primary = [cell for cell in plan if cell.category == "PRIMARY"]
    sensitivity = [cell for cell in plan if cell.category == "SENSITIVITY"]
    assert len([c for c in primary if c.strategy in PRIMARY_STRATEGIES]) == 45
    assert {
        (c.strategy, c.regime, c.arrival_scale)
        for c in primary
        if "EVICTION" in c.strategy
    } == {
        ("EVICTION_LRU", "MIXED", 1.5),
        ("CENSUS_EVICTION", "MIXED", 1.5),
    }
    assert {(c.strategy, c.arrival_scale) for c in sensitivity} == {
        ("DECODE_NO_GATE", 1.5),
        ("DECODE_ORACLE", 1.5),
        ("DECODE_ORACLE_NOISED", 1.5),
    }
    assert {cell.arrival_scale for cell in primary} == set(ARRIVAL_SCALES)


def test_process_guard_detects_only_active_placement_runner() -> None:
    assert placement_run_active(["123 python scripts/run_m12_placement.py"])
    assert not placement_run_active(
        ["123 python scripts/run_m12_final.py", "grep placement"]
    )


def test_artifacts_are_atomic_deterministic_and_load_workload_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    loads = 0

    def loader(_path):
        nonlocal loads
        loads += 1
        return ("frozen-workload", {"trace_sha256": "abc", "record_count": 3})

    left = tmp_path / "left"
    right = tmp_path / "right"
    binding = {("MIXED", 1.5)}
    first = run_artifacts(
        tmp_path / "trace.jsonl",
        left,
        executor=lambda _workload, cell: fake_result(cell),
        workload_loader=loader,
        binding_cells=binding,
        rss_reader=lambda: 0,
    )
    second = run_artifacts(
        tmp_path / "trace.jsonl",
        right,
        executor=lambda _workload, cell: fake_result(cell),
        workload_loader=loader,
        binding_cells=binding,
        rss_reader=lambda: 0,
    )
    assert loads == 2  # exactly once per complete experiment, never per cell
    plan = build_cell_plan(binding)
    expected_progress = [
        f"[{index:03d}/{len(plan):03d}] {cell.cell_id}"
        for index, cell in enumerate(plan, start=1)
    ]
    assert capsys.readouterr().err.splitlines() == expected_progress * 2
    assert first == second
    manifest = json.loads((left / "MANIFEST.json").read_text())
    assert {
        "primary.csv",
        "constraints.csv",
        "explanation.csv",
        "sensitivities.csv",
        "pareto.json",
        "attribution.json",
        "falsification/visibility-delay.json",
        "crossovers.csv",
        "failures.csv",
        "gates/g12-3.json",
        "gates/g12-4.json",
        "diagnostics/rss-watchdog.json",
        "contract.json",
        "config.json",
        "provenance.json",
    } == set(manifest["files"])
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((left / name).read_bytes()).hexdigest() == digest
        assert (left / name).read_bytes() == (right / name).read_bytes()
    constraints_header = (left / "constraints.csv").read_text().splitlines()[0]
    assert "queue_p95_normalized_work" in constraints_header
    assert "p99" not in constraints_header
    contract = json.loads((left / "contract.json").read_text())
    assert contract["schema_version"] == "m12-final-contract-v2"
    mixed = next(item for item in contract["regimes"] if item["regime_id"] == "MIXED")
    assert mixed["prefill_token_work"] == 0.06
    assert mixed["decode_token_work"] == 1.0
    assert mixed["kvs_token_work"] == 0.01
    assert mixed["kvs_bytes_per_token"] == 65_536
    provenance = json.loads((left / "provenance.json").read_text())
    assert set(provenance["imported_script_sha256"]) == {
        "scripts/run_m12_final.py",
        "scripts/run_m12_placement.py",
    }
    gate = json.loads((left / "gates/g12-3.json").read_text())
    assert all(
        not cell["passed"] and not cell["deployable"]
        for cell in gate["cells"]
        if "ORACLE" in cell["strategy"]
    )


def test_rss_watchdog_interrupts_cell_stops_grid_and_cleans_thread(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def blocked(_workload, cell):
        calls.append(cell.cell_id)
        while True:
            time.sleep(0.01)

    previous = signal.getsignal(signal.SIGUSR1)

    def prior_handler(_signum, _frame):
        return None

    signal.signal(signal.SIGUSR1, prior_handler)
    try:
        with pytest.raises(RssGridAborted, match="RSS hard limit exceeded"):
            run_artifacts(
                tmp_path / "trace.jsonl",
                tmp_path / "out",
                executor=blocked,
                workload_loader=lambda _path: ("workload", {"trace_sha256": "x"}),
                binding_cells=set(),
                rss_reader=lambda: 300,
                rss_soft_limit_bytes=100,
                rss_hard_limit_bytes=200,
                rss_poll_interval_seconds=0.001,
            )
        assert signal.getsignal(signal.SIGUSR1) is prior_handler
    finally:
        signal.signal(signal.SIGUSR1, previous)
    assert len(calls) == 1
    diagnostic = json.loads(
        (tmp_path / "out/diagnostics/rss-watchdog.json").read_text()
    )
    assert diagnostic["soft_limit_exceeded"] is True
    assert diagnostic["hard_limit_exceeded"] is True
    assert diagnostic["aborted_cell_id"] == calls[0]
    failures = (tmp_path / "out/failures.csv").read_text()
    assert "_RssHardLimitExceeded" in failures
    assert not (tmp_path / "out/MANIFEST.json").exists()
    assert not (tmp_path / "out/contract.json").exists()
    abort_manifest = json.loads((tmp_path / "out/ABORT_MANIFEST.json").read_text())
    assert abort_manifest["status"] == "ABORTED"
    assert json.loads((tmp_path / "out/ABORTED.json").read_text())["status"] == (
        "ABORTED_RSS_HARD_LIMIT"
    )
    assert not any(
        thread.name.startswith("m12-rss-watchdog-")
        for thread in threading.enumerate()
    )


def test_rss_watchdog_soft_limit_reports_without_false_hard_trip(
    tmp_path: Path,
) -> None:
    artifacts = run_artifacts(
        tmp_path / "trace.jsonl",
        tmp_path / "out",
        executor=lambda _workload, cell: (time.sleep(0.002), fake_result(cell))[1],
        workload_loader=lambda _path: ("workload", {"trace_sha256": "x"}),
        binding_cells=set(),
        rss_reader=lambda: 150,
        rss_soft_limit_bytes=100,
        rss_hard_limit_bytes=200,
        rss_poll_interval_seconds=0.001,
    )
    diagnostic = json.loads(artifacts["diagnostics/rss-watchdog.json"])
    assert diagnostic["soft_limit_exceeded"] is True
    assert diagnostic["hard_limit_exceeded"] is False
    assert diagnostic["aborted_cell_id"] is None
    assert not any(
        thread.name.startswith("m12-rss-watchdog-")
        for thread in threading.enumerate()
    )


def test_rss_watchdog_also_guards_visibility_reruns(tmp_path: Path) -> None:
    calls: list[str] = []

    def execute(_workload, cell):
        calls.append(cell.cell_id)
        time.sleep(0.002)
        return fake_result(cell)

    def reader() -> int:
        return 300 if len(calls) > len(build_cell_plan(set())) else 0

    with pytest.raises(RssGridAborted):
        run_artifacts(
            tmp_path / "trace.jsonl",
            tmp_path / "out",
            executor=execute,
            workload_loader=lambda _path: ("workload", {"trace_sha256": "x"}),
            binding_cells=set(),
            rss_reader=reader,
            rss_soft_limit_bytes=100,
            rss_hard_limit_bytes=200,
            rss_poll_interval_seconds=0.001,
        )
    assert len(calls) == len(build_cell_plan(set())) + 1
    diagnostic = json.loads(
        (tmp_path / "out/diagnostics/rss-watchdog.json").read_text()
    )
    assert diagnostic["hard_limit_exceeded"] is True
    assert not (tmp_path / "out/MANIFEST.json").exists()


def test_guarded_real_visibility_rerun_preserves_requested_delay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.run_m12_final as runner

    observed_delays: list[float] = []

    def fake_execute(_workload, cell, *, visibility_delay_work=0.0):
        observed_delays.append(visibility_delay_work)
        return replace(
            fake_result(cell), visibility_delay_work=visibility_delay_work
        )

    monkeypatch.setattr(runner, "execute_cell", fake_execute)
    monkeypatch.setattr(
        runner, "_build_artifacts", lambda *args, **kwargs: {"MANIFEST.json": b"{}"}
    )
    runner.run_artifacts(
        tmp_path / "trace.jsonl",
        tmp_path / "out",
        workload_loader=lambda _path: ("workload", {"trace_sha256": "x"}),
        binding_cells=set(),
        rss_reader=lambda: 0,
    )
    assert 0.0 in observed_delays
    assert 1.0 in observed_delays


def test_artifact_directory_transitions_remove_stale_success_and_abort(
    tmp_path: Path,
) -> None:
    output = tmp_path / "out"

    def success() -> None:
        run_artifacts(
            tmp_path / "trace.jsonl",
            output,
            executor=lambda _workload, cell: fake_result(cell),
            workload_loader=lambda _path: ("workload", {"trace_sha256": "x"}),
            binding_cells=set(),
            rss_reader=lambda: 0,
        )

    success()
    assert (output / "MANIFEST.json").exists()

    with pytest.raises(RssGridAborted):
        run_artifacts(
            tmp_path / "trace.jsonl",
            output,
            executor=lambda _workload, _cell: time.sleep(10),
            workload_loader=lambda _path: ("workload", {"trace_sha256": "x"}),
            binding_cells=set(),
            rss_reader=lambda: 300,
            rss_soft_limit_bytes=100,
            rss_hard_limit_bytes=200,
            rss_poll_interval_seconds=0.001,
        )
    assert set(path.name for path in output.iterdir()) == {
        "ABORTED.json",
        "ABORT_MANIFEST.json",
        "diagnostics",
        "failures.csv",
    }
    assert not (output / "MANIFEST.json").exists()

    success()
    assert (output / "MANIFEST.json").exists()
    assert not (output / "ABORT_MANIFEST.json").exists()
    assert not (output / "ABORTED.json").exists()


def test_atomic_directory_swap_restores_old_tree_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.run_m12_final as runner

    output = tmp_path / "out"
    output.mkdir()
    (output / "old.txt").write_text("old")
    real_replace = os.replace
    calls = 0

    def fail_staging_swap(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected staging swap failure")
        return real_replace(source, destination)

    monkeypatch.setattr(runner.os, "replace", fail_staging_swap)
    with pytest.raises(RuntimeError, match="injected staging swap failure"):
        runner._write_atomic(output, {"MANIFEST.json": b"new"})
    assert {path.name for path in output.iterdir()} == {"old.txt"}
    assert (output / "old.txt").read_text() == "old"
    assert not list(tmp_path.glob(".m12-final-*"))
    assert not list(tmp_path.glob(".m12-previous-*"))


def test_atomic_directory_swap_keeps_commit_on_backup_cleanup_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.run_m12_final as runner

    output = tmp_path / "out"
    output.mkdir()
    (output / "old.txt").write_text("old")
    real_rmtree = shutil.rmtree
    interrupted = False

    def interrupt_backup_cleanup(path, *args, **kwargs):
        nonlocal interrupted
        candidate = Path(path)
        if candidate.name.startswith(".m12-previous-") and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("injected cleanup interrupt")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(runner.shutil, "rmtree", interrupt_backup_cleanup)
    with pytest.raises(KeyboardInterrupt, match="injected cleanup interrupt"):
        runner._write_atomic(output, {"MANIFEST.json": b"new"})
    assert {path.name for path in output.iterdir()} == {"MANIFEST.json"}
    assert (output / "MANIFEST.json").read_bytes() == b"new"
    assert not list(tmp_path.glob(".m12-final-*"))
    assert not list(tmp_path.glob(".m12-previous-*"))


def test_atomic_directory_swap_never_restores_partially_deleted_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.run_m12_final as runner

    output = tmp_path / "out"
    output.mkdir()
    (output / "old.txt").write_text("old")
    real_rmtree = shutil.rmtree
    interrupted = False

    def partially_delete_then_interrupt(path, *args, **kwargs):
        nonlocal interrupted
        candidate = Path(path)
        if candidate.name.startswith(".m12-previous-") and not interrupted:
            interrupted = True
            (candidate / "old.txt").unlink()
            raise KeyboardInterrupt("injected partial cleanup interrupt")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(runner.shutil, "rmtree", partially_delete_then_interrupt)
    with pytest.raises(KeyboardInterrupt, match="injected partial cleanup interrupt"):
        runner._write_atomic(output, {"MANIFEST.json": b"new"})
    assert {path.name for path in output.iterdir()} == {"MANIFEST.json"}
    assert (output / "MANIFEST.json").read_bytes() == b"new"
    assert not list(tmp_path.glob(".m12-final-*"))
    assert not list(tmp_path.glob(".m12-previous-*"))


def test_rss_watchdog_restores_handler_when_thread_start_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = signal.getsignal(signal.SIGUSR1)

    def prior_handler(_signum, _frame):
        return None

    def fail_start(_thread):
        raise RuntimeError("start failed")

    signal.signal(signal.SIGUSR1, prior_handler)
    monkeypatch.setattr(threading.Thread, "start", fail_start)
    diagnostic = {
        "last_rss_bytes": 0,
        "peak_rss_bytes": 0,
        "soft_limit_exceeded": False,
    }
    try:
        with (
            pytest.raises(RuntimeError, match="start failed"),
            _RssWatchdog(
                "cell",
                reader=lambda: 0,
                soft_limit_bytes=100,
                hard_limit_bytes=200,
                poll_interval_seconds=0.01,
                diagnostic=diagnostic,
            ),
        ):
            pass
        assert signal.getsignal(signal.SIGUSR1) is prior_handler
    finally:
        signal.signal(signal.SIGUSR1, previous)


def test_binding_probes_have_separate_progress_and_only_binding_cache_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.run_m12_final as runner

    monkeypatch.setattr(
        runner, "_build_artifacts", lambda *args, **kwargs: {"MANIFEST.json": b"{}"}
    )
    monkeypatch.setattr(runner, "_visibility_delay_audit", lambda *args, **kwargs: {})
    workload = (
        KernelRequestSpec(
            LogicalRequestSpec("a", "t", "STANDARD", 0, 1, 1), ("K",), (1,)
        ),
        KernelRequestSpec(
            LogicalRequestSpec("b", "t", "STANDARD", 1, 1, 1), ("K",), (1,)
        ),
    )
    calls: list[str] = []

    def execute(_workload, cell):
        calls.append(cell.cell_id)
        result = fake_result(cell)
        return replace(
            result,
            token_hit_rate=(0.4 if cell.regime == "MIXED" else 0.5),
        )

    run_artifacts(
        tmp_path / "trace.jsonl",
        tmp_path / "out",
        executor=execute,
        workload_loader=lambda _path: (workload, {"trace_sha256": "x"}),
    )
    progress = capsys.readouterr().err.splitlines()
    assert progress[:15] == [
        f"[probe {index}/15] {regime}-{scale:.1f}x"
        for index, (regime, scale) in enumerate(
            (
                (item.regime_id.value, scale)
                for item in SERVICE_REGIMES
                for scale in ARRIVAL_SCALES
            ),
            start=1,
        )
    ]
    assert calls.count("MIXED-0.8x-EVICTION_LRU") == 1
    assert calls.count("COMPUTE_BOUND-0.8x-EVICTION_LRU") == 1
    source = inspect.getsource(run_artifacts)
    assert source.index("if is_binding:") < source.index(
        "cached_results[probe.cell_id]"
    )


def test_cache_digest_is_order_independent_and_hashseed_stable(tmp_path: Path) -> None:
    program = r'''
from scripts.run_m12_final import _CacheDigestLedger
from prefill_cache_sim.m12_kernel import CacheMutation
import sys
keys = tuple(sys.argv[1].split(','))
ledger = _CacheDigestLedger()
ledger.apply('p0', keys, frozenset(), CacheMutation(True))
print(ledger.snapshot('p0'))
'''
    outputs = []
    for seed, keys in (("1", "A,B,😀"), ("7", "😀,B,A")):
        completed = subprocess.run(
            [sys.executable, "-c", program, keys],
            check=True,
            capture_output=True,
            text=True,
            env=dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH="src:."),
        )
        outputs.append(completed.stdout.strip())
    assert outputs[0] == outputs[1]


def test_cache_digest_updates_without_raw_snapshot_or_occupancy_scan() -> None:
    ledger = _CacheDigestLedger()
    ledger.apply("p0", ("A", "B"), frozenset(), CacheMutation(True))
    before = ledger.snapshot("p0")
    ledger.apply(
        "p0", ("C",), frozenset({"A", "B"}), CacheMutation(True, ("A",))
    )
    after = ledger.snapshot("p0")
    assert before[0] == after[0] == 2
    assert before[1] != after[1]
    source = inspect.getsource(_CacheDigestLedger)
    assert "sorted(" not in source
    assert "set(resident" not in source
    assert "census_input_keys" not in inspect.getsource(_DecisionLedgerPolicy)


def test_census_age_is_measured_from_actual_refresh_at_decision() -> None:
    workload = tuple(
        KernelRequestSpec(
            LogicalRequestSpec(str(index), "t", "STANDARD", index * 0.2, 1, 1),
            (chr(ord("A") + index),),
            (1,),
        )
        for index in range(8)
    )
    cell = next(
        item
        for item in build_cell_plan({("MIXED", 1.5)})
        if item.strategy == "CENSUS_EVICTION"
    )
    result = execute_cell(workload, cell, visibility_delay_work=2)
    decisions = [
        json.loads(item)
        for item in result.decision_log
        if json.loads(item)["decision_kind"] == "EVICTION"
    ]
    measured = [
        item["decision_time_work"] - item["census_refreshed_at_work"]
        for item in decisions
        if item["census_refreshed_at_work"] is not None
    ]
    assert measured
    assert [
        item["census_age_work"]
        for item in decisions
        if item["census_refreshed_at_work"] is not None
    ] == pytest.approx(measured)
    assert result.census_age_work == pytest.approx(max(measured))
    assert result.census_age_work != result.visibility_delay_work


def test_same_victims_different_census_digest_is_first_divergence() -> None:
    def record(digest: str) -> str:
        return json.dumps(
            {
                "logical_id": "r",
                "attempt_index": 0,
                "decision_kind": "EVICTION",
                "decision_time_work": 1,
                "sequence": 0,
                "victims": ["V"],
                "census_input_count": 2,
                "census_input_digest": digest,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    divergence = _first_decision_diff((record("a"),), (record("b"),))
    assert divergence is not None
    assert divergence["before"]["victims"] == divergence["after"]["victims"]
    assert divergence["before"]["census_input_digest"] != divergence["after"][
        "census_input_digest"
    ]


def test_runner_refuses_empty_or_nonfinite_results(tmp_path: Path) -> None:
    bad = fake_result(build_cell_plan(set())[0])
    with pytest.raises(ValueError, match="finite"):
        replace(bad, strict_goodput=float("nan"))


@pytest.mark.parametrize(
    "strategy",
    [
        "BASELINE",
        "PRICED_SPILL",
        "DECODE_CAUSAL",
        "DECODE_ORACLE_NOISED",
        "EVICTION_LRU",
        "CENSUS_EVICTION",
    ],
)
def test_real_executor_smoke_uses_fixed_workload(strategy: str) -> None:
    workload = tuple(
        KernelRequestSpec(
            LogicalRequestSpec(identity, "tenant", "STANDARD", arrival, 1, 2),
            (key,),
            (1,),
        )
        for identity, key, arrival in (("a", "A", 0), ("b", "A", 3))
    )
    result = execute_cell(
        workload,
        build_cell_plan({("MIXED", 1.5)})[
            next(
                index
                for index, cell in enumerate(build_cell_plan({("MIXED", 1.5)}))
                if cell.regime == "MIXED"
                and cell.strategy == strategy
                and (cell.arrival_scale == 1.5 or "ORACLE" in strategy)
            )
        ],
    )
    assert result.offered_requests == 2
    assert result.offered_tokens == 6
    assert result.total_work == pytest.approx(result.accounted_work)
    assert result.p_to_d_debt == 0


def test_predeclared_causal_cell_exercises_abort_retry_pressure() -> None:
    workload = tuple(
        KernelRequestSpec(
            LogicalRequestSpec(
                f"retry-{index}", "tenant", "STANDARD", 0, 2, 20
            ),
            (f"K-{index}",),
            (2,),
        )
        for index in range(33)
    )
    cell = next(
        cell
        for cell in build_cell_plan(set())
        if cell.regime == "MIXED"
        and cell.arrival_scale == 1.5
        and cell.strategy == "DECODE_CAUSAL"
    )
    result = execute_cell(workload, cell)
    assert result.attempt_count > result.offered_requests
    assert result.retry_count == result.attempt_count - result.offered_requests
    assert result.decode_report is not None
    assert result.decode_report.preemptions > 0
    assert result.congestion_action == "GATED_DP"
    assert result.gated_retry_count > 0
    no_gate_cell = next(
        item
        for item in build_cell_plan(set())
        if item.regime == "MIXED" and item.strategy == "DECODE_NO_GATE"
    )
    no_gate = execute_cell(workload, no_gate_cell)
    gate = _g12_3(
        (no_gate, result), expected_cells=(no_gate_cell, cell)
    )
    assert gate["retry_pressure_covered"] is True

    configured_but_unproven = replace(result, gated_retry_count=0)
    assert _g12_3(
        (no_gate, configured_but_unproven), expected_cells=(no_gate_cell, cell)
    )["retry_pressure_covered"] is False

    defer_result = replace(result, congestion_action="DEFER")
    assert _g12_3(
        (no_gate, defer_result), expected_cells=(no_gate_cell, cell)
    )["retry_pressure_covered"] is False

    wrong_cell_result = replace(
        result,
        cell=replace(cell, regime="COMPUTE_BOUND"),
    )
    assert _g12_3(
        (no_gate, wrong_cell_result),
        expected_cells=(no_gate_cell, wrong_cell_result.cell),
    )["retry_pressure_covered"] is False


def test_retry_fences_are_predeclared_without_future_output_labels() -> None:
    workload = (
        KernelRequestSpec(
            LogicalRequestSpec("short", "t", "STANDARD", 0, 1, 1),
            ("A",),
            (1,),
        ),
        KernelRequestSpec(
            LogicalRequestSpec("long", "t", "STANDARD", 0, 1, 100),
            ("B",),
            (1,),
        ),
    )
    fences = _retry_pressure_abort_fences(workload, enabled=True)
    assert set(fences) == {"short", "long"}
    assert all(fence.allows_abort for fence in fences.values())
    assert _retry_pressure_abort_fences(workload, enabled=False) == {}


def test_capacity_binding_ceiling_is_causal_contiguous_prefix() -> None:
    workload = (
        KernelRequestSpec(
            LogicalRequestSpec("a", "t", "STANDARD", 0, 2, 1),
            ("A", "B"),
            (1, 1),
        ),
        KernelRequestSpec(
            LogicalRequestSpec("b", "t", "STANDARD", 1, 2, 1),
            ("X", "B"),
            (1, 1),
        ),
        KernelRequestSpec(
            LogicalRequestSpec("c", "t", "STANDARD", 2, 2, 1),
            ("A", "B"),
            (1, 1),
        ),
    )
    assert _causal_hit_ceiling(workload) == pytest.approx(2 / 6)


def test_visibility_delay_is_an_independent_policy_knowledge_rerun() -> None:
    cell = build_cell_plan(set())[0]

    def visible(_workload, _cell):
        return replace(
            fake_result(cell),
            decision_log=("r0:p0:0", "r1:p1:1"),
            decision_fingerprint="fixed",
            census_age_work=2,
        )

    zero = with_visibility_delay(visible, 0)("workload", cell)
    delayed = with_visibility_delay(visible, 5)("workload", cell)
    assert zero is not delayed
    assert delayed.decision_log == zero.decision_log
    assert delayed.decision_fingerprint == zero.decision_fingerprint
    # Generic executors are rerun, never patched to manufacture older metadata.
    assert delayed.census_age_work == 2
    assert delayed.census_age_work >= zero.census_age_work
    assert delayed.offered_requests == zero.offered_requests
    assert delayed.offered_tokens == zero.offered_tokens
    assert delayed.total_work == delayed.accounted_work


def test_non_census_strategy_never_reports_census_age() -> None:
    cell = next(item for item in build_cell_plan(set()) if item.strategy == "BASELINE")
    assert execute_cell((), cell).census_age_work is None


def test_real_visibility_delay_changes_only_policy_knowledge() -> None:
    cell = next(
        item
        for item in build_cell_plan(set())
        if item.strategy == "DECODE_CAUSAL" and item.arrival_scale == 0.8
    )
    workload = tuple(
        KernelRequestSpec(
            LogicalRequestSpec(identity, "t", "STANDARD", arrival, 1, output),
            ("family",),
            (1,),
        )
        for identity, arrival, output in (("a", 0, 4), ("b", 10, 20))
    )
    visible = execute_cell(workload, cell)
    delayed = execute_cell(workload, cell, visibility_delay_work=1_000)
    assert visible.offered_requests == delayed.offered_requests
    assert visible.offered_tokens == delayed.offered_tokens
    assert delayed.total_work == delayed.accounted_work
    visible_placement = [
        json.loads(item)
        for item in visible.decision_log
        if json.loads(item)["decision_kind"] == "PLACEMENT"
    ]
    delayed_placement = [
        json.loads(item)
        for item in delayed.decision_log
        if json.loads(item)["decision_kind"] == "PLACEMENT"
    ]
    assert visible_placement == delayed_placement
    assert visible.decision_fingerprint != delayed.decision_fingerprint


def test_first_divergence_aligns_ledger_by_key_not_tuple_position() -> None:
    def record(logical_id, kind, sequence, **extra):
        return json.dumps(
            {
                "logical_id": logical_id,
                "attempt_index": 0,
                "decision_kind": kind,
                "decision_time_work": sequence,
                "sequence": sequence,
                **extra,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    placement_a = record("a", "PLACEMENT", 0, node="p0")
    placement_b = record("b", "PLACEMENT", 2, node="p1")
    inserted_decode = record("a", "DECODE", 1, admission_action="ADMIT")
    divergence = _first_decision_diff(
        (placement_a, placement_b),
        (placement_a, inserted_decode, placement_b),
    )
    assert divergence is not None
    assert divergence["decision_layer"] == "DECODE"
    assert divergence["change_type"] == "ADDED"
    assert divergence["key"] == ["a", 0, "DECODE"]


def test_delayed_census_remove_cancels_pending_add_and_refresh() -> None:
    census = _DelayedCensus(CensusConfig(8, 100), 5)
    census.observe("k", "c", "p0", at_work=1, recovery_work=1)
    census.remove("k", "c", "p0")
    assert census.lookup("k", "c", now_work=10) is None

    census.observe("k", "c", "p0", at_work=10, recovery_work=1)
    assert census.lookup("k", "c", now_work=15) is not None
    census.observe("k", "c", "p0", at_work=16, recovery_work=2)
    census.remove("k", "c", "p0")
    assert census.lookup("k", "c", now_work=30) is None


def test_delayed_census_age_ignores_future_pending_refresh() -> None:
    census = _DelayedCensus(CensusConfig(8, 100), 5)
    census.observe("k", "c", "p0", at_work=0, recovery_work=1)
    assert census.lookup("k", "c", now_work=5) is not None
    assert census.latest_visible_snapshot_work == 0

    census.observe("k", "c", "p0", at_work=10, recovery_work=2)
    assert census.lookup("k", "c", now_work=12) is not None
    assert 12 - census.latest_visible_snapshot_work == 12

    assert census.lookup("k", "c", now_work=15) is not None
    assert census.latest_visible_snapshot_work == 10
    assert 15 - census.latest_visible_snapshot_work == 5


def test_eviction_mutation_uses_immutable_resident_without_full_copy() -> None:
    source = inspect.getsource(M12EvictionPolicy.cache_mutation)
    assert "set(view.cache_by_node" not in source
    assert "resident | additions" not in source

    class CountingLru(OrderedDict[str, None]):
        iterations = 0

        def __iter__(self):
            for key in super().__iter__():
                self.iterations += 1
                yield key

    keys = tuple(f"K-{index}" for index in range(10_000))
    resident = frozenset(keys)
    request = KernelRequestSpec(
        LogicalRequestSpec("new", "t", "STANDARD", 0, 1, 1),
        ("NEW",),
        (1,),
    )
    placement = M12PlacementPolicy(
        PlacementMode.PRICED_SPILL,
        FrozenKernelCostModel(1, 0, 0, 1),
        kvs_enabled=False,
    )
    policy = M12EvictionPolicy(
        placement,
        M12EvictionConfig(EvictionMode.LRU, len(keys), 1, 0, {}),
        ClusterCacheCensus(CensusConfig(len(keys) + 1, 100)),
        cache_key_cohorts={**dict.fromkeys(keys, "c"), "NEW": "c"},
    )
    lru = CountingLru.fromkeys(keys)
    policy._lru["p0"] = lru
    view = CausalView(
        0,
        resident,
        {"p0": 0},
        {"d0": 0},
        {"p0": resident},
        {"p0": 0},
    )
    attempt = next(iter(placement.plan_attempts(request, view)))
    mutation = policy.cache_mutation(request, attempt, view)
    assert mutation.evict_keys == (keys[0],)
    assert lru.iterations == 1


def test_delayed_census_same_time_remove_then_add_is_deterministic() -> None:
    census = _DelayedCensus(CensusConfig(8, 100), 5)
    census.observe("k", "c", "p0", at_work=1, recovery_work=1)
    census.remove("k", "c", "p0")
    census.observe("k", "c", "p0", at_work=1, recovery_work=3)
    entry = census.lookup("k", "c", now_work=6)
    assert entry is not None
    assert entry.recovery_work_by_holder["p0"] == 3


def test_frozen_gate_rejects_tier_set_or_two_percent_regression() -> None:
    plan = build_cell_plan(set())
    baseline_cell = next(item for item in plan if item.strategy == "BASELINE")
    candidate_cell = next(
        item
        for item in plan
        if item.strategy == "PRICED_SPILL"
        and item.regime == baseline_cell.regime
        and item.arrival_scale == baseline_cell.arrival_scale
    )
    baseline = replace(fake_result(baseline_cell), per_tier={"A": 0.9, "B": 0.9})
    wrong_set = replace(fake_result(candidate_cell), per_tier={"A": 0.9})
    regression = replace(fake_result(candidate_cell), per_tier={"A": 0.879, "B": 0.9})
    for candidate in (wrong_set, regression):
        assert (
            candidate.cell.cell_id
            not in _pareto((baseline, candidate))["groups"][0]["frontier_cell_ids"]
        )
        assert _crossovers((baseline, candidate)) == []


def test_missing_entire_expected_group_is_incomplete_everywhere() -> None:
    binding = {("MIXED", 1.5)}
    expected = build_cell_plan(binding)
    assert _pareto((), expected_cells=expected)["overall_verdict"] == "INCOMPLETE"
    assert _attribution((), expected_cells=expected)["overall_verdict"] == "INCOMPLETE"


def test_artifacts_match_across_python_hash_seeds(tmp_path: Path) -> None:
    program = r"""
import hashlib
import sys
from pathlib import Path
from scripts.run_m12_final import CellResult, run_artifacts
def execute(_workload, cell):
    return CellResult(cell, 1, 2, 1, 1, 1, .9, .95, {'STANDARD': 1}, 0,
                      .5, 0, 1, 0, .1, .1, 0, 1, 1, False)
run_artifacts(Path('unused'), Path(sys.argv[1]), executor=execute,
              workload_loader=lambda _: ('w', {'trace_sha256': 'x'}),
              binding_cells=set(), rss_reader=lambda: 0)
print(hashlib.sha256((Path(sys.argv[1]) / 'MANIFEST.json').read_bytes()).hexdigest())
"""
    digests = []
    for seed in ("1", "3"):
        environment = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=".")
        completed = subprocess.run(
            [sys.executable, "-c", program, str(tmp_path / seed)],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        digests.append(completed.stdout.strip())
    assert digests[0] == digests[1]


def test_eviction_composite_preserves_decode_ledger_and_predictor_lifecycle() -> None:
    cost = FrozenKernelCostModel(0.1, 0, 0, 0.5)
    config = KernelConfig(0, 100, ("p0",), ("d0",), {"STANDARD": 100}, 8, cost)
    workload = tuple(
        KernelRequestSpec(
            LogicalRequestSpec(identity, "t", "STANDARD", arrival, 1, output),
            ("family",),
            (1,),
        )
        for identity, arrival, output in (
            ("a", 0, 4),
            ("b", 5, 6),
            ("c", 10, 8),
        )
    )

    def decode_policy():
        return DecodeCapacityPolicy(
            M12PlacementPolicy(PlacementMode.PRICED_SPILL, cost, kvs_enabled=False),
            DecodeAdmissionConfig(DecodeAdmissionMode.CAUSAL, 100),
            predictor=PrefixFamilyPredictor(default_output_tokens=2),
        )

    standalone = decode_policy()
    standalone_result = CausalKernel(config).run(workload, standalone)
    inner = decode_policy()
    composite = _FinalEvictionPolicy(
        inner,
        M12EvictionConfig(EvictionMode.LRU, 8, 0.1, 0, {}),
        ClusterCacheCensus(CensusConfig(8, 100)),
        cache_key_cohorts={"family": "cohort"},
    )
    composite_result = CausalKernel(config).run(workload, composite)
    assert [item.predicted_output_tokens for item in inner.decisions] == [
        item.predicted_output_tokens for item in standalone.decisions
    ]
    assert inner.ledger.actual_decode_work == standalone.ledger.actual_decode_work
    assert inner.ledger.p_to_d_debt_credits == 0
    assert composite_result.metrics == standalone_result.metrics


def test_single_switch_attribution_uses_first_diff_or_interaction() -> None:
    plan = build_cell_plan(set())
    baseline_cell = next(cell for cell in plan if cell.strategy == "BASELINE")
    priced_cell = next(
        cell
        for cell in plan
        if cell.strategy == "PRICED_SPILL"
        and cell.regime == baseline_cell.regime
        and cell.arrival_scale == baseline_cell.arrival_scale
    )
    decode_cell = next(
        cell
        for cell in plan
        if cell.strategy == "DECODE_CAUSAL"
        and cell.regime == baseline_cell.regime
        and cell.arrival_scale == baseline_cell.arrival_scale
    )
    baseline = replace(
        fake_result(baseline_cell),
        strict_goodput=1,
        decision_log=("same",),
        decision_fingerprint="a",
    )
    priced = replace(
        fake_result(priced_cell),
        strict_goodput=2,
        decision_log=("same",),
        decision_fingerprint="a",
    )
    decode = replace(
        fake_result(decode_cell),
        strict_goodput=3,
        decision_log=("different",),
        decision_fingerprint="b",
    )
    records = _attribution((baseline, priced, decode))["records"]
    by_switch = {record["single_switch"]: record for record in records}
    assert by_switch["PLACEMENT"]["classification"] == "INTERACTION"
    assert (
        by_switch["DECODE_CREDITS"]["classification"] == "ATTRIBUTED_FIRST_DIVERGENCE"
    )
    assert by_switch["DECODE_CREDITS"]["first_divergent_decision"]["index"] == 0
