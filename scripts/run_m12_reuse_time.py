#!/usr/bin/env python3
"""Generate deterministic M12.0 reuse-time artifacts."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prefill_cache_sim.config import git_provenance  # noqa: E402
from prefill_cache_sim.reuse_time import analyze_reuse_time  # noqa: E402
from prefill_cache_sim.trace import load_trace  # noqa: E402

TRACE_SHA256 = "b434f1816a707f4bac697235588184ebc374c9907cb981bb65fb0643471fe711"
CONFIG = {
    "schema_version": "m12-reuse-time-config-v1",
    "block_size_tokens": 512,
    "windows_ms": [10, 50, 100, 250, 1000, 5000, 60000, 300000],
    "gate": {
        "caliber": "token_weighted_causal_hot_excluded",
        "window_ms": 250,
        "threshold": 0.10,
    },
    "hot": {"percentile": 99.0, "minimum_references": 8, "refresh_requests": 256},
    "family_anchor_depth": 4,
}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _csv_bytes(rows: list[dict[str, object]]) -> bytes:
    output = io.StringIO(newline="")
    fields = list(rows[0])
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def _svg_bytes(rows: list[dict[str, object]]) -> bytes:
    width, height = 900, 430
    left, top, plot_w, plot_h = 95, 45, 735, 285
    windows = CONFIG["windows_ms"]
    colors = ["#4338ca", "#15803d", "#b45309", "#b91c1c"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<g font-family="-apple-system,PingFang SC,Arial" '
        'font-size="12" fill="#171a21">',
        '<text x="450" y="24" text-anchor="middle" font-size="15" '
        'font-weight="650">Reuse delta CDF（window ms）</text>',
        f'<path d="M{left} {top}V{top + plot_h}H{left + plot_w}" '
        'stroke="#8a93a3" fill="none"/>',
    ]
    for tick in range(0, 6):
        y = top + plot_h - tick * plot_h / 5
        parts.append(f'<path d="M{left} {y:.1f}H{left + plot_w}" stroke="#e4e7ec"/>')
        parts.append(
            f'<text x="{left - 12}" y="{y + 4:.1f}" '
            f'text-anchor="end">{tick * 20}%</text>'
        )
    for index, window in enumerate(windows):
        x = left + index * plot_w / (len(windows) - 1)
        parts.append(
            f'<text x="{x:.1f}" y="{top + plot_h + 24}" '
            f'text-anchor="middle">{window:g}</text>'
        )
    for row_index, row in enumerate(rows):
        points = []
        for index, window in enumerate(windows):
            x = left + index * plot_w / (len(windows) - 1)
            ratio = float(row[f"ratio_{window}ms"])
            y = top + plot_h * (1 - ratio)
            points.append(f"{x:.1f},{y:.1f}")
        color = colors[row_index]
        parts.append(
            f'<polyline points="{" ".join(points)}" fill="none" '
            f'stroke="{color}" stroke-width="2.5"/>'
        )
        legend_y = 360 + row_index * 17
        parts.append(
            f'<path d="M{left} {legend_y}h22" stroke="{color}" stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{left + 30}" y="{legend_y + 4}">{row["caliber"]}</text>'
        )
    parts.append("</g></svg>\n")
    return "".join(parts).encode()


def _write_atomic(output: Path, artifacts: dict[str, bytes]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as staging_name:
        staging = Path(staging_name)
        for name, content in artifacts.items():
            path = staging / name
            path.write_bytes(content)
            os.replace(path, output / name)


def main() -> int:
    trace_path = ROOT / "mooncake_trace.jsonl"
    trace = load_trace(trace_path, block_size_tokens=512)
    if trace.sha256 != TRACE_SHA256:
        raise RuntimeError(f"trace SHA mismatch: {trace.sha256}")
    analysis = analyze_reuse_time(trace)
    provenance = git_provenance(ROOT)
    config_bytes = _json_bytes(CONFIG)
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    rows: list[dict[str, object]] = []
    for row in analysis.rows:
        rendered: dict[str, object] = {
            "caliber": row.caliber,
            "denominator_weight": row.denominator_weight,
            "reused_events": row.reused_events,
        }
        for window in analysis.windows_ms:
            rendered[f"within_{window}ms_weight"] = row.within_window_weight[
                str(window)
            ]
            rendered[f"ratio_{window}ms"] = (
                f"{row.within_window_ratio[str(window)]:.12f}"
            )
        rows.append(rendered)
    report = {
        "truth_basis": "TRACE_TIMESTAMP",
        "trace_sha256": trace.sha256,
        "config_sha256": config_sha,
        "git_sha": provenance.sha,
        "git_dirty": provenance.dirty,
        "analysis": analysis.to_dict(),
        "notes": {
            "raw_coarrival_is_upper_bound": True,
            "pass_is_provisional": True,
            "mfu_claim_allowed": False,
        },
    }
    artifacts = {
        "config.json": config_bytes,
        "results.csv": _csv_bytes(rows),
        "report.json": _json_bytes(report),
        "reuse_cdf.svg": _svg_bytes(rows),
    }
    manifest = {
        "schema_version": "m12-reuse-time-manifest-v1",
        "files": {
            name: hashlib.sha256(content).hexdigest()
            for name, content in sorted(artifacts.items())
        },
    }
    artifacts["MANIFEST.json"] = _json_bytes(manifest)
    _write_atomic(ROOT / "results/m12-reuse-time", artifacts)
    print(
        json.dumps(
            {"gate_ratio": analysis.gate_ratio, "gate_verdict": analysis.gate_verdict},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
