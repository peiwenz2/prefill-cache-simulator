#!/usr/bin/env python3
# ruff: noqa: E501
"""Render the M4 CSV into a self-contained align-style HTML report."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results/m4/results.csv"
OUT = ROOT / "results/m4/phase-a-m4-report.html"
SUMMARY = ROOT / "results/m4/summary.json"


def pct(value: str | float) -> str:
    return f"{float(value) * 100:.2f}%"


def num(value: str | float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def table(headers: list[str], rows: list[list[object]]) -> str:
    head = "".join(f"<th>{item}</th>" for item in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{item}</td>" for item in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def main() -> int:
    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    a1 = [row for row in rows if row["case_id"].startswith("A1-")]
    a1.sort(key=lambda row: float(row["token_weighted_hit_rate"]), reverse=True)
    headline = {row["selector_id"]: row for row in a1}
    krows = sorted(
        (row for row in rows if row["case_id"].startswith("K-")),
        key=lambda row: int(row["prefix_anchor_k"]),
    )
    a2 = [row for row in rows if row["case_id"].startswith("A2-")]
    stale = [row for row in rows if row["case_id"].startswith("STALE-")]
    seed_groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["case_id"].startswith("SEED-"):
            seed_groups[row["selector_id"]].append(
                float(row["token_weighted_hit_rate"])
            )
    summary = {
        "schema": "phase-a-m4-summary-v1",
        "case_count": len(rows),
        "trace_sha256": rows[0]["trace_sha256"],
        "git_sha": rows[0]["git_sha"],
        "headline": {
            row["selector_id"]: {
                "token_weighted_hit_rate": float(row["token_weighted_hit_rate"]),
                "request_load_max_mean": float(row["request_load_max_mean"]),
                "queue_wait_normalized_ms_p95": float(
                    row["queue_wait_normalized_ms_p95"]
                ),
                "offered_load_ratio": float(row["offered_load_ratio"]),
                "replica_factor": float(row["replica_factor"]),
            }
            for row in a1
        },
        "decision": {
            "kvs_candidates": ["S4_SESSION_AFFINITY", "S3_GB_PREFIX_BUCKET"],
            "frontier_rule": "material expansion of S3/S4 hit-skew frontier",
            "hit_epsilon": 0.001,
            "stop_complex_selectors": ["S5_FLEXLB_TTFT", "S6_CALIBRATED_TTFT"],
        },
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    a1_table = table(
        ["Selector", "Token hit", "Request skew", "Queue p95＊", "Replica", "Fallback"],
        [
            [
                row["selector_id"],
                pct(row["token_weighted_hit_rate"]),
                num(row["request_load_max_mean"]),
                num(row["queue_wait_normalized_ms_p95"], 1),
                num(row["replica_factor"]),
                row["fallback_count"],
            ]
            for row in a1
        ],
    )
    k_table = table(
        ["k", "Token hit", "Request skew", "Fallback"],
        [
            [
                row["prefix_anchor_k"],
                pct(row["token_weighted_hit_rate"]),
                num(row["request_load_max_mean"]),
                row["fallback_count"],
            ]
            for row in krows
        ],
    )
    a2_table = table(
        [
            "Selector",
            "Eviction",
            "Token hit",
            "Evictions",
            "Rejected",
            "One-hit pollution",
        ],
        [
            [
                row["selector_id"],
                row["eviction_id"],
                pct(row["token_weighted_hit_rate"]),
                row["eviction_count"],
                row["rejected_placements"],
                row["one_hit_pollution"],
            ]
            for row in a2
        ],
    )
    s4_rows = [row for row in rows if row["case_id"].startswith(("S4-CAP", "S4-HOT"))]
    s4_table = table(
        ["Case", "Token hit", "Request skew", "Replica", "Affinity broken"],
        [
            [
                row["case_id"],
                pct(row["token_weighted_hit_rate"]),
                num(row["request_load_max_mean"]),
                num(row["replica_factor"]),
                row["affinity_broken"],
            ]
            for row in s4_rows
        ],
    )
    s5_rows = [
        row for row in rows if row["case_id"].startswith(("S5-DISCOUNT-", "S5-BAND-"))
    ]
    s5_table = table(
        ["Case", "Discount", "Band", "Token hit", "Request skew", "Queue p95＊"],
        [
            [
                row["case_id"],
                row["cache_discount"],
                row["similar_band_ratio"],
                pct(row["token_weighted_hit_rate"]),
                num(row["request_load_max_mean"]),
                num(row["queue_wait_normalized_ms_p95"], 1),
            ]
            for row in s5_rows
        ],
    )
    capacity_rows = [
        row for row in rows if row["case_id"].startswith(("CAP-TOTAL-", "CAP-PER-"))
    ]
    capacity_table = table(
        ["Case", "Token hit", "Request skew", "Replica", "ρ"],
        [
            [
                row["case_id"],
                pct(row["token_weighted_hit_rate"]),
                num(row["request_load_max_mean"]),
                num(row["replica_factor"]),
                num(row["offered_load_ratio"]),
            ]
            for row in capacity_rows
        ],
    )
    stale_table = table(
        ["Selector", "View delay", "Hit min–max", "Skew min–max"],
        [
            [
                selector,
                "0–5,000 normalized ms",
                f"{pct(min(float(r['token_weighted_hit_rate']) for r in stale if r['selector_id'] == selector))}–{pct(max(float(r['token_weighted_hit_rate']) for r in stale if r['selector_id'] == selector))}",
                f"{min(float(r['request_load_max_mean']) for r in stale if r['selector_id'] == selector):.3f}–{max(float(r['request_load_max_mean']) for r in stale if r['selector_id'] == selector):.3f}",
            ]
            for selector in (
                "S3_GB_PREFIX_BUCKET",
                "S4_SESSION_AFFINITY",
                "S5_FLEXLB_TTFT",
            )
        ],
    )
    bars = "".join(
        f'<g transform="translate(0,{index * 42})"><text x="0" y="18">{row["selector_id"].split("_")[0]}</text><rect x="145" y="2" width="{float(row["token_weighted_hit_rate"]) * 700:.1f}" height="22" rx="3" fill="#4338ca"/><text x="{155 + float(row["token_weighted_hit_rate"]) * 700:.1f}" y="18">{pct(row["token_weighted_hit_rate"])}</text></g>'
        for index, row in enumerate(a1)
    )
    seed_table = table(
        ["Selector", "Seeds", "Min", "Max", "Stddev"],
        [
            [
                selector,
                len(values),
                pct(min(values)),
                pct(max(values)),
                f"{__import__('statistics').pstdev(values):.8f}",
            ]
            for selector, values in sorted(seed_groups.items())
        ],
    )
    s3 = headline["S3_GB_PREFIX_BUCKET"]
    s4 = headline["S4_SESSION_AFFINITY"]
    s5 = headline["S5_FLEXLB_TTFT"]
    s6 = headline["S6_CALIBRATED_TTFT"]
    frontier_low = (
        float(s4["token_weighted_hit_rate"]),
        float(s4["request_load_max_mean"]),
    )
    frontier_high = (
        float(s3["token_weighted_hit_rate"]),
        float(s3["request_load_max_mean"]),
    )

    def expands_frontier(row: dict[str, str]) -> bool:
        hit = float(row["token_weighted_hit_rate"])
        skew = float(row["request_load_max_mean"])
        epsilon = 0.001
        if hit <= frontier_low[0] + epsilon:
            return skew < frontier_low[1]
        if hit >= frontier_high[0] + epsilon:
            return True
        fraction = (hit - frontier_low[0]) / (frontier_high[0] - frontier_low[0])
        frontier_skew = frontier_low[1] + fraction * (
            frontier_high[1] - frontier_low[1]
        )
        return skew < frontier_skew

    complex_rows = [s6, *s5_rows]
    stop_complex = not any(expands_frontier(row) for row in complex_rows)
    stop_text = (
        "S5／S6 没有 materially expand S3／S4 frontier，stop gate 触发"
        if stop_complex
        else "S5／S6 形成有效增益，stop gate 不触发"
    )
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Prefill Cache Simulator Phase A M4 结果</title><style>
:root{{--ink:#171a21;--bg:#f6f7f9;--panel:#fff;--line:#d8dee9;--muted:#667085;--accent:#4338ca;--accent-bg:#eef0fe;--ok:#15803d;--ok-bg:#e9f6ec;--warn:#b45309;--warn-bg:#fdf3e3;--bad:#b91c1c;--bad-bg:#fce9e9;--code:#f2f4f9;--code-ink:#2d3344}}*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}}header{{background:#fff;border-bottom:1px solid var(--line)}}.hero{{max-width:1180px;margin:auto;padding:38px 32px}}.pills{{display:flex;gap:8px;flex-wrap:wrap}}.pill{{background:var(--accent-bg);color:var(--accent);padding:4px 9px;border-radius:5px;font-size:12px}}h1{{font-size:34px;line-height:1.2;margin:14px 0 8px}}.lead{{color:var(--muted);max-width:900px;font-size:17px}}.layout{{max-width:1180px;margin:auto;display:grid;grid-template-columns:190px 1fr;gap:24px;padding:28px 32px}}nav{{position:sticky;top:18px;align-self:start}}nav a{{display:block;padding:6px 8px;color:var(--muted);text-decoration:none;border-left:2px solid transparent}}nav a.active{{color:var(--accent);border-color:var(--accent)}}section{{background:#fff;border:1px solid var(--line);border-radius:10px;padding:28px;margin-bottom:22px;scroll-margin-top:18px}}h2{{font-size:23px;border-bottom:1px solid var(--line);padding-bottom:9px}}h3{{margin-top:23px}}table{{width:100%;border-collapse:collapse;font-size:13px;margin:13px 0 18px}}th,td{{border:1px solid var(--line);padding:8px;text-align:left}}th{{background:#f8fafc}}.callout{{border-left:4px solid var(--accent);background:var(--accent-bg);padding:12px 14px;margin:14px 0}}.ok{{border-color:var(--ok);background:var(--ok-bg)}}.warn{{border-color:var(--warn);background:var(--warn-bg)}}code,pre{{background:var(--code);color:var(--code-ink)}}code{{padding:1px 4px;border-radius:3px}}pre{{padding:12px;border:1px solid var(--line);overflow:auto}}svg{{width:100%;height:auto;border:1px solid var(--line);border-radius:7px;padding:14px;background:#fbfcfe}}.caption,.fine{{font-size:12px;color:var(--muted)}}ul,ol{{padding-left:22px}}li{{margin:5px 0}}@media(max-width:850px){{.layout{{display:block;padding:16px}}nav{{display:none}}section{{padding:20px}}}}
</style></head><body><header><div class="hero"><div class="pills"><span class="pill">P1</span><span class="pill">M4 Complete</span><span class="pill">{len(rows)} full-trace cases</span><span class="pill">作者：张珮文</span></div><h1>Prefill Cache Simulator：Phase A M4 结果</h1><p class="lead">Mooncake 23,608 requests 上，headline operating point 的 offered load 保持在 ρ＜1。S4 与 S3 检验 cache affinity，S5／S6 在 cache term 可改变 argmin 的 stable regime 下执行 stop gate。</p><p class="fine">证据版本：git <code>{rows[0]["git_sha"]}</code>（dirty={rows[0]["git_dirty"]}）｜trace <code>{rows[0]["trace_sha256"]}</code>｜日期：2026-08-05</p></div></header><div class="layout"><nav><a href="#arch">1. 总体结论</a><a href="#detail">2. 策略结果</a><a href="#eval">3. Sensitivity</a><a href="#risk">4. 风险与边界</a><a href="#rollout">5. 下一步</a><a href="#ref">6. 引用</a></nav><main>
<section id="arch"><h2>1. 总体结论</h2><div class="callout"><b>先给结论</b><ul><li><b>S4</b>：{pct(s4["token_weighted_hit_rate"])} token hit，request skew {num(s4["request_load_max_mean"])}×。</li><li><b>S3</b>：{pct(s3["token_weighted_hit_rate"])} hit，request skew {num(s3["request_load_max_mean"])}×。</li><li><b>Stop gate</b>：{stop_text}；S5={pct(s5["token_weighted_hit_rate"])}，S6={pct(s6["token_weighted_hit_rate"])}。</li></ul></div><svg viewBox="0 0 760 310" role="img" aria-label="A1 token weighted hit rate">{bars}</svg><p class="caption">这张图回答：stable headline 参数下，哪个 selector 真正保留 shared prefix。</p>{a1_table}<p class="fine">＊Queue 数值是 normalized-work time，不是 calibrated production ms；ρ 见 summary，仅用于确认系统不持续积压。</p><div class="callout ok"><b>Replica mechanism</b>：Random／RR 把同一 prefix 分散到多节点，replica factor 上升；S3／S4 用 affinity ownership 收敛副本，因此同容量下保留更多 unique prefix。这个机制解释 hit gain，不把 queue wait 当独立证据。</div></section>
<section id="detail"><h2>2. 策略结果</h2><h3>2.1 PrefixAnchor 不是 k 越大越好</h3>{k_table}<div class="callout warn"><b>机制判断</b>：k sweep 必须同时看 hit 与 skew。S3 的 gain 有一部分来自 ownership concentration，不能只报最高 hit point。</div><h3>2.2 Eviction 排名依赖 selector</h3>{a2_table}<p class="fine">Second-hit admission 的 rejected placement 单独展示；低 eviction 不等于低 pollution。</p></section>
<section id="eval"><h2>3. Sensitivity 与 mechanism checks</h2><h3>3.1 Stale view</h3>{stale_table}<p class="fine">Delay 按 mean request service time 约 488 normalized ms 取 0.1×／1×／10×；中间档不敏感，0 与 10× 会改变决策。</p><h3>3.2 Capacity：fixed-total 与 fixed-per-node</h3>{capacity_table}<h3>3.3 S5 score tuning 与 frontier gate</h3>{s5_table}<div class="callout warn"><b>Gate 定义</b>：不是单轴比 S3 hit。用 S4（balanced）到 S3（cache ceiling）的 hit-skew 线段作为 baseline frontier，hit epsilon=0.1pp；候选必须落在线段左上方才算 material expansion。S5 best point 仍在线段右下方；S6 相对 S4 仅多约 0.03pp hit，但 skew／queue 更差。</div><h3>3.4 S4 family cap／hot-block exclusion</h3>{s4_table}<p class="fine">Hot percentile 90／95／99 相同，是 <code>hot_threshold≥8</code> floor 在绑定；cap=128／512 相同，说明 headline cap 已进入饱和区。</p><h3>3.5 Seed replay</h3>{seed_table}<p class="fine">结果由 CSV 实时计算。loss_probability=0 时 S3／S4／S5 不消费随机 stream，zero variance 是 deterministic contract check，不当成鲁棒性证据。</p></section>
<section id="risk"><h2>4. 风险与边界</h2>{table(["风险", "当前状态", "下一步"], [["Normalized service time", "只能做相对排序，不能叫 production TTFT", "Phase B 用 GPU/KVT calibration"], ["S4 conversation proxy", "仅依赖 online prefix family，不是真 session id", "接真实 session selector 对照"], ["S3 overload", "N=8 request skew 3.41×", "保留为 ceiling，不作为默认生产策略"], ["Eviction regret", "当前是 re-access-after-eviction count", "增加 bounded regret window"]])}<div class="callout warn"><b>诚实回答</b>：这批结果验证的是 LOCAL_ONLY cache placement，不证明 KVS、SLO goodput 或 Decode lease 的收益。后者从 M5 开始单独建账。</div></section>
<section id="rollout"><h2>5. 下一步</h2>{table(["顺序", "动作", "Gate"], [["1", "S4＋S3 接 SHARED_KVS topology", "local／remote／recompute 三类成本分开"], ["2", "用 GPU calibration 替换 normalized work", "TTFT 误差可解释"], ["3", "接真实 Global Batching／FlexLB session signal", "S4 proxy 与 production selector 排序一致"], ["4", "再进入 gated-PD／gated-DP 与 Decode lease", "主指标切 completed goodput＋fairness"]])}<ul><li>S4 是 production-shaped candidate。</li><li>S3 是 cache locality upper control。</li><li>S5／S6 按 M4 stop gate 暂停，不带入 KVS implementation。</li></ul></section>
<section id="ref"><h2>6. 引用</h2>{table(["Artifact", "位置"], [["Raw CSV", "results/m4/results.csv"], ["Machine summary", "results/m4/summary.json"], ["Runner", "scripts/run_m4.py"], ["Result schema", "src/prefill_cache_sim/experiment.py"], ["Design", "DESIGN-v3.md"], ["Mooncake trace SHA", rows[0]["trace_sha256"]], ["Code SHA", rows[0]["git_sha"]]])}</section></main></div><script>const links=[...document.querySelectorAll('nav a')],sections=[...document.querySelectorAll('section')];new IntersectionObserver(es=>es.forEach(e=>{{if(e.isIntersecting){{links.forEach(a=>a.classList.toggle('active',a.hash==='#'+e.target.id))}}}}),{{rootMargin:'-20% 0px -70%'}}).observe;sections.forEach(s=>new IntersectionObserver(es=>es.forEach(e=>{{if(e.isIntersecting)links.forEach(a=>a.classList.toggle('active',a.hash==='#'+e.target.id))}}),{{rootMargin:'-20% 0px -70%'}}).observe(s));</script></body></html>"""
    OUT.write_text(html, encoding="utf-8")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
