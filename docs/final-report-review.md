# Prefill Cache Simulation final report review

作者：张珮文

审阅对象：`/Users/zhangpeiwen/Downloads/Prefill+Cache+Simulation.pdf`

证据版本：`prefill-cache-sim@22e49bd`＋clean sizing artifacts

## 1. 总体判断

前四页的主线成立：先说明 trace，再比较 4-node local-cache selector，最后把问题扩展到 shared KVS 下的 resource sizing。但当前草稿混用了三套不能横比的实验合同：trace-only ceiling、M4 selector replay、M12 causal／sizing replay。给老板讲时必须先分开。

最重要的结果修订：旧 sizing 把 local-only 的 S4 routing 与 shared KVS 的 HYBRID routing 放在一起比较，因此 `3→2`、`7→4` 同时包含 selector 变化和 KVS 变化。修复为相同 HYBRID routing 后：80% floor 是 `2 vs 2`，95% floor 是 `4 vs 4`。旧资源节省 headline 撤回。

## 2. PDF 逐项修改

| 草稿内容 | 判断 | 建议直接替换 |
|---|---|---|
| 理论 token hit 为 57.04% | 错 | Infinite／no-expiry token-weighted ceiling 是 `115,733,271／202,791,701＝57.0700%`；57.04% 是 1,800s TTL 档。Block-ref ceiling 是 55.2551%。 |
| Ceiling 假设包含 LRU／hit renew | 不准确 | 无限容量下 LRU 不发生 eviction，因此不会改变 ceiling。Ceiling 的关键条件是 causal、continuous prefix、single global pool、no expiry。 |
| k=2 只匹配前 1,024 tokens | 错 | k=2 表示用第 2 个 chained block 作为 stable owner anchor。实际 cache hit 仍继续按完整 continuous prefix 查找，不会截断在 1,024 tokens。 |
| `hash % 4` | 过度简化 | 对 anchor key 做 SHA-256，取稳定整数后 `% N`；N 是 P node 数，本实验 N=4。 |
| overload 后只比较 idx＋1／idx＋2 | 不完整 | 最多检查 ring 上两个 secondary candidate；选 eligible 且不过载的 least-work。都不合格时 fail-open 到 cluster least-work。 |
| k=2 是 hit 与 load 都最好 | 错 | k=2 hit 最高 54.0086%，但 load max／mean=1.8218。k=16 的 load 更均衡 1.0549，但 hit 降至 51.9639%。k=2 是 trace-specific trade-off。 |
| SessionAffinity 超过 2 blocks 才匹配 | 错 | 实现是至少共享 2 blocks，包含恰好 2；结果 53.3366% 来自 online prefix-family proxy，不是真实 session_id。 |
| 中心化 master 的 0→5s delay 影响单请求 SLO | 未证明 | M4 没有 request SLO。可说 hit 从 53.0076% 降到 51.3302%，queue p95 从 1,220.23 增到 9,123.40 normalized work；不能写成 production ms／TTFT。 |
| 中心化 master 同时 assign P／D | 超出实现 | 当前数字只验证 P selection；联合 P／D routing 是后续架构设想。 |
| 大多数系统已有足够大的 Global KVS | 无证据 | 改成“下面把 shared KVS 作为 modeled topology”；不要声称行业普遍性或无限容量。 |
| 所有 P 完成后都可任意 get KV | 过度简化 | Shared visibility 仍受 capacity、transfer price、bandwidth、contention、timeout 和 failure 约束；当前 sizing 特意关闭 eviction，只隔离 topology。 |

## 3. 一条请求到底怎么算

当前 M12 sizing 使用 MIXED normalized cost model，没有 hardware calibration，也没有把 batching-saving curve应用到 kernel service time。

```text
arrival
  -> P queue wait
  -> prefill 未命中部分
  -> 可选 remote KV transfer
  -> D queue wait
  -> 完整 decode
  -> finish
```

公式：

```text
uncached_tokens = input_tokens - local_hit_tokens - remote_hit_tokens
P_work   = uncached_tokens × 0.06 work/token
KVS_work = remote_hit_tokens × 0.01 work/token
KVS_bytes = remote_hit_tokens × 65,536 bytes/token
D_work   = output_tokens × 1.00 work/token

request_slo_met = completed_full_output
                  AND finish_work_time - arrival_work_time <= tier_deadline_work
```

真实例子：trace request `...00001829` 的 input=6,509、output=30，共 13 个 cacheable blocks，最后一块 365 tokens，synthetic tier=STANDARD，budget=20,000。

| 路径 | P／KVS work | 加上 Decode 后，不含 queue |
|---|---:|---:|
| 前 10 blocks local hit | `1,389×0.06＝83.34` | 113.34 |
| 前 10 blocks remote fetch | `1,389×0.06＋5,120×0.01＝134.54` | 164.54 |
| 完全 location miss，全部重算 | `6,509×0.06＝390.54` | 420.54 |

结论不是“remote 永远最好”：local hit 最便宜。当前 frozen price 里，同一个 token 的 remote fetch 单价被定义为 `0.01 work`，recompute 单价被定义为 `0.06 work`，所以只对 fetched portion 而言便宜 6 倍。这个 6 倍是 scenario assumption，不是实验测出的收益；local hit＝0 也是 assumption，忽略 lookup／assembly，会高估 cache 收益。整条 request 还包含未命中的 1,389 tokens、queue 与 Decode，不能直接套 6 倍。

## 4. SLO 是什么

Mooncake trace 没有 tenant、tier、SLO 或真实 latency。当前实验按 request_id 的 SHA-256 确定性生成：16 个 synthetic tenants；20% STRICT、60% STANDARD、20% RELAXED。

`1 normalized work` 定义为 Decode 1 token 的 modeled service cost；它是工作量单位。Kernel 里每个 P 每个 work-time 处理 1 work，因此 work-time 是 synthetic 时间单位。Arrival 把 raw trace timestamp_ms 的数值放到这个坐标轴，但这不代表真实硬件已经证明 `1 work-time＝1ms`。Queue ceiling=20,000 与 STANDARD budget=20,000 数值相同也是人为 contract 选择，不是数据集推导出的物理常数。

| Tier | Request 数 | 每条 request 的 synthetic completion deadline | 能否写成秒 |
|---|---:|---:|
| STRICT | 4,780 | 5,000 work-time | 不能；缺 hardware calibration |
| STANDARD | 14,105 | 20,000 work-time | 不能；缺 hardware calibration |
| RELAXED | 4,723 | 100,000 work-time | 不能；缺 hardware calibration |

如果老板希望看 1s／1.5s／2s，正确做法不是改表头，而是先对目标 GPU 测出 `service_rate_work_per_ms`，然后用 `deadline_work=deadline_ms×service_rate_work_per_ms` 重跑。比如测得 8 work／ms 时，1s deadline 才能换成 8,000 work-time。当前 M9-HW 没有完成，所以报告保留具体的 synthetic deadline，并明确不冒充秒。

Input length 会改变 P work，但不会改变 tier deadline；tier 与长度独立。Fully-cold 下只有 28／4,780 条 STRICT request 自身 token work 已超过 5,000，占 0.59%；decode-only 不可达的是 0 条。因此 P=1 的 STRICT attainment=0 主要是 queue collapse，不是长请求天然无解。

## 5. 54 个 cell 到底是什么

一个 cell 不是一个 request。一个 cell 是固定全部参数后，从空 cache 开始完整 replay 23,608 条 request 一次。

```text
45 primary full-trace replays
= 3 synthetic cost regimes
× 5 arrival-pressure scales
× 3 candidate policies

9 sensitivity full-trace replays
= 3 cost regimes
× 1 个固定 1.5× stress point
× 3 个 No-Gate／Oracle／Noised-Oracle controls
```

这 54 次 replay 回答“Decode／eviction 想法是否值得继续”，不是 resource sizing。MIXED 1.5× Decode Causal 是专门构造的 overload stress：只有这一格把 decode credits 从 4,096 降到 128，开启 abort／retry fence 并使用 `GATED_DP`。它的 76.11% hit、17,795 retries、14,881 gated 不能写成策略平均表现。

| 被检验的想法 | 结果 | 大白话 |
|---|---|---|
| Priced Spill／cluster eviction | Null result | 45 个 primary cells 全部没有 capacity binding；实验没有真正触发“cache 满了该淘汰谁”。 |
| Decode overload gate | Narrow to overload-only | Stress cell queue p95 下降 71.96%，但 strict output goodput 相对 No-Gate 下降 9.55%，minimum tier 从 0.9358 降到 0.7531。 |
| Router hold 等待复用 | Kill | 可等待到的复用量不足以覆盖 hold；保留 locality routing，不继续做 router-side hold。 |

## 6. Resource sizing 是另一套 24-cell 实验

```text
24 sizing cells = 3 KVS topologies × P count {1..8}
```

固定条件：raw trace timestamps、D=8、MIXED cost、相同 HYBRID routing、cache 足以容纳全部 unique keys、no eviction。三组只改变 remote visibility／price：local-only、shared KVS、zero-transfer-price control。

Gate：

| Gate | Threshold | 来源 | 实际是否 binding |
|---|---:|---|---|
| Completion | 100% | 人为 quality floor | 含 drain window，24 cells 全部通过 |
| Tier deadline | 5,000／20,000／100,000 work-time | 人为 synthetic completion SLO | 决定每条 request 是否按时 |
| Minimum tier attainment | 80%；95% sensitivity | 三个 tier 分别算 `按时完成数÷offered 数` 后取最小值 | P≥2 后决定 N* |
| Jain fairness | ≥0.90 | 人为 fairness floor | 只在 P=1 失败 |
| P queue p95 | ≤20,000 work | 人为 backlog guardrail；不是 elapsed time 或 ms | 只在 P=1 失败 |
| KVS bytes／work | ≤1,000,000 | 人为 transfer guardrail | 全 grid 最大为 ceiling 的 36.24% |

为什么不能无限排队：P=1 的 completion=100%，说明所有 request 最终都跑完；但 minimum tier=0、Jain=0.7154、queue backlog p95=3,851,640.90 work。排队不会丢 request，却会让 request 越过自己的 deadline，因此不能算 healthy capacity。

隔离 routing 后的最终 N*：

| 最差 tier 必须达到的按时率 | Local-only | Shared KVS | Zero-transfer control | 结论 |
|---:|---:|---:|---:|---|
| 75% | 2 | 2 | 2 | 无资源差异 |
| 78%～80% | 3 | 2 | 2 | v1.2 中 priced Shared 只在这段门槛少 1 P |
| 81%～82% | 3 | 3 | 2 | priced Shared 与 Local 相同；zero-price 仍为 2 |
| 83%～93% | 3 | 3 | 3 | 无资源差异 |
| 94% | 3 | 4 | 3 | priced Shared 反而多 1 P，根因待验证 |
| 95% | 4 | 4 | 4 | 旧 `7→4` 撤回 |
| 96% | 5 | 6 | 5 | priced Shared 反而多 1 P；禁止从局部门槛外推 |
| 97% | P≤8 内不可行 | P≤8 内不可行 | P≤8 内不可行 | Grid exhausted，不能外推 |

同 P 对比是在回答“remote reuse 的收益是否超过 transfer cost”，不是为了证明 Shared 一定赢。24 个 cell 的最差档都是 STRICT（4,780 条）。P=2 时，Local 3,701 条按时，Shared 3,866 条按时，即多 165 条；queue backlog p95 从 12,615.42 降到 11,446.36 work（－9.27%），与拥塞点有帮助一致。P=3 时，Shared 4,485／4,780＝93.83% 反而低于 Local 4,504／4,780＝94.23%，直接造成 94% floor 下 Shared 多用 1 P；机制根因待验证。P=4 时，按时率只从 95.82% 升到 95.90%，queue backlog p95 反而从 2,436.54 升到 2,445.12 work（＋0.35%），收益接近零。当前 tenant／tier 只使用一组 deterministic hash assignment，threshold frontier 尚未经过换 seed robustness 检查。逐 tier 整数列将在正在重跑的 `m12-sizing-v2.1` artifact 中发布，重跑完成前这些计数只可由 minimum ratio 与三个 tier 分母交叉复算。

## 7. 可以说／不能说

可以说：在这份 trace 和 frozen normalized model 内，Shared KVS 在 P=2 拥塞点改善最差 tier attainment 与 queue；仅 78%～80% floor band 少 1 个 P，94%／96% 反而多 1 个 P。完整 threshold frontier 可复现。

不能说：真实 GPU 节省、真实 QPS／MFU、production TTFT、Global KVS 容量已经足够、transfer price 已经证明鲁棒。下一步必须做 KVS-only price sweep、COMPUTE／MEMORY regime sizing、hardware calibration 和有限容量 eviction。
