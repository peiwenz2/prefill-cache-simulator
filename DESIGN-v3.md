# Prefill Cache Simulator v3：从 selector baseline 到 KVS／SLO lifecycle planner

> 状态：Fable Reviewed，等待张珮文确认后开工。  
> 作者：张珮文  
> 日期：2026-08-05  
> 输入：`mooncake_trace.jsonl`，SHA-256 `b434f1816a707f4bac697235588184ebc374c9907cb981bb65fb0643471fe711`。  
> 目标：先完整回答 original task，再验证 KVS／SLO／Decode lease 能否把 cache hit 转成更高的 completed goodput 和更公平的服务。

---

## 0. 先给结论

### 0.1 可行，但必须分成两个 abstraction axis

```text
CacheTopology × ExecutionProtocol

CacheTopology：LOCAL_ONLY ｜ SHARED_KVS ｜ HYBRID
ExecutionProtocol：
  HandoffProtocol：PD ｜ DP ｜ gated-PD ｜ gated-DP ｜ KVS_DECOUPLED
  DecodePolicy：D0 ｜ D1 fixed lease ｜ D1.5 adaptive lease ｜ D2 cooperative preemption
```

- `CacheTopology` 回答：prefix 在哪里、命中了多少、需要 local load／remote KVT／recompute、谁负责 eviction。
- `ExecutionProtocol` 内部再拆 handoff 与 Decode policy，回答什么时候选 P、什么时候锁 D、何时 COMMIT、何时 hold／reject、Decode 是否续租／迁移。
- selector 只选 P，是 original task 的 scope；`RequestLifecyclePlanner` 是张珮文要验证的扩展，不强塞进 selector interface。

两个 axis 不是全笛卡尔积。协议可以声明 topology capability requirement，但不能绕过统一接口直接读取 store 内部状态：

| CacheTopology／Protocol | P0 PD | P1 DP | P2 gated-PD | P3 gated-DP | P4 KVS decoupled |
|---|---:|---:|---:|---:|---:|
| `LOCAL_ONLY` | 合法 | 合法 | 合法 | 合法 | **非法**：P-now／D-later 没有持久载体 |
| `SHARED_KVS` | 合法 | 合法 | 合法 | 合法 | 合法 |
| `HYBRID` | 合法 | 合法 | 合法 | 合法 | 合法 |

另有一个独立 capability：D2-R2 要求 `DecodeCheckpointStore`。它不能由 `SHARED_KVS` 自动推出，因为 prefix object 与 Decode checkpoint 的生命周期、一致性和 fencing 不同。

### 0.2 三个明确保留的跨 axis coupling

| Coupling | 为什么无法消除 | 接口边界 |
|---|---|---|
| Eviction externality | selector 把请求放到某个 P，会改变该节点的 eviction pressure | topology 返回 placement／eviction cost，selector 只消费 estimate |
| P4 store occupancy | P4 的 P-now 会产生中间 KV，占用 topology capacity | protocol 发 placement intent，topology 决定 admit／evict／reject |
| Decode continuation repricing | D1／D2 的 sent tokens 会改变下次 prefill／KVT 成本 | `ContinuationPrefixBuilder` 生成 block refs，再统一走 topology lookup |

这三点在进入对应 phase 的 major schema 时写入 executable contract 和 decision trace。`scenario.schema.json` 1.x 只冻结 Phase A，不提前承诺 B–D 的配置语义。除此之外，selector 不感知 protocol 名称，protocol 也不直接实现 cache policy。

### 0.3 Decode abort 可做，但产品语义必须叫 cooperative preemption

允许三种动作，按风险递增：

| Level | 动作 | 当前基础 | 用户流语义 |
|---|---|---|---|
| D0 | 不动 Decode | 现状 | 请求跑到结束 |
| D1 | token lease 到期，自然 `LENGTH`，client 吞掉中间终态后 continuation | 已有 Decode lease 实践 | 无中间 `LENGTH`，换 D 继续 |
| D2 | lease checkpoint 主动 `abort_self`，再 continuation | 已有 Passport／action grant 设计基础 | cooperative preemption |
| D3 | 任意时刻 hard abort | 只作 overload／故障止损 | 必须有 checkpoint，否则不可作为优化路径 |

**第一阶段只模拟 D1。** D2 只有在 D1 实锤收益、且 continuation recovery cost 可控后才进入实现。D3 不进入正常调度策略。

### 0.4 优化目标不是 raw cache hit，也不是 raw token throughput

Phase C 之后的主指标采用 strict cliff：

```text
strict_completed_goodput
  = Σ 完成且满足请求 SLO 的 useful tokens / wall time

lenient_completed_goodput
  = Σ 所有完成请求的 useful tokens / wall time
```

- strict 是主指标；完成但 SLO 违约的请求贡献 0。
- lenient 是辅助指标，用于识别策略是否只在 deadline 边界刷 strict 分数。
- partial output 只单列，不进入 completed goodput。
- Phase A 没有 synthetic SLO，主指标只用 token／block-ref hit 与 load skew，不报告 goodput。
- 如果用单一 scalar 排序，全部权重写入 report，并做 sensitivity；不能藏在 config 里。

必须同时报告：

- cache：local／remote／recompute token；
- SLO：TTFT、TPOT、completion deadline attainment；
- waste：aborted compute、re-prefill、duplicate output、idle D reservation；
- fairness：per-tenant normalized service、worst wait、preempt 次数；
- amplification：每个 logical request 的 P／D attempt 数和 KVT 次数。

只让部分请求更快、但让更多请求做一半被杀，不算吞吐提升。

---

## 1. Evidence base：不另起炉灶

### 1.1 本地代码与既有实践

| 素材 | 已有机制 | 本设计如何复用 |
|---|---|---|
| Turbo `CacheAwareScheduler` | pull-based queue；实例按 cache bucket range 拉请求；同时受 batch、running token、prefill concurrency、prefill token 四道容量限制 | 模拟 `GBPrefixBucketSelector`＋`AdmissionView` |
| Turbo `StickySessionScheduler` | bucket primary／alternative range；找不到匹配 engine 时不硬 fallback；同样带 batch／token／prefill capacity gate | 模拟 `StickySessionSelector`＋bounded fallback |
| Turbo `PdPrefillScheduler` | request／token capacity、length capacity、engine 上报的 cached token queue snapshot | 定义 selector 可见的 `NodeSnapshot` |
| FlexLB `ShortestTTFTStrategy` | `TTFT = queueTime + tokens - 0.7×hitCacheTokens`；top 30%＋similar band＋last-selected CAS fairness | 模拟 `FlexLbTtftSelector`，保持公式版本可替换 |
| Session affinity shadow 实践 | synchronous、bounded、fail-open；mock 不能冒充 production semantics | `AffinityKeyProvider` 独立，trace 无 session ID 时明确叫 proxy |
| KV-Ledger／S6 | logical request 首次 P 命中口径与后续 retry／resched 的 engine work 口径分开 | 指标分 user view 与 engine view |
| V6dCacheQueryUtil | prefill 前可用 token IDs 查询 KVS 命中，失败可降级 unknown | `KvsLookup` 可插拔，先 simulation、后 shadow query |
| gated-PD | P PREPARE／probe、D RESERVE、P COMMIT／ROLLBACK、pending token ledger | `GatedPdProtocol` 状态机 |
| Decode lease | token cap → continuation resched；无 retry penalty／backoff；attempt ID 唯一；中间 `LENGTH` 不透用户 | `LeasedDecodeProtocol` v1 |
| autoTPM Decode preemption | high 抢 decode-low；`Throttling.Aborted` 后 fresh-from-P；历史暴露 low starvation 风险 | 作为 aggressive baseline，不作为默认答案 |
| Passport／epoch-seq fence | owner、attempt、grant、checkpoint、late frame drop | D2 cooperative preemption 的正确性底座 |

代码证据版本：

- dashscope-platform：本地缓存的 `origin/main@afb893c7e826`。本轮 fetch 因内网凭证失败；相关 selector 文件与当前 working commit 内容一致，但开工前仍必须重新 fetch。
- RTP-LLM：`origin/main@cd77137bf307`；working tree 比它少 18 commit，但本轮引用的两个 FlexLB 文件在两者之间无 diff。
- dashserving：已 fetch `aone/main@fe13ca3f1c3b`；当前 checkout 有用户改动，不读取 checkout 作为 main 真值。
- dashllm：Aone fetch 成功；实现阶段重新钉目标 branch／commit。

关键 file anchors：

- `dashscope-platform/dashscope-turbo/dashscope-turbo-server/src/main/java/com/alibaba/dashscope/turbo/batch/scheduler/CacheAwareScheduler.java:1196-1202`：四维 capacity gate。
- `dashscope-platform/dashscope-turbo/dashscope-turbo-server/src/main/java/com/alibaba/dashscope/turbo/batch/scheduler/CacheAwareScheduler.java:1295-1339`：priority queue＋primary／secondary bucket fetch。
- `dashscope-platform/dashscope-turbo/dashscope-turbo-server/src/main/java/com/alibaba/dashscope/turbo/batch/scheduler/StickySessionScheduler.java:613-670`：bucket → primary／alternative engine。
- `dashscope-platform/dashscope-turbo/dashscope-turbo-server/src/main/java/com/alibaba/dashscope/turbo/batch/scheduler/StickySessionScheduler.java:752-770`：capacity gate＋range fetch。
- `RTP-LLM/rtp_llm/flexlb/flexlb-common/src/main/java/org/flexlb/dao/master/TaskInfo.java:32-38`：FlexLB prefill cost。
- `RTP-LLM/rtp_llm/flexlb/flexlb-sync/src/main/java/org/flexlb/balance/strategy/ShortestTTFTStrategy.java:187-205`：cache hit＋queue 合并成 TTFT。
- `RTP-LLM/rtp_llm/flexlb/flexlb-sync/src/main/java/org/flexlb/balance/strategy/ShortestTTFTStrategy.java:287-377`：similar band＋last-selected fairness。
- `~/dev-trees/DECODE_LEASE_HANDOFF.md`：Decode lease 已落地语义与 safety invariants。
- `~/Code/tongyi/github/HANDOFF-gated-pd-ttft-admission.md`：gated-PD PREPARE／RESERVE／COMMIT。
- `~/Downloads/client_token_scheduling_v0.2.md`：PP1／PP2、SLO slack、公平性、capacity lease。

### 1.2 Mooncake trace 能证明什么

本地 trace：

| 字段／事实 | 数值 | 边界 |
|---|---:|---|
| requests | 23,608 | 1 小时 |
| input 平均 | 8,589.96 tokens | 与论文正文 7,590 不同，不能混用 |
| output 平均 | 182.13 tokens | 可用于 synthetic decode duration |
| block refs | 409,356 | block size 512 |
| unique block IDs | 183,166 | block-ref compulsory miss 口径 |
| block-ref infinite ceiling | 55.255% | `(409,356−183,166)／409,356` |
| token-weighted infinite ceiling | 57.070% | partial last block 按实际 token 数加权 |
| timestamp | 0～3,600,000 ms | 可做 arrival replay |

trace 没有：

- session ID；
- tenant／user；
- service tier／deadline；
- P／D 实测 latency；
- KVT latency／bandwidth；
- GPU batch state。

因此：

1. 基础 cache result 可以直接由 trace 得出。
2. SessionAffinity 只能使用明确标注的 `AffinityKeyProxy`，不能声称是真 session。
3. SLO／fairness 实验必须使用 deterministic overlay＋多 seed sensitivity，不能冒充生产分布。
4. GPU 秒、KVT 时间只在校准后进入正式结论。

#### Hit metric contract

以下两个口径必须同时产出，禁止在同一列里混用：

```text
block_ref_hit_rate
  = continuous-prefix hit block refs / total block refs

token_weighted_hit_rate
  = continuous-prefix hit tokens / total input tokens
```

`57.070% > 55.255%` 是 partial block token weighting 导致，不是有限容量超过 compulsory-miss ceiling。§3.4 临时表测的是 `token_weighted_hit_rate`。与 Mooncake 论文约 50% 的结果对照前，必须先确认论文图表使用的 metric／block completion 语义，不能仅按容量数值直接比较。

---

## 2. Simulator domain model

### 2.1 Request

```python
@dataclass(frozen=True)
class Request:
    request_id: str
    arrival_ms: int
    input_tokens: int
    output_tokens: int
    prefix_blocks: tuple[int, ...]
    block_token_sizes: tuple[int, ...]   # 最后一个 block 不一定是 512
    affinity_key: str | None             # proxy，不能冒充 session ID
    tenant_id: str | None                 # advanced overlay
    service_tier: str | None              # advanced overlay
    ttft_deadline_ms: float | None
    tpot_deadline_ms: float | None
```

Block identity 使用 typed namespace，避免 trace integer ID 与 generated ID 碰撞：

```python
BlockId = TraceBlockId(value: int) | GeneratedBlockId(value: str)

generated_id_j = sha256(
    len64("generated-sha256-v1") || utf8("generated-sha256-v1")
    || raw32(trace_sha256) || u64be(block_size_tokens) || u64be(trace_line_index)
    || u8(parent_tag) || parent_value || u64be(generated_index)
)
```

`ContinuationPrefixBuilder` 负责把 `original_input + sent_tokens` 重新切成 block refs：

- trace 没有 request ID；M1 固定生成 `logical_request_id = "trace:" + trace_sha256 + ":" + zero_padded_line_index`，不能依赖随机 UUID；该 display ID 不参与 generated block hash；
- generated IDs 跨进程／跨 seed 稳定，不能使用 Python process-randomized `hash()`；
- `parent_tag` 固定为 trace block／generated block／empty prefix 三种；parent value 分别编码为 `u64-be`／32-byte raw digest／空；所有字段与 3 组 golden vectors 见 `docs/identity-contract-v1.md`；
- 第一个 generated block 正确处理 original input 的 partial tail；
- 同一 logical request 的后续 attempt 复用同一 generated chain；不同 request 即使 output length 相同也不假设内容相同；
- 该 schema 让 R0／R1 能通过真实 topology lookup 定价，不在 M7 临时拍 recovery ratio。

Randomness 同样是 reproducibility contract：scenario 只给 root seed，
`named-sha256-v1` 按稳定 consumer name 派生 selector／tie-break／cluster-view loss／eviction 子 stream。切换 selector 不能扰动 loss draws，保证 paired A／B 可比；完整 bytes 见 `docs/randomness-contract-v1.md`。

硬语义：

- 命中只算从 block 0 开始的 continuous prefix。
- `hash_ids[i]` 已包含 preceding blocks 的 hash 语义，但 KV object 仍按 block 存；中间缺块后，后续 resident block 记作 stranded，不算可用命中。
- token hit 需要按 `input_length` 截断最后一个 partial block。
- logical request 与 physical attempts 分开计数。

### 2.2 NodeSnapshot

```python
@dataclass(frozen=True)
class NodeSnapshot:
    node_id: str
    alive: bool
    cache_capacity_blocks: int
    resident_blocks: int
    running_requests: int
    running_tokens: int
    prefilling_requests: int
    queued_uncached_tokens: int
    available_at_ms: float
    last_selected_ms: float
```

View modes：

- `ORACLE`：当前 simulator 真值。
- `DELAYED(delta_ms)`：周期上报，模拟 Turbo／FlexLB state staleness。
- `LOSSY(p)`：部分 node 缺 cache stats，验证 fail-open。

### 2.3 五个基础接口

```python
class PrefillNodeSelector(Protocol):
    def select(self, request: Request, view: ClusterView) -> Selection: ...

class BlockEvictionPolicy(Protocol):
    def on_lookup(self, block_id: int, now_ms: int) -> None: ...
    def admit(self, block: BlockMeta, now_ms: int) -> bool: ...
    def evict(self, required_blocks: int, now_ms: int) -> list[int]: ...

class CacheTopology(Protocol):
    def lookup(self, node_id: str, request: Request) -> CacheLookup: ...
    def place(self, node_id: str, request: Request) -> PlacementResult: ...

class PrefillRequestProcessor(Protocol):
    def process(self, request: Request) -> PrefillResult: ...

class Simulator(Protocol):
    def run(self, trace: Iterable[Request], scenario: Scenario) -> Report: ...
```

`Selection` 必须带 decision trace：

```python
Selection(
    node_id="p3",
    score=812.4,
    components={
        "local_hit_tokens": 6144,
        "queued_uncached_tokens": 4096,
        "affinity": 1,
        "eviction_cost": 12.3,
    },
    fallback_reason=None,
)
```

### 2.4 Replay mode 与 cache visibility

两种 replay 不混榜：

| Mode | 时间语义 | 可比较策略 | 用途 |
|---|---|---|---|
| `CACHE_ONLY` | 严格按 arrival 顺序处理，不建 queue／service time | S0／S1／S3／eviction | compulsory ceiling、论文 sanity anchor |
| `QUEUED` | event loop 驱动 arrival／start／finish，使用 normalized service model | S0-S6 | A1 headline、load／queue／staleness 排名 |

同一个 LRU 在两种 mode 下可以得到不同数字，因此 report key 必须包含 `replay_mode`。A1 headline 固定使用 `QUEUED + DELAYED`；`CACHE_ONLY` 只做 analyzer／论文口径 sanity，不参与 S2／S5／S6 排名。

Cache visibility 显式配置：

| Visibility | 语义 | 默认／用途 |
|---|---|---|
| `INSERT_AT_COMPLETION` | prefill finish 后 block 才进入 resident set | 默认，保守且与 §3.4 可比 |
| `INFLIGHT_DEDUP_WAIT` | 相同 block 已在计算时，后来请求等待 producer 完成 | A3 sensitivity；单列 wait tokens／ms，不冒充即时 cache hit |

inflight block 在完成前始终 pinned，不参与 eviction；producer 失败时 waiter 回到正常 lookup／compute。

---

## 3. Phase A：基础 selector／eviction 策略

### 3.1 Selector versions

先固定 `LOCAL_ONLY + LRU`，只换 selector，避免 selector 与 eviction contribution 混在一起。

| ID | Selector | 参考 | 精确定义 | 回答的问题 |
|---|---|---|---|---|
| S0 | `RandomSelector` | original task | alive nodes 等概率，seed 固定 | 最低基线 |
| S1 | `RoundRobinSelector` | 常规均衡 | alive nodes 轮转 | 仅均衡是否已经够用 |
| S2 | `LeastWorkSelector` | Global Batching capacity view | 最小 `queued_uncached_tokens + running_prefill_tokens` | 只看负载的收益 |
| S3 | `GBPrefixBucketSelector` | CacheAware／Sticky bucket ownership | `AffinityKeyProxy → bucket → primary node`；soft overload 时 bounded secondary／LeastWork | 静态亲和的 hit／skew tradeoff |
| S4 | `SessionAffinitySelector` | StickySession | 已见 affinity key 优先 last node；超过 overload threshold 后 fail-open LeastWork | session continuity 是否比 prefix bucket 更稳 |
| S5 | `FlexLbTtftSelector` | FlexLB | `queue_ms + input_tokens - 0.7×hit_tokens`；top 30% similar band 内 LRU node | cache＋load 统一是否优于硬 sticky |
| S6 | `CalibratedTtftSelector` | 本设计 basic+ | `queue_ms + prefill_ms(uncached_tokens, state)` | 替掉 FlexLB 拍脑袋 0.7 |
| S7 | `CacheSloSelector` | 本设计 advanced | 在 S6 上按 normalized slack／risk 选，cache warmth 只在相似 risk band 内 tie-break | SLO 是否自然化解 affinity／balance 冲突 |

`GBPrefixBucketSelector` 不是逐行复制 Turbo。Turbo 是 engine pull queue，本 simulator 是 request push API；二者通过相同 bucket ownership／primary-secondary range／capacity gate 做语义等价。

S3 与 S4 必须拆成两个 selector，保证 attribution；只共享两个 helper：

```text
CapacityGate
  hard gate：真实 batch／token／prefill capacity 上限
  soft affinity break：primary_load > α × mean_eligible_load

BoundedFallback
  primary → bounded secondary candidates → LeastWork fail-open
```

`α ∈ {1.10, 1.25, 1.50, 2.00}` 做 sensitivity。cluster mean 为 0 时不触发 soft break；hard gate 永远优先。相对 soft gate 保证 fixed-total 与 fixed-per-node sweep 下 overload 语义不漂移。

### 3.2 AffinityKeyProvider

trace 没有 session ID，provider 必须独立可替换：

| Provider | 定义 | 用途 |
|---|---|---|
| `NoneKey` | 无 affinity | S0／S1／S2 |
| `PrefixAnchor(k)` | 第 `k` 个 prefix hash；不足 k 取最后一个 | S3 sanity baseline |
| `OnlineConversationLinker` | 严格只看历史：时间窗内最长公共 prefix 超阈值则继承 family ID | S4 主实验 |
| `RealSessionKey` | 未来有真实 session 字段时注入 | production replay |

必须扫 `k ∈ {1,2,4,8,12,16}`、link TTL、minimum shared blocks。不能挑一个最好看的 k 当最终结果。

`OnlineConversationLinker` 额外满足：

- hot-block exclusion：超过 hotness percentile／fanout threshold 的共同 prefix 不建立 family；
- family size cap：超过上限后停止吸附，避免 shared system prompt 形成巨型假 session；
- 所有 linkage decision 记录当时可见历史，不允许离线回看 future request；
- 预期 S4 与 S5 的 hit 接近，主要比较 skew、fallback 和 stale-view stability，而不是强求 hit 拉开。

### 3.3 Eviction versions

Selector sweep 后固定 Top-2 selector，再换 eviction：

| ID | Policy | 定义 | 目的 |
|---|---|---|---|
| E0 | FIFO | 首次 insert 顺序淘汰，hit 不刷新 | original task 基线 |
| E1 | LRU | hit／request compute 后刷新 | Mooncake trace 论文基线 |
| E2 | SLRU | probation＋protected 两段 | 防 scan pollution |
| E3 | SecondHitAdmission＋LRU | ghost 首见；第二次才进入 protected cache | 验证 78%+ one-hit block 的 pollution |
| E4 | ChainAwareValue | `reuse_prob × recompute_cost ÷ replicas`，同时计 stranded descendants | advanced，只在 E0-E3 出数后做 |

约束：

- inflight／pinned block 不可淘汰。
- 请求大于 node capacity：允许 streaming compute，但最终只留下 eviction policy 接受的 suffix／hot blocks；行为必须显式记录。
- eviction 后又在 reuse window 内访问，计 `eviction_regret`。
- 早期 block 被逐导致后续 resident blocks 不可用，计 `stranded_block_time`。

### 3.4 Provisional trace sanity，不是最终实验

下面只用于验证 simulator 方向。临时 read-only script：continuous-prefix lookup、partial last block、request 完成后全 block resident、LRU；没有 queue／service time。

固定**集群总容量**，N 增大时 per-node capacity=`total/N`。下表单位统一为 **continuous-prefix token-weighted hit rate**：

| N／selector | 1k | 10k | 30k | 50k | 100k |
|---|---:|---:|---:|---:|---:|
| 1 node＋LRU | 35.18% | 47.56% | 55.46% | 56.91% | 57.04% |
| 4 Random＋LRU | 34.32% | 38.35% | 42.10% | 43.96% | 45.01% |
| 4 PrefixAnchor(2)＋LRU | 35.15% | 47.47% | 55.44% | 56.92% | 57.02% |
| 8 Random＋LRU | 31.97% | 36.57% | 38.85% | 39.95% | 40.94% |
| 8 PrefixAnchor(2)＋LRU | 34.99% | 47.09% | 55.45% | 56.88% | 57.04% |

关键观察：

1. Random 把相同 prefix 打散，8 nodes／50k 总容量损失约 17pp。
2. PrefixAnchor(2) 几乎恢复 token-weighted infinite ceiling `57.070%`，但负载很歪：8 nodes request max／mean=`3.55×`，input token max／mean=`2.98×`。
3. 这正是 FlexLB／S6 要回答的 tradeoff：不要只看 hit，也不要只看 balance。
4. `PrefixAnchor(1)` 受超热共同首块影响，8 nodes／50k 只有 43.73%；anchor choice 不能写死。

最终实现必须用 versioned config 重跑，同时输出 block-ref 与 token-weighted 两张表；以上数字不作为正式结论。

---

## 4. Phase A experiment matrix

### 4.1 两种 capacity sweep

必须同时跑：

1. **Fixed total capacity**：隔离 routing fragmentation／replication cost。
2. **Fixed per-node capacity**：模拟真实 scale-out 后 aggregate cache 增长。

参数：

```text
N                    = 1, 2, 4, 8, 16
total cache blocks   = 1k, 10k, 30k, 50k, 100k
per-node blocks      = 1k, 5k, 10k, 30k
view delay           = 0, 100ms, 1s, 5s, 10s
replay speed         = 1×, 2×, 4×
replay mode          = CACHE_ONLY, QUEUED
cache visibility     = INSERT_AT_COMPLETION, INFLIGHT_DEDUP_WAIT
affinity soft α      = 1.10, 1.25, 1.50, 2.00
seed                 = 713..720
```

基础 cache-only policy 是 deterministic 时 seed 只影响 Random／tie-break；advanced overlay 必须全 8 seeds。

Headline realism 默认 `DELAYED`，`ORACLE` 仅作为 upper bound，`LOSSY` 用于 fail-open robustness。不得用 ORACLE 主榜夸大 S5／S6 优势。

### 4.2 三轮实验，禁止全笛卡尔积

| Round | 固定 | 扫描 | 产出 |
|---|---|---|---|
| A1 selector | `LOCAL_ONLY + LRU + QUEUED + DELAYED` | S0-S6 | hit／load／queue Pareto＋replica mechanism |
| A2 eviction | A1 Top-2 selector | E0-E3 | admission／eviction contribution |
| A3 combine | Top-3 combinations | capacity／N／staleness／visibility | final baseline frontier |

### 4.3 Phase A metrics

| 类别 | 指标 |
|---|---|
| Cache | token-weighted hit、block-ref hit、各自 ceiling attainment、unique resident、replica factor |
| Load | requests／tokens／uncached tokens per node、max／mean、CV、Gini |
| Queue | estimated queue ms p50／p95／p99、head-of-line wait |
| Eviction | eviction count、regret、one-hit pollution、stranded block time |
| Decision | affinity kept／broken、fallback count、no-stat fallback、score components |

Phase A stop gate：

- 同配置可重复；
- invariant tests 全绿；
- one-node infinite／large-capacity ceiling 与 independent analyzer 对齐；
- A1 能复现“hard affinity hit 高但 skew 高”；
- A1 用 replica factor 解释 Random fragmentation／PrefixAnchor 去重机制，不只报告 outcome；
- 不引入 synthetic SLO 就不报告 SLO gain。

---

## 5. Phase B：KVS-aware CacheTopology

### 5.1 三种 topology

```text
LOCAL_ONLY
  P.HBM miss → recompute

SHARED_KVS
  P.HBM miss → shared KVS hit → remote KVT → recompute remainder

HYBRID
  local HBM → local DRAM／v6d → remote DRAM／KVS → recompute
```

统一 lookup result：

```python
CacheLookup(
    local_hit_tokens=4096,
    local_dram_hit_tokens=1024,
    remote_hit_tokens=2048,
    recompute_tokens=1421,
    transfer_bytes=...,
    source_segments=[...],
)
```

### 5.2 CPU simulation 不跑真实 KVT

参数化：

```text
kvt_ms(bytes) = fixed_latency_ms + bytes / effective_bandwidth
effective_cost = max(non_overlapped_kvt_ms, recompute_ms)
```

扫：

- bandwidth：25／50／100／200 Gbps；
- fixed latency；
- layer-wise overlap：0%／50%／90%；
- local／remote capacity；
- KVS lookup timeout／unknown rate；
- hot block replica factor。

只有当策略排序对这些参数敏感，才借 GPU／RDMA 机器校准。cache policy 本身不需要 GPU。

### 5.3 KVS 改变 selector 的目标

`LOCAL_ONLY`：选错 P 就 miss。  
`SHARED_KVS`：选错 P 仍可 remote hit，但会付 KVT／hotspot cost。

因此 S6 扩展成：

```text
plan_cost(P)
  = queue_ms(P)
  + min(local_compute_plan, remote_fetch_plan)
  + eviction_externality(P)
```

不要直接把 remote hit 当 local hit。二者必须分账。

Phase B 增加具名对照 `S6-KVS／MooncakeAlgo1`：

```text
best_prefix_len, best_holder = FindBestPrefixMatch(...)

for each candidate P with local prefix_len:
  if best_prefix_len / prefix_len < kvcache_balancing_threshold:
    cost(P) = queue(P) + prefill(full_len, prefix_len)
  else:
    transfer_len = best_prefix_len - prefix_len
    cost(P) = transfer(P, best_holder, transfer_len)
              + queue(P)
              + prefill(full_len, best_prefix_len)

select P with minimum predicted TTFT
```

零 prefix 的除法按显式 sentinel／branch 处理，不能依赖浮点无穷。以上复刻自 Mooncake Algorithm 1 第 4～23 行；还要单列第 28～30 行触发的 hot-spot migration。记录 `kvcache_balancing_threshold` sensitivity。该配置用于与论文直接对话；calibrated S6 可以扩展它，但不能取代这个复刻锚点。

---

## 6. Phase C：SLO-aware RequestLifecyclePlanner

### 6.1 新 interface，不污染 selector

```python
class RequestLifecyclePlanner(Protocol):
    def prepare(self, request, cluster_view) -> ExecutionPlan: ...
    def on_prefill_probe(self, plan, p_probe) -> LifecycleAction: ...
    def on_decode_probe(self, plan, d_probe) -> LifecycleAction: ...
    def on_decode_checkpoint(self, plan, checkpoint) -> LifecycleAction: ...
```

```python
class LifecycleAction(Enum):
    HOLD = auto()
    SELECT_P = auto()
    SELECT_D = auto()
    COMMIT_P = auto()
    ROLLBACK_P = auto()
    CONTINUE_D = auto()
    RESCHEDULE_D = auto()
    REJECT = auto()
    TERMINAL_FAIL = auto()
```

### 6.2 Protocol implementations

| ID | Protocol | 时序 | 主要成本 |
|---|---|---|---|
| P0 | PD | P compute → 选 D | D 无位时可能浪费 P |
| P1 | DP | reserve D → P compute | D hold／idle |
| P2 | gated-PD | P PREPARE probe → 判 → D RESERVE → P COMMIT | probe＋pending ledger |
| P3 | gated-DP | D capacity probe／soft lease → P PREPARE／COMMIT | soft lease staleness／hold |
| P4 | KVS decoupled | P-now → KV in Store → D-later | Store occupancy＋D pull latency |

所有 protocol 使用相同 selector／topology／workload，避免架构名词对比变成不同输入对比。

### 6.3 SLO model

Trace 没 tier。提供两种模式：

1. `RelativeSloOverlay`：对齐 Mooncake paper 思路，TTFT／TBT 相对 isolated latency 的倍数。
2. `TieredSloOverlay`：strict／standard／relaxed，比例和 deadline 全在 config，8 seeds sensitivity。

每个 request 维护：

```text
ttft_slack = ttft_deadline - predicted_ttft
tpot_slack = tpot_deadline - predicted_tpot
completion_slack = completion_deadline - predicted_completion
```

Cache warmth 只在相似 risk band 内作为 tie-break。不能让 cold request 永久没机会。

Phase C+ 同时输出：

| Metric | 定义 | 用途 |
|---|---|---|
| `strict_completed_goodput` | 完成且满足该请求全部 SLO 的 useful tokens／wall time | 主指标 |
| `lenient_completed_goodput` | 所有完整完成请求的 useful tokens／wall time | cliff／边界行为审计 |
| `partial_output_tokens` | 未完成请求已 emit 的 tokens | 单列，不计 goodput |

若 policy search 需要 scalar，`tier_weight`、`λ_gpu`、`λ_stall`、`λ_risk` 全部进入 report provenance，并至少做 one-at-a-time sensitivity。

---

## 7. Phase D：Decode lease／preemption

### 7.1 为什么 client 有这个位置

LLMClientV1 同时拥有：

- logical request identity；
- P／D 两条独立 stream；
- 已 emit tokens；
- retry／continuation；
- SLO／priority；
- P probe、D probe、KVS estimate；
- next attempt 的路由入口。

Engine 适合做 token scheduling 和 local preemption；Turbo／FlexLB 适合选 instance；只有 client 能把“当前 D 不值得继续”转换成“带 checkpoint 的另一次 D attempt”，并保证用户 output 不重复。

### 7.2 不做“谁低优就杀谁”

每个 Decode checkpoint 比较两个 plan：

```text
Keep(r, d)
  = completion_value(r) × P(按 SLO 完成 | keep)
  - λ_gpu × remaining_decode_gpu_ms(r, d)
  - λ_interference × interference_gpu_ms(r, d)

Move(r, d→d2)
  = completion_value(r) × P(按 SLO 完成 | move)
  - λ_gpu × [checkpoint_gpu_ms(r) + re_prefill_or_kvt_gpu_ms(r)]
  - λ_stall × migration_stall_ms(r)
  - λ_risk × duplicate_risk(r)

Preempt only if Move - Keep > safety_margin
```

所有项先映射到同一个 normalized utility 尺度；`λ_*` 由离线 replay 或 shadow telemetry 标定，不能直接把“价值”和“毫秒”裸相减。`completion_value` 由 tier／tenant weight／aging 组成，但必须有 admission-time snapshot，防止请求执行中被任意改 priority。

### 7.3 Fairness invariants

| Invariant | 规则 |
|---|---|
| F1 Minimum quantum | 每次获得 D 后至少执行 `min_tokens` 或 `min_ms`，除非故障／硬 overload |
| F2 Bounded preemption | 每个 logical request 最多 `max_preemptions`；到顶后 continue 或 terminal，不无限搬 |
| F3 Aging | wait／stall 越久，virtual priority 单调上升 |
| F4 Service fairness | 用 `served_decode_ms / tenant_weight` 或 served tokens 的 virtual runtime 排序 |
| F5 Cache tie-break only | cache warmth 只在同 tier／相似 slack band 内加分 |
| F6 No duplicate output | epoch＋output_seq fence；旧 attempt late frame drop |
| F7 Grant conservation | retry／resched／hedge 用 allocation grant，不共改易丢的 counter |
| F8 Fail-open boundary | planner state unavailable 时回到 D1 fixed lease／现状，不 hard abort |

需要同时报：

- Jain fairness index；
- per-tier completion ratio；
- per-tenant normalized attained service；
- max wait／max stall；
- preemptions per completed request；
- 被连续 preempt ≥2 次的 request 比例。

### 7.4 D1 → D2 的落地阶梯

#### D1 Fixed token lease，先模拟已有机制

- 每个 D attempt cap=`min(remaining_budget, lease_tokens)`。
- lease 到期自然 `LENGTH`。
- client 不向用户 emit 中间 `LENGTH`。
- continuation input=`original_input + sent_tokens`。
- resched 不消耗 fault retry budget、不带 cache penalty header、无 backoff。
- `max_rounds` 到顶后最后一次放开。

#### D1.5 Adaptive lease，不主动 abort

每次发 D 前决定下一 lease：

```text
lease_tokens
  = clamp(base_lease
          × slo_slack_factor
          × fairness_debt_factor
          × cache_recovery_factor,
          min_lease,
          max_lease)
```

- tight SLO／已经多次 stall → lease 变长。
- 新来的 high utility request 多、当前 request slack 宽 → lease 变短。
- recovery cost 高／无 KVS checkpoint → lease 变长，减少搬迁。

#### D2 Cooperative preemption

只有到固定 checkpoint 才允许 `abort_self`：

1. D 上报 checkpoint／progress／TPOT／KV handle。
2. client 计算 Keep／Move。
3. 决定 Move 时，先创建新 epoch／grant。
4. 旧 D 收 `abort_self(epoch)`，停止后确认 terminal checkpoint。
5. 新 D 从 checkpoint resume；无法 resume 则走 continuation recompute。
6. output authority 仍在 client，按 seq fence emit。

### 7.5 KVS 是 D2 的经济基础，但不是自动成立

当前 continuation 能把已 emit tokens 拼回 prompt，但不等于 decode 产生的 KV 已存在 global KVS。

必须分三档验证：

| Recovery | 需要什么 | 成本 |
|---|---|---|
| R0 Recompute continuation | 现有 client continuation | 重新 prefill 全 context，可能被 prefix cache 命中 |
| R1 Prefix-KVS assisted | exact continuation prefix 已进 KVS | remote load＋小段 recompute |
| R2 Decode checkpoint handle | D 定期把 KV checkpoint 放可跨实例读的 store | 最低重算，但要 store lifecycle／fence |

如果只有 R0，频繁 abort 很可能让 throughput 更差。D2 开工 gate：R0／R1 recovery cost 的 p95 必须先量出来。

---

## 8. Advanced simulation scenarios

### 8.1 WorkloadOverlay

保持 Mooncake arrival／input／output／hash 不变，只叠 synthetic fields：

```python
WorkloadOverlay(
    tenant_distribution=...,
    tier_distribution=...,
    deadline_model=...,
    decode_slowdown_noise=...,
)
```

RNG stream 分开：

- trace／routing；
- tenant／tier；
- service time noise；
- fault；
- policy tie-break。

Baseline 与 candidate 共用相同 seeds 713～720。

### 8.2 对照组

| Group | Selector | Protocol | Decode policy |
|---|---|---|---|
| B0 | Random | PD | no lease |
| B1 | GBPrefixBucket | PD | no lease |
| B2 | FlexLB | PD | no lease |
| B3 | CalibratedTTFT | gated-PD | no lease |
| B4 | CacheSLO | gated-PD | fixed lease D1 |
| B5 | CacheSLO | gated-PD | adaptive lease D1.5 |
| B6 | CacheSLO | gated-PD | cooperative preemption D2 |
| Aggressive control | priority victim | PD | autoTPM-like abort／fresh retry |

Aggressive control 是为了测 wasted compute／low starvation，不作为推荐 production policy。

### 8.3 决策门

| Gate | 通过条件 |
|---|---|
| G-A | selector Pareto 优于 Random／RR，且不是只靠 load skew 换 hit |
| G-B | KVS-aware 结果在合理 bandwidth sensitivity 下稳定 |
| G-C | TTFT prediction 排序稳定；校准前只报 normalized work，不报 ms 真值 |
| G-D1 | fixed lease 提升 completed goodput，p95 migration stall 可接受 |
| G-D2 | cooperative preemption 相对 D1 仍有稳定增益；p99 stall、waste、最低 tier completion 不劣化；`preemptions_per_completed_request` 不超过硬上限 |
| G-Prod | simulation 排名在 GPU replay 中一致，不要求绝对值完全相等 |

---

## 9. Detail todolist

### M0｜Review freeze

- [x] 接受 `CacheTopology × ExecutionProtocol`，同时加入 validity matrix 和三个显式 coupling。
- [x] S3／S4 拆开；共享 `CapacityGate`／`BoundedFallback` helper。
- [x] 接受严格 online 的 `AffinityKeyProxy`；加入 hot-block exclusion 和 family size cap。
- [x] FlexLB v1 先严格复刻，再实现 calibrated v2。
- [x] Phase A 先固定 LRU 排 selector，再扫 Top-2 eviction。
- [x] D1／D1.5 是 advanced implementation ceiling；D2 只做 gated spike，不直接实现 arbitrary abort。
- [x] Phase C+ 主报 strict completed goodput，lenient／partial 单列；Phase A 不报 goodput。
- [x] 第一期 fairness 使用 tenant-weighted virtual runtime，不把 quota／TPM ledger 塞进 overlay。
- [x] R2 Decode checkpoint store 独立于 Mooncake prefix Store。
- [x] 文档归属：`DESIGN.md` v2 保留 concept／insight；`DESIGN-v3` 是 build spec，互相引用、不合并。
- [x] 张珮文确认本轮修订。
- [x] `scenario.schema.json` 1.0.0 冻结为 strict Phase A-only；B–D 配置留在 design example，进入新 phase 时升 major。
- [x] 把 G1 metric unit、G2 synthetic ID、G3 visibility、G4 replay mode 固化到 `scenario.schema.json`。
- [x] G2 使用 byte-exact `generated-sha256-v1`，覆盖 trace parent／generated parent／empty prefix 三组 golden vectors。
- [x] 固化 schema version policy、exact-file-byte config SHA 和 M1 runtime validation 清单。

交付：review resolution＋冻结的 `scenario.schema.json`。schema 未冻结前不进入 M1 implementation。

### M1｜Trace analyzer／provenance

- [x] 建 package skeleton、config loader、deterministic named-stream seed manager。
- [x] parse／validate 四字段，校验 hash chain length 与 input length。
- [x] 正确计算 partial last block。
- [x] 输出 request／block／reuse-distance／hotness／prefix-family stats（anchor depth 1–4）。
- [x] 输出 trace SHA、exact-byte config SHA、git SHA＋dirty state 到 report。
- [x] independent analyzer 分别计算 block-ref `55.2550836%` 与 token-weighted `57.0700233%` ceiling。
- [x] 定义 typed `TraceBlockId／GeneratedBlockId` 并实现已冻结的 deterministic `generated-sha256-v1` hash contract。
- [x] 从 trace SHA＋line index 生成 stable logical request ID。
- [x] 核对 Mooncake Table 1：论文只称 cache hit ratio，未完整给 denominator／insertion timing；保留 `Inf=0.51` 为 paper reference，不与两种 ceiling 强行等同。
- [x] 一次 streaming read＋紧凑 `array("Q")` block storage；23.6k requests 不引入分布式并行框架。

Tests：空 trace、坏 JSON、重复 timestamp、短于一 block、partial block、超长 request。

### M2｜Core interfaces／LOCAL_ONLY

- [ ] `Request`／`NodeSnapshot`／`ClusterView`／`Selection`。
- [ ] `PrefillNodeSelector`／`BlockEvictionPolicy`／`CacheTopology`／`PrefillRequestProcessor`。
- [ ] `ContinuationPrefixBuilder`，覆盖 partial input tail 与 generated chain。
- [ ] continuous-prefix lookup。
- [ ] pinned block／placement／eviction／stranded accounting。
- [ ] `CACHE_ONLY／QUEUED` 两种 replay mode；event loop 用 heap 管 arrival、prefill start／finish、cache insert／evict。
- [ ] `INSERT_AT_COMPLETION／INFLIGHT_DEDUP_WAIT` visibility；默认前者。
- [ ] `ORACLE`／`DELAYED`／`LOSSY` view。

Invariant tests：capacity 不超；pinned 不驱逐；hit 不越过首 miss；logical／attempt token 不重复计。

### M3｜Baseline selectors／eviction

- [ ] S0 Random、S1 RR、S2 LeastWork。
- [ ] 共用 `CapacityGate`：hard capacity＋`primary_load > α×cluster_mean` soft break。
- [ ] 共用 `BoundedFallback`：primary／secondary range＋LeastWork fail-open。
- [ ] S3 GBPrefixBucket：静态 prefix ownership 薄壳。
- [ ] S4 SessionAffinity：online linker＋hot-block exclusion＋family cap 薄壳。
- [ ] S5 FlexLB：公式 v1＋top30／similar band／last-selected fairness。
- [ ] S6 CalibratedTTFT：先用 normalized work model。
- [ ] E0 FIFO、E1 LRU、E2 SLRU、E3 SecondHitAdmission。
- [ ] 每次 selection 保留 component trace，可按 request sample debug。

交付：A1／A2／A3 CSV＋图＋策略排名；CSV schema 从这里稳定，含 trace／config／git SHA、metric unit、replay mode、view mode、visibility；明确 fixed-total 与 fixed-per-node。

### M4｜Base result review

- [ ] 验证 PrefixAnchor sensitivity，禁止只留 k=2。
- [ ] 检查 hit gain 是否由单节点 overload 换来。
- [ ] 检查 SessionAffinity proxy 是否把 shared system prompt 错当 session，验证 hot-block exclusion／family cap。
- [ ] 预期 S4／S5 hit 接近；重点比较 skew、fallback、stale-view stability，不能以“hit 没拉开”判失败。
- [ ] A1 用 replica factor 解释 Random fragmentation 和 affinity 去重机制。
- [ ] 对 Top-3 做 8-seed／stale-view replay。
- [ ] 选择进入 KVS phase 的 2 个 selector。

Stop：如果 S5／S6 相对简单 S3 没有 Pareto 增益，不进入复杂 selector implementation。

M1–M4 作为独立可交付：S0／S1／S2＋FIFO／LRU＋正式 hit report 已完整回答 original task，不等待后续 KVS／SLO phase。

### M5｜KVS topology

- [ ] `SHARED_KVS`／`HYBRID` lookup／placement。
- [ ] local／remote／recompute 分账。
- [ ] 参数化 KVT fixed latency／bandwidth／overlap。
- [ ] remote store capacity／replica／eviction。
- [ ] KVS timeout／unknown fail-open。
- [ ] hot block transfer hotspot metric。
- [ ] `S6-KVS／MooncakeAlgo1` 严格复刻＋`kvcache_balancing_threshold` sensitivity。

交付：bandwidth／capacity sensitivity；回答“要不要借 GPU 量 KVT”。

### M6｜Time／SLO model

- [ ] normalized service model：uncached prefill work、decode token work。
- [ ] `RelativeSloOverlay`。
- [ ] `TieredSloOverlay`＋tenant overlay＋独立 RNG streams。
- [ ] PD／DP／gated-PD／gated-DP lifecycle state machine。
- [ ] hold／reserve／commit／rollback／reject／waste accounting。
- [ ] strict／lenient completed goodput、partial output、SLO、fairness metrics。
- [ ] scalar 权重进入 provenance，并做 sensitivity。

交付：同一 selector 下四 protocol Pareto；不使用未经校准的“真实 ms”措辞。

### M7｜Decode lease simulation

- [ ] D0 no lease。
- [ ] D1 fixed lease，复现当前 `8／8／14`、max-round、kill-switch、failure downgrade invariants。
- [ ] R0 continuation 通过 `ContinuationPrefixBuilder`＋topology lookup 计算 recompute cost。
- [ ] R1 exact generated prefix 通过同一 synthetic block chain 计算 KVS-assisted cost。
- [ ] D1.5 adaptive lease，只在 attempt boundary 调整。
- [ ] fairness debt／minimum quantum／bounded preemption。

交付：D0／D1／D1.5 completed goodput、stall、waste、fairness。

### M8｜Cooperative preemption spike

- [ ] `DecodeCheckpoint` schema：epoch、seq、emitted tokens、KV handle、TPOT、served quantum。
- [ ] Keep／Move decision＋safety margin。
- [ ] D action grant：首期只 `report_checkpoint`／`abort_self`。
- [ ] old epoch late frame drop。
- [ ] max preemption／min quantum／aging。
- [ ] `preemptions_per_completed_request` hard gate。
- [ ] R2 global KV checkpoint 作为独立 topology，不能假设已存在。

进入条件：M7 通过 G-D1，且 R0／R1 recovery p95 有实测或可信校准。

### M9｜1-GPU calibration

- [ ] 扫 uncached input tokens／batch state，拟合 `prefill_ms`。
- [ ] 扫 output length／batch state，拟合 TPOT／decode work。
- [ ] 记录 estimate residual，不强行正态。
- [ ] 如 M5 敏感，再加 KVT benchmark；否则不借 RDMA 集群。
- [ ] simulator 参数版本化。

机器：先 1 台目标机型；此阶段不需要 4P+4D。

### M10｜Multi-GPU replay

- [ ] N=4 P＋D pool，trace 1×／2× replay。
- [ ] Random／S3／S5／winner 对照。
- [ ] engine hit 真值、client TTFT／TPOT、attempt trace 三源对账。
- [ ] shadow gated decision，不 enforce。
- [ ] simulator vs real 排名一致性。

通过后才决定是否接 LLMClientV1 enforce／Decode D2。

### M11｜Production implementation，另立 RFC／MR

- [ ] selector 接入点单独拍板：Turbo／FlexLB master／LLMClientV1 SDK，不重复造全局 map。
- [ ] Client 保留 request ledger／SLO／grant／continuation owner。
- [ ] `off｜shadow｜enforce`，默认 off。
- [ ] required mode 仅 test workspace，防 fallback 盖问题。
- [ ] chain mock：timeout、stale view、P rollback、D reject、lease expiry、late frame、client crash。
- [ ] canary 看 completed goodput／waste／fairness，不只看 hit。

---

## 10. Fable review resolution

| Topic | Resolution | Design impact |
|---|---|---|
| Abstraction boundary | 接受，但不是全笛卡尔积 | 增加 validity matrix、capability requirement、三个 coupling |
| S3／S4 | 拆成 independent selector | 共享 `CapacityGate`／`BoundedFallback`，结果可归因 |
| Session proxy | 接受 strict online linker | 增加 hot-block exclusion／family cap；主看 skew／staleness |
| FlexLB | 先复刻 `0.7`，再 calibrated | v1 是解释 diff 的锚点 |
| Experiment order | LRU 排 selector，再扫 eviction | 避免全笛卡尔积 |
| Decode scope | D1／D1.5 是 implementation ceiling | D2 仅在 recovery gate 通过后做 spike |
| Goodput | strict completed 为主，lenient 为辅 | partial output 永远单列 |
| Fairness | v1 使用 tenant-weighted virtual runtime | quota／TPM ledger 暂不进入 overlay |
| R2 | 独立 Decode checkpoint capability | 不与 prefix Store 混为一体 |
| G-D2 | 增益＋尾延迟／waste／fairness 不劣化 | 增加 preemptions per completed request 硬上限 |

Review 额外提出 G1–G7，均已吸收到正文和 milestone：metric unit、synthetic block ID、cache visibility、replay mode、relative capacity gate、replica mechanism attribution、Mooncake Algorithm 1 具名对照。

---

## 11. References

- Mooncake paper：https://arxiv.org/abs/2407.00079
- Mooncake repository：https://github.com/kvcache-ai/Mooncake
- Existing v2 concept／insight doc：`/Users/zhangpeiwen/Downloads/tongyi/github/prefill-cache-sim/DESIGN.md`
- Fable review：`/Users/zhangpeiwen/.codex/attachments/aed241a6-3244-439c-8b5e-16decf9de6f9/pasted-text.txt`
- Decode lease handoff：`/Users/zhangpeiwen/dev-trees/DECODE_LEASE_HANDOFF.md`
- gated-PD handoff：`/Users/zhangpeiwen/Code/tongyi/github/HANDOFF-gated-pd-ttft-admission.md`
- Client KV／SLO admission：`/Users/zhangpeiwen/Downloads/client_token_scheduling_v0.2.md`
- KVS architecture memory：`~/.claude/projects/-Users-zhangpeiwen/memory/project_kvstore_architecture.md`
- Session affinity memory：`~/.claude/projects/-Users-zhangpeiwen/memory/project_inference_session_affinity.md`
- autoTPM memory：`~/.claude/projects/-Users-zhangpeiwen/memory/reference_autotpm_priority_admission.md`
- KV hit audit memory：`~/.claude/projects/-Users-zhangpeiwen/memory/reference_engine_vs_reported_cache_hit.md`
