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

- [x] `Request`／`NodeSnapshot`／`ClusterView`／`Selection`。
- [x] `PrefillNodeSelector`／`BlockEvictionPolicy`／`CacheTopology`／`PrefillRequestProcessor`。
- [x] `ContinuationPrefixBuilder`，覆盖 partial input tail、generated chain 与 multi-round continuation。
- [x] continuous-prefix lookup。
- [x] pinned block／placement／eviction／stranded accounting；当前 request 不允许 self-eviction。
- [x] `CACHE_ONLY／QUEUED` 两种 replay mode；event loop 用 heap 管 arrival、prefill start／finish、cache insert／evict。
- [x] `INSERT_AT_COMPLETION／INFLIGHT_DEDUP_WAIT` visibility；wait token／ms 独立记账。
- [x] `ORACLE`／`DELAYED` view＋per-node lossy cache stats；与 schema 1.0.0 vocabulary 对齐。

Invariant tests：capacity 不超；pinned 不驱逐；hit 不越过首 miss；logical／attempt token 不重复计。

### M3｜Baseline selectors／eviction

- [x] S0 Random、S1 RR、S2 LeastWork。
- [x] 共用 `CapacityGate`：hard capacity＋`primary_load > α×cluster_mean` soft break。
- [x] 共用 `BoundedFallback`：primary／secondary range＋LeastWork fail-open。
- [x] S3 GBPrefixBucket：静态 prefix ownership 薄壳。
- [x] S4 SessionAffinity：online linker＋hot-block exclusion＋family cap 薄壳。
- [x] S5 FlexLB：公式 v1＋top30／similar band／last-selected fairness。
- [x] S6 CalibratedTTFT：先用 normalized work model。
- [x] E0 FIFO、E1 LRU、E2 SLRU、E3 SecondHitAdmission。
- [x] 每次 selection 保留 component trace，可按 request sample debug。

交付：A1／A2／A3 CSV＋图＋策略排名；CSV schema 从这里稳定，含 trace／config／git SHA、metric unit、replay mode、view mode、visibility；明确 fixed-total 与 fixed-per-node。

### M4｜Base result review

- [x] 验证 PrefixAnchor sensitivity，禁止只留 k=2。
- [x] 检查 hit gain 是否由单节点 overload 换来。
- [x] 检查 SessionAffinity proxy 是否把 shared system prompt 错当 session，验证 hot-block exclusion／family cap。
- [x] S4／S5 比较 skew、fallback、stale-view stability；实测 S5 没有形成 Pareto gain。
- [x] A1 用 replica factor 解释 Random fragmentation 和 affinity 去重机制。
- [x] 对 Top-3 做 8-seed／stale-view replay。
- [x] 选择进入 KVS phase 的 2 个 selector：S4 production-shaped candidate＋S3 cache ceiling control。

Stop：如果 S5／S6 没有 materially expand S3／S4 的 hit-skew frontier，不进入复杂 selector implementation。判据使用 0.1pp hit epsilon；不是单轴只比 S3 hit。

M4 结论：stop gate 已触发。S5／S6 暂停；M5 只带 S4／S3。完整 97-case CSV 与报告见 `results/m4/`。

M1–M4 作为独立可交付：S0／S1／S2＋FIFO／LRU＋正式 hit report 已完整回答 original task，不等待后续 KVS／SLO phase。

### M5｜KVS topology

- [x] `SHARED_KVS`／`HYBRID` lookup／placement。
- [x] local／remote／recompute 分账。
- [x] 参数化 KVT fixed latency／bandwidth／overlap。
- [x] remote store capacity／replica／eviction。
- [x] KVS timeout／unknown fail-open。
- [x] hot block transfer hotspot metric。
- [x] `S6-KVS／MooncakeAlgo1` 严格复刻＋`kvcache_balancing_threshold` sensitivity。

交付：bandwidth／capacity sensitivity；回答“要不要借 GPU 量 KVT”。

M5 结论：shared tier 在当前 modeled capacity 下相对 `LOCAL_ONLY` 降低约 4.7% prefill cost；这是“增加 shared tier”的结果，不是 equal-capacity 胜负。最差 KVT corner 仍有约 1.03% modeled gain。M6–M8 排名不需要先借 GPU／RDMA；真实 KVT calibration 留给 M9。完整 30-case CSV 见 `results/m5/`。

### M6｜Time／SLO model

- [x] normalized service model：uncached prefill work、decode token work。
- [x] `RelativeSloOverlay`。
- [x] `TieredSloOverlay`＋tenant overlay＋独立 RNG streams。
- [x] PD／DP／gated-PD／gated-DP lifecycle state machine。
- [x] hold／reserve／commit／rollback／reject／waste accounting。
- [x] strict／lenient completed goodput、partial output、SLO、fairness metrics。
- [x] scalar 权重进入 provenance，并做 sensitivity。

交付：同一 selector 下四 protocol Pareto；不使用未经校准的“真实 ms”措辞。

M6 结论：五种 protocol 已形成不同 timeline／resource ledger，deadline、partial、waste、hold、rollback 与 store cost 均做 conservation test。所有结果只使用 `NORMALIZED_WORK`；49-case CSV 见 `results/m6/`。经过三轮 Opus review 后通过 milestone gate。

### M7｜Decode lease simulation

- [x] D0 no lease。
- [x] D1 fixed lease，复现当前 `8／8／14`、max-round、kill-switch、failure downgrade invariants。
- [x] R0 continuation 通过 `ContinuationPrefixBuilder`＋topology lookup 计算 recompute cost。
- [x] R1 exact generated prefix 通过同一 synthetic block chain 计算 KVS-assisted cost。
- [x] D1.5 adaptive lease，只在 attempt boundary 调整。
- [x] fairness debt／minimum quantum／bounded preemption。

交付：D0／D1／D1.5 completed goodput、stall、waste、fairness。

M7 结论：shared K-server queue 可以表达真实 crossover。1 个 D node 的 severe HOL 场景中，D1 strict goodput 从 `0.00718` 提升到 `0.14181`；2 个 D nodes 时 D0 反胜，8 个 D nodes 时基本持平。结论是 pressure-aware enable，不是全局开启 lease。forced migration 的 R0／R1 recovery storm 单列为 sensitivity。完整 11-case CSV 见 `results/m7/`，Opus gate 为 PASS。

### M8｜Cooperative preemption spike

- [x] `DecodeCheckpoint` schema：epoch、seq、emitted tokens、KV handle、TPOT、served quantum。
- [x] Keep／Move decision＋safety margin。
- [x] D action grant：首期只 `report_checkpoint`／`abort_self`。
- [x] old epoch late frame drop。
- [x] max preemption／min quantum／aging。
- [x] `preemptions_per_completed_request` hard gate。
- [x] R2 global KV checkpoint 作为独立 topology，不能假设已存在。

进入条件：M7 通过 G-D1，且 R0／R1 recovery p95 有实测或可信校准。

M8 结论：single-threaded safety spike 已通过 Opus gate；epoch／grant／durability／output fence／minimum quantum／bounded preemption 均有反例测试。trace replay 只支持 ordinal claim：abort rate 随 safety margin 单调下降，且 recovery assumption 会移动 gate crossover。CSV 明示 `ASSUME_RESUME_SUCCESS_UPPER_BOUND`、`SYNTHETIC_SCORE`、`SYNTHETIC_UNCALIBRATED`；不能解读为真实 goodput 或 production-ready。完整 12-case CSV 见 `results/m8/`。

### M9｜1-GPU calibration

代码就绪，执行 BLOCKED_NO_ENGINE_ACCESS（本仓库无 accelerator 访问）：

- [x] 版本化 hardware gate、endpoint contract、transport seam：`calibration/hardware.py`（`HARDWARE_SCHEMA_VERSION = m9-hardware-v1`）、`calibration/transport.py`（`HttpJsonTransport` 拒绝 `file://`、`StubTransport`）、`calibration/http_endpoint.py`（`/v1/describe`＋`/v1/measure`、echo 校验）。CPU 可跑通 plumbing，不可产出 GPU calibration。
- [x] `scripts/run_m9_hardware.py`：无 `--endpoint-url` 时写出 `BLOCKED_NO_ENGINE_ACCESS` 报告到 `results/m9-hardware-blocked/`，exit 2；accepted 路径永不触及——synthetic 无法获得 `HW_CALIBRATED`／`HW_VALIDATED`（`endpoint_is_synthetic` exact-type 检查）。
- [x] atomic artifacts/manifests（`os.replace` staging，`MANIFEST.json` 最后写入）；`validate-only handshake`（`--dry-run` → `BLOCKED_DRY_RUN_ONLY`）；require complete `MachineProvenance`（`HardwareContext.complete`）。
- [x] tests：gate（accepted/blocked/round-trip/hand-edited rejected）、transport（scheme/timeout/stub）、endpoint（describe/measure/echo mismatch）、script blocked path。见 `tests/test_hardware.py`。
- [ ] **执行 G0→G1**：G0 = 1 台目标机型 engine access；G1 = 产出 accepted `results/m9-hardware/`（`HW_CALIBRATED`）。Tooling complete，stage not passed。当前环境不可执行。

机器：先 1 台目标机型；此阶段不需要 4P+4D。

M9 结论：hardware harness 代码就绪并测试（tooling complete）；但无 accelerator 访问，所有 blocker 以机器可读方式发布（`HardwareBlocker` StrEnum），`BLOCKED_NO_ENGINE_ACCESS` 为当前 honest 结果而非 failure。CPU 可验证 plumbing（handshake、gate 逻辑、artifact 原子写），不可产出 GPU calibration。Gate 顺序：G0 = engine access；G1 = accepted M9-HW（`HW_CALIBRATED`）。Tooling complete ≠ stage passed。

### M10｜Multi-GPU replay

代码就绪，执行 BLOCKED_NO_ENGINE_ACCESS（依赖 accepted M9-HW）：

- [x] 版本化 replay hardware gate（`replay/hardware.py`，`REPLAY_HARDWARE_SCHEMA_VERSION = m10-hardware-v1`）：frozen plan digest（`plan_digest(ReplayPlan()) == FROZEN_PLAN_DIGEST`）、tau-b gate、reconciliation fractions、fault-injection detection；accepted M9-HW prerequisite（`REQUIRED_CALIBRATION_TIER = HW_CALIBRATED`）。
- [x] `scripts/run_m10_hardware.py`：无 `--calibration`／`--observed` 时写 `BLOCKED_NO_ENGINE_ACCESS` 到 `results/m10-hardware-blocked/`，exit 2；modeled 侧永远 `SYNTHETIC_REPLAY`／`NORMALIZED_WORK`。
- [x] atomic artifacts/manifests；`HW_VALIDATED` 只允许出现在 accepted `GATE.json`；`_assert_no_stronger_claim` 逐 artifact 扫描，per-artifact permission；worst-cell aggregation（tau-b／reconciliation）。
- [x] tests：gate、report round-trip、hand-edited rejected、script blocked path、dry-run。见 `tests/test_hardware.py`。
- [ ] **执行 G2**：accepted M9-HW (G1) + measured engine bundles + M10-HW measured replay。Production rollout gates: R0 metrics → R1 3-day shadow → R2 canary（selector 与 baseline 差异可解释、无新增 error）。Tooling complete，stage not passed。当前环境不可执行。

通过后才决定是否接 LLMClientV1 enforce／Decode D2。

M10 结论：replay hardware gate 代码就绪并测试（tooling complete）；但无 engine 访问，`BLOCKED_NO_ENGINE_ACCESS` 为 honest 结果。synthetic 无法获得 `HW_VALIDATED`；modeled 侧永远 `SYNTHETIC_REPLAY`。Gate 顺序：G2 = M10-HW measured replay（依赖 G1 accepted M9-HW）。Production rollout: R0 metrics → R1 3-day shadow → R2 canary。Tooling complete ≠ stage passed。Q5/Q6/Turbo pull-mode unresolved（见 M11 RFC §10）。

### M11｜Production implementation，另立 RFC／MR

RFC 与 chain mock 部分（本仓库内可交付）：

- [x] selector 接入点单独拍板：向已持有 map 的一方（FlexLB master／Turbo `CacheAwareScheduler`）投递带版本号的 scoring function，不重复造全局 map。owner 差异在 mock 中可执行：`owner=NONE` 拒绝非 off 模式，Turbo pull 无 push preference 通道；pull 派发循环本身未建模。见 `docs/m11-production-rfc.md` §2。
- [x] Client 保留 request ledger／SLO／grant／continuation owner；WHERE 与 lifecycle 归属分离。`exclude_hosts` 只能来自本请求 failed-leg ledger；admission host 必须等于 owner 选中实例；admission 消耗 selection 与 prefill commit，sibling admission 被拒绝，decode reject 消耗本次 pending admission（同一决策不能重复 reject），因此每个 leg 都是 owner-mediated 且 attempt identity 唯一。见 RFC §3。
- [x] output 去重只在 `OutputAuthority` fence：decode leg 只持 leg-local seq／identity；旧 epoch／重复／乱序 frame 分计数丢弃；lease 边界推进 epoch，过期 leg 的在飞 frame 被 fence 挡下；`max_rounds` 耗尽同样先推进 fence 再拒绝，过期 leg 不能继续投递；`report_checkpoint` 绑定 reporting leg——无 leg／superseded／sibling 的 report 直接拒绝，checkpoint 只携带 leg-local epoch＋seq。见 RFC §7.5、§7.6。
- [x] crash 恢复只读持久状态：resume 位点 = durable output-ack，R2 checkpoint 只是 KV 材料；无 durable ack 则 fail closed。epoch 持久 source of truth 是 client durable ledger；controller 视图落后时 D2 fail-open 到 `REPORT_CHECKPOINT`，不抛异常。见 RFC §7.7、§10 Q5／Q6。
- [x] `off｜shadow｜enforce`，默认 off；retry budget 耗尽后拒绝一切新 attempt，只允许一个 `ERROR` terminal。见 RFC §4、§5、`ChainConfig.mode`。
- [x] required mode 仅 test workspace，防 fallback 盖问题；生产读到 `required` 直接拒绝启动，不静默降级；五类 fail-open reason（timeout／error／stale／capability／planner）一律**先计数、后硬失败**。见 RFC §4、§5、`RequiredModeRejected`。
- [x] chain mock：timeout、stale view、P rollback、D reject、lease expiry、late frame、client crash 七场景，另有敌意用例（superseded-leg race、sibling admission 拒绝、leg-bound checkpoint、max_rounds fence、无 durable ack、epoch 回退 view、伪造 epoch、budget 耗尽）。见 `src/prefill_cache_sim/chain/scenarios.py`、`tests/test_chain_mocks.py`。
- [x] fail-open 边界与 capability／version handshake 定义；未握手默认零 capability，不推断——未握手或缺 `PREFIX_CACHE_QUERY` 时 selection 的 hit 项按 0 计、退回 baseline 顺序且不发布 selector score；`served_quantum=0` 拒绝而非改写。见 RFC §5、§6、§7。
- [x] canary 看 completed goodput／waste／fairness，不只看 hit。见 RFC §8、§9。
- [x] R0/R1 decision logging：`chain/decision_log.py`。versioned privacy-safe `DecisionRecord`（online+shadow、feature/view/capability/fallback/timing、`logical_request_id`+`attempt_index`、enforcement OFF）；bounded-cardinality metrics（`DECISION_METRIC_NAMES` closed vocabulary）；append-only crash-safe JSONL sink with injectable clock；R1 `PairReporter`＋`DiffReport` with atomic artifact/gate；structural no-enforce（`DecisionEnforcementError` on `enforced=True`）；generic `PushObserver` protocol wired into `ChainHarness._finalize_selection`。Turbo pull shadow = `PairStatus.UNRESOLVED_OWNER_SIGNOFF`（machine-readable，需 owner sign-off；Q5/Q6 unresolved）。WHERE 与 lifecycle/output authority 分离已落实。见 `tests/test_decision_log.py`。

Production 部分（本次不做，保持未完成）：

- [ ] production MR：改 FlexLB master 或 Turbo 的实际 selector 代码。
- [ ] LLMClientV1 侧 `LegRoute` 字段落地与 owner 团队接口对齐。
- [ ] 真实硬件验证与 canary 执行（依赖 M9／M10）。
- [ ] D2 开闸评估（RFC §9 R4 独立 gate，当前保持关闭）。

交付：`docs/m11-production-rfc.md`＋`src/prefill_cache_sim/chain/`＋`tests/test_chain_mocks.py`。

M11 结论：RFC 已就三件事拍板——selector 接入点、lifecycle 归属、上线形态；chain mock 把 RFC §7 的不变量做成可执行断言，覆盖 fail-open（五类 reason 全部计数、required 模式先计数后硬失败）、stale／epoch 回退 view 降级、rollback 不产生用户可见 output、lease expiry 不消耗 retry budget 且在边界推进 epoch fence（`max_rounds` 耗尽也先 fence 再拒绝，过期 leg 不能继续投递）、decode leg 只持 leg-local seq 而去重只在 fence（superseded-leg race 有测试，sibling admission 被拒绝，每个 leg attempt identity 唯一且 owner-mediated）、`report_checkpoint` 绑定 reporting leg（absent／superseded／sibling report 拒绝，checkpoint 携带 leg-local epoch＋seq）、cache-aware 打分依赖握手协商到 `PREFIX_CACHE_QUERY`（否则 hit 项按 0 计、退回 baseline 且不发布 score）、crash 恢复只读 durable output-ack（无 ack 则 fail closed）、controller epoch 视图落后时 D2 fail-open 而非抛异常。**这是跨组件协议／状态机测试，不是 production E2E**；全部数值标记 `TruthBasis.SYNTHETIC_FIXTURE`，不可作为测量值发布。D2 在 mock 中保持 gated：闸关闭或 `COOPERATIVE_PREEMPT` 未协商时只可能产出 `REPORT_CHECKPOINT`，`hard_abort()` 无条件抛异常。未覆盖项如实列在 RFC §11：attempt 边界模式切换未测试，Turbo pull 派发循环未建模，durable ack 介质与 D2 re-registration 是 open question。production MR、硬件验证、canary 均未开始。

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
