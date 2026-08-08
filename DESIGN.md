# Prefill 路由 × KV Cache × SLO：从 basic 到 inspired 的统一设计

> 输入：Mooncake FAST'25 trace（`mooncake_trace.jsonl`，本目录）
> 语境：本地仓库实锤（rtp-llm centralized_master、dashscope-platform turbo、dashserving LLMClientV1、
> dashllm、vllm_pai、v6d、blade-kvt、tilert-serve-pd、Gated-PD 设计稿）
> 状态：**v2**。v1 的"策略分家式"设计压缩为 Solution 1；v2 的主体是一个统一目标函数
> （下称 **magic function**），把 KV cache 命中、request SLO、instance stats、PD reservation
> 放进同一笔账，并给出 模拟 → 实机部署 的完整实验路径。

---

## 0. 数据画像：三个不变的事实

对 trace（23,608 请求 / 1 小时 / 512-token block / 平均 17.3 block·8,590 tokens 每请求）：

| 事实 | 数值 | 设计含义 |
|---|---|---|
| **F1 理论命中率上限**（单节点∞容量） | **55.3%** | 一切算法的天花板，报告命中率必须归一化（达成率 = 实际/55.3%） |
| **F2 极端头部倾斜** | top-10 块占 22.9% 引用、top-100 占 34.4%；**78.5% 的块只被引用一次** | 热块复制几乎免费；大多数块 cache 了就是污染 → 准入 > 淘汰 |
| **F3 复用距离双峰** | p50 = 4 个请求（多轮会话立即复用）；p90 = 1,629 请求 / 247s | 短距复用靠亲和路由；长距复用靠聚合容量（全量唯一 KV 93.8M tokens，单机装不下） |

一个 v2 新增的度量观点：**token 命中率不是目标，GPU 秒才是**。命中长前缀省下的
prefill FLOPs 超线性于 token 数（后续 token 的 attention 随位置增长），所以指标体系里
要同时报 `FLOPs-weighted 命中率`（= 省下的 prefill GPU 秒 / 总 prefill GPU 秒）。两个
路由策略 token 命中率相同时，把命中集中在长前缀请求上的那个更值钱。

---

## 1. 现状盘点：为什么"策略分家"是错的，centralized_master 已经证明了一半

### 1.1 Turbo 的问题：N 个 selector 类，没有一笔总账

dashscope-turbo 把关注点拆成互斥的策略类：`CacheAwareScheduler`、`StickySessionScheduler`、
`PdPrefillScheduler`（四维容量）、长短隔离……每个类只看一个维度，维度之间靠配置串联和
优先级硬编码。后果：

- cache 亲和与负载均衡"打架"没有裁判——亲和策略把长会话黏死在热节点，容量策略再一刀切拒掉；
- SLO 完全不在路由决策里（EngineCapacityCalculator 的 `A·P+B` SLO 模型只用于容量规划，不用于选节点）；
- 没有任何决策考虑"这次路由会挤掉哪些 cache、值多少"。

**这不是缺一个更聪明的 selector，是缺一个公共的目标函数。**

### 1.2 centralized_master 的启发：半个 magic function（代码实锤）

内部中心化 master 的 `ShortestTTFTStrategy` 是本地代码里最接近答案的实现（具体路径脱敏）：

```java
// centralized master .. TaskInfo.estimatePrefillTimeMs
prefillTime = tokens * 1.0 - hitCacheTokens * 0.7      // 命中 token 打 3 折
// centralized master .. ShortestTTFTStrategy.scoreWorkers
TTFT(worker) = prefillTime + worker.runningQueueTime    // ← 相对 TTFT
// selectBestWorker: TTFT 升序 → top 30% 候选 → max(0.1·minTTFT, 0.5·stddev) 相似带
//                → lastSelectedTime 公平性 CAS
```

它已经把 **cache 亲和（hitCacheTokens）、负载（queueTime）、公平性（lastSelectedTime）
统一进一次打分**——session affinity 在这里是涌现行为而非独立策略：会话上一跳的节点
天然 hitCacheTokens 最大，自动被选中。这证明了统一函数的可行性。

但它只是半个答案，四个结构性缺陷：

| 缺陷 | 具体表现 | v2 的修法 |
|---|---|---|
| D1 延迟模型是拍脑袋线性 | `1.0 ms/token`、命中折扣固定 0.7（命中 token 仍计 30% 成本，未标定） | forward-pass 模型 + 实机校准（§3.2） |
| D2 无 SLO 语义 | 所有请求同权，最小化绝对 TTFT ≠ 最大化 SLO 达成 | slack 归一化 + 违约概率项（§3.3） |
| D3 只看 P，不看 D | decode 侧容量/TPOT/reservation 完全不在函数里 | D-reserve 项 + TPOT 约束（§4） |
| D4 与淘汰/准入无关 | 路由是路由，engine LRU 是 LRU，两本账 | cache 外部性项 + 统一价值函数（§3.4） |

### 1.3 其余现成积木（v2 只依赖这些，**不依赖 DashNext**）

- 前缀哈希已在数据面：api-server `PrefixUtils.hashTokens()`（murmur3 多长度）→ `x-prefix-hash`
- vLLM 块级事件底座：`ExternalBlockHash` / KV events（stored/removed 可上报）
- 引擎命中真值：`Prefix cache hit rate` stats 行 + `ds-ms-stats prompt_cached_token_num`（实验闭环用）
- v6d `KVCacheStoreWrapper`：`get_matched_num()/store_layer()/load()`（L2 层）
- blade-kvt / Mooncake push-direct：KV 传输的两种数据面（Current PD / TileRT）
- LLMClientV1 4-tier decode 路由 + `instance_selector.rs`（Rust 侧留白 = 我们的注入点）
- tilert-serve-pd `/pd/reserve`（D-first reservation）与 Gated-PD probe/COMMIT（P-only TTFT 预测 + 准入）

---

## 2. Solution 1（basic 层）：模拟框架抽象 + 基线策略

> 这是任务 1–4 的直接答案与实验底座，v1 内容压缩保留。它的价值：定义领域模型和
> 可插拔接口，让 §3 的所有高级策略都能作为同一接口的实现被公平对比。

### 2.1 领域模型与接口

```
Request:  req_id, arrival_ms, input_tokens, blocks(hash_ids 前缀链), output_tokens
          + slo: {ttft_ms, tpot_ms, tier}          # v2 新增，trace 无此字段 → §6.1 合成

ClusterView（决策时可见的世界，可配置精度）:
    ORACLE（完美实时）| MIRROR（路由侧影子副本，无淘汰上报会漂移）| STALE(Δt)（周期上报）
    node_view(n): capacity/used, inflight_reqs/tokens, running_batch_tokens,
                  cached_prefix_len(blocks), d_pool_view(decode 侧容量/预约水位)   # v2 新增

接口 1  PrefillNodeSelector.select(req, view) -> NodeId | REJECT     # v2: 可返回拒绝
接口 2  BlockEvictionPolicy: on_insert/on_access/evict(need)/admit(block, meta)
接口 3  PrefillHandler.handle(req) -> PrefillResult{node, hit_blocks, hit_tokens, evicted,
                                                    ttft_est, slo_met}
接口 4  Simulator(trace, N, capacity, selector, eviction, view_mode) -> 指标集
```

语义约定（写进实现）：**连续前缀命中**（中断即停）；**inflight 块 pin 住不可淘汰**；
命中率分子只算从块 0 起的连续段。

### 2.2 基线实现矩阵

| Selector | Eviction |
|---|---|
| Random（任务基线）/ RoundRobin / LeastLoaded | FIFO（任务基线）/ LRU |
| PrefixHash（`consistent_hash(blocks[0..k])`，零同步） | LRU-ChainAware（只逐前缀链叶子） |
| SessionAffinity（最长已见前缀查表） | SecondRefAdmission（ghost list，第二次引用才缓存，打 F2） |
| CacheAware（`α·cached_len − β·load`，≈centralized_master 简化版） | TTL（标定 F3 的 247s 窗口） |

指标：token/block/FLOPs-weighted 命中率、归一化达成率(/55.3%)、per-node 负载方差、
淘汰浪费率（被逐后又被引用/总淘汰）、有效容量利用率、**SLO 达成率（p99 slack）**。

Solution 1 的定位：跑出三个"定价数字"——路由 vs 淘汰的贡献差、STALE(Δt) vs ORACLE 的
视图折旧、SecondRefAdmission 的准入收益——它们是给 §3 的高级设计定投入优先级的标尺。

---

## 3. The Magic Function：一个函数，五种决策（v2 主体）

### 3.0 核心主张

系统里全部五种决策——**路由、淘汰、准入(early reject)、D-reserve、复制/预取**——本质上
都是同一种赌注：**花不花这一份 GPU 秒，换不换那一份 SLO 概率**。所以不该有五套策略类，
应该有一个统一的账本单位（GPU 秒）和一个目标函数，五种决策是它在五个决策点的求值。

```
J(r, π) =  C_compute(π)                    # 重算未命中段的 GPU 秒
         + C_transfer(π)                   # 从邻居/L2 拉 KV 的传输时间成本
         + C_interference(π)               # 挤占目标节点在跑请求的代价（batch 变慢）
         + λ_ttft(r) · P[TTFT(π) > slo.ttft]      # SLO 违约概率 × 该请求的违约价格
         + λ_tpot(r) · P[TPOT(π) > slo.tpot]
         + ΔV_cache(π)                     # cache 外部性：被挤掉的块价值 − 新驻留块价值

π = 一个"执行计划"：去哪个 P 节点、前缀链各段从哪来（本地 HBM / 邻居 / L2 / 重算）、
    何时以何种方式锁 D（P-first / D-first reserve / gated probe）

决策 = argmin_π J(r, π)；若 min J 超过该请求的价值预算 → REJECT（early rejection 是特例）
```

关键点：**centralized_master 的 ShortestTTFT 是 J 在
`λ=0, ΔV=0, C_transfer=0, C_interference=0, P(·) 退化为期望值` 下的特例**。v2 不是推翻
它，是把它没记的账补上。下面按"从 basic 到 advanced"分四级展开，每一级都可独立落地，
且是上一级的严格超集——这就是渐进路线。

### 3.1 M1（basic+）：可标定的相对 TTFT —— centralized_master 重建版

```
TTFT_est(r, n) = queue_delay(n)                          # instance stats: 在排 tokens / 吞吐
               + kv_fetch(π)                             # M1 先取 0（只用本地命中）
               + forwards(r, n) × step_time(node_state)

forwards(r, n) = ceil( (input_tokens − cached_consecutive(r, n)) / chunk_size )
step_time(·)   = 查表：实机测得的 每-forward 耗时 ~ f(running_batch_tokens)   # 非线性，校准得来
select         = argmin TTFT_est，加 centralized_master 式相似带 + 公平性 tie-break（防抖动照抄）
```

对 centralized_master 的两处修正：① `forwards×step_time` 替换 `1.0·tokens`（EngineCapacityCalculator
的 forward-pass 视角 + chunked prefill 语义，命中 token 成本自然为 0 而非 0.7 折）；
② `step_time` 是**实测校准表**不是常数——这正是"找机器建部署"阶段的第一个产出（§6.2）。

M1 的性质：确定性、无状态可复制、单次打分 O(N)。它已经统一了 cache 亲和 + 负载 + 会话
亲和（涌现），可以直接作为 `instance_selector.rs` 的第一版实现。

### 3.2 M2：SLO 归一化 + D 侧进账 —— "最小化违约概率，不是最小化延迟"

M1 的隐藏错误：最小化绝对 TTFT 会把好节点浪费在宽 SLO 请求上。改成按 **slack** 记账：

```
slack(r, n)  = (slo.ttft − TTFT_est(r, n)) / slo.ttft          # 归一化余量
risk(r, n)   = P[TTFT > slo.ttft]  ≈  Φ((TTFT_est − slo.ttft)/σ_n)   # σ_n: 该节点 TTFT 估计误差
                                                                      # （校准阶段直接测残差分布）
score(r, n)  = risk(r, n) + β·risk_tpot(r, d_pool)             # D 侧同理进账
select       = argmin score
```

行为变化（这是 magic 开始显形的地方）：

- **紧 SLO 请求自动获得 cache 亲和优先权**——它们经不起重算，risk 对 cached_len 最敏感；
- **宽 SLO 请求自动成为负载均衡的填充物**——去冷节点重算 risk 依然≈0，于是系统自发把
  "亲和造成的热点"用宽 SLO 流量对冲。**亲和 vs 均衡的矛盾不再靠拍系数 α/β，而是被 SLO
  异质性天然化解**。这是分家式 selector 结构上做不到的。
- D 侧：`risk_tpot` 用 decode 池的 running tokens/卡 与 reservation 水位估计。若所有 P 选择
  都无法保 TPOT（D 池满）→ score 全体饱和 → 触发 REJECT，比"P 白算完发现 D 没位"省一次
  prefill——**Mooncake early-rejection 的教训在函数里自动出现，不需要单独设计**。

### 3.3 M3：cache 外部性 + 统一块价值 —— 路由和淘汰合并成一本账

定义**块价值**（唯一的新概念，之后处处复用）：

```
V(b) = p_reuse(b) × recompute_cost(b) / replicas(b)

p_reuse(b): 复用概率估计。basic 版 = 分段常数：
    新块（未复用过）      → 0.2      # F2: 78.5% 一次性 → 先验 ≈ 0.215
    复用过 ≥1 次          → 0.7      # 复用是强信号（TinyLFU 直觉）
    距上次访问 > 300s     → 衰减 ×0.3 # F3: p90 复用窗口 247s
    advanced 版 = logistic(链深度, 引用次数, 距上次访问, 会话活跃标志)，trace 可直接拟合
recompute_cost(b): 该块 512 token 在其链位置的 prefill GPU 秒（位置越深 attention 越贵）
replicas(b): 集群内副本数（多一份副本，单份价值摊薄）
```

然后三件事同时被这一个 V 解决：

```
淘汰   evict = argmin V(b)（pin 块除外）
         ▸ FIFO/LRU/TTL/SecondRef 全是 V 的退化特例：
           LRU = 只用"距上次访问"；SecondRef = 对 p_reuse 的两段先验；TTL = 对衰减项的硬阈值
准入   admit(b) ⇔ V(b) > min V(驻留块)（放进来必须比挤出去的值钱；78.5% 一次性块自动被拒）
路由   J 里的 ΔV_cache(π) = Σ V(被 π 挤掉的块) − Σ V(π 新写入块)
         ▸ 效果：把长会话路由到一个会挤掉一堆热 system-prompt 块的小容量节点，会被这一项
           罚出局——**路由第一次开始对"它对 cache 版图的破坏"负责**
复制   对块 b 在节点 m 加副本 ⇔ b 在 m 的预期命中流量 × 单次节省 > m 上的 min V（挤占成本）
         ▸ top-100 热块（51K tokens）在任何节点都满足 → F2 的"免费午餐"自动成立
预取   会话 decode 进行中 → 下轮 p_reuse ≈ 1 → 提前 pin/预热，把 F3 的 p50=4 复用变 100% 命中
```

**这就是对"CacheAware 和 SessionAffinity 有没有综合方式"的最终回答：不但路由内部综合了，
路由和淘汰、准入、复制也综合成了同一个函数的不同投影。**

### 3.4 M4（inspired 终局）：KV 组装计划 + 全局放置

M3 仍是贪心的（逐请求局部最优）。M4 把 π 的搜索空间放开：

- **prefill = KV 组装计划**：一条前缀链的不同段可以来自不同源（本地 HBM 段 + 邻居 RDMA 段
  + v6d L2 段 + 重算段），`C_transfer` 与 `C_compute` 在段粒度上权衡。拉取赢过重算的分界：
  `bytes(b)/BW_rdma < recompute_seconds(b)`——对深链位置的块几乎总是拉取赢（KV 字节数固定，
  重算成本随位置涨）。这是 Mooncake Store 的思想，但用 J 表达后它不是新架构，只是 π 的
  更大候选集。
- **全局放置**：一个轻量 ledger（单 region 单 shard 足够：本 trace 块写入 ~100/s）维护
  block → 副本位置，周期性（秒级）跑副本数再平衡 = 对全体块按 V 排序做水位调整。
  路由用的还是同一个 J，只是 ClusterView 从"每节点独立视图"升级为 ledger 全局视图。
- **decode-side KV 复用**：多轮会话本轮 output 是下轮 input 的一部分——decode 节点产生的
  KV 登记进 ledger，下轮 prefill 的组装计划可以引用它（拉回或就地）。这是 trace 里
  看不到（hash_ids 只覆盖 input）但线上真实存在的额外复用空间，M4 独有。

M4 的验证顺序：**先在模拟器里用 ORACLE 视图算出它相对 M3 的上限增益**，超过 ~5pp 再谈
工程投入，否则停在 M3。

---

## 4. PD 贯穿：reservation 的经济学（D-reserve / TileRT / Gated-PD 进账）

> 素材实锤：《Current PD × TileRT × Gated-PD》设计稿（tilert-pd-artifacts, 2026-08-03）。
> 三种形态不是三选一的架构信仰，在 J 的视角下它们是**同一个 reservation 决策的三个定价点**。

### 4.1 三形态在 J 里的位置

| 形态 | reservation 语义 | 在 J 里等价于 |
|---|---|---|
| **Current PD**（P-first） | 无 D admission gate，P 算完靠 Pkg⓪ 建 D 流 | `C_reserve = 0`，但承担 `P[D 不可用] × wasted_prefill` 的尾部风险 |
| **TileRT**（D-first） | 进程级独占 reserve（单 active rid，TTL 120s/300s，busy 429） | 支付**整个 D engine 的持有成本**（reserve 期间 D 空转）换 `P[D 不可用]→0` + push-direct 的确定目标 |
| **Gated-PD**（probe+COMMIT） | 请求级 ledger（`pending_gated_tokens` 可并发预测），P probe 后 client 算 P-only TTFT 可提前 reject | 花一次**廉价 probe** 买到 J 的低成本估计，再决定 COMMIT/ROLLBACK——**它就是 J 的两阶段求值** |

统一决策规则（per-request，而非全局定死）：

```
reserve D at t ⇔ P[D 池在 prefill 完成时刻无位] × wasted_prefill_seconds(r)
                > hold_cost(D, prefill_duration_est)

推论：
- D 池水位低 → 右边贵左边便宜 → P-first（Current PD）就是最优，reservation 是浪费
- D 池水位高 / 长 prefill（wasted 大）→ D-first / gated 胜出
- TileRT 的进程级独占使 hold_cost 极高 → 只适合"确定会执行"的请求（reserve 前应先过
  一遍 J 的 admission）；Gated-PD 的请求级 ledger hold_cost 低 → 适合作为默认形态
```

**J 让"P-first vs D-first"从架构选择降格为按请求水位定价的运行时决策**——这是把 PD 知识
贯穿进 magic function 的核心收益。

### 4.2 与 cache 的耦合点

- **Gated-PD 的 probe 就是 J 的天然载体**：P Pkg⓪ 返回 probe（cached_len、队列水位）→
  client（LLMClientV1，我们的地盘）在 COMMIT 前算完 J 的 TTFT/risk 项 → reject 时 P 尚未
  enqueue、D 未 add_request，**J 的 admission 分支有了零浪费的执行机制**。
- **TileRT push-direct 影响 C_transfer 的形状**：P 主动 push 到 armed buffer，传输与 prefill
  流水线重叠 → `C_transfer` 近似隐藏；Current PD 的 D-pull（Blade KVT load）在 decode 关键
  路径上 → 计入 TTFT。同一个 J，两种 KVT 形态只是参数不同。
- **decode-KV 复用（M4）在 TileRT 下更顺**：D-first 意味着会话的 D 节点先确定——下轮
  prefill 的组装计划可以把"从该 D 节点拉上轮 KV"作为候选段源，rid fence 机制已经提供了
  按请求的传输隔离。

### 4.3 如果引入 Mooncake Store：配对与准入的解耦

> §4.1 三形态有一个未言明的共同假设：**P→D 直连传 KV**（push 或 pull），所以才存在
> "配对时机"（reservation）问题。Mooncake Store（P 写本地 DRAM + 全局索引，D 调度到了
> 再 RDMA 拉取）恰恰拆掉这个假设——P 和 D 在时间上解耦，既不是 P-first 也不是 D-first，
> 而是 **"P-now, D-later"（无序化）**。

**核心判断：Mooncake 替代的是 Gated-PD 的"配对"半边，不是"准入"半边。**

| Gated-PD 的功能 | Mooncake Store 下还需要吗 |
|---|---|
| **配对承诺**（COMMIT 时锁定 P→D 直连目标，防 D 无位时 P 白算） | ❌ 不需要。KV 停在 store 里等 D，"prefill 完成时 D 池无位"不再导致白算 |
| **准入/early rejection**（probe 算 P-only TTFT，超 SLO 提前拒） | ✅ 仍然需要。Mooncake 论文自己就有 Conductor early rejection——过载时 KV 停在 store 不免费（DRAM 占用 + 最终还是要 decode） |

对 §4.1 定价规则的影响：`reserve ⇔ P[D无位]×wasted_prefill > hold_cost` 在 Store 下
**左边趋近 0**（白算风险结构性消失），换来的新账单是：DRAM 容量成本 + store 淘汰风险
（KV 被逐出→重算）+ decode 起步时的 pull 延迟进 TTFT。

#### 可行路径全集（pingpong 素材，给下一个 agent）

| 路径 | 形态 | gate 存在形式 | 优势 | 代价/风险 | 适用期 |
|---|---|---|---|---|---|
| **A. Current PD** | P-first 直连，无 gate | 无（timeout 兜底） | 零改动 | 尾部白算风险、retry storm | 现状基线 |
| **B. Gated-PD** | 直连 + probe/COMMIT | 请求级 ledger，**测量式**拒绝（probe 拿实测 cached_len/水位） | 不需要任何 store 基建；client 侧就能落（我们的地盘）；probe = J 两阶段求值 | COMMIT 后仍硬配对——D 必须在 prefill 时长内出位 | **近期首选** |
| **C. TileRT D-first** | 进程级 reserve + push-direct | reserve TTL 即 gate | 传输与 prefill 重叠、确定目标 | hold_cost 极高（D 空转），单 active rid | TileRT 引擎专用，niche |
| **D. Mooncake Store 全解耦** | P 写本地 DRAM + 全局索引，D 拉取 | gate 退化为 router 处的**预测式**准入（无 per-request 握手） | 白算问题结构性消失；D 选择推迟到 decode 调度时（信息更新鲜）；**M4 的组装计划/decode 侧复用只有这条路能做**；HBM+DRAM 两级淘汰统一进 V(b) | 要建 store 基建（v6d/blade-kvt 有部分积木但不是完整 store）；pull 延迟进 TTFT；DRAM 成本；KV 生命周期管理 | M3/M4 期终局 |
| **E. Hybrid：让 J 选** | 传输/配对方式本身进入 π | probe 保留为测量工具 | `π ∈ {直连push(D 现在有位, TTFT 最低), 停store(D 忙, 零白算, D 延后选), REJECT(都不行)}` per-request 定价 | 两套传输路径并存的复杂度 | magic function 哲学下的诚实答案 |

#### 倾向（待 pingpong 确认）

1. **B → D 不是二选一，是时间轴**：B 零基建、client 侧可控、迭代最快，是 Phase C 实机的
   载体；D 是 M4 终局（decode 侧 KV 复用、全局 ledger 都依赖 store 层存在）。**B 的 probe
   机制在 D 里也不浪费**——store 化后 probe 从"配对握手"退役，转岗为"给 J 喂实测数据 vs
   依赖 5s 陈旧视图"的校准通道。
2. **别为了 Mooncake 提前弃 gate**：预测式准入（D 路线）依赖 ClusterView 够准；测量式
   准入（B 路线）不依赖。在视图新鲜度 Δt≈5–10s 的现实下，probe 的实测值更稳。
3. **E 是设计上的正确答案**："要不要 gate、走直连还是走 store"本身就该是 J 的一个决策
   维度，而不是架构信仰——这和 §4.1 把 P-first/D-first 降格为定价决策是同一个动作的延续。

### 4.4 timeout 体系是 J 的运行时护栏

设计稿里的 timeout 链（HANDSHAKE 300s / prefill bucket / DECODE_CONNECT 240s floor /
FIRSTPKG / TPOT 30s / TileRT RESERVE_TTL 120s）在 v2 视角下的意义：每个 timeout 都是对
J 某一项估计错误的止损。实验阶段（§6）应把"timeout 触发率"作为 J 估计质量的反向指标——
估得准，护栏就不该被撞。

---

## 5. 性能 × 命中 × 成本三角：把取舍变成一条 Pareto 前沿

目标函数定了，剩下的取舍是**预算旋钮**。统一记账单位：GPU 秒（成本侧再乘单价）。

| 旋钮 | 花的是什么 | 买到什么 | 定价方式（模拟器输出） |
|---|---|---|---|
| N（P 节点数） | 卡 | 吞吐 + 聚合 cache 容量 | 命中率-vs-N 曲线（F3：聚合容量吃长距复用） |
| 单节点 HBM cache 预留 | 挤占 batch 的显存 | 命中率 | 拐点曲线（v1 预判：聚合 ~20M tokens 后归零） |
| L2 容量（v6d/DRAM） | 内存 + 拉取带宽 | 长距复用（F3 p90） | `L2 命中增量 × recompute_saved` vs 内存成本 |
| 副本因子（热块） | 容量摊薄 | 热点解耦 + 均衡自由度 | V(b) 复制判据直接给出最优 top-K |
| 视图新鲜度 Δt | 上报带宽/工程复杂度 | 决策质量 | STALE(Δt) vs ORACLE 的命中差（v1 预判：10s 几乎免费） |
| D reservation 形态 | D 空转 / probe 往返 | TPOT SLO + 防 P 白算 | §4.1 规则的触发率分布 |
| λ（SLO 违约价格） | 命中率/吞吐 | SLO 达成率 | **扫 λ 得到 吞吐-vs-SLO 达成的 Pareto 前沿**——这是给容量规划的最终交付物 |

最终报告形态：一张三轴图（GPU 成本 / FLOPs-weighted 命中率 / SLO 达成率），M0–M4 各是
图上一条前沿线。**"某策略好不好"从此变成"它的前沿在不在别人外面"**，不再吵参数。

---

## 6. 实验路径：模拟 → 校准 → 实机（对齐"结合数据集、找机器建部署、拿结果"）

### Phase A：模拟器（纯 Python，接口即 §2，其他 agent 可直接领走）

1. 实现 M0（Random+FIFO）→ M1 → M2 → M3；M4 只做 ORACLE 上限版。
2. **SLO 合成**（trace 无 SLO，必须造）：三档 tier——
   `strict`(TTFT 1.5s / 20%)、`standard`(4s / 60%)、`relaxed`(15s / 20%)，
   叠加规则 `slo.ttft ≥ input_tokens × 校准前的名义 step_time × 1.5`（避免物理不可达）。
   档位比例是 pingpong 参数。
3. 输出 §5 的定价表 + 三个关键数字（路由 vs 淘汰贡献差 / 视图折旧 / 准入收益）。

### Phase B：校准（1 台真机，半天量级）

| 校准对象 | 方法 | 喂给 |
|---|---|---|
| `step_time(batch_tokens)` 表 | 单实例扫 batch 压力，读引擎 stats | M1 的 TTFT 模型 |
| TTFT 估计残差 σ | 同上，逐请求 est vs 实测 | M2 的 risk 项 |
| KVT 带宽/延迟（blade-kvt & Mooncake push） | kvt-pressure-test 现成 skill | C_transfer |
| chunk_size / HBM cache 实际预留 | 部署配置直读 | 模拟器容量档 |

### Phase C：实机对照（N=4 prefill + decode 池，Spectrum 测试 workspace）

1. **路由注入点**：首选一个独立轻量 router（Python/Rust 单进程，前置于 N 个 prefill），
   实现 M1/M2 打分——迭代最快、不动生产件；第二步再沉淀进 `instance_selector.rs`
   （LLMClientV1 是我们自己的地盘，Tier 结构现成）。**不走 DashNext。**
2. **回放器**：按 trace timestamp 回放（可时间压缩 2–4×），hash_ids → 确定性合成 prompt
   （同 hash 同内容，保证引擎 prefix cache 真实命中）。
3. **真值闭环**：命中率读引擎 `Prefix cache hit rate` stats 行 + `ds-ms-stats
   prompt_cached_token_num`；TTFT/TPOT 读回放器端到端 + `ds-ms-stats`；对照模拟器预测值
   ——**模拟-实机偏差本身是交付物**（校准模型的可信度）。
4. 对照组：Random（基线）vs M1 vs M2，各跑同一 trace 段；有余力加"engine 原生 LRU vs
   SecondRefAdmission"（后者需 dashllm 侧小改，评估后再定）。

### 里程碑判据

- A 完成：M2 相对 Random 的命中率增益、SLO 达成增益出数；
- B 完成：TTFT 估计 p50 误差 < 15%；
- C 完成：实机复现模拟趋势（排序一致即可，绝对值允许偏差）→ 决定 M3 投入。

---

## 7. Pingpong 议题（v2）

1. **SLO 合成档位**（§6.1 的三档比例/数值）你来定，或者直接从 ds-ms-stats 抽一份线上
   TTFT 分布反推真实 tier 结构？
2. **M2 的 risk 形式**：Φ(正态) 够不够，还是直接用校准残差的经验分位数（无分布假设）？
3. **Gated-PD 作为 J 的执行载体**（§4.2）：这条把 magic function 落进你现有 Gated-PD
   工作流的路径，是否就是实机 Phase C 之后的正式形态？若是，probe 包里还缺哪些字段
   （cached_len 有了，D 池水位要不要进 probe）？
4. **实验机型**：Phase B/C 用哪个池（L20 / H20）、N 取 4 还是 8、decode 池借现有测试
   部署还是新建？
5. **Mooncake Store 立项时机**（§4.3）：Phase C 之后是止步 M3（Gated-PD 直连 + probe
   准入），还是立项 store 层走 D/E 路线（解锁 M4 的 decode 侧复用与全局 ledger）？
   若立项，store 用 v6d 扩建还是引入 Mooncake Store 开源实现？
5. **M4 上限值多少才值得做**：我提议 ≥5pp（相对 M3）为启动线，你的阈值？
