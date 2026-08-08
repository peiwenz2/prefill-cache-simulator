# M11 Production RFC：cache-aware selector 接入与 decode lifecycle 归属

作者：张珮文
状态：RFC，未实现。本仓库只交付 RFC 与 chain mock，不含 production MR。
适用范围：DashScope PD 链路（Turbo／中心化 master／LLMClientV1）。

## 0. 这份 RFC 决定什么

本 RFC 只决定三件事，其余留给后续 MR：

1. cache-aware selector 的**接入点**（WHERE），以及为什么不再造一份全局 map。
2. decode **lifecycle 归属**（WHO owns identity／budget／continuation／output authority）。
3. 上线形态：`off｜shadow｜enforce`、fail-open 边界、capability handshake、observability、canary gate。

本 RFC 明确**不**决定：engine 内部 token scheduling、KV store 选型、D2 是否开闸。D2 仍然 gated，且只允许 cooperative checkpoint，不允许任意 hard abort。

## 1. 背景与既有事实

| 组件 | 现状 | 与 selector 的关系 |
|---|---|---|
| Turbo `CacheAwareScheduler` | **pull-based**：engine 按 cache bucket range 主动拉请求；Turbo 侧已持有 bucket→engine 映射与四维容量闸 | 已经是一份 map owner。没有外部 push 的接入位 |
| Turbo `StickySessionScheduler` | primary／alternative bucket，无 hard fallback | 亲和性已在同一份 map 内表达 |
| 中心化 master `ShortestTTFTStrategy` | **push-based**：`TTFT = queueTime + tokens − 0.7×hitCacheTokens`，top-30% 带内 + last-selected CAS 公平 | 已经是一份 map owner，且已经用到 cached tokens |
| LLMClientV1（Rust，dashserving） | 已上线 decode lease v1：`resolve_leg_route(ctx, snap) -> LegRoute`、`should_resched(ctx, snap, signals) -> Decision` 两个单点 seam | 已持有 logical request identity、attempt id、retry／resched budget、continuation |

DESIGN-v3 §7.1 已给出分工判断：engine 适合做 token scheduling 与 local preemption；Turbo／中心化 master 适合选 instance；只有 client 能把"当前 D 不值得继续"翻译成"带 checkpoint 的另一次 D attempt"，并保证用户 output 不重复。本 RFC 把这句话落成接口。

## 2. 决策一：selector 接入点

**决策：不新增 selector 服务，不新增全局 map。M11 向"已经拥有 map 的那一方"投递一个带版本号的 scoring function。**

| 部署形态 | selector 归属（WHERE） | M11 交付物 |
|---|---|---|
| 中心化 master 在链路上 | 中心化 master | 替换 `ShortestTTFTStrategy` 的 score 项，新增 `selector_score_version` |
| Turbo batching 为入口 | Turbo `CacheAwareScheduler` | 调整 bucket 排序权重，新增 `selector_score_version` |
| 两者都不在（直连） | 不接入 | 保持现状，`off` |

理由：

- **避免双份真相。** 中心化 master 与 Turbo 各自已经是权威 map。再造第三份 map 会引入 view skew，而 stale view 恰好是本 RFC 要防的故障之一（见 §7）。
- **selector 是无状态打分。** 它只需要 `(candidate, snapshot) -> score`，可以作为库嵌入现有 owner，不需要独立生命周期。
- **client 不做 placement。** client 只表达 preference，见 §3。

被否决的方案：

| 方案 | 否决理由 |
|---|---|
| 独立 selector service | 第三份 map；多一跳 RTT；stale view 面积翻倍 |
| client 直接选实例 | client 视野是单请求，没有全局负载；且会与 Turbo pull 模型冲突 |
| engine 自选 | engine 不知道别的 engine 的队列，只能做 local |

owner 归属在 chain mock 中是可执行行为，不只是文档字段。诚实口径：以下三条是 mock 覆盖的**全部** owner 差异——

- `owner=NONE` 时 `ChainConfig` 拒绝任何非 `off` 模式：没有 map owner 就没有接入点（`test_no_owner_means_no_selector_attach_point`）。
- `owner=TURBO_CACHE_AWARE`（pull-based）没有 push preference 通道：`prefer_host` 即使恰好等于选中实例也结构性不可达，一律计入 `selector_pref_ignored_total`（`test_turbo_pull_owner_has_no_push_preference_channel`）。
- Turbo 的 pull 派发循环（engine 按 bucket range 拉请求）本身**不在 mock 内**；bucket 排序权重如何进 `CacheAwareScheduler` 属于 production MR，未验证。

## 3. 决策二：WHERE 与 lifecycle 归属分离

这是本 RFC 最重要的一条边界。**"去哪台"与"这条请求归谁管"是两件事，不能合并。**

| 关注点 | Owner | 说明 |
|---|---|---|
| WHERE：候选打分、实例选择 | 中心化 master／Turbo | 已有 map owner。selector 只贡献 score |
| logical request identity | LLMClientV1 | `logical_request_id` 跨 attempt 稳定 |
| attempt identity | LLMClientV1 | 每次新 leg 一个 `attempt_index` |
| retry／resched budget | LLMClientV1 | lease expiry 不消耗 retry budget；budget 耗尽后**拒绝一切新 attempt**，唯一合法出口是一个 `ERROR` terminal |
| continuation 拼装 | LLMClientV1 | `original_input + sent_tokens` |
| output authority（去重 fence） | LLMClientV1 | `epoch + output_seq`。fence 是**唯一**去重点：decode leg 只持有自己的 local seq／leg identity，不查询 fence 计数器 |
| durable output-ack ledger | LLMClientV1（持久介质） | 用户**确实收到**的 seq 与 epoch 的唯一持久记录。重启恢复只读它；没有它就 fail closed，见 §7.7 |
| R2 decode checkpoint | R2CheckpointStore | KV 恢复材料，**不是** output 恢复位点。checkpoint 落后于 ack 不能回退流，超前于 ack 不能跳过未确认 output |
| SLO deadline 判定 | LLMClientV1 | 输入的 class→deadline 表见 §10 open question |
| grant 签发（D2，gated） | LLMClientV1 | engine 只能 `report_checkpoint`，不能自裁 |
| token scheduling／local preemption | engine | 不在本 RFC |

client 侧的表达形式：`resolve_leg_route` 返回的是 **preference／constraint**，不是 placement。

```text
LegRoute {
    prefer_host:      Option<HostId>,     // 亲和倾向，可被 owner 忽略
    exclude_hosts:    Vec<HostId>,        // 硬约束：只能来自本请求的 failed-leg ledger
    hint_input_tokens: Option<u64>,       // 让 owner 自己算 TTFT，不替它算
}
```

owner 可以**完全忽略** `prefer_host`。忽略必须计数（`selector_pref_ignored_total`），但不得报错。这条保证 client 不会变成隐形的第二个 selector。

`exclude_hosts` 也不是自由裁量：它只能引用本请求 failed-leg ledger 上的实例（当前即 decode reject 过的 host）。引用 ledger 外的实例是协议错误，计入 `exclude_hosts_unproven_total` 后拒绝——否则 exclusion 就成了变相 placement。同理，decode admission 时 client 报的 host 只是校验 token，必须等于 owner 选中的实例；不一致计入 `admission_host_mismatch_total` 并拒绝。

admission 与 rejection 都是一次性的：admission **消耗**本次 owner selection 与 prefill commit，活跃 leg 存在时 sibling admission 直接拒绝——新 leg 只能经 lease expiry／checkpoint grant／crash recovery 之一产生，而这三条路径都推进 attempt index，因此每个 leg 的 attempt identity 唯一且 owner-mediated。decode reject 拒绝的是"一次 pending admission"（selection＋prefill commit）并同样消耗两者：同一决策不能重复 reject，再次 reject 必须经 owner 重新选点并重新 commit。

## 4. Enforcement mode

四态，默认 `off`。模式是 owner 侧配置，不是 client 侧。

| mode | selector 是否计算 | 是否影响选择 | 允许环境 |
|---|---|---|---|
| `off` | 否 | 否 | 全部（**默认**） |
| `shadow` | 是，只记录 | 否 | 全部 |
| `enforce` | 是 | 是，可 fail-open 降级 | 生产 canary 起 |
| `required` | 是 | 是，**不允许 fail-open** | **仅 test workspace** |

`required` 的唯一用途：在测试 workspace 里让 fail-open 路径失效，暴露"平时被 fallback 盖住"的问题。生产环境若读到 `required`，必须拒绝启动并落一条 `selector_required_mode_rejected` 日志——**不允许静默降级成 `enforce`**，否则这条闸就白设了。

模式切换只在 attempt 边界生效，不影响在飞请求。诚实口径：chain mock 的 `ChainConfig` 一次构造后不可变，**不覆盖**运行中切换模式；这条边界目前是 production 要求，不是可执行证据。

## 5. Fail-open 边界

原则：**selector 不可用时退回既有 baseline，绝不拒绝请求，绝不 abort 在飞 decode。** 这是 DESIGN-v3 §7.3 的 F8。

| 故障 | `enforce` 下行为 | 计数器 |
|---|---|---|
| selector 打分超时 | 用 baseline score（现有 `ShortestTTFTStrategy`／bucket 顺序），**不发布 selector score** | `selector_timeout_total` |
| snapshot 过期超阈值 | 降级为 `shadow`，本次不生效 | `selector_stale_view_total` |
| snapshot epoch 回退（即使 age 很新） | 同 stale view：降级为 `shadow`，本次不生效 | `selector_stale_view_total` |
| capability 不匹配 | 该实例退出 selector 候选，仍可被 baseline 选中，**不发布 selector score** | `selector_capability_mismatch_total` |
| planner／grant 服务不可用（D2） | 退回 D1 固定 lease，**不 abort** | `preemption_fail_open_total` |
| score 抛异常 | 记录并用 baseline，**不发布 selector score** | `selector_error_total` |
| `COOPERATIVE_PREEMPT` 未协商（D2） | `report_checkpoint` 只回 `REPORT_CHECKPOINT` | `preemption_capability_fail_open_total` |
| controller epoch 视图落后（client 主导的 epoch 前进：lease expiry／重启之后） | `report_checkpoint` 只回 `REPORT_CHECKPOINT`，**不抛异常** | `preemption_epoch_desync_fail_open_total` |

明确禁止的降级动作：

- 禁止把 selector 故障转成 `REJECT` 或 429。
- 禁止把 selector 故障转成 hard abort 或 kill decode leg。
- 禁止在 fail-open 时静默不计数。
- 禁止为超时／异常的 selector 发布"它本来会选什么"——没打出来的分不存在。

`required` 模式下上述 fail-open 一律改为**失败**（拒绝该次选择／该次 checkpoint 并显式报错），覆盖全部五个 `FailOpenReason`——timeout、error、stale view、capability mismatch、planner unavailable——且**先计数、后失败**：required 的目的就是暴露 fallback 盖住的问题，把信号吞掉等于自毁。只在 test workspace 生效。

## 6. Capability／version handshake

selector 不能假设对端支持什么。握手在实例注册时完成一次，snapshot 里携带摘要。

| 字段 | 类型 | 语义 |
|---|---|---|
| `selector_protocol_version` | `u32` | 不兼容变更时 +1。owner 与实例必须相等，否则该实例降级为 baseline-only |
| `selector_score_version` | `str` | 打分函数版本，如 `m11-score-v1`。仅影响可比性，不影响兼容性 |
| `capabilities` | `set[str]` | 见下表 |
| `snapshot_epoch` | `u64` | 单调递增，用于 stale view 判定 |
| `snapshot_age_ms` | `u64` | 采样到使用的时延 |

capability 取值：

| capability | 含义 | 缺失时 |
|---|---|---|
| `PREFIX_CACHE_QUERY` | 能报 prefix hit token 数 | hit 项按 0 计，不猜 |
| `DECODE_LEASE_V1` | 支持 lease expiry 自然 `LENGTH` | 不下发 lease，走原有 max_tokens |
| `R2_CHECKPOINT_STORE` | 支持 decode checkpoint 存取 | R2 恢复不可用，退回 R0 重算 |
| `COOPERATIVE_PREEMPT` | 支持 `report_checkpoint` | D2 对该实例永久关闭 |

规则：

- **capability 缺失一律退化，不一律报错。** 只有 `selector_protocol_version` 不等才把实例移出 selector 候选。
- capability 不允许"推断"。没上报就是没有。可执行口径：mock 里未握手的实例默认 **零 capability**（`test_without_a_handshake_capabilities_default_to_zero`、`test_selection_without_a_handshake_scores_no_cache_hits`）——未握手或缺 `PREFIX_CACHE_QUERY` 时 selection 的 hit 项按 0 计，排序退回 baseline 顺序且**不发布、不推断** selector score；D2 gate 即使打开，没有协商到 `COOPERATIVE_PREEMPT` 的实例也只会得到 `REPORT_CHECKPOINT`。
- 版本协商结果要落一条启动日志，便于 canary 时对齐。

## 7. Chain 不变量（对应 chain mock 七个场景）

本仓库交付的 chain mock 逐条断言下列不变量。**这是跨组件协议／状态机测试，不是 production E2E**，不能当成上线证据。

| # | 场景 | 不变量 | 依据 |
|---|---|---|---|
| 7.1 | timeout | selector／planner 超时必须 fail-open 到 baseline；不得 `REJECT`，不得 abort；超时的 selector 不发布 score | F8 |
| 7.2 | stale view | `enforce` 遇到过期或 epoch 回退的 snapshot 降级为 `shadow`；client 只出 preference 不出 placement | §3、§5 |
| 7.3 | P rollback | `ROLLBACK_P` 计入 prefill waste；**不产生任何用户可见 output**；epoch 不前进；decode admission 之后不再允许 rollback | F7 |
| 7.4 | D reject | 容量拒绝算 fault：消耗 retry budget，产生新 attempt id；被拒 host 进 failed-leg ledger（此后才可 exclude）；重选走既有 owner，不由 client 指定实例；reject 消耗本次 selection 与 prefill commit，同一决策不能重复 reject；budget 耗尽后拒绝一切新 attempt，只允许一个 `ERROR` terminal | §3 |
| 7.5 | lease expiry | 中间 `LENGTH` 不外泄；continuation = `original_input + sent_tokens`；不消耗 retry budget、不 backoff；用户只看到一个 terminal；**lease 边界推进 output epoch**，过期 leg 的在飞 frame 被 fence 丢弃；expiry 要求存在活跃 leg；`max_rounds` 耗尽同样**先推进 epoch fence 再拒绝**（`lease_rounds_exhausted_total`），过期 leg 不能继续投递，此后唯一出口是一个 `ERROR` terminal | Decode lease v1、F6 |
| 7.6 | late frame | decode leg 只持有 leg-local seq／leg identity；去重只发生在 `OutputAuthority` fence：旧 epoch frame、重复 seq、乱序 seq 分别丢弃并分计数（`stale_epoch_frame_dropped_total`／`duplicate_frame_dropped_total`／`out_of_order_frame_dropped_total`）；`output_seq` 单调；无重复 output；宣称未签发 epoch 的 frame 是伪造，直接拒绝；`report_checkpoint` 绑定 reporting leg：无 leg／superseded／sibling leg 的 report 在进 controller 前拒绝，checkpoint 只携带该 leg 自己的 epoch 与 local seq | F6 |
| 7.7 | client crash | 重启只读**持久**状态：resume 位点 = durable output-ack，R2 checkpoint 只是 KV 恢复材料——旧 checkpoint 不能回退进度，超前 checkpoint 不能跳过未确认 output；没有 durable ack 时恢复 **fail closed**，不假装 volatile 内存幸存；存活的 decode leg 靠 lease 自然结束，不被 kill | R2、F6 |

跨场景断言：

- **D2 gate 关闭时，`enforce` 只可能产出 `REPORT_CHECKPOINT`，不可能产出 `ABORT_SELF`。**
- epoch 的持久 source of truth 是 client 的 durable ledger；`OutputAuthority` 是它的内存镜像。controller 只认识自己 grant 过的 epoch：client 主导的 epoch 前进（lease expiry／重启）之后，`report_checkpoint` fail-open 到 `REPORT_CHECKPOINT` 而不是抛异常；只有 controller 自己 grant 的 `ABORT_SELF` 同步推进两侧视图。
- 每个 decode leg 都是 owner-mediated 且 attempt identity 唯一：admission 消耗 selection 与 prefill commit，sibling admission 被拒绝，supersession（lease／grant／crash）都推进 attempt index。
- `served_quantum=0` 是非法进度，直接拒绝，不得被静默改写成默认 quantum。

## 8. Observability

必须能回答三个问题：selector 是否生效、生效后是否更好、坏了有没有被盖住。

按 `(mode, selector_score_version, deployment)` 打标。

| 指标 | 类型 | 用途 |
|---|---|---|
| `strict_completed_goodput` | counter（tokens） | **主指标**。只计完整完成的请求 |
| `lenient_completed_goodput` | counter | 与 strict 对比，看 cliff |
| `partial_output_tokens` | counter | **单列，永远不计入 goodput** |
| `prefill_waste_work` | counter | rollback／SLO miss 造成的浪费 |
| `jain_fairness` | gauge | 按 tenant normalized attained service |
| `per_tier_completion_ratio` | gauge | 低优先级是否被饿死 |
| `preemptions_per_completed_request` | gauge | D2 硬闸，上限 0.25 |
| `block_ref_hit_rate` / `token_weighted_hit_rate` | gauge | **两条都要**，口径不同不可互换 |
| `selector_pref_ignored_total` | counter | client preference 被忽略次数 |
| `selector_timeout_total` / `selector_stale_view_total` / `selector_capability_mismatch_total` / `selector_error_total` | counter | fail-open 是否在盖问题 |
| `stale_epoch_frame_dropped_total` / `duplicate_frame_dropped_total` / `out_of_order_frame_dropped_total` | counter | fence 丢弃按种类分列；总量另有 `late_frame_dropped_total` |
| `exclude_hosts_unproven_total` / `admission_host_mismatch_total` | counter | client 越权 placement 的审计线 |
| `retry_budget_exhausted_total` | counter | 耗尽即拒绝新 attempt，只出一个 `ERROR` terminal |
| `lease_rounds_exhausted_total` | counter | `max_rounds` 耗尽：先 fence 过期 leg 再拒绝，只剩一个 `ERROR` terminal |
| `preemption_epoch_desync_fail_open_total` / `preemption_capability_fail_open_total` | counter | D2 在 controller 视图落后／capability 未协商时的 fail-open |
| `crash_recovery_fail_closed_total` | counter | 无 durable ack 时恢复拒绝续流的次数 |
| `lease_expiry_total` / `lease_rounds_histogram` | counter／histogram | lease 是否设得太小 |

`shadow` 模式必须同时记录 baseline 选择与 selector 选择，供离线对比。只记一个没有意义。

## 9. Rollout 与 canary gate

**gate 一律不看 hit rate 单指标。** hit 高而 goodput 不涨，是把算力花在没完成的请求上。

| 阶段 | 配置 | 放行条件 |
|---|---|---|
| R0 | `off` | 指标管道打通，两条 hit 口径都能出数 |
| R1 | `shadow`，全量 | 连续 3 天：selector 与 baseline 选择差异可解释；无新增 error |
| R2 | `enforce`，单 deployment canary | `strict_completed_goodput` 不劣于对照；`prefill_waste_work` 不升；`jain_fairness` 不降；各 tier `per_tier_completion_ratio` 不降 |
| R3 | `enforce`，扩量 | R2 条件维持 7 天；fail-open 计数器无异常尖峰 |
| R4 | D2 评估 | **独立 gate**，见下 |

D2（cooperative preemption）单独闸，全部满足才评估：

- `strict_completed_goodput` 有正收益。
- P99 TTFT／TPOT 不回归。
- `prefill_waste_work` 不升。
- 低优先级 tier 完成率不降。
- `preemptions_per_completed_request ≤ 0.25` 硬上限。
- 全程只出 `REPORT_CHECKPOINT` 路径，无 hard abort。

任一条不满足，D2 保持关闭。D2 关闭不影响 D1 lease。

回滚：任何阶段把 mode 调回 `off` 即可，攻击面为零——因为 selector 从不改变 lifecycle 归属。

## 10. Open questions（本 RFC 不决定）

| # | 问题 | 现状 | 默认 |
|---|---|---|---|
| Q1 | SLO class → deadline 表归谁维护 | `client_token_scheduling_v0.2` §11 标为未决 | 暂定配置服务（uniconfig）下发，client 只读。**需产品与 SRE 确认** |
| Q2 | Turbo 与 中心化 master 同时在链路上时谁是 owner | 未见共存部署 | 就近原则：谁直接面对 engine 谁是 owner |
| Q3 | `selector_score_version` 跨 deployment 不一致时如何比对 | 未定 | canary 期间强制同版本；不同版本不做横向比较 |
| Q4 | R2 checkpoint store 的容量与淘汰策略 | 未定 | D2 未开闸前不阻塞 |
| Q5 | client 主导 epoch 前进（lease expiry／重启）后，planner／controller 如何重新注册该请求 | mock 无 re-auth 通道，此后 D2 对该请求**永久 fail-open** 到 `REPORT_CHECKPOINT` | fail-open 是安全侧默认；re-registration 协议属于 D2 production 设计，未开闸前不阻塞 |
| Q6 | durable output-ack 的持久介质与逐 frame 写入成本 | mock 假设同步持久 ack；无 ack 则恢复 fail closed | 介质选型（本地 WAL／R2 同域）与批量 ack 粒度归 production MR |

Q1 属于配置归属，不改变本 RFC 的架构归属，因此不阻塞本 RFC。

## 11. 状态矩阵（诚实口径）

| 交付项 | 状态 | 证据 | 备注 |
|---|---|---|---|
| M11 RFC（本文） | ✅ 完成 | 本文件 | 设计文档，非实现 |
| Chain mock，7 场景＋敌意用例 | ✅ 完成 | `src/prefill_cache_sim/chain/`、`tests/test_chain_mocks.py` | **协议／状态机不变量测试**，含 durable-ack 恢复、lease 边界 fence、superseded-leg race（sibling admission 被拒绝）、leg-bound checkpoint report、max_rounds 耗尽先 fence 再拒绝、未握手零 capability 选点退回 baseline、controller epoch desync、required 模式五类 fail-open 硬失败 |
| Chain mock 产物 provenance | ✅ 完成 | `TruthBasis.SYNTHETIC_FIXTURE` | 数据全部构造，**不可作为测量值发布** |
| selector 接入点决策 | ✅ 完成 | §2、§3 | 待 owner 团队确认；owner 差异中只有 §2 列出的三条是可执行行为，Turbo pull 派发循环未建模 |
| Enforcement mode 定义 | ✅ 完成 | §4 | attempt 边界模式切换是 production 要求，mock 配置不可变，**未测试** |
| Fail-open 边界定义 | ✅ 完成 | §5 | 五类 reason 全部有计数与 required 硬失败测试 |
| Capability／version handshake 定义 | ✅ 完成 | §6 | 未握手默认零 capability，有测试 |
| Observability 指标定义 | ✅ 完成 | §8 | |
| Canary gate 定义 | ✅ 完成 | §9 | |
| durable output-ack 介质／D2 re-registration | ⚠️ open question | §10 Q5、Q6 | mock 中分别以 fail-closed／永久 fail-open 兜底，production 设计未定 |
| **Production MR** | ⛔ 阻塞 | — | 本任务范围明确排除。未改任何 production repo |
| **真实硬件验证** | ⛔ 阻塞 | — | 无 GPU 实测。simulator 数据不等于 production |
| **Canary 执行** | ⛔ 阻塞 | — | 无部署、无 Spectrum 变更 |
| **G-Prod gate**（仿真排序 vs GPU replay） | ⛔ 阻塞 | — | 依赖上一条 |
| D2 开闸 | ⛔ 未评估 | — | gate 见 §9 R4，保持关闭 |

一句话：**RFC 与 chain mock 已完成；production 实现、硬件验证、canary 全部未开始，且本次不做。**

## 12. 参考

- DESIGN-v3 §7（Phase D 分工与 F1–F8）、§8.3（决策闸）、§10（Fable review 结论）。
- `docs/runtime-validation-v1.md`、`docs/identity-contract-v1.md`（既有契约文体）。
- LLMClientV1 decode lease v1 交接说明（`resolve_leg_route`／`should_resched` 两个 seam，`DS_LLM_PD_DECODE_LEASE_TOKENS`／`DS_LLM_PD_DECODE_LEASE_MAX_ROUNDS`）。
- `client_token_scheduling_v0.2`（PP1／PP2 准入、class A／B／C、§11 未决项）。
- M8 safety spike：`src/prefill_cache_sim/preemption.py`、`tests/test_preemption.py`。
