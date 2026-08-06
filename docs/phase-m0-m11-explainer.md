# prefill-cache-sim：M0–M11 阶段新人向说明书

- 作者：张珮文
- 基准提交：`cfb643c`（M9=`b651bf9`，M10=`4ca340a`，M11=`cfb643c`，工作区 clean）
- 时间单位声明：所有仿真结果的时间量纲是 `NORMALIZED_WORK`（归一化工作量），**不是硬件毫秒**。带 `_ms` 后缀的字段同样是归一化工作量（见 `results/m6/results.csv` 的 `metric_unit` 列与 M10 的 `unit_note`）。

---

## 0. 我们究竟在做什么

这个仓库是一个**决策实验室**，不是一份 GPU 性能报告。它回答一类问题：在 prefill 缓存命中、节点选择、淘汰、共享 KVS、decode 续约与抢占这些策略之间，**哪些组合在同一条真实 trace 上排序更优、哪些会互相打架**。

- 输入：一条固定的生产形态 trace（23,608 条请求，SHA-256 `b434f181…f711`），加上确定性随机源（`docs/randomness-contract-v1.md`）。
- 输出：同一 trace 下不同策略组合的命中率、goodput、排队分位数、抢占率的**相对排序**。
- 明确不输出：TTFT 毫秒、SLO 达标率、生产收益预测。这些必须等硬件标定（M9-HW）之后才有资格谈。

一句话给新人：**这里所有数字只用于"比大小"，不用于"报快慢"。**

---

## 1. 一个请求的旅程

跟着一条请求从到达走到终点，就能把整个系统看完。每一步对应一个源码模块。

| 步骤 | 发生什么 | 模块 |
|---|---|---|
| ① 到达 | 请求带着 `input_tokens`、block 引用序列、session 归属进入仿真时钟 | `src/prefill_cache_sim/simulator.py` |
| ② 选节点（WHERE） | Selector 决定发给哪个 prefill 节点：随机？找缓存最热的？跟着 session 走？ | `src/prefill_cache_sim/selectors.py` |
| ③ 查本地缓存 | 节点上逐 block 前缀匹配，命中的 block 不用重算 | `src/prefill_cache_sim/simulator.py` |
| ④ 淘汰 | 容量满了，淘汰策略决定谁出局（FIFO／LRU／SLRU／二次命中准入／链感知） | `src/prefill_cache_sim/eviction.py` |
| ⑤ 共享 KVS | 本地 miss 时可查共享层；拓扑决定"本地／共享／混合"三种形态 | `src/prefill_cache_sim/kvs.py`、`topology.py` |
| ⑥ SLO 与生命周期 | 请求带截止期与重试预算；attempt 身份按 `docs/identity-contract-v1.md` 记账 | `src/prefill_cache_sim/slo.py`、`lifecycle.py` |
| ⑦ decode 续约 | prefill 完成后进入 D 流；D1/D1.5 给 decode 发 token 租约，到期自然 LENGTH 截断，续段 = 原输入 + 已发 token | `src/prefill_cache_sim/decode_lease.py`、`continuation.py` |
| ⑧ 协作抢占 | D2 用 checkpoint + `abort_self` 让 decode 主动让位（M8 spike，仅上界结论） | `src/prefill_cache_sim/preemption.py` |
| ⑨ 记账与对账 | 输出进 ledger；M10 把 ENGINE_HIT／CLIENT_LATENCY／ATTEMPT_TRACE 三源日志按 `(logical_request_id, attempt_index)` join 对账 | `src/prefill_cache_sim/replay/` |

最重要的口径纪律（贯穿全部里程碑）：

- 命中率必须同时报 `block_ref_hit_rate` 与 `token_weighted_hit_rate`，**不许混在一列**。
- goodput 主口径是 `strict_completed_goodput`（只算完整完成的请求）；`partial_output_tokens` 永远不进 goodput。

---

## 2. 策略分类：一张新人速查表

### 2.1 Selector（S 系，选"发给谁"）

| 编号 | 名字 | 一句话解释 | 现实原型 |
|---|---|---|---|
| S0 | Random | 抛硬币 | 基线 |
| S1 | RoundRobin | 轮流发 | 基线 |
| S2 | LeastWork | 谁最闲发谁 | 朴素负载均衡 |
| S3 | GBPrefixBucket | 按前缀桶粘住节点，让同前缀请求聚堆 | Turbo CacheAwareScheduler／Sticky bucket（`DESIGN-v3.md` §1 引 `CacheAwareScheduler.java:1196-1202`） |
| S4 | SessionAffinity | 同 session 粘同节点，在线建链 + 热 block 排除 + family 上限 | Session 亲和 |
| S5 | FlexLbTtft | 打分 `queue_ms + input_tokens − 0.7×hit_tokens`，取 top-30% 带内随机 | RTP-LLM ShortestTTFTStrategy |
| S6 | CalibratedTtft | S5 加标定系数 | 同上 |
| S7 | CacheSlo | 缓存与 SLO 混合打分 | 混合 |

M4 结论：S5/S6 触发 stop gate（复杂度不换收益），**S3 + S4 晋级**为 KVS 候选（`results/m4/summary.json` 的 `decision` 字段）。

### 2.2 淘汰（E 系）

E0 FIFO、E1 LRU、E2 SLRU、E3 二次命中准入 + LRU、E4 链感知价值淘汰（`eviction.py`）。

### 2.3 decode 策略阶梯（D 系）

| 编号 | 机制 | 关键点 |
|---|---|---|
| D0 | 无租约 | decode 无限占用，容量紧张时头阻塞 |
| D1 | 固定 token 租约 | 租约到期走**自然 LENGTH 结束**，不是 abort；续段重新排队 |
| D1.5 | 自适应租约 | 按压力调租约长度，边界数从 12.92 降到 4.95 次/请求（`results/m7/results.csv`） |
| D2 | 协作抢占 | checkpoint + grant 下的 `abort_self`（M8 spike） |
| D3 | 硬 abort | **排除在优化空间外**，只作对照 |

### 2.4 恢复阶梯（R 系）与拓扑×协议

- R0 重算续段 ／ R1 前缀 KVS 辅助 ／ R2 decode checkpoint 句柄。
- 两轴抽象：`CacheTopology`（LOCAL_ONLY｜SHARED_KVS｜HYBRID）×`ExecutionProtocol`（P0_PD／P1_DP／P2_GATED_PD／P3_GATED_DP／KVS_DECOUPLED），见 `DESIGN-v3.md` §0。

---

## 3. M0–M11 里程碑分类账

每行回答三件事：建了什么、怎么验收、证据强度。完整台账在 `DESIGN-v3.md` §9。

| 里程碑 | 建了什么 | 验收方式 | 证据强度 |
|---|---|---|---|
| M0–M2 | trace 分析器、确定性内核、schema 合同（`docs/scenario-versioning-v1.md` 等四份合同） | 单测 + 上限复算 | SYNTHETIC_FIXTURE |
| M3 | 基线 selector + 淘汰矩阵 | 固定 trace 回放 | SYNTHETIC_REPLAY |
| M4 | 7 个 selector × 淘汰全量对比，97 case | stop-gate 判据先行（hit_epsilon=0.001） | SYNTHETIC_REPLAY |
| M5 | 共享 KVS 层建模，30 case | 与 M4 冠军组合对比 | SYNTHETIC_REPLAY |
| M6 | 协议阶梯（P0–P3），49 case，三轮外部评审 | strict goodput 对比 | SYNTHETIC_REPLAY |
| M7 | decode 租约 D0/D1/D1.5，11 case | 容量扫描找 crossover | SYNTHETIC_REPLAY |
| M8 | 协作抢占 spike，12 case | 只做序数结论 + 上界模型 | SYNTHETIC_REPLAY（ASSUME_RESUME_SUCCESS_UPPER_BOUND） |
| M9 | 标定 harness（拟合器 + 溯源门禁） | MockEngine 已知真值回收 | SYNTHETIC_REPLAY，`calibration_status=SYNTHETIC_UNCALIBRATED` |
| M10 | 回放验证 harness（三源对账 + 影子决策 + 故障注入） | 控制实验 3 注入 = 3 回收 | SYNTHETIC_REPLAY，`hardware_validation=BLOCKED_NO_ENGINE_ACCESS` |
| M11 | 生产 RFC + 全链路 mock（7 场景协议不变量） | mock 断言，非 E2E | HARNESS_ONLY／RFC |

本地验证记录（非 artifact，属会话记录）：682 个测试通过，ruff clean。**这只证明代码自洽，不是 E2E 或 canary 证据。**

---

## 4. 当前的准确数字（全部为 NORMALIZED_WORK，SYNTHETIC_REPLAY）

### 4.1 trace 上限（M0，`README.md`）

- block-ref 命中上限：`226190 / 409356 = 55.2550836%`
- token-weighted 上限：`115733271 / 202791701 = 57.0700233%`
- trace：23,608 请求，input 均值 8,589.96，output 均值 182.13。

### 4.2 selector 对比（M4，`results/m4/summary.json`）

| Selector | token_weighted_hit_rate | queue_wait p95（归一化） | 负载偏斜 request_load_max_mean |
|---|---|---|---|
| S0 Random | 0.4430 | — | — |
| S3 GBPrefixBucket | 0.5401 | 3103.48 | 1.822 |
| S4 SessionAffinity | 0.5334 | 3011.46 | 1.047 |
| S5 FlexLbTtft | 0.5232 | 6589.08 | — |

结论：S3 命中最高但偏斜大；S4 略低但均衡好；S5 排队恶化、被 stop gate 淘汰。

### 4.3 KVS 与协议（M5/M6）

- M5：共享层带来约 4.7% 效果，最差 KVT 角落约 1.03%（`DESIGN-v3.md` §9 M5 结论；`results/m5/results.csv` M5-BASE `effective_prefix_rate=0.5676`）。
- M6 strict_completed_goodput：P2_GATED_PD=0.6870（最优）＞P3_GATED_DP=0.5787＞P0_PD=0.4883＞P1_DP=0.2939（`results/m6/results.csv`）。

### 4.4 decode 租约 crossover（M7，`results/m7/results.csv`）

| decode 节点数 | D0 strict goodput | D1 strict goodput | 谁赢 |
|---|---|---|---|
| 1（极端头阻塞） | 0.00718 | 0.14181 | D1（约 19.7×） |
| 2 | 0.68462 | 0.62351 | D0 |
| 4 | 0.74183 | 0.73939 | 打平偏 D0 |
| 8 | 0.74275 | 0.74275 | 完全打平 |

附加事实：`migrate_on_boundary=True` 使 goodput 崩到 0.00153；D1.5 把边界次数从 12.92 降到 4.95。结论是**压力感知启用**：租约只在容量紧张时开。

### 4.5 抢占上界（M8，`results/m8/results.csv`）

safety_margin 0.1→50→100→200 时 `preemptions_per_completed_request` 0.3795→0.3391→0.2978→0.2231；硬门槛 ≤0.25 仅 margin=200 在**上界模型**下通过。只有序数结论，无绝对收益结论。

### 4.6 标定 harness（M9，`results/m9-synthetic/results.csv`）

| 拟合 | intercept | token 系数 | batch 系数 | residual_p99 | mock 真值 |
|---|---|---|---|---|---|
| M9-PREFILL | 3.0925 | 0.0199965 | 1.4943 | 0.5885 | `3.0 + 0.02·tokens + 1.5·batch` |
| M9-DECODE | 7.9297 | 0.0004463 | 0.9036 | 0.5364 | `8.0 + 0.0004·tokens + 0.9·batch` |

含义：拟合管线能回收已知真值（噪声 0.5 下），**但被拟合的对象是 mock，不是 GPU**。外插风险已披露：prefill 网格 1,807–61,623 vs trace 890–125,546（2.04× 超出）；decode 2.23× 超出（`results/m9-synthetic/PROVENANCE.md`）。

### 4.7 回放验证（M10，`results/m10-synthetic/`）

- 4 个 arm：S0（BASELINE）、S3（CANDIDATE）、S5（STOP_GATED）、S4（M4_WINNER）；1× 与 2× 回放速度。
- 排序稳定性：KENDALL_TAU_B=1.0，6 concordant／0 discordant。**PROVENANCE 自己声明这是弱证据**（S0 两个速度下分数逐位相同 0.34403309236012575；2× 下 queue_wait p95 仅升约 6%，压力不足以逼出翻转）。
- 三源对账：建模 ledger 为 0 行分歧——但因为三源都是同一份日志的投影，**0 分歧本身不证明对账逻辑正确**；证明力来自故障注入控制实验：注入 3（drop／duplicate／perturb）＝回收 3，逐条精确匹配。
- 影子决策：S3、S4 = SHADOW_RECOMMENDED；S5 = SHADOW_WITHHELD（不赢基线）。影子决策从未被执行（`ShadowEnforcementError` 挡住）。
- 防伪门禁：runner 落盘前扫描字节，出现 MILLISECONDS／HW_CALIBRATED／HW_VALIDATED 即 fail-closed 拒写。

### 4.8 尚未测量（NOT_MEASURED）

真实 TTFT 毫秒、真实 TPOT、KVT 传输带宽、GPU 上任何策略的绝对收益、生产命中率变化。**引用本仓库时这些字段一律写"未测量"。**

---

## 5. 不可宣称框（must-not-claim）

以下句式在任何汇报里都**不允许**出现：

1. ❌ "仿真显示 TTFT 降低 X ms／SLO 提升 X%" ——时间单位是 NORMALIZED_WORK，无毫秒语义。
2. ❌ "682 测试通过，可以上线" ——测试只覆盖仿真器与 mock 链路，不含任何生产 E2E／canary。
3. ❌ "M9 已完成标定" ——M9 只有 harness，`calibration_status=SYNTHETIC_UNCALIBRATED`，M9-HW 被真实 `MachineProvenance` 缺失阻塞。
4. ❌ "M10 tau_b=1.0 证明排序在生产可复现" ——那是弱证据，压力不足；需要 M10-HW 在硬件校准后重测。
5. ❌ "S3 比 S4 好（或反之）" ——两者是不同 trade-off（命中 vs 偏斜），M4 只给 diff 不下总评。
6. ❌ "D2 抢占收益已验证" ——M8 是 ASSUME_RESUME_SUCCESS_UPPER_BOUND 上界模型。
7. ❌ 任何把 `_ms` 字段当 wall-clock 的引用。
8. ❌ "S3 比线上提升 9.71 pp" ——9.71 pp 是相对 Random；相对 S5／S6 simulator baseline 只有约 0.6～1.7 pp，production headroom 未测量。

---

## 6. 下一步：两条并行线

前提铁律：**CPU 机器可以跑通 replay 与 shadow 的管线（plumbing），但产不出 GPU 硬件标定数据。**所以硬件线与生产接入线互不阻塞、并行推进。

### 6.1 线 A：硬件标定线（M9-HW → M10-HW）

| 门 | 内容 | 进入条件 | 退出条件 |
|---|---|---|---|
| G1 M9-HW | 在真实引擎上重跑标定扫描 | 拿到带 `MachineProvenance`（host_id／accelerator_model／engine_version／captured_at_utc）的 `EngineEndpoint` | 产出 `HW_CALIBRATED` 系数 + KVT 基准（M5 遗留 DEFERRED_PER_M5） |
| G2 | 用标定系数重跑 M4–M8 合成扫描 | G1 完成 | 结论排序不变或差异有解释 |
| G3 M10-HW | 冻结 tau-b 对比 G-Prod | G2 完成 | tau-b 在真实压力下仍达标 |

现成命令（合成版，硬件版待改造 endpoint）：`.venv/bin/python scripts/run_m9_synthetic.py`、`.venv/bin/python scripts/run_m10_synthetic.py`。

### 6.2 线 B：生产接入线（G0 → R0 → R1）

| 门 | 内容 | 关键条件 |
|---|---|---|
| G0 | owner sign-off | FlexLB（push 版图）与 Turbo（pull 版图）owner 对 RFC §2 attach 方式签字；Turbo pull-shadow 语义需 owner 确认 |
| G1' R0 | enforcement=off，只建指标管线 | RFC §8 观测指标落地 |
| R1 | shadow 3 天 | 影子决策与线上决策 diff 报告，零 enforce |
| G4 R2 | 单 deployment canary enforce | R1 通过后 |
| G5/G6 | 解决 Q6（durable ack 写序）、Q5（D2 re-registration） | 见 §7 |
| G7 R4 | D2 评估独立门 | `preemptions_per_completed_request ≤ 0.25` 硬上限（真实恢复语义下，非上界模型） |

### 6.3 M11 RFC 已定的关键决策（`docs/m11-production-rfc.md`）

- attach 点：把**带版本的打分函数**交付给现有版图 owner（FlexLB master 是 push 型，Turbo CacheAwareScheduler 是 pull 型），**不建第三张全局图**。
- WHERE 与生命周期分离：selector 只回答 WHERE（`LegRoute { prefer_host, exclude_hosts, hint_input_tokens }`）；LLMClientV1 保有 logical_request_id、attempt 身份、重试预算、续段、OutputAuthority（epoch + output_seq 围栏）、durable output-ack ledger。
- 四档 enforcement：off｜shadow｜enforce｜required；fail-open 五种原因计数（timeout／error／stale view／capability mismatch／planner unavailable）。
- 能力握手：PREFIX_CACHE_QUERY、DECODE_LEASE_V1、R2_CHECKPOINT_STORE、COOPERATIVE_PREEMPT；**未握手实例按零能力对待**。
- 链路 mock 七场景（超时／stale view／P 回滚／D 拒绝／租约到期／迟到帧／client 崩溃）是**协议不变量测试，不是生产 E2E**。

---

## 7. 开放决策（需要人拍板的三件事）

| 编号 | 问题 | 需要的决策 |
|---|---|---|
| Q6 | durable output-ack 的写序：ack 落盘和向用户发帧的先后顺序未定，影响崩溃后"重复发帧 vs 丢帧"的取舍 | 选定持久化介质与写序语义，并接受对应的每帧写放大成本 |
| Q5 | D2 下 client 主导 epoch 前进后，decode 侧 controller 的 re-registration 协议未定义 | 定义 re-registration 时序与失败语义，否则 R4 门无法评估 |
| Turbo shadow | Turbo 是 pull 型版图，shadow 模式"记录但不执行"的语义在 pull 路径上如何实现需要 owner 确认 | Turbo owner 对 pull-shadow 方案签字（G0 的一部分） |

---

## 8. 审读清单（给张珮文自查"这阶段是否真的做完了"）

- [ ] HEAD 是 `cfb643c` 且 `git status` clean。
- [ ] `results/m9-synthetic/` 与 `results/m10-synthetic/` 存在，PROVENANCE 声明 SYNTHETIC_REPLAY，且不含 MILLISECONDS／HW_CALIBRATED 字样。
- [ ] `docs/m11-production-rfc.md` §11 状态矩阵中，生产 MR／硬件／canary／G-Prod 仍为 ⛔（未做，且文档如实标注）。
- [ ] 本文 §4 每个数字都能在对应 artifact 中找到原值（抽查 M7 表与 M9 系数）。
- [ ] 本文 §5 的七条禁句没有在任何对外材料中出现。
- [ ] 两条并行线的进入条件里，M9-HW 的 `MachineProvenance` 阻塞项与 G0 的 owner 名单都还没解除——若已解除，本文需要更新。
- [ ] 682 测试／ruff clean 只作为"代码自洽"证据引用，未被升格为验收证据。

---

## 9. 引用

- 设计与台账：`DESIGN-v3.md`（§0 结论、§1 证据基础、§8.3 门禁、§9 里程碑台账）
- 生产 RFC：`docs/m11-production-rfc.md`
- 四份合同：`docs/identity-contract-v1.md`、`docs/randomness-contract-v1.md`、`docs/runtime-validation-v1.md`、`docs/scenario-versioning-v1.md`
- artifact：`results/m4/summary.json`、`results/m5..m8/results.csv`、`results/m9-synthetic/{PROVENANCE.md,results.csv}`、`results/m10-synthetic/{PROVENANCE.md,results.csv,ranking.csv,shadow_decisions.csv,fault_injection.csv}`
- 源码：`src/prefill_cache_sim/`（selectors.py、eviction.py、kvs.py、slo.py、lifecycle.py、decode_lease.py、preemption.py、continuation.py、calibration/、replay/、chain/）
- 复现：`.venv/bin/python scripts/run_m9_synthetic.py`；`.venv/bin/python scripts/run_m10_synthetic.py`
- 上限口径：`README.md`（block-ref 55.2550836%，token-weighted 57.0700233%；Mooncake Inf=0.51 仅作论文参照，分母与插入时机未指明）
