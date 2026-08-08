# M12 Metric Contract v1.1（历史版本）

> 本文件只用于复现旧 artifacts，已由 `m12-metric-contract-v1.2.md` 取代。不要把本页的 `0.0568` 或 byte-priced KVS 与新结果混用。

状态：FROZEN FOR SYNTHETIC EVALUATION  
Truth basis：`SYNTHETIC_SERVICE_REGIME`  
作者：张珮文

## 1. 排名目标

| 指标 | 定义 | Gate |
|---|---|---|
| `strict_useful_token_goodput` | SLO 内完成、每个 logical request 只计一次的 input＋output tokens／固定 horizon | Primary ranking |
| `strict_useful_input_token_goodput` | strict useful input tokens／固定 horizon | 必须单列 |
| `strict_useful_output_token_goodput` | strict useful output tokens／固定 horizon | 相对 baseline 退化 `≤0.1%` |
| `strict_useful_tokens_per_gpu_work` | strict useful tokens／全部 attempt 的 P＋D GPU work | Pareto efficiency |
| `request_goodput` | SLO 内完成的 logical requests／固定 horizon | 防止长请求掩盖饥饿 |

最终排名先看 combined strict goodput。offered load、output-side goodput、SLO 和 fairness gate 全部通过后，效率才参与 Pareto。

## 2. Workload truth 与 attempt execution

| 项 | 冻结规则 |
|---|---|
| Workload | 独立 `LogicalRequestSpec` 列表定义 identity、tenant、tier、arrival、input、true output；不是从 attempt 反推 |
| Zero-attempt drop | 仍在 offered／tier／tenant demand 分母中，不能被策略藏掉 |
| True output | evaluator 冻结；completed attempt 必须精确 emit 该长度 |
| Stop／EOS／SLO | harness 判断；policy 不得改变 true output 或自行宣布 SLO success |
| Retry | 每个 attempt 都收费；一个 logical request 最多一个 strict winner 获 useful credit |
| Horizon | caller 冻结 start／end；策略不能用自己的最后完成时间改分母 |

## 3. Work 与成本分类账

| 分类 | 规则 |
|---|---|
| Prefill GPU work | 基于实际 uncached tokens 计算；cache hit 只减少一次 prefill work |
| Decode GPU work | 所有 attempt 的 issued decode work，包括 retry／abort |
| KVS work | bytes 与 normalized work 都报告；不混入 GPU efficiency 分母，但 M12.2 必须进入 finish／SLO scheduling |
| True waste | abort／failed attempt 中明确不可复用的 wasted P＋D work |
| SLO-missed | completed 但 strict SLO failed 的非 waste work |
| Winner-attributable | strict winner 的非 waste work |
| Unclassified | 其他 issued attempt work 的 residual bucket |

守恒式：`total_gpu_work = waste + slo_missed + winner_attributable + unclassified`。`kvs_bytes_per_horizon` 是传输速率，`kvs_normalized_work` 是独立成本维度。

## 4. Fairness gate

| Gate | 冻结值 |
|---|---:|
| 每个出现的 SLO tier attainment floor | `≥0.80` |
| tenant served／demand ratio 的 Jain fairness | `≥0.90` |
| 相对 baseline 的单 tier最大退化 | `≤0.02 absolute` |
| strict output goodput 相对退化 | `≤0.001 relative` |

## 5. Synthetic service regimes

| Regime | Prefill token work | Decode token work | Prefill batch saving | Decode batch saving | KVS byte work |
|---|---:|---:|---:|---:|---:|
| COMPUTE_BOUND | 0.0800 | 1.0000 | 0.45 | 0.20 | 1.0e-7 |
| MEMORY_BOUND | 0.0400 | 1.4000 | 0.15 | 0.35 | 2.5e-7 |
| MIXED | 0.0568 | 1.0000 | 0.30 | 0.25 | 1.5e-7 |

参数只是 sensitivity grid，不是硬件拟合值。normalized kill 是 final；pass 只是 provisional。

M12.2 grid 必须包含 KVS-disabled、KVS-expensive extreme，以及至少一个 decode-binding cell。KVS cost 必须推进 completion time 并影响 strict SLO，而不只是出现在报表里。

## 6. Conservation invariants

1. Workload identity 与 `(logical_request_id, attempt_index)` 分别唯一。
2. Attempt 必须引用 workload，并完全匹配冻结 shape。
3. completed output 必须等于 frozen true output。
4. offered requests／tokens／horizon 不随 attempt 数量变化。
5. retry 可增加 raw output、GPU／KVS work，不能重复增加 useful tokens。
6. P／D 分池 utilization 只称 normalized utilization；M9-HW 前禁止称 MFU。

实现：`src/prefill_cache_sim/m12_metrics.py`。  
验证：`tests/test_m12_metrics.py`、`results/m12-metrics/`。
