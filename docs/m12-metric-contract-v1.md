# M12 Metric Contract v1

状态：FROZEN FOR SYNTHETIC EVALUATION  
Truth basis：`SYNTHETIC_SERVICE_REGIME`  
作者：张珮文

## 1. 排名目标

策略不能只报 hit 或 raw TPS。M12 同时冻结两个主观察量，并用 SLO／fairness 作硬约束。

| 指标 | 定义 | 防止什么 gaming |
|---|---|---|
| `strict_useful_token_goodput` | SLO 内完成、每个 logical request 只计一次的 input＋output tokens／固定 observation horizon | retry 重复记产出；策略改变分母 |
| `strict_useful_tokens_per_gpu_work` | strict useful tokens／全部 attempt 的 P＋D GPU work | 高 hit 但大量 retry／abort 浪费 |
| `request_goodput` | SLO 内完成的 logical requests／固定 horizon | 用长请求 token 数掩盖请求饥饿 |

最终策略排名先看 `strict_useful_token_goodput`；效率指标只能在 offered load、SLO 和 fairness gate 全部通过后参与 Pareto。

## 2. 分子与分母

| 项 | 规则 |
|---|---|
| Offered load | 按 `logical_request_id` 去重；input／requested output shape 在 retry 间必须一致 |
| Useful tokens | 一个 logical request 最多由一个 strict winner 贡献 `input_tokens + emitted_output_tokens` |
| GPU work | 所有 attempt 的 Prefill／Decode work 全部计入，包括失败、retry、abort |
| Raw output | 所有 attempt emitted output 都计入，只作诊断，不能当 goodput |
| Waste | `wasted_prefill_work + wasted_decode_work`；必须分别不超过 issued work |
| SLO-missed work | completed 但未过 strict SLO 的非 waste GPU work；单列，不冒充 true waste |
| Horizon | caller 显式冻结 `observation_start_work`／`observation_end_work`；策略不能按自身最后完成时间改分母 |
| Utilization | P／D 分池报告；aggregate 只称 normalized utilization，M9-HW 前禁称 MFU |

## 3. Fairness gate

| Gate | 冻结值 |
|---|---:|
| 每个出现的 SLO tier attainment floor | `≥ 0.80` |
| tenant served／demand ratio 的 Jain fairness | `≥ 0.90` |
| 相对 baseline 的单 tier 最大退化 | `≤ 0.02 absolute` |

三个条件独立报告。绝对 floor 通过不代表允许相对 baseline 退化；后者在 M12.2 横比时启用。

## 4. 三套 synthetic service regimes

这些参数是 sensitivity grid，不是硬件拟合值。所有输出保持 `NORMALIZED_WORK`。

| Regime | Prefill token work | Decode token work | Prefill batch max saving | Decode batch max saving | KVS byte work |
|---|---:|---:|---:|---:|---:|
| COMPUTE_BOUND | 0.0800 | 1.0000 | 0.45 | 0.20 | 1.0e-7 |
| MEMORY_BOUND | 0.0400 | 1.4000 | 0.15 | 0.35 | 2.5e-7 |
| MIXED | 0.0568 | 1.0000 | 0.30 | 0.25 | 1.5e-7 |

Batch saving 使用冻结的 piecewise-linear saturation；它只制造三种排序压力。normalized simulation 的 kill 可以终止方案，pass 只算 provisional。

## 5. Conservation invariants

1. `(logical_request_id, attempt_index)` 全局唯一。
2. `total_gpu_work = prefill_gpu_work + decode_gpu_work`。
3. `wasted_gpu_work ≤ total_gpu_work`。
4. strict completed request 不能没有 finish；未完成 attempt 不能命中 strict SLO。
5. offered logical requests／tokens／horizon 不随 attempt 数量变化。
6. retry 可以增加 raw output 和 GPU work，不能重复增加 useful tokens。

实现：`src/prefill_cache_sim/m12_metrics.py`。  
验证：`tests/test_m12_metrics.py`、`results/m12-metrics/`。
