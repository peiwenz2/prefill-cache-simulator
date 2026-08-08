# M12 metric contract v1.2：可解释的 synthetic work 尺子

作者：张珮文

## 1. 先定义统一尺子

`work` 不是毫秒、FLOPs 或 GPU 利用率。它只是一把人为定义的相对成本尺子：

```text
Decode 1 个 token 的 modeled cost = 1 work
```

因此 Decode 30 tokens 就记为 `30 × 1 = 30 work`。这一步只是选单位，类似把某段长度定义成 1 米。

## 2. 再定义三种 synthetic world

| World | Prefill 一个未命中 token | Decode 一个 token | Remote fetch 一个 token | 大白话 |
|---|---:|---:|---:|---|
| COMPUTE_BOUND | 0.08 work | 1.00 work | 0.01 work | Prefill 相对更贵 |
| MEMORY_BOUND | 0.04 work | 1.40 work | 0.02 work | Decode 与搬运相对更贵 |
| MIXED | 0.06 work | 1.00 work | 0.01 work | 中间档，用于 headline sizing |

这些数字是圆整的 sensitivity assumptions，不是 Mooncake trace 字段，不是论文常数，也不是硬件 benchmark。

## 3. KV 大小与 KVS cost 是两件事

```text
KV bytes per token = 65,536 bytes = 64 KiB
```

这个假设只用于报告“搬了多少 bytes”。真实值取决于模型层数、KV heads、head dimension 和 dtype：

```text
真实 KV bytes/token
= 2（K＋V）× layers × kv_heads × head_dim × bytes_per_element
```

调度成本直接使用上表中的 `remote fetch work/token`，不再从一个很小的 `work/byte` 反推，避免制造虚假精度。

## 4. 一条 request 的公式

```text
uncached_tokens = input_tokens - local_hit_tokens - remote_hit_tokens

P_work   = uncached_tokens × prefill_work_per_token
KVS_work = remote_hit_tokens × remote_fetch_work_per_token
D_work   = output_tokens × decode_work_per_token
```

- `local_hit_tokens`：本节点已有 KV；在当前模型中不收重算／搬运 work。
- `remote_hit_tokens`：本节点没有，但 shared KVS 有；只收 remote fetch work。
- `uncached_tokens`：两处都没有；必须重新 Prefill。
- 三类 input tokens 互斥，并且总和必须等于 input tokens。

## 5. MIXED world 的白话比例

```text
Local hit 1 token    = 0 work
Remote fetch 1 token = 0.01 work
Recompute 1 token    = 0.06 work
Decode 1 token       = 1.00 work
```

因此，在这个人为世界里，local hit 最便宜；local miss 时，remote fetch 的单 token cost 是 recompute 的六分之一。这是实验输入，不是实验结论。

## 6. 当前不能声称什么

- 不能把 `work` 写成 ms。
- 不能把资源下限写成真实 GPU 数量。
- 不能声称 remote fetch 在线上一定比 recompute 便宜 6 倍。
- 没有 M9 hardware calibration 前，normalized simulation 的 pass 只算 provisional；它可以淘汰明显失败的想法，不能形成生产容量承诺。
