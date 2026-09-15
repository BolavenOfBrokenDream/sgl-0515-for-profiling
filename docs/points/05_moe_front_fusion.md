# 05. MoE 路由前段融合（moe_front_fusion）

- 实验序：5（0517 `f91b9b5`）
- 开关：`SGLANG_MOE_FRONT_FUSION`（默认 0）
- 适用：仅 35B（MoE）；9B 跳过
- 联动：为 #3 GMM2 提供 exclusive offsets 直喂；为 #13 vgmm1 C1 提供 rank_hist partials

## 改动摘要

- gating + renorm 单算子化：`npu_moe_gating_top_k_softmax` + `l1_norm` 两算子 → 官方 `npu_moe_gating_top_k(renorm=1)` 单算子（bf16 直喂属 #9 cast_elimination 增量）。
- init_routing 重构：stock `npu_moe_init_routing_v2`（黑盒，约 17–18µs/层，占前段六至七成）→ 三个 Triton kernel（`_rank_hist_kernel` / `_partials_cumsum_kernel` / `_routing_gather_kernel`），原生产出 counts / exclusive / inclusive 前缀和。

## A/B 方法

- A：`SGLANG_MOE_FRONT_FUSION=0`；B：`=1`。
- 生效范围：无 grouped topk、无 correction bias 的 fast path；renorm 单算子另需 renormalize 且无 fused shared expert；init_routing 形态门 M≤512、M%8==0、H 为 2 的幂、bf16；形态门外逐点回退 stock（prefill 大尺寸自动回退）。
- profiling 对比重点：路由前段窗口（gating → init_routing → GMM1 前）逐 kernel 耗时。

## 已知收益

（待补充——区域探针：stock init_routing 约 17–18µs/层；renorm 单算子约 5.2–5.8µs vs 两算子 9.4–10.4µs）

## 结果（Qwen3.5-35B-A3B）

| 臂 | decode ms/step | 前段窗口 µs/层 | 备注 |
|---|---|---|---|
| A（关） | | | |
| B（开） | | | |

profiling 归档：`results/qwen3.5-35b-a3b/profiling/05_moe_front_fusion/{a,b}/`

## 备注

开启后请顺带复核 #3 GMM2 段：offsets 前置 kernel 应消失（直喂）。#9 的 gating bf16 直喂只有在 #5 开时才生效。
