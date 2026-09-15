# 03. persistent GMM2（down_proj）Triton 分组矩阵乘（gmm）

- 实验序：3（0517 `a7a1c0d`）
- 开关：`SGLANG_GMM2_TRITON`（默认 0）
- 适用：仅 35B（MoE）；9B 跳过
- 依赖：与 `SGLANG_MOE_FRONT_FUSION=1`（实验序 5）同开时，v2.2 init 原生 exclusive offsets 直喂，省内置 offsets 前置 kernel

## 改动摘要

stock `npu_grouped_matmul`（mac_ratio 仅约 6.9%，任务边界流水线排空）→ Triton persistent kernel：约 AIC 核数个常驻 program 静态 round-robin 消费 `(expert, n_tile)` 任务，空专家整任务跳过；`_gmm2_offsets_kernel` 单 program `tl.cumsum` 推导 exclusive offsets（或外部直喂）。w2 以 `[E,N,K]` ND 直喂不再 transpose；开关开时 `process_weights_after_loading` 对 w2 `.contiguous()` 兜底。

## A/B 方法

- A：`SGLANG_GMM2_TRITON=0`；B：`=1`。
- 按累积协议，实验序 3 时 moe_front_fusion（序 5）尚未开，B 臂走内置 offsets kernel；待序 5 开启后 offsets 直喂收益自动叠加，可在序 5 的笔记中复核 GMM2 段耗时变化。
- 生效条件：NPU BF16 无量化、无 bias；有 bias 回退 stock。

## 已知收益

（待补充——0517 报告留空；单测口径：skinny grouped GEMM，每专家约 1 行场景）

## 结果（Qwen3.5-35B-A3B）

| 臂 | decode ms/step | GMM2 kernel µs/层 | 备注 |
|---|---|---|---|
| A（关，stock） | | | |
| B（开） | | | |

profiling 归档：`results/qwen3.5-35b-a3b/profiling/03_gmm/{a,b}/`

## 备注

`num_progs=24`（= AIC 核数）是静态负载均衡前提，勿随意改。精度：fp32 累加，与 stock 数值语义一致。
