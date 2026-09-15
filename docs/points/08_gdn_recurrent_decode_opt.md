# 08. GDN decode 专用 recurrent kernel（gdn_recurrent_decode_opt）

- 实验序：8（0517 `c2a4e31`）
- 开关：`SGLANG_NPU_GDN_UPDATE_FUSED`（默认 0）
- 适用：9B + 35B（GDN decode）
- 互斥：与 #12 `SGLANG_NPU_GDN_RECURRENT_ASCENDC` 同一调用点三选一，#12 优先；两者同开时本点不生效

## 改动摘要

sgl-kernel-npu PR #740 的 decode 专用 Triton recurrent kernel + strided 适配：1-D grid 封顶 AIV 核数（program 内循环处理 (sequence, value-head) tile）、gating 标量跨 V 分块复用（BHV=1 vs 通用版 2）、`num_warps=4/num_stages=1`；叠加 `q/k/v_row_stride` 直读、wrapper `_maybe_contiguous` 轻量检查、`initial_state_source` 硬断言连续、`T==N` decode-only 守卫。

## A/B 方法

- A：`SGLANG_NPU_GDN_UPDATE_FUSED=0`（stock Triton 通用版）；B：`=1`。
- 确保实验时 `SGLANG_NPU_GDN_RECURRENT_ASCENDC=0`（累积协议下序 12 未开，天然满足）。
- profiling 对比重点：GDN recurrent kernel 时长与调度结构（program 数、gating 重复计算消除）。

## 已知收益

（待补充——decode 场景 kernel 主体为状态池读写，调度与标量冗余占比不可忽视）

## 结果

### Qwen3.5-9B

| 臂 | decode ms/step | recurrent kernel µs/层 | 备注 |
|---|---|---|---|
| A（关） | | | |
| B（开） | | | |

profiling 归档：`results/qwen3.5-9b/profiling/08_gdn_recurrent_decode_opt/{a,b}/`

### Qwen3.5-35B-A3B

| 臂 | decode ms/step | recurrent kernel µs/层 | 备注 |
|---|---|---|---|
| A（关） | | | |
| B（开） | | | |

profiling 归档：`results/qwen3.5-35b-a3b/profiling/08_gdn_recurrent_decode_opt/{a,b}/`

## 备注

kernel 算术与通用版逐语句一致，差异仅在调度结构与寻址。#12 的 AscendC 版兼容并演进自本点——两点结果横向对比可裁决 decode recurrent 的最终形态。
