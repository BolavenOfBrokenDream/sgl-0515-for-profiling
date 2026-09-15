# 07. decode 图内 MoE 专家权重 L2 预取（moe_weight_prefetch）

- 实验序：7（0517 `4501149`）
- 开关：`SGLANG_NPU_MOE_PREFETCH`（默认 0）+ `SGLANG_NPU_MOE_PREFETCH_OPS/MODE/CHUNK_MIB/BUDGET_MIB`
- 适用：仅 35B（MoE）
- **本仓实验状态：不做，恒关。**

## 不做的原因（最优实践裁决）

最优实践为 tp8 + 每 engine 8die，w13 / w2 分别 128MiB / 64MiB，GDN 计算期间难以全量预取；综合带宽抢占问题，最优实践中不开启 prefetch。因此 A/B 序列中本点恒关，所有后续点的两臂均保持 `SGLANG_NPU_MOE_PREFETCH=0`。

## 机制速览（备查）

aclgraph capture 期在专用预取流上按 16MiB 分块发射 CMO 预取（`torch_npu.npu_prefetch`，SDMA 通道），仅 GDN 层、静态全量（auto≡full，active 图内不可实现）；发射点在本层 prepare_attn 之后、attention 之前（TP/EP 两线都位于全部层间通信之后）；best-effort 跨流结构（每层一条 fork 边 + step 末 join）；容量门控 fail-closed（0.8×L2 预算，查询失败整体禁用）。

## 若日后翻案

设置 `SGLANG_NPU_MOE_PREFETCH=1`（其余配置默认），A/B 两臂重启服务；注意每层新增跨流同步边的固定调度开销。记录写回本文件。
