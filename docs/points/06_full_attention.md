# 06. decode 全注意力段三件套融合（full_attention）

- 实验序：6（0517 `95461e4` → `d62cb3b`，交付态含升级版：A1b 守卫放宽 q∈{1,2,4}，9B-TP4 特化）
- 开关：`SGLANG_NPU_FULL_ATTN_FUSION`（默认 1）；`SGLANG_NPU_FULL_ATTN_FUSION_DEBUG`（默认 0，打印未命中原因）
- 适用：9B + 35B（全注意力层）

## 改动摘要

- A1b：`split_qkvgate_gemma_rmsnorm_rope` + backend 两次 `npu_scatter_nd_update_` → 行并行 Triton kernel 一次完成 split+norm+rope+KV scatter（k/v 按 `out_cache_loc` 直写 cache，调用方 `save_kv_cache=False`）。
- A2：`attn_output.mul_(torch.sigmoid(gate))` 双算子 → 单 kernel（中间 bf16 舍入逐位复刻基线）。
- A3a：`add_gemma_rms_norm` 行并行版（`grid=(batch,)`，小 batch 行内并行度不足问题）。

## A/B 方法

- A：`SGLANG_NPU_FULL_ATTN_FUSION=0`；B：`=1`。
- 生效守卫：decode、每 rank q∈{1,2,4}、kv=1、head_dim=256、rope=64、带 gate、bf16 连续、use_fia 布局非 hybrid SWA；未命中自动回退——若 B 臂无变化，先开 `_DEBUG=1` 查未命中原因。
- profiling 对比重点：全注意力层 9 个算子窗口（基线约 121µs/层，其中 7 个微型 kernel 几乎全是 launch 固定开销）。

## 已知收益

（待补充——动机数据：TP8 / bs32 / 不开 MTP，10 层全注意力约 1.2ms/step）

## 结果

### Qwen3.5-9B

| 臂 | decode ms/step | FA 层窗口 µs/层 | 备注 |
|---|---|---|---|
| A（关） | | | |
| B（开） | | | |

profiling 归档：`results/qwen3.5-9b/profiling/06_full_attention/{a,b}/`

### Qwen3.5-35B-A3B

| 臂 | decode ms/step | FA 层窗口 µs/层 | 备注 |
|---|---|---|---|
| A（关） | | | |
| B（开） | | | |

profiling 归档：`results/qwen3.5-35b-a3b/profiling/06_full_attention/{a,b}/`

## 备注

收益机制为固定开销消除，小 batch decode 最大，batch 增大转向带宽受限后相对收益稀释。A2 与 TP 线 `fused_sigmoid_mul_mm`（已下线桩）有联动接线，不影响本点独立 A/B。
