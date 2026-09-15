# 10. GDN 输入投影 qkvz+ba 权重打包（gdn_qkvzba_pack，TP 融合包）

- 实验序：10（0517 `1e04211` 拆分）
- 开关：`SGLANG_NPU_GDN_QKVZBA_PACK`（默认 **1**）；`SGLANG_NPU_GDN_QKVZBA_PACK_MAX_M`（EnvInt，默认 256，小 M 门控）
- 适用：9B + 35B（GDN 层输入投影）
- 互斥：与双流（`SGLANG_NPU_USE_MULTI_STREAM=1` alt 流）分支互斥——双流命中时不走打包路径

## 改动摘要

两次独立 GEMM（`in_proj_qkvz` 大 N + `in_proj_ba` 小 N）合并为一次打包 GEMM：权重沿 N 维拼接为 `[N_pack, K]`（补零行至 16 倍数），单次 stock linear 后按行切片成两段行距视图。下游三条消费路径原生支持行距视图：`fused_qkvzba_split_reshape_cat_contiguous` 按物理 `stride(0)` 取行距、`fused_qkvzba_conv1d`（#11）原生接受、`fix_query_key_value_ordering` 回退路径本就要拷贝。`weight.data` 重绑为打包缓冲行切片，loader / RL 同步的 in-place `narrow+copy_` 天然落在打包缓冲上；指针漂移时 data_ptr 校验自动重新物化。

## A/B 方法

- A：`SGLANG_NPU_GDN_QKVZBA_PACK=0`；B：`=1`。
- 生效条件：NPU、非量化 bf16/fp16 2D 权重、无 bias、非双流分支、`seq_len ≤ MAX_M`；capture 时分支固化进图。
- 若实验环境开双流，本点不生效，需在笔记中注明。

## 已知收益

（待补充——小 M 段省一次小 GEMM 固定开销；大 M 下非规则 tiling 惩罚会反超，故有 MAX_M 门控）

## 结果

### Qwen3.5-9B

| 臂 | decode ms/step | 备注 |
|---|---|---|
| A（关） | | |
| B（开） | | |

profiling 归档：`results/qwen3.5-9b/profiling/10_gdn_qkvzba_pack/{a,b}/`

### Qwen3.5-35B-A3B

| 臂 | decode ms/step | 备注 |
|---|---|---|
| A（关） | | |
| B（开） | | |

profiling 归档：`results/qwen3.5-35b-a3b/profiling/10_gdn_qkvzba_pack/{a,b}/`

## 备注

数学上严格等价（拼接 GEMM 每输出列独立，pad 列无消费者），逐 bit 不变。
