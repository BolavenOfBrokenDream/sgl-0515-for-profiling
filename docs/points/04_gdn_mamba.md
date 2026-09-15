# 04. GDN decode 融合 qkvzba split 与递推 kernel strided 直读（gdn_mamba）

- 实验序：4（0517 `fe997a5`）
- 开关：`SGLANG_NPU_GDN_MAMBA_OPT`（本仓补丁 `0003` 新增，默认 1；基线为总是开启无开关）
- 适用：9B + 35B（GDN 层）

## 改动摘要

- 方案A（sglang，`qwen3_5.py`）：解除 NPU 禁用，GDN 输入投影 split 走 sgl-kernel-npu 融合 kernel 单次产出 `mixed_qkv/z/b/a`，消除 fallback 的重排 + 2 份 contiguous + cat。置 0 恢复 `and not _is_npu` 旧 fallback。
- 方案B（sgl-kernel-npu，`fla/fused_sigmoid_gating_recurrent.py`）：recurrent kernel 按行 stride 直读非连续视图，wrapper 去除 `@input_guard` 全量物化（原每层 3 份 q/k/v 拷贝）。置 0 时 wrapper 对 q/k/v 显式 contiguous 物化，恢复旧行为。

## A/B 方法

- A：`SGLANG_NPU_GDN_MAMBA_OPT=0`；B：`=1`。
- 方案A/B 共用一个开关；如需拆分诊断可临时只改一侧，正式结果以整点为准。

## 已知收益

（待补充——消除的是每层 2+3 份纯数据搬运拷贝，decode 图模式口径计量）

## 结果

### Qwen3.5-9B

| 臂 | decode ms/step | 备注 |
|---|---|---|
| A（关） | | |
| B（开） | | |

profiling 归档：`results/qwen3.5-9b/profiling/04_gdn_mamba/{a,b}/`

### Qwen3.5-35B-A3B

| 臂 | decode ms/step | 备注 |
|---|---|---|
| A（关） | | |
| B（开） | | |

profiling 归档：`results/qwen3.5-35b-a3b/profiling/04_gdn_mamba/{a,b}/`

## 备注

连续输入下方案B 与原版逐 bit 一致（行 stride = packed 宽度时寻址恒等）。下游 recurrent 实现（#8 / #12）均兼容 strided 输入。
