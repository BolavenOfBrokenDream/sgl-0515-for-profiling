# 12. AscendC GDN decode recurrent 算子（recurrent_ascendc，TP 融合包 op2）

- 实验序：12（0517 `1e04211` 拆分）
- 开关：`SGLANG_NPU_GDN_RECURRENT_ASCENDC`（默认 0，显式置 1 启用）
- 适用：9B + 35B（GDN decode）
- 互斥：decode 三选一最高优先级（本点 > #8 > stock Triton）；import 期绑定，capture 后改 env 无效

## 改动摘要

生产 Triton recurrent kernel 按 BV=64 半行模式读写 state，有效带宽仅标定 1/3~1/4；AscendC 重写利用每个 (slot, head) state 恰为 32KB 全连续块的事实，单发 DataCopyPad 整头搬入 UB、计算后整块写回。grid = min(N×HV, AIV 核数)，双缓冲软流水重叠 MTE2 与向量计算；特化 head_dim=128。sglang 侧同签名 drop-in wrapper（`fused_sigmoid_gating_delta_rule_update_ascendc`），守卫未命中逐调用回退 stock Triton。

## A/B 方法

- A：`SGLANG_NPU_GDN_RECURRENT_ASCENDC=0`；B：`=1`（此时若 #8 曾开启会被本点压制——注意累积协议下 #8 已开，B 臂实际对比的是「#8 Triton 优化版 → AscendC 版」的增量；如需对比 stock，另设一臂 #8=0 且 #12=0）。
- 运行期守卫未命中静默回退，`SGLANG_NPU_TP_ASCENDC_FUSION_DEBUG=1` 打印原因。
- profiling 对比重点：recurrent kernel 时长与访存模式（半行 → 整块 DataCopyPad）。

## 精度验收口径

非逐 bit：4 处归约加法顺序 + Exp/Ln/Div 指令级差异不可先验保证逐 bit 一致。走组合闸门：有界误差（bf16 臂逐元素误差与符号翻转率预算）→ 长程漂移（多步连解误差有界）→ e2e 同 seed A/B。

## 已知收益

（待补充——decode 段带宽瓶颈，访存模式重构属代际收益候选）

## 结果

### Qwen3.5-9B

| 臂 | decode ms/step | recurrent kernel µs/层 | 备注 |
|---|---|---|---|
| A（#8 版/stock） | | | |
| B（AscendC） | | | |

profiling 归档：`results/qwen3.5-9b/profiling/12_recurrent_ascendc/{a,b}/`

### Qwen3.5-35B-A3B

| 臂 | decode ms/step | recurrent kernel µs/层 | 备注 |
|---|---|---|---|
| A（#8 版/stock） | | | |
| B（AscendC） | | | |

profiling 归档：`results/qwen3.5-35b-a3b/profiling/12_recurrent_ascendc/{a,b}/`

## 备注

依赖 sgl-kernel-npu wheel 含 `fused_sigmoid_gating_recurrent`（bf16/fp32 一 kernel 一文件）。与 #9 的 dt_bias fp32 联动：#9 开启时 wrapper 的 `.float()` 为 no-op。
