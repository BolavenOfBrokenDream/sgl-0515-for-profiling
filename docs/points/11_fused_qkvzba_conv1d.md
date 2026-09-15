# 11. GDN split+causal_conv1d 单 kernel 融合（fused_qkvzba_conv1d，TP 融合包 op1）

- 实验序：11（0517 `1e04211` 拆分）
- 开关：`SGLANG_NPU_TP_ASCENDC_FUSION`（默认 0）；`SGLANG_NPU_TP_ASCENDC_FUSION_QKVZBA`（op1 分开关，未设置随总开关）；`SGLANG_NPU_TP_ASCENDC_FUSION_DEBUG=1` 打印守卫未命中原因
- 适用：9B + 35B（GDN decode）

## 改动摘要

AscendC 融合算子 `fused_qkvzba_conv1d`：单 kernel 完成 split + `causal_conv1d(run_mode=1)`，接收未拆分 qkvz/ba 投影，直接产出 `(mixed_qkv, z, b, a)` 并原地更新 conv_states，消除 `mixed_qkv` 等中间张量 HBM 写读与一次 kernel 启动。接线为「tuple 进 → (out, z) 出」穿过 attention backend，模型与 backend 签名零改动；三层守卫（init 形状守卫 / 运行期张量属性判定 / 算子注册探测），未命中静默回退 split + stock conv。支持行距视图输入（列内连续、行距 ≥ 逻辑宽度且 16 倍数），与 #10 打包输出直送衔接。

## A/B 方法

- A：`SGLANG_NPU_TP_ASCENDC_FUSION=0`；B：`=1`。
- 仅 decode 生效；若 B 臂无变化，`SGLANG_NPU_TP_ASCENDC_FUSION_DEBUG=1` 查未命中 key。
- profiling 对比重点：GDN 输入投影后 split + conv 窗口（两 kernel + 中间张量流量 → 单 kernel）。

## 已知收益

（待补充——整条融合候选链中精度风险最低：split 纯搬运 + 卷积算术与 stock 同源，按全逐 bit 验收）

## 结果

### Qwen3.5-9B

| 臂 | decode ms/step | split+conv 窗口 µs/层 | 备注 |
|---|---|---|---|
| A（关） | | | |
| B（开） | | | |

profiling 归档：`results/qwen3.5-9b/profiling/11_fused_qkvzba_conv1d/{a,b}/`

### Qwen3.5-35B-A3B

| 臂 | decode ms/step | split+conv 窗口 µs/层 | 备注 |
|---|---|---|---|
| A（关） | | | |
| B（开） | | | |

profiling 归档：`results/qwen3.5-35b-a3b/profiling/11_fused_qkvzba_conv1d/{a,b}/`

## 备注

依赖 sgl-kernel-npu wheel 含 `fused_qkvzba_conv1d`（csrc 新算子，交付仓 6bcca68 已含）。**改动算子签名类变更须整批部署重建 wheel**；python 补丁最后打。
