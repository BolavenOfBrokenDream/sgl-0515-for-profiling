# 14. MoE 层尾 fin+add+AR+norm 四节点融合（moe_tail_fusion，TP 融合包）

- 实验序：14（0517 `1e04211` 拆分；0515 侧 `2b0034e` 按 0515 原生架构重实现 + `2eb3bc8` v4→v4.1，**未上机验证过**）
- 开关：`SGLANG_NPU_MOE_TAIL_FUSION`（默认 0）；`SGLANG_NPU_MOE_TAIL_FUSION_AR`（EnvStr，`spin` 默认 / `hccl`）；`SGLANG_NPU_MOE_TAIL_FUSION_DEBUG`；`SGLANG_NPU_FUSED_TAIL_PROBE`（eager 取证专用，**严禁生产**）
- 适用：仅 35B（MoE）；9B 跳过
- 前提：`SGLANG_NPU_USE_MULTI_STREAM=1`（链中 add 是双流汇合特有节点）；仅 decode 激活；spin 变体需对称内存链路（libshmem + 每 engine 独立端口 `SHMEM_UID_SESSION_ID`）

## 改动摘要

层尾四节点链 `MoeFinalizeRoutingV2 → add → all_reduce → add_gemma_rms_norm` 内有三条跨流同步边（multi-stream join + AR fork/join），图模式下每次跨流 wait 解除后首个 kernel 派发约 5µs 固定延迟，合计约 15µs/层。两变体：

- **spin（默认，小 M 约 ≤32）**：单 kernel `fused_fin_ar_norm` 全链融合。add 借 `MoeFinalizeRoutingV2` 原生 skip1 残差输入消除（fp32 累加，精度变化 ≤1ulp）；AR 重构为 AIV 自旋 one-shot（share memory 直读写对端 + flag 自旋，不走 HCCL 节点）；配套 `fused_tail_zero` 清 flag 区。
- **hccl（大 M 约 ≥64）**：`fused_fin_add`（finalize+skip1 本地前段）+ stock HCCL AR + norm。无共享内存操作。

## A/B 方法

- A：`SGLANG_NPU_MOE_TAIL_FUSION=0`；B：`=1`（spin / hccl 分两次实验更佳）。
- 本仓实验核心目的：**0515 重实现首次上机验证**（正确性 + 收益）。gate 设计为通过即不可回退——前提不满足必须 fail-loud，gate 判定本身异常才静默回退 stock。
- profiling 对比重点：层尾窗口跨流同步边数量与窗口总时长；逐层约 15µs 跨流开销是否消除。
- v4.1 已含 capture-born tiling 修复（pinned 保活 + eri/norm_w 改 tensor 参数 + bf16 scales 直喂），若复测发现逐层 cast+memcpy 小算子流说明 wheel 非 v4.1。

## 精度验收口径

跨 rank 求和顺序变化，不承诺逐 bit 一致，走既定有界误差验收。

## 已知收益

（待补充——模拟单测：fin+add 融合在 HCCL AR 下收益恒正；自旋 AR+norm 融合增量仅小 M 域为正，M=64 开始劣化）

## 结果（Qwen3.5-35B-A3B）

| 臂 | decode ms/step | 层尾窗口 µs/层 | 备注 |
|---|---|---|---|
| A（关） | | | |
| B（spin） | | | |
| C（hccl） | | | |

profiling 归档：`results/qwen3.5-35b-a3b/profiling/14_moe_tail_fusion/{a,b,c}/`

## 备注

移植细节见 `npu-tp-fusion-v4-0515-rearch-anchor.md`（挂点落 `forward_npu(deferred=True)`，gate 用 `get_moe_a2a_backend().is_none()` + 量化方法类标记）。
