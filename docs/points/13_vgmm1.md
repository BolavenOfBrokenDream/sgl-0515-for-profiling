# 13. vendored GMM1（vgmm1，TP 融合包）

- 实验序：13（0517 `1e04211` 拆分；0515 侧 `2eb3bc8` 移植，**未上机验证过**）
- 开关：`SGLANG_NPU_VGMM1`（默认 0）；`SGLANG_NPU_VGMM1_MAX_M`（EnvInt，默认 1024 形态门）；`SGLANG_NPU_VGMM1_DEBUG`（排障）
- 适用：仅 35B（MoE，w13 gate_up_proj 的 GMM1）；9B 跳过
- 依赖：C1 收益要求 `SGLANG_MOE_FRONT_FUSION=1` 同开（累积协议下 #5 已开）

## 改动摘要

stock `aclnnGroupedMatmulV5` 约 30µs 固定开销下限，其中约 15µs 为每核 256 组存在性扫描。方案为「调度前置 + 读表直取」：`vgmm1_sched`（AIV 单核建 row_offsets + block_table）+ `vgmm1_main`（复刻 stock GEMM、删扫描循环、按核号查表）。C1 增量：`vgmm1_sched_partial` 直接消费 front fusion 的 rank_hist partials，单核列归约出 counts/row_offsets 并建表，Triton `partials_cumsum` 节点整体退场。

## A/B 方法

- A：`SGLANG_NPU_VGMM1=0`；B：`=1`（#5 保持开）。
- 预期裁决（沿用 dev 线上机结论）：本体+sched **无收益**（热态 -17.6µs 是假象；生产冷态 memory bound 下可兑现约 8µs，恰被 sched 节点底价约 7µs + 每块 24ns 抵消）；**唯一收益来源是 C1**（消 Triton cumsum 节点，轻微收益）。
- 本仓实验的核心目的：**验证 0515 移植是否成立**（正确性 + C1 是否兑现轻微收益），而非翻案本体。
- profiling 对比重点：GMM1 窗口内 `partials_cumsum` 节点是否退场、`vgmm1_sched(_partial)` 节点开销、GMM1 主 kernel 时长。
- 守卫：单 tensor ND 连续、groupType=0 splitM、`group_list_type=1`、权重 `[E,N,K]` 连续；其余形态显式回退 stock；算子未注册（wheel 未重建）时 `hasattr` 探测静默回退。

## 已知收益

本体+sched 无收益（默认关的原因）；C1 轻微收益（dev 线结论，0515 待复测）。

## 结果（Qwen3.5-35B-A3B）

| 臂 | decode ms/step | GMM1 窗口 µs/层 | 备注 |
|---|---|---|---|
| A（关，stock+FF） | | | |
| B（开，C1 全链） | | | |

profiling 归档：`results/qwen3.5-35b-a3b/profiling/13_vgmm1/{a,b}/`

## 备注

主 kernel GEMM 定序与 stock 完全一致，数值逐 bit 一致（dev 线 T0/T1 7/7 case max_abs=0）。0515 挂点全部落在 `UnquantizedFusedMoEMethod.forward_npu`（未回移 0517 的四站管线重构）。
