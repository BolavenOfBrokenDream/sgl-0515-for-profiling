# sgl-0515-for-profiling

Qwen3.5-9B（dense）与 Qwen3.5-35B-A3B（MoE，GDN 线性注意力 + 全注意力混合架构）在昇腾 910C 上的推理调优笔记仓。核心目标：**将 14 个已交付优化点逐点拆开**，支持每个优化点独立 A/B 实验（确认性能收益）与 A/B profiling 采集。

## 基线与优化点来源

- **代码基线**：0515 交付仓 [`sglang-npu-opt-0515-transfer`](https://github.com/BolavenOfBrokenDream/sglang-npu-opt-0515-transfer) `main@6bcca68`（sglang 0.5.15 现网对齐线，sglang + sgl-kernel-npu 同仓两目录树）。14 个优化点已全部合入该基线。
- **优化点清单**：《NPU sglang 推理优化点交付总结（0517 优化线 / 0515 交付仓）》，实验顺序按 0517 开发仓 `sglang-npu-opt-stack` 的上仓顺序，见 `docs/experiment-plan.md`。
- **本仓补丁**：`patches/` 下 4 个补丁，为基线中"总是开启、无开关"的 4 个优化点（mrope / post_sample 常驻部分 / gdn_mamba / cast_elimination）补运行时开关，**默认开、行为与基线逐位一致，显式置 0 回退优化前行为**。其余 10 个点基线已自带开关，无需补丁。

## 目录结构

```
patches/                  # 基于 0515 main@6bcca68 的开关补丁（git apply 按序套用）
scripts/                  # 服务器侧脚本：开关集中定义、A/B 实验、decode 压测、profiling 采集
docs/
  switch-matrix.md        # 14 点 × 开关 × 默认值 × 生效条件 总表
  experiment-plan.md      # 实验顺序（0517 上仓顺序）、A/B 协议、profiling 协议
  points/NN_<name>.md     # 每个优化点一篇笔记：改动点、开关、A/B 方法、9B/35B 结果、profiling 记录
results/
  qwen3.5-9b/             # 9B 实验数据（csv）与 profiling 归档（人工回拷）
  qwen3.5-35b-a3b/        # 35B 同上
```

## 使用流程（服务器侧）

```bash
# 0. 部署基线（详见交付仓 deliver/README.md 的挂载/编译指南），然后套用开关补丁
cd <ROOT>   # 0515 交付仓 checkout @6bcca68
git apply <本仓>/patches/0001_mrope_fastpath_switch.patch
git apply <本仓>/patches/0002_post_sample_always_on_switch.patch
git apply <本仓>/patches/0003_gdn_mamba_opt_switch.patch
git apply <本仓>/patches/0004_cast_elimination_switch.patch

# 1. 纯 python 改动，无需重编 wheel；按部署铁律：若重装了 wheel，补丁最后重新打

# 2. 逐点 A/B（顺序见 docs/experiment-plan.md）
source scripts/envs.sh
make_env off 01_mrope > /tmp/exp_off.env   # 全关基线
make_env on  01_mrope > /tmp/exp_on.env    # 基线 + 该点
# 分别 source 后启动服务 → scripts/bench_decode.sh → scripts/collect_profiling.sh
# 结果按 results/<model>/ 下的命名约定归档并人工回拷
```

注意：绝大多数开关在 NPU 图模式（aclgraph）capture 期固化进计算图，**必须在服务启动前设置，capture 后修改无效**；A/B 两臂各需一次完整服务重启。

## 14 个优化点速览（实验顺序 = 0517 上仓顺序）

| 顺序 | 点 | 短名 | 主开关 | 默认 |
|---|---|---|---|---|
| 1 | mRoPE device positions 快路径 | mrope | `SGLANG_NPU_MROPE_FASTPATH`（本仓补丁新增） | 1 |
| 2 | 异步 exponential-race 采样与小算子链 | post_sample | `SGLANG_NPU_ASYNC_EXPONENTIAL` 等 3 个 | 0/1/1 |
| 3 | persistent GMM2（down_proj） | gmm | `SGLANG_GMM2_TRITON` | 0 |
| 4 | GDN 融合 qkvzba split + strided 直读 | gdn_mamba | `SGLANG_NPU_GDN_MAMBA_OPT`（本仓补丁新增） | 1 |
| 5 | MoE 路由前段融合 | moe_front_fusion | `SGLANG_MOE_FRONT_FUSION` | 0 |
| 6 | decode 全注意力段三件套融合 | full_attention | `SGLANG_NPU_FULL_ATTN_FUSION` | 1 |
| 7 | MoE 专家权重 L2 预取 | moe_weight_prefetch | `SGLANG_NPU_MOE_PREFETCH` | 0 |
| 8 | GDN decode 专用 recurrent kernel | gdn_recurrent_decode_opt | `SGLANG_NPU_GDN_UPDATE_FUSED` | 0 |
| 9 | decode 图内边角 cast 消除 | cast_elimination | `SGLANG_NPU_CAST_ELIMINATION`（本仓补丁新增） | 1 |
| 10 | GDN 输入投影 qkvz+ba 打包 | gdn_qkvzba_pack | `SGLANG_NPU_GDN_QKVZBA_PACK` | 1 |
| 11 | split+causal_conv1d 单 kernel 融合 | fused_qkvzba_conv1d | `SGLANG_NPU_TP_ASCENDC_FUSION` | 0 |
| 12 | AscendC GDN decode recurrent | recurrent_ascendc | `SGLANG_NPU_GDN_RECURRENT_ASCENDC` | 0 |
| 13 | vendored GMM1 | vgmm1 | `SGLANG_NPU_VGMM1` | 0 |
| 14 | MoE 层尾 fin+add+AR+norm 融合 | moe_tail_fusion | `SGLANG_NPU_MOE_TAIL_FUSION` | 0 |
