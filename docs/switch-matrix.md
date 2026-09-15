# 14 优化点开关矩阵

基线：0515 交付仓 `main@6bcca68`。标注 *(补丁)* 的开关由本仓 `patches/` 新增，默认 1（保持基线行为），显式置 0 回退优化前行为。除注明外均为 `EnvBool`（`1`/`0`），且在 aclgraph capture 期固化，**须服务启动前设置**。

| # | 短名 | 开关 | 默认 | 生效条件 / 回退 |
|---|------|------|------|----------------|
| 1 | mrope | `SGLANG_NPU_MROPE_FASTPATH` *(补丁)* | 1 | 非 MTP、非扩散 LM，且 `rl_on_policy_target` 已设置或全 batch 无多模态输入；含真实多模态自动回退原逐 request 构造 |
| 2 | post_sample | `SGLANG_NPU_ASYNC_EXPONENTIAL` | 0 | 异步采样总开关；仅 910C（设备名前缀 `Ascend910_93`）、decode、非投机/verify、无 grammar、非全 greedy、无 per-request seed、无 top-k/p/min-p |
|  |  | `SGLANG_NPU_EXP_RACE_TRITON` | 1 | Triton 融合消费 kernel，仅在上者实际启用时生效 |
|  |  | `SGLANG_NPU_POST_SAMPLE_OPT` *(补丁)* | 1 | 常驻部分总开关：全 1.0 温度跳过 RealDiv、softmax 不写回、logprob clamp 下沉 + gather 提取；置 0 逐点回退 stock |
| 3 | gmm | `SGLANG_GMM2_TRITON` | 0 | 仅 NPU BF16 无量化、无 bias；与 `SGLANG_MOE_FRONT_FUSION=1` 同开时 exclusive offsets 直喂省前置 kernel |
| 4 | gdn_mamba | `SGLANG_NPU_GDN_MAMBA_OPT` *(补丁)* | 1 | 方案A（NPU 融合 qkvzba split，qwen3_5.py）+ 方案B（recurrent kernel strided 直读，sgl-kernel-npu fla）共用；置 0 时方案A 走旧 fallback、方案B wrapper 物化 contiguous |
| 5 | moe_front_fusion | `SGLANG_MOE_FRONT_FUSION` | 0 | 仅无 grouped topk、无 correction bias 的 fast path；renorm 单算子另需 renormalize 且无 fused shared expert；init_routing 形态门 M≤512 且 M%8==0、H 为 2 的幂、bf16；仅 BF16 无量化 |
| 6 | full_attention | `SGLANG_NPU_FULL_ATTN_FUSION` | 1 | scatter：decode、每 rank q∈{1,2,4}、kv=1、head_dim=256、rope=64、带 gate、bf16 连续、use_fia 布局；sigmoid_mul：同形 bf16 连续；norm v2：2D bf16 连续、hidden 为 2 的幂。未命中自动回退 |
|  |  | `SGLANG_NPU_FULL_ATTN_FUSION_DEBUG` | 0 | 打印未启用原因 |
| 7 | moe_weight_prefetch | `SGLANG_NPU_MOE_PREFETCH` | 0 | **本仓实验恒关（不做）**。仅 aclgraph capture 期发射、仅 GDN 层、容量门控 fail-closed；配套 `SGLANG_NPU_MOE_PREFETCH_OPS/MODE/CHUNK_MIB/BUDGET_MIB` |
| 8 | gdn_recurrent_decode_opt | `SGLANG_NPU_GDN_UPDATE_FUSED` | 0 | decode 专用 Triton recurrent；优先级低于 `SGLANG_NPU_GDN_RECURRENT_ASCENDC`，同开不生效 |
| 9 | cast_elimination | `SGLANG_NPU_CAST_ELIMINATION` *(补丁)* | 1 | 三处：GDN 元数据 int64 上提 + int32 影子 buffer；MoE gating bf16 直喂（以 #5 开为前提）；dt_bias 源头 fp32。置 0 逐处回退 |
| 10 | gdn_qkvzba_pack | `SGLANG_NPU_GDN_QKVZBA_PACK` | 1 | 仅 NPU 非量化 bf16/fp16；与双流分支互斥 |
|  |  | `SGLANG_NPU_GDN_QKVZBA_PACK_MAX_M` (EnvInt) | 256 | 小 M 门控，超过不生效 |
| 11 | fused_qkvzba_conv1d | `SGLANG_NPU_TP_ASCENDC_FUSION` | 0 | op1 总开关；仅 decode 且 init 形状守卫命中；运行期未命中静默回退 split + stock conv |
|  |  | `SGLANG_NPU_TP_ASCENDC_FUSION_QKVZBA` | 随总开关 | op1 分开关，显式置 0 可单独关 op1 |
| 12 | recurrent_ascendc | `SGLANG_NPU_GDN_RECURRENT_ASCENDC` | 0 | decode 三选一最高优先级（ASCENDC > #8 > stock Triton）；import 期绑定，运行期守卫未命中逐调用回退 stock |
| 13 | vgmm1 | `SGLANG_NPU_VGMM1` | 0 | 本体+sched 无收益（热态假象），收益仅来自 C1（与 #5 同开时 `vgmm1_sched_partial` 消 cumsum）；`SGLANG_NPU_VGMM1_MAX_M` (EnvInt)=1024 形态门；`_DEBUG` 排障 |
| 14 | moe_tail_fusion | `SGLANG_NPU_MOE_TAIL_FUSION` | 0 | 仅 `SGLANG_NPU_USE_MULTI_STREAM=1` 下生效、仅 decode；`SGLANG_NPU_MOE_TAIL_FUSION_AR` (EnvStr)=`spin`/`hccl`（spin 需对称内存链路：libshmem + 每 engine 独立端口 `SHMEM_UID_SESSION_ID`）；`_DEBUG` 打印守卫；`SGLANG_NPU_FUSED_TAIL_PROBE` 严禁生产 |

## 关键交互关系

- #3 GMM2 的 offsets 直喂收益要求 #5 同开。
- #9 的 MoE gating bf16 直喂落在 #5 的 fast path 内，#5 关则该部分不生效；#9 部分收益以 #11/#12 开启为前提，#12 关时零副作用。
- #13 的 C1 收益要求 #5 同开；本体+sched 已裁决无收益，实验中建议 #13 只在与 #5 组合时验证 C1。
- #14 要求双流 `SGLANG_NPU_USE_MULTI_STREAM=1`（基线中 qwen2_moe 双流重排由该开关控制）。
- #8 与 #12 互斥（#12 优先），二者是同一调用点的两个候选实现，应分别实验。
- #10 与双流分支互斥：双流命中时不走打包路径。
- MoE 相关点（#3/#5/#7/#13/#14）仅适用于 35B-A3B，9B dense 无 MoE，实验中跳过。
