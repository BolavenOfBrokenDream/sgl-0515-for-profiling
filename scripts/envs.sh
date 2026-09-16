#!/bin/bash
# 14 优化点开关集中定义（实验顺序 = 0517 上仓顺序，prefetch 恒关不做）。
# 用法：
#   source scripts/envs.sh
#   make_env off 06_full_attention   # 打印 A 臂 export 行（开启实验序 < 6 的所有点）
#   make_env on  06_full_attention   # 打印 B 臂 export 行（开启实验序 <= 6 的所有点）
#   make_env baseline                # 打印全关基线 export 行
#   env_all_off                      # 直接在当前 shell 生效（export）全关基线
#   env_up_to 06_full_attention      # 直接在当前 shell 生效累计开启状态
#
# 开关均在 aclgraph capture 期固化：设置后必须重启服务才生效。

# 实验序列（13 点；07_moe_weight_prefetch 恒关，不入列）
POINTS=(
  01_mrope
  02_post_sample
  03_gmm
  04_gdn_mamba
  05_moe_front_fusion
  06_full_attention
  08_gdn_recurrent_decode_opt
  09_cast_elimination
  10_gdn_qkvzba_pack
  11_fused_qkvzba_conv1d
  12_recurrent_ascendc
  13_vgmm1
  14_moe_tail_fusion
)

# ---- 单点 ON（只列该点相对全关基线需要改动的变量；OFF 状态由 env_all_off 统一显式给出）----

_on_01_mrope() { # 本仓补丁 0001 新增开关
  echo "export SGLANG_NPU_MROPE_FASTPATH=1"
}

_on_02_post_sample() {
  echo "export SGLANG_NPU_POST_SAMPLE_OPT=1"      # 本仓补丁 0002：常驻小算子链优化
  echo "export SGLANG_NPU_ASYNC_EXPONENTIAL=1"    # 异步采样（仅 910C 生效）
  echo "export SGLANG_NPU_EXP_RACE_TRITON=1"      # Triton 融合消费 kernel
}

_on_03_gmm() {
  echo "export SGLANG_GMM2_TRITON=1"
}

_on_04_gdn_mamba() { # 本仓补丁 0003 新增开关（方案A split + 方案B strided 一体）
  echo "export SGLANG_NPU_GDN_MAMBA_OPT=1"
}

_on_05_moe_front_fusion() {
  echo "export SGLANG_MOE_FRONT_FUSION=1"
}

_on_06_full_attention() {
  echo "export SGLANG_NPU_FULL_ATTN_FUSION=1"
}

_on_08_gdn_recurrent_decode_opt() { # 与 12 同一调用点三选一，12 优先
  echo "export SGLANG_NPU_GDN_UPDATE_FUSED=1"
}

_on_09_cast_elimination() { # 本仓补丁 0004 新增开关
  echo "export SGLANG_NPU_CAST_ELIMINATION=1"
}

_on_10_gdn_qkvzba_pack() {
  echo "export SGLANG_NPU_GDN_QKVZBA_PACK=1"
  echo "export SGLANG_NPU_GDN_QKVZBA_PACK_MAX_M=256"
}

_on_11_fused_qkvzba_conv1d() {
  echo "export SGLANG_NPU_TP_ASCENDC_FUSION=1"
}

_on_12_recurrent_ascendc() { # 优先级高于 08
  echo "export SGLANG_NPU_GDN_RECURRENT_ASCENDC=1"
}

_on_13_vgmm1() { # 收益仅 C1（需 05 同开，累计协议下天然满足）
  echo "export SGLANG_NPU_VGMM1=1"
  echo "export SGLANG_NPU_VGMM1_MAX_M=1024"
}

_on_14_moe_tail_fusion() { # 需双流；spin 变体另需 libshmem + SHMEM_UID_SESSION_ID
  echo "export SGLANG_NPU_USE_MULTI_STREAM=1"
  echo "export SGLANG_NPU_MOE_TAIL_FUSION=1"
  echo "export SGLANG_NPU_MOE_TAIL_FUSION_AR=spin"
}

# ---- 全关基线（显式逐变量置关，含基线默认开的点）----

_all_off_lines() {
  cat <<'EOF'
export SGLANG_NPU_MROPE_FASTPATH=0
export SGLANG_NPU_POST_SAMPLE_OPT=0
export SGLANG_NPU_ASYNC_EXPONENTIAL=0
export SGLANG_NPU_EXP_RACE_TRITON=0
export SGLANG_GMM2_TRITON=0
export SGLANG_NPU_GDN_MAMBA_OPT=0
export SGLANG_MOE_FRONT_FUSION=0
export SGLANG_NPU_FULL_ATTN_FUSION=0
export SGLANG_NPU_MOE_PREFETCH=0
export SGLANG_NPU_GDN_UPDATE_FUSED=0
export SGLANG_NPU_CAST_ELIMINATION=0
export SGLANG_NPU_GDN_QKVZBA_PACK=0
export SGLANG_NPU_TP_ASCENDC_FUSION=0
export SGLANG_NPU_GDN_RECURRENT_ASCENDC=0
export SGLANG_NPU_VGMM1=0
export SGLANG_NPU_MOE_TAIL_FUSION=0
EOF
}

# ---- 组合逻辑 ----

_point_index() { # $1 = 点名，返回在 POINTS 中的下标，未找到返回 -1
  local i
  for i in "${!POINTS[@]}"; do
    [ "${POINTS[$i]}" = "$1" ] && { echo "$i"; return 0; }
  done
  echo "-1"
}

# make_env baseline           → 全关基线
# make_env off <NN_name>      → A 臂：开启实验序严格小于该点的所有点
# make_env on  <NN_name>      → B 臂：开启实验序小于等于该点的所有点
make_env() {
  local mode="$1" point="$2" idx limit=-1 i
  case "$mode" in
    baseline) limit=-1 ;;
    off) idx=$(_point_index "$point"); [ "$idx" = "-1" ] && { echo "unknown point: $point" >&2; return 1; }; limit=$((idx - 1)) ;;
    on)  idx=$(_point_index "$point"); [ "$idx" = "-1" ] && { echo "unknown point: $point" >&2; return 1; }; limit=$idx ;;
    *) echo "usage: make_env {baseline|off <point>|on <point>}" >&2; return 1 ;;
  esac
  _all_off_lines
  for ((i = 0; i <= limit; i++)); do
    "_on_${POINTS[$i]}"
  done
}

# 直接在当前 shell 生效（供 source 后手动起服务用）
env_all_off() { eval "$(_all_off_lines)"; }
env_up_to() { eval "$(make_env on "$1")"; }

# source 时直接执行不做事；作为脚本运行时打印用法
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  sed -n '1,12p' "$0"
fi
