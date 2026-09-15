#!/bin/bash
# 单优化点 A/B 驱动：生成两臂 env 文件并（可选）自动跑两轮 bench。
# 用法：
#   scripts/run_ab.sh <NN_point> <model_tag>        例：scripts/run_ab.sh 06_full_attention qwen3.5-35b-a3b
#
# 默认只生成 env 文件并打印手动步骤（服务启动方式因部署而异）。
# 若定义了 SGL_SERVER_START_CMD / SGL_SERVER_STOP_CMD（在命令前可用的启动/停止命令），
# 则自动执行：起 A → bench → 停 → 起 B → bench → 停。bench 由 scripts/bench_decode.sh 执行。

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

POINT="${1:?usage: run_ab.sh <NN_point> <model_tag>}"
MODEL_TAG="${2:?usage: run_ab.sh <NN_point> <model_tag>}"

source "$SCRIPT_DIR/envs.sh"

OUT_DIR="$REPO_DIR/results/$MODEL_TAG"
mkdir -p "$OUT_DIR"

A_ENV="$OUT_DIR/${POINT}_a.env"
B_ENV="$OUT_DIR/${POINT}_b.env"
make_env off "$POINT" > "$A_ENV"
make_env on  "$POINT" > "$B_ENV"

echo "A 臂（对照，实验序 < $POINT 累计开启）env: $A_ENV"
echo "B 臂（实验，实验序 <= $POINT 累计开启）env: $B_ENV"
diff "$A_ENV" "$B_ENV" || true

run_arm() { # $1 = arm(a|b), $2 = env file
  local arm="$1" env_file="$2"
  echo "===== arm $arm : source $env_file ====="
  set -a; source "$env_file"; set +a
  if [ -n "${SGL_SERVER_START_CMD:-}" ]; then
    eval "$SGL_SERVER_START_CMD"
    "$SCRIPT_DIR/bench_decode.sh" "$MODEL_TAG" "$POINT" "$arm"
    eval "${SGL_SERVER_STOP_CMD:-true}"
  else
    cat <<EOF
[手动步骤] 在当前 shell 执行：
  set -a; source $env_file; set +a
  <按部署方式启动服务，等待 capture 完成与 warmup>
  $SCRIPT_DIR/bench_decode.sh $MODEL_TAG $POINT $arm
  <停止服务>
EOF
  fi
}

run_arm a "$A_ENV"
run_arm b "$B_ENV"

echo "结果 csv 位于 $OUT_DIR/${POINT}_{a,b}.csv，profiling 用 collect_profiling.sh 采集。"
echo "汇总读数填入 docs/points/${POINT}.md 的结果表。"
