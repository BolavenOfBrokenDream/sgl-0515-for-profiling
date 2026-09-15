#!/bin/bash
# A/B profiling 采集：对当前已启动的服务，经 sglang /start_profile /stop_profile
# 端点采集一段稳态 decode 窗口的 torch_npu profiler 数据并归档。
# 前提：服务以 SGLANG_TORCH_PROFILER_DIR=<可写目录> 启动（NPU 上 sglang 路由到 torch_npu profiler）。
# 用法：scripts/collect_profiling.sh <model_tag> <NN_point> <arm>
# 可调环境变量：SGL_HOST/SGL_PORT、PROF_NUM_PROMPTS（默认 64，短窗口）、
#   PROF_INPUT_LEN/PROF_OUTPUT_LEN（默认 1024/128——decode 窗口为主，output 不必长）
# 产物：results/<model_tag>/profiling/<NN_point>/<arm>/（原始 trace + 汇总）

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

MODEL_TAG="${1:?usage: collect_profiling.sh <model_tag> <NN_point> <arm>}"
POINT="${2:?}"
ARM="${3:?}"

HOST="${SGL_HOST:-127.0.0.1}"
PORT="${SGL_PORT:-30000}"
NUM_PROMPTS="${PROF_NUM_PROMPTS:-64}"
INPUT_LEN="${PROF_INPUT_LEN:-1024}"
OUTPUT_LEN="${PROF_OUTPUT_LEN:-128}"
CONCURRENCY="${BENCH_MAX_CONCURRENCY:-32}"

OUT_DIR="$REPO_DIR/results/$MODEL_TAG/profiling/$POINT/$ARM"
mkdir -p "$OUT_DIR"

: "${SGLANG_TORCH_PROFILER_DIR:?需与服务启动时的 SGLANG_TORCH_PROFILER_DIR 一致（服务端目录）}"

echo "[1/3] start_profile -> $SGLANG_TORCH_PROFILER_DIR"
curl -sS -X POST "http://$HOST:$PORT/start_profile" \
  -H "Content-Type: application/json" \
  -d "{\"output_dir\": \"$SGLANG_TORCH_PROFILER_DIR\", \"activities\": [\"CPU\", \"NPU\"], \"with_stack\": false}"

echo "[2/3] 发压测流量（稳态 decode 窗口）"
python3 -m sglang.bench_serving \
  --backend sglang --host "$HOST" --port "$PORT" \
  --dataset-name random \
  --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
  --num-prompts "$NUM_PROMPTS" --max-concurrency "$CONCURRENCY" \
  --output-file "$OUT_DIR/traffic_bench.json"

echo "[3/3] stop_profile 并归档"
curl -sS -X POST "http://$HOST:$PORT/stop_profile"
# profiler 产物在服务端 SGLANG_TORCH_PROFILER_DIR 下，按部署方式拷贝回 $OUT_DIR
# （挂载共享盘时直接 cp -r "$SGLANG_TORCH_PROFILER_DIR"/* "$OUT_DIR"/）
if [ -d "$SGLANG_TORCH_PROFILER_DIR" ]; then
  cp -r "$SGLANG_TORCH_PROFILER_DIR"/* "$OUT_DIR"/ 2>/dev/null || true
fi

cat > "$OUT_DIR/META.txt" <<EOF
model=$MODEL_TAG
point=$POINT
arm=$ARM
num_prompts=$NUM_PROMPTS input_len=$INPUT_LEN output_len=$OUTPUT_LEN concurrency=$CONCURRENCY
profiler_dir=$SGLANG_TORCH_PROFILER_DIR
EOF

echo "归档于 $OUT_DIR；汇总读数填入 docs/points/$POINT.md。"
