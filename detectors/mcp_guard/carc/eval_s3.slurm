#!/bin/bash --login
# Full MCP-Guard (Stage I+II+III) on one split, with an open LLM served by vLLM on the same GPU node.
#
#   sbatch detectors/mcp_guard/carc/eval_s3.slurm                      # defaults below
#   sbatch --export=ALL,MODEL=meta-llama/Meta-Llama-3-8B-Instruct,SPLIT=val,MODE=released \
#          detectors/mcp_guard/carc/eval_s3.slurm
#
# Gated models (Llama, Gemma, Mistral) need HF_TOKEN in the repo's .env and accepted licenses.
# Never set SPLIT=test unless the team has agreed the final evaluation run is happening.
#SBATCH --job-name=mcpguard-s3
#SBATCH --account=xiangren_1987
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/project2/xiangren_1987/group_12/logs/%x-%j.out

set -euo pipefail
GROUP=/project2/xiangren_1987/group_12
REPO=$GROUP/MCP-PROBE
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
SPLIT=${SPLIT:-tune}
MODE=${MODE:-paper}
PROMPT=${PROMPT:-upstream}
PORT=${PORT:-$((8000 + SLURM_JOB_ID % 1000))}

export UV_CACHE_DIR=/scratch1/$USER/.cache/uv
export HF_HOME=/scratch1/$USER/.cache/huggingface
export PATH=$HOME/.local/bin:$PATH
cd "$REPO"
set -a; [ -f .env ] && . ./.env; set +a

if [ "$SPLIT" = "test" ] && [ "${FINAL_EVAL:-}" != "1" ]; then
  echo "Refusing SPLIT=test without FINAL_EVAL=1"; exit 1
fi

"$GROUP/envs/vllm/bin/vllm" serve "$MODEL" --port "$PORT" --max-model-len 4096 \
  --gpu-memory-utilization 0.80 --disable-log-requests > "$GROUP/logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

for i in $(seq 1 120); do
  curl -sf "http://localhost:$PORT/v1/models" >/dev/null && break
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM exited; see logs/vllm-$SLURM_JOB_ID.log"; exit 1; }
  sleep 10
done

TAG=$(echo "$MODEL" | tr '/:' '__')_${PROMPT}
uv run python -m detectors.mcp_guard.evaluate --split "$SPLIT" --mode "$MODE" --stages s1,s2,s3 \
  --s3-model "$MODEL" --s3-base-url "http://localhost:$PORT/v1" --s3-prompt "$PROMPT" --tag "$TAG"
