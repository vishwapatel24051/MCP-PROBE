#!/bin/bash --login
# One-time MCP-Guard setup on USC CARC Discovery (CPU node, no GPU needed).
#   - clones this repo's mcp-guard branch into the team dir (or updates it)
#   - uv env + MCP-Guard extras (and, with INSTALL_VLLM=1, a separate vLLM env for Stage III)
#   - MCPTox at the pinned commit + processed splits
#   - pre-downloads the MCP-Guard checkpoint and BERT tokenizer
# Submit from a login node:  sbatch detectors/mcp_guard/carc/setup.slurm
#SBATCH --job-name=mcpguard-setup
#SBATCH --account=xiangren_1987
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/project2/xiangren_1987/group_12/logs/%x-%j.out

set -euo pipefail
GROUP=/project2/xiangren_1987/group_12
REPO=$GROUP/MCP-PROBE
REPO_URL=${REPO_URL:-https://github.com/naman-gupta-02/MCP-PROBE.git}
BRANCH=${BRANCH:-mcp-guard}
MCPTOX_COMMIT=f85189f9ad12504c197c7f920ab818a40657b1fa

# Keep caches off the 100 GiB home quota.
export UV_CACHE_DIR=/scratch1/$USER/.cache/uv
export HF_HOME=/scratch1/$USER/.cache/huggingface
export PATH=$HOME/.local/bin:$PATH

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

if [ -d "$REPO/.git" ]; then
  git -C "$REPO" fetch origin && git -C "$REPO" checkout "$BRANCH" && git -C "$REPO" pull --ff-only
else
  git clone -b "$BRANCH" "$REPO_URL" "$REPO"
fi
cd "$REPO"

uv sync
uv pip install -r detectors/mcp_guard/requirements.txt

if [ ! -d data/raw/MCPTox-Benchmark ]; then
  git clone https://github.com/zhiqiangwang4/MCPTox-Benchmark data/raw/MCPTox-Benchmark
fi
git -C data/raw/MCPTox-Benchmark checkout -q "$MCPTOX_COMMIT"
uv run python scripts/build_splits.py

uv run python -c "from detectors.mcp_guard.stage2 import NeuralStage; NeuralStage(device='cpu'); print('stage2 ok')"

# Separate env for the vLLM server (Stage III with open models); opt in with INSTALL_VLLM=1.
# vLLM pins its own torch, so it gets its own venv.
if [ "${INSTALL_VLLM:-0}" = "1" ] && [ ! -x "$GROUP/envs/vllm/bin/vllm" ]; then
  uv venv --python 3.11 "$GROUP/envs/vllm"
  uv pip install --python "$GROUP/envs/vllm/bin/python" vllm
fi

uv run pytest -q
chmod -R g+rwX "$REPO" "$GROUP/envs" 2>/dev/null || true
echo SETUP_OK
