#!/bin/bash --login
# Fine-tune MCP-Guard Stage II on MCPTox tune for every init x scope x seed, then
# evaluate each checkpoint (Stage I+II, paper mode) on tune and val. Never touches test.
#
#   sbatch detectors/mcp_guard/carc/train_sweep.slurm
#   sbatch --export=ALL,INITS="mcpguard",SCOPES="full",SEEDS="0 1 2" detectors/mcp_guard/carc/train_sweep.slurm
#SBATCH --job-name=mcpguard-train
#SBATCH --account=xiangren_1987
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=/project2/xiangren_1987/group_12/logs/%x-%j.out

set -euo pipefail
GROUP=/project2/xiangren_1987/group_12
REPO=$GROUP/MCP-PROBE
CKPT_ROOT=${CKPT_ROOT:-$GROUP/checkpoints/mcp_guard}
INITS=${INITS:-"mcpguard e5base"}
SCOPES=${SCOPES:-"full head"}
SEEDS=${SEEDS:-"0 1 2 3 4"}

export UV_CACHE_DIR=/scratch1/$USER/.cache/uv
export HF_HOME=/scratch1/$USER/.cache/huggingface
export PATH=$HOME/.local/bin:$PATH
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# Fail fast instead of silently training on CPU if torch can't see the GPU (e.g. driver too old).
uv run python -c "import torch, sys; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)"

# Frozen MCP-Guard reference on the same splits.
for SPLIT in tune val; do
  uv run python -m detectors.mcp_guard.evaluate --split $SPLIT --stages s1,s2 --tag frozen
done

for INIT in $INITS; do
  for SCOPE in $SCOPES; do
    for SEED in $SEEDS; do
      RUN=${INIT}_${SCOPE}_seed${SEED}
      uv run python -m detectors.mcp_guard.train --init $INIT --scope $SCOPE --seed $SEED --out-root "$CKPT_ROOT"
      for SPLIT in tune val; do
        uv run python -m detectors.mcp_guard.evaluate --split $SPLIT --stages s1,s2 \
          --s2-path "$CKPT_ROOT/$RUN" --tag "ft_$RUN"
      done
    done
  done
done

uv run python -m detectors.mcp_guard.summarize_finetune
chmod -R g+rwX "$CKPT_ROOT" results/mcp_guard 2>/dev/null || true
