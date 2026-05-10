#!/bin/bash
#SBATCH -J psegnet_train
#SBATCH --mail-type=END,FAIL
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24G
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=6
#SBATCH --output=sbatch_log/%j.out
#SBATCH --error=sbatch_log/%j.out

# Usage:
#   sbatch run_euler_psegnet.sh [options]
#
# Options are passed directly to scripts/train_psegnet.py. Common ones:
#   --batch_size 8          Default
#   --epochs 200            Default
#   --no_wandb              Disable W&B (useful for quick tests)
#   --resume PATH           Resume from a checkpoint .pth file
#   --use_all_sequences     Include held-out test sequences (final model)
#
# Dataset and output paths default to local folder (set below).
#
# Examples:
#   sbatch scripts/run_euler_psegnet.sh
#   sbatch scripts/run_euler_psegnet.sh --batch_size 4 --no_wandb

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_PATH="data/TrackPlant3D/versions"
OUTPUT_DIR="outputs/psegnet_trackplant3d"

# ── Environment setup ─────────────────────────────────────────────────────────
module purge
module load stack/2024-05 gcc/13.2.0 cuda/12.2.1 eth_proxy
echo "Loaded modules:"
module list
echo "Conda env: $CONDA_DEFAULT_ENV"

echo "=========================================="
echo "PSegNet Training - Euler Cluster"
echo "=========================================="
echo "Dataset path: $DATASET_PATH"
echo "Output dir:   $OUTPUT_DIR"
echo "Extra args:   $@"
echo "=========================================="
echo ""

nvidia-smi
echo ""

mkdir -p sbatch_log

python scripts/train_psegnet.py \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    "$@"

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
