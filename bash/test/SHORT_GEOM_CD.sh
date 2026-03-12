#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short
#SBATCH --time=04:45:00
#SBATCH --job-name=PRISM_GEOMETRY_TEST_GEN
#SBATCH --mail-user=wolf7055@ox.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --output=jobs_files/10_03_2026_prism_geometry_CD_test_gen.log
#SBATCH --error=jobs_files/10_03_2026_prism_geometry_CD_test_gen.log

module purge
module load Anaconda3
source activate /data/stat-cadd/wolf7055/conda/envs/PRISM_25

# --- PATHS ---
export PROJECT_ROOT="/data/stat-cadd/wolf7055/PRISM"
CHECKPOINT="${PROJECT_ROOT}/Log_Results/short_crossdocked_dataset_geometry_training/checkpoints/7275716/seed=42/epoch=09-reward=0.68.pt"
TEST_DIR="${PROJECT_ROOT}/data/cross_dock/processed_crossdock_noH_full_temp/test"
OUTDIR="${PROJECT_ROOT}/results/prism/geometry_reward/seed42"
CONFIG="${PROJECT_ROOT}/configs/ppo_config.yaml"
SCRIPT="${PROJECT_ROOT}/scripts/test.py"

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Checkpoint:   $CHECKPOINT"
echo "Test dir:     $TEST_DIR"
echo "Output dir:   $OUTDIR"
echo "=========================================="

# --- CHECKS ---
if [ ! -f "$CHECKPOINT" ]; then
    echo "[ERROR] Checkpoint not found: $CHECKPOINT"
    exit 1
fi

if [ ! -d "$TEST_DIR" ]; then
    echo "[ERROR] Test directory not found: $TEST_DIR"
    exit 1
fi

mkdir -p "$OUTDIR"
mkdir -p jobs_files

cd $PROJECT_ROOT

# --- GPU CHECK ---
which python
python - << 'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
nvidia-smi

# --- GENERATION ---
echo "Starting PRISM geometry reward test generation..."
7276030
srun python "$SCRIPT" \
    "$CHECKPOINT" \
    --config "$CONFIG" \
    --test_dir "$TEST_DIR" \
    --outdir "$OUTDIR" \
    --n_samples 100 \
    --batch_size 120 \
    --sanitize \
    --skip_existing

echo "Generation complete!"
echo "Output written to: $OUTDIR"