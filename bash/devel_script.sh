#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=short
#SBATCH --time 04:00:00
#SBATCH --job-name=weighted_sum_cd_brd4_bd1
#SBATCH --output=jobs_files/weighted_sum_cd_brd4_bd1_%a.log
#SBATCH --array=0-1


# --- Environment Setup ---
module purge
module load Anaconda3
source activate /data/stat-cadd/wolf7055/conda/envs/PRISM_25


echo "Python executable: $(which python)"

# --- DEFINITIONS ---

export PROJECT_ROOT="/data/stat-cadd/wolf7055/PRISM"

PERMANENT_DATA_ROOT="${PROJECT_ROOT}/data/BRD4_BD1"

# Scratch location
WORK_DIR="${TMPDIR}"
SCRATCH_ROOT="${WORK_DIR}/BRD4_BD1_test"
SCRATCH_DATASET_DIR="${SCRATCH_ROOT}/03_final_dataset"

export DEBUG_PPO=0

# Seeds array
SEEDS=(42 976)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Checkpoint path for this seed
CHECKPOINT_PATH="/data/stat-cadd/wolf7055/PRISM/Log_Results/molecular_props_geom_step2/molecular_props_geom_step2/checkpoints/BRD4_BD1/seed=42/last.ckpt"
# CHECKPOINT_PATH="${CHECKPOINT_DIR}/seed=${SEED}/last.ckpt"

echo "=========================================="
echo "Project Root: ${PROJECT_ROOT}"
echo "Source Data:  ${PERMANENT_DATA_ROOT}"
echo "Scratch Dir:  ${SCRATCH_DATASET_DIR}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Using Seed: ${SEED}"
echo "Resuming from: ${CHECKPOINT_PATH}"
echo "=========================================="

cd $PROJECT_ROOT

# --- Permissions ---
chmod +x "${PROJECT_ROOT}/val_analysis/smina.static"

# --- STAGE IN: Copy Data to Scratch ---
echo "Step 1: Staging data to fast storage..."
echo "Start time: $(date)"

mkdir -p "$SCRATCH_ROOT"
rsync -ah "${PERMANENT_DATA_ROOT}/02_preprocessed/" "${SCRATCH_ROOT}/02_preprocessed/"
rsync -ah "${PERMANENT_DATA_ROOT}/03_final_dataset/" "${SCRATCH_ROOT}/03_final_dataset/"

echo "Data staging complete."
ls -F "$SCRATCH_DATASET_DIR" | head -n 5
echo "=========================================="

# --- Diagnostics ---
echo "Initial GPU Usage:"
nvidia-smi

# --- RUN TRAINING ---
echo "Starting PRISM training using Scratch Data..."

srun python "${PROJECT_ROOT}/scripts/train.py" \
    --config "${PROJECT_ROOT}/configs/weighted_sum_cd.yaml" \
    --resume_from_checkpoint "${CHECKPOINT_PATH}" \
    --datadir "${SCRATCH_DATASET_DIR}" \
    --seed ${SEED}

# --- CLEANUP ---
echo "Training completed."
echo "Cleaning up scratch space..."
rm -rf "$SCRATCH_ROOT"

echo "Final GPU Usage:"
nvidia-smi