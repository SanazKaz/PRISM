#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --partition=devel
#SBATCH --time 00:10:00
#SBATCH --job-name=silly_walks_test2
#SBATCH --output=jobs_files/silly_walks_test2.log


# --- Environment Setup ---
module purge
module load Anaconda3
source activate /data/stat-cadd/wolf7055/conda/envs/prism_backup
echo "Python executable: $(which python)"

# --- DEFINITIONS ---
export PROJECT_ROOT="/data/stat-cadd/wolf7055/PRISM"
PERMANENT_DATA_DIR="${PROJECT_ROOT}/data/AMPC_beta_lactamase/03_final_dataset"

# Scratch location
WORK_DIR="${TMPDIR}"
SCRATCH_DATASET_DIR="${WORK_DIR}/AMPC_beta_lactamase_test"

# Seeds array
SEEDS=(42 976 123 789)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Checkpoint path for this seed
CHECKPOINT_DIR="/data/stat-cadd/wolf7055/PRISM/Log_Results/linear_squared_feature_density_300_timesteps_lr_1e-5/checkpoints/tmp"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/seed=${SEED}/last.ckpt"

echo "=========================================="
echo "Project Root: ${PROJECT_ROOT}"
echo "Source Data:  ${PERMANENT_DATA_DIR}"
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

mkdir -p "$SCRATCH_DATASET_DIR"
rsync -ah "${PERMANENT_DATA_DIR}/" "${SCRATCH_DATASET_DIR}/"

echo "Data staging complete."
ls -F "$SCRATCH_DATASET_DIR" | head -n 5
echo "=========================================="

# --- Diagnostics ---
echo "Initial GPU Usage:"
nvidia-smi

# --- RUN TRAINING ---
echo "Starting PRISM training using Scratch Data..."

srun python "${PROJECT_ROOT}/scripts/train.py" \
    --config "${PROJECT_ROOT}/configs/ppo_devel.yaml" \
    --resume_from_checkpoint "${CHECKPOINT_PATH}" \
    --datadir "${SCRATCH_DATASET_DIR}" \
    --seed ${SEED}

# --- CLEANUP ---
echo "Training completed."
echo "Cleaning up scratch space..."
rm -rf "$SCRATCH_DATASET_DIR"

echo "Final GPU Usage:"
nvidia-smi