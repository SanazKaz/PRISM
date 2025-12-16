#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --partition=devel
#SBATCH --time 00:10:00
#SBATCH --job-name=PRISM_scratch_test
#SBATCH --output=jobs_files/PRISM_scratch_test.log

# --- Environment Setup ---
module purge
module load Anaconda3
source activate /data/stat-cadd/wolf7055/conda/envs/prism_backup
echo "Python executable: $(which python)"

# --- DEFINITIONS ---
export PROJECT_ROOT="/data/stat-cadd/wolf7055/PRISM"
# 1. Define your Safe Data Source (The Warehouse)
PERMANENT_DATA_DIR="${PROJECT_ROOT}/data/AMPC_beta_lactamase/03_final_dataset"

chmod +x /data/stat-cadd/wolf7055/PRISM/val_analysis/smina.static # make exc # SBATCH --mem=GB


# 2. Define your Fast Scratch Location (The Workbench)
# ARC uses $TMPDIR for local node storage.
WORK_DIR="${TMPDIR}"
# We create a subfolder so we don't mess up the root of scratch
SCRATCH_DATASET_DIR="${WORK_DIR}/AMPC_beta_lactamase_test"

echo "Project Root: ${PROJECT_ROOT}"
echo "Source Data:  ${PERMANENT_DATA_DIR}"
echo "Scratch Dir:  ${SCRATCH_DATASET_DIR}"

# Change to the project directory
cd $PROJECT_ROOT

# --- Permissions ---
chmod +x "${PROJECT_ROOT}/val_analysis/smina.static"

# --- STAGE IN: Copy Data to Scratch ---
echo "=========================================="
echo "Step 1: Staging data to fast storage..."
echo "Start time: $(date)"

mkdir -p "$SCRATCH_DATASET_DIR"
# Copy contents of permanent dir to scratch dir
rsync -ah "${PERMANENT_DATA_DIR}/" "${SCRATCH_DATASET_DIR}/"

echo "Data staging complete."
echo "Listing files in scratch to verify:"
ls -F "$SCRATCH_DATASET_DIR" | head -n 5
echo "=========================================="

# --- Diagnostics ---
echo "Initial GPU Usage:"
nvidia-smi

# --- RUN TRAINING ---
# We point --datadir to the SCRATCH_DATASET_DIR
echo "Starting PRISM training using Scratch Data..."

srun python "${PROJECT_ROOT}/scripts/train.py" \
    --config "${PROJECT_ROOT}/configs/ppo_devel.yaml" \
    --resume_from_checkpoint "/data/stat-cadd/wolf7055/PRISM/Log_Results/PB_Geo_Flat_1.0/checkpoints/seed=789/epoch=09-reward=0.83.ckpt" \
    --datadir "${SCRATCH_DATASET_DIR}"

# --- CLEANUP ---
# ARC says TMPDIR is auto-cleaned, but this confirms it for your own peace of mind
echo "Training completed."
echo "Cleaning up scratch space..."
rm -rf "$SCRATCH_DATASET_DIR"

echo "Final GPU Usage:"
nvidia-smi