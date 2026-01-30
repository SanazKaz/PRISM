#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --partition=devel
#SBATCH --time 00:10:00
#SBATCH --job-name=PPO_DEVEL_SILLY
#SBATCH --output=jobs_files/PPO_DEVEL_SILLY.log


# --- Environment Setup ---
module purge
module load Anaconda3
source activate /data/stat-cadd/wolf7055/conda/envs/PRISM_25
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH



echo "Python executable: $(which python)"

# --- DEFINITIONS ---
export PROJECT_ROOT="/data/stat-cadd/wolf7055/PRISM"
PERMANENT_DATA_DIR="${PROJECT_ROOT}/data/BRD4_BD1/03_final_dataset"

# Scratch location
WORK_DIR="${TMPDIR}"
SCRATCH_DATASET_DIR="${WORK_DIR}/BRD4_BD1_test"

export DEBUG_PPO=0

# Seeds array
SEEDS=(42 976 123 789)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Checkpoint path for this seed
CHECKPOINT_DIR="/data/stat-cadd/wolf7055/PRISM/Log_Results/geometry_checks_0.5_flatness_checks_0.5_BD1_BRD4/checkpoints/tmp"
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
    --config "${PROJECT_ROOT}/configs/binding_moad_fa_ppo.yaml" \
    --warm_start_from_ddpm "/data/stat-cadd/wolf7055/PRISM/checkpoints/moad_fullatom_cond.ckpt" \
    --datadir "${SCRATCH_DATASET_DIR}" \
    --seed ${SEED}

# --- CLEANUP ---
echo "Training completed."
echo "Cleaning up scratch space..."
rm -rf "$SCRATCH_DATASET_DIR"

echo "Final GPU Usage:"
nvidia-smi