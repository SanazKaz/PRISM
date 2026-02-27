#!/bin/bash

# --- DEFINITIONS ---
export PROJECT_ROOT="/data/stat-cadd/wolf7055/PRISM"

WORK_DIR="/tmp"

PERMANENT_DATA_ROOT="${PROJECT_ROOT}/data/BRD4_BD1"
SCRATCH_ROOT="${WORK_DIR}/BRD4_BD1"
SCRATCH_DATASET_DIR="${SCRATCH_ROOT}/03_final_dataset"

export DEBUG_PPO=0

SEED=42
CHECKPOINT_PATH="/data/stat-cadd/wolf7055/PRISM/Log_Results/molecular_props_geom_step2/molecular_props_geom_step2/checkpoints/BRD4_BD1/seed=42/last.ckpt"

echo "=========================================="
echo "Project Root: ${PROJECT_ROOT}"
echo "Source Data:  ${PERMANENT_DATA_ROOT}"
echo "Scratch Dir:  ${SCRATCH_DATASET_DIR}"
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
echo "Starting PRISM training..."

python "${PROJECT_ROOT}/scripts/train.py" \
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