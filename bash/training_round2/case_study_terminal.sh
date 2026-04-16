#!/bin/bash

module purge
module load Anaconda3
source activate /data/stat-cadd/wolf7055/conda/envs/PRISM_25

# =============================================================================
# 1. PATH DEFINITIONS
# =============================================================================
export PROJECT_ROOT="/data/stat-cadd/wolf7055/PRISM"

RESUME_FROM_CHECKPOINT="${PROJECT_ROOT}/Log_Results/serious/final_geometry_checks_CD/checkpoints/BRD4_BD1/seed=976/epoch=43-reward=0.76.ckpt"
SOURCE_DATASET_ROOT="${PROJECT_ROOT}/data/BRD4_BD1"
HOTSPOT_PATH="${PROJECT_ROOT}/data/BRD4_BD1/hotspot_analysis/BRD4_BD1_hotspot_data.pkl"
CHECKPOINT_OUTPUT_DIR="${PROJECT_ROOT}/Log_Results/case_studies/ignore"
DATASET_NAME=$(basename "$SOURCE_DATASET_ROOT")

echo "=========================================="
echo "ROUND 2 TRAINING - Terminal Test Run"
echo "=========================================="
echo "Dataset:    $DATASET_NAME"
echo "Checkpoint: $RESUME_FROM_CHECKPOINT"
echo "Hotspot:    $HOTSPOT_PATH"
echo "Output dir: $CHECKPOINT_OUTPUT_DIR"
echo "=========================================="

cd "$PROJECT_ROOT"

# Validate files exist
if [ ! -f "${RESUME_FROM_CHECKPOINT}" ]; then
    echo "[ERROR] Checkpoint not found: ${RESUME_FROM_CHECKPOINT}"
    exit 1
fi

if [ ! -f "${HOTSPOT_PATH}" ]; then
    echo "[ERROR] Hotspot PKL not found: ${HOTSPOT_PATH}"
    exit 1
fi

# --- Python/GPU check ---
which python
python - << 'PY'
import torch, os
print("CVD", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("avail", torch.cuda.is_available(), "count", torch.cuda.device_count())
PY
nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_PPO=0

# --- Run training ---
echo "Starting training..."
python "${PROJECT_ROOT}/scripts/train.py" \
    --config "${PROJECT_ROOT}/configs/exp_specific/case_study_dock.yaml" \
    --resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}" \
    --datadir "${SOURCE_DATASET_ROOT}/03_final_dataset" \
    --logdir "$CHECKPOINT_OUTPUT_DIR" \
    --dataset_name "$DATASET_NAME" \
    --hotspot_path "$HOTSPOT_PATH"

echo "Training completed!"
nvidia-smi