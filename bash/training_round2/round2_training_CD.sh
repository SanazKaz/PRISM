#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short
#SBATCH --time 12:00:00
#SBATCH --job-name=feat_2dprop_geom_cd_curriculum
#SBATCH --mail-user=wolf7055@ox.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-23
#SBATCH --output=/dev/null 
#SBATCH --error=/dev/null   

module purge
module load Anaconda3
source activate /data/stat-cadd/wolf7055/conda/envs/PRISM_25

# =============================================================================
# 1. PATH DEFINITIONS (INPUTS & OUTPUTS)
# =============================================================================
export PROJECT_ROOT="/data/stat-cadd/wolf7055/PRISM"

# --- INPUT PATHS: Checkpoints & Data ---
ROUND1_CKPT_BASE="${PROJECT_ROOT}/Log_Results/final_geometry_checks_CD/checkpoints"

RESUME_FROM_CHECKPOINTS=(
    "${ROUND1_CKPT_BASE}/BRD4_BD1/seed=976/epoch=43-reward=0.76.ckpt"
    "${ROUND1_CKPT_BASE}/Factor_Xa/seed=976/epoch=40-reward=0.76.ckpt"
    "${ROUND1_CKPT_BASE}/Carb_Anh_II/seed=976/epoch=44-reward=0.78.ckpt"
    "${ROUND1_CKPT_BASE}/EGFR/seed=123/epoch=42-reward=0.80.ckpt"
    "${ROUND1_CKPT_BASE}/Estrogen_recep_alpha/seed=123/epoch=30-reward=0.90.ckpt"
    "${ROUND1_CKPT_BASE}/HIV_1_Protease/seed=976/epoch=44-reward=0.84.ckpt"
)

DATASETS=(
    "${PROJECT_ROOT}/data/BRD4_BD1"
    "${PROJECT_ROOT}/data/Factor_Xa"
    "${PROJECT_ROOT}/data/Carb_Anh_II"
    "${PROJECT_ROOT}/data/EGFR"
    "${PROJECT_ROOT}/data/Estrogen_recep_alpha"
    "${PROJECT_ROOT}/data/HIV_1_Protease"
)

HOTSPOT_PKLS=(
    "${PROJECT_ROOT}/data/BRD4_BD1/hotspot_analysis/BRD4_BD1_hotspot_data.pkl"
    "${PROJECT_ROOT}/data/Factor_Xa/hotspot_analysis/Factor_Xa_hotspot_data.pkl"
    "${PROJECT_ROOT}/data/Carb_Anh_II/hotspot_analysis/Carb_Anh_II_hotspot_data.pkl"
    "${PROJECT_ROOT}/data/EGFR/hotspot_analysis/EGFR_hotspot_data.pkl"
    "${PROJECT_ROOT}/data/Estrogen_recep_alpha/hotspot_analysis/Estrogen_recep_alpha_hotspot_data.pkl"
    "${PROJECT_ROOT}/data/HIV_1_Protease/hotspot_analysis/HIV_1_Protease_hotspot_data.pkl"
)

# --- OUTPUT PATHS: Logs & Checkpoints ---
CHECKPOINT_OUTPUT_DIR="${PROJECT_ROOT}/Log_Results/feat_2dprop_geom_cd_curriculum"
JOB_NAME="feat_2dprop_geom_cd_curriculum"
SLURM_LOG_BASE_DIR="${PROJECT_ROOT}/jobs_files/feat_2dprop_geom_cd_curriculum/${JOB_NAME}"

# =============================================================================
# 2. TASK INDEXING & LOGGING SETUP
# =============================================================================
SEEDS=(42 976 123 789)
NUM_SEEDS=${#SEEDS[@]}
NUM_DATASETS=${#DATASETS[@]}

# Calculate indices
SEED_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))
DATASET_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))

# Assign indexed variables
SEED=${SEEDS[$SEED_IDX]}
SOURCE_DATASET_ROOT=${DATASETS[$DATASET_IDX]}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINTS[$DATASET_IDX]}
HOTSPOT_PATH=${HOTSPOT_PKLS[$DATASET_IDX]}
DATASET_NAME=$(basename "$SOURCE_DATASET_ROOT")

# Setup Logging
SLURM_LOG_DIR="${SLURM_LOG_BASE_DIR}/${DATASET_NAME}"
mkdir -p "$SLURM_LOG_DIR" 
LOG_FILE="${SLURM_LOG_DIR}/seed_${SEED}_taskid_${SLURM_ARRAY_TASK_ID}.log"

# Redirect all output to log file 
exec 1>"$LOG_FILE" 2>&1

echo "=========================================="
echo "ROUND 2 TRAINING - Sillwalks CD"
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Using seed: $SEED"
echo "Dataset: $DATASET_NAME"
echo "Hotspot Path: $HOTSPOT_PATH"
echo "Round 1 checkpoint: $RESUME_FROM_CHECKPOINT"
echo "Log file: $LOG_FILE"
echo "Checkpoint output: $CHECKPOINT_OUTPUT_DIR"
echo "=========================================="

cd "$PROJECT_ROOT"

# Validate files exist
if [ ! -f "${RESUME_FROM_CHECKPOINT}" ]; then
    echo "[ERROR] Round 1 checkpoint not found: ${RESUME_FROM_CHECKPOINT}"
    exit 1
fi

if [ ! -f "${HOTSPOT_PATH}" ]; then
    echo "[ERROR] Hotspot PKL not found: ${HOTSPOT_PATH}"
    exit 1
fi

# --- 3. STAGE IN (Copy ENTIRE dataset directory to Scratch) ---
SCRATCH_WORK_DIR="${TMPDIR}/${SLURM_JOB_ID}/${DATASET_NAME}"
export DEBUG_PPO=0

echo "Starting Data Stage-in..."
mkdir -p "$SCRATCH_WORK_DIR"
start_time=$(date +%s)
rsync -ah "$SOURCE_DATASET_ROOT/" "$SCRATCH_WORK_DIR/"
end_time=$(date +%s)
echo "Data staged in $((end_time - start_time)) seconds."

# --- 4. PYTHON SETUP ---
which python
python - << 'PY'
import torch, os
print("CVD", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("avail", torch.cuda.is_available(), "count", torch.cuda.device_count())
PY
nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_PPO=0

# --- 5. RUN TRAINING ---
echo "Starting Round 2 training..."
srun python "${PROJECT_ROOT}/scripts/train.py" \
    --config "${PROJECT_ROOT}/configs/ppo_config.yaml" \
    --resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}" \
    --seed "$SEED" \
    --datadir "${SCRATCH_WORK_DIR}/03_final_dataset" \
    --logdir "$CHECKPOINT_OUTPUT_DIR" \
    --dataset_name "$DATASET_NAME" \
    --hotspot_path "$HOTSPOT_PATH"

echo "Training completed!"

# --- 6. CLEANUP ---
echo "Cleaning up scratch..."
rm -rf "${TMPDIR}/${SLURM_JOB_ID}"
nvidia-smi