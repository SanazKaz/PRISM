#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short
#SBATCH --time 08:00:00
#SBATCH --job-name=silly_walks_Aro_bonus_Posebusters_PRISM
#SBATCH --mail-user=wolf7055@ox.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-19
#SBATCH --output=/dev/null 
#SBATCH --error=/dev/null   

module purge
module load Anaconda3
source activate /data/stat-cadd/wolf7055/conda/envs/prism_backup

# --- 1. SETUP PATHS ---
SEEDS=(42 976 123 789)

# Base path for trained model checkpoints
CKPT_BASE="/data/stat-cadd/wolf7055/PRISM/Log_Results/PB_Final_Run_0.5_0.5_to_0.7_0.3/checkpoints"

# Parallel arrays: datasets and their corresponding best model checkpoints
DATASETS=(
    "/data/stat-cadd/wolf7055/PRISM/data/AMPC_beta_lactamase/03_final_dataset"
    "/data/stat-cadd/wolf7055/PRISM/data/Carb_Anh_II/03_final_dataset"
    "/data/stat-cadd/wolf7055/PRISM/data/covid19_main_protease/03_final_dataset"
    "/data/stat-cadd/wolf7055/PRISM/data/EGFR/03_final_dataset"
    "/data/stat-cadd/wolf7055/PRISM/data/Estrogen_recep_alpha/03_final_dataset"
)

# Model paths with .ckpt extension (same structure as .pt but different extension)
MODEL_CKPTS=(
    "${CKPT_BASE}/AMPC_beta_lactamase/seed=42/epoch=33-reward=1.28.ckpt"
    "${CKPT_BASE}/Carb_Anh_II/seed=42/epoch=33-reward=1.23.ckpt"
    "${CKPT_BASE}/covid19_main_protease/seed=123/epoch=31-reward=1.18.ckpt"
    "${CKPT_BASE}/EGFR/seed=42/epoch=34-reward=1.20.ckpt"
    "${CKPT_BASE}/Estrogen_recep_alpha/seed=789/epoch=34-reward=1.23.ckpt"
)

NUM_SEEDS=${#SEEDS[@]}
NUM_DATASETS=${#DATASETS[@]}

# Calculate indices
SEED_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))
DATASET_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))

SEED=${SEEDS[$SEED_IDX]}
SOURCE_DATASET_PATH=${DATASETS[$DATASET_IDX]}
MODEL_CKPT=${MODEL_CKPTS[$DATASET_IDX]}

# Extract dataset name
DATASET_NAME=$(basename $(dirname $SOURCE_DATASET_PATH))

# --- 2. LOGGING SETUP (Permanent Storage) ---
JOB_NAME="sw_aro_b_pb_prism"
BASE_LOG_DIR="/data/stat-cadd/wolf7055/PRISM/jobs_files/silly_walks_Aro_bonus_Posebusters_PRISM" 
LOG_DIR="${BASE_LOG_DIR}/${JOB_NAME}/${DATASET_NAME}"
mkdir -p $LOG_DIR 

LOG_FILE="${LOG_DIR}/seed_${SEED}_taskid_${SLURM_ARRAY_TASK_ID}.log"

# Redirect all output to this log file
exec 1>$LOG_FILE 2>&1

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Using seed: $SEED"
echo "Dataset: $DATASET_NAME"
echo "Source Path: $SOURCE_DATASET_PATH"
echo "Model checkpoint: $MODEL_CKPT"
echo "Log file: $LOG_FILE"
echo "=========================================="

# Validate model checkpoint exists
if [ ! -f "${MODEL_CKPT}" ]; then
    echo "[ERROR] Model checkpoint not found: ${MODEL_CKPT}"
    exit 1
fi

# --- 3. STAGE IN (Copy to Scratch) ---
SCRATCH_WORK_DIR="${TMPDIR}/${SLURM_JOB_ID}/${DATASET_NAME}"

echo "Starting Data Stage-in..."
echo "Copying from: $SOURCE_DATASET_PATH"
echo "Copying to:   $SCRATCH_WORK_DIR"
start_time=$(date +%s)

mkdir -p "$SCRATCH_WORK_DIR"
rsync -ah "$SOURCE_DATASET_PATH/" "$SCRATCH_WORK_DIR/"

end_time=$(date +%s)
echo "Data staged in $((end_time - start_time)) seconds."

# --- 4. PYTHON SETUP ---
which python

python - << 'PY'
import torch, os
print("CVD", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("avail", torch.cuda.is_available(), "count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("name0", torch.cuda.get_device_name(0))
PY

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_PPO=0

# --- 5. RUN TRAINING ---
echo "Starting training..."
srun python scripts/train.py \
--config "configs/ppo_config.yaml" \
--resume_from_checkpoint "${MODEL_CKPT}" \
--seed $SEED \
--datadir "$SCRATCH_WORK_DIR" \
--logdir "$BASE_LOG_DIR" \
--dataset_name "$DATASET_NAME"

echo "Training completed!"

# --- 6. CLEANUP ---
echo "Cleaning up scratch..."
rm -rf "${TMPDIR}/${SLURM_JOB_ID}"

echo "Final Memory Usage:"
free -h
echo "Final GPU Usage:"
nvidia-smi