#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short
#SBATCH --time 04:00:00
#SBATCH --job-name=PB_Final_Run_0.5_0.5_to_0.7_0.3
#SBATCH --mail-user=wolf7055@ox.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-23
#SBATCH --output=/dev/null 
#SBATCH --error=/dev/null   

module purge
module load Anaconda3
source activate /data/stat-cadd/wolf7055/conda/envs/prism_backup

# --- 1. SETUP PATHS ---
SEEDS=(42 976 123 789)
DATASETS=(
    "/data/stat-cadd/wolf7055/PRISM/data/AMPC_beta_lactamase/03_final_dataset"
    "/data/stat-cadd/wolf7055/PRISM/data/Carb_Anh_II/03_final_dataset"
    "/data/stat-cadd/wolf7055/PRISM/data/covid19_main_protease/03_final_dataset"
    "/data/stat-cadd/wolf7055/PRISM/data/EGFR/03_final_dataset"
    "/data/stat-cadd/wolf7055/PRISM/data/Estrogen_recep_alpha/03_final_dataset"
    "/data/stat-cadd/wolf7055/PRISM/data/HIV_1_Protease/03_final_dataset"
)

NUM_SEEDS=${#SEEDS[@]}
NUM_DATASETS=${#DATASETS[@]}

# Calculate indices
SEED_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))
DATASET_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))

SEED=${SEEDS[$SEED_IDX]}
# This is the PERMANENT source path
SOURCE_DATASET_PATH=${DATASETS[$DATASET_IDX]}

# Extract dataset name
DATASET_NAME=$(basename $(dirname $SOURCE_DATASET_PATH))

# --- 2. LOGGING SETUP (Permanent Storage) ---
# Create organized log directory structure
JOB_NAME="PB_Geo_Flat_1.0_cont_0.7_0.3_epoch_30_45"
# Ensure this path is on /data or /home, NOT scratch
BASE_LOG_DIR="/data/stat-cadd/wolf7055/PRISM/jobs_files" 
LOG_DIR="${BASE_LOG_DIR}/${JOB_NAME}/${DATASET_NAME}"
mkdir -p $LOG_DIR 

# Create log file with descriptive name
LOG_FILE="${LOG_DIR}/seed_${SEED}_taskid_${SLURM_ARRAY_TASK_ID}.log"

# Redirect all output to this log file
exec 1>$LOG_FILE 2>&1

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Using seed: $SEED"
echo "Dataset: $DATASET_NAME"
echo "Source Path: $SOURCE_DATASET_PATH"
echo "Log file: $LOG_FILE"
echo "=========================================="

# --- 3. STAGE IN (Copy to Scratch) ---
# Define the fast local scratch directory
# Using a specific subfolder prevents file collisions
SCRATCH_WORK_DIR="${TMPDIR}/${SLURM_JOB_ID}/${DATASET_NAME}"

echo "Starting Data Stage-in..."
echo "Copying from: $SOURCE_DATASET_PATH"
echo "Copying to:   $SCRATCH_WORK_DIR"
start_time=$(date +%s)

mkdir -p "$SCRATCH_WORK_DIR"
# Copy data
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
# CRITICAL: 
# --datadir points to SCRATCH (Read Fast)
# --default_root_dir points to PERMANENT LOG_DIR (Write Safe)

echo "Starting training..."
srun python scripts/train.py \
--config "configs/ppo_config.yaml" \
--warm_start_from_ddpm "/data/stat-cadd/wolf7055/PRISM/checkpoints/crossdocked_fa_cond_temp.ckpt" \
--seed $SEED \
--datadir "$SCRATCH_WORK_DIR" \
--logdir "$BASE_LOG_DIR" \
--dataset_name "$DATASET_NAME" \

echo "Training completed!"

# --- 6. CLEANUP ---
# Remove the scratch data to be polite (even if system does it later)
echo "Cleaning up scratch..."
rm -rf "${TMPDIR}/${SLURM_JOB_ID}"

echo "Final Memory Usage:"
free -h
echo "Final GPU Usage:"
nvidia-smi