#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short
#SBATCH --time 06:00:00
#SBATCH --job-name=single_objective_QED_CD
#SBATCH --mail-user=wolf7055@ox.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-23
#SBATCH --output=/dev/null 
#SBATCH --error=/dev/null   

module purge
module load Anaconda3
source activate /data/stat-cadd/wolf7055/conda/envs/PRISM_25


# --- 1. SETUP PATHS ---
export PROJECT_ROOT="/data/stat-cadd/wolf7055/PRISM"
SEEDS=(42 976 123 789)

# Single warm-start model for all datasets
WARM_START_CKPT="${PROJECT_ROOT}/checkpoints/crossdocked_fa_cond_temp.ckpt"

# Where checkpoints should be saved
CHECKPOINT_OUTPUT_DIR="${PROJECT_ROOT}/Log_Results"

# Datasets to train on (pointing to parent dirs now)
DATASETS=(
    "${PROJECT_ROOT}/data/BRD4_BD1"
    "${PROJECT_ROOT}/data/Factor_Xa"
    "${PROJECT_ROOT}/data/Carb_Anh_II"
    "${PROJECT_ROOT}/data/EGFR"
    "${PROJECT_ROOT}/data/Estrogen_recep_alpha"
    "${PROJECT_ROOT}/data/HIV_1_Protease"
)

NUM_SEEDS=${#SEEDS[@]}
NUM_DATASETS=${#DATASETS[@]}

# Calculate indices
SEED_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))
DATASET_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))

SEED=${SEEDS[$SEED_IDX]}
SOURCE_DATASET_ROOT=${DATASETS[$DATASET_IDX]}

# Extract dataset name
DATASET_NAME=$(basename $SOURCE_DATASET_ROOT)

# --- 2. LOGGING SETUP (for SLURM logs only) ---
JOB_NAME="single_objective_QED_CD"
SLURM_LOG_DIR="${PROJECT_ROOT}/single_objective_QED_CD/${JOB_NAME}/${DATASET_NAME}"
mkdir -p $SLURM_LOG_DIR 

LOG_FILE="${SLURM_LOG_DIR}/seed_${SEED}_taskid_${SLURM_ARRAY_TASK_ID}.log"

# Redirect all output to this log file
exec 1>$LOG_FILE 2>&1

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Using seed: $SEED"
echo "Dataset: $DATASET_NAME"
echo "Source Root: $SOURCE_DATASET_ROOT"
echo "Warm start checkpoint: $WARM_START_CKPT"
echo "Log file: $LOG_FILE"
echo "Checkpoint output: $CHECKPOINT_OUTPUT_DIR"
echo "=========================================="

cd $PROJECT_ROOT

# Validate warm start checkpoint exists
if [ ! -f "${WARM_START_CKPT}" ]; then
    echo "[ERROR] Warm start checkpoint not found: ${WARM_START_CKPT}"
    exit 1
fi

# --- 3. STAGE IN (Copy ENTIRE dataset directory to Scratch) ---
SCRATCH_WORK_DIR="${TMPDIR}/${SLURM_JOB_ID}/${DATASET_NAME}"

export DEBUG_PPO=0

echo "Starting Data Stage-in..."
echo "Copying from: $SOURCE_DATASET_ROOT"
echo "Copying to:   $SCRATCH_WORK_DIR"
start_time=$(date +%s)

mkdir -p "$SCRATCH_WORK_DIR"
# Copy the entire dataset directory (including 02_preprocessed and 03_final_dataset)
rsync -ah "$SOURCE_DATASET_ROOT/" "$SCRATCH_WORK_DIR/"

end_time=$(date +%s)
echo "Data staged in $((end_time - start_time)) seconds."

# Verify the structure was copied correctly
echo "Verifying copied structure:"
ls -lh "$SCRATCH_WORK_DIR/"

# --- 4. PYTHON SETUP ---
which python

python - << 'PY'
import torch, os
print("CVD", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("avail", torch.cuda.is_available(), "count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("name0", torch.cuda.get_device_name(0))
PY

nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_PPO=0

# --- 5. RUN TRAINING ---
echo "Starting training..."
# Now pass the 03_final_dataset subdirectory from scratch
srun python "${PROJECT_ROOT}/scripts/train.py" \
    --config "${PROJECT_ROOT}/configs/exp_specific/single_obj_qed.yaml" \
    --warm_start_from_ddpm "${WARM_START_CKPT}" \
    --seed $SEED \
    --datadir "${SCRATCH_WORK_DIR}/03_final_dataset" \
    --logdir "$CHECKPOINT_OUTPUT_DIR" \
    --dataset_name "$DATASET_NAME"

echo "Training completed!"

# --- 6. CLEANUP ---
echo "Cleaning up scratch..."
rm -rf "${TMPDIR}/${SLURM_JOB_ID}"

echo "Final Memory Usage:"
free -h
echo "Final GPU Usage:"
nvidia-smi