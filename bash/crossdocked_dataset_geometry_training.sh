#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short
#SBATCH --time 03:00:00
#SBATCH --job-name=12-3-26_prod_multiobjective_CD
#SBATCH --mail-user=wolf7055@ox.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-3
#SBATCH --output=jobs_files/12-3-prod_multiobjective_CD_%a.log
#SBATCH --error=jobs_files/12-3_prod_multiobjective_CD_%a.log

module purge
module load Anaconda3
source activate /data/stat-cadd/wolf7055/conda/envs/PRISM_25


# --- 1. SETUP PATHS ---
export PROJECT_ROOT="/data/stat-cadd/wolf7055/PRISM"

SEEDS=(42 976 123 789)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

WARM_START_CKPT="${PROJECT_ROOT}/Log_Results/short_crossdocked_dataset_geometry_training/checkpoints/7275716/seed=42/last.ckpt"
CHECKPOINT_OUTPUT_DIR="${PROJECT_ROOT}/Log_Results"
DATADIR="${PROJECT_ROOT}/data/cross_dock/processed_crossdock_noH_full_temp"

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Using seed: $SEED"
echo "Warm start checkpoint: $WARM_START_CKPT"
echo "Checkpoint output: $CHECKPOINT_OUTPUT_DIR"
echo "Data directory: $DATADIR"
echo "=========================================="

cd $PROJECT_ROOT

if [ ! -f "${WARM_START_CKPT}" ]; then
    echo "[ERROR] Warm start checkpoint not found: ${WARM_START_CKPT}"
    exit 1
fi


# --- 2. PYTHON SETUP ---
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


# --- 3. RUN TRAINING ---
echo "Starting training..."
srun python "${PROJECT_ROOT}/scripts/train.py" \
    --config "${PROJECT_ROOT}/configs/ppo_config.yaml" \
    --resume_from_checkpoint "${WARM_START_CKPT}" \
    --seed $SEED \
    --datadir "$DATADIR" \
    --logdir "$CHECKPOINT_OUTPUT_DIR"

echo "Training completed!"

echo "Final Memory Usage:"
free -h
echo "Final GPU Usage:"
nvidia-smi