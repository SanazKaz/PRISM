#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short
#SBATCH --time=24:00:00
#SBATCH --job-name=egnn_unfreeze_aromatic_crossdocked_geom_dock_customqed_sa
#SBATCH --mail-user=sanaz.kazeminia@stats.ox.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-3
#SBATCH --output=jobs_files/egnn_unfreeze_aromatic_crossdocked_geom_dock_customqed_sa_%a.log
#SBATCH --error=jobs_files/egnn_unfreeze_aromatic_crossdocked_geom_dock_customqed_sa_%a.log

module purge
module load Anaconda3
source activate /data/stat-cadd/wolf7055/conda/envs/PRISM_25


# --- 1. SETUP PATHS ---
export PROJECT_ROOT="/data/stat-cadd/wolf7055/PRISM"

SEEDS=(42 789 976 123)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

RESUME_FROM_CHECKPOINT="${PROJECT_ROOT}/Log_Results/serious/short_crossdocked_dataset_geometry_training/checkpoints/7275716/seed=42/epoch=09-reward=0.68.ckpt"

# logdir must be the ROOT log dir — run_identifier from the yaml creates the subdir
LOGDIR="${PROJECT_ROOT}/Log_Results"
DATADIR="${PROJECT_ROOT}/data/cross_dock/processed_crossdock_noH_full_temp"

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Using seed: $SEED"
echo "Resume checkpoint: $RESUME_FROM_CHECKPOINT"
echo "Log dir: $LOGDIR"
echo "Data directory: $DATADIR"
echo "=========================================="

cd $PROJECT_ROOT

if [ ! -f "${RESUME_FROM_CHECKPOINT}" ]; then
    echo "[ERROR] Resume checkpoint not found: ${RESUME_FROM_CHECKPOINT}"
    exit 1
fi

if [ ! -f "${DATADIR}/size_distribution.npy" ]; then
    echo "[ERROR] size_distribution.npy not found in: ${DATADIR}"
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
    --config "${PROJECT_ROOT}/configs/diffsbdd/ablations/egnn_unfreeze.yaml" \
    --warm_start_from_ddpm "${RESUME_FROM_CHECKPOINT}" \
    --seed $SEED \
    --datadir "$DATADIR" \
    --logdir "$LOGDIR" \
    --dataset_name "crossdock"

echo "Training completed!"

echo "Final Memory Usage:"
free -h
echo "Final GPU Usage:"
nvidia-smi
