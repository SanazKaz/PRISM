#!/bin/bash
#SBATCH --job-name=egnn_unfreeze_aromatic_crossdocked_geom_dock_customqed_sa
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --output=jobs_files/egnn_unfreeze_aromatic_crossdocked_geom_dock_customqed_sa_%a.log
#SBATCH --error=jobs_files/egnn_unfreeze_aromatic_crossdocked_geom_dock_customqed_sa_%a.log
#SBATCH --mail-user=sanaz.kazeminia@stats.ox.ac.uk
#SBATCH --mail-type=END,FAIL

module purge
module load Anaconda3
source activate /data/stat-cadd/wolf7055/conda/envs/PRISM_25

export PROJECT_ROOT="/data/stat-cadd/wolf7055/PRISM"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_PPO=0

SEEDS=(42 789 976 123)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

RESUME_FROM_CHECKPOINT="${PROJECT_ROOT}/Log_Results/serious/short_crossdocked_dataset_geometry_training/checkpoints/7275716/seed=42/epoch=09-reward=0.68.ckpt"
LOGDIR="${PROJECT_ROOT}/Log_Results"
DATADIR="${PROJECT_ROOT}/data/cross_dock/processed_crossdock_noH_full_temp"

echo "Job: $SLURM_JOB_ID | Array task: $SLURM_ARRAY_TASK_ID | Seed: $SEED"

if [ ! -f "${RESUME_FROM_CHECKPOINT}" ]; then
    echo "[ERROR] Resume checkpoint not found: ${RESUME_FROM_CHECKPOINT}"
    exit 1
fi

if [ ! -f "${DATADIR}/size_distribution.npy" ]; then
    echo "[ERROR] size_distribution.npy not found in: ${DATADIR}"
    exit 1
fi

cd "$PROJECT_ROOT"
nvidia-smi

echo "Starting training..."
srun python scripts/train.py \
    --config configs/diffsbdd/ablations/egnn_unfreeze.yaml \
    --warm_start_from_ddpm "${RESUME_FROM_CHECKPOINT}" \
    --seed "${SEED}" \
    --datadir "${DATADIR}" \
    --logdir "${LOGDIR}" \
    --dataset_name crossdock

echo "Done."