#!/bin/bash
#SBATCH --job-name=targetdiff_penlogp_CD
#SBATCH --partition=short
#SBATCH --qos=ecr
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:2
#SBATCH --array=0-0
#SBATCH --output=jobs_files/%x_%a.log
#SBATCH --error=jobs_files/%x_%a.log
#SBATCH --mail-user=wolf7055@ox.ac.uk
#SBATCH --mail-type=END,FAIL

module purge
module load Anaconda3
source activate /data/stat-cadd/wolf7055/conda/envs/PRISM_25

export PROJECT_ROOT="/data/stat-cadd/wolf7055/PRISM"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_PPO=0

SEEDS=(42 976 123 789)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

WARM_START_CKPT="${PROJECT_ROOT}/checkpoints/targetdiff_pretrained_diffusion.pt"
DATADIR="${PROJECT_ROOT}/data/cross_dock/processed_crossdock_targetdiff"
CONFIG="${PROJECT_ROOT}/configs/targetdiff/crossdocked/p_logp.yaml"
LOGDIR="${PROJECT_ROOT}/Log_Results/targetdiff_penlogp_CD"

echo "Job: $SLURM_JOB_ID | Array task: $SLURM_ARRAY_TASK_ID | Seed: $SEED"

if [ ! -f "${WARM_START_CKPT}" ]; then
    echo "[ERROR] Checkpoint not found: ${WARM_START_CKPT}"
    exit 1
fi

cd "$PROJECT_ROOT"

echo "Starting training..."
srun python scripts/train.py \
    --config "${CONFIG}" \
    --warm_start_from_ddpm "${WARM_START_CKPT}" \
    --seed "${SEED}" \
    --datadir "${DATADIR}" \
    --logdir "${LOGDIR}"

echo "Done."
