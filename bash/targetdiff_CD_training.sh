#!/bin/bash
#SBATCH --job-name=20_05_2026_4gpu_64_batch_700T_TargetDiff_CD_plogp
#SBATCH --partition=short
#SBATCH --qos=ecr
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:4
#SBATCH --array=0-1
#SBATCH --output=jobs_files/targetdiff/%x_%a.log
#SBATCH --error=jobs_files/targetdiff/%x_%a.log
#SBATCH --mail-user=wolf7055@ox.ac.uk
#SBATCH --mail-type=END,FAIL

module purge
module load Anaconda3
source activate /data/stat-cadd/wolf7055/conda/envs/PRISM_25

export PROJECT_ROOT="/data/stat-cadd/wolf7055/PRISM"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_PPO=0

SEEDS=(42 976)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

WARM_START_CKPT="${PROJECT_ROOT}/checkpoints/targetdiff_pretrained_models/targetdiff_pretrained_diffusion.pt"
DATADIR="${PROJECT_ROOT}/data/cross_dock/processed_crossdock_targetdiff"
CONFIG="${PROJECT_ROOT}/configs/targetdiff/crossdocked/p_logp_4gpu.yaml"
LOGDIR="${PROJECT_ROOT}/Log_Results/targetdiff_penlogp_CD"

echo "Job: $SLURM_JOB_ID | Array task: $SLURM_ARRAY_TASK_ID | Seed: $SEED"

if [ ! -f "${WARM_START_CKPT}" ]; then
    echo "[ERROR] Checkpoint not found: ${WARM_START_CKPT}"
    exit 1
fi

cd "$PROJECT_ROOT"
nvidia-smi
# record the time
start_time=$(date +%s)
echo "Start time: $(date +%T)"

echo "Starting training..."
srun python scripts/train.py \
    --config "${CONFIG}" \
    --warm_start_from_ddpm "${WARM_START_CKPT}" \
    --seed "${SEED}" \
    --datadir "${DATADIR}" \
    --logdir "${LOGDIR}"

echo "Done."
end_time=$(date +%s)
echo "End time: $(date +%T)"

# calculate the time taken
time_taken=$((end_time - start_time))
echo "Time taken: ${time_taken} seconds"