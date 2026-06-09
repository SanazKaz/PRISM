#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=short
#SBATCH --time=03:00:00
#SBATCH --job-name=TESTING_ligand_efficiency
#SBATCH --output=jobs_files/TESTING_ligand_efficiency.log

# --- Environment Setup ---
module purge
module load Anaconda3
source activate /data/stat-cadd/wolf7055/conda/envs/PRISM_25

echo "Python executable: $(which python)"

export DEBUG_PPO=0

# Seeds array
SEEDS=(42 976 123 789)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Checkpoint path (hardcoded)
CHECKPOINT_PATH="/data/stat-cadd/wolf7055/PRISM/Log_Results/axis_2_sillwalks_CD/axis_2_sillwalks_CD/checkpoints/BRD4_BD1/seed=42/epoch=58-reward=0.49.ckpt"

echo "=========================================="
echo "Project Root: /data/stat-cadd/wolf7055/PRISM"
echo "Data Dir:     /data/stat-cadd/wolf7055/PRISM/data/BRD4_BD1/03_final_dataset"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Using Seed: ${SEED}"
echo "Resuming from: ${CHECKPOINT_PATH}"
echo "=========================================="

cd /data/stat-cadd/wolf7055/PRISM

# --- Permissions ---
chmod +x /data/stat-cadd/wolf7055/PRISM/val_analysis/smina.static

# --- Diagnostics ---
echo "Initial GPU Usage:"
nvidia-smi

# --- RUN TRAINING ---
echo "Starting PRISM training..."

srun python /data/stat-cadd/wolf7055/PRISM/scripts/train.py \
    --config /data/stat-cadd/wolf7055/PRISM/configs/ppo_config.yaml \
    --resume_from_checkpoint "${CHECKPOINT_PATH}" \
    --datadir /data/stat-cadd/wolf7055/PRISM/data/BRD4_BD1/03_final_dataset \
    --seed ${SEED}

echo "Training completed."

echo "Final GPU Usage:"
nvidia-smi
