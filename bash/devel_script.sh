#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --partition=devel
#SBATCH --time 00:10:00 # Increased time slightly for testing
#SBATCH --job-name=PRISM_test
#SBATCH --output=jobs_files/PRISM_test.log # Add job ID to log name

# --- Environment Setup ---
module purge
module load Anaconda3
source activate /data/stat-cadd/wolf7055/conda/envs/TEST_ENV
echo "Python executable: $(which python)"

# --- BEST PRACTICE: Define Project Root ---
# This makes all your paths robust and easy to change later.
export PROJECT_ROOT="/data/stat-cadd/wolf7055/PRISM"
echo "Project Root: ${PROJECT_ROOT}"

# Change to the project directory to ensure relative paths work
cd $PROJECT_ROOT

# --- Permissions (if needed) ---
# Assuming smina.static is in the analysis folder at the root of PRISM
# chmod +x analysis/smina.static 

# --- Logging and Diagnostics ---
echo "Initial GPU Usage:"
nvidia-smi

# --- Run the NEW Training Script ---
echo "Starting PRISM training..."
srun python "${PROJECT_ROOT}/scripts/train.py" \
    --config "${PROJECT_ROOT}/configs/ppo_config.yaml" \
    --warm_start_from_ddpm "/data/stat-cadd/wolf7055/diffsbdd-ppo/checkpoints/crossdocked_fa_cond_temp.ckpt"

# --- Final Diagnostics ---
echo "Training completed."
echo "Final GPU Usage:"
nvidia-smi