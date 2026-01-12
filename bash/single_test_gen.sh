#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --partition=short 
#SBATCH --gres=gpu:h100:1
#SBATCH --time 03:00:00
#SBATCH --job-name=MOLS_linear_squared_feature_density_300_timesteps_lr_1e-5_cont_5ang
#SBATCH --mail-user=wolf7055@ox.ac.uk
#SBATCH --output=jobs_files/MOLS_linear_squared_feature_density_300_timesteps_lr_1e-5_cont_5ang%j.log
exec 2>&1

module purge
module load Anaconda3
source activate /data/stat-cadd/wolf7055/conda/envs/TEST_ENV

echo "Python executable: $(which python)"
chmod +x /data/stat-cadd/wolf7055/PRISM/val_analysis/smina.static

# --- CONFIGURATION ---
PRISM_ROOT="/data/stat-cadd/wolf7055/PRISM"

# Model checkpoint - UPDATE THIS
MODEL_PATH="${PRISM_ROOT}/Log_Results/linear_squared_feature_density_300_timesteps_lr_1e-5_cont_5ang/checkpoints/tmp/seed=976/epoch=141-reward=0.36.pt"

# Target to test
TARGET="AMPC_beta_lactamase"

# Output directory
OUTDIR="${PRISM_ROOT}/Generated_Mols/linear_squared_feature_density_300_timesteps_lr_1e-5_cont_5ang"

# Generation settings
N_SAMPLES=10000
BATCH_SIZE=75

# --- DIAGNOSTICS ---
echo "=========================================="
echo "PRISM Single Model Test"
echo "Started at: $(date)"
echo "Node: $(hostname)"
echo "=========================================="
echo ""
echo "Model: ${MODEL_PATH}"
echo "Target: ${TARGET}"
echo "Output: ${OUTDIR}"
echo "N samples: ${N_SAMPLES}"
echo ""

echo "Initial GPU Usage:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# --- RUN ---
cd "${PRISM_ROOT}/src/models/diffsbdd"
export PYTHONPATH="${PRISM_ROOT}/src/models/diffsbdd:${PYTHONPATH}"

python "${PRISM_ROOT}/scripts/test.py" "${MODEL_PATH}" \
    --config "${PRISM_ROOT}/configs/ppo_config.yaml" \
    --outdir "${OUTDIR}" \
    --target "${TARGET}" \
    --n_samples ${N_SAMPLES} \
    --batch_size ${BATCH_SIZE} \
    --sanitize

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Completed: ${TARGET}"
echo "Exit code: ${EXIT_CODE}"
echo "Finished at: $(date)"
echo "=========================================="

echo "Final GPU Usage:"
nvidia-smi

exit ${EXIT_CODE}