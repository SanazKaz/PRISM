#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --partition=short
#SBATCH --gres=gpu:h100:1
#SBATCH --time 12:00:00
#SBATCH --output=jobs_files/DiffSBDD/crossdocked_fa_cond_temp_%x_%j.log
exec 2>&1

# =============================================================================
# PRISM Evaluation: Single target job (for parallel submission)
# Usage: sbatch --job-name=TARGET_NAME run_single_target.sh TARGET_NAME
# =============================================================================

# Get target from command line argument
TARGET=$1

if [ -z "${TARGET}" ]; then
    echo "[ERROR] No target specified!"
    echo "Usage: sbatch run_single_target.sh TARGET_NAME"
    exit 1
fi

module purge
module load Anaconda3
source activate /data/stat-cadd/wolf7055/conda/envs/TEST_ENV

echo "============================================="
echo "PRISM Evaluation: ${TARGET}"
echo "Started at: $(date)"
echo "Node: $(hostname)"
echo "============================================="

# -----------------------------------------------------------------------------
# CONFIGURATION - Edit these paths
# -----------------------------------------------------------------------------

PRISM_ROOT="/data/stat-cadd/wolf7055/PRISM"
SCRIPT_PATH="${PRISM_ROOT}/scripts/test.py"
CONFIG_PATH="${PRISM_ROOT}/configs/ppo_config.yaml"

# Model checkpoint - UPDATE THIS
MODEL_PATH="${PRISM_ROOT}/checkpoints/crossdocked_fa_cond_temp.ckpt"

# Output directory
RUN_NAME="crossdocked_fa_cond_temp_10k_Evaluation"
OUTDIR="${PRISM_ROOT}/Generated_Mols/${RUN_NAME}"

# Generation settings
N_SAMPLES=10000
BATCH_SIZE=75

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------

echo ""
echo "Model: ${MODEL_PATH}"
echo "Target: ${TARGET}"
echo "Output: ${OUTDIR}/${TARGET}"
echo ""

nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

cd "${PRISM_ROOT}/src/models/diffsbdd"

export PYTHONPATH="${PRISM_ROOT}/src/models/diffsbdd:${PYTHONPATH}" 


python ${SCRIPT_PATH} "${MODEL_PATH}" \
    --config "${CONFIG_PATH}" \
    --outdir "${OUTDIR}" \
    --target "${TARGET}" \
    --n_samples ${N_SAMPLES} \
    --batch_size ${BATCH_SIZE} \
    --sanitize \
    --skip_existing

EXIT_CODE=$?

echo ""
echo "============================================="
echo "Completed: ${TARGET}"
echo "Exit code: ${EXIT_CODE}"
echo "Finished at: $(date)"
echo "============================================="

exit ${EXIT_CODE}