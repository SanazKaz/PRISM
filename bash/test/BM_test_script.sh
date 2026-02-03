#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --partition=short
#SBATCH --gres=gpu:h100:1
#SBATCH --time 05:00:00
#SBATCH --output=bin_BM/BM_DiffSBDD/BM_DiffSBDD_baseline_%x_%j.log
exec 2>&1

# =============================================================================
# DiffSBDD Baseline Evaluation: Single target job (for parallel submission)
# 
# Arguments:
#   $1 - Target name (e.g., BRD4_BD1_4whw)
#   $2 - Path to model checkpoint
#
# Usage: sbatch test_file.sh TARGET_NAME MODEL_PATH
# =============================================================================

TARGET=$1
MODEL_PATH=$2

if [ -z "${TARGET}" ]; then
    echo "[ERROR] No target specified!"
    echo "Usage: sbatch BM_test_script.sh TARGET_NAME MODEL_PATH"
    exit 1
fi

if [ -z "${MODEL_PATH}" ]; then
    echo "[ERROR] No model path specified!"
    echo "Usage: sbatch BM_test_script.sh TARGET_NAME MODEL_PATH"
    exit 1
fi

if [ ! -f "${MODEL_PATH}" ]; then
    echo "[ERROR] Model file not found: ${MODEL_PATH}"
    exit 1
fi

module purge
module load Anaconda3
source activate /data/stat-cadd/wolf7055/conda/envs/PRISM_25

echo "============================================="
echo "BM DiffSBDD Baseline Evaluation: ${TARGET}"
echo "Started at: $(date)"
echo "Node: $(hostname)"
echo "============================================="

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

PRISM_ROOT="/data/stat-cadd/wolf7055/PRISM"
SCRIPT_PATH="${PRISM_ROOT}/scripts/test.py"
CONFIG_PATH="${PRISM_ROOT}/configs/binding_moad_fa_ppo.yaml"

# Output directory for DiffSBDD baseline
OUTDIR="/data/stat-cadd/wolf7055/PRISM/bin_BM/BM_DiffSBDD"

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