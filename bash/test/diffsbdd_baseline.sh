#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --partition=short
#SBATCH --gres=gpu:h100:1
#SBATCH --time 00:30:00
#SBATCH --output=jobs_files/diffsbdd_baseline_targets/%x_%j.log
exec 2>&1

# =============================================================================
# diffsbdd_baseline.sh
#
# Generates ligands for a single target using the DiffSBDD baseline model.
#
# Arguments:
#   $1 - Target name (e.g., BRD4_BD1_4whw)
#   $2 - Path to model checkpoint (.ckpt file)
#
# Usage: Called by submit_diffsbdd_baseline.sh
# =============================================================================

TARGET=$1
MODEL_PATH=$2

if [ -z "${TARGET}" ]; then
    echo "[ERROR] No target specified!"
    echo "Usage: sbatch diffsbdd_baseline.sh TARGET_NAME MODEL_PATH"
    exit 1
fi

if [ -z "${MODEL_PATH}" ]; then
    echo "[ERROR] No model path specified!"
    echo "Usage: sbatch diffsbdd_baseline.sh TARGET_NAME MODEL_PATH"
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
echo "DiffSBDD Baseline Generation: ${TARGET}"
echo "Started at: $(date)"
echo "Node: $(hostname)"
echo "============================================="

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

PRISM_ROOT="/data/stat-cadd/wolf7055/PRISM"
SCRIPT_PATH="${PRISM_ROOT}/scripts/test_targets.py"
CONFIG_PATH="${PRISM_ROOT}/configs/ppo_config.yaml"

OUTDIR="${PRISM_ROOT}/generation_results/diffsbdd_baseline_targets"
mkdir -p "${OUTDIR}"

N_SAMPLES=1000
BATCH_SIZE=100

# -----------------------------------------------------------------------------
# RUN
# -----------------------------------------------------------------------------

echo ""
echo "Model:  ${MODEL_PATH}"
echo "Config: ${CONFIG_PATH}"
echo "Target: ${TARGET}"
echo "Output: ${OUTDIR}"
echo ""

nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

cd "${PRISM_ROOT}/src/models/diffsbdd"
export PYTHONPATH="${PRISM_ROOT}/src/models/diffsbdd:${PYTHONPATH}"

python "${SCRIPT_PATH}" "${MODEL_PATH}" \
    --config "${CONFIG_PATH}" \
    --outdir "${OUTDIR}" \
    --target "${TARGET}" \
    --n_samples ${N_SAMPLES} \
    --batch_size ${BATCH_SIZE} \
    --sanitize \
    --fix_n_nodes \
    --skip_existing

EXIT_CODE=$?

echo ""
echo "============================================="
echo "Completed: ${TARGET}"
echo "Exit code: ${EXIT_CODE}"
echo "Finished at: $(date)"
echo "============================================="

nvidia-smi

exit ${EXIT_CODE}