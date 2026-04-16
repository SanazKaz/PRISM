#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --partition=short
#SBATCH --gres=gpu:h100:1
#SBATCH --time 00:45:00
#SBATCH --output=jobs_files/03_04_dock_molprops_geom/test_targets/%x_%j.log
exec 2>&1

# =============================================================================
# CD Geometry Model Evaluation: Generate ligands from trained checkpoints
# 
# Arguments:
#   $1 - Target name (e.g., BRD4_BD1)
#   $2 - Path to model checkpoint (.pt file)
#
# Usage: Called by submit_all_targets_test.sh
# =============================================================================

TARGET=$1
MODEL_PATH=$2

if [ -z "${TARGET}" ]; then
    echo "[ERROR] No target specified!"
    echo "Usage: sbatch test_file.sh TARGET_NAME MODEL_PATH"
    exit 1
fi

if [ -z "${MODEL_PATH}" ]; then
    echo "[ERROR] No model path specified!"
    echo "Usage: sbatch test_file.sh TARGET_NAME MODEL_PATH"
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
echo "CD Geometry Model Evaluation: ${TARGET}"
echo "Started at: $(date)"
echo "Node: $(hostname)"
echo "============================================="

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

PRISM_ROOT="/data/stat-cadd/wolf7055/PRISM"
SCRIPT_PATH="${PRISM_ROOT}/scripts/test_targets.py"
CONFIG_PATH="${PRISM_ROOT}/configs/ppo_config.yaml"

# Output directory for CD geometry model generations
OUTDIR="/data/stat-cadd/wolf7055/PRISM/results/case_studies/03_04_dock_molprops_geom/test_targets"
mkdir -p ${OUTDIR}

# Generation settings
N_SAMPLES=1000
BATCH_SIZE=100

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
    --fix_n_nodes

EXIT_CODE=$?

echo ""
echo "============================================="
echo "Completed: ${TARGET}"
echo "Exit code: ${EXIT_CODE}"
echo "Finished at: $(date)"
echo "============================================="

exit ${EXIT_CODE}