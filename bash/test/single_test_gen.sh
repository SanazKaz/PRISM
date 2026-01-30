#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --partition=short 
#SBATCH --gres=gpu:h100:1
#SBATCH --time 05:00:00
#SBATCH --job-name=anamoly_test_moad
#SBATCH --mail-user=wolf7055@ox.ac.uk
#SBATCH --output=jobs_files/anamoly_detection_moad/anamoly_test_moad%j.log
exec 2>&1

module purge
module load Anaconda3
source activate /data/stat-cadd/wolf7055/conda/envs/PRISM_25

echo "Python executable: $(which python)"
chmod +x /data/stat-cadd/wolf7055/PRISM/val_analysis/smina.static

# --- CONFIGURATION ---
PRISM_ROOT="/data/stat-cadd/wolf7055/PRISM"

# Model checkpoint - UPDATE THIS
MODEL_PATH="${PRISM_ROOT}/checkpoints/moad_fullatom_cond.ckpt"

# Target to test
TARGET="BRD4_BD1_6fo5"

# Output directory
OUTDIR="${PRISM_ROOT}/jobs_files/anamoly_detection_moad"
mkdir -p "${OUTDIR}" # Create output directory if it doesn't exist

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
    --config "${PRISM_ROOT}/configs/binding_moad_fa_ppo.yaml" \
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