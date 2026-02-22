#!/bin/bash
# =============================================================================
# submit_round2_training.sh
# 
# Submits 24 parallel training jobs (6 targets × 4 seeds) for Round 2.
# Usage: ./submit_round2_training.sh
# =============================================================================

# =============================================================================
# 1. PATH & VARIABLE DEFINITIONS
# =============================================================================
PROJECT_ROOT="/data/stat-cadd/wolf7055/PRISM"
ROUND1_CKPT_BASE="${PROJECT_ROOT}/Log_Results/final_geometry_checks_CD/checkpoints"

TRAINING_SCRIPT_PATH="bash/training_round2/round2_training_CD.sh"
LOG_DIR="${PROJECT_ROOT}/jobs_files/2D_property_CD_geom"


# Round 1 best checkpoints (Must match order of TARGETS)
RESUME_FROM_CHECKPOINTS=(
    "${ROUND1_CKPT_BASE}/BRD4_BD1/seed=976/epoch=43-reward=0.76.ckpt"
    "${ROUND1_CKPT_BASE}/Factor_Xa/seed=976/epoch=40-reward=0.76.ckpt"
    "${ROUND1_CKPT_BASE}/Carb_Anh_II/seed=976/epoch=44-reward=0.78.ckpt"
    "${ROUND1_CKPT_BASE}/EGFR/seed=123/epoch=42-reward=0.80.ckpt"
    "${ROUND1_CKPT_BASE}/Estrogen_recep_alpha/seed=123/epoch=30-reward=0.90.ckpt"
    "${ROUND1_CKPT_BASE}/HIV_1_Protease/seed=976/epoch=44-reward=0.84.ckpt"
)

TARGETS=(
    "BRD4_BD1"
    "Factor_Xa"
    "Carb_Anh_II"
    "EGFR"
    "Estrogen_recep_alpha"
    "HIV_1_Protease"
)

SEEDS=(42 976 123 789)

# =============================================================================
# 2. PRE-FLIGHT VERIFICATION
# =============================================================================
echo "============================================="
echo "Round 2 Training Submission"
echo "============================================="
echo "Targets: ${#TARGETS[@]}"
echo "Seeds per target: ${#SEEDS[@]}"
echo "Total jobs: $((${#TARGETS[@]} * ${#SEEDS[@]}))"
echo ""

# Verify all checkpoints exist
echo "Verifying Round 1 checkpoints..."
for i in "${!TARGETS[@]}"; do
    TARGET="${TARGETS[$i]}"
    CKPT="${RESUME_FROM_CHECKPOINTS[$i]}"
    if [ ! -f "${CKPT}" ]; then
        echo "[WARNING] Checkpoint not found: ${CKPT}"
    else
        echo "[OK] ${TARGET}: $(basename "${CKPT}")"
    fi
done
echo ""

# Verify the SLURM script exists
if [ ! -f "${PROJECT_ROOT}/${TRAINING_SCRIPT_PATH}" ]; then
    echo "[ERROR] SLURM training script not found at ${PROJECT_ROOT}/${TRAINING_SCRIPT_PATH}"
    exit 1
fi

# Create log directory
mkdir -p "${LOG_DIR}"

# =============================================================================
# 3. SUBMISSION
# =============================================================================
echo "Submitting SLURM array job (Tasks 0-23)..."
echo ""

JOB_OUTPUT=$(sbatch "${PROJECT_ROOT}/${TRAINING_SCRIPT_PATH}")
JOB_ID=$(echo "${JOB_OUTPUT}" | awk '{print $4}')

# =============================================================================
# 4. OUTPUT SUMMARY
# =============================================================================
echo "============================================="
echo "Job submitted successfully!"
echo "============================================="
echo "Job ID:      ${JOB_ID}"
echo "Array tasks: 0-23"
echo ""
echo "Monitor with:"
echo "  squeue -j ${JOB_ID}"
echo ""
echo "Check progress (Wait a few seconds for logs to initialize):"
echo "  tail -f ${LOG_DIR}/*/seed_*.log"
echo ""
echo "Cancel job:"
echo "  scancel ${JOB_ID}"