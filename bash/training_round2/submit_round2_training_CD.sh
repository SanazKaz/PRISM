#!/bin/bash
# =============================================================================
# submit_round2_training.sh
# 
# Submits 24 parallel training jobs (6 targets × 4 seeds) for Round 2.
# Each job continues training from the best Round 1 checkpoint using the
# new reward function defined in ppo_config.yaml.
#
# Round 1: Baseline DiffSBDD → Train with Reward A → best checkpoint
# Round 2: Best checkpoint → Train with Reward B → new checkpoint
#
# Usage: ./submit_round2_training.sh
# =============================================================================

# Base path for Round 1 checkpoints
ROUND1_CKPT_BASE="/data/stat-cadd/wolf7055/PRISM/Log_Results/final_geometry_checks_CD/checkpoints"

# Round 1 best checkpoints (must match order in training script!)
RESUME_FROM_CHECKPOINTS=(
    "${ROUND1_CKPT_BASE}/BRD4_BD1/seed=976/epoch=43-reward=0.76.ckpt"
    "${ROUND1_CKPT_BASE}/Factor_Xa/seed=976/epoch=40-reward=0.76.ckpt"
    "${ROUND1_CKPT_BASE}/Carb_Anh_II/seed=976/epoch=44-reward=0.78.ckpt"
    "${ROUND1_CKPT_BASE}/EGFR/seed=123/epoch=42-reward=0.80.ckpt"
    "${ROUND1_CKPT_BASE}/Estrogen_recep_alpha/seed=123/epoch=30-reward=0.90.ckpt"
    "${ROUND1_CKPT_BASE}/HIV_1_Protease/seed=976/epoch=44-reward=0.84.ckpt"
)

# Target names (must match order of RESUME_FROM_CHECKPOINTS!)
TARGETS=(
    "BRD4_BD1"
    "Factor_Xa"
    "Carb_Anh_II"
    "EGFR"
    "Estrogen_recep_alpha"
    "HIV_1_Protease"
)

# Seeds to use (4 seeds per target = 24 jobs total)
SEEDS=(42 976 123 789)

PROJECT_ROOT="/data/stat-cadd/wolf7055/PRISM"

echo "============================================="
echo "Round 2 Training Submission"
echo "Starting from Round 1 best checkpoints"
echo "============================================="
echo ""
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
        echo "          Target: ${TARGET}"
    else
        echo "[OK] ${TARGET}: $(basename ${CKPT})"
    fi
done
echo ""

# Create log directory
LOG_DIR="${PROJECT_ROOT}/axis_2_sillwalks_CD"
mkdir -p "${LOG_DIR}"

echo "Submitting SLURM array job (24 tasks)..."
echo ""

# Submit the array job
JOB_OUTPUT=$(sbatch bash/training_round2/round2_training_CD.sh)
JOB_ID=$(echo "${JOB_OUTPUT}" | awk '{print $4}')

echo "============================================="
echo "Job submitted!"
echo "============================================="
echo ""
echo "Job ID: ${JOB_ID}"
echo "Array tasks: 0-23 (24 total)"
echo ""
echo "Task mapping:"
for i in "${!TARGETS[@]}"; do
    TARGET="${TARGETS[$i]}"
    CKPT="${RESUME_FROM_CHECKPOINTS[$i]}"
    echo "  Dataset ${i} (${TARGET}):"
    for s in "${!SEEDS[@]}"; do
        TASK_ID=$((i * ${#SEEDS[@]} + s))
        echo "    Task ${TASK_ID}: seed=${SEEDS[$s]} → ${CKPT}"
    done
done
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  squeue -j ${JOB_ID}"
echo ""
echo "Check logs:"
echo "  tail -f ${LOG_DIR}/*/seed_*.log"
echo ""
echo "Cancel job:"
echo "  scancel ${JOB_ID}"