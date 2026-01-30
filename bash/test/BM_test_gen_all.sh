#!/bin/bash
# =============================================================================
# submit_baseline_generation.sh
# 
# Submits parallel SLURM jobs to generate molecules using DiffSBDD baseline
# (Binding MOAD full-atom conditional model) across all protein targets.
#
# Usage: ./submit_baseline_generation.sh
# =============================================================================

# Binding MOAD full-atom conditional checkpoint
CKPT_PATH="/data/stat-cadd/wolf7055/PRISM/checkpoints/moad_fullatom_cond.ckpt"

# All targets to evaluate
TARGETS=(
    # BRD4_BD1 - 3 test structures
    "BRD4_BD1_4whw"
    "BRD4_BD1_6fo5"
    "BRD4_BD1_6xvc"
    
    # Carb_Anh_II - 3 test structures
    "Carb_Anh_II_6rl9"
    "Carb_Anh_II_3k34"
    "Carb_Anh_II_5n0d"
    
    # EGFR - 3 test structures
    "EGFR_8a27"
    "EGFR_3poz"
    "EGFR_4wkq"
    
    # Estrogen_recep_alpha - 3 test structures
    "Estrogen_recep_alpha_4ivy"
    "Estrogen_recep_alpha_5kct"
    "Estrogen_recep_alpha_2qzo"
    
    # Factor_Xa - 3 test structures
    "Factor_Xa_1ezq"
    "Factor_Xa_2p3t"
    "Factor_Xa_3kl6"

    # HIV_1_Protease - 3 test structures
    "HIV_1_Protease_2qnn"
    "HIV_1_Protease_3t11"
    "HIV_1_Protease_1hos"
)

# Create jobs directory if needed
mkdir -p bin_size_distribution_BM/BM_DiffSBDD


echo "============================================="
echo "Submitting ${#TARGETS[@]} parallel BM DiffSBDD baseline jobs"
echo "Model: Binding MOAD Full-Atom Conditional"
echo "============================================="
echo ""
echo "Checkpoint: ${CKPT_PATH}"
echo ""

# Track job IDs
declare -a JOB_IDS

for TARGET in "${TARGETS[@]}"; do
    # Submit job with target name and model path
    JOB_OUTPUT=$(sbatch --job-name="MOAD_${TARGET}" bash/test/BM_test_script.sh "${TARGET}" "${CKPT_PATH}")
    JOB_ID=$(echo "${JOB_OUTPUT}" | awk '{print $4}')
    JOB_IDS+=("${JOB_ID}")
    
    echo "Submitted ${TARGET}"
    echo "  Job ID: ${JOB_ID}"
    echo ""
done

echo "============================================="
echo "All jobs submitted!"
echo "============================================="
echo ""
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "Cancel all:"
echo "  scancel ${JOB_IDS[*]}"