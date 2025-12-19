#!/bin/bash
# =============================================================================
# submit_all_targets.sh
# 
# Submits 6 parallel SLURM jobs, one per protein target.
# Each job generates 10k molecules independently.
#
# Usage: ./submit_all_targets.sh
# =============================================================================

# All 6 targets (must match keys in test_prism_targets.py)
TARGETS=(
    "AMPC_beta_lactamase"
    "Carb_Anh_II"
    "COVID19_main_protease"
    "Estrogen_recep_alpha"
    "EGFR"
    "HIV_1_Protease"
)

# Create jobs directory if needed
mkdir -p jobs_files

echo "============================================="
echo "Submitting ${#TARGETS[@]} parallel PRISM evaluation jobs"
echo "============================================="
echo ""

# Track job IDs
declare -a JOB_IDS

for TARGET in "${TARGETS[@]}"; do
    # Submit job with target-specific name
    JOB_OUTPUT=$(sbatch --job-name="${TARGET}" bash/test_file.sh "${TARGET}")
    JOB_ID=$(echo "${JOB_OUTPUT}" | awk '{print $4}')
    JOB_IDS+=("${JOB_ID}")
    
    echo "Submitted ${TARGET}: Job ID ${JOB_ID}"
done

echo ""
echo "============================================="
echo "All jobs submitted!"
echo "============================================="
echo ""
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "Check logs:"
echo "  tail -f jobs_files/prism_TARGET_JOBID.log"
echo ""
echo "Cancel all:"
echo "  scancel ${JOB_IDS[*]}"