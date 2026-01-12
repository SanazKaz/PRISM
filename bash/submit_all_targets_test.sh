#!/bin/bash
# =============================================================================
# submit_all_targets.sh
# 
# Submits 5 parallel SLURM jobs, one per protein target.
# Each job generates 10k molecules using target-specific trained models.
#
# Usage: ./submit_all_targets.sh
# =============================================================================

# Base path for checkpoints
CKPT_BASE="/data/stat-cadd/wolf7055/PRISM/Log_Results/PB_Final_Run_0.5_0.5_to_0.7_0.3/checkpoints"

# Declare associative array mapping targets to their best checkpoint paths
declare -A TARGET_MODELS
TARGET_MODELS["AMPC_beta_lactamase"]="${CKPT_BASE}/AMPC_beta_lactamase/seed=42/epoch=33-reward=1.28.pt"
TARGET_MODELS["Carb_Anh_II"]="${CKPT_BASE}/Carb_Anh_II/seed=42/epoch=33-reward=1.23.pt"
TARGET_MODELS["COVID19_main_protease"]="${CKPT_BASE}/covid19_main_protease/seed=123/epoch=31-reward=1.18.pt"
TARGET_MODELS["EGFR"]="${CKPT_BASE}/EGFR/seed=42/epoch=34-reward=1.20.pt"
TARGET_MODELS["Estrogen_recep_alpha"]="${CKPT_BASE}/Estrogen_recep_alpha/seed=789/epoch=34-reward=1.23.pt"

# Create jobs directory if needed
mkdir -p jobs_files

echo "============================================="
echo "Submitting ${#TARGET_MODELS[@]} parallel PRISM evaluation jobs"
echo "============================================="
echo ""

# Track job IDs
declare -a JOB_IDS

for TARGET in "${!TARGET_MODELS[@]}"; do
    MODEL_PATH="${TARGET_MODELS[$TARGET]}"
    
    # Submit job with target name and model path
    JOB_OUTPUT=$(sbatch --job-name="${TARGET}" bash/test_file.sh "${TARGET}" "${MODEL_PATH}")
    JOB_ID=$(echo "${JOB_OUTPUT}" | awk '{print $4}')
    JOB_IDS+=("${JOB_ID}")
    
    echo "Submitted ${TARGET}"
    echo "  Model: ${MODEL_PATH}"
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
echo "Check logs:"
echo "  tail -f jobs_files/DiffSBDD/crossdocked_fa_cond_temp_TARGET_JOBID.log"
echo ""
echo "Cancel all:"
echo "  scancel ${JOB_IDS[*]}"