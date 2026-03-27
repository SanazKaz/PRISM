#!/bin/bash
# =============================================================================
# submit_diffsbdd_baseline.sh
#
# Submits 18 parallel SLURM jobs to generate molecules using the DiffSBDD
# baseline model across all test structures (6 targets x 3 structures each).
#
# Usage: ./submit_diffsbdd_baseline.sh
# =============================================================================

MODEL_PATH="/data/stat-cadd/wolf7055/PRISM/checkpoints/crossdocked_fa_cond_temp.ckpt"

TARGETS=(
    "BRD4_BD1_4whw"
    "BRD4_BD1_6fo5"
    "BRD4_BD1_6xvc"
    "Carb_Anh_II_6rl9"
    "Carb_Anh_II_3k34"
    "Carb_Anh_II_5n0d"
    "EGFR_8a27"
    "EGFR_3poz"
    "EGFR_4wkq"
    "Estrogen_recep_alpha_4ivy"
    "Estrogen_recep_alpha_5kct"
    "Estrogen_recep_alpha_2qzo"
    "Factor_Xa_1ezq"
    "Factor_Xa_2p3t"
    "Factor_Xa_3kl6"
    "HIV_1_Protease_2qnn"
    "HIV_1_Protease_3t11"
    "HIV_1_Protease_1hos"
)

mkdir -p jobs_files/diffsbdd_baseline_targets

echo "============================================="
echo "Submitting ${#TARGETS[@]} DiffSBDD baseline jobs"
echo "Model: ${MODEL_PATH}"
echo "============================================="
echo ""

declare -a JOB_IDS

for TARGET in "${TARGETS[@]}"; do

    if [ ! -f "${MODEL_PATH}" ]; then
        echo "[ERROR] Model file not found: ${MODEL_PATH}"
        exit 1
    fi

    JOB_OUTPUT=$(sbatch --job-name="DIFFSBDD_${TARGET}" \
                        bash/test/diffsbdd_baseline.sh "${TARGET}" "${MODEL_PATH}")
    JOB_ID=$(echo "${JOB_OUTPUT}" | awk '{print $4}')
    JOB_IDS+=("${JOB_ID}")

    echo "Submitted ${TARGET} | Job ID: ${JOB_ID}"
done

echo ""
echo "============================================="
echo "All ${#JOB_IDS[@]} jobs submitted!"
echo "============================================="
echo ""
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "Cancel all:"
echo "  scancel ${JOB_IDS[*]}"