#!/bin/bash
# =============================================================================
# submit_all_targets.sh
# 
# Submits 18 parallel SLURM jobs to generate molecules using target-specific
# PRISM models across all test structures (6 targets × 3 structures each).
#
# Usage: ./submit_all_targets.sh
# =============================================================================

# Base path for checkpoints
CKPT_BASE="/data/stat-cadd/wolf7055/PRISM/Log_Results/molecular_props_geom_step2/molecular_props_geom_step2/checkpoints"

# Map each test structure to its trained model checkpoint
declare -A MODEL_MAP
# BRD4_BD1 structures → BRD4 model - good
MODEL_MAP["BRD4_BD1_4whw"]="${CKPT_BASE}/BRD4_BD1/seed=42/last.pt"
MODEL_MAP["BRD4_BD1_6fo5"]="${CKPT_BASE}/BRD4_BD1/seed=42/last.pt"
MODEL_MAP["BRD4_BD1_6xvc"]="${CKPT_BASE}/BRD4_BD1/seed=42/last.pt"

# Carb_Anh_II structures → Carb_Anh_II model - good
MODEL_MAP["Carb_Anh_II_6rl9"]="${CKPT_BASE}/Carb_Anh_II/seed=123/epoch=90-reward=0.56.pt"
MODEL_MAP["Carb_Anh_II_3k34"]="${CKPT_BASE}/Carb_Anh_II/seed=123/epoch=90-reward=0.56.pt"
MODEL_MAP["Carb_Anh_II_5n0d"]="${CKPT_BASE}/Carb_Anh_II/seed=123/epoch=90-reward=0.56.pt"

# EGFR structures → EGFR model - BAD
MODEL_MAP["EGFR_8a27"]="${CKPT_BASE}/EGFR/seed=42/last.pt"
MODEL_MAP["EGFR_3poz"]="${CKPT_BASE}/EGFR/seed=42/last.pt"
MODEL_MAP["EGFR_4wkq"]="${CKPT_BASE}/EGFR/seed=42/last.pt"

# Estrogen_recep_alpha structures → Estrogen model - mediocre
MODEL_MAP["Estrogen_recep_alpha_4ivy"]="${CKPT_BASE}/Estrogen_recep_alpha/seed=976/last.pt"
MODEL_MAP["Estrogen_recep_alpha_5kct"]="${CKPT_BASE}/Estrogen_recep_alpha/seed=976/last.pt"
MODEL_MAP["Estrogen_recep_alpha_2qzo"]="${CKPT_BASE}/Estrogen_recep_alpha/seed=976/last.pt"

# Factor_Xa structures → Factor_Xa model bad
MODEL_MAP["Factor_Xa_1ezq"]="${CKPT_BASE}/Factor_Xa/seed=789/epoch=86-reward=0.60.pt"
MODEL_MAP["Factor_Xa_2p3t"]="${CKPT_BASE}/Factor_Xa/seed=789/epoch=86-reward=0.60.pt"
MODEL_MAP["Factor_Xa_3kl6"]="${CKPT_BASE}/Factor_Xa/seed=789/epoch=86-reward=0.60.pt"

# HIV_1_Protease structures → HIV model good
MODEL_MAP["HIV_1_Protease_2qnn"]="${CKPT_BASE}/HIV_1_Protease/seed=123/last.pt"
MODEL_MAP["HIV_1_Protease_3t11"]="${CKPT_BASE}/HIV_1_Protease/seed=123/last.pt"
MODEL_MAP["HIV_1_Protease_1hos"]="${CKPT_BASE}/HIV_1_Protease/seed=123/last.pt"

# All targets to evaluate (18 test structures)
TARGETS=(
    # BRD4_BD1 - 3 test structures
    "BRD4_BD1_4whw"
    "BRD4_BD1_6fo5"
    "BRD4_BD1_6xvc"
    
    # Carb_Anh_II - 3 test structures
    # "Carb_Anh_II_6rl9"
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
    # "HIV_1_Protease_2qnn"
    # "HIV_1_Protease_3t11"
    # "HIV_1_Protease_1hos"
)

# Create jobs directory if needed
mkdir -p jobs_files/generating_molecular_props_geom_step2/quick_test

echo "============================================="
echo "Submitting ${#TARGETS[@]} parallel PRISM CD geometry evaluation jobs"
echo "============================================="
echo ""

# Track job IDs
declare -a JOB_IDS

for TARGET in "${TARGETS[@]}"; do
    MODEL_PATH="${MODEL_MAP[$TARGET]}"
    
    if [ -z "${MODEL_PATH}" ]; then
        echo "[ERROR] No checkpoint found for ${TARGET}"
        continue
    fi
    
    if [ ! -f "${MODEL_PATH}" ]; then
        echo "[ERROR] Model file not found: ${MODEL_PATH}"
        continue
    fi
    
    # Submit job with target name and model path
    JOB_OUTPUT=$(sbatch --job-name="CD_${TARGET}" bash/test/trained_model_test_script.sh "${TARGET}" "${MODEL_PATH}")
    JOB_ID=$(echo "${JOB_OUTPUT}" | awk '{print $4}')
    JOB_IDS+=("${JOB_ID}")
    
    echo "Submitted ${TARGET}"
    echo "  Model: ${MODEL_PATH}"
    echo "  Job ID: ${JOB_ID}"
    echo ""
done

echo "============================================="
echo "All ${#JOB_IDS[@]} jobs submitted!"
echo "============================================="
echo ""
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "Check individual logs:"
echo "  tail -f jobs_files/generating_molecular_props_geom_step2/quick_test/CD_BRD4_BD1_4whw_*.log"
echo ""
echo "Cancel all:"
echo "  scancel ${JOB_IDS[*]}"