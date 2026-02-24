#!/bin/bash

# --- CONFIGURATION ---
CONFIG="/data/stat-cadd/wolf7055/PRISM/configs/ppo_config.yaml"
TARGET="BRD4_BD1_4whw"
N_SAMPLES=1000
BATCH_SIZE=64
PROJECT_ROOT="/data/stat-cadd/wolf7055/PRISM"

# Checkpoint 1: floored
CKPT_1="/data/stat-cadd/wolf7055/PRISM/Log_Results/floored_property_2d_fused_rings_test_chiralitywidened/checkpoints/tmp/seed=976/last.pt"
OUTDIR_1="MOL_PROPS_TESTS/floored"
JOBNAME_1="floored"

# Checkpoint 2: logp_floored
CKPT_2="//data/stat-cadd/wolf7055/PRISM/Log_Results/log_p_floored_property_2d_fused_rings_test_chiralitywidened/checkpoints/tmp/seed=976/last.pt"
OUTDIR_2="MOL_PROPS_TESTS/logp_floored"
JOBNAME_2="test_chiralitywidened_logp_floored"

# --- SUBMIT JOBS ---
for i in 1 2; do
    CKPT_VAR="CKPT_${i}"
    OUTDIR_VAR="OUTDIR_${i}"
    JOBNAME_VAR="JOBNAME_${i}"

    CKPT="${!CKPT_VAR}"
    OUTDIR="${!OUTDIR_VAR}"
    JOBNAME="${!JOBNAME_VAR}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --job-name=${JOBNAME}
#SBATCH --output=jobs_files/${JOBNAME}.log

# --- Environment Setup ---
module purge
module load Anaconda3
source activate /data/stat-cadd/wolf7055/conda/envs/PRISM_25

echo "Python executable: \$(which python)"

export PROJECT_ROOT="${PROJECT_ROOT}"
export DEBUG_PPO=0

echo "=========================================="
echo "Checkpoint:  ${CKPT}"
echo "Output Dir:  ${OUTDIR}"
echo "Target:      ${TARGET}"
echo "=========================================="

cd \$PROJECT_ROOT

echo "Initial GPU Usage:"
nvidia-smi

srun python "\${PROJECT_ROOT}/scripts/test.py" \
    "${CKPT}" \
    --config "${CONFIG}" \
    --outdir "${OUTDIR}" \
    --target "${TARGET}" \
    --n_samples ${N_SAMPLES} \
    --batch_size ${BATCH_SIZE} \
    --sanitize

echo "Job complete."
echo "Final GPU Usage:"
nvidia-smi
EOF

    echo "Submitted job: ${JOBNAME}"
done