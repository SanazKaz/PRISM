#!/bin/bash
#SBATCH --job-name=case_study_train
#SBATCH --partition=medium
#SBATCH --qos=ecr
#SBATCH --time=35:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h100:2
#SBATCH --array=0-23
#SBATCH --output=targetdiff_jobs/%x_%A_%a.log
#SBATCH --error=targetdiff_jobs/%x_%A_%a.log
#SBATCH --mail-user=wolf7055@ox.ac.uk
#SBATCH --mail-type=END,FAIL

# ---------------------------------------------------------------------------
# Case-study PPO fine-tuning. The array sweeps the 4 canonical seeds x 6 targets
# (24 tasks). The config is the single arg, so the same script drives each
# reward set — submit once per config:
#
#   sbatch bash/target_diff/train_case_study_targets.sh \
#       configs/targetdiff/case_studies/multi_objective_pharm_product.yaml
#   sbatch bash/target_diff/train_case_study_targets.sh \
#       configs/targetdiff/case_studies/multi_objective_pharm_weighted_sum.yaml
#
# Per (seed,target) it overrides datadir / hotspot pkl / target_name /
# dataset_name so the single config (default BRD4_BD1) is retargeted.
# Checkpoints land under: Log_Results/case_studies/<run_identifier>/checkpoints/<TARGET>/seed=<SEED>/
# SBATCH directives mirror multi_obj_product_seeds.sh (medium H100s, qos=ecr).
# ---------------------------------------------------------------------------

CONFIG="${1:?usage: sbatch bash/target_diff/train_case_study_targets.sh <CONFIG>}"

module purge
module load Anaconda3
source activate /data/stat-cadd/wolf7055/conda/envs/PRISM_25

export PROJECT_ROOT="/data/stat-cadd/wolf7055/PRISM"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_PPO=0
export PRISM_GRAD_CKPT=1

# Canonical seeds x case-study targets. Decode the 0-23 array id:
#   target = id % 6   (each has 03_final_dataset_targetdiff + hotspot pkl + a
#                       propeties_ref.json entry keyed by this exact name)
#   seed   = id / 6   (canonical project seeds)
SEEDS=(42 976 123 789)
TARGETS=(BRD4_BD1 Carb_Anh_II EGFR Estrogen_recep_alpha Factor_Xa HIV_1_Protease)
TARGET=${TARGETS[$(( SLURM_ARRAY_TASK_ID % 6 ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID / 6 ))]}

WARM_START_CKPT="${PROJECT_ROOT}/checkpoints/targetdiff_pretrained_models/targetdiff_pretrained_diffusion.pt"
DATADIR="${PROJECT_ROOT}/data/${TARGET}/03_final_dataset_targetdiff"
HOTSPOT="${PROJECT_ROOT}/data/${TARGET}/hotspot_analysis/${TARGET}_hotspot_data.pkl"
LOGDIR="${PROJECT_ROOT}/Log_Results/case_studies"

echo "Job: $SLURM_JOB_ID | Array task: $SLURM_ARRAY_TASK_ID | Target: $TARGET | Seed: $SEED"
echo "Config:  $CONFIG"
echo "Datadir: $DATADIR"
echo "Hotspot: $HOTSPOT"

[ -f "${WARM_START_CKPT}" ] || { echo "[ERROR] Checkpoint not found: ${WARM_START_CKPT}"; exit 1; }
[ -f "${CONFIG}" ]          || { echo "[ERROR] Config not found: ${CONFIG}"; exit 1; }
[ -f "${HOTSPOT}" ]         || { echo "[ERROR] Hotspot not found: ${HOTSPOT}"; exit 1; }
[ -d "${DATADIR}" ]         || { echo "[ERROR] Datadir not found: ${DATADIR}"; exit 1; }

cd "$PROJECT_ROOT"
nvidia-smi
start_time=$(date +%s)
echo "Start time: $(date +%T)"

echo "Starting training..."
srun python scripts/train.py \
    --config "${CONFIG}" \
    --warm_start_from_ddpm "${WARM_START_CKPT}" \
    --seed "${SEED}" \
    --datadir "${DATADIR}" \
    --hotspot_path "${HOTSPOT}" \
    --target_name "${TARGET}" \
    --dataset_name "${TARGET}" \
    --logdir "${LOGDIR}"

echo "Done."
end_time=$(date +%s)
echo "End time: $(date +%T)"
echo "Time taken: $((end_time - start_time)) seconds"
