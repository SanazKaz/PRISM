#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short
#SBATCH --time=02:30:00
#SBATCH --job-name=MOAD_RUN_GEOMETRY_TRAINING
#SBATCH --mail-user=wolf7055@ox.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-3
#SBATCH --output=jobs_files/short_moad_run_geometry_training_%a.log
#SBATCH --error=jobs_files/short_moad_run_geometry_training_%a.log

module purge
module load Anaconda3
source activate /data/stat-cadd/wolf7055/conda/envs/PRISM_25


# --- 1. SETUP PATHS ---
export PROJECT_ROOT="/data/stat-cadd/wolf7055/PRISM"

SEEDS=(42 976 123 789)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

WARM_START_CKPT="${PROJECT_ROOT}/checkpoints/moad_fullatom_cond.ckpt"
CHECKPOINT_OUTPUT_DIR="${PROJECT_ROOT}/Log_Results"

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Using seed: $SEED"
echo "Warm start checkpoint: $WARM_START_CKPT"
echo "Checkpoint output: $CHECKPOINT_OUTPUT_DIR"
echo "=========================================="

cd $PROJECT_ROOT

if [ ! -f "${WARM_START_CKPT}" ]; then
    echo "[ERROR] Warm start checkpoint not found: ${WARM_START_CKPT}"
    exit 1
fi


# --- 2. STAGE IN (Copy CrossDocked dataset to scratch) ---
SOURCE_DATADIR=$(python -c "
import yaml
with open('${PROJECT_ROOT}/configs/ppo_config.yaml') as f:
    cfg = yaml.safe_load(f)
print(cfg['datadir'])
")

SCRATCH_WORK_DIR="${TMPDIR}/${SLURM_JOB_ID}/crossdocked"

export DEBUG_PPO=0

echo "Starting Data Stage-in..."
echo "Copying from: $SOURCE_DATADIR"
echo "Copying to:   $SCRATCH_WORK_DIR"
start_time=$(date +%s)

mkdir -p "$SCRATCH_WORK_DIR"
rsync -ah "$SOURCE_DATADIR/" "$SCRATCH_WORK_DIR/"

end_time=$(date +%s)
echo "Data staged in $((end_time - start_time)) seconds."

echo "Verifying copied structure:"
ls -lh "$SCRATCH_WORK_DIR/"


# --- 3. PYTHON SETUP ---
which python

python - << 'PY'
import torch, os
print("CVD", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("avail", torch.cuda.is_available(), "count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("name0", torch.cuda.get_device_name(0))
PY

nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_PPO=0


# --- 4. RUN TRAINING ---
echo "Starting training..."
srun python "${PROJECT_ROOT}/scripts/train.py" \
    --config "${PROJECT_ROOT}/configs/ppo_config.yaml" \
    --warm_start_from_ddpm "${WARM_START_CKPT}" \
    --seed $SEED \
    --datadir "$SCRATCH_WORK_DIR" \
    --logdir "$CHECKPOINT_OUTPUT_DIR"

echo "Training completed!"


# --- 5. CLEANUP ---
echo "Cleaning up scratch..."
rm -rf "${TMPDIR}/${SLURM_JOB_ID}"

echo "Final Memory Usage:"
free -h
echo "Final GPU Usage:"
nvidia-smi