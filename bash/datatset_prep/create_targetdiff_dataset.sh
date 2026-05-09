#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --job-name=create_targetdiff_dataset
#SBATCH --mail-user=wolf7055@ox.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --output=jobs_files/create_targetdiff_dataset.log
#SBATCH --error=jobs_files/create_targetdiff_dataset.log

module purge
module load Anaconda3
source activate /data/stat-cadd/wolf7055/conda/envs/PRISM_25

export PROJECT_ROOT="/data/stat-cadd/wolf7055/PRISM"

CROSSDOCKED_DIR="${PROJECT_ROOT}/data/cross_dock/crossdocked_pocket10"
SPLIT_PATH="${PROJECT_ROOT}/data/cross_dock/split_by_name.pt"
OUTPUT_DIR="${PROJECT_ROOT}/data/cross_dock/processed_crossdock_targetdiff"

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Input:  $CROSSDOCKED_DIR"
echo "Split:  $SPLIT_PATH"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

cd "$PROJECT_ROOT"
which python

python -m scripts.process_crossdock_targetdiff \
    --crossdocked_dir "$CROSSDOCKED_DIR" \
    --split_path      "$SPLIT_PATH" \
    --output_dir      "$OUTPUT_DIR"

echo "Done!"
ls -lh "$OUTPUT_DIR/"
