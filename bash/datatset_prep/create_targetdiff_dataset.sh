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
CROSSDOCKED_DIR="${PROJECT_ROOT}/data/cross_dock/processed_crossdock_noH_full_temp"

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Creating TargetDiff 27-dim dataset from:"
echo "  $CROSSDOCKED_DIR"
echo "Output will be:"
echo "  ${CROSSDOCKED_DIR}/03_final_dataset_targetdiff"
echo "=========================================="

cd "$PROJECT_ROOT"

which python
python -c "import numpy; print('numpy:', numpy.__version__)"

echo "Starting dataset creation..."
python -m scripts.process_data \
    --skip_fetch \
    --pdb_dir "${CROSSDOCKED_DIR}/02_preprocessed" \
    --output_dir "${CROSSDOCKED_DIR}" \
    --model targetdiff

echo "Done!"
ls -lh "${CROSSDOCKED_DIR}/03_final_dataset_targetdiff/"
