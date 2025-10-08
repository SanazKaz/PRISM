#!/bin/bash
#SBATCH --cpus-per-task=12
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1            
#SBATCH --time=03:00:00
#SBATCH --partition=short
#SBATCH --job-name=preproc_full
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wolf7055@ox.ac.uk
#SBATCH --output=jobs_files/preprocessing_%j.log
#SBATCH --error=jobs_files/preprocessing_%j.err

set -e                              # stop on first error
module purge
module load Anaconda3
# Load required module
source activate /data/stat-cadd/wolf7055/conda/envs/TEST_ENV

echo "Python: $(which python)"
free -h
nvidia-smi

# ---------------------------------------------------------------------------
# 1) Build full-atom dataset
# ---------------------------------------------------------------------------
OUTDIR=data/processed_crossdock_noH_full_temp   # ✱ renamed _ca_ instead of _full_
python process_crossdock.py \
       /data/stat-cadd/wolf7055/diffsbdd-ppo/data/crossdocked_pocket10 \
       --no_H \
       --outdir "${OUTDIR}"

# ---------------------------------------------------------------------------
# 2) Quick sanity check
# ---------------------------------------------------------------------------
python - << PY
import numpy as np, sys, pathlib
d = np.load(pathlib.Path("${OUTDIR}") / "train.npz")
print("✅  pocket_one_hot dim :", d["pocket_one_hot"].shape[1])  # expect 20
print("✅  ligand_one_hot dim :", d["lig_one_hot"].shape[1])     # expect 10
PY
