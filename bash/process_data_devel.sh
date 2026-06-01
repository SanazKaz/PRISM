#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --ntasks-per-node=1
#SBATCH --partition=devel
#SBATCH --time=01:00:00
#SBATCH --job-name=prism_process_data
#SBATCH --mail-user=wolf7055@ox.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --output=jobs_files/process_data_%j.log
#SBATCH --error=jobs_files/process_data_%j.log

module purge
module load Anaconda3
source activate /data/stat-cadd/wolf7055/conda/envs/PRISM_25

export PROJECT_ROOT="/data/stat-cadd/wolf7055/PRISM"
OUTPUT_BASE="/tmp/custom_diffsbdd"
OUTPUT_TD="/tmp/custom_targetdiff"

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Project root: $PROJECT_ROOT"
echo "=========================================="

cd $PROJECT_ROOT
which python

# --- Step 1+2+3: DiffSBDD (downloads PDBs from RCSB) ---
echo ""
echo "=== Running process_data: DiffSBDD ==="
python -m scripts.process_data \
    --pdb_list "${PROJECT_ROOT}/data/example_pdbs.txt" \
    --output_dir "${OUTPUT_BASE}" \
    --model diffsbdd \
    --test_pdbs none

echo ""
echo "=== Running process_data: TargetDiff (reuses downloaded PDBs) ==="
python -m scripts.process_data \
    --skip_fetch \
    --pdb_dir "${OUTPUT_BASE}/01_raw_pdbs" \
    --output_dir "${OUTPUT_TD}" \
    --model targetdiff \
    --test_pdbs none

# --- Verify output shapes ---
echo ""
echo "=== Verifying NPZ shapes ==="
python - << 'PY'
import numpy as np, sys
for label, path, expect_poc, expect_lig in [
    ("diffsbdd",   "/tmp/custom_diffsbdd/03_final_dataset/train.npz",              10, 10),
    ("targetdiff", "/tmp/custom_targetdiff/03_final_dataset_targetdiff/train.npz", 27, 10),
]:
    try:
        d = np.load(path, allow_pickle=True)
        poc = d['pocket_one_hot'].shape[-1]
        lig = d['lig_one_hot'].shape[-1]
        status = "OK" if poc == expect_poc and lig == expect_lig else "MISMATCH"
        print(f"[{status}] {label}: pocket={poc} (expect {expect_poc})  lig={lig} (expect {expect_lig})")
    except FileNotFoundError:
        print(f"[MISSING] {label}: {path}", file=sys.stderr)
PY

echo ""
echo "Done. Logs: jobs_files/process_data_${SLURM_JOB_ID}.log"
echo "Final memory usage:"
free -h
