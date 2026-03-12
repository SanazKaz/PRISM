#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=short
#SBATCH --time 02:30:00
#SBATCH --job-name=process
#SBATCH --output=jobs_files/crossdocked_process.log

###############################################################################
# Process Crossdocked Datasets Script
#
# This script processes the crossdocked datasets and saves them to the data/cross_dock/crossdocked_pocket10 directory.
#
# Usage: sbatch bash/datatset_prep/process_crossdocked.sh
###############################################################################

set -e

# --- Environment Setup ---
module purge
module load Anaconda3
source activate /data/stat-cadd/wolf7055/conda/envs/PRISM_25
echo "Python executable: $(which python)"
echo ""

python "/data/stat-cadd/wolf7055/PRISM/src/models/diffsbdd/process_crossdock.py" "/data/stat-cadd/wolf7055/PRISM/data/cross_dock/crossdocked_pocket10" --no_H