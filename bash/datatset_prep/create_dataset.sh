#!/bin/bash
#SBATCH --cpus-per-task=6
#SBATCH --mem=64GB
#SBATCH --partition=short
#SBATCH --time 01:00:00
#SBATCH --job-name=PRISM_Regen
#SBATCH --output=jobs_files/Regenerate_Final_Datasets.log

###############################################################################
# Regenerate Final Datasets Script
#
# This script regenerates ONLY the 03_final_dataset/ folders from existing
# 02_preprocessed/ data. Use this to rebuild .npz files with different
# parameters (e.g., changing dist_cutoff from 5.0 to 8.0).
#
# Usage: sbatch bash/regenerate_final_datasets.sh
###############################################################################

set -e

# --- Environment Setup ---
module purge
module load Anaconda3
source activate /data/stat-cadd/wolf7055/conda/envs/prism_backup
echo "Python executable: $(which python)"
echo ""

# Configuration
PROJECT_ROOT="/data/stat-cadd/wolf7055/PRISM"
DATA_DIR="${PROJECT_ROOT}/data"
CREATE_DATASET_SCRIPT="${PROJECT_ROOT}/data/preprocessing/create_dataset.py"

# Key parameter - change this to rebuild with different cutoff
DIST_CUTOFF=8.0

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Regenerating Final Datasets${NC}"
echo -e "${BLUE}  Distance Cutoff: ${DIST_CUTOFF}Å${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo ""

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Find datasets with existing preprocessed data
PREPROCESSED_DIRS=$(find ${DATA_DIR} -maxdepth 2 -type d -name "02_preprocessed" | grep -v "preprocessing" | grep -v "old_data")

if [ -z "$PREPROCESSED_DIRS" ]; then
    echo -e "${RED}No 02_preprocessed directories found in ${DATA_DIR}${NC}"
    exit 1
fi

DATASET_COUNT=$(echo "$PREPROCESSED_DIRS" | wc -l | tr -d ' ')
echo -e "${BLUE}Found ${DATASET_COUNT} datasets with preprocessed data${NC}"
echo ""

# List datasets
echo -e "${YELLOW}Datasets to regenerate:${NC}"
for PREPROC_DIR in $PREPROCESSED_DIRS; do
    DATASET_NAME=$(basename $(dirname "$PREPROC_DIR"))
    echo "  - ${DATASET_NAME}"
done
echo ""

# Process each dataset
CURRENT=0
SUCCESSFUL=0
FAILED=0

for PREPROC_DIR in $PREPROCESSED_DIRS; do
    CURRENT=$((CURRENT + 1))
    
    DATASET_DIR=$(dirname "$PREPROC_DIR")
    DATASET_NAME=$(basename "$DATASET_DIR")
    OUTPUT_DIR="${DATASET_DIR}/03_final_dataset"
    SPLIT_FILE="all_data.txt"
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}[${CURRENT}/${DATASET_COUNT}] Regenerating: ${DATASET_NAME}${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo "Input: ${PREPROC_DIR}"
    echo "Output: ${OUTPUT_DIR}"
    echo "Time: $(date)"
    echo ""
    
    # Check that required files exist
    if [ ! -f "${PREPROC_DIR}/${SPLIT_FILE}" ]; then
        echo -e "${RED}Missing ${SPLIT_FILE} in ${PREPROC_DIR}. Skipping.${NC}"
        FAILED=$((FAILED + 1))
        continue
    fi
    
    # Remove only the final dataset folder
    if [ -d "${OUTPUT_DIR}" ]; then
        echo -e "${YELLOW}Removing existing 03_final_dataset/...${NC}"
        rm -rf "${OUTPUT_DIR}"
    fi
    
    # Run create_dataset.py
    echo -e "${YELLOW}Running create_dataset.py with dist_cutoff=${DIST_CUTOFF}...${NC}"
    CMD="python ${CREATE_DATASET_SCRIPT} -i ${PREPROC_DIR} -o ${OUTPUT_DIR} --split_file ${SPLIT_FILE} --dist_cutoff ${DIST_CUTOFF}"
    echo "Command: ${CMD}"
    echo ""
    
    if ${CMD}; then
        echo ""
        echo -e "${GREEN}✓ Successfully regenerated ${DATASET_NAME}${NC}"
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        echo ""
        echo -e "${RED}✗ Failed to regenerate ${DATASET_NAME}${NC}"
        FAILED=$((FAILED + 1))
    fi
    
    echo ""
done

# Final summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Regeneration Complete${NC}"
echo -e "${BLUE}========================================${NC}"
echo "End time: $(date)"
echo "Distance cutoff used: ${DIST_CUTOFF}Å"
echo ""
echo "Total datasets: ${DATASET_COUNT}"
echo -e "${GREEN}Successful: ${SUCCESSFUL}${NC}"
if [ ${FAILED} -gt 0 ]; then
    echo -e "${RED}Failed: ${FAILED}${NC}"
fi
echo ""

echo "Job completed: ${SLURM_JOB_ID}"