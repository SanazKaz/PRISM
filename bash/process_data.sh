#!/bin/bash
#SBATCH --cpus-per-task=6
#SBATCH --mem=64GB
#SBATCH --partition=short
#SBATCH --time 02:00:00        # Increased to 2 hours for structural processing
#SBATCH --job-name=PRISM_All
#SBATCH --output=jobs_files/Process_All_Datasets_CIF_DEDUPLICATE.log

###############################################################################
# Process All Datasets Script - HPC Version
#
# This script automatically processes all protein datasets in the data/ folder
# that contain a pdb_list.txt file. For each dataset:
# 1. Cleans up existing processed data directories (01, 02, 03)
# 2. Runs the full data processing pipeline (fetch, preprocess, create dataset)
#
# Usage: sbatch bash/process_all_datasets_hpc.sh
###############################################################################

set -e  # Exit on any error

# --- Environment Setup ---
module purge
module load Anaconda3
source activate /data/stat-cadd/wolf7055/conda/envs/prism_backup
echo "Python executable: $(which python)"
echo ""

# Configuration
PROJECT_ROOT="/data/stat-cadd/wolf7055/PRISM"
DATA_DIR="${PROJECT_ROOT}/data"
PROCESS_SCRIPT="${PROJECT_ROOT}/scripts/process_data.py"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Processing All Protein Datasets${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo ""

# Set PYTHONPATH and change to base directory
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
echo "Working directory: $(pwd)"
echo ""

# --- Find Dataset Lists ---
# We look 2 levels deep (data/TARGET/pdb_list.txt) 
# We explicitly ignore "old_data" and "preprocessing" utility folders
PDB_LISTS=$(find ${DATA_DIR} -maxdepth 2 -name "pdb_list.txt" -type f | grep -v "preprocessing" | grep -v "old_data")

if [ -z "$PDB_LISTS" ]; then
    echo -e "${RED}No pdb_list.txt files found in ${DATA_DIR}${NC}"
    exit 1
fi

# Count datasets
DATASET_COUNT=$(echo "$PDB_LISTS" | wc -l | tr -d ' ')
echo -e "${BLUE}Found ${DATASET_COUNT} datasets to process${NC}"
echo ""

# List all datasets to be processed
echo -e "${YELLOW}Datasets queued for processing:${NC}"
for PDB_LIST in $PDB_LISTS; do
    DATASET_NAME=$(basename $(dirname "$PDB_LIST"))
    echo "  - ${DATASET_NAME}"
done
echo ""

# Process each dataset
CURRENT=0
SUCCESSFUL=0
FAILED=0

for PDB_LIST in $PDB_LISTS; do
    CURRENT=$((CURRENT + 1))
    
    # Get the parent directory (e.g., data/EGFR)
    DATASET_DIR=$(dirname "$PDB_LIST")
    DATASET_NAME=$(basename "$DATASET_DIR")
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}[${CURRENT}/${DATASET_COUNT}] Processing: ${DATASET_NAME}${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo "Dataset directory: ${DATASET_DIR}"
    echo "PDB list: ${PDB_LIST}"
    echo "Time: $(date)"
    echo ""
    
    # Clean up existing data directories (Overwrite mode)
    echo -e "${YELLOW}Cleaning up existing data...${NC}"
    
    if [ -d "${DATASET_DIR}/01_raw_pdbs" ]; then
        echo "  Removing 01_raw_pdbs/"
        rm -rf "${DATASET_DIR}/01_raw_pdbs"
    fi
    
    if [ -d "${DATASET_DIR}/02_preprocessed" ]; then
        echo "  Removing 02_preprocessed/"
        rm -rf "${DATASET_DIR}/02_preprocessed"
    fi
    
    if [ -d "${DATASET_DIR}/03_final_dataset" ]; then
        echo "  Removing 03_final_dataset/"
        rm -rf "${DATASET_DIR}/03_final_dataset"
    fi
    
    echo -e "${GREEN}✓ Cleanup complete${NC}"
    echo ""
    
    # Run the processing pipeline
    echo -e "${YELLOW}Running processing pipeline...${NC}"
    # Note: --deduplicate is False by default in the python script 
    echo "Command: python ${PROCESS_SCRIPT} --pdb_list ${PDB_LIST} -o ${DATASET_DIR}"
    echo ""
    
    if python "${PROCESS_SCRIPT}" --pdb_list "${PDB_LIST}" -o "${DATASET_DIR}"; then
        echo ""
        echo -e "${GREEN}✓ Successfully processed ${DATASET_NAME}${NC}"
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        echo ""
        echo -e "${RED}✗ Failed to process ${DATASET_NAME}${NC}"
        echo -e "${RED}Continuing with next dataset...${NC}"
        FAILED=$((FAILED + 1))
    fi
    
    echo ""
done

# Final summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Processing Complete${NC}"
echo -e "${BLUE}========================================${NC}"
echo "End time: $(date)"
echo ""
echo "Total datasets: ${DATASET_COUNT}"
echo -e "${GREEN}Successful: ${SUCCESSFUL}${NC}"
if [ ${FAILED} -gt 0 ]; then
    echo -e "${RED}Failed: ${FAILED}${NC}"
fi
echo ""

echo "Processed datasets results:"
for PDB_LIST in $PDB_LISTS; do
    DATASET_DIR=$(dirname "$PDB_LIST")
    DATASET_NAME=$(basename "$DATASET_DIR")
    
    # Check if final dataset folder actually contains data
    if [ -d "${DATASET_DIR}/03_final_dataset" ]; then
        echo -e "  ${GREEN}✓${NC} ${DATASET_NAME}"
    else
        echo -e "  ${RED}✗${NC} ${DATASET_NAME}"
    fi
done
echo ""

echo "Job completed: ${SLURM_JOB_ID}"