#!/bin/bash
#SBATCH --cpus-per-task=6
#SBATCH --mem=64GB
#SBATCH --partition=short
#SBATCH --time 01:30:00
#SBATCH --job-name=PRISM_HIV1
#SBATCH --output=jobs_files/Process_HIV1_Protease.log

###############################################################################
# Process HIV-1 Protease Dataset Script - HPC Version
#
# This script processes the HIV-1 Protease dataset through the full pipeline:
# 1. Cleans up existing processed data directories (01, 02, 03)
# 2. Runs the full data processing pipeline (fetch, preprocess, create dataset)
#
# Usage: sbatch bash/process_hiv1_protease_hpc.sh
###############################################################################

set -e  # Exit on any error

# --- Environment Setup ---
module purge
module load Anaconda3
source activate /data/stat-cadd/wolf7055/conda/envs/PRISM_25
echo "Python executable: $(which python)"
echo ""

# Configuration
PROJECT_ROOT="/data/stat-cadd/wolf7055/PRISM"
DATASET_DIR="${PROJECT_ROOT}/data/HIV_1_Protease"
PDB_LIST="${DATASET_DIR}/pdb_list.txt"
PROCESS_SCRIPT="${PROJECT_ROOT}/scripts/process_data.py"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Processing HIV-1 Protease Dataset${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo ""

# Set PYTHONPATH and change to base directory
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
echo "Working directory: $(pwd)"
echo "Dataset directory: ${DATASET_DIR}"
echo ""

# Verify pdb_list.txt exists
if [ ! -f "$PDB_LIST" ]; then
    echo -e "${RED}Error: pdb_list.txt not found at ${PDB_LIST}${NC}"
    exit 1
fi

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

echo -e "${GREEN}Cleanup complete${NC}"
echo ""

# Run the processing pipeline
echo -e "${YELLOW}Running processing pipeline...${NC}"
echo "Command: python ${PROCESS_SCRIPT} --pdb_list ${PDB_LIST} -o ${DATASET_DIR}"
echo ""

if python "${PROCESS_SCRIPT}" --pdb_list "${PDB_LIST}" -o "${DATASET_DIR}"; then
    echo ""
    echo -e "${GREEN}Successfully processed HIV-1 Protease dataset${NC}"
else
    echo ""
    echo -e "${RED}Failed to process HIV-1 Protease dataset${NC}"
    exit 1
fi

# Final summary
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Processing Complete${NC}"
echo -e "${BLUE}========================================${NC}"
echo "End time: $(date)"
echo ""

# Check if final dataset was created
if [ -d "${DATASET_DIR}/03_final_dataset" ]; then
    echo -e "${GREEN}Final dataset created successfully at:${NC}"
    echo "  ${DATASET_DIR}/03_final_dataset"
else
    echo -e "${RED}Warning: 03_final_dataset directory not found${NC}"
fi
echo ""

echo "Job completed: ${SLURM_JOB_ID}"