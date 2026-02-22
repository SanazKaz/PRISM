#!/bin/bash
# =============================================================================
# Pharmacophore Hotspot Generation Pipeline
# =============================================================================
# This script processes all 6 targets through:
#   1. PyMOL preprocessing (align, clean, extract ligands)
#   2. DBSCAN hotspot generation (cluster features, generate pkl + figures)
#
# Usage (in interactive node):
#   bash run_hotspot_pipeline.sh
#
# Or to run a single target:
#   bash run_hotspot_pipeline.sh BRD4_BD1
# =============================================================================

set -e  # Exit on error

# --- Configuration ---
BASE_DIR="/data/stat-cadd/wolf7055/PRISM/data"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# All targets
ALL_TARGETS=(
    "BRD4_BD1"
    "Carb_Anh_II"
    "EGFR"
    "Estrogen_recep_alpha"
    "Factor_Xa"
    "HIV_1_Protease"
)

# --- Environment Setup ---
echo "=============================================="
echo "Pharmacophore Hotspot Generation Pipeline"
echo "=============================================="
echo ""

# Activate conda environment (adjust if needed)
if command -v conda &> /dev/null; then
    echo "Activating conda environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate /data/stat-cadd/wolf7055/conda/envs/PRISM_25
fi

echo "Python: $(which python)"
echo "Base directory: ${BASE_DIR}"
echo ""

# --- Determine which targets to process ---
if [ $# -eq 0 ]; then
    # No arguments - process all targets
    TARGETS=("${ALL_TARGETS[@]}")
    echo "Processing ALL targets: ${TARGETS[*]}"
else
    # Process only specified targets
    TARGETS=("$@")
    echo "Processing specified targets: ${TARGETS[*]}"
fi

echo ""

# --- Main Processing Loop ---
for TARGET in "${TARGETS[@]}"; do
    echo ""
    echo "######################################################################"
    echo "# Processing: ${TARGET}"
    echo "######################################################################"
    echo ""
    
    TARGET_DIR="${BASE_DIR}/${TARGET}"
    
    # Check target directory exists
    if [ ! -d "${TARGET_DIR}" ]; then
        echo "ERROR: Target directory not found: ${TARGET_DIR}"
        echo "Skipping ${TARGET}..."
        continue
    fi
    
    # Check required input directories
    RAW_PDB_DIR="${TARGET_DIR}/01_raw_pdbs"
    SDF_DIR="${TARGET_DIR}/02_preprocessed/sdf_files"
    
    if [ ! -d "${RAW_PDB_DIR}" ]; then
        echo "ERROR: Raw PDB directory not found: ${RAW_PDB_DIR}"
        echo "Skipping ${TARGET}..."
        continue
    fi
    
    if [ ! -d "${SDF_DIR}" ]; then
        echo "ERROR: SDF directory not found: ${SDF_DIR}"
        echo "Skipping ${TARGET}..."
        continue
    fi
    
    # Output directories
    PROTEIN_OUT="${TARGET_DIR}/04_aligned_clean_proteins"
    LIGAND_OUT="${TARGET_DIR}/05_ligand_only_within_8A"
    ALIGNED_SDF="${TARGET_DIR}/FEATURE_MAP_ALIGNED"
    HOTSPOT_OUT="${TARGET_DIR}/hotspot_analysis"
    
    # =========================================================================
    # Step 1: PyMOL Preprocessing
    # =========================================================================
    echo ""
    echo "--- Step 1: PyMOL Preprocessing ---"
    echo "Input:  ${RAW_PDB_DIR}"
    echo "Output: ${ALIGNED_SDF}"
    echo ""
    
    # Check if already processed
    if [ -d "${ALIGNED_SDF}" ] && [ "$(ls -A ${ALIGNED_SDF} 2>/dev/null)" ]; then
        echo "FEATURE_MAP_ALIGNED already exists with files. Skipping preprocessing..."
        echo "(Delete ${ALIGNED_SDF} to re-run)"
    else
        echo "Running PyMOL preprocessing..."
        python "${SCRIPT_DIR}/preprocess_pdbs.py" \
            --target_dir "${TARGET_DIR}" \
            --raw_pdb_subdir "01_raw_pdbs" \
            --sdf_subdir "02_preprocessed/sdf_files" \
            --protein_out_subdir "04_aligned_clean_proteins" \
            --ligand_out_subdir "05_ligand_only_within_8A" \
            --final_sdf_subdir "FEATURE_MAP_ALIGNED"
        
        echo "PyMOL preprocessing complete."
    fi
    
    # =========================================================================
    # Step 2: DBSCAN Hotspot Generation
    # =========================================================================
    echo ""
    echo "--- Step 2: DBSCAN Hotspot Generation ---"
    echo "Input:  ${ALIGNED_SDF}"
    echo "Output: ${HOTSPOT_OUT}"
    echo ""
    
    # Check if aligned SDFs exist
    if [ ! -d "${ALIGNED_SDF}" ] || [ -z "$(ls -A ${ALIGNED_SDF} 2>/dev/null)" ]; then
        echo "ERROR: No aligned SDFs found in ${ALIGNED_SDF}"
        echo "Skipping hotspot generation for ${TARGET}..."
        continue
    fi
    
    # Create output directory
    mkdir -p "${HOTSPOT_OUT}"
    
    echo "Running DBSCAN hotspot generation..."
    python "${SCRIPT_DIR}/generate_hotspots.py" \
        --sdf_dir "${ALIGNED_SDF}" \
        --output_dir "${HOTSPOT_OUT}" \
        --target_name "${TARGET}" \
        --eps 0.5 \
        --min_samples 10 \
        --min_count 10
    
    echo "Hotspot generation complete."
    
    # =========================================================================
    # Summary for this target
    # =========================================================================
    echo ""
    echo "--- ${TARGET} Complete ---"
    echo "Outputs:"
    echo "  - Aligned proteins: ${PROTEIN_OUT}"
    echo "  - Filtered ligands: ${LIGAND_OUT}"
    echo "  - Aligned SDFs:     ${ALIGNED_SDF}"
    echo "  - Hotspot data:     ${HOTSPOT_OUT}/${TARGET}_hotspot_data.pkl"
    echo "  - Summary plot:     ${HOTSPOT_OUT}/${TARGET}_hotspot_summary.png"
    echo "  - 3D features plot: ${HOTSPOT_OUT}/${TARGET}_features_3d.png"
    echo ""
    
done

# --- Final Summary ---
echo ""
echo "######################################################################"
echo "# Pipeline Complete!"
echo "######################################################################"
echo ""
echo "Processed targets: ${TARGETS[*]}"
echo ""
echo "To view results:"
echo "  ls -la ${BASE_DIR}/*/hotspot_analysis/"
echo ""