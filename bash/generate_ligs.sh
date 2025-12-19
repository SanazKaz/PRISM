#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --partition=short 
#SBATCH --gres=gpu:h100:1
#SBATCH --time 00:30:00
#SBATCH --job-name=multi_protein_gen
#SBATCH --mail-user=wolf7055@ox.ac.uk
#SBATCH --output=jobs_files/GEN_FILE.log
exec 2>&1

module purge
module load Anaconda3
source activate /data/stat-cadd/wolf7055/conda/envs/TEST_ENV
which python
chmod +x /data/stat-cadd/wolf7055/PRISM/val_analysis/smina.static

# Define run name
RUN_NAME="PB_Final_Run_0.5_0.5_to_0.7_0.3"
BASE_OUTPUT_DIR="/data/stat-cadd/wolf7055/PRISM/Generated_Mols/${RUN_NAME}"

# Define models with their corresponding protein data
declare -A MODELS=(
    ["Carb_Anh_II_seed42"]="/data/stat-cadd/wolf7055/PRISM/Log_Results/PB_Final_Run_0.5_0.5_to_0.7_0.3/checkpoints/Carb_Anh_II/seed=42/epoch=33-reward=1.23.pt"
    ["covid19_main_protease_seed123"]="/data/stat-cadd/wolf7055/PRISM/Log_Results/PB_Final_Run_0.5_0.5_to_0.7_0.3/checkpoints/covid19_main_protease/seed=123/epoch=31-reward=1.18.pt"
    ["EGFR_seed42"]="/data/stat-cadd/wolf7055/PRISM/Log_Results/PB_Final_Run_0.5_0.5_to_0.7_0.3/checkpoints/EGFR/seed=42/epoch=34-reward=1.20.pt"
    ["Estrogen_recep_alpha_seed789"]="/data/stat-cadd/wolf7055/PRISM/Log_Results/PB_Final_Run_0.5_0.5_to_0.7_0.3/checkpoints/Estrogen_recep_alpha/seed=789/epoch=34-reward=1.23.pt"
)

# Protein-specific data paths (YOU NEED TO FILL THESE IN!)
declare -A PROTEIN_PDBS=(
    ["Carb_Anh_II"]="/data/stat-cadd/wolf7055/PRISM/data/Carb_Anh_II/02_preprocessed/pocket_files/1bcd_FMS_C_500_pocket.pdb"
    ["covid19_main_protease"]="/data/stat-cadd/wolf7055/PRISM/data/covid19_main_protease/02_preprocessed/pocket_files/5r7y_JFM_B_1001_pocket.pdb"
    ["EGFR"]="/data/stat-cadd/wolf7055/PRISM/data/EGFR/02_preprocessed/pocket_files/1m17_AQ4_B_999_pocket.pdb"
    ["Estrogen_recep_alpha"]="/data/stat-cadd/wolf7055/PRISM/data/Estrogen_recep_alpha/02_preprocessed/pocket_files/1a52_AU_D_2_pocket.pdb"
)

declare -A PROTEIN_REF_LIGS=(
    ["Carb_Anh_II"]="/data/stat-cadd/wolf7055/PRISM/data/Carb_Anh_II/02_preprocessed/sdf_files/1bcd_FMS_C_500.sdf"
    ["covid19_main_protease"]="/data/stat-cadd/wolf7055/PRISM/data/covid19_main_protease/02_preprocessed/sdf_files/5r7y_JFM_B_1001.sdf"
    ["EGFR"]="/data/stat-cadd/wolf7055/PRISM/data/EGFR/02_preprocessed/sdf_files/1m17_AQ4_B_999.sdf"
    ["Estrogen_recep_alpha"]="/data/stat-cadd/wolf7055/PRISM/data/Estrogen_recep_alpha/02_preprocessed/sdf_files/1a52_AU_D_2.sdf"
)

# Print initial resource usage
echo "Initial Memory Usage:"
free -h
echo "Initial GPU Usage:"
nvidia-smi

echo "Starting generation for ${#MODELS[@]} models..."

# Iterate through each model
for MODEL_KEY in "${!MODELS[@]}"; do
    MODEL_PATH="${MODELS[$MODEL_KEY]}"
    
    # Extract protein name and seed
    PROTEIN_NAME=$(echo "$MODEL_KEY" | sed 's/_seed[0-9]*//')
    SEED=$(echo "$MODEL_KEY" | grep -oP 'seed\K\d+')
    
    # Create output directory
    PROTEIN_DIR="${BASE_OUTPUT_DIR}/${PROTEIN_NAME}"
    SEED_DIR="${PROTEIN_DIR}/seed_${SEED}"
    mkdir -p "${SEED_DIR}"
    
    # Define output file path
    OUTFILE="${SEED_DIR}/generated_molecules.sdf"
    
    # Get protein-specific paths
    PDBFILE="${PROTEIN_PDBS[$PROTEIN_NAME]}"
    REF_LIGAND="${PROTEIN_REF_LIGS[$PROTEIN_NAME]}"
    
    echo "========================================="
    echo "Protein: ${PROTEIN_NAME}"
    echo "Seed: ${SEED}"
    echo "Model: ${MODEL_PATH}"
    echo "PDB: ${PDBFILE}"
    echo "Ref Ligand: ${REF_LIGAND}"
    echo "Output: ${OUTFILE}"
    echo "========================================="
    
    python scripts/generate_ligands.py \
        "${MODEL_PATH}" \
        --config /data/stat-cadd/wolf7055/PRISM/configs/ppo_config.yaml \
        --pdbfile "${PDBFILE}" \
        --ref_ligand "${REF_LIGAND}" \
        --n_samples 100 \
        --num_nodes_lig 25 \
        --batch_size 50 \
        --sanitize \
        --outfile "${OUTFILE}"
    
    echo "Completed generation for ${PROTEIN_NAME} seed ${SEED}"
    echo ""
done

echo "Generation completed for all models!"
echo "Output directory: ${BASE_OUTPUT_DIR}"

echo "Final Memory Usage:"
free -h
echo "Final GPU Usage:"
nvidia-smi
```

**You need to fill in the actual PDB and SDF file names for each protein!** The structure will be:
```
Generated_Mols/
└── PB_Final_Run_0.5_0.5_to_0.7_0.3/
    ├── Carb_Anh_II/
    │   └── seed_42/
    │       └── generated_molecules.sdf
    ├── covid19_main_protease/
    │   └── seed_123/
    │       └── generated_molecules.sdf
    ├── EGFR/
    │   └── seed_42/
    │       └── generated_molecules.sdf
    └── Estrogen_recep_alpha/
        └── seed_789/
            └── generated_molecules.sdf