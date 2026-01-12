#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --partition=devel 
#SBATCH --gres=gpu:h100:1
#SBATCH --time 03:00:00
#SBATCH --job-name=aromatic_bonus_mols
#SBATCH --mail-user=wolf7055@ox.ac.uk
#SBATCH --output=jobs_files/GEN_FILE_aromatic_bonus_mols.log
# Redirect stderr to stdout
exec 2>&1
# Clear pre-loaded modules to ensure clean state // --constraint='gpu_mem:24GB'#SBATCH --gres=gpu:h100:1#SBATCH --array=0-7--constraint=gpu_sku:H100


module purge
module load Anaconda3
# Load required module
source activate /data/stat-cadd/wolf7055/conda/envs/TEST_ENV
which python
chmod +x /data/stat-cadd/wolf7055/PRISM/val_analysis/smina.static # make exc # SBATCH --mem=GB

    

# Print initial resource usage for logging
echo "Initial Memory Usage:"
free -h
echo "Initial GPU Usage:"
nvidia-smi

echo "Starting generation..."


python scripts/generate_ligands.py \
'/data/stat-cadd/wolf7055/PRISM/Log_Results/aromatic_bonus_trial_ampc_from_PB_Final_Run/checkpoints/tmp/seed=42/epoch=60-reward=0.39.ckpt' \
--config /data/stat-cadd/wolf7055/PRISM/configs/ppo_config.yaml \
--pdbfile /data/stat-cadd/wolf7055/PRISM/data/AMPC_beta_lactamase/02_preprocessed/pocket_files/1pi4_SM3_E_401_pocket.pdb \
--ref_ligand /data/stat-cadd/wolf7055/PRISM/data/AMPC_beta_lactamase/02_preprocessed/sdf_files/1pi4_SM3_E_401.sdf \
--n_samples 10100 \
--num_nodes_lig 25 \
--batch_size 50 \
--sanitize \
--outfile /data/stat-cadd/wolf7055/PRISM/Generated_Mols/Aromatic_Bonus_Mols.sdf

echo "Generation completed, stopping resource monitoring..."

echo "Final Memory Usage:"
free -h
echo "Final GPU Usage:"
nvidia-smi