#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --partition=devel 
#SBATCH --gres=gpu:1
#SBATCH --time 00:10:00
#SBATCH --job-name=ampc_1c3b_QED_SA_SCORE_gen
#SBATCH --mail-user=wolf7055@ox.ac.uk
#SBATCH --output=jobs_files/GEN_FILE.log
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
'/data/stat-cadd/wolf7055/PRISM/Log_Results/dbscan_aromatic_bonus/checkpoints/tmp/seed=42/epoch=31-reward=0.27.pt' \
--config /data/stat-cadd/wolf7055/PRISM/configs/ppo_config.yaml \
--pdbfile /data/stat-cadd/wolf7055/PRISM/data/AMPC_beta_lactamase/02_preprocessed/pocket_files/1c3b_BZB_C_362_pocket.pdb \
--ref_ligand /data/stat-cadd/wolf7055/PRISM/data/AMPC_beta_lactamase/02_preprocessed/sdf_files/1c3b_BZB_C_362.sdf \
--n_samples 100 \
--num_nodes_lig 25 \
--batch_size 50 \
--sanitize \
--outfile /data/stat-cadd/wolf7055/PRISM/Generated_Mols/dbscan_aromatic_bonus_epoch31_reward0.27_ampcs/Seed42.sdf



echo "Generation completed, stopping resource monitoring..."

echo "Final Memory Usage:"
free -h
echo "Final GPU Usage:"
nvidia-smi