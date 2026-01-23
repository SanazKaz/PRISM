#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --partition=devel 
#SBATCH --gres=gpu:1
#SBATCH --time 00:10:00
#SBATCH --job-name=cd_posebusters_mols
#SBATCH --mail-user=wolf7055@ox.ac.uk
#SBATCH --output=jobs_files/Single_Objective_updated_CD_Posebusters_Training.log
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
'/data/stat-cadd/wolf7055/PRISM/Log_Results/Single_Objective_updated_CD_Posebusters_Training/checkpoints/BRD4_BD1/seed=789/epoch=13-reward=0.81.pt' \
--config /data/stat-cadd/wolf7055/PRISM/configs/ppo_config.yaml \
--pdbfile /data/stat-cadd/wolf7055/PRISM/data/BRD4_BD1/02_preprocessed/pocket_files/6fo5_DZH_B_201_pocket.pdb \
--ref_ligand /data/stat-cadd/wolf7055/PRISM/data/BRD4_BD1/02_preprocessed/sdf_files/6fo5_DZH_B_201.sdf \
--n_samples 100 \
--num_nodes_lig 30 \
--batch_size 50 \
--sanitize \
--outfile /data/stat-cadd/wolf7055/PRISM/Generated_Mols/Single_Objective_updated_CD_Posebusters_Training_789.sdf

echo "Generation completed, stopping resource monitoring..."

echo "Final Memory Usage:"
free -h
echo "Final GPU Usage:"
nvidia-smi