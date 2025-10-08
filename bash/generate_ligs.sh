#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=short 
#SBATCH --time 03:00:00
#SBATCH --job-name=Crossdock_QED(1.0)_lr3e-5_generation
#SBATCH --mail-user=wolf7055@ox.ac.uk
#SBATCH --output=jobs_files/generation_CD_qed_1.0.log
# Redirect stderr to stdout
exec 2>&1
# Clear pre-loaded modules to ensure clean state // --constraint='gpu_mem:24GB'  #SBATCH --array=0-7--constraint=gpu_sku:H100


module purge
module load Anaconda3
# Load required module
source activate /data/stat-cadd/wolf7055/conda/envs/TEST_ENV
which python
chmod +x analysis/smina.static # make exc # SBATCH --mem=GB



# Print initial resource usage for logging
echo "Initial Memory Usage:"
free -h
echo "Initial GPU Usage:"
nvidia-smi

echo "Starting generation..."
# python generate_ligands.py \
#     Log_Results/brd4_SAS_QED_7mra_lr3e-5/checkpoints/ppo-epoch=68-train_reward_mean_epoch=0.00.ckpt \
# --pdbfile data/brd_structures/test/processed_ligand_free_pockets_test/7t2i_F_E9F_pocket_only.pdb\
# --outfile example/brd4_pdbs/brd4_sas_qed_e68_7t2i_F_E9F_25_nodes.sdf \
# --ref_ligand data/brd_structures/7t2i_F_E9F.sdf \
# --n_samples 50 \
# --timesteps 500 \
# --num_nodes_lig 25


python test.py \
    'Log_Results/Crossdock_QED(1.0)_lr3e-5/checkpoints/ppo-epoch=37-train_reward_mean_epoch=0.00-v1.ckpt' \
    --test_dir /data/stat-cadd/wolf7055/diffsbdd-ppo/data/processed_crossdock_noH_full_temp/test \
    --outdir Results/cd2020_sampling_results_no_relax_qed \
    --n_samples 100 \
    --batch_size 64 \
    --sanitize \
    --skip_existing

echo "Generation completed, stopping resource monitoring..."

echo "Final Memory Usage:"
free -h
echo "Final GPU Usage:"
nvidia-smi