#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --partition=short 
#SBATCH --gres=gpu:h100:1
#SBATCH --time 6:00:00
#SBATCH --job-name=Diffsbdd_150x150_gen
#SBATCH --mail-user=wolf7055@ox.ac.uk
#SBATCH --output=jobs_files/diffsbdd_150x150_gen.log
# Redirect stderr to stdout
exec 2>&1
# Clear pre-loaded modules to ensure clean state // --constraint='gpu_mem:24GB'#SBATCH --gres=gpu:h100:1#SBATCH --array=0-7--constraint=gpu_sku:H100


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


python src/models/diffsbdd/generate_ligands.py \
    checkpoints/crossdocked_fa_cond_temp.ckpt \
    --pdbfile src/models/diffsbdd/data/drd2_strucutres/7e2z.pdb \
    --outfile results/diffsbdd_150x150_gen/7e2z_gen_traj_diffsbdd.sdf \
    --ref_ligand src/models/diffsbdd/data/drd2_strucutres/7e2z_9sc.sdf \
    --n_samples 1 \
    --timesteps 500 \
    --num_nodes_lig 32 \
    --save_traj \

# python test.py \
#     'Log_Results/Crossdock_QED(1.0)_lr3e-5/checkpoints/ppo-epoch=37-train_reward_mean_epoch=0.00-v1.ckpt' \
#     --test_dir /data/stat-cadd/wolf7055/diffsbdd-ppo/data/processed_crossdock_noH_full_temp/test \
#     --outdir Results/cd2020_sampling_results_no_relax_qed \
#     --n_samples 100 \
#     --batch_size 64 \
#     --sanitize \
#     --skip_existing

echo "Generation completed, stopping resource monitoring..."

echo "Final Memory Usage:"
free -h
echo "Final GPU Usage:"
nvidia-smi