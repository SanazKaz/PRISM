#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1  # node is auto as 1, then n tasks per node should match num of gpus #SBATCH --constraint=gpu_sku:H100
#SBATCH --partition=short
#SBATCH --time 04:00:00
#SBATCH --job-name=DBSCAN_eps_0.5_min_10_AMPC_centered
#SBATCH --mail-user=wolf7055@ox.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-2
#SBATCH --output=jobs_files/DBSCAN_eps_0.5_min_10_AMPC_centered-%A_%a.log 
# Redirect stderr to stdout
exec 2>&1 

# #SBATCH --gres=gpu:h100:1

module purge
module load Anaconda3
# Load required module
source activate /data/stat-cadd/wolf7055/conda/envs/prism_backup
which python
# chmod +x analysis/smina.static # make exc

# Define seeds array
SEEDS=(42 976 123)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "Array Task ID: $SLURM_ARRAY_TASK_ID, Using seed: $SEED"

# Print initial resource usage for logging
echo "Initial Memory Usage:"
free -h
echo "Initial GPU Usage:"
nvidia-smi

python - << 'PY'
import torch, os
print("CVD", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("avail", torch.cuda.is_available(), "count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("name0", torch.cuda.get_device_name(0))
PY


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_PPO=0

# Run the training script
echo "Starting training..."
srun python scripts/train.py \
--config "configs/ppo_config.yaml" \
--warm_start_from_ddpm "checkpoints/crossdocked_fa_cond_temp.ckpt" \
--seed $SEED
# After training completes, stop the monitoring processes
echo "Training completed, stopping resource monitoring..."


echo "Final Memory Usage:"
free -h
echo "Final GPU Usage:"
nvidia-smi
