#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1  # node is auto as 1, then n tasks per node should match num of gpus #SBATCH --constraint=gpu_sku:H100
#SBATCH --partition=short
#SBATCH --time 04:00:00
#SBATCH --job-name=Crossdock_QED(1.0)_lr3e-5_resume
#SBATCH --mail-user=wolf7055@ox.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-2
#SBATCH --output=jobs_files/Crossdock_QED(1.0)_lr3e-5_resume-%A_%a.log 
# Redirect stderr to stdout
exec 2>&1 


module purge
module load Anaconda3
# Load required module
source activate /data/stat-cadd/wolf7055/conda/envs/TEST_ENV
which python
chmod +x analysis/smina.static # make exc

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
srun python /data/stat-cadd/wolf7055/diffsbdd-ppo/train.py \
--config "/data/stat-cadd/wolf7055/diffsbdd-ppo/configs/ppo_config.yaml" \
--resume "/data/stat-cadd/wolf7055/diffsbdd-ppo/Log_Results/Crossdock_QED(1.0)_lr3e-5/checkpoints/ppo-epoch=37-train_reward_mean_epoch=0.00-v1.ckpt" \
--seed $SEED
# After training completes, stop the monitoring processes
echo "Training completed, stopping resource monitoring..."


echo "Final Memory Usage:"
free -h
echo "Final GPU Usage:"
nvidia-smi
