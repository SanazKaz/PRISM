# PRISM: Policy-Reinforced Iterative Structure-Based Molecular Diffusion

A reinforcement learning framework for structure-based molecular generation.

## Setup

### Environment Installation

- Need to edit the pyproject.toml to remove the torch, torch-scatter and pytorch-lightning dependencies after installing manually with cuda support.

```bash
# Create conda environment from pyproject.toml (Not tested)
conda env create -n prism python=3.12
conda activate prism
# install torch with cuda support
pip install torch==2.8.0 
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu129/
pip install pytorch-lightning
# Install repo
pip install -e .
```




```bash
# Create uv environment from pyproject.toml
uv venv
source .venv/bin/activate
# install torch with cuda support
uv pip install torch==2.8.0
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu129/
uv pip install pytorch-lightning
# Install repo
uv pip install -e .
```



```bash
# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```


## Data Processing

Provide a uniprot id or a list of pdbs in a text file and the processing script will fetch them, pre-process them and finally generate the dataset for use. The pockets and sdfs will be stored as well as the final dataset in data.

A sample pdb id list is provided in data 

```bash
python -m scripts.process_data \ 
    --pdb_list /home/alexi/Documents/PRISM/data/example_pdb_list.txt \ 
    --output_dir /home/alexi/Documents/PRISM/data/test

```
## Generating Ligands

We provide a standalone generation script that decouples inference from the training logic. This script automatically detects and supports both:
* **Lightning Checkpoints (`.ckpt`)**: Useful for evaluating models directly after training.
* **Clean Weights (`.pt`)**: Lightweight files containing only the model state dictionary (no optimizer/scheduler states).

### 1. Get Pre-trained Models

Obtain the original diffusion model checkpoint from DiffSBDD:
```bash
wget -P checkpoints/ https://zenodo.org/record/8183747/files/crossdocked_fullatom_cond.ckpt
```
obtain our  model checkpoint from DiffSBDD using:

    some zenodo link here when models are done

Generate molecules from a trained checkpoint using a protein structure:

```bash
python scripts/generate_ligands.py \ 
    checkpoints/crossdocked_fullatom_cond.ckpt \ 
    --config configs/ppo_config.yaml \ 
    --pdbfile data/test/02_preprocessed/pocket_files/1cil_ETS_C_263_pocket.pdb \ 
    --outfile data/test/results/generated_ligands.sdf \ 
    --ref_ligand data/test/02_preprocessed/sdf_files/1cil_ETS_C_263.sdf \ 
    --n_samples 100 \ 
    --batch_size 25 \ 
    --timesteps 500 \ 
    --num_nodes_lig 25 \ 
    --sanitize \ 
    --relax

```

### Configuration

Training is controlled through `configs/ppo_config.yaml`. Key parameters:

**Basic Settings:**
- `run_identifier`: Experiment name for logging
- `datadir`: Path to processed training data
- `batch_size`: Number of protein pockets to sample per batch
- `lr`: Learning rate for the diffusion model (typically 1e-6)

**PPO Parameters:**
- `num_outer_epochs`: Number of PPO training cycles (e.g., 30)
- `num_inner_epochs`: PPO updates per rollout (e.g., 2)
- `n_steps`: Number of molecules generated per rollout (e.g., 108)
- `ppo_batch_size`: Batch size for PPO updates
- `clip_range`: PPO clipping parameter (typically 0.1)
- `lr`: PPO optimizer learning rate (e.g., 4e-5)
- `num_train_timesteps`: Number of diffusion timesteps to train on (e.g., 150)

**Reward Functions:**
Configure reward weights in the `reward_params` section. Set weight to 1.0 to enable:
- `qed`: Drug-likeness (Quantitative Estimate of Drug-likeness)
- `sa_score`: Synthetic accessibility
- `lipinski`: Rule of five compliance
- `sucos`: Shape and pharmacophore similarity to reference
- `interaction_fingerprints`: Protein-ligand interaction matching

**Diffusion Model Parameters:**
The base diffusion model configuration (EGNN architecture, diffusion schedule) must be included as PRISM fine-tunes the pretrained diffusion model. These settings typically remain unchanged from the base checkpoint.

## Training

Train PRISM with reinforcement learning:
```bash
python scripts/train.py \ 
    --config configs/ppo_config.yaml \ 
    --warm_start_from_ddpm checkpoints/crossdocked_fullatom_cond.ckpt \ 
    --seed 42 \

```

The training configuration is managed through `configs/ppo_config.yaml`. Key settings include:
- Reward function specifications
- PPO hyperparameters
- Checkpoint frequency
- Batch sizes

### Multiple Seeds


## Project Structure
```
.
├── configs/              # Configuration files
├── checkpoints/          # Model checkpoints
├── data/                 # Processed datasets + processing code
├── scripts/              # Training and data processing scripts
├── src/models/diffsbdd/  # Core diffsbdd model code
├── src/prism/            # PPO code with lightning
└── results/              # Generation outputs
```

