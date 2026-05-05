# PRISM: Policy-Reinforced Iterative Structure-Based Molecular Diffusion

A reinforcement learning framework for structure-based molecular generation.

## Setup

### Environment Installation

Choose one of the following methods:

#### Option 1: Conda 

```bash
# Create and activate environment
conda env create -f environment.yml
conda activate prism
```

For GPU support (Linux/Windows with CUDA):

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

For Mac (Metal/MPS):

```bash
# torch already installed via environment.yml
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
```

#### Option 2: pip with toml (CPU or custom CUDA setup)

```bash
# Create and activate virtual environment
python3.12 -m venv prism_env
source prism_env/bin/activate  # On Windows: prism_env\Scripts\activate

# Install base package
pip install -e .

# Install torch-scatter (required, not in pyproject.toml due to build dependencies)
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
```

#### Option 3: uv (Fast alternative to pip)

```bash
# Create and activate environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install base package
uv pip install -e .

# Install torch-scatter
uv pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
```

For GPU support with uv:

```bash
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

#### Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
python -c "import torch_scatter; print('torch-scatter imported successfully')"
python -c "import prism; print('PRISM imported successfully')"
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

- **Lightning Checkpoints (`.ckpt`)**: Useful for evaluating models directly after training.
- **Clean Weights (`.pt`)**: Lightweight files containing only the model state dictionary (no optimizer/scheduler states).

### 1. Get Pre-trained Models

Obtain the original diffusion model checkpoint from DiffSBDD:

```bash
wget -P checkpoints/ https://zenodo.org/record/8183747/files/crossdocked_fullatom_cond.ckpt
```

obtain our  model checkpoint from DiffSBDD using:

```
some zenodo link here when models are done
```

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

