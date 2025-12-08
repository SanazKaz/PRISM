# PRISM: Policy-Reinforced Iterative Structure-Based Molecular Diffusion

A reinforcement learning framework for structure-based molecular generation.

## Setup

### Environment Installation
```bash
# Create conda environment from yaml file
conda env create -f environment.yml
conda activate prism

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Data Processing

Provide a uniprot id or a list of pdbs in a text file and the processing script will fetch them, pre-process them and finally generate the dataset for use. The pockets and sdfs will be stored as well as the final dataset in data.

A sample pdb id list is provided in data 

```bash
python scripts/process_data.py \
    --input_dir data/pdb_list.txt \
    --output_dir data/Carbonic_Anhydrase_II 
```

## Generating Ligands
### TODO:  ignore for now - still to decouple from diffsbdd as it is their script.

obtain the diffusion model checkpoint from DiffSBDD using:
```
wget -P checkpoints/ https://zenodo.org/record/8183747/files/crossdocked_fullatom_cond.ckpt 
```
obtain our  model checkpoint from DiffSBDD using:

some zenodo link here when models are done

Generate molecules from a trained checkpoint using a protein structure:
```bash
python src/models/diffsbdd/generate_ligands.py \
    checkpoints/your_checkpoint.ckpt \
    --pdbfile path/to/protein.pdb \
    --outfile results/generated_ligands.sdf \
    --ref_ligand path/to/reference_ligand.sdf \
    --n_samples 100 \
    --timesteps 500 \
    --num_nodes_lig 32
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
    --warm_start_from_ddpm checkpoints/crossdocked_fa_cond_temp.ckpt \
    --seed 42
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

