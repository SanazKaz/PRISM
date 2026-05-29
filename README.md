# PRISM: Policy-Reinforced Iterative Structure-Based Molecular Diffusion

A reinforcement learning framework for structure-based _de novo_ diffusion models

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
pip install torch-scatter==2.1.2 torch-cluster==1.6.3 torch-geometric==2.7.0 \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

For Mac (Metal/MPS):

```bash
# torch already installed via environment.yml
pip install torch-scatter==2.1.2 torch-cluster==1.6.3 torch-geometric==2.7.0 \
    -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
```

#### Option 2: pip with toml (CPU or custom CUDA setup)

```bash
# Create and activate virtual environment
python3.12 -m venv prism_env
source prism_env/bin/activate  # On Windows: prism_env\Scripts\activate

# Install base package
pip install -e .

# Install PyG packages (required, version-specific — not in pyproject.toml)
pip install torch-scatter==2.1.2 torch-cluster==1.6.3 torch-geometric==2.7.0 \
    -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
```

#### Option 3: uv (Fast alternative to pip)

```bash
# Create and activate environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install base package
uv pip install -e .

# Install PyG packages (required, version-specific — not in pyproject.toml)
uv pip install torch-scatter==2.1.2 torch-cluster==1.6.3 torch-geometric==2.7.0 \
    -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
```

For GPU support with uv:

```bash
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install torch-scatter==2.1.2 torch-cluster==1.6.3 torch-geometric==2.7.0 \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

#### Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
python -c "import torch_scatter; print('torch-scatter imported successfully')"
python -c "import torch_geometric; print('torch-geometric imported successfully')"
python -c "import prism; print('PRISM imported successfully')"

# Run unit tests
python -m pytest tests/unit/ -v
```

---

## Data Preparation

PRISM uses paired **binding-pocket PDB** + **ligand SDF** files that are
pre-processed into compressed `.npz` arrays. The pipeline has three steps:

```
Step 1  PDB IDs ──► download .pdb/.cif files    (01_raw_pdbs/)
Step 2  .pdb files ──► pocket .pdb + ligand .sdf (02_preprocessed/)
Step 3  pocket/sdf pairs ──► train/val/test .npz  (03_final_dataset/)
```

All three steps are run by a single script:

```
scripts/process_data.py
```

### Option A — From a list of PDB IDs (downloads from RCSB)

Provide a plain-text file of PDB IDs (comma- or newline-separated). The
pipeline fetches the structures from RCSB automatically.

```bash
python -m scripts.process_data \
    --pdb_list data/example_pdbs.txt \
    --output_dir data/my_dataset
```

A sample list (`data/example_pdbs.txt`) with ten PDB IDs is included for
testing.

### Option B — From local PDB/CIF files you already have

If you have a directory of `.pdb` or `.cif` files on disk, skip the download
step with `--skip_fetch`:

```bash
python -m scripts.process_data \
    --skip_fetch \
    --pdb_dir /path/to/your/pdb_files \
    --output_dir data/my_dataset
```

### Option C — CrossDocked dataset (DiffSBDD or TargetDiff)

The CrossDocked2020 benchmark dataset used to train both models is distributed
by [Pocket2Mol](https://github.com/pengxingang/Pocket2Mol).

**Step 1 — Download from Pocket2Mol**

Follow the Pocket2Mol data instructions to download:

- `crossdocked_pocket10.tar.gz` — pocket PDB files
- `split_by_name.pt` — train/test split

**Step 2 — Extract**

```bash
tar -xzvf crossdocked_pocket10.tar.gz
# produces: crossdocked_pocket10/
```

**Step 3 — Process**

Use the unified `process_crossdock.py` script with `--model diffsbdd` or
`--model targetdiff`:

```bash
# DiffSBDD (10-dim element one-hots)
python -m scripts.process_crossdock \
    --crossdocked_dir /path/to/crossdocked_pocket10 \
    --split_path      /path/to/split_by_name.pt \
    --output_dir      data/crossdock_diffsbdd \
    --model           diffsbdd

# TargetDiff (27-dim element + AA + backbone features)
python -m scripts.process_crossdock \
    --crossdocked_dir /path/to/crossdocked_pocket10 \
    --split_path      /path/to/split_by_name.pt \
    --output_dir      data/crossdock_targetdiff \
    --model           targetdiff
```

Run `--smoke_test` first to verify shapes on 2 pairs before the full ~2 h job.

Point `datadir` in `configs/ppo_config.yaml` (DiffSBDD) or
`configs/targetdiff_ppo.yaml` (TargetDiff) at the output directory.

### Output structure

After the pipeline completes, `--output_dir` will contain:

```
my_dataset/
├── 01_raw_pdbs/              # Downloaded .pdb / .cif files (Option A only)
├── 02_preprocessed/
│   ├── pocket_files/         # Ligand-free pocket .pdb files
│   ├── sdf_files/            # Ligand .sdf files
│   ├── all_data.txt          # All extracted basename pairs
│   └── train_data.txt        # After test-set filtering
└── 03_final_dataset/
    ├── train.npz             # Training set
    ├── val.npz               # Validation set
    ├── test.npz              # Test set
    ├── train_smiles.npy      # Pre-computed SMILES for training ligands
    ├── size_distribution.npy # Joint ligand/pocket size histogram
    └── summary.txt           # Atom/AA histograms and dataset statistics
```

Set `datadir` in your training config to the `03_final_dataset/` path.

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--pdb_list` | — | Text file of PDB IDs to download |
| `--skip_fetch` | off | Use existing local PDB files instead |
| `--pdb_dir` | — | Directory of local PDB files (required with `--skip_fetch`) |
| `--output_dir` | `data/custom_<name>_data` | Root output directory |
| `--preprocess_distance` | `15.0` | Å cutoff for initial pocket extraction |
| `--dataset_distance` | `5.0` | Å cutoff for final pocket definition in .npz |
| `--test_pdbs` | `data/crossdocked_test_pdbs.txt` | Test PDB exclusion list; pass `none` to skip |
| `--keep_duplicates` | off | Keep all chain instances of the same ligand |
| `--include_common` | off | Include crystallographic additives (skips block list) |
| `--dataset_info_key` | `crossdock_full` | Atom encoder key from `constants.py` |

### Reference files in `data/`

| File | Purpose |
|------|---------|
| `example_pdbs.txt` | 10 PDB IDs for a quick test run |
| `crossdocked_train_pdbs.txt` | Full CrossDocked training PDB list |
| `crossdocked_test_pdbs.txt` | CrossDocked test set (excluded from training) |
| `pdb_block_list.txt` | Crystallographic additives / solvents to ignore |

---

## Generating Ligands

### Single pocket

Two scripts handle single-pocket generation — one per model architecture.
Both accept a checkpoint, a config, and a pocket PDB file.

```bash
# DiffSBDD
python -m scripts.generate_diffsbdd \
    checkpoints/crossdocked_fullatom_cond.ckpt \
    --config configs/ppo_config.yaml \
    --pdbfile path/to/pocket.pdb \
    --outfile results/generated.sdf \
    --n_samples 100 --batch_size 25 --sanitize

# TargetDiff
python -m scripts.generate_targetdiff \
    checkpoints/targetdiff.pt \
    --config configs/targetdiff_ppo.yaml \
    --pdbfile path/to/pocket.pdb \
    --outfile results/generated.sdf \
    --n_samples 100
```

Obtain the original DiffSBDD checkpoint:

```bash
wget -P checkpoints/ https://zenodo.org/record/8183747/files/crossdocked_fullatom_cond.ckpt
```

### Test set / custom target set

`scripts/test.py` generates ligands for a directory of pockets and works
with both models via `--model`:

```bash
python -m scripts.test \
    checkpoints/my_run.ckpt \
    --model diffsbdd \          # or targetdiff
    --config configs/ppo_config.yaml \
    --test_dir /path/to/test_pockets \
    --outdir results/test \
    --n_samples 100 --batch_size 120
```

Expected layout under `--test_dir`:

```
test_pockets/
├── <stem>.sdf        reference ligand (used as output name key)
├── <pdb_id>.pdb      pocket structure
└── <stem>.txt        residue list (DiffSBDD only; optional)
```

For TargetDiff the `.txt` file is not needed. Any directory of `.pdb` + `.sdf`
pairs works — including a custom set of a few targets.

### Held-out evaluation targets

`scripts/test_targets.py` runs generation over a fixed set of 6 proteins
(18 structures) and also supports `--model diffsbdd|targetdiff`:

```bash
python -m scripts.test_targets \
    checkpoints/my_run.ckpt \
    --model diffsbdd \
    --config configs/ppo_config.yaml \
    --targets_dir /data/targets \
    --outdir results/eval_targets \
    --n_samples 10000

# Single target (useful for parallel submission)
python -m scripts.test_targets ... --target BRD4_BD1_4whw
```

---

## Training

Train PRISM with reinforcement learning:

```bash
python scripts/train.py \
    --config configs/ppo_config.yaml \
    --warm_start_from_ddpm checkpoints/crossdocked_fullatom_cond.ckpt \
    --seed 42
```

### Configuration

Training is controlled through `configs/ppo_config.yaml`. Key parameters:

**Top-level settings:**

| Key | Description |
|-----|-------------|
| `run_identifier` | Experiment name for logging |
| `datadir` | Path to the `03_final_dataset/` directory |
| `batch_size` | Number of protein pockets per data-loading batch |
| `freeze_except` | List of EGNN blocks to unfreeze (e.g. `['e_block_3', 'e_block_4']`) |

**`model:` section:**

| Key | Description |
|-----|-------------|
| `total_timesteps` | Total diffusion timesteps (must match the pre-trained checkpoint) |

**`ppo:` section:**

| Key | Description |
|-----|-------------|
| `num_outer_epochs` | Number of PPO training cycles |
| `num_inner_epochs` | PPO gradient updates per rollout |
| `n_steps` | Molecules generated per rollout |
| `batch_size` | Mini-batch size for PPO updates |
| `clip_range` | PPO clipping parameter (typically `0.1`) |
| `entropy_coef` | Entropy bonus coefficient |
| `lr` | Optimizer learning rate |
| `train_timesteps` | Diffusion timesteps sampled during training |
| `gradient_accumulation_steps` | Steps before an optimizer update |
| `target_kl` | Early-stop KL threshold per inner epoch |
| `num_nodes_lig` | Expected ligand size for rollout sampling |

**`reward_params:` section:**

Set a weight > 0 to enable a reward component. All weights should sum to 1.

| Key | Description |
|-----|-------------|
| `smina_docking` | Docking score (Smina/Gnina) |
| `custom_qed` | Drug-likeness (QED) |
| `custom_sa_score` | Synthetic accessibility |
| `geometry_checks` | 3-D geometry validity |
| `feature_density` | Hotspot pharmacophore overlap |
| `property_2d` | 2-D molecular property matching |

---

## Project Structure

```
.
├── configs/                     # YAML configuration files
│   ├── ppo_config.yaml          # Main training config
│   ├── ablations/               # Ablation study configs
│   └── exp_specific/            # Experiment-specific configs
├── checkpoints/                 # Model checkpoints
├── data/
│   ├── example_pdbs.txt         # 10-entry PDB list for quick tests
│   ├── crossdocked_train_pdbs.txt
│   ├── crossdocked_test_pdbs.txt
│   └── pdb_block_list.txt
├── scripts/
│   ├── process_data.py                 # Data pipeline: PDB IDs → NPZ (custom datasets)
│   ├── process_crossdock.py            # CrossDocked pocket10 → NPZ (--model diffsbdd|targetdiff)
│   ├── process_crossdock_targetdiff.py # Legacy: TargetDiff CrossDocked processor
│   ├── train.py                        # PPO training entry point
│   ├── generate_diffsbdd.py            # Single-pocket inference (DiffSBDD)
│   ├── generate_targetdiff.py          # Single-pocket inference (TargetDiff)
│   ├── test.py                         # Test-set generation (--model diffsbdd|targetdiff)
│   └── test_targets.py                 # Eval-target generation (--model diffsbdd|targetdiff)
├── src/
│   ├── models/diffsbdd/         # DiffSBDD diffusion model (vendored)
│   ├── models/targetdiff/       # TargetDiff diffusion model (vendored)
│   └── prism/
│       ├── data_processing/     # fetch_pdbs, preprocess_data, create_dataset
│       ├── data_modules/        # PyTorch Lightning DataModule + Dataset
│       ├── models/              # Policy wrappers + targetdiff_inference helpers
│       ├── ppo_tuner/           # PPO algorithm, rollout, loss, Lightning module
│       └── reward/              # Reward function implementations
└── results/                     # Generation outputs
```
