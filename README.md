# PRISM: Policy-Reinforced Iterative Structure-Based Molecular Diffusion

Official implementation of PRISM: A Hybrid Diffusion-Reinforcement Learning Framework for 3D Structure-based De Novo Design. 

PRISM framework takes a pretrained **DiffSBDD** or **TargetDiff** checkpoint and tunes layers so generated
ligands optimise a chemistry reward while staying close to the pretrained prior.

## Workshop Paper
PRISM: A Hybrid Diffusion-Reinforcement Learning Framework for 3D Structure-based De Novo Design
Sanaz Kazeminia, Lewis R. Vidler, Pushkar G. Ghanekar, Nele P Quast, Garrett M Morris
[![Paper](https://img.shields.io/badge/OpenReview-TAeVNZ77sH-8C1B13)](https://openreview.net/forum?id=TAeVNZ77sH)

## Setup

```bash
conda env create -f environment.yml
conda activate PRISM_25
pip install -e .
```

`environment.yml` pins torch 2.6.0+cu124 and the CUDA-specific PyG wheels
(`torch-scatter`, `torch-cluster`, `torch-geometric`), so there is nothing else
to install.

On the group cluster the environment already exists:

```bash
module load Mamba && conda activate PRISM_25
```

Check it worked:

```bash
python -c "import torch, torch_scatter, torch_geometric, prism; print(torch.cuda.is_available())"
python -m pytest tests/unit -q
```

## Train

```bash
python scripts/train.py \
    --config configs/targetdiff/crossdocked/geometry.yaml \
    --warm_start_from_ddpm checkpoints/targetdiff_pretrained_models/targetdiff_pretrained_diffusion.pt \
    --seed 42
```

For DiffSBDD, swap in `configs/diffsbdd/ppo_config.yaml` and a DiffSBDD
checkpoint; the config's `model_type` selects the backbone (DiffSBDD is the
default when the key is absent).

These CLI flags override the YAML, which is how one config is reused across a
seed × target sweep: `--seed`, `--datadir`, `--logdir`, `--dataset_name`,
`--target_name`, `--hotspot_path`, `--resume_from_checkpoint`.

Checkpoints land in
`Log_Results/<logdir>/<run_identifier>/checkpoints/<dataset>/seed=<S>/`, saved as
both `.ckpt` (Lightning) and `.pt` (native backbone format — **this is the one
you generate with**). Top-3 by `train/reward_mean`, plus `last`.

### Config

One YAML per experiment, under `configs/<backbone>/`. Clone the closest existing
config and change only what the experiment is asking about.

| Section | What it controls |
|---------|------------------|
| top level | `run_identifier` (names the output dir), `datadir`, `gpus`, `freeze_except` (which blocks to unfreeze — everything else stays frozen to anchor the prior) |
| `model:` | checkpoint path, dims, `total_timesteps` (must match the pretrained checkpoint) |
| `ppo:` | `num_outer_epochs`, `num_inner_epochs`, `n_steps` (mols per rollout), `lr`, `clip_range`, `target_kl`, `ref_kl_coef`, `train_timesteps`, `timestep_window` |
| `reward_params:` | `aggregation` (`weighted_sum` or `product`), per-reward weights under `rewards:`, plus `reward_paths:` and `target_name` |

Rewards with weight `0.0` are inactive — only weights > 0 get instantiated. Under
`product` aggregation the weights act as exponents, not a partition of 1.

Commonly used reward keys (full list in `src/prism/reward/factory.py`):

| Key | What it scores |
|-----|----------------|
| `geometry_checks` | 3-D geometry validity (PoseBusters bounds) |
| `smina_docking` | Docking score via Smina |
| `feature_density` | Pharmacophore hotspot overlap — needs a hotspot `.pkl` |
| `property_2d` | 2-D property match to a reference binder set |
| `custom_qed`, `custom_sa_score` | Capped/normalised drug-likeness and synthesisability |
| `penalised_logp` | Penalised LogP |

To add one: implement `BaseReward` in `src/prism/reward/scoring/`, register it in
`factory.py`, give it a weight key.

## Generate

Three entry points, all taking a `.pt`/`.ckpt` checkpoint, the run's config, and
`--model diffsbdd|targetdiff`.

**Held-out evaluation targets** — the fixed 6-protein / 18-structure set:

```bash
python -m scripts.test_targets \
    <checkpoint> \
    --model targetdiff \
    --config configs/targetdiff/crossdocked/geometry.yaml \
    --targets_dir data \
    --outdir results/targetdiff/<run> \
    --n_samples 1000 --batch_size 84 --num_steps 1000

# one structure at a time, for parallel SLURM submission
python -m scripts.test_targets ... --target BRD4_BD1_4whw
```

Target keys are defined in `scripts/test_targets.py::_TARGET_SPECS`. Output is
`<outdir>/<TARGET>/<TARGET>_processed.sdf` plus a `_stats.txt`. Pass
`--targets_file <json>` instead of `--targets_dir` to use your own pocket/ligand
pairs.

**CrossDocked test set** — a directory of pocket/ligand pairs:

```bash
python -m scripts.test_crossdocked \
    <checkpoint> \
    --model targetdiff \
    --config configs/targetdiff/crossdocked/geometry.yaml \
    --test_dir data/cross_dock/.../test \
    --outdir results/targetdiff/<run> \
    --n_samples 100 --batch_size 84
```

The script pairs each `<stem>.sdf` reference ligand (which names the output) with
`<stem>_pocket.pdb` (what `process_crossdock.py` produces), falling back to
`<pdb_id>.pdb` for raw full-structure PDBs. DiffSBDD additionally accepts an
optional `<stem>.txt` residue list.

**A single pocket** — `scripts.generate_diffsbdd` / `scripts.generate_targetdiff`:

```bash
python -m scripts.generate_targetdiff \
    <checkpoint> \
    --config configs/targetdiff/crossdocked/geometry.yaml \
    --pdbfile path/to/pocket.pdb \
    --outfile results/generated.sdf \
    --n_samples 100
```

Evaluate the resulting SDFs with `val_analysis/metrics.py` (QED, SA, diversity,
PoseBusters) and `val_analysis/smina_docking.py`.

## Data preparation

Datasets are paired **pocket PDB** + **ligand SDF**, processed into `.npz`.
`--model` picks the pocket featurisation: `diffsbdd` (10-dim element one-hots) or
`targetdiff` (27-dim element + amino acid + backbone).

**Your own targets** — `scripts/process_data.py` runs fetch → preprocess → dataset
in one go:

```bash
# from a list of PDB IDs (downloaded from RCSB)
python -m scripts.process_data \
    --pdb_list data/example_pdbs.txt \
    --output_dir data/my_dataset \
    --model targetdiff

# from PDB/CIF files you already have
python -m scripts.process_data \
    --skip_fetch --pdb_dir /path/to/pdbs \
    --output_dir data/my_dataset \
    --model targetdiff
```

**CrossDocked** — download `crossdocked_pocket10.tar.gz` and `split_by_name.pt`
from [Pocket2Mol](https://github.com/pengxingang/Pocket2Mol), extract, then:

```bash
python -m scripts.process_crossdock \
    --crossdocked_dir /path/to/crossdocked_pocket10 \
    --split_path      /path/to/split_by_name.pt \
    --output_dir      data/crossdock_targetdiff \
    --model           targetdiff
```

Run with `--smoke_test` first to check shapes on 2 pairs before the full ~2 h job.

Either way you get:

```
my_dataset/
├── 01_raw_pdbs/          # downloaded .pdb / .cif
├── 02_preprocessed/      # pocket_files/, sdf_files/, basename lists
└── 03_final_dataset/     # train/val/test .npz, size_distribution.npy, summary.txt
```

Point `datadir` (or `--datadir`) at the `03_final_dataset/` path.

Useful flags: `--preprocess_distance` (15 Å, initial pocket cut),
`--dataset_distance` (5 Å, final pocket in the `.npz`), `--test_pdbs` (exclusion
list, `none` to skip), `--keep_duplicates`, `--include_common`. Reference lists
live in `data/`: `example_pdbs.txt` (10 IDs for a quick test),
`crossdocked_{train,test}_pdbs.txt`, `pdb_block_list.txt`.

## Layout

```
configs/            one YAML per experiment, split by backbone
scripts/            train.py, generate_*, test_*, process_*
bash/               SLURM submission scripts (gitignored)
src/
  models/           vendored DiffSBDD and TargetDiff backbones
  prism/            PRISM's own code
    models/         policy wrappers + factories (the backbone abstraction)
    ppo_tuner/      PPO algorithm, rollout, loss, Lightning module
    reward/         reward factory, manager, and scoring/*
    data_modules/   Lightning DataModule + Dataset
    data_processing/ PDB → NPZ pipeline
val_analysis/       post-hoc metrics + Smina docking
Log_Results/        training outputs (gitignored)
results/            generation outputs (SDFs)
```

Get the pretrained DiffSBDD checkpoint with:

```bash
wget -P checkpoints/ https://zenodo.org/record/8183747/files/crossdocked_fullatom_cond.ckpt
```
