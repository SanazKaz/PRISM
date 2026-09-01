# PRISM: Policy-Reinforced Iterative Structure-Based Molecular Diffusion

Official implementation of PRISM: A Hybrid Diffusion-Reinforcement Learning Framework for 3D Structure-based De Novo Design. 

PRISM framework takes a pretrained **DiffSBDD** or **TargetDiff** checkpoint and tunes layers so generated
ligands optimise a chemistry reward while staying close to the pretrained prior.

## Workshop Paper

**PRISM: A Hybrid Diffusion-Reinforcement Learning Framework for 3D Structure-based De Novo Design**

Sanaz Kazeminia, Lewis R. Vidler, Pushkar G. Ghanekar, Nele P Quast, Garrett M Morris

[![Paper](https://img.shields.io/badge/OpenReview-TAeVNZ77sH-8C1B13)](https://openreview.net/forum?id=TAeVNZ77sH)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![PRISM training loop](assets/prism_overview.png)

## Environment

```bash
conda env create -f environment.yml
conda activate PRISM_25
pip install -e .
```

`environment.yml` pins torch 2.6.0+cu124 and the CUDA-specific PyG wheels
(`torch-scatter`, `torch-cluster`, `torch-geometric`).

Check it worked:

```bash
python -c "import torch, torch_scatter, torch_geometric, prism; print(torch.cuda.is_available())"
python -m pytest tests/unit -q
```

## Checkpoints and data

Fine-tuned checkpoints from the paper, and the estrogen receptor alpha
case-study inputs: **https://zenodo.org/records/22229237**

Each `.pt` is the native backbone format the generation scripts load.

| File | Backbone | Reward | Seed | Epoch |
|---|---|---|---|---|
| `prism_diffsbdd_penalised_logp_seed42_ep25.pt` | DiffSBDD | penalised LogP | 42 | 25 |
| `prism_targetdiff_penalised_logp_seed976_ep13.pt` | TargetDiff | penalised LogP | 976 | 13 |
| `prism_diffsbdd_geometry_seed42_ep09.pt` | DiffSBDD | geometry | 42 | 9 |
| `prism_diffsbdd_multiobj_seed976_ep27.pt` | DiffSBDD | geometry + docking + QED + SA (weighted sum) | 976 | 27 |
| `prism_targetdiff_multiobj_seed789_ep17.pt` | TargetDiff | geometry + docking + QED + SA (product) | 789 | 17 |
| `prism_targetdiff_case_estrogen_receptor_alpha_seed976_ep54.pt` | TargetDiff | ERα: feature density + property_2d + geometry (product) | 976 | 54 |

`era_case_study_data.zip` holds what the ERα case study needs beyond the code:
the PDB list fed to `scripts/process_data.py`, the pharmacophore hotspot `.pkl`
for `feature_density`, `propeties_ref.json` for `property_2d`, and the three
held-out pocket/ligand pairs used for generation. Everything else in the
pipeline regenerates from the PDB list.

PPO training is not bitwise reproducible across GPUs, so use these checkpoints
to reproduce the paper numbers rather than retraining.

## Train

PPO fine-tunes a pretrained checkpoint, so download one first and put it under
`checkpoints/`:

- DiffSBDD — https://github.com/arneschneuing/DiffSBDD
- TargetDiff — https://github.com/guanjq/targetdiff

```bash
python scripts/train.py \
    --config configs/targetdiff/targetdiff_default.yaml \
    --warm_start_from_ddpm checkpoints/targetdiff_pretrained_models/targetdiff_pretrained_diffusion.pt
```

For DiffSBDD, swap in `configs/diffsbdd/diffsbdd_default.yaml` and a DiffSBDD
checkpoint; the config's `model_type` selects the backbone.

Both default configs reward QED and synthetic accessibility only
(`weighted_sum`, 0.5 each). Their PPO settings match the validated
multi-objective runs, so they are a working baseline to add rewards to.

These CLI flags override the YAML, which is how one config is reused across a
seed × target sweep: `--seed`, `--datadir`, `--logdir`, `--dataset_name`,
`--target_name`, `--hotspot_path`, `--resume_from_checkpoint`.

Checkpoints land in
`Log_Results/<logdir>/<run_identifier>/checkpoints/<dataset>/seed=<S>/`, saved as
both `.ckpt` (Lightning) and `.pt` (native backbone format — **this is the one
you generate with**). Top-3 by `train/reward_mean`, plus `last`.

### Config

One YAML per experiment, under `configs/<backbone>/`. Clone
`targetdiff_default.yaml` or `diffsbdd_default.yaml` and change only what the
experiment is asking about.

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
    --config configs/targetdiff/targetdiff_default.yaml \
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
    --config configs/targetdiff/targetdiff_default.yaml \
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
    --config configs/targetdiff/targetdiff_default.yaml \
    --pdbfile path/to/pocket.pdb \
    --outfile results/generated.sdf \
    --n_samples 100
```

Evaluate the resulting SDFs with `val_analysis/metrics.py` (QED, SA, diversity,
PoseBusters) and `val_analysis/smina_docking.py`.

## Case study: estrogen receptor alpha

The case studies fine-tune on one target instead of the whole CrossDocked set.
The reward is a pharmacophore hotspot match, a 2D property match to known
binders, and geometry, combined as a product. Here is the full ERα run.

**1. Get the inputs.** Download `era_case_study_data.zip` from the Zenodo
record above. It has the PDB list, the hotspot `.pkl`, `propeties_ref.json`,
and the three held-out pockets we generate for.

**2. Build the dataset.** The PDB list is every ERα structure from an RCSB
sequence search. `process_data.py` fetches them, cuts the pockets, and writes
the `.npz` files. It also drops the three held-out structures, so the model
never sees the pockets it is later asked to design for:

```bash
python -m scripts.process_data \
    --pdb_list estrogen_recep_alpha_pdb_list.txt \
    --output_dir data/Estrogen_recep_alpha \
    --model targetdiff
```

Put the hotspot `.pkl` in `data/Estrogen_recep_alpha/hotspot_analysis/` and
`propeties_ref.json` in `src/prism/reward/scoring/reward_data/`.

**3. Train.** `--target_name` picks both the hotspot and the reference-binder
stats for that target, so one config serves every case study:

```bash
python scripts/train.py \
    --config configs/targetdiff/case_studies/multi_objective_pharm_product.yaml \
    --warm_start_from_ddpm checkpoints/targetdiff_pretrained_models/targetdiff_pretrained_diffusion.pt \
    --datadir data/Estrogen_recep_alpha/03_final_dataset_targetdiff \
    --hotspot_path data/Estrogen_recep_alpha/hotspot_analysis/Estrogen_recep_alpha_hotspot_data.pkl \
    --target_name Estrogen_recep_alpha \
    --dataset_name Estrogen_recep_alpha \
    --logdir Log_Results/case_studies \
    --seed 976
```

To sweep seeds and all six targets on SLURM instead, see
`bash/target_diff/train_case_study_targets.sh`.

**4. Generate.** Pick the checkpoint with the best `train/reward_mean`, or skip
training and use ours from the Zenodo record
(`prism_targetdiff_case_estrogen_receptor_alpha_seed976_ep54.pt`):

```bash
python -m scripts.test_targets <checkpoint> \
    --model targetdiff \
    --config configs/targetdiff/case_studies/multi_objective_pharm_product.yaml \
    --targets_dir data \
    --target Estrogen_recep_alpha_2qzo \
    --outdir results/era \
    --n_samples 1000 --batch_size 84 --num_steps 1000
```

Drop `--target` to run all three ERα structures. Score the SDFs with
`val_analysis/metrics.py` and `val_analysis/smina_docking.py`.

The other five targets work the same way — swap the target name and its data
directory. The configs are in `configs/targetdiff/case_studies/`.

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
`case_study_train_pdbs.txt`, `heldout_eval_pdbs.txt`, `pdb_block_list.txt`.

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

## Citation

If you use PRISM in your work, please cite:

```bibtex
@inproceedings{kazeminia2026prism,
    title     = {{PRISM}: A Hybrid Diffusion-Reinforcement Learning Framework for 3D Structure-based De Novo Design},
    author    = {Sanaz Kazeminia and Lewis R. Vidler and Pushkar G. Ghanekar and Nele P Quast and Garrett M Morris},
    booktitle = {ICLR 2026 Workshop on Generative and Experimental Perspectives for Biomolecular Design},
    year      = {2026},
    url       = {https://openreview.net/forum?id=TAeVNZ77sH}
}
```

We thank the authors of DiffSBDD and TargetDiff as this repo builds on their work.

DiffSBDD: https://github.com/arneschneuing/DiffSBDD

TargetDiff: https://github.com/guanjq/targetdiff


