# Adding a New Diffusion Model to PRISM

This guide walks through every file that must be created or modified to integrate
a new pretrained diffusion model into the PRISM PPO fine-tuning framework.
Follow the steps in order. TargetDiff is used as the reference example throughout.

---

## Overview

PRISM's PPO loop calls exactly three things on any model:

1. **`sample_given_pocket`** — generate molecules during rollout collection
2. **`log_p_zs_given_zt`** — recompute log π(z_s | z_t) during the PPO update
3. **`get_ligand_and_pocket`** — convert a dataloader batch into the dicts those two functions expect

Everything else (checkpoint loading, architecture construction, dataset info) is
handled by a factory function that you add to `policy_factory.py`.

---

## Step 1 — Vendor the model source

Place the model's source code under `src/models/<model_name>/`. Do not modify
the vendored source. PRISM adds a `sys.path` entry to make it importable.

```
src/models/
├── diffsbdd/       # DiffSBDD (reference)
├── targetdiff/     # TargetDiff (reference)
└── <new_model>/    # your model here
```

Verify the model can be imported from the project root:

```python
import sys
sys.path.insert(0, 'src/models/<new_model>')
import <new_model_module>  # should not raise
```

---

## Step 2 — Understand the model's key dimensions

Before writing any code, find the following in the model source:

| Dimension | Where to find it | Example (TargetDiff) |
|---|---|---|
| `protein_feat_dim` | Input to the protein embedding layer | 27 (6 element + 20 AA + 1 backbone) |
| `ligand_atom_types` | Number of atom-type classes | 13 (add_aromatic encoding) |
| `n_dims` | Spatial dimensions | 3 |
| `total_timesteps` | Length of the diffusion chain | 1000 |
| Checkpoint key | How weights are stored in the `.pt` file | `ckpt['model']` |
| Named parameter path | Run `dict(model.named_parameters()).keys()` | `refine_net.base_block.0.*` |

**Critical:** print the named parameter paths from the actual pretrained model
before deciding which layers to freeze. Do not guess layer names from the source
code — run this:

```python
ckpt = torch.load('checkpoints/<model>.pt', map_location='cpu')
state_dict = ckpt['model']  # adjust key as needed
for k in state_dict.keys():
    print(k)
```

This is how we caught the `refine_net.blocks.0.layers.6` vs `refine_net.base_block.6`
bug — the yaml had the wrong path because we read the source instead of printing
the actual keys.

---

## Step 3 — Create the data processing script

**File:** `scripts/process_crossdock_<model_name>.py`

The model almost certainly uses different protein feature dimensions than DiffSBDD's
10-dim element one-hots. You need a separate NPZ dataset.

Use `scripts/process_crossdock_targetdiff.py` as the template. The key function
to replace is `featurize_pocket_<model_name>()`. It must:

- Use the **model's own parser** if one exists (e.g. TargetDiff's `PDBProtein`)
  rather than reimplementing atom parsing. Find it in the vendored source.
- Return `(coords [N, 3], features [N, protein_feat_dim])` where the features
  are **bit-for-bit identical** to what the pretrained model was trained on.
- Use `np.atleast_1d` / `np.atleast_2d` on all arrays returned by the parser
  to guard against single-atom degenerate pockets.
- Raise an exception for pockets with fewer than 10 atoms (these are malformed
  PDB files, not real binding sites).

**Always keep ligand features as DiffSBDD-style element one-hots.** The reward
pipeline and SMILES computation depend on this encoding. Only the protein
featurisation changes.

**Add a `--smoke_test` flag** that runs the full pipeline on 2 pairs only,
prints shapes, and asserts dimensions before a full multi-hour job. Pattern:

```python
parser.add_argument('--smoke_test', action='store_true')
args = parser.parse_args()
if args.smoke_test:
    smoke_test(args)
else:
    main(args)
```

The smoke test must verify:
- `poc_coords.ndim == 2 and poc_coords.shape[1] == 3`
- `poc_one_hot.ndim == 2 and poc_one_hot.shape[1] == protein_feat_dim`
- `np.concatenate` across multiple pockets does not crash

**`sys.path` ordering matters.** If the model source contains a bare `import utils`
or `import constants`, those must resolve to the model's own files, not PRISM's
`utils` package. Put the model's root first:

```python
sys.path.insert(0, str(_MODEL_ROOT))   # model's bare imports resolve here
sys.path.insert(1, str(_PROJECT_ROOT))  # PRISM imports resolve here
```

---

## Step 4 — Create the factory function

**File:** `src/prism/models/policy_factory.py`

Add two things:

### 4a. Dataset info dict

Define `_<MODEL>_DATASET_INFO` with the model's atom decoder, encoder, colours,
and radii. The decoder is the list of element symbols in class-index order (e.g.
`['H', 'C', 'C', 'N', ...]` for TargetDiff's 13-class add_aromatic encoding).
This dict is passed to reward functions and evaluation metrics.

```python
_NEWMODEL_ATOM_DECODER = ['C', 'N', 'O', ...]  # one entry per class index
_NEWMODEL_DATASET_INFO = {
    'atom_decoder': _NEWMODEL_ATOM_DECODER,
    'atom_encoder': {sym: i for i, sym in enumerate(_NEWMODEL_ATOM_DECODER)},
    'colors_dic':   [...],  # one hex colour per class
    'radius_dic':   [...],  # one float radius per class
}
```

### 4b. Build function

```python
def build_<newmodel>_policy(config, device, warm_start_checkpoint):
    model_cfg = getattr(config, 'model', None)

    # CLI arg takes priority over yaml — always.
    checkpoint_path = (
        warm_start_checkpoint
        or (getattr(model_cfg, 'checkpoint', None) if model_cfg else None)
    )
    if checkpoint_path is None:
        raise ValueError("<NewModel> requires a checkpoint path.")

    protein_feat_dim  = getattr(model_cfg, 'protein_feat_dim',  <default>) if model_cfg else <default>
    ligand_atom_types = getattr(model_cfg, 'ligand_atom_types', <default>) if model_cfg else <default>

    policy = load_<newmodel>_policy(
        checkpoint_path=checkpoint_path,
        device=device,
        protein_atom_feature_dim=protein_feat_dim,
        ligand_atom_feature_dim=ligand_atom_types,
    )
    dataset_info = {k: list(v) if isinstance(v, list) else v
                    for k, v in _NEWMODEL_DATASET_INFO.items()}
    return policy, dataset_info
```

Create a separate `src/prism/models/<newmodel>_factory.py` for the actual
`load_<newmodel>_policy()` function that handles checkpoint loading. See
`targetdiff_factory.py` for the pattern, especially the shape-mismatch stripping
loop and the `strict=False` load.

---

## Step 5 — Create the policy wrapper

**File:** `src/prism/models/<newmodel>_policy.py`

Subclass `BaseDiffusionPolicy` and implement all four abstract requirements:

```python
from src.prism.models.base_policy import BaseDiffusionPolicy

class NewModelPolicy(BaseDiffusionPolicy):

    def sample_given_pocket(self, pocket, num_nodes_lig, **kwargs):
        """Full reverse diffusion trajectory. Returns 6 things — see base class docstring."""
        ...

    def log_p_zs_given_zt(self, s, t, z_t, z_s, xh_pock, lig_mask, poc_mask):
        """Re-evaluate log π(z_s | z_t) with current weights. Returns [n_mols]."""
        ...

    def get_ligand_and_pocket(self, data):
        """Convert dataloader batch to (ligand_dict, pocket_dict, names)."""
        ...

    @property
    def atom_nf(self): return self._num_atom_types

    @property
    def n_dims(self): return 3
```

### `sample_given_pocket` requirements

- Must capture **per-step log-probabilities** into `mol_log_probs` (list of
  `[n_mols]` tensors, one per reverse step) and **latent states** into
  `z_states` (list of `[N_atoms, n_dims + atom_nf]` tensors). The rollout
  buffer needs both.
- Latent states must be packed as `torch.cat([pos, one_hot_float], dim=-1)`.
  The PPO loss slices `z[:, :n_dims]` for positions and `z[:, n_dims:]` for
  atom types.
- Return exactly: `xh_lig, xh_pocket, lig_mask, pocket_mask, mol_log_probs, z_states`

### `log_p_zs_given_zt` requirements

- `t` and `s` arrive as normalised floats `[n_mols, 1]` in `[0, 1]`. Convert
  to integer timesteps with `t_int = (t.squeeze(-1) * total_timesteps).round().long()`.
- `z_t` and `z_s` are `[N_atoms, n_dims + atom_nf]`. Unpack with:
  ```python
  pos_t = z_t[:, :self._n_dims]
  v_t   = z_t[:, self._n_dims:].argmax(dim=-1)  # back to integer indices
  ```
- Must return `[n_mols]` — one scalar per molecule, averaged over atoms with
  `scatter_mean`. Do not return per-atom values.
- Must use the **current** model weights (i.e. a live forward pass). Do not
  cache or reuse the rollout's predictions.
- Must **not** be wrapped in `torch.no_grad()` — gradients must flow through
  this for the PPO loss to train.

### `parameters()` and `named_parameters()` routing

If the policy wrapper has attributes that are not `nn.Parameter` (scalars,
lists, etc.), you must override `parameters()` and `named_parameters()` to
delegate to the inner model only. Otherwise PyTorch will try to recurse into
non-parameter attributes and raise:

```python
def parameters(self, recurse=True):
    return self._model.parameters(recurse=recurse)

def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
    return self._model.named_parameters(prefix=prefix, recurse=recurse,
                                        remove_duplicate=remove_duplicate)
```

---

## Step 6 — Wire into `lightning_module.py`

**File:** `src/prism/ppo_tuner/lightning_module.py`

Add a new branch in `PPOFineTuner.__init__`:

```python
model_type = getattr(self.config, 'model_type', 'diffsbdd')

if model_type == 'targetdiff':
    self.policy, self.dataset_info = build_targetdiff_policy(...)
    self.ddpm_model = None
elif model_type == '<new_model>':                          # ADD THIS
    self.policy, self.dataset_info = build_<newmodel>_policy(...)
    self.ddpm_model = None
else:
    self.policy, self.ddpm_model, self.dataset_info = build_diffsbdd_policy(...)
```

Also update the `validation_step` and `train.py` histogram logic if needed —
both already gate on `model_type == 'targetdiff'` to skip DiffSBDD-specific
steps. Add `or model_type == '<new_model>'` to those conditions.

---

## Step 7 — Create the YAML config

**File:** `configs/<newmodel>_ppo.yaml`

Minimum required fields:

```yaml
run_identifier: '<newmodel>_ppo_cd'
logdir: 'Log_Results'
model_type: <new_model>          # must match the string in lightning_module.py

freeze_except:                   # see Step 8 — get these from actual named_parameters()
  - 'some.layer.name'

model:
  checkpoint: 'checkpoints/<newmodel>_pretrained.pt'
  protein_feat_dim: <N>          # must match pretrained checkpoint
  ligand_atom_types: <M>
  total_timesteps: <T>

datadir: '/path/to/processed_crossdock_<newmodel>'
dataset: 'crossdock'
batch_size: 8
num_workers: 4
gpus: 1
n_epochs: 1

ppo:
  num_outer_epochs: 60
  num_inner_epochs: 2
  n_steps: 64
  ligand_chunk_size: 32
  batch_size: 64
  clip_range: 0.1
  entropy_coef: 0.001
  kl_coef: 0.0
  max_grad_norm: 1.0
  gradient_accumulation_steps: 10
  train_timesteps: <T/2>         # typically last half of the chain
  target_kl: 0.03
  lr: 3.0e-5
  num_nodes_lig: 25

reward_params:
  ...                            # same structure as existing configs

eval_params:
  smiles_file: '/path/to/train_smiles.npy'
  ...
```

---

## Step 8 — Set freeze_except correctly

**This is the most error-prone step.** The substring matching in
`freeze_parameters()` will silently freeze layers if the names are wrong.

Do this on the cluster with the real checkpoint:

```python
import torch, sys
sys.path.insert(0, 'src/models/<new_model>')
from <new_model_module> import <ModelClass>

ckpt = torch.load('checkpoints/<newmodel>_pretrained.pt', map_location='cpu')
model = <ModelClass>(...)
model.load_state_dict(ckpt['model'])

for name, param in model.named_parameters():
    print(name, param.shape)
```

Identify the layer groups you want to keep trainable (typically the last
transformer/attention blocks and any output heads). Copy the exact dotted
name prefixes into `freeze_except`. Then verify:

```python
trainable = [n for n, p in model.named_parameters()
             if any(x in n for x in freeze_except)]
frozen    = [n for n, p in model.named_parameters()
             if not any(x in n for x in freeze_except)]
print(f"Trainable: {len(trainable)}, Frozen: {len(frozen)}")
# Manually inspect `trainable` to confirm the right layers appear
```

---

## Step 9 — Create the SLURM script

**File:** `bash/<newmodel>_CD_training.sh`

Use `bash/targetdiff_CD_training.sh` as the template. Key points:

- `--logdir` must point to the **root** log dir (e.g. `$PROJECT_ROOT/Log_Results`),
  not the run subdirectory. `train.py` appends `run_identifier/checkpoints/dataset/seed=N`
  automatically. Passing the full subdir path causes double-nesting.
- `--dataset_name <new_model>_crossdock` — prevents the code from falling back
  to the parent directory name when constructing the checkpoint path.
- Add a pre-flight check that the dataset dir exists and contains
  `size_distribution.npy` (or skip this check if the new model doesn't need it —
  check `train.py`'s `model_type` guard).

---

## Step 10 — Add unit tests

**File:** `tests/unit/test_<newmodel>_log_probs.py`

At minimum, test the mathematical properties of `log_p_zs_given_zt`:

1. Output shape is `[n_mols]` for any number of atoms.
2. All values are finite (no NaN, no Inf).
3. Log-prob is maximised when `z_s == predicted_mean` (vs displaced by an offset).
4. If the model has a categorical atom-type component, log-prob is higher when
   `v_s == argmax(predicted_distribution)` than when using the worst type.

See `tests/unit/test_log_probs.py` for the TargetDiff and DiffSBDD pattern.
Extract the formula as a standalone function so you can test it without
loading the full model.

Run all unit tests before submitting any job:

```bash
python -m pytest tests/unit/ -v
```

---

---

## Step 11 — Wire up the reconstruction pipeline (critical)

This step is easy to overlook and causes silent, catastrophic training failures
if skipped. Every model encodes atom types differently. If the wrong decoder is
used during rollout reconstruction, molecules are built from wrong atom types,
rewards are computed on chemical nonsense, and training diverges silently.

### The rule

> **Each model must decode its own atom-type indices using its own pipeline.**
> Never use DiffSBDD's `build_molecule` for a non-DiffSBDD model.

| Model | Atom-type encoding | Reconstruction pipeline |
|---|---|---|
| DiffSBDD | 10-class element one-hot | `build_molecule` + `process_molecule` (molecule_builder.py) |
| TargetDiff | 13-class `add_aromatic` | `reconstruct_from_generated(basic_mode=False)` via OpenBabel |
| **New model** | **check the model source** | **must use the model's own pipeline** |

### How to add the reconstruction function

In `src/prism/models/targetdiff_inference.py` (or a new `<newmodel>_inference.py`),
add two functions following the TargetDiff pattern:

**1. A rollout reconstruction callable** for `build_molecules_from_batch`:

```python
def make_<newmodel>_reconstruction_fn():
    """Returns (coords, atom_indices) -> Mol | None using <NewModel>'s own pipeline."""
    def _reconstruct(coords, atom_indices):
        try:
            # decode atom_indices using the model's own vocabulary
            atomic_nums = <newmodel_decode_atoms>(atom_indices)
            aromatic    = <newmodel_decode_aromatic>(atom_indices)  # if applicable
            mol = <newmodel_reconstruct>(coords, atomic_nums, aromatic, basic_mode=False)
            smi = Chem.MolToSmiles(mol)
            return mol if '.' not in smi else None
        except Exception:
            return None
    return _reconstruct
```

**2. A batch reconstruction function** for test-time scripts:

```python
def reconstruct_molecules_<newmodel>(all_pred_pos, all_pred_v, debug=False):
    """Convert list of (pos, atom_indices) predictions to RDKit mols."""
    molecules = []
    for mol_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_pos, all_pred_v)):
        try:
            mol = <newmodel_reconstruct>(pred_pos, pred_v, basic_mode=False)
            smi = Chem.MolToSmiles(mol)
            molecules.append(mol if '.' not in smi else None)
        except Exception:
            molecules.append(None)
    return molecules
```

### Wire it into the training loop

In `src/prism/ppo_tuner/lightning_module.py`, add a branch for your model:

```python
elif model_type == '<new_model>':
    self.policy, self.dataset_info = build_<newmodel>_policy(...)
    self.ddpm_model = None
    from src.prism.models.<newmodel>_inference import make_<newmodel>_reconstruction_fn
    reconstruction_fn = make_<newmodel>_reconstruction_fn()
```

`reconstruction_fn` is passed to `get_reward_manager(reconstruction_fn=reconstruction_fn)`,
which passes it to `RewardManager`, which passes it to `build_molecules_from_batch`.
The chain is already in place — you only need to supply the function.

### `basic_mode` — always False

TargetDiff's `reconstruct_from_generated` has a `basic_mode` parameter.
**Always pass `basic_mode=False`.**

- `basic_mode=True` (the vendor default) silently ignores aromatic flags —
  OpenBabel assigns bonds without aromatic hints, producing wrong SMILES.
- `basic_mode=False` passes aromatic information to OpenBabel, producing
  chemically correct aromatic rings.

The vendor's own evaluation script (`evaluate_diffusion.py`) uses the default
`basic_mode=True`, which is a known limitation. PRISM intentionally deviates
from this to produce better molecules.

If a new model has its own `basic_mode`-equivalent flag, always choose the
mode that **uses all available atom-type information**.

### Verification checklist for the reconstruction pipeline

Run this before any training job:

```python
# 1. Confirm atom-type dimension matches between NPZ and model
import numpy as np
data = np.load('/path/to/your_dataset/train.npz')
print("lig_one_hot shape:", data['lig_one_hot'].shape[-1])    # should match ligand_atom_types in config
print("pocket_one_hot shape:", data['pocket_one_hot'].shape[-1])  # should match protein_feat_dim in config

# 2. Generate one batch and inspect the decoded SMILES
# Run with model_type=<new_model> and 1 outer epoch, 1 pocket.
# Check the reward table printed by RewardManager — SMILES should look like
# real drug-like molecules, not gibberish or single atoms.

# 3. Sanity-check atom-type index mapping
from utils import transforms as trans  # or your model's equivalent
for i in range(num_atom_types):
    print(i, trans.get_atomic_number_from_index([i], mode='add_aromatic'))
# Compare against atom_decoder in policy_factory.py — they must agree.
```

---

## Checklist

- [ ] Model source vendored under `src/models/<new_model>/`
- [ ] Named parameter paths printed from the real checkpoint
- [ ] Data processing script created with `--smoke_test` flag
- [ ] Smoke test passes: correct shapes, concatenation works
- [ ] **`lig_one_hot.shape[-1]` matches `ligand_atom_types` in config**
- [ ] **`pocket_one_hot.shape[-1]` matches `protein_feat_dim` in config**
- [ ] Factory function added to `policy_factory.py`
- [ ] **`atom_decoder` in `_<MODEL>_DATASET_INFO` matches the model's atom-type index order**
- [ ] Policy wrapper implements all 4 abstract methods + `parameters()` routing
- [ ] **`make_<newmodel>_reconstruction_fn()` created and wired into `lightning_module.py`**
- [ ] **`reconstruct_molecules_<newmodel>()` created and used in test scripts**
- [ ] **`basic_mode=False` (or equivalent) confirmed in all reconstruction calls**
- [ ] `lightning_module.py` branched on `model_type`
- [ ] `freeze_except` verified against real named_parameters output
- [ ] YAML config has correct `protein_feat_dim` matching the checkpoint
- [ ] SLURM script uses root `--logdir`, not subdir
- [ ] Unit tests added and passing
- [ ] **Smoke rollout: run 1 epoch, inspect reward table SMILES for chemical sanity**
- [ ] Smoke test submitted before full job
