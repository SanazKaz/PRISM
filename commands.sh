
uv venv
source .venv/bin/activate
# install torch with cuda support
uv pip install torch==2.8.0
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu129/
uv pip install pytorch-lightning
# Install repo
uv pip install -e .

python -m scripts.process_data --pdb_list /home/alexi/Documents/PRISM/data/example_pdb_list.txt --output_dir /home/alexi/Documents/PRISM/data/test

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




python scripts/train.py \
    --config configs/ppo_config.yaml \
       --warm_start_from_ddpm checkpoints/crossdocked_fullatom_cond.ckpt \
       --seed 42 \
