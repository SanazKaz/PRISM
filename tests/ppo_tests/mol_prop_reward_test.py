import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]   # PRISM/ (zwei Ebenen hoch)
sys.path.insert(0, str(project_root))
sys.path.insert(1, str(project_root / "src" / "models" / "diffsbdd"))

# --- ab hier deine eigentlichen Imports ---
from rdkit import Chem
from src.prism.reward.scoring.my_mol_prop_reward import TargetProfileReward

reward = TargetProfileReward(sharpness=5.0)
smiles = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "Cc1ccc(cc1)Nc1nccc(n1)-c1cccnc1",
    "Cc1ccccc1",
]
mols = [Chem.MolFromSmiles(s) for s in smiles]
scores = reward(mols)
for smi, sc in zip(smiles, scores):
    print(f"{sc.item():.4f}  {smi}")