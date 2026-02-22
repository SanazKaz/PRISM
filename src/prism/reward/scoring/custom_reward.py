from rdkit import Chem
from rdkit.Chem import Descriptors


mol = 'CCC1=CC=CC=C1'
mol = Chem.MolFromSmiles(mol)