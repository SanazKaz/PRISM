import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED
from tqdm import tqdm

# Assuming you have this file, otherwise you might need to comment this import out
# or use rdkit.Chem.RDConfig.RDDataDir to find the SA score script
try:
    from analysis.SA_Score.sascorer import calculateScore
except ImportError:
    print("Warning: SA Scorer not found. SA score will be 0.")
    def calculateScore(mol): return 10.0 # Dummy function

class MoleculeProperties:
    @staticmethod
    def calculate_qed(rdmol):
        return QED.qed(rdmol)

    @staticmethod
    def calculate_sa(rdmol):
        try:
            sa = calculateScore(rdmol)
            return round((10 - sa) / 9, 2)
        except:
            return 0.0

    @staticmethod
    def calculate_logp(rdmol):
        return Crippen.MolLogP(rdmol)

    @staticmethod
    def calculate_mwt(rdmol):
        return Descriptors.ExactMolWt(rdmol)

    @staticmethod
    def calculate_hbd(rdmol):
        return Lipinski.NumHDonors(rdmol)

    @staticmethod
    def calculate_hba(rdmol):
        return Lipinski.NumHAcceptors(rdmol)

    @staticmethod
    def calculate_aromatic_rings(rdmol):
        return Lipinski.NumAromaticRings(rdmol)

    @staticmethod
    def calculate_lipinski(rdmol):
        # Rule of 5 violations
        rule_1 = Descriptors.ExactMolWt(rdmol) < 500
        rule_2 = Lipinski.NumHDonors(rdmol) <= 5
        rule_3 = Lipinski.NumHAcceptors(rdmol) <= 10
        rule_4 = (logp := Crippen.MolLogP(rdmol) >= -2) & (logp <= 5)
        rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(rdmol) <= 10
        return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])

    @classmethod
    def calculate_diversity(cls, pocket_mols):
        if len(pocket_mols) < 2:
            return 0.0
        
        # Use Tanimoto similarity on Morgan Fingerprints
        fps = [Chem.RDKFingerprint(x) for x in pocket_mols]
        n_mols = len(pocket_mols)
        
        similarity_sum = 0
        count = 0
        for i in range(n_mols):
            for j in range(i + 1, n_mols):
                similarity_sum += DataStructs.TanimotoSimilarity(fps[i], fps[j])
                count += 1
        
        avg_similarity = similarity_sum / count if count > 0 else 0.0
        return 1.0 - avg_similarity

    def evaluate_mean(self, rdmols):
        """
        Run full evaluation and return dictionary of means
        """
        valid_mols = []
        for mol in rdmols:
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                    valid_mols.append(mol)
                except:
                    continue

        if len(valid_mols) < 1:
            return {}

        metrics = {
            'val/QED': np.mean([self.calculate_qed(m) for m in valid_mols]),
            'val/SA': np.mean([self.calculate_sa(m) for m in valid_mols]),
            'val/LogP': np.mean([self.calculate_logp(m) for m in valid_mols]),
            'val/MWT': np.mean([self.calculate_mwt(m) for m in valid_mols]),
            'val/HBD': np.mean([self.calculate_hbd(m) for m in valid_mols]),
            'val/HBA': np.mean([self.calculate_hba(m) for m in valid_mols]),
            'val/AromaticRings': np.mean([self.calculate_aromatic_rings(m) for m in valid_mols]),
            'val/Lipinski': np.mean([self.calculate_lipinski(m) for m in valid_mols]),
            'val/Diversity': self.calculate_diversity(valid_mols)
        }
        
        return metrics