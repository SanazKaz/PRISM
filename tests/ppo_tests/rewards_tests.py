def test_medchem_scoring():
    """Test the MedChem scoring with known molecules."""
    reward = MedChemReward()
    
    # Test with a simple molecule
    aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    aspirin_mol = Chem.MolFromSmiles(aspirin_smiles)
    score = reward.lilly_medchem_score(aspirin_mol)
    print(f"Aspirin score: {score}")
    
    # You can test with other molecules too
    print("\nTesting complete!")

if __name__ == "__main__":
    test_medchem_scoring()