"""
Test for permute_timesteps function to verify it maintains data alignment.
"""
import torch
import unittest
from utils import permute_timesteps
from debug_alignment import assert_same_ids, assert_step_match, assert_atom_mol_match, dbg


class TestPermuteTimesteps(unittest.TestCase):
    def setUp(self):
        # Use CPU for testing to ensure deterministic behavior
        self.device = torch.device('cpu')
        
        # Test parameters
        self.B = 3  # Number of molecules
        self.T = 4  # Number of timesteps
        self.F = 5  # Number of features
        
        # Create mock data that simulates a realistic scenario
        self.create_mock_data()
    
    def create_mock_data(self):
        """Create mock rollout data that resembles real-world data"""
        
        # Create molecule IDs (0, 1, 2)
        molecule_ids = torch.arange(self.B, device=self.device)
        
        # Create ligand atoms with varying counts per molecule
        # Molecule 0: 5 atoms
        # Molecule 1: 3 atoms
        # Molecule 2: 7 atoms
        atoms_per_molecule = [5, 3, 7]
        total_atoms = sum(atoms_per_molecule)
        
        # Create ligand mask: [0,0,0,0,0,1,1,1,2,2,2,2,2,2,2]
        lig_mask = torch.cat([
            torch.full((n,), i, device=self.device, dtype=torch.long)
            for i, n in enumerate(atoms_per_molecule)
        ])
        
        # Create pocket atoms (more atoms than ligand)
        # Molecule 0: 20 atoms
        # Molecule 1: 15 atoms
        # Molecule 2: 25 atoms
        pocket_atoms_per_molecule = [20, 15, 25]
        total_pocket_atoms = sum(pocket_atoms_per_molecule)
        
        # Create pocket mask
        pocket_mask = torch.cat([
            torch.full((n,), i, device=self.device, dtype=torch.long)
            for i, n in enumerate(pocket_atoms_per_molecule)
        ])
        
        # Create timesteps: [B, T]
        timesteps = torch.arange(self.T, device=self.device).unsqueeze(0).expand(self.B, -1)
        
        # Create old_log_probs: [B, T]
        old_log_probs = torch.randn(self.B, self.T, device=self.device)
        
        # Create latents: [N, T, F]
        # Each atom has a unique pattern for all timesteps and features
        # We'll make it deterministic based on atom index for easy verification
        latents = torch.zeros(total_atoms, self.T, self.F, device=self.device)
        for atom_idx in range(total_atoms):
            for t in range(self.T):
                for f in range(self.F):
                    # Encode the atom_idx, timestep and feature in a deterministic way
                    latents[atom_idx, t, f] = 100 * atom_idx + 10 * t + f
        
        # Create next_latents: [N, T, F]
        # Should be identical to latents but shifted one timestep
        next_latents = torch.zeros_like(latents)
        next_latents[:, :-1] = latents[:, 1:]  # Shift one timestep
        next_latents[:, -1] = latents[:, 0]    # Wrap around for the last timestep
        
        # Store everything in rollout_data
        self.rollout_data = {
            "masks": (lig_mask, pocket_mask),
            "timesteps": timesteps,
            "old_log_probs": old_log_probs,
            "latents": latents,
            "next_latents": next_latents
        }
        
        # Store original values for comparison
        self.original_lig_mask = lig_mask.clone()
        self.original_pocket_mask = pocket_mask.clone()
        self.original_latents_sum = latents.sum().item()
        self.original_next_latents_sum = next_latents.sum().item()
    
    def test_permute_timesteps(self):
        """Test that permute_timesteps maintains data alignment"""
        print("\nTesting permute_timesteps function...")
        
        # Checkpoint 1: Before permutation
        lig_mask, pocket_mask = self.rollout_data['masks']
        print("Before permutation:")
        print(f"  lig_mask unique: {torch.unique(lig_mask).tolist()}")
        print(f"  pocket_mask unique: {torch.unique(pocket_mask).tolist()}")
        print(f"  latents shape: {self.rollout_data['latents'].shape}")
        print(f"  next_latents shape: {self.rollout_data['next_latents'].shape}")
        
        # Verify masks match before permutation
        assert_same_ids("before_permute", lig_mask, pocket_mask)
        assert_step_match("before_permute", 
                         self.rollout_data['latents'], 
                         self.rollout_data['next_latents'])
        
        # Apply permutation
        permuted_data = permute_timesteps(self.rollout_data.copy(), self.device)
        
        # Checkpoint 2: After permutation
        lig_mask, pocket_mask = permuted_data['masks']
        print("\nAfter permutation:")
        print(f"  lig_mask unique: {torch.unique(lig_mask).tolist()}")
        print(f"  pocket_mask unique: {torch.unique(pocket_mask).tolist()}")
        print(f"  latents shape: {permuted_data['latents'].shape}")
        print(f"  next_latents shape: {permuted_data['next_latents'].shape}")
        
        # Test 1: Masks should be unchanged
        self.assertTrue(torch.equal(lig_mask, self.original_lig_mask), 
                      "Ligand mask was modified during permutation")
        self.assertTrue(torch.equal(pocket_mask, self.original_pocket_mask), 
                      "Pocket mask was modified during permutation")
        
        # Test 2: Verify masks still match after permutation
        assert_same_ids("after_permute", lig_mask, pocket_mask)
        
        # Test 3: Tensor shapes should be preserved
        self.assertEqual(permuted_data['latents'].shape, self.rollout_data['latents'].shape,
                       "Latents shape changed after permutation")
        self.assertEqual(permuted_data['next_latents'].shape, self.rollout_data['next_latents'].shape,
                       "Next_latents shape changed after permutation")
        
        # Test 4: Step match should still be valid
        assert_step_match("after_permute", 
                         permuted_data['latents'], 
                         permuted_data['next_latents'])
        
        # Test 5: Data values should be preserved (just in different order)
        self.assertAlmostEqual(permuted_data['latents'].sum().item(), 
                             self.original_latents_sum,
                             places=4,
                             msg="Sum of latents changed after permutation")
        self.assertAlmostEqual(permuted_data['next_latents'].sum().item(), 
                             self.original_next_latents_sum,
                             places=4,
                             msg="Sum of next_latents changed after permutation")
        
        # Test 6: Verify timesteps were actually permuted (different from original)
        timesteps_changed = not torch.equal(permuted_data['timesteps'], self.rollout_data['timesteps'])
        self.assertTrue(timesteps_changed, "Timesteps were not actually permuted")
        
        # Test 7: Check that each molecule got its own unique permutation
        # Extract permutation patterns for each molecule from the latents
        mol_perms = {}
        for mol_id in range(self.B):
            # Get atoms for this molecule
            mol_mask = lig_mask == mol_id
            # Get first atom's latents (all should have same permutation)
            first_atom_idx = torch.where(mol_mask)[0][0]
            # Extract permutation pattern by tracking feature 0
            orig_pattern = self.rollout_data['latents'][first_atom_idx, :, 0]
            new_pattern = permuted_data['latents'][first_atom_idx, :, 0]
            
            # Find permutation by matching values
            perm = torch.zeros(self.T, dtype=torch.long)
            for t in range(self.T):
                orig_val = orig_pattern[t]
                for new_t in range(self.T):
                    if new_pattern[new_t] == orig_val:
                        perm[t] = new_t
                        break
            
            mol_perms[mol_id] = perm
        
        # Check that each molecule got a different permutation
        all_same = True
        for i in range(self.B):
            for j in range(i+1, self.B):
                if not torch.equal(mol_perms[i], mol_perms[j]):
                    all_same = False
        
        self.assertFalse(all_same, "All molecules received the same permutation")
        
        print("\nAll permutation tests passed! ✅")

if __name__ == "__main__":
    unittest.main()