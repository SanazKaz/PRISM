from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def score_mol(self, mol: Mol) -> float:
    if mol is None:
        return 0.0
    
    try:
        raw_feats = self.fdef.GetFeaturesForMol(mol)
        mol_feats = [f for f in raw_feats if f.GetFamily() in self.keep]
    except Exception:
        return 0.0
    
    if len(mol_feats) == 0:
        return 0.0
    
    # Group generated features by type
    feats_by_type = {}
    for feat in mol_feats:
        ft = feat.GetFamily()
        if ft not in feats_by_type:
            feats_by_type[ft] = []
        feats_by_type[ft].append(feat)
    
    total_score = 0.0
    total_max_score = 0.0
    
    # Iterate over each feature type (Donors, Acceptors, etc.) separately
    for feat_type in self.keep: # iterate over 'keep' to ensure we check all targets
        
        # 1. Get Target Clusters for this type (ALL OF THEM - don't slice top N!)
        # We store (index, center, count)
        targets = [(i, self.cluster_centers[i], self.cluster_counts[i]) 
                   for i in range(len(self.cluster_features)) 
                   if self.cluster_features[i] == feat_type]
        
        if not targets:
            continue
            
        target_centers = np.array([t[1] for t in targets])
        target_weights = np.array([t[2] for t in targets])
        
        # Add to total theoretical max (normalization factor)
        # We sum the weights of the Top N ideal clusters for normalization
        # to keep the score between 0 and 1 roughly.
        ideal_count = self.target_profile.get(feat_type, (0, 1))[0]
        if ideal_count > 0:
            sorted_weights = sorted(target_weights, reverse=True)
            total_max_score += sum(sorted_weights[:ideal_count])
        else:
            # If ideal is 0, we don't expect this feature, max_score contribution is 0
            pass

        # 2. Get Generated Features for this type
        gen_feats = feats_by_type.get(feat_type, [])
        if not gen_feats:
            continue
            
        # CORRECT COORDINATES: Subtract Global Centroid!
        gen_coords = []
        for f in gen_feats:
            pos = f.GetPos()
            # Apply the shift to match the PDB frame
            shifted_pos = np.array([pos.x, pos.y, pos.z]) - self.metadata['global_centroid']
            gen_coords.append(shifted_pos)
        gen_coords = np.array(gen_coords)
        
        # 3. Build Cost Matrix (Distance Matrix)
        # Shape: (Num_Generated, Num_Targets)
        dists = cdist(gen_coords, target_centers)
        
        # 4. Hungarian Algorithm (Linear Sum Assignment)
        # It matches indices to minimize total distance
        row_ind, col_ind = linear_sum_assignment(dists)
        
        # 5. Calculate Score for the optimal pairs
        current_type_score = 0.0
        for r, c in zip(row_ind, col_ind):
            d = dists[r, c]
            weight = target_weights[c]
            
            # Soft Gaussian Score (No Hard Cutoff needed, or use a loose one)
            # Using a slightly wider sigma (1.5) to help gradients
            term = np.exp(-0.5 * (d / 1.5) ** 2)
            current_type_score += term * weight
            
        total_score += current_type_score

    # Normalize
    placement_score = total_score / total_max_score if total_max_score > 0 else 0.0
    
    # --- Profile Score (Count Penalty) ---
    # This remains similar to your code, ensuring we don't just spam features elsewhere
    profile_score = 0.0
    for feat_type, (ideal_count, tolerance) in self.target_profile.items():
        actual_count = len(feats_by_type.get(feat_type, []))
        if ideal_count == 0:
            if actual_count == 0: profile_score += 1.0
        else:
            # Gaussian penalty for count mismatch
            profile_score += np.exp(-0.5 * ((actual_count - ideal_count) / tolerance) ** 2)
    profile_score /= len(self.target_profile)
    
    final_score = self.placement_weight * placement_score + self.profile_weight * profile_score
    return float(max(0.0, min(1.0, final_score)))