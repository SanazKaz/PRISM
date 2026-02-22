"""
DBSCAN Pharmacophore Hotspot Generation.

This script clusters pharmacophore features from aligned ligands using DBSCAN
to identify hotspot regions for structure-based drug design.

Features:
- Global centroid centering (preserves alignment)
- Per-feature-type clustering
- Customizable min_samples and min_count thresholds
- Generates summary plots and saves hotspot data as pickle

Usage:
    python generate_hotspots.py --sdf_dir /path/to/aligned/sdfs --output_dir /path/to/output
"""

import argparse
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import Counter

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.FeatMaps import FeatMaps
from rdkit import RDConfig
from sklearn.cluster import DBSCAN


# ============== SETUP ==============
fdef = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))

fmParams = {}
for k in fdef.GetFeatureFamilies():
    fparams = FeatMaps.FeatMapParams()
    fmParams[k] = fparams

keep = ('Donor', 'Acceptor', 'NegIonizable', 'PosIonizable', 'ZnBinder',
        'Aromatic', 'Hydrophobe', 'LumpedHydrophobe')


# ============== LOAD AND CENTER WITH GLOBAL CENTROID ==============
def get_feat_map_centered(path_to_dir: str):
    """
    Load molecules from SDF files and center using global centroid.
    
    Args:
        path_to_dir: Directory containing SDF files
        
    Returns:
        feat_list, fms, mol_list, global_centroid
    """
    dir_list = os.listdir(path_to_dir)
    sdf_list = [os.path.join(path_to_dir, f) for f in dir_list if f.endswith('.sdf')]
    
    if not sdf_list:
        raise FileNotFoundError(f"No SDF files found in {path_to_dir}")
    
    # First pass: load all mols and find global centroid
    mol_list = []
    all_centroids = []
    for sdf_path in sdf_list:
        sup = Chem.SDMolSupplier(sdf_path)
        for m in sup:
            if m is not None:
                mol_list.append(m)
                coords = m.GetConformer(0).GetPositions()
                all_centroids.append(coords.mean(axis=0))
    
    if not mol_list:
        raise ValueError(f"No valid molecules loaded from {path_to_dir}")
    
    # Global centroid across all molecules
    global_centroid = np.mean(all_centroids, axis=0)
    print(f"    Global centroid: {global_centroid}")
    
    # Second pass: shift all mols by the SAME offset
    for mol in mol_list:
        conf = mol.GetConformer(0)
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, (pos.x - global_centroid[0], 
                                      pos.y - global_centroid[1], 
                                      pos.z - global_centroid[2]))
    
    feat_list = []
    for m in mol_list:
        raw_feats = fdef.GetFeaturesForMol(m)
        feat_list.append([f for f in raw_feats if f.GetFamily() in keep])
    
    fms = [FeatMaps.FeatMap(feats=x, weights=[1] * len(x), params=fmParams) for x in feat_list]
    
    print(f"    Loaded {len(mol_list)} molecules, extracted {sum(len(f) for f in feat_list)} features")
    return feat_list, fms, mol_list, global_centroid


# ============== CLUSTERING BY FEATURE TYPE ==============
def cluster_features_by_type(feat_list, eps=0.5, min_samples=10, min_samples_override=None):
    """
    Cluster pharmacophore features by type using DBSCAN.
    
    Args:
        feat_list: List of feature lists per molecule
        eps: DBSCAN epsilon (max distance between points in cluster)
        min_samples: Default minimum samples for DBSCAN
        min_samples_override: Dict mapping feature types to custom min_samples
                              e.g. {'Aromatic': 5}
    """
    if min_samples_override is None:
        min_samples_override = {}
    
    all_coords = []
    mol_indices = []
    feat_indices = []
    feat_types = []
    
    for mol_idx, feats in enumerate(feat_list):
        for feat_idx, f in enumerate(feats):
            all_coords.append(f.GetPos())
            mol_indices.append(mol_idx)
            feat_indices.append(feat_idx)
            feat_types.append(f.GetFamily())
    
    X = np.array([[c.x, c.y, c.z] for c in all_coords])
    feat_types = np.array(feat_types)
    mol_indices = np.array(mol_indices)
    feat_indices = np.array(feat_indices)
    
    all_labels = np.full(len(X), -1)
    cluster_offset = 0
    
    for feat_type in keep:
        # Use override if specified, otherwise default
        ms = min_samples_override.get(feat_type, min_samples)
        
        mask = feat_types == feat_type
        if mask.sum() < ms:
            print(f"    {feat_type}: {mask.sum()} features -> skipped (need {ms})")
            continue
        
        X_subset = X[mask]
        clustering = DBSCAN(eps=eps, min_samples=ms).fit(X_subset)
        
        labels_subset = clustering.labels_.copy()
        labels_subset[labels_subset != -1] += cluster_offset
        all_labels[mask] = labels_subset
        
        n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        cluster_offset += n_clusters
        print(f"    {feat_type}: {mask.sum()} features -> {n_clusters} clusters (min_samples={ms})")
    
    return all_labels, mol_indices, feat_indices, X, feat_types


# ============== HOTSPOT EXTRACTION ==============
def get_cluster_hotspots(X, labels, mol_indices, feat_indices, feat_list, min_count=10, min_count_override=None):
    """
    Extract hotspot centers and metadata from clusters.
    
    Args:
        min_count: Default minimum count for a cluster to be kept
        min_count_override: Dict mapping feature types to custom min_count
                            e.g. {'Aromatic': 5}
    """
    if min_count_override is None:
        min_count_override = {}
    
    cluster_centers = []
    cluster_radii = []
    cluster_ids = []
    cluster_counts = []
    cluster_features = []
    
    unique_labels = [l for l in np.unique(labels) if l != -1]
    
    for label in unique_labels:
        mask = labels == label
        count = mask.sum()
        
        # Determine feature type for this cluster
        feat_types_in_cluster = []
        for i, is_in_cluster in enumerate(mask):
            if is_in_cluster:
                mol_idx = mol_indices[i]
                feat_idx = feat_indices[i]
                feat_types_in_cluster.append(feat_list[mol_idx][feat_idx].GetFamily())
        
        dominant_feat = Counter(feat_types_in_cluster).most_common(1)[0][0]
        
        # Use override min_count if specified for this feature type
        mc = min_count_override.get(dominant_feat, min_count)
        
        if count < mc:
            continue
        
        points = X[mask]
        center = points.mean(axis=0)
        radius = np.max(np.linalg.norm(points - center, axis=1))
        
        cluster_centers.append(center)
        cluster_radii.append(radius)
        cluster_ids.append(label)
        cluster_counts.append(count)
        cluster_features.append(dominant_feat)
    
    sorted_idx = np.argsort(cluster_counts)[::-1]
    cluster_centers = np.array(cluster_centers)[sorted_idx] if cluster_centers else np.array([])
    cluster_radii = [cluster_radii[i] for i in sorted_idx]
    cluster_ids = [cluster_ids[i] for i in sorted_idx]
    cluster_counts = [cluster_counts[i] for i in sorted_idx]
    cluster_features = [cluster_features[i] for i in sorted_idx]
    
    return cluster_centers, cluster_radii, cluster_ids, cluster_counts, cluster_features


# ============== CLUSTER COMPOSITION ==============
def get_cluster_composition(labels, mol_indices, feat_indices, feat_list):
    """Get breakdown of feature types within each cluster."""
    cluster_info = {}
    
    for i, label in enumerate(labels):
        if label == -1:
            continue
        
        mol_idx = mol_indices[i]
        feat_idx = feat_indices[i]
        feat_type = feat_list[mol_idx][feat_idx].GetFamily()
        
        if label not in cluster_info:
            cluster_info[label] = []
        cluster_info[label].append(feat_type)
    
    cluster_summary = {}
    for label, feat_types_list in cluster_info.items():
        cluster_summary[label] = dict(Counter(feat_types_list))
    
    return cluster_summary


# ============== DATASET ANALYSIS ==============
def analyse_dataset(mol_list, profile_override=None):
    """
    Analyse reference dataset to build target profile.
    
    Args:
        mol_list: List of reference molecules
        profile_override: Dict to override specific feature profiles
                          e.g. {'Aromatic': (2, 1)} for ideal=2, tolerance=1
    """
    if profile_override is None:
        profile_override = {}
    
    all_feat_counts = []
    
    for mol in mol_list:
        raw_feats = fdef.GetFeaturesForMol(mol)
        mol_feats = [f for f in raw_feats if f.GetFamily() in keep]
        
        feat_counts = {}
        for f in mol_feats:
            ft = f.GetFamily()
            feat_counts[ft] = feat_counts.get(ft, 0) + 1
        all_feat_counts.append(feat_counts)
    
    print(f"\n    Analysed {len(all_feat_counts)} molecules\n")
    
    target_profile = {}
    for feat_type in keep:
        counts = [fc.get(feat_type, 0) for fc in all_feat_counts]
        
        # Check for override first
        if feat_type in profile_override:
            ideal, tolerance = profile_override[feat_type]
        else:
            ideal = int(np.median(counts))
            tolerance = max(1, int(np.std(counts)))  # Never allow tolerance of 0
        
        target_profile[feat_type] = (ideal, tolerance)
        print(f"    {feat_type}: ideal={ideal}, tolerance={tolerance}, data_range=[{np.min(counts)}, {np.max(counts)}]")
    
    return target_profile


# ============== PLOTTING ==============
def plot_cluster_summary(cluster_ids, cluster_counts, cluster_features, output_path):
    """Plot bar chart of cluster sizes by feature type."""
    if not cluster_ids:
        print("    No clusters to plot")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    feature_colours = {
        'Donor': 'blue',
        'Acceptor': 'red',
        'Aromatic': 'orange',
        'NegIonizable': 'green',
        'PosIonizable': 'purple',
        'ZnBinder': 'cyan',
        'Hydrophobe': 'brown',
        'LumpedHydrophobe': 'pink'
    }
    
    colours = [feature_colours.get(f, 'grey') for f in cluster_features]
    xlabels = [f'C{cid} ({feat})' for cid, feat in zip(cluster_ids, cluster_features)]
    
    ax.bar(range(len(cluster_counts)), cluster_counts, color=colours)
    ax.set_xticks(range(len(cluster_counts)))
    ax.set_xticklabels(xlabels, rotation=45, ha='right')
    ax.set_xlabel('Cluster (Feature Type)')
    ax.set_ylabel('Count')
    ax.set_title('Pharmacophore Hotspots by Density (Global Centered)')
    
    from matplotlib.patches import Patch
    present_features = list(set(cluster_features))
    legend_handles = [Patch(color=feature_colours[f], label=f) for f in present_features]
    ax.legend(handles=legend_handles, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"    Saved: {output_path}")
    return fig


def plot_all_features(X, labels, mol_indices, feat_indices, feat_list, min_count, output_path):
    """Plot 3D scatter of all clustered features."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    feature_colours = {
        'Donor': 'blue',
        'Acceptor': 'red',
        'Aromatic': 'orange',
        'NegIonizable': 'green',
        'PosIonizable': 'purple',
        'ZnBinder': 'cyan',
        'Hydrophobe': 'brown',
        'LumpedHydrophobe': 'pink'
    }
    
    feature_points = {ft: [] for ft in feature_colours}
    
    for i, label in enumerate(labels):
        if label == -1:
            continue
        mol_idx = mol_indices[i]
        feat_idx = feat_indices[i]
        feat_type = feat_list[mol_idx][feat_idx].GetFamily()
        if feat_type in feature_colours:
            feature_points[feat_type].append(X[i])
    
    for feat_type, points in feature_points.items():
        if len(points) >= min_count:
            points = np.array(points)
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                       c=feature_colours[feat_type], label=f'{feat_type} (n={len(points)})', alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title('All Features (Global Centered)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved: {output_path}")
    return fig


def run_hotspot_generation(sdf_dir, output_dir, target_name, eps=0.5, min_samples=10, min_count=10,
                           min_samples_override=None, min_count_override=None, profile_override=None):
    """
    Run the complete hotspot generation pipeline.
    
    Args:
        sdf_dir: Directory containing aligned SDF files
        output_dir: Directory to save outputs (pkl and figures)
        target_name: Name of the target for output filenames
        eps: DBSCAN epsilon parameter
        min_samples: Default DBSCAN min_samples
        min_count: Default minimum cluster count
        min_samples_override: Dict of feature-specific min_samples overrides
        min_count_override: Dict of feature-specific min_count overrides
        profile_override: Dict of feature-specific profile overrides
    """
    if min_samples_override is None:
        min_samples_override = {'Aromatic': 5}
    if min_count_override is None:
        min_count_override = {'Aromatic': 5}
    if profile_override is None:
        profile_override = {'Aromatic': (2, 1)}
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"DBSCAN Pharmacophore Hotspot Generation: {target_name}")
    print("(Global Centroid Centering - Preserves Alignment)")
    print("=" * 60)
    
    # 1. Load and center with global centroid
    print("\n[1] Loading molecules and centering with global centroid...")
    feat_list, fms, mol_list, global_centroid = get_feat_map_centered(sdf_dir)
    
    # 2. Cluster by feature type
    print("\n[2] Clustering features by type...")
    labels, mol_indices, feat_indices, X, feat_types = cluster_features_by_type(
        feat_list, 
        eps=eps, 
        min_samples=min_samples,
        min_samples_override=min_samples_override
    )
    
    # 3. Get composition
    print("\n[3] Cluster composition:")
    summary = get_cluster_composition(labels, mol_indices, feat_indices, feat_list)
    for cluster_id in sorted(summary.keys(), reverse=True):
        print(f"    Cluster {cluster_id}: {summary[cluster_id]}")
    
    # 4. Extract hotspots
    print("\n[4] Extracting hotspots...")
    cluster_centers, cluster_radii, cluster_ids, cluster_counts, cluster_features = get_cluster_hotspots(
        X, labels, mol_indices, feat_indices, feat_list, 
        min_count=min_count,
        min_count_override=min_count_override
    )
    
    print("\n    Hotspot Summary:")
    for i in range(len(cluster_ids)):
        print(f"    Cluster {cluster_ids[i]}: {cluster_features[i]}, count={cluster_counts[i]}, radius={cluster_radii[i]:.2f}A, center={cluster_centers[i]}")
    
    # 5. Target profile
    print("\n[5] Analysing dataset for target profile...")
    target_profile = analyse_dataset(mol_list, profile_override=profile_override)
    
    # 6. Plot
    print("\n[6] Plotting...")
    summary_plot_path = os.path.join(output_dir, f'{target_name}_hotspot_summary.png')
    features_plot_path = os.path.join(output_dir, f'{target_name}_features_3d.png')
    
    plot_cluster_summary(cluster_ids, cluster_counts, cluster_features, summary_plot_path)
    plot_all_features(X, labels, mol_indices, feat_indices, feat_list, min_count=5, output_path=features_plot_path)
    
    # 7. Save pickle
    print("\n[7] Saving pkl...")
    pkl_path = os.path.join(output_dir, f'{target_name}_hotspot_data.pkl')
    
    hotspot_data = {
        'cluster_centers': cluster_centers,
        'cluster_radii': cluster_radii,
        'cluster_ids': cluster_ids,
        'cluster_counts': cluster_counts,
        'cluster_features': cluster_features,
        'target_profile': target_profile,
        'keep': keep,
        'global_centroid': global_centroid,
        'metadata': {
            'eps': eps,
            'min_samples': min_samples,
            'min_samples_override': min_samples_override,
            'min_count': min_count,
            'min_count_override': min_count_override,
            'profile_override': profile_override,
            'sigma': 0.5,
            'cutoff_distance': 2.0,
            'n_clusters': len(cluster_centers) if len(cluster_centers) > 0 else 0,
            'n_molecules_analysed': len(mol_list),
            'source_dir': sdf_dir,
            'centering_method': 'global_centroid',
            'global_centroid': global_centroid.tolist(),
            'notes': 'Clustered by feature type with aromatic-specific min_samples=5. All ligands shifted by same global centroid to preserve alignment.'
        }
    }
    
    with open(pkl_path, 'wb') as f:
        pickle.dump(hotspot_data, f)
    
    print(f"\n    Saved to {pkl_path}")
    print(f"    Global centroid used: {global_centroid}")
    print(f"    Number of clusters: {len(cluster_centers) if len(cluster_centers) > 0 else 0}")
    print(f"    Aromatic clusters: {sum(1 for f in cluster_features if f == 'Aromatic')}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    
    return hotspot_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate pharmacophore hotspots from aligned SDF files using DBSCAN"
    )
    parser.add_argument(
        "--sdf_dir",
        required=True,
        help="Directory containing aligned SDF files (e.g., FEATURE_MAP_ALIGNED)"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save outputs (pkl and figures)"
    )
    parser.add_argument(
        "--target_name",
        required=True,
        help="Name of the target for output filenames"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.5,
        help="DBSCAN epsilon parameter (default: 0.5)"
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=10,
        help="DBSCAN min_samples parameter (default: 10)"
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=10,
        help="Minimum cluster count to keep (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Default overrides for sparser feature types
    min_samples_override = {'Aromatic': 5}
    min_count_override = {'Aromatic': 5}
    profile_override = {'Aromatic': (2, 1)}
    
    run_hotspot_generation(
        sdf_dir=args.sdf_dir,
        output_dir=args.output_dir,
        target_name=args.target_name,
        eps=args.eps,
        min_samples=args.min_samples,
        min_count=args.min_count,
        min_samples_override=min_samples_override,
        min_count_override=min_count_override,
        profile_override=profile_override
    )


if __name__ == "__main__":
    main()