"""Radar plot analysis for chemical property distributions.

adapted from 
https://github.com/jianingli-purdue/Benchmarking_gene_model/blob/main/Picture_drawing.ipynb
Yang et al https://pubs.acs.org/doi/10.1021/acs.jmedchem.5c01706
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
import json
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski
from rdkit.Chem import Crippen

# changed: removed src.utils import, no longer needed
# from src.utils import load_molecules, save_figure

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# =============================================================================
# changed: paths to SDF files
# =============================================================================
DATA_DIR   = Path("/data/stat-cadd/stat0548/PRISM/results/diffsbdd/Test3protein_ligands")  
OUTPUT_DIR = Path("/data/stat-cadd/stat0548/PRISM/analysis/mol_prop/radar_diffsbdd/withoutreward_summary_plot")  

# changed: automatically load all SDF files from folder
INPUT_FILES = {
    sdf.stem: sdf
    for sdf in sorted(DATA_DIR.glob("*/*_processed.sdf"))
}

# =============================================================================
# Property denominators from the paper for normalization
# =============================================================================
PROPERTY_DENOMINATORS = {
    'MW':      500,
    'AliR_C':  4,
    'AroR_C':  3,
    'ChiA_C':  6,
    'SA':      6,
    'NHOH_C':  6,
    'HetA_C':  10,
    'RotB_C':  8,
    'BriA_C':  2,
    'HBD_C':   5,
    'HBA_C':   10,
    'FusedR_C': 6,
    'LogP_C':  5,
}

PROPERTY_ORDER = [
    'MW', 'AliR_C', 'AroR_C', 'ChiA_C', 'SA', 'NHOH_C',
    'HetA_C', 'RotB_C', 'BriA_C', 'HBD_C', 'HBA_C', 'FusedR_C', 'LogP_C']

# changed: automatic color palette for 41 files
COLORS = [
    '#E74C3C', '#E67E22', '#F1C40F', '#27AE60', '#1ABC9C',
    '#2980B9', '#8E44AD', '#2C3E50', '#D35400', '#16A085',
    '#C0392B', '#7D3C98', '#1F618D', '#148F77', '#B7950B',
    '#784212', '#1A5276', '#0E6655', '#6E2F1A', '#4A235A',
]


# =============================================================================
# changed: load_molecules_from_sdf replaces load_molecules from src.utils
# =============================================================================
def load_molecules_from_sdf(sdf_path) -> list:
    """
    Load molecules from an SDF file.
    Replaces load_molecules() from src.utils.
    
    Args:
        sdf_path: Path to SDF file
        
    Returns:
        List of RDKit molecule objects
    """
    path = Path(sdf_path)
    if not path.exists():
        print(f"  File not found: {path}")
        return []
    mols = []
    n_failed = 0
    for mol in Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True):
        if mol is not None:
            mols.append(mol)
        else:
            n_failed += 1
    print(f"  {len(mols)} molecules loaded ({n_failed} invalid skipped)")
    return mols


# changed: save_figure now defined locally, no longer from src.utils
def save_figure(fig, output_path: Path, formats: list = None, dpi: int = 300):
    """
    Save matplotlib figure in one or more formats.
    Replaces save_figure() from src.utils.
    
    Args:
        fig:         Matplotlib figure
        output_path: Path WITHOUT file extension
        formats:     List of formats e.g. ['png', 'svg']
        dpi:         Resolution for raster formats
    """
    if formats is None:
        formats = ['png']
    for fmt in formats:
        full_path = output_path.with_suffix(f'.{fmt}')
        full_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(full_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', transparent=False)
        print(f"  Saved: {full_path}")


# =============================================================================
# Property calculation (unchanged)
# =============================================================================
def calculate_sa_score(mol):
    """
    Calculate synthetic accessibility score.
    
    Tries to import sascorer, falls back to a simple heuristic if unavailable.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        SA score (1-10 scale, lower is more accessible)
    """
    try:
        from sascorer import calculateScore
        return calculateScore(mol)
    except ImportError:
        # Fallback: simple heuristic based on complexity
        n_rings      = rdMolDescriptors.CalcNumRings(mol)
        n_stereo     = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        n_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        score = 2.0 + (n_rings * 0.5) + (n_stereo * 0.3) + (n_bridgehead * 0.5)
        return min(score, 10.0)


def compute_properties_for_mol(mol) -> dict:
    """
    Compute the 13 normalized properties for a single molecule.
    
    Each property is divided by its denominator as per the paper's methodology.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Dictionary of property_name -> normalized value (0-1 scale)
    """
    if mol is None:
        return None
    try:
        # 1) Molecular weight
        mol_wt           = Descriptors.MolWt(mol)
        # 2) Aliphatic ring count
        aliph_ring_count = Lipinski.NumAliphaticRings(mol)
        # 3) Aromatic ring count
        arom_ring_count  = Lipinski.NumAromaticRings(mol)
        # 4) Chiral atom count
        chiral_count     = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        # 5) Synthetic accessibility score
        sa_score         = calculate_sa_score(mol)
        # 6) NH/OH count
        nh_oh_count      = Lipinski.NHOHCount(mol)
        # 7) Heteroatom count (not C or H)
        heteroatom_count = sum(1 for atom in mol.GetAtoms()
                               if atom.GetAtomicNum() not in (1, 6))
        # 8) Rotatable bonds
        rot_bonds        = Descriptors.NumRotatableBonds(mol)
        # 9) Bridgehead atom count
        bridgehead_count = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        # 10) Hydrogen bond donors and acceptors
        hbd_count        = rdMolDescriptors.CalcNumHBD(mol)
        hba_count        = rdMolDescriptors.CalcNumHBA(mol)
        # 11) Fused ring count
        ring_info        = mol.GetRingInfo()
        fused_count      = sum(1 for i in range(ring_info.NumRings())
                               if ring_info.IsRingFused(i))
        # 12) LogP
        logp_count       = Crippen.MolLogP(mol)

        # Return normalized values (divided by denominators)
        return {
            'MW':       mol_wt           / PROPERTY_DENOMINATORS['MW'],
            'AliR_C':   aliph_ring_count / PROPERTY_DENOMINATORS['AliR_C'],
            'AroR_C':   arom_ring_count  / PROPERTY_DENOMINATORS['AroR_C'],
            'ChiA_C':   chiral_count     / PROPERTY_DENOMINATORS['ChiA_C'],
            'SA':       sa_score         / PROPERTY_DENOMINATORS['SA'],
            'NHOH_C':   nh_oh_count      / PROPERTY_DENOMINATORS['NHOH_C'],
            'HetA_C':   heteroatom_count / PROPERTY_DENOMINATORS['HetA_C'],
            'RotB_C':   rot_bonds        / PROPERTY_DENOMINATORS['RotB_C'],
            'BriA_C':   bridgehead_count / PROPERTY_DENOMINATORS['BriA_C'],
            'HBD_C':    hbd_count        / PROPERTY_DENOMINATORS['HBD_C'],
            'HBA_C':    hba_count        / PROPERTY_DENOMINATORS['HBA_C'],
            'FusedR_C': fused_count      / PROPERTY_DENOMINATORS['FusedR_C'],
            'LogP_C':   logp_count       / PROPERTY_DENOMINATORS['LogP_C'],
        }
    except Exception as e:
        print(f"  Warning: Could not calculate properties: {e}")
        return None


def compute_mean_properties(molecules: list) -> dict:
    """
    Compute mean normalized properties across a list of molecules.
    
    Args:
        molecules: List of RDKit molecule objects
        
    Returns:
        Dictionary of property_name -> mean normalized value
    """
    properties_list = {prop: [] for prop in PROPERTY_ORDER}
    for mol in molecules:
        props = compute_properties_for_mol(mol)
        if props is not None:
            for k, v in props.items():
                properties_list[k].append(v)
    return {
        k: np.mean(v) if v else 0.0
        for k, v in properties_list.items()
    }


def compute_property_statistics(molecules: list) -> dict:
    """
    Compute detailed statistics (mean, std, min, max, range) for properties.
    
    Args:
        molecules: List of RDKit molecule objects
        
    Returns:
        Dictionary of property_name -> dict of statistics
    """
    properties_list = {prop: [] for prop in PROPERTY_ORDER}
    for mol in molecules:
        props = compute_properties_for_mol(mol)
        if props is not None:
            for k, v in props.items():
                properties_list[k].append(v)
    statistics = {}
    for prop, values in properties_list.items():
        if values:
            statistics[prop] = {
                'mean':      float(np.mean(values)),
                'std':       float(np.std(values)),
                'min':       float(np.min(values)),
                'max':       float(np.max(values)),
                'range':     float(np.max(values) - np.min(values)),
                'median':    float(np.median(values)),
                'n_samples': len(values),
            }
        else:
            statistics[prop] = {
                'mean': 0.0, 'std': 0.0, 'min': 0.0,
                'max': 0.0, 'range': 0.0, 'median': 0.0, 'n_samples': 0
            }
    return statistics


def _stats_to_json_serializable(stats: dict) -> dict:
    """Convert per-property stats (with numpy types) to native Python for JSON."""
    out = {}
    for prop, d in stats.items():
        out[prop] = {
            'mean':      float(d['mean']),
            'std':       float(d['std']),
            'min':       float(d['min']),
            'max':       float(d['max']),
            'range':     float(d['range']),
            'median':    float(d['median']),
            'n_samples': int(d['n_samples']),
        }
    return out


# =============================================================================
# Plot functions
# =============================================================================
def plot_radar_single(property_dict: dict, title: str, output_path: Path,
                      color: str = '#408EC6', figure_formats: list = None,
                      dpi: int = 300):
    """
    Plot a single radar chart matching the paper's style.
    
    Args:
        property_dict:  Dictionary of property_name -> normalized value
        title:          Plot title
        output_path:    Path for output figure
        color:          Fill/line color
        figure_formats: List of formats to save
        dpi:            Resolution for raster formats
    """
    if figure_formats is None:
        figure_formats = ['png', 'svg']

    values = [property_dict[cat] for cat in PROPERTY_ORDER]
    N      = len(PROPERTY_ORDER)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # close the plot
    values += values[:1]  # close the plot

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'polar': True})
    fig.patch.set_facecolor('white')  # changed: white instead of transparent
    ax.set_facecolor('white')         # changed: white instead of transparent
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(PROPERTY_ORDER, fontsize=10, fontweight='bold')
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=8)
    ax.spines['polar'].set_color('black')
    ax.spines['polar'].set_linewidth(2)
    ax.xaxis.grid(True, color='grey', linewidth=1, alpha=0.8)
    ax.yaxis.grid(True, color='grey', linewidth=1, alpha=0.8)
    ax.plot(angles, values, color=color, linewidth=3, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.3)
    ax.set_title(title, size=14, y=1.08, fontweight='bold', color='black')
    plt.tight_layout()
    save_figure(fig, output_path, formats=figure_formats, dpi=dpi)
    plt.close()


def plot_radar_grid(all_properties: dict, output_path: Path,
                    figure_formats: list = None, dpi: int = 300):
    """
    Create a grid of radar plots like the paper figure.
    
    Args:
        all_properties: Dict mapping source names to their property dicts
        output_path:    Path for output figure
        figure_formats: List of formats to save
        dpi:            Resolution for raster formats
    """
    if figure_formats is None:
        figure_formats = ['png', 'svg']

    sources  = list(all_properties.keys())
    n_cols   = min(3, len(sources))
    n_rows   = (len(sources) + n_cols - 1) // n_cols
    N        = len(PROPERTY_ORDER)
    angles   = [n / float(N) * 2 * pi for n in range(N)]
    angles  += angles[:1]

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 5 * n_rows),
                             subplot_kw={'polar': True})
    fig.patch.set_facecolor('white')  # changed
    axes = [axes] if len(sources) == 1 else axes.flatten()

    for idx, source in enumerate(sources):
        ax     = axes[idx]
        values = [all_properties[source][cat] for cat in PROPERTY_ORDER]
        values += values[:1]
        color  = COLORS[idx % len(COLORS)]  # changed: automatic color per file

        ax.set_facecolor('white')  # changed
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(PROPERTY_ORDER, fontsize=9, fontweight='bold')
        ax.set_rlabel_position(0)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=7)
        ax.spines['polar'].set_color('black')
        ax.spines['polar'].set_linewidth(2)
        ax.xaxis.grid(True, color='grey', linewidth=1, alpha=0.8)
        ax.yaxis.grid(True, color='grey', linewidth=1, alpha=0.8)
        ax.plot(angles, values, color=color, linewidth=3, linestyle='solid')
        ax.fill(angles, values, color=color, alpha=0.3)
        ax.set_title(source, size=8, y=1.08, fontweight='bold')  # changed: smaller font for long names

    # Hide empty subplots
    for idx in range(len(sources), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    save_figure(fig, output_path, formats=figure_formats, dpi=dpi)
    plt.close()


def plot_radar_overlay(all_properties: dict, title: str, output_path: Path,
                       figure_formats: list = None, dpi: int = 300):
    """
    Create a single radar plot with all sources overlaid for comparison.
    
    Args:
        all_properties: Dict mapping source names to their property dicts
        title:          Plot title
        output_path:    Path for output figure
        figure_formats: List of formats to save
        dpi:            Resolution for raster formats
    """
    if figure_formats is None:
        figure_formats = ['png', 'svg']

    N       = len(PROPERTY_ORDER)
    angles  = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'polar': True})
    fig.patch.set_facecolor('white')  # changed
    ax.set_facecolor('white')         # changed
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    for idx, (source, property_dict) in enumerate(all_properties.items()):
        values  = [property_dict[cat] for cat in PROPERTY_ORDER]
        values += values[:1]
        color   = COLORS[idx % len(COLORS)]  # changed: automatic color
        ax.plot(angles, values, color=color, linewidth=2.5,
                linestyle='solid', label=source)
        ax.fill(angles, values, color=color, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(PROPERTY_ORDER, fontsize=11, fontweight='bold')
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=9)
    ax.spines['polar'].set_color('black')
    ax.spines['polar'].set_linewidth(2)
    ax.xaxis.grid(True, color='grey', linewidth=1, alpha=0.8)
    ax.yaxis.grid(True, color='grey', linewidth=1, alpha=0.8)
    ax.set_title(title, size=14, y=1.08, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1.15, 0.5),
              frameon=True, fontsize=7)  # changed: smaller font for many entries

    plt.tight_layout(rect=[0, 0, 0.75, 1])  # changed: more space for legend
    save_figure(fig, output_path, formats=figure_formats, dpi=dpi)
    plt.close()


# =============================================================================
# changed: run_radar_analysis completely rewritten - no config object needed
# =============================================================================
def run_radar_analysis(input_files: dict, output_dir: Path,
                       formats: list = None, dpi: int = 300):
    """
    Main function: loads all SDF files, computes properties, creates plots.

    Args:
        input_files: Dict { name: sdf_path }
        output_dir:  Output folder
        formats:     Plot formats
        dpi:         Resolution
    """
    if formats is None:
        formats = ['png', 'svg']

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print("RADAR PLOT ANALYSIS")
    print(f"{'='*60}")
    print(f"SDF files found: {len(input_files)}")
    print(f"Output folder: {output_dir.resolve()}")

    all_mean_properties = {}
    all_statistics      = {}
    all_raw_rows        = []

    for idx, (name, sdf_path) in enumerate(input_files.items()):
        print(f"\n{'─'*50}")
        print(f"[{idx+1}/{len(input_files)}] {name}")

        # changed: load SDF instead of CSV
        mols = load_molecules_from_sdf(sdf_path)
        if not mols:
            print(f"  WARNING: No molecules found - skipping")
            continue

        mean_props = compute_mean_properties(mols)
        stats      = compute_property_statistics(mols)
        all_mean_properties[name] = mean_props
        all_statistics[name]      = stats

        print(f"  Properties computed for {len(mols)} molecules")

        # Collect raw data
        for mol in mols:
            props = compute_properties_for_mol(mol)
            if props:
                props['source'] = name
                all_raw_rows.append(props)

        # Single radar plot per SDF
        color = COLORS[idx % len(COLORS)]
        plot_radar_single(
            mean_props,
            title=name,
            output_path=output_dir / f'{name}_radar',
            color=color,
            figure_formats=formats,
            dpi=dpi
        )

    if not all_mean_properties:
        print("\nERROR: No data loaded - aborting.")
        return

    # Grid plot (all together)
    print(f"\n{'─'*50}")
    print("Creating grid plot...")
    plot_radar_grid(
        all_mean_properties,
        output_dir / 'all_radar_grid',
        figure_formats=formats,
        dpi=dpi
    )

    # Overlay plot
    print("Creating overlay plot...")
    plot_radar_overlay(
        all_mean_properties,
        title="DiffSBDD - Chemical Property Comparison",
        output_path=output_dir / 'all_radar_overlay',
        figure_formats=formats,
        dpi=dpi
    )

    # Save mean properties CSV
    mean_df = pd.DataFrame(all_mean_properties).T
    mean_df.index.name = 'Source'
    mean_df.to_csv(output_dir / 'mean_properties.csv')
    print(f"\nSaved means: {output_dir / 'mean_properties.csv'}")

    # Save raw data CSV
    if all_raw_rows:
        pd.DataFrame(all_raw_rows).to_csv(
            output_dir / 'all_raw_properties.csv', index=False)
        print(f"Saved raw data: {output_dir / 'all_raw_properties.csv'}")

    # Save statistics JSON
    with open(output_dir / 'property_statistics.json', 'w') as f:
        json.dump(
        {k: _stats_to_json_serializable(stats)
         for k, stats in all_statistics.items()},
        f, indent=2
    )
    print(f"Saved statistics: {output_dir / 'property_statistics.json'}")

    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}\n")


# =============================================================================
# changed: entry point without config object
# =============================================================================
if __name__ == "__main__":
    run_radar_analysis(
        input_files=INPUT_FILES,
        output_dir=OUTPUT_DIR,
        formats=['png', 'svg'],
        dpi=300
    )