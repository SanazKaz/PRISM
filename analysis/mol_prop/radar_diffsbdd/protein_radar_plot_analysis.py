from math import pi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("/data/stat-cadd/stat0548/PRISM/analysis/mol_prop/radar_diffsbdd/molprop_reward_protein_plot")
PROPERTY_ORDER = ['MW','AliR_C','AroR_C','ChiA_C','SA','NHOH_C','HetA_C','RotB_C','BriA_C','HBD_C','HBA_C','FusedR_C','LogP_C']

PROTEIN_GROUPS = {
    'ABL':  ['ABL_dataset_2HZ0_GIN_processed', 'ABL_dataset_2HZI_JIN_processed', 'ABL_dataset_5HU9_66K_processed'],
    'DPP':  ['DPP_dataset_1RWQ_5AP_processed', 'DPP_dataset_2G63_AAF_processed', 'DPP_dataset_2OGZ_U1N_processed',
             'DPP_dataset_2QJR_PZF_processed', 'DPP_dataset_3EIO_AJH_processed', 'DPP_dataset_3SWW_KXB_processed',
             'DPP_dataset_3VJM_W61_processed', 'DPP_dataset_5KBY_6RL_processed'],
    'MTB':  ['MTB_dataset_4UNR_QZE_processed', 'MTB_dataset_6YT1_PK5_processed'],
    'PARP': ['PARP_dataset_5WS1_7U9_processed', 'PARP_dataset_7AAC_78P_processed', 'PARP_dataset_7ONT_VKQ_processed'],
}

COLORS = ['#408EC6', '#E84646', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C', '#E74C3C', '#3498DB']


def make_radar(ax, values, labels, color, alpha=0.25, label=None):
    N = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    vals = list(values) + [values[0]]
    ax.set_facecolor('white')
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9, fontweight='bold')
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=7)
    ax.spines['polar'].set_color('black')
    ax.spines['polar'].set_linewidth(2)
    ax.xaxis.grid(True, color='grey', linewidth=1, alpha=0.8)
    ax.yaxis.grid(True, color='grey', linewidth=1, alpha=0.8)
    ax.plot(angles, vals, color=color, linewidth=3, linestyle='solid', label=label)
    ax.fill(angles, vals, color=color, alpha=alpha)


def main():
    df = pd.read_csv(Path("/data/stat-cadd/stat0548/PRISM/analysis/mol_prop/radar_diffsbdd/molprop_reward_summary_plot/all_raw_properties.csv"))

    for protein, sources in PROTEIN_GROUPS.items():
        subset = df[df['source'].isin(sources)]

        # --- kombinierter Plot (alle Moleküle zusammen) ---
        combined_means = subset[PROPERTY_ORDER].mean().values
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        make_radar(ax, combined_means, PROPERTY_ORDER, color='#408EC6', label=protein)
        ax.set_title(f"{protein} – combined", size=13, pad=15)
        fig.savefig(OUTPUT_DIR / f"{protein}_combined_radar.png", dpi=300, bbox_inches='tight')
        fig.savefig(OUTPUT_DIR / f"{protein}_combined_radar.svg", bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {protein}_combined_radar")

        # --- Overlay Plot (jede PDB einzeln) ---
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        for i, src in enumerate(sources):
            grp = df[df['source'] == src][PROPERTY_ORDER].mean().values
            pdb = src.split('_')[2]  # z.B. '2HZ0'
            make_radar(ax, grp, PROPERTY_ORDER, color=COLORS[i % len(COLORS)], alpha=0.15, label=pdb)
        ax.set_title(f"{protein} – overlay by PDB", size=13, pad=15)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
        fig.savefig(OUTPUT_DIR / f"{protein}_overlay_radar.png", dpi=300, bbox_inches='tight')
        fig.savefig(OUTPUT_DIR / f"{protein}_overlay_radar.svg", bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {protein}_overlay_radar")

        # --- Gesamt-Overlay: alle Proteine kombiniert ---
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for i, (protein, sources) in enumerate(PROTEIN_GROUPS.items()):
        subset = df[df['source'].isin(sources)]
        means = subset[PROPERTY_ORDER].mean().values
        vals = list(means) + [means[0]]
        angles = [n / float(len(PROPERTY_ORDER)) * 2 * np.pi for n in range(len(PROPERTY_ORDER))]
        angles += angles[:1]
        ax.plot(angles, vals, color=COLORS[i], linewidth=3, linestyle='solid', label=protein)
        ax.fill(angles, vals, color=COLORS[i], alpha=0.1)

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
    ax.set_title("All Proteins – combined overlay", size=14, y=1.08, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    fig.savefig(OUTPUT_DIR / "all_proteins_overlay.png", dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / "all_proteins_overlay.svg", bbox_inches='tight')
    plt.close(fig)
    print("Saved: all_proteins_overlay")

    print("\nDONE!")


if __name__ == "__main__":
    main()
