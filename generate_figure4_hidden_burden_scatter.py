from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from figure_utils import setup_publication_style, save_figure

setup_publication_style()

# Load model from the same directory as this script
script_dir = Path(__file__).resolve().parent
model_path = script_dir / "assisted_ventilation_model_final.py"

spec = importlib.util.spec_from_file_location("model", model_path)
model = importlib.util.module_from_spec(spec)
sys.modules["model"] = model
spec.loader.exec_module(model)

df = model.run_parameter_sweep()

# A single colorbar requires the same ECF normalization in every panel.
ecf_min = float(df["ECF"].min())
ecf_max = float(df["ECF"].max())

phenotypes = [
    "compliance_dominant",
    "resistance_dominant",
    "mixed",
    "severe_mixed",
]

panel_titles = {
    "compliance_dominant": "A. Compliance-dominant",
    "resistance_dominant": "B. Resistance-dominant",
    "mixed": "C. Mixed",
    "severe_mixed": "D. Severe mixed",
}

# Global axis limits for direct comparison across panels
xmin = float(df["MP_vent_Jmin"].min())
xmax = float(df["MP_vent_Jmin"].max())
ymin = float(df["MP_tot_Jmin"].min())
ymax = float(df["MP_tot_Jmin"].max())

lower = 0.0
upper = max(xmax, ymax) * 1.05

fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.0), sharex=True, sharey=True)

sc = None

for ax, phenotype in zip(axes.flat, phenotypes):
    sub = df[df["phenotype"] == phenotype].copy()

    sc = ax.scatter(
        sub["MP_vent_Jmin"],
        sub["MP_tot_Jmin"],
        c=sub["ECF"],
        vmin=ecf_min,
        vmax=ecf_max,
        s=52,
        alpha=0.90,
        edgecolors="none"
    )

    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        linewidth=1.5,
        color="black",
        alpha=0.8
    )

    ax.set_title(panel_titles[phenotype], fontsize=10)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)

    ax.text(
        0.03,
        0.97,
        r"MP$_{tot}$/MP$_{vent}$: "
        + f"{sub['HBR'].min():.2f}–{sub['HBR'].max():.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="0.5",
            linewidth=0.8
        )
    )

for ax in axes[1, :]:
    ax.set_xlabel("Ventilator-derived mechanical power (J/min)")
for ax in axes[:, 0]:
    ax.set_ylabel("Total modeled mechanical power (J/min)")

# Reserve a separate right-hand strip; the colorbar must not overlap a panel.
fig.subplots_adjust(
    left=0.10,
    right=0.86,
    bottom=0.10,
    top=0.93,
    wspace=0.22,
    hspace=0.28
)
colorbar_ax = fig.add_axes([0.895, 0.21, 0.025, 0.59])
cbar = fig.colorbar(sc, cax=colorbar_ax)
cbar.set_label("ECF (dimensionless)")

save_figure(fig, "Figure4_hidden_burden_scatter")