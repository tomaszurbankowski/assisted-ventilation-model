from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import numpy as np
import matplotlib.pyplot as plt


# ======================================
# Paths
# ======================================
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = BASE_DIR / "assisted_ventilation_model_final.py"


# ======================================
# Publication-style defaults
# ======================================
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ======================================
# Dynamic import of model
# ======================================
spec = importlib.util.spec_from_file_location("assisted_model", MODEL_PATH)
model = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = model
spec.loader.exec_module(model)


# ======================================
# Data preparation
# ======================================
df = model.run_parameter_sweep()

# Figure 3: fix effort duration at 1.0 s
df = df[df["Tmus_s"] == 1.0].copy()

phenotypes = [
    "compliance_dominant",
    "resistance_dominant",
    "mixed",
    "severe_mixed",
]

phenotype_titles = {
    "compliance_dominant": "Compliance-\ndominant",
    "resistance_dominant": "Resistance-\ndominant",
    "mixed": "Mixed",
    "severe_mixed": "Severe\nmixed",
}

metrics = [
    ("MP_tot_Jmin", "MPtot (J/min)"),
    ("EII", "EII"),
    ("HBR", "HBR"),
]

pressure_support_levels = [5.0, 10.0, 15.0]
pmus_levels = [12.0, 9.0, 6.0, 3.0, 0.0]   # descending for display


# ======================================
# Precompute pivots and row-wise scales
# ======================================
pivot_data = {}
row_limits = {}

for metric, _ in metrics:
    row_min = np.inf
    row_max = -np.inf
    pivot_data[metric] = {}

    for phenotype in phenotypes:
        sub = df[df["phenotype"] == phenotype]

        pivot = (
            sub.pivot(
                index="Pmus_peak_cmH2O",
                columns="PS_cmH2O",
                values=metric,
            )
            .reindex(index=pmus_levels, columns=pressure_support_levels)
        )

        pivot_data[metric][phenotype] = pivot
        row_min = min(row_min, np.nanmin(pivot.values))
        row_max = max(row_max, np.nanmax(pivot.values))

    row_limits[metric] = (row_min, row_max)


# ======================================
# Figure generation
# ======================================
fig, axes = plt.subplots(
    nrows=len(metrics),
    ncols=len(phenotypes),
    figsize=(11.2, 7.4),
    sharex=False,
    sharey=False,
)

# leave room on the right for one shared colorbar per row
fig.subplots_adjust(
    left=0.08,
    right=0.90,
    bottom=0.09,
    top=0.96,
    wspace=0.18,
    hspace=0.18,
)

for r, (metric, row_label) in enumerate(metrics):
    vmin, vmax = row_limits[metric]
    last_im = None

    for c, phenotype in enumerate(phenotypes):
        ax = axes[r, c]
        pivot = pivot_data[metric][phenotype]

        im = ax.imshow(
            pivot.values,
            aspect="auto",
            origin="upper",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        last_im = im

        # Column titles
        if r == 0:
            ax.set_title(phenotype_titles[phenotype], pad=8)

        # X axis
        ax.set_xticks(range(len(pressure_support_levels)))
        ax.set_xticklabels([f"{x:.0f}" for x in pressure_support_levels])
        if r == len(metrics) - 1:
            ax.set_xlabel("Pressure support (cmH₂O)")
        else:
            ax.set_xlabel("")

        # Y axis
        ax.set_yticks(range(len(pmus_levels)))
        ax.set_yticklabels([f"{y:.0f}" for y in pmus_levels])
        if c == 0:
            ax.set_ylabel(f"{row_label}\nPmus peak (cmH₂O)")
        else:
            ax.set_ylabel("")

        # Thin white grid between cells
        ax.set_xticks(np.arange(-0.5, len(pressure_support_levels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(pmus_levels), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

    # One shared colorbar per row
    cbar = fig.colorbar(
        last_im,
        ax=axes[r, :],
        location="right",
        fraction=0.025,
        pad=0.02,
    )
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_linewidth(0.8)


# ======================================
# Save
# ======================================
out_png = OUTPUT_DIR / "Figure3_parameter_sweep_heatmaps.png"
out_svg = OUTPUT_DIR / "Figure3_parameter_sweep_heatmaps.svg"

plt.savefig(out_png, bbox_inches="tight")
plt.savefig(out_svg, bbox_inches="tight")
plt.close(fig)

print("Saved:")
print(out_png)
print(out_svg)