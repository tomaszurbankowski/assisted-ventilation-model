from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def setup_publication_style() -> None:
    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })


def save_figure(fig: plt.Figure, filename_base: str, output_dir: str = "outputs") -> None:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    png_path = outdir / f"{filename_base}.png"
    pdf_path = outdir / f"{filename_base}.pdf"

    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


setup_publication_style()

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "assisted_ventilation_model_final.py"

spec = importlib.util.spec_from_file_location("model", MODEL_PATH)
model = importlib.util.module_from_spec(spec)
sys.modules["model"] = model
spec.loader.exec_module(model)

df = model.run_parameter_sweep()
mv = model.matched_pairs(df, match_variable="MP_vent_Jmin", tolerance=0.05)
mt = model.matched_pairs(df, match_variable="MP_tot_Jmin", tolerance=0.05)

metrics = [
    ("delta_MP_tot", "A. Δ total MP", "J/min"),
    ("delta_EII", "B. Δ EII", "Absolute difference"),
    ("delta_REF1", "C. Δ EF₁", "Absolute difference"),
    ("delta_ECF", "D. Δ ECF", "Absolute difference"),
    ("delta_HBR", "E. Δ HBR", "Absolute difference"),
]

fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.6))

box_facecolors = ["#bdbdbd", "#e0e0e0"]
edgecolor = "black"

for ax, (metric, title, ylabel) in zip(axes, metrics):
    vals = [mv[metric].values, mt[metric].values]

    bp = ax.boxplot(
        vals,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        tick_labels=["Vent MP", "Total MP"],
        medianprops=dict(color="black", linewidth=1.5),
        boxprops=dict(linewidth=1.0, color=edgecolor),
        whiskerprops=dict(linewidth=1.0, color=edgecolor),
        capprops=dict(linewidth=1.0, color=edgecolor),
    )

    for patch, fc in zip(bp["boxes"], box_facecolors):
        patch.set_facecolor(fc)
        patch.set_edgecolor(edgecolor)

    ax.set_title(title, pad=8)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.15, linewidth=0.5)
    ax.set_ylim(bottom=0)

    ax.text(
        0.5, 0.92,
        f"n={len(mv)} vs n={len(mt)}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7,
        bbox=dict(
            boxstyle="round,pad=0.15",
            facecolor="white",
            edgecolor="0.8"
        )
    )

fig.tight_layout(w_pad=0.8)
save_figure(fig, "Figure5_matched_pair_summary")