from pathlib import Path
import importlib.util
import sys

import matplotlib.pyplot as plt
import numpy as np


# ======================================
# Paths
# ======================================
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = BASE_DIR / "assisted_ventilation_model_final.py"


# ======================================
# Dynamic import of model
# ======================================
spec = importlib.util.spec_from_file_location("assisted_model", MODEL_PATH)
model = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = model
spec.loader.exec_module(model)

VentilatorSettings = model.VentilatorSettings
EffortSettings = model.EffortSettings
phenotype_definition = model.phenotype_definition
simulate_breath = model.simulate_breath


# ======================================
# Figure generation
# ======================================
def main():
    # Representative scenario for waveform illustration
    phenotype = "mixed"

    ph = phenotype_definition(phenotype)
    comp1 = ph["comp1"]
    comp2 = ph["comp2"]

    vent = VentilatorSettings(
        peep=5.0,
        pressure_support=5.0,
        respiratory_rate=18.0,
        inspiratory_time=1.0,
        dt=0.001,
    )

    effort = EffortSettings(
        pmus_peak=3.0,
        effort_duration=1.0,
        onset=0.0,
    )

    sim = simulate_breath(comp1, comp2, vent, effort)

    t = np.asarray(sim["t"])
    paw_above_peep = np.asarray(sim["paw_above_peep"])
    pmus = np.asarray(sim["pmus"])
    pdist = np.asarray(sim["pdist"])

    f1 = np.asarray(sim["f1"])
    f2 = np.asarray(sim["f2"])
    f_tot = np.asarray(sim["f_tot"])

    v1 = np.asarray(sim["v1"])
    v2 = np.asarray(sim["v2"])
    vtot = np.asarray(sim["vtot"])

    cycle_time = 60.0 / vent.respiratory_rate

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.2), sharex=True)

    # Common dash-dot pattern for total / combined variables
    dashdot_style = (0, (7, 2, 1.6, 2))

    # Panel A: Pressures
    ax = axes[0]
    ax.plot(t, paw_above_peep, linewidth=2.0, label="Paw above PEEP")
    ax.plot(t, pmus, linewidth=2.0, linestyle="--", label="Pmus")
    ax.plot(t, pdist, linewidth=2.2, linestyle=dashdot_style, label="Pdist")
    ax.set_ylabel("Pressure (cmH₂O)", fontsize=10)
    ax.text(
        0.01, 0.93, "A",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )

    # Panel B: Flows
    ax = axes[1]
    ax.plot(t, f1, linewidth=1.8, label="Flow compartment 1")
    ax.plot(t, f2, linewidth=1.8, linestyle="--", label="Flow compartment 2")
    ax.plot(t, f_tot, linewidth=2.2, linestyle=dashdot_style, label="Total flow")
    ax.set_ylabel("Flow (L/s)", fontsize=10)
    ax.text(
        0.01, 0.93, "B",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )

    # Panel C: Volumes
    ax = axes[2]
    ax.plot(t, v1, linewidth=1.8, label="Volume compartment 1")
    ax.plot(t, v2, linewidth=1.8, linestyle="--", label="Volume compartment 2")
    ax.plot(t, vtot, linewidth=2.2, linestyle=dashdot_style, label="Total volume")
    ax.set_ylabel("Volume (L)", fontsize=10)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.text(
        0.01, 0.93, "C",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )

    # Shared styling
    for ax in axes:
        ax.set_xlim(0.0, cycle_time)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=9)

    # Separate legends for readability
    legend_kwargs = dict(
        frameon=False,
        fontsize=8,
        loc="upper right",
        handlelength=3.0,
    )
    axes[0].legend(**legend_kwargs)
    axes[1].legend(**legend_kwargs)
    axes[2].legend(**legend_kwargs)

    plt.subplots_adjust(
        top=0.97,
        bottom=0.08,
        left=0.12,
        right=0.98,
        hspace=0.22,
    )

    out_png = OUTPUT_DIR / "Figure2_representative_waveforms.png"
    out_svg = OUTPUT_DIR / "Figure2_representative_waveforms.svg"

    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)

    print("Saved:")
    print(out_png)
    print(out_svg)


if __name__ == "__main__":
    main()