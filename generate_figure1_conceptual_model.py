from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from figure_utils import setup_publication_style, save_figure


setup_publication_style()

fig, ax = plt.subplots(figsize=(10.8, 4.8))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


def box(x, y, w, h, text, fc="#f7f7f7", ec="black", lw=1.0, fontsize=9):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
    )


def arrow(x1, y1, x2, y2):
    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.0,
        color="black",
        shrinkA=6,
        shrinkB=6,
    )
    ax.add_patch(patch)


# Left-side inputs
box(
    0.06,
    0.62,
    0.20,
    0.20,
    "Ventilator\nPEEP + pressure support",
    fc="#f7f7f7",
    fontsize=9,
)

box(
    0.06,
    0.18,
    0.20,
    0.20,
    "Patient effort\nhalf-sine Pmus(t)",
    fc="#f7f7f7",
    fontsize=9,
)

# Central block
cx, cy, cw, ch = 0.39, 0.37, 0.24, 0.28
box(
    cx,
    cy,
    cw,
    ch,
    "Net distending pressure\nPdist(t) = Paw(t) - PEEP + Pmus(t)",
    fc="#eef5ff",
    fontsize=8.8,
)

# Right-side compartments
box(
    0.75,
    0.61,
    0.18,
    0.18,
    "Compartment 1\nC1 = 0.04 L/cmH2O\nR1 = 8 cmH2O·s/L",
    fc="#fafafa",
    fontsize=8.8,
)

box(
    0.75,
    0.21,
    0.18,
    0.18,
    "Compartment 2\nphenotype-specific\nC2, R2",
    fc="#fafafa",
    fontsize=8.8,
)

# Arrows into the left edge of the central box
arrow(0.26, 0.71, cx, cy + 0.18)
arrow(0.26, 0.28, cx, cy + 0.10)

# Arrows out from the right edge of the central box
arrow(cx + cw, cy + 0.18, 0.75, 0.70)
arrow(cx + cw, cy + 0.10, 0.75, 0.29)

save_figure(fig, "Figure1_conceptual_model")
plt.close(fig)