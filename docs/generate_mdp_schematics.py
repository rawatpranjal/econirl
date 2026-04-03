#!/usr/bin/env python3
"""Generate schematic MDP structure diagrams for example documentation pages.

Each diagram shows the state space, action set, and transition structure
for one problem domain in a clean, textbook-style layout.

Usage:
    python docs/generate_mdp_schematics.py
    python docs/generate_mdp_schematics.py --only rust_bus
    python docs/generate_mdp_schematics.py --only frozen_lake,keane_wolpin

Outputs:
    docs/_static/mdp_schematic_<name>.png  (17 files)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = Path(__file__).resolve().parent / "_static"

# ── Palette ──────────────────────────────────────────────────────────

STATE = "#D4E6F1"      # light blue — normal states
START = "#D5F5E3"      # light green — initial / goal
TERM  = "#FADBD8"      # light red — terminal / absorbing
HILITE = "#FEF9E7"     # light yellow — highlighted / ego
GRID_F = "#EBF5FB"     # very light blue — safe grid cells
EDGE  = "#2C3E50"      # dark blue-grey — borders and text

C1 = "#2C3E50"         # primary action
C2 = "#E74C3C"         # secondary
C3 = "#27AE60"         # tertiary
C4 = "#F39C12"         # quaternary
C5 = "#8E44AD"         # quinary

FONT = "sans-serif"
DPI = 180


# ── Drawing Primitives ──────────────────────────────────────────────

def _node(ax, x, y, label, sub=None, color=STATE, w=1.2, h=0.55, fs=9):
    """Rounded-rectangle state node with optional sublabel."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.08",
        fc=color, ec=EDGE, lw=1.5, zorder=3)
    ax.add_patch(box)
    ty = y + (0.07 if sub else 0)
    ax.text(x, ty, label, ha="center", va="center", fontsize=fs,
            fontweight="bold", family=FONT, zorder=4)
    if sub:
        ax.text(x, y - 0.15, sub, ha="center", va="center",
                fontsize=6.5, color="#555", family=FONT, zorder=4)


def _arrow(ax, x1, y1, x2, y2, label="", color=EDGE, rad=0.0,
           lw=1.5, fs=8, lx=None, ly=None, sA=25, sB=25, ls="-"):
    """Labelled arrow between two points with optional curvature."""
    cs = f"arc3,rad={rad}"
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", color=color,
        mutation_scale=14, lw=lw, connectionstyle=cs,
        shrinkA=sA, shrinkB=sB, linestyle=ls, zorder=2)
    ax.add_patch(arr)
    if label and lx is not None:
        ax.text(lx, ly, label, ha="center", va="center", fontsize=fs,
                fontstyle="italic", color=color, family=FONT, zorder=5)


def _selfloop(ax, x, y, label, color=EDGE, above=True, dx=0.0, fs=8):
    """Self-loop arrow above or below a node."""
    dy = 0.28 if above else -0.28
    r = -0.8 if above else 0.8
    arr = FancyArrowPatch(
        (x + dx - 0.18, y + dy), (x + dx + 0.18, y + dy),
        arrowstyle="-|>", color=color, mutation_scale=11, lw=1.3,
        connectionstyle=f"arc3,rad={r}", shrinkA=0, shrinkB=0, zorder=2)
    ax.add_patch(arr)
    ldy = 0.62 if above else -0.62
    ax.text(x + dx, y + ldy, label, ha="center", va="center",
            fontsize=fs, fontstyle="italic", color=color,
            family=FONT, zorder=5)


def _dots(ax, x, y, fs=16):
    """Ellipsis placeholder between states."""
    ax.text(x, y, "\u00b7\u00b7\u00b7", ha="center", va="center",
            fontsize=fs, color="#888", family=FONT, zorder=4)


def _note(ax, x, y, text, fs=8, color="#555"):
    """Annotation text."""
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fs, color=color, family=FONT, zorder=5)


def _rect(ax, x, y, w, h, color=STATE, lw=1.0, label="", fs=7):
    """Simple rectangle for grid cells."""
    rect = mpatches.Rectangle(
        (x - w / 2, y - h / 2), w, h,
        fc=color, ec=EDGE, lw=lw, zorder=3)
    ax.add_patch(rect)
    if label:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", family=FONT, zorder=4)


def _draw_grid(ax, rows, cols, x0, y0, cs, colors, labels):
    """Draw a grid of square cells. colors and labels are (row, col) dicts."""
    for r in range(rows):
        for c in range(cols):
            cx = x0 + c * cs
            cy = y0 + (rows - 1 - r) * cs
            col = colors.get((r, c), STATE)
            lab = labels.get((r, c), "")
            _rect(ax, cx, cy, cs * 0.92, cs * 0.92, color=col, label=lab)


def _fig(w=8, h=2.8):
    """Create figure with axes off and equal aspect."""
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _save(fig, name):
    """Save figure to _static directory."""
    p = OUT_DIR / f"mdp_schematic_{name}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white",
                edgecolor="none", pad_inches=0.15)
    plt.close(fig)
    print(f"  \u2713 {p.name}")


# ══════════════════════════════════════════════════════════════════════
# Group A: Optimal Stopping / Degradation
# ══════════════════════════════════════════════════════════════════════

def _optimal_stopping_chain(name, node_labels, node_subs, note_text,
                            keep_label="Keep", replace_label="Replace"):
    """Shared layout for linear-chain keep/replace problems."""
    fig, ax = _fig(8, 2.8)
    xs = [1.0, 3.2, 5.0, 7.0]

    _node(ax, xs[0], 0, node_labels[0], node_subs[0], color=START)
    _node(ax, xs[1], 0, node_labels[1], node_subs[1])
    _dots(ax, xs[2], 0)
    _node(ax, xs[3], 0, node_labels[2], node_subs[2])

    # Keep arrows along chain
    _arrow(ax, xs[0], 0, xs[1], 0, keep_label, C1, lx=2.1, ly=0.38)
    _arrow(ax, xs[1], 0, xs[2], 0, "", C1, sB=8)
    _arrow(ax, xs[2], 0, xs[3], 0, "", C1, sA=8)

    # Replace arrow (curved above, dashed)
    _arrow(ax, xs[3], 0, xs[0], 0, replace_label, C2,
           rad=-0.22, ls="--", lx=4.0, ly=1.45)

    _note(ax, 4.0, -0.8, note_text)

    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-1.2, 2.0)
    _save(fig, name)


def generate_rust_bus():
    _optimal_stopping_chain(
        "rust_bus",
        node_labels=["$s_0$", "$s_1$", "$s_{89}$"],
        node_subs=["0 mi", None, "max mi"],
        replace_label="Replace (cost $R_C$)",
        note_text="90 mileage bins  \u00b7  2 actions: Keep or Replace",
    )


def generate_scania_component():
    _optimal_stopping_chain(
        "scania_component",
        node_labels=["$d_0$", "$d_1$", "$d_{49}$"],
        node_subs=["healthy", None, "degraded"],
        keep_label="Operate",
        replace_label="Replace",
        note_text=(
            "50 sensor-derived degradation states  \u00b7  "
            "2 actions: Operate or Replace"
        ),
    )


def generate_rdw_scrappage():
    """RDW scrappage: age chain with defect dimension and scrap exit."""
    fig, ax = _fig(8, 3.2)

    xs = [1.0, 3.2, 5.0, 7.0]
    y_top = 0.6

    _node(ax, xs[0], y_top, "age 3", "low defects", color=START, w=1.3)
    _node(ax, xs[1], y_top, "age 4", None, w=1.3)
    _dots(ax, xs[2], y_top)
    _node(ax, xs[3], y_top, "age 20", None, w=1.3)

    # Keep arrows along age chain
    _arrow(ax, xs[0], y_top, xs[1], y_top, "Keep", C1,
           lx=2.1, ly=y_top + 0.38)
    _arrow(ax, xs[1], y_top, xs[2], y_top, "", C1, sB=8)
    _arrow(ax, xs[2], y_top, xs[3], y_top, "", C1, sA=8)

    # Scrap terminal
    y_bot = -0.7
    _node(ax, 4.0, y_bot, "Scrapped", color=TERM, w=1.4)

    _arrow(ax, xs[0], y_top, 4.0, y_bot, "Scrap", C2,
           rad=0.15, lx=1.8, ly=-0.15, ls="--")
    _arrow(ax, xs[3], y_top, 4.0, y_bot, "", C2, rad=-0.15, ls="--")

    # Defect note
    _note(ax, 7.0, -0.1,
          "Defect severity evolves\nstochastically at each age",
          fs=7, color="#888")

    _note(ax, 4.0, -1.35,
          "17 age bins \u00d7 defect severity  \u00b7  2 actions: Keep or Scrap")

    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-1.7, 1.6)
    _save(fig, "rdw_scrappage")


# ══════════════════════════════════════════════════════════════════════
# Group B: Grid Navigation
# ══════════════════════════════════════════════════════════════════════

def generate_frozen_lake():
    """FrozenLake 4x4 grid with NSEW actions and slippery note."""
    fig, ax = _fig(8, 3.5)

    cs = 0.7
    x0, y0 = 0.5, 0.5

    # Cell colours and labels
    fl_colors = {
        (0, 0): START, (3, 3): "#82E0AA",
        (1, 1): TERM, (1, 3): TERM, (3, 0): TERM,
    }
    fl_labels = {
        (0, 0): "S", (3, 3): "G",
        (1, 1): "H", (1, 3): "H", (3, 0): "H",
    }
    for r in range(4):
        for c in range(4):
            if (r, c) not in fl_colors:
                fl_colors[(r, c)] = GRID_F
                fl_labels[(r, c)] = "F"
    _draw_grid(ax, 4, 4, x0, y0, cs, fl_colors, fl_labels)

    # Highlight agent cell at (2, 1)
    acx = x0 + 1 * cs
    acy = y0 + (3 - 2) * cs
    _rect(ax, acx, acy, cs * 0.92, cs * 0.92, color=HILITE, lw=2.0,
          label="F")

    # NSEW arrows from agent cell
    d = 0.55
    _arrow(ax, acx, acy + 0.2, acx, acy + d + 0.15, "", C1, sA=3, sB=3)
    _arrow(ax, acx, acy - 0.2, acx, acy - d - 0.15, "", C1, sA=3, sB=3)
    _arrow(ax, acx + 0.2, acy, acx + d + 0.15, acy, "", C1, sA=3, sB=3)
    _arrow(ax, acx - 0.2, acy, acx - d - 0.15, acy, "", C1, sA=3, sB=3)

    # Right side explanation
    rx = 4.5
    _note(ax, rx, 2.6, "Agent chooses N, S, E, or W", fs=9, color=EDGE)
    _note(ax, rx, 2.1, "Slippery surface:", fs=8)
    _note(ax, rx, 1.7, "P(intended) = \u2153", fs=8)
    _note(ax, rx, 1.3, "P(each perpendicular) = \u2153", fs=8)

    # Legend
    ly = 0.3
    for yoff, col, lab in [
        (0, START, "Start"), (-0.4, "#82E0AA", "Goal"),
        (-0.8, TERM, "Hole"), (-1.2, GRID_F, "Frozen"),
    ]:
        _rect(ax, 3.8, ly + yoff, 0.25, 0.25, color=col)
        _note(ax, 4.2, ly + yoff, lab, fs=7)

    _note(ax, 3.0, -1.0,
          "16 states  \u00b7  4 actions: N, S, E, W  \u00b7  "
          "Slippery transitions")

    ax.set_xlim(-0.3, 5.8)
    ax.set_ylim(-1.4, 3.2)
    _save(fig, "frozen_lake")


def generate_taxi_gridworld():
    """Taxi gridworld: 5x5 grid with NSEW + Stay."""
    fig, ax = _fig(8, 3.5)

    cs = 0.6
    x0, y0 = 0.5, 0.5

    colors, labels = {}, {}
    for r in range(5):
        for c in range(5):
            colors[(r, c)] = GRID_F
            labels[(r, c)] = ""
    colors[(0, 0)] = START
    labels[(0, 0)] = "S"
    colors[(4, 4)] = "#82E0AA"
    labels[(4, 4)] = "G"
    _draw_grid(ax, 5, 5, x0, y0, cs, colors, labels)

    # Highlight agent at centre (2, 2)
    cx = x0 + 2 * cs
    cy = y0 + 2 * cs
    _rect(ax, cx, cy, cs * 0.92, cs * 0.92, color=HILITE, lw=2.0)

    # NSEW arrows
    d = 0.48
    for ddx, ddy in [(0, d), (0, -d), (d, 0), (-d, 0)]:
        _arrow(ax, cx + ddx * 0.4, cy + ddy * 0.4,
               cx + ddx * 1.3, cy + ddy * 1.3, "", C1, sA=2, sB=2)

    # Stay self-loop
    _selfloop(ax, cx, cy, "Stay", C4, above=True, fs=7)

    # Right side
    rx = 4.8
    _note(ax, rx, 2.8, "5 actions per cell:", fs=9, color=EDGE)
    _note(ax, rx, 2.35, "N, S, E, W, Stay", fs=8)
    _note(ax, rx, 1.7, "Deterministic transitions", fs=8)
    _note(ax, rx, 1.25, "Goal at corner (4, 4)", fs=8)

    _note(ax, 3.2, -0.5,
          "25 grid cells  \u00b7  5 actions  \u00b7  "
          "Deterministic transitions")

    ax.set_xlim(-0.3, 6.2)
    ax.set_ylim(-0.9, 3.8)
    _save(fig, "taxi_gridworld")


def generate_beijing_taxi():
    """Beijing taxi: 15x15 grid with 8 compass actions."""
    fig, ax = _fig(8, 3.2)

    cs = 0.7
    x0, y0 = 0.5, 0.5
    colors, labels = {}, {}
    for r in range(3):
        for c in range(3):
            colors[(r, c)] = GRID_F
            labels[(r, c)] = ""
    colors[(1, 1)] = HILITE
    _draw_grid(ax, 3, 3, x0, y0, cs, colors, labels)

    # 8 compass arrows from centre
    cx = x0 + 1 * cs
    cy = y0 + 1 * cs
    compass = [
        (0, 1), (0, -1), (1, 0), (-1, 0),
        (0.71, 0.71), (-0.71, 0.71), (0.71, -0.71), (-0.71, -0.71),
    ]
    for ddx, ddy in compass:
        _arrow(ax, cx + ddx * 0.25, cy + ddy * 0.25,
               cx + ddx * 0.85, cy + ddy * 0.85,
               "", C1, sA=2, sB=2, lw=1.2)

    # Dots showing grid extends
    _dots(ax, x0 + 3.2 * cs, y0 + 1 * cs, fs=12)
    _dots(ax, x0 + 1 * cs, y0 + 3.2 * cs, fs=12)
    _dots(ax, x0 + 1 * cs, y0 - 1.2 * cs, fs=12)

    # Right side
    rx = 4.5
    _note(ax, rx, 2.2, "8 compass actions", fs=9, color=EDGE)
    _note(ax, rx, 1.75, "N, S, E, W, NE, NW, SE, SW", fs=8)
    _note(ax, rx, 1.2, "Discretised from GPS trajectories", fs=8)

    _note(ax, 3.2, -0.6,
          "225 grid cells (15\u00d715)  \u00b7  8 compass actions")

    ax.set_xlim(-0.5, 6.0)
    ax.set_ylim(-1.0, 3.0)
    _save(fig, "beijing_taxi")


def generate_wulfmeier_deep_maxent():
    """Wulfmeier deep MaxEnt: Objectworld grid with nonlinear rewards."""
    fig, ax = _fig(8, 3.2)

    cs = 0.65
    x0, y0 = 0.5, 0.5
    colors, labels = {}, {}
    for r in range(4):
        for c in range(4):
            colors[(r, c)] = GRID_F
            labels[(r, c)] = ""
    colors[(1, 1)] = HILITE
    _draw_grid(ax, 4, 4, x0, y0, cs, colors, labels)

    # Coloured objects in some cells
    obj_cells = [(0, 2), (2, 0), (3, 3), (1, 3)]
    obj_colors = ["#E74C3C", "#3498DB", "#F39C12", "#27AE60"]
    for (r, c), oc in zip(obj_cells, obj_colors):
        ocx = x0 + c * cs
        ocy = y0 + (3 - r) * cs
        circle = mpatches.Circle((ocx, ocy), 0.12, fc=oc, ec=EDGE,
                                 lw=1.0, zorder=5)
        ax.add_patch(circle)

    # NSEW + Stay from highlighted cell (1, 1)
    cx = x0 + 1 * cs
    cy = y0 + 2 * cs
    d = 0.5
    for ddx, ddy in [(0, d), (0, -d), (d, 0), (-d, 0)]:
        _arrow(ax, cx + ddx * 0.3, cy + ddy * 0.3,
               cx + ddx * 1.2, cy + ddy * 1.2, "", C1, sA=2, sB=2)
    _selfloop(ax, cx, cy, "Stay", C4, above=True, fs=7)

    # Right side
    rx = 4.8
    _note(ax, rx, 2.8, "Objectworld / Binaryworld", fs=9, color=EDGE)
    _note(ax, rx, 2.3, "Coloured objects create", fs=8)
    _note(ax, rx, 1.9, "nonlinear reward landscape", fs=8)
    _note(ax, rx, 1.3, "5 actions: N, S, E, W, Stay", fs=8)

    _note(ax, 3.5, -0.6,
          "Grid cells  \u00b7  5 actions  \u00b7  "
          "Nonlinear reward from object distances")

    ax.set_xlim(-0.3, 6.5)
    ax.set_ylim(-1.0, 3.6)
    _save(fig, "wulfmeier_deep_maxent")


# ══════════════════════════════════════════════════════════════════════
# Group C: Binary / Simple Choice with Multi-Dimensional State
# ══════════════════════════════════════════════════════════════════════

def generate_entry_exit():
    """Dixit entry/exit: two status rows with profit transitions."""
    fig, ax = _fig(8, 3.4)

    y_act = 1.0
    y_inact = -0.6
    xs = [1.5, 3.5, 5.5, 7.0]
    pi_labels = ["$\\pi_1$", "$\\pi_2$", "$\\pi_3$", "$\\pi_4$"]

    # Active row
    for i, x in enumerate(xs):
        _node(ax, x, y_act, pi_labels[i], "active", color=STATE, w=1.1)

    # Inactive row
    for i, x in enumerate(xs):
        _node(ax, x, y_inact, pi_labels[i], "inactive", color=GRID_F, w=1.1)

    # Profit transition arrows within active row
    for i in range(len(xs) - 1):
        _arrow(ax, xs[i], y_act, xs[i + 1], y_act, "", C1,
               lw=1.0, sA=22, sB=22)

    # Entry arrow (inactive -> active)
    _arrow(ax, xs[1], y_inact, xs[1], y_act, "Enter", C3,
           lx=xs[1] + 0.6, ly=0.2, sA=20, sB=20)
    _note(ax, xs[1] + 0.6, -0.1, "(sunk cost)", fs=6.5, color=C3)

    # Exit arrow (active -> inactive)
    _arrow(ax, xs[2], y_act, xs[2], y_inact, "Exit", C2,
           lx=xs[2] + 0.55, ly=0.2, sA=20, sB=20)
    _note(ax, xs[2] + 0.55, 0.5, "(sunk cost)", fs=6.5, color=C2)

    # Row labels
    _note(ax, 0.2, y_act, "Active", fs=8, color=EDGE)
    _note(ax, 0.2, y_inact, "Inactive", fs=8, color=EDGE)

    _note(ax, 4.0, -1.5,
          "10 profit bins \u00d7 2 status  \u00b7  "
          "2 actions: Enter or Exit  \u00b7  "
          "Sunk costs create hysteresis")

    ax.set_xlim(-0.7, 8.2)
    ax.set_ylim(-1.9, 1.8)
    _save(fig, "entry_exit")


def generate_instacart():
    """Instacart: frequency x recency with reorder/skip."""
    fig, ax = _fig(8, 3.2)

    cs = 0.85
    gx, gy = 1.0, 0.5

    # Axis labels
    _note(ax, gx - 0.9, gy + 1 * cs, "Freq \u2191", fs=8, color=EDGE)
    _note(ax, gx + 1 * cs, gy + 2.6 * cs, "Recency \u2192", fs=8, color=EDGE)

    grid_labels = {
        (0, 0): "hi,lo", (0, 1): "hi,md", (0, 2): "hi,hi",
        (1, 0): "md,lo", (1, 1): "md,md", (1, 2): "md,hi",
        (2, 0): "lo,lo", (2, 1): "lo,md", (2, 2): "lo,hi",
    }
    colors = {}
    for r in range(3):
        for c in range(3):
            colors[(r, c)] = GRID_F
    colors[(1, 1)] = HILITE
    _draw_grid(ax, 3, 3, gx, gy, cs, colors, grid_labels)

    # Reorder: increase freq, reset recency
    cx, cy = gx + 1 * cs, gy + 1 * cs
    tx, ty = gx + 0 * cs, gy + 2 * cs
    _arrow(ax, cx, cy, tx, ty, "Reorder", C3,
           rad=-0.15, lx=(cx + tx) / 2 - 0.55, ly=(cy + ty) / 2 + 0.3)

    # Skip: increase recency
    sx, sy = gx + 2 * cs, gy + 1 * cs
    _arrow(ax, cx, cy, sx, sy, "Skip", C2,
           lx=(cx + sx) / 2, ly=cy - 0.4)

    # Right side
    rx = 5.5
    _note(ax, rx, 2.3, "State = (freq, recency)", fs=9, color=EDGE)
    _note(ax, rx, 1.8, "Reorder: increase frequency,", fs=8)
    _note(ax, rx, 1.4, "   reset recency", fs=8)
    _note(ax, rx, 0.8, "Skip: increase recency,", fs=8)
    _note(ax, rx, 0.4, "   frequency unchanged", fs=8)

    _note(ax, 3.5, -0.6,
          "Frequency \u00d7 Recency states  \u00b7  "
          "2 actions: Reorder or Skip")

    ax.set_xlim(-1.2, 7.2)
    ax.set_ylim(-1.0, 3.2)
    _save(fig, "instacart")


def generate_citibike_usage():
    """Citi Bike daily usage: habit stock chain with ride/skip."""
    fig, ax = _fig(8, 2.8)

    xs = [1.0, 3.0, 5.0, 7.0]
    y = 0

    _node(ax, xs[0], y, "$h_0$", "no habit", color=GRID_F)
    _node(ax, xs[1], y, "$h_1$", None)
    _dots(ax, xs[2], y)
    _node(ax, xs[3], y, "$h_K$", "strong", color=START)

    # Ride arrows (rightward)
    _arrow(ax, xs[0], y, xs[1], y, "Ride", C3, lx=2.0, ly=0.38)
    _arrow(ax, xs[1], y, xs[2], y, "", C3, sB=8)
    _arrow(ax, xs[2], y, xs[3], y, "", C3, sA=8)

    # Skip arrows (leftward, curved below)
    _arrow(ax, xs[3], y, xs[2], y, "", C2, rad=0.25, sA=8)
    _arrow(ax, xs[2], y, xs[1], y, "", C2, rad=0.25, sB=8, sA=8)
    _arrow(ax, xs[1], y, xs[0], y, "Skip", C2,
           rad=0.25, lx=2.0, ly=-0.45)

    _note(ax, 4.0, -0.85,
          "Habit stock \u00d7 day type  \u00b7  2 actions: Ride or Skip  \u00b7  "
          "Habit decays on skip, grows on ride")

    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-1.2, 1.0)
    _save(fig, "citibike_usage")


# ══════════════════════════════════════════════════════════════════════
# Group D: Network / Route Choice
# ══════════════════════════════════════════════════════════════════════

def generate_ngsim_lane_change():
    """NGSIM: three highway lanes with lane-change actions."""
    fig, ax = _fig(8, 3.0)

    lane_ys = [0.0, 1.2, 2.4]
    lane_labels = ["Lane 1", "Lane 2", "Lane 3"]

    # Lane rects start at x=1.5 so lane labels at x=0.6 are clearly outside
    lane_cx = 5.0   # centre of lane rect
    lane_w  = 6.5   # width — left edge at 5.0 - 3.25 = 1.75
    for ly, ll in zip(lane_ys, lane_labels):
        _rect(ax, lane_cx, ly, lane_w, 0.9, color=GRID_F, lw=0.8)
        _note(ax, 0.6, ly, ll, fs=7, color="#888")

    # Ego vehicle — shifted right relative to lane centre
    ego_x, ego_y = 3.8, lane_ys[1]
    _rect(ax, ego_x, ego_y, 0.9, 0.5, color=HILITE, lw=2.0, label="Ego")

    # Other vehicles
    for ox, oy in [(6.5, lane_ys[0]), (6.2, lane_ys[2]), (2.5, lane_ys[0])]:
        _rect(ax, ox, oy, 0.7, 0.4, color="#D5DBDB", lw=0.8)

    # Stay — label clearly above arrow
    _arrow(ax, ego_x + 0.5, ego_y, ego_x + 2.0, ego_y,
           "Stay", C1, lx=ego_x + 1.25, ly=ego_y + 0.42)

    # Left lane change — label well below
    _arrow(ax, ego_x, ego_y - 0.3, ego_x + 1.0, lane_ys[0] + 0.3,
           "Left", C3, lx=ego_x + 1.1, ly=ego_y - 0.88)

    # Right lane change — label well above
    _arrow(ax, ego_x, ego_y + 0.3, ego_x + 1.0, lane_ys[2] - 0.3,
           "Right", C2, lx=ego_x + 1.1, ly=ego_y + 1.0)

    # Traffic flow annotation — far right, out of the way
    _arrow(ax, 7.8, -0.55, 8.5, -0.55, "", "#888", sA=3, sB=3, lw=1.0)
    _note(ax, 8.15, -0.28, "flow", fs=6.5, color="#888")

    _note(ax, 4.5, -1.0,
          "Lanes \u00d7 speed bins  \u00b7  3 actions: Left, Stay, Right")

    ax.set_xlim(-0.5, 8.8)
    ax.set_ylim(-1.4, 3.2)
    _save(fig, "ngsim_lane_change")


def generate_shanghai_route():
    """Shanghai route: intersection fan-out with road segment choices."""
    fig, ax = _fig(8, 3.0)

    cx, cy = 2.0, 0.8
    _node(ax, cx, cy, "Intersection\n$v_t$", color=HILITE, w=1.6, h=0.7)

    targets = [
        (5.5, 2.0, "$v_{t+1}^a$", "road A"),
        (5.5, 0.8, "$v_{t+1}^b$", "road B"),
        (5.5, -0.4, "$v_{t+1}^c$", "road C"),
    ]
    acolors = [C1, C3, C4]
    for (tx, ty, tl, ts), ac in zip(targets, acolors):
        _node(ax, tx, ty, tl, ts, w=1.4)
        _arrow(ax, cx, cy, tx, ty, "", ac, sA=30, sB=30)

    # Continue dots
    _dots(ax, 7.3, 2.0)
    _arrow(ax, 5.5, 2.0, 7.3, 2.0, "", C1, sB=8)

    _note(ax, 3.8, 1.75, "choose road A", fs=7, color=C1)
    _note(ax, 3.8, 1.0, "choose road B", fs=7, color=C3)
    _note(ax, 3.8, 0.0, "choose road C", fs=7, color=C4)

    _note(ax, 4.0, -1.2,
          "Road network intersections  \u00b7  "
          "Action = next road segment  \u00b7  "
          "Stochastic travel time")

    ax.set_xlim(-0.2, 8.2)
    ax.set_ylim(-1.6, 2.8)
    _save(fig, "shanghai_route")


def generate_citibike_route():
    """Citi Bike route: hub-and-spoke destination choice."""
    fig, ax = _fig(8, 3.2)

    ox, oy = 2.0, 0.8
    _node(ax, ox, oy, "Origin\ncluster", color=HILITE, w=1.4, h=0.7)

    dests = [
        (5.5, 2.2, "Midtown", "#AED6F1"),
        (6.0, 0.8, "Downtown", "#A9DFBF"),
        (5.5, -0.6, "Uptown", "#F9E79F"),
        (3.8, 2.2, "Brooklyn", "#F5CBA7"),
    ]
    dcolors = [C1, C3, C4, C5]
    for (dx, dy, dl, dbg), dc in zip(dests, dcolors):
        _node(ax, dx, dy, dl, color=dbg, w=1.3, h=0.5, fs=8)
        _arrow(ax, ox, oy, dx, dy, "", dc, sA=28, sB=25)

    _note(ax, 4.0, -1.3,
          "Station clusters  \u00b7  "
          "Action = destination choice  \u00b7  "
          "Time-of-day and membership effects")

    ax.set_xlim(-0.2, 7.5)
    ax.set_ylim(-1.7, 3.0)
    _save(fig, "citibike_route")


# ══════════════════════════════════════════════════════════════════════
# Group E: Multi-Action Complex State
# ══════════════════════════════════════════════════════════════════════

def generate_keane_wolpin():
    """Keane-Wolpin: state with 4 career-action branches."""
    fig, ax = _fig(8, 3.6)

    sx, sy = 1.2, 0.0
    _node(ax, sx, sy, "$(s, e_w, e_b)$", "period $t$", color=HILITE,
          w=1.7, h=0.65)

    outcomes = [
        (6.5,  1.5, "$(s+1, e_w, e_b)$",  "School",       C1),
        (6.5,  0.5, "$(s, e_w+1, e_b)$",   "White-collar", C3),
        (6.5, -0.5, "$(s, e_w, e_b+1)$",   "Blue-collar",  C4),
        (6.5, -1.5, "$(s, e_w, e_b)$",      "Home",         C2),
    ]
    for ox, oy, ol, act, ac in outcomes:
        _node(ax, ox, oy, ol, None, w=2.0, h=0.5, fs=8)
        _arrow(ax, sx, sy, ox, oy, act, ac,
               lx=(sx + ox) / 2, ly=oy + 0.28, sA=30, sB=35)

    # Period timeline
    _note(ax, 6.5, -2.2, "period $t + 1$", fs=8, color="#888")

    _note(ax, 4.0, -2.7,
          "704 states (schooling \u00d7 experience)  \u00b7  "
          "4 actions  \u00b7  10 periods (ages 17\u201326)")

    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-3.1, 2.2)
    _save(fig, "keane_wolpin")


def generate_icu_sepsis():
    """ICU sepsis: clinical state with treatment grid and absorbing states."""
    fig, ax = _fig(8, 3.2)

    # Patient state (left)
    _node(ax, 1.2, 0.5, "Patient $s_t$", "716 states", color=HILITE,
          w=1.5, h=0.65)

    # Action label
    ax.text(4.0, 1.85, "Action: (IV dose, Vasopressor dose)",
            ha="center", va="center", fontsize=8, color=EDGE,
            family=FONT, zorder=5)

    # Mini 5x5 dosing grid
    gx0, gy0 = 3.1, 0.0
    gs = 0.3
    for r in range(5):
        for c in range(5):
            gcx = gx0 + c * gs
            gcy = gy0 + (4 - r) * gs
            intensity = (r + c) / 8.0
            col = plt.cm.YlOrRd(0.1 + intensity * 0.5)
            _rect(ax, gcx, gcy, gs * 0.88, gs * 0.88, color=col, lw=0.5)

    _note(ax, gx0 + 2 * gs, gy0 - 0.35, "IV dose \u2192", fs=6)
    ax.text(gx0 - 0.35, gy0 + 2 * gs, "Vaso\ndose \u2192",
            ha="center", va="center", fontsize=6, color="#555",
            family=FONT, rotation=90, zorder=5)

    # Arrow from patient to action grid
    _arrow(ax, 1.2, 0.5, gx0, 0.5, "", C1, sA=28, sB=8)

    # Next state (right top)
    _node(ax, 7.0, 1.0, "$s_{t+1}$", "next state", w=1.3, h=0.55)
    _arrow(ax, gx0 + 4 * gs + 0.2, 0.8, 7.0, 1.0, "", C1, sA=8, sB=25)

    # Absorbing state (right bottom)
    _node(ax, 7.0, -0.3, "Discharge\nor Death", color=TERM,
          w=1.5, h=0.6, fs=8)
    _arrow(ax, gx0 + 4 * gs + 0.2, 0.3, 7.0, -0.3, "", C2, sA=8, sB=25)

    _note(ax, 4.0, -1.2,
          "716 clinical states  \u00b7  25 actions (5\u00d75 dosing grid)  \u00b7  "
          "4-hour decision windows")

    ax.set_xlim(-0.3, 8.3)
    ax.set_ylim(-1.6, 2.4)
    _save(fig, "icu_sepsis")


def generate_trivago_search():
    """Trivago hotel search: session state with browse/refine loops."""
    fig, ax = _fig(8, 3.2)

    sx, sy = 3.5, 1.0
    _node(ax, sx, sy, "Session\nState", color=HILITE, w=1.4, h=0.7)

    # Self-loops — spread widely to avoid label crowding
    _selfloop(ax, sx, sy, "Browse", C1, above=True, dx=-0.85, fs=8)
    _selfloop(ax, sx, sy, "Refine", C3, above=True, dx=0.85, fs=8)

    # Terminal: Booking (left)
    _node(ax, 1.5, -0.6, "Booking", color=START, w=1.2, h=0.5)
    _arrow(ax, sx, sy, 1.5, -0.6, "Clickout", C4,
           lx=2.0, ly=0.45, sA=28, sB=22)

    # Terminal: Exit (right)
    _node(ax, 5.5, -0.6, "Exit", color=TERM, w=1.2, h=0.5)
    _arrow(ax, sx, sy, 5.5, -0.6, "Abandon", C2,
           lx=5.0, ly=0.45, sA=28, sB=22)

    _note(ax, 3.5, -1.4,
          "Search session state  \u00b7  "
          "4 actions: Browse, Refine, Clickout, Abandon")

    ax.set_xlim(-0.2, 7.2)
    ax.set_ylim(-1.8, 2.5)
    _save(fig, "trivago_search")


def generate_supermarket():
    """Supermarket: inventory x price state with pricing and ordering."""
    fig, ax = _fig(8, 3.2)

    cs = 0.85
    gx, gy = 1.2, 0.3

    _note(ax, gx - 0.9, gy + 1 * cs, "Inv. \u2191", fs=8, color=EDGE)
    _note(ax, gx + 1 * cs, gy + 2.7 * cs, "Price level \u2192", fs=8,
          color=EDGE)

    colors, labels = {}, {}
    for r in range(3):
        for c in range(3):
            colors[(r, c)] = GRID_F
            labels[(r, c)] = ""
    labels[(0, 0)] = "hi,lo"
    labels[(0, 2)] = "hi,hi"
    labels[(2, 0)] = "lo,lo"
    labels[(2, 2)] = "lo,hi"
    labels[(1, 1)] = "md,md"
    colors[(1, 1)] = HILITE
    _draw_grid(ax, 3, 3, gx, gy, cs, colors, labels)

    cx, cy = gx + 1 * cs, gy + 1 * cs

    # Set price (horizontal)
    _arrow(ax, cx, cy, gx + 2 * cs, cy, "Set price \u2192", C4,
           lx=gx + 1.5 * cs, ly=cy + 0.38)

    # Order (vertical)
    _arrow(ax, cx, cy, cx, gy + 2 * cs, "Order \u2191", C3,
           lx=cx - 0.6, ly=gy + 1.5 * cs)

    # Demand (downward, dashed)
    _arrow(ax, cx + 0.3, cy, cx + 0.3, gy + 0 * cs,
           "Demand \u2193", "#999", ls="--",
           lx=cx + 0.9, ly=gy + 0.5 * cs)

    # Right side
    rx = 5.8
    _note(ax, rx, 2.3, "State = (inventory, price)", fs=9, color=EDGE)
    _note(ax, rx, 1.8, "Actions: set price,", fs=8)
    _note(ax, rx, 1.4, "   place replenishment order", fs=8)
    _note(ax, rx, 0.8, "Demand is stochastic", fs=8)
    _note(ax, rx, 0.4, "   and price-sensitive", fs=8)

    _note(ax, 3.8, -0.7,
          "Inventory \u00d7 price states  \u00b7  "
          "Price and order actions  \u00b7  Stochastic demand")

    ax.set_xlim(-1.0, 7.8)
    ax.set_ylim(-1.1, 3.2)
    _save(fig, "supermarket")


# ══════════════════════════════════════════════════════════════════════
# Registry and CLI
# ══════════════════════════════════════════════════════════════════════

GENERATORS = {
    "rust_bus":              generate_rust_bus,
    "scania_component":      generate_scania_component,
    "rdw_scrappage":         generate_rdw_scrappage,
    "frozen_lake":           generate_frozen_lake,
    "taxi_gridworld":        generate_taxi_gridworld,
    "beijing_taxi":          generate_beijing_taxi,
    "wulfmeier_deep_maxent": generate_wulfmeier_deep_maxent,
    "entry_exit":            generate_entry_exit,
    "instacart":             generate_instacart,
    "citibike_usage":        generate_citibike_usage,
    "ngsim_lane_change":     generate_ngsim_lane_change,
    "shanghai_route":        generate_shanghai_route,
    "citibike_route":        generate_citibike_route,
    "keane_wolpin":          generate_keane_wolpin,
    "icu_sepsis":            generate_icu_sepsis,
    "trivago_search":        generate_trivago_search,
    "supermarket":           generate_supermarket,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", default="",
        help="Comma-separated list of diagrams to generate (default: all)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.only:
        names = [n.strip() for n in args.only.split(",")]
    else:
        names = list(GENERATORS.keys())

    print(f"Generating {len(names)} MDP schematics...")
    for name in names:
        if name not in GENERATORS:
            print(f"  \u2717 Unknown diagram: {name}")
            continue
        GENERATORS[name]()
    print("Done.")
