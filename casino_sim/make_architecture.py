#!/usr/bin/env python3
"""Render the system architecture diagram → casino_sim/figures/architecture.png
How the code works, at a glance. Run: python3 casino_sim/make_architecture.py"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

fig, ax = plt.subplots(figsize=(11, 8.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

C = {"in": "#e2e8f0", "model": "#bee3f8", "prob": "#9ae6b4", "branch": "#fbd38d",
     "out": "#d6bcfa", "line": "#fed7d7"}


def box(x, y, w, h, text, color, bold=False, fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2",
                                fc=color, ec="#2d3748", lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)


def arrow(x1, y1, x2, y2, style="-", color="#2d3748", lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=16, color=color, lw=lw, linestyle=style,
                 shrinkA=2, shrinkB=2))


ax.text(50, 97, "Line-free sports model  →  being the casino", ha="center",
        fontsize=15, fontweight="bold")

# Row 1: inputs
box(4, 84, 27, 8, "intl_results.csv\n49k international matches (results only)", C["in"], fs=9)
box(36.5, 84, 27, 8, "ClubElo snapshot\n(pre-tournament, point-in-time)", C["in"], fs=9)
box(69, 84, 27, 8, "Announced squads\n(player → club rosters)", C["in"], fs=9)

# Row 2: models
box(10, 68, 36, 9, "Walk-forward Elo\n(only matches BEFORE kickoff)", C["model"], fs=9)
box(54, 68, 36, 9, "Squad-strength Poisson /\nDixon–Coles (goals-only fit)", C["model"], fs=9)
for sx in (22, 28, 40):
    arrow(sx, 84, 28, 77)
arrow(82, 84, 72, 77)
arrow(50, 84, 60, 77)

# Row 3: probabilities (the product)
box(22, 53, 56, 9, "LINE-FREE 3-way probabilities\nP(home) · P(draw) · P(away)", C["prob"], bold=True, fs=11)
arrow(28, 68, 40, 62)
arrow(72, 68, 60, 62)

# the line, kept to the side — comparison only
box(80, 40, 18, 16, "Sportsbook / Kalshi\nLINE\n(used ONLY after\nprediction, never\nan input)", C["line"], fs=8.5)

# Row 4: three branches
box(3, 30, 28, 12, "Leakage audit\n3 adversarial auditors:\nis the line an input? → NO", C["branch"], fs=8.8)
box(35, 30, 30, 12, "Compare vs real books\nPinnacle · DraftKings · Bet365\n(de-vig → consensus)", C["branch"], fs=8.8)
box(69, 22, 28, 12, "Casino Monte Carlo ($50M)\nvig vs line-shading,\n20k seasons", C["branch"], fs=8.8)
arrow(40, 53, 22, 42)
arrow(50, 53, 50, 42)
arrow(60, 53, 80, 34)
arrow(89, 40, 50, 42, style="--", color="#c53030")   # line → compare only
arrow(89, 40, 83, 34, style="--", color="#c53030")   # line → casino sim (as market ref)

# Row 5: outputs
box(3, 8, 28, 10, "Independence verdict\n(LEAKAGE_AUDIT.md)", C["out"], fs=9)
box(35, 8, 30, 10, "Spreadsheet + graphs\n(book_comparison.csv, figures/)", C["out"], fs=9)
box(69, 8, 28, 10, "House economics\n(HOUSE_RESULTS.md)", C["out"], fs=9)
arrow(17, 30, 17, 18)
arrow(50, 30, 50, 18)
arrow(83, 22, 83, 18)

fig.tight_layout()
out = os.path.join(FIG, "architecture.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
