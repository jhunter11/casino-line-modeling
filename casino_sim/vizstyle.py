"""One look for every figure in this repo.

Charts here exist to make a negative result legible, so the styling gets out of
the way: no chart junk, recessive axes, one horizontal grid, and values written
next to the marks that carry the argument. Import it and call :func:`use` right
after importing pyplot.

The categorical hues are a validated colour-blind-safe set — blue / orange /
aqua, in that fixed order, never cycled.
"""

# Categorical slots, assigned in order. Two-hue charts use SERIES[0:2].
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

INK = "#0b0b0b"        # titles, values
INK_2 = "#52514e"      # axis labels, ticks
INK_3 = "#8a8985"      # annotations, reference lines
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"
GOOD = "#1baf7a"
BAD = "#eb6834"


def use():
    """Apply the house style to the current matplotlib session."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": ["DejaVu Sans"],
        "font.size": 10,
        "text.color": INK,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.labelsize": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "semibold",
        "axes.titlecolor": INK,
        "axes.titlepad": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
    })


def horizontal_grid_only(ax):
    """Vertical gridlines rarely help a categorical axis; drop them."""
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", visible=True)


def zero_line(ax, label=None, y=0.0):
    """The reference the reader is comparing against, drawn once and named."""
    ax.axhline(y, color=INK_3, lw=1.1, zorder=1)
    if label:
        ax.annotate(
            label,
            xy=(1.0, y),
            xycoords=("axes fraction", "data"),
            xytext=(-2, 4),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=8.5,
            color=INK_3,
        )


def label_bars(ax, bars, fmt="{:+.1f}%", fontsize=9):
    """Write the value at the end of each bar, outside the mark."""
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            fmt.format(h),
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4 if h >= 0 else -13),
            textcoords="offset points",
            ha="center",
            fontsize=fontsize,
            color=INK,
        )


def reliability(ax, bins, *, color=None, xlabel="Predicted probability (our model)"):
    """A reliability curve where the dot size carries the sample count.

    The point of these plots is *where* a model is wrong, and the honest answer
    depends heavily on how many observations sit in each bucket. Drawing a bin
    of 3 the same size as a bin of 400 invites the reader to believe a tail that
    is pure noise — so the marker area is proportional to n, and only the bins
    with enough data to argue from are labelled.
    """
    color = color or SERIES[0]
    preds = [b["pred"] for b in bins]
    obs = [b["obs"] for b in bins]
    ns = [b["n"] for b in bins]
    biggest = max(ns) if ns else 1

    ax.plot([0, 1], [0, 1], "--", color=INK_3, lw=1, zorder=1,
            label="perfectly calibrated")
    ax.plot(preds, obs, "-", color=color, lw=1.6, zorder=2, alpha=0.75)
    ax.scatter(preds, obs, s=[40 + 360 * (n / biggest) ** 0.5 for n in ns],
               color=color, zorder=3, linewidths=1.5, edgecolors=SURFACE,
               label="our model  (area ∝ n)")

    for b in bins:
        if b["n"] >= max(5, 0.08 * sum(ns)):
            # clear the marker, whose radius grows with n
            pad = 7 + (40 + 360 * (b["n"] / biggest) ** 0.5) ** 0.5 * 0.6
            ax.annotate(f"n={b['n']}", (b["pred"], b["obs"]),
                        textcoords="offset points", xytext=(0, -pad),
                        ha="center", va="top", fontsize=8, color=INK_2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Actual win frequency")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    horizontal_grid_only(ax)
    ax.grid(axis="x", visible=True)


def caption(fig, text, y=-0.02):
    """A single line under the figure carrying the takeaway."""
    fig.text(0.5, y, text, ha="center", va="top", fontsize=9, color=INK_2, wrap=True)
