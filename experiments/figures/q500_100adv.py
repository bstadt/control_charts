"""Standalone presentation figure: the Q=500, 100-adversary detect_grid column.

Panels run LEFT TO RIGHT (infection, then alarm ratio) instead of stacked,
so the figure fits a slide. Data is pre-aggregated in q500_100adv_data.json
(built from the 900 r30 Volume cells via plot_grids.agg) — edit and re-run
this script freely, no Modal access needed.

Usage: python q500_100adv.py [out.png]
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "q500_100adv_data.json")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "q500_100adv.png")

# ---- layout knobs -----------------------------------------------------------
NS = [1000, 10000, 100000]            # x axis (no an100 data at N=100)
DEGS = [9, 99, 999, 9999, 99999]      # y axis (degree 4 trimmed, as in the 3-col figure)
FIGSIZE = (11.0, 5.6)                 # wide: two panels side by side
TITLE = "100 Sleeper Agents Defecting Gradually"
CELL_FONT, LABEL_FONT, TITLE_FONT = 10, 10, 12
# -----------------------------------------------------------------------------

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
cmap_seq = LinearSegmentedColormap.from_list("seqblue", SEQ)
cmap_div = LinearSegmentedColormap.from_list("ratio", [
    (0.00, "#6b0f0f"), (0.50, "#d03b3b"), (0.56, "#e88a83"),
    (0.63, "#f0efec"), (0.75, "#86b6ef"), (1.00, "#104281"),
])
LOGLIM = 1.5
nfmt = {100: "100", 1000: "1k", 10000: "10k", 100000: "100k"}
dfmt = {4: "4", 9: "9", 99: "99", 999: "999", 9999: "9,999", 99999: "99,999"}


def rfmt(r):
    if r >= 10: return f"{r:.0f}×"
    if r >= 1:  return f"{r:.1f}×"
    return f"{r:.2f}×"


data = json.load(open(DATA))
cells = {(c["N"], c["degree"]): c for c in data["cells"]}
seeds = data["seeds"]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=FIGSIZE, facecolor=SURFACE)
fig.subplots_adjust(left=0.09, right=0.98, top=0.83, bottom=0.21, wspace=0.28)


def rect(a, x, y, color, outline=False):
    a.add_patch(plt.Rectangle((x + 0.04, y + 0.04), 0.92, 0.92,
                              facecolor=color,
                              edgecolor=INK if outline else "none",
                              linewidth=2.8 if outline else 0))


# panel 1 (left): infected fraction of victims
for xi, N in enumerate(NS):
    for yi, dg in enumerate(DEGS):
        e = cells.get((N, dg))
        if e is None or e["inf"] is None:
            continue
        rect(ax, xi, yi, cmap_seq(e["inf"]))
        ax.text(xi + .5, yi + .5, f"{e['inf']:.2f}", ha="center", va="center",
                fontsize=CELL_FONT, color="#ffffff" if e["inf"] > .55 else INK)
ax.set_title("infected fraction of victims", fontsize=TITLE_FONT - 1, color=INK, pad=8)

# panel 2 (right): attack alarm rate / baseline alarm rate
for xi, N in enumerate(NS):
    for yi, dg in enumerate(DEGS):
        e = cells.get((N, dg))
        if e is None or e["ratio"] is None:
            continue
        t = (np.clip(np.log10(e["ratio"]), -LOGLIM, LOGLIM) + LOGLIM) / (2 * LOGLIM)
        # threshold 1.05, not 1.0: any cell that DISPLAYS as "1.0x" should box —
        # (100k, 9999) is 1.024 only via +1 smoothing on a dead-even 41v40 split
        taken = e["inf"] is not None and e["inf"] > 0.5
        blind = taken and e["ratio"] <= 1.05
        rect(ax2, xi, yi, cmap_div(t), outline=blind)
        dark = t <= .53 or t > .82
        ax2.text(xi + .5, yi + .5, rfmt(e["ratio"]), ha="center", va="center",
                 fontsize=CELL_FONT, color="#ffffff" if dark else INK,
                 fontweight="bold" if blind else "normal")
ax2.set_title("attack alarm rate ÷ baseline alarm rate",
              fontsize=TITLE_FONT - 1, color=INK, pad=8)

for a in (ax, ax2):
    a.set_facecolor(SURFACE)
    a.set_xlim(0, len(NS)); a.set_ylim(0, len(DEGS))
    a.set_xticks([i + .5 for i in range(len(NS))])
    a.set_xticklabels([nfmt[n] for n in NS], fontsize=LABEL_FONT - 1, color=INK2)
    a.set_yticks([i + .5 for i in range(len(DEGS))])
    a.set_yticklabels([dfmt[d] for d in DEGS], fontsize=LABEL_FONT - 1, color=INK2)
    a.set_xlabel("N (agents)", fontsize=LABEL_FONT, color=INK2)
    a.set_ylabel("mean degree", fontsize=LABEL_FONT, color=INK2)
    for s in a.spines.values():
        s.set_visible(False)
    a.tick_params(length=0)

legend = [Patch(facecolor="#d03b3b", label="≤ 1× — attack alarms no more than baseline (UNDETECTABLE)"),
          Patch(facecolor="#2a78d6", label="≫ 1× — attack alarms more (detectable)"),
          Line2D([0], [0], marker="s", linestyle="none", markersize=11,
                 markerfacecolor="none", markeredgecolor=INK, markeredgewidth=2.8,
                 label="outlined = takeover (victim infection > 0.5) AND alarm ratio ≤ 1×")]
fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False,
           fontsize=8.5, bbox_to_anchor=(0.5, 0.005))
fig.suptitle(TITLE, fontsize=TITLE_FONT + 2, color=INK, y=0.955)
fig.savefig(OUT, dpi=170, facecolor=SURFACE)
print(OUT)
