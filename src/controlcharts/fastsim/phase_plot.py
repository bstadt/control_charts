"""Aggregate phase-sweep cell results into the (N, density) phase diagram.

Reads <out_dir>/cells/*.json, averages over seeds, and renders a heatmap over
(N, mean_degree) whose colour encodes the regime:
  0  no intrusion            (quine dies below the epidemic threshold)
  1  intrusion, detected     (sharp takeover the adaptive chart catches)
  2  intrusion, UNDETECTED   (slow takeover absorbed as baseline drift)  <-- target
Cell value = mean over seeds of the per-seed regime; a fractional value marks
a boundary cell where seeds disagree.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


def regime(c):
    if not c["intrusion"]:
        return 0
    if c["detected"]:
        return 1
    return 2  # intrusion and not detected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out_dir)

    cells = [json.load(open(f)) for f in glob.glob(str(out / "cells" / "*.json"))]
    if not cells:
        print("no cells"); return

    Ns = sorted({c["N"] for c in cells})
    ks = sorted({c["mean_degree"] for c in cells})
    # regime + infected fraction, averaged over seeds
    reg = np.full((len(ks), len(Ns)), np.nan)
    inf = np.full((len(ks), len(Ns)), np.nan)
    und = np.full((len(ks), len(Ns)), np.nan)   # fraction of seeds undetected-intrusion
    for ik, k in enumerate(ks):
        for iN, N in enumerate(Ns):
            grp = [c for c in cells if c["mean_degree"] == k and c["N"] == N]
            if grp:
                reg[ik, iN] = np.mean([regime(c) for c in grp])
                inf[ik, iN] = np.mean([c["infected_frac"] for c in grp])
                und[ik, iN] = np.mean([1.0 if regime(c) == 2 else 0.0 for c in grp])

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    cmap = ListedColormap(["#2c3e50", "#27ae60", "#c0392b"])  # none / detected / UNDETECTED
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    ax = axes[0]
    im = ax.imshow(np.round(reg), origin="lower", aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(Ns))); ax.set_xticklabels(Ns)
    ax.set_yticks(range(len(ks))); ax.set_yticklabels([f"{k:g}" for k in ks])
    ax.set_xlabel("N (agents)"); ax.set_ylabel("mean degree (density axis)")
    ax.set_title("Regime  (rounded seed-mean)")
    for ik in range(len(ks)):
        for iN in range(len(Ns)):
            if not np.isnan(und[ik, iN]):
                ax.text(iN, ik, f"{und[ik, iN]:.0%}", ha="center", va="center",
                        color="white", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["no intrusion", "detected", "UNDETECTED"])

    ax = axes[1]
    im2 = ax.imshow(inf, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(Ns))); ax.set_xticklabels(Ns)
    ax.set_yticks(range(len(ks))); ax.set_yticklabels([f"{k:g}" for k in ks])
    ax.set_xlabel("N (agents)"); ax.set_ylabel("mean degree")
    ax.set_title("Final infected fraction")
    fig.colorbar(im2, ax=ax)

    fig.suptitle("Intrusion detectability phase diagram (cell label = % seeds undetected-intrusion)")
    fig.tight_layout()
    outpath = args.out or str(out / "phase_diagram.png")
    fig.savefig(outpath, dpi=140)
    print(f"wrote {outpath}")

    # text summary
    print("\nUNDETECTED-INTRUSION cells (the vulnerable region):")
    for c in sorted(cells, key=lambda c: (c["N"], c["mean_degree"])):
        if regime(c) == 2:
            print(f"  N={c['N']} k={c['mean_degree']:g} density={c['density']:.1e} "
                  f"seed={c['seed']} infected={c['infected_frac']:.2f}")


if __name__ == "__main__":
    main()
