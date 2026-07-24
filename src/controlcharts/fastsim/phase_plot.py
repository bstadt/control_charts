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
    # detection rate (detected replications / total replications) and mean
    # infected fraction, per (mean_degree, N) cell
    det_rate = np.full((len(ks), len(Ns)), np.nan)
    det_lbl = np.full((len(ks), len(Ns)), "", dtype=object)
    inf = np.full((len(ks), len(Ns)), np.nan)
    for ik, k in enumerate(ks):
        for iN, N in enumerate(Ns):
            grp = [c for c in cells if c["mean_degree"] == k and c["N"] == N]
            if grp:
                nd = sum(1 for c in grp if c["detected"])
                det_rate[ik, iN] = nd / len(grp)
                det_lbl[ik, iN] = f"{nd}/{len(grp)}"
                inf[ik, iN] = np.mean([c["infected_frac"] for c in grp])

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Detection rate: darker = more detections (Greys maps high->black)
    ax = axes[0]
    im = ax.imshow(det_rate, origin="lower", aspect="auto", cmap="Greys",
                   vmin=0, vmax=1)
    ax.set_xticks(range(len(Ns))); ax.set_xticklabels(Ns)
    ax.set_yticks(range(len(ks))); ax.set_yticklabels([f"{k:g}" for k in ks])
    ax.set_xlabel("N (agents)"); ax.set_ylabel("mean degree (density axis)")
    ax.set_title("Detections / replications  (darker = more detected)")
    for ik in range(len(ks)):
        for iN in range(len(Ns)):
            if det_lbl[ik, iN]:
                v = det_rate[ik, iN]
                ax.text(iN, ik, det_lbl[ik, iN], ha="center", va="center",
                        color=("white" if v > 0.5 else "black"), fontsize=8)
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 0.5, 1])
    cbar.set_label("fraction of replications detected")

    ax = axes[1]
    im2 = ax.imshow(inf, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(Ns))); ax.set_xticklabels(Ns)
    ax.set_yticks(range(len(ks))); ax.set_yticklabels([f"{k:g}" for k in ks])
    ax.set_xlabel("N (agents)"); ax.set_ylabel("mean degree")
    ax.set_title("Final infected fraction")
    fig.colorbar(im2, ax=ax)

    fig.suptitle("Intrusion detectability phase diagram (cell = detections/replications)")
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
