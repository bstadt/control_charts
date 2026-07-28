"""detect_grid comparison figure, one column per grid.

Row 1: infected fraction among VICTIMS (non-adversary agents) — for scaled
       adversary populations the adversaries would otherwise floor the metric.
Row 2: alarm rate during the attack RELATIVE TO BASELINE (out-of-band rate of
       the adaptive 3sigma chart in adv runs / same in clean runs, steps >=
       ATTACK, pooled over seeds, +1-smoothed). <= 1 renders red (undetectable);
       only ratios above 1 shade toward blue.
Full-mesh cells sit at their true mean degree (N-1); no separate "full" row.
Cells with takeover AND alarm ratio <= 1 (total compromise, no alarm) are outlined.

Usage: plot_grids.py <out.png> <label>=<glob-or-json> ...
"""
import json, glob, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
cmap_seq = LinearSegmentedColormap.from_list("seqblue", SEQ)
cmap_div = LinearSegmentedColormap.from_list("ratio", [
    (0.00, "#6b0f0f"), (0.50, "#d03b3b"), (0.56, "#e88a83"),
    (0.63, "#f0efec"), (0.75, "#86b6ef"), (1.00, "#104281"),
])

BURN, ATTACK, WINDOW_ITERS, K, LOGLIM = 100, 400, 200, 3.0, 1.5


def alarm_counts(steps, iso, k=K):
    steps = np.asarray(steps); iso = np.asarray(iso)
    interval = int(steps[1] - steps[0]) if len(steps) > 1 else 1
    window = max(1, WINDOW_ITERS // interval)
    burn_idx = int(np.searchsorted(steps, BURN))
    alarms = evals = 0
    for i in range(len(steps)):
        if i - window < burn_idx or steps[i] < ATTACK:
            continue
        w = iso[i - window:i]
        m, sd = w.mean(), w.std()
        evals += 1
        if iso[i] > m + k * sd or iso[i] < m - k * sd:
            alarms += 1
    return alarms, evals


def load(src):
    if src.endswith(".json") and "*" not in src:
        return json.load(open(src))
    return [json.load(open(f)) for f in glob.glob(src)]


def true_degree(N, deg):
    return N - 1 if (deg is None or float(deg) >= N - 1) else int(deg)


def _binom_sf(k, n, p=0.5):
    """P(X >= k) for X ~ Binomial(n, p), exact via the regularized incomplete
    beta identity. Used to ask whether the attack's alarms outnumber the
    baseline's by more than chance."""
    if n == 0:
        return 1.0
    if k <= 0:
        return 1.0
    from math import lgamma, exp
    total = 0.0
    for i in range(int(k), int(n) + 1):
        logc = lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1)
        total += exp(logc + i * np.log(p) + (n - i) * np.log(1 - p))
    return min(1.0, total)


def agg(cells):
    g = defaultdict(lambda: dict(inf=[], adv=[0, 0], base=[0, 0], nadv=[]))
    for r in cells:
        e = g[(r["N"], true_degree(r["N"], r["degree"]))]
        a, n = alarm_counts(r["iso_steps"], r["iso_values"]) if r.get("iso_steps") else (0, 0)
        if r["adversary"]:
            # victims-only where available (scaled-adversary runs), else the
            # whole-network fraction (single-adversary runs: 1 agent, no skew)
            v = r.get("infected_frac_victims")
            e["inf"].append(v if v is not None else r["infected_frac"])
            e["nadv"].append(r.get("n_adversaries") or 1)
            e["adv"][0] += a; e["adv"][1] += n
        else:
            e["base"][0] += a; e["base"][1] += n
    out = {}
    for k_, e in g.items():
        ar = (e["adv"][0] + 1) / (e["adv"][1] + 1) if e["adv"][1] else None
        br = (e["base"][0] + 1) / (e["base"][1] + 1) if e["base"][1] else None
        # Is the attack's alarm count actually above the baseline's, or is the
        # ratio just the +1 smoothing talking? Both arms get the same number of
        # evaluations, so under H0 each alarm is equally likely to land in
        # either arm: one-sided binomial on the split. Without this a single
        # alarm vs zero reads as a decisive "2.0x".
        na, nb = e["adv"][0], e["base"][0]
        pval = _binom_sf(na, na + nb, 0.5) if (na + nb) else 1.0
        out[k_] = dict(inf=np.mean(e["inf"]) if e["inf"] else None,
                       ratio=(ar / br) if ar and br else None,
                       nadv=int(np.mean(e["nadv"])) if e["nadv"] else None,
                       adv_alarms=na, base_alarms=nb, evals=e["adv"][1],
                       sig=(pval < 0.05))
    return out


def rfmt(r):
    if r >= 10: return f"{r:.0f}×"
    if r >= 1:  return f"{r:.1f}×"
    return f"{r:.2f}×"


def main():
    # --ns / --degs restrict the axes (e.g. --ns 100,1000,10000,100000 drops the
    # N=5 and N=10 columns). Stripped before the label=glob positionals.
    argv, ns_sel, degs_sel = [], None, None
    it = iter(sys.argv[1:])
    for a in it:
        if a == "--ns":
            ns_sel = [int(x) for x in next(it).split(",")]
        elif a == "--degs":
            degs_sel = [int(x) for x in next(it).split(",")]
        else:
            argv.append(a)

    out_png = argv[0]
    grids = {}
    seeds_seen = set()
    for arg in argv[1:]:
        # rsplit: labels themselves contain '=' (e.g. "Q=500, 20% adversarial")
        label, src = arg.rsplit("=", 1)
        cells = load(src)
        if not cells:
            print(f"WARN: no cells for {label} ({src})"); continue
        grids[label] = agg(cells)
        seeds_seen.update(r["seed"] for r in cells)
        print(f"{label}: {len(cells)} cells, {len({r['seed'] for r in cells})} seeds")
    if not grids:
        sys.exit("no data")

    NS = ns_sel or [5, 10, 100, 1000, 10000, 100000]
    present = {k_[1] for g in grids.values() for k_ in g}
    DEGS = [d for d in (degs_sel or [4, 9, 99, 999, 9999, 99999]) if d in present]
    nfmt = {5: "5", 10: "10", 100: "100", 1000: "1k", 10000: "10k", 100000: "100k"}
    dfmt = {4: "4", 9: "9", 99: "99", 999: "999", 9999: "9,999", 99999: "99,999"}

    ncol = len(grids)
    fig, axes = plt.subplots(2, ncol, figsize=(4.9 * ncol + 1.0, 9.0),
                             facecolor=SURFACE, squeeze=False)
    fig.subplots_adjust(left=0.075, right=0.98, top=0.895, bottom=0.205,
                        hspace=0.40, wspace=0.22)

    def rect(ax, x, y, color, outline=False):
        # outline marks the claim cell: takeover with no alarm increase.
        ax.add_patch(plt.Rectangle((x + 0.04, y + 0.04), 0.92, 0.92,
                                   facecolor=color,
                                   edgecolor=INK if outline else "none",
                                   linewidth=2.8 if outline else 0))

    for col, (label, g) in enumerate(grids.items()):
        ax = axes[0][col]; ax.set_facecolor(SURFACE)
        for xi, N in enumerate(NS):
            for yi, dg in enumerate(DEGS):
                e = g.get((N, dg))
                if e is None or e["inf"] is None:
                    continue
                rect(ax, xi, yi, cmap_seq(e["inf"]))
                ax.text(xi + .5, yi + .5, f"{e['inf']:.2f}", ha="center", va="center",
                        fontsize=9, color="#ffffff" if e["inf"] > .55 else INK)
        ax.set_title(f"{label}\ninfected fraction of victims", fontsize=10.5,
                     color=INK, pad=8)

        ax2 = axes[1][col]; ax2.set_facecolor(SURFACE)
        for xi, N in enumerate(NS):
            for yi, dg in enumerate(DEGS):
                e = g.get((N, dg))
                if e is None or e["ratio"] is None:
                    continue
                # Color is the ratio, full stop: <=1 red (undetectable), >1
                # shading to blue, lower = hotter.
                t = (np.clip(np.log10(e["ratio"]), -LOGLIM, LOGLIM) + LOGLIM) / (2 * LOGLIM)
                taken = e["inf"] is not None and e["inf"] > 0.5
                blind = taken and e["ratio"] <= 1.0
                rect(ax2, xi, yi, cmap_div(t), outline=blind)
                dark = t <= .53 or t > .82
                ax2.text(xi + .5, yi + .5, rfmt(e["ratio"]), ha="center", va="center",
                         fontsize=9, color="#ffffff" if dark else INK,
                         fontweight="bold" if blind else "normal")
        ax2.set_title(f"{label}\nattack alarm rate ÷ baseline alarm rate",
                      fontsize=10.5, color=INK, pad=8)

    for row in axes:
        for ax in row:
            ax.set_xlim(0, len(NS)); ax.set_ylim(0, len(DEGS))
            ax.set_xticks([i + .5 for i in range(len(NS))])
            ax.set_xticklabels([nfmt[n] for n in NS], fontsize=9, color=INK2)
            ax.set_yticks([i + .5 for i in range(len(DEGS))])
            ax.set_yticklabels([dfmt[d] for d in DEGS], fontsize=9, color=INK2)
            ax.set_xlabel("N (agents)", fontsize=9.5, color=INK2)
            ax.set_ylabel("mean degree", fontsize=9.5, color=INK2)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.tick_params(length=0)

    legend = [Patch(facecolor="#d03b3b", label="≤ 1× — attack alarms no more than baseline (UNDETECTABLE)"),
              Patch(facecolor="#2a78d6", label="≫ 1× — attack alarms more (detectable)"),
              Line2D([0], [0], marker="s", linestyle="none", markersize=11,
                     markerfacecolor="none", markeredgecolor=INK, markeredgewidth=2.8,
                     label="outlined = takeover (victim infection > 0.5) AND alarm ratio ≤ 1×")]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 0.005))
    ns = len(seeds_seen)
    fig.text(0.5, 0.090,
             f"{ns} seeds per cell. Diagonal cells (mean degree = N−1) are the full-mesh runs. Alarm rate = share of adaptive-chart\n"
             f"evaluations (3σ trailing window) out of band during steps ≥ 400, pooled over the {ns} runs, +1-smoothed.",
             ha="center", fontsize=7.5, color=INK2, style="italic", linespacing=1.5)
    fig.suptitle("Quine takeover and detectability vs question-pool size and adversary population",
                 fontsize=13.5, color=INK, y=0.972)
    fig.savefig(out_png, dpi=170, facecolor=SURFACE)
    print(out_png)


if __name__ == "__main__":
    main()
