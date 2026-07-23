"""Compare fastsim runs against archived original-simulator runs.

Usage:
  python -m controlcharts.fastsim.validate \
      --orig experiments/results/nollm-pcross075-s4* \
      --fast experiments/results/nollm-pcross075-s4*_fast_* \
      --out validate_pcross075.png

Compares, per step: mean known questions per agent, queries per step, quine
fraction in snapshot responses, temporal staleness, and the iso-mirror
trajectory (Isomap over TDKPS embeddings, as in figure3).
"""

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QUINE = "i lost the game"


def load_orig_history(run_dir: Path):
    with open(run_dir / "results.json") as f:
        data = json.load(f)
    steps, known, queries = [], [], []
    for entry in data["hook_history"]:
        steps.append(entry["step"])
        known.append(np.mean([a["known_count"] for a in entry["agents"]]))
    for r in data["results"]:
        queries.append(r["num_queries"])
    return np.array(steps), np.array(known), np.array(queries)


def load_fast_history(run_dir: Path):
    with open(run_dir / "results_summary.json") as f:
        data = json.load(f)
    h = data["history"]
    steps = np.array([r["step"] for r in h])
    known = np.array([r["mean_known"] for r in h])
    queries = np.array([r["num_queries"] for r in h])
    return steps, known, queries


def load_snapshot_series(run_dir: Path):
    """Per snapshot step: quine fraction, temporal staleness, temporal wrong-type frac."""
    metas = sorted(glob.glob(str(run_dir / "snapshots" / "snapshot_step_*_meta.json")))
    steps, quine_frac, staleness, t_wrong = [], [], [], []
    for mp in metas:
        with open(mp) as f:
            m = json.load(f)
        steps.append(m["step"])
        responses = m["responses"]
        flat = [r for row in responses for r in row]
        quine_frac.append(np.mean([QUINE in r.lower() for r in flat]))

        tmask = m.get("temporal_mask")
        tvals = m.get("temporal_values", {})
        if tmask and any(tmask) and tvals:
            qs = m["questions"]
            stales, wrong = [], 0
            n_t = 0
            for row in responses:
                for j, r in enumerate(row):
                    if not tmask[j]:
                        continue
                    n_t += 1
                    cur = tvals.get(qs[j])
                    nums = re.findall(r"\d+", r)
                    if nums and QUINE not in r.lower():
                        stales.append(cur - int(nums[0]))
                    else:
                        wrong += 1
            staleness.append(np.mean(stales) if stales else np.nan)
            t_wrong.append(wrong / max(n_t, 1))
        else:
            staleness.append(np.nan)
            t_wrong.append(np.nan)
    return np.array(steps), np.array(quine_frac), np.array(staleness), np.array(t_wrong)


def load_isomirror(run_dir: Path):
    from sklearn.manifold import Isomap
    npz_files = glob.glob(str(run_dir / "*_tdkps_embeddings.npz"))
    if not npz_files:
        return None, None
    d = np.load(npz_files[0], allow_pickle=True)
    em, steps = d["embedding_matrix"], d["steps"]
    flat = em.reshape(em.shape[0], -1)
    dist = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=2)
    iso = Isomap(n_components=1, metric="precomputed",
                 n_neighbors=min(5, len(steps) - 1)).fit_transform(dist)[:, 0]
    # orientation and offset are arbitrary per fit: anchor start at 0, end positive
    iso = iso - iso[0]
    if iso[-1] < 0:
        iso = -iso
    return steps, iso


def band(ax, series, color, label):
    """Plot mean with min-max band over runs; series = list of (x, y)."""
    if not series:
        return
    xs = series[0][0]
    ys = np.stack([y for _, y in series if len(y) == len(xs)])
    ax.plot(xs, np.nanmean(ys, axis=0), color=color, label=f"{label} (n={len(ys)})", lw=2)
    ax.fill_between(xs, np.nanmin(ys, axis=0), np.nanmax(ys, axis=0), color=color, alpha=0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", nargs="+", required=True)
    ap.add_argument("--fast", nargs="+", required=True)
    ap.add_argument("--out", default="fastsim_validation.png")
    args = ap.parse_args()

    orig_dirs = [Path(d) for pat in args.orig for d in sorted(glob.glob(pat))]
    fast_dirs = [Path(d) for pat in args.fast for d in sorted(glob.glob(pat))]
    print(f"orig runs: {[d.name for d in orig_dirs]}")
    print(f"fast runs: {[d.name for d in fast_dirs]}")

    o_hist = [load_orig_history(d) for d in orig_dirs]
    f_hist = [load_fast_history(d) for d in fast_dirs]
    o_snap = [load_snapshot_series(d) for d in orig_dirs]
    f_snap = [load_snapshot_series(d) for d in fast_dirs]
    o_iso = [load_isomirror(d) for d in orig_dirs]
    f_iso = [load_isomirror(d) for d in fast_dirs]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    ax = axes[0, 0]
    band(ax, [(s, k) for s, k, _ in o_hist], "C0", "original")
    band(ax, [(s, k) for s, k, _ in f_hist], "C1", "fastsim")
    ax.set_title("Mean known questions / agent"); ax.set_xlabel("step"); ax.legend()

    ax = axes[0, 1]
    band(ax, [(s, q) for s, _, q in o_hist], "C0", "original")
    band(ax, [(s, q) for s, _, q in f_hist], "C1", "fastsim")
    ax.set_title("Queries per step"); ax.set_xlabel("step"); ax.legend()

    ax = axes[0, 2]
    band(ax, [(s, q) for s, q, _, _ in o_snap], "C0", "original")
    band(ax, [(s, q) for s, q, _, _ in f_snap], "C1", "fastsim")
    ax.set_title("Quine fraction in snapshot responses"); ax.set_xlabel("step"); ax.legend()

    ax = axes[1, 0]
    band(ax, [(s, st) for s, _, st, _ in o_snap], "C0", "original")
    band(ax, [(s, st) for s, _, st, _ in f_snap], "C1", "fastsim")
    ax.set_title("Temporal staleness (steps behind)"); ax.set_xlabel("step"); ax.legend()

    ax = axes[1, 1]
    band(ax, [(s, w) for s, _, _, w in o_snap], "C0", "original")
    band(ax, [(s, w) for s, _, _, w in f_snap], "C1", "fastsim")
    ax.set_title("Temporal wrong-type fraction (quine/IDK)"); ax.set_xlabel("step"); ax.legend()

    ax = axes[1, 2]
    for s, iso in o_iso:
        if s is not None:
            ax.plot(s, iso, color="C0", alpha=0.6)
    for s, iso in f_iso:
        if s is not None:
            ax.plot(s, iso, color="C1", alpha=0.6)
    ax.set_title("Iso-mirror trajectories (blue=orig, orange=fast)")
    ax.set_xlabel("step")

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")

    # numeric summary over the post-attack window
    def window_mean(series_list, lo, hi):
        vals = []
        for s, y in series_list:
            selm = (s >= lo) & (s <= hi)
            vals.append(np.nanmean(y[selm]))
        return np.mean(vals), np.std(vals)

    for label, oh, fh in [("known/agent", [(s, k) for s, k, _ in o_hist], [(s, k) for s, k, _ in f_hist]),
                          ("queries/step", [(s, q) for s, _, q in o_hist], [(s, q) for s, _, q in f_hist])]:
        om, osd = window_mean(oh, 1000, 2000)
        fm, fsd = window_mean(fh, 1000, 2000)
        print(f"{label} [1000,2000]: orig {om:.2f}±{osd:.2f}  fast {fm:.2f}±{fsd:.2f}")
    for label, oS, fS, idx in [("quine frac", o_snap, f_snap, 1), ("staleness", o_snap, f_snap, 2)]:
        om, osd = window_mean([(s[0], s[idx]) for s in oS], 1000, 2000)
        fm, fsd = window_mean([(s[0], s[idx]) for s in fS], 1000, 2000)
        print(f"{label} [1000,2000]: orig {om:.3f}±{osd:.3f}  fast {fm:.3f}±{fsd:.3f}")


if __name__ == "__main__":
    main()
