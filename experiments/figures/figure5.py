#!/usr/bin/env python3
"""Figure 5: Precision-Recall curves for attack detection at varying k thresholds."""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import Isomap


RESULTS_DIR = os.path.expanduser("~/root/controlcharts/experiments/results")
BURN_IN = 100
WINDOW_SIZE_ITERS = 200
K_RANGE = np.arange(2, 9)  # sigma 2 through 8


def load_isomirror(result_dir):
    """Load embeddings, compute isomap 1D, return (steps, iso_values)."""
    npz_files = glob.glob(os.path.join(result_dir, "*_tdkps_embeddings.npz"))
    if not npz_files:
        return None, None
    d = np.load(npz_files[0], allow_pickle=True)
    embedding_matrix = d["embedding_matrix"]
    steps = d["steps"]
    n_timesteps = embedding_matrix.shape[0]

    flat = embedding_matrix.reshape(n_timesteps, -1)
    dist_matrix = np.zeros((n_timesteps, n_timesteps))
    for i in range(n_timesteps):
        for j in range(n_timesteps):
            dist_matrix[i, j] = np.linalg.norm(flat[i] - flat[j])

    isomap = Isomap(n_components=1, metric="precomputed", n_neighbors=min(5, n_timesteps - 1))
    iso_values = isomap.fit_transform(dist_matrix)[:, 0]
    return steps, iso_values


def compute_exceeded(steps, iso_values, k, burn_in=BURN_IN, window_size_iters=WINDOW_SIZE_ITERS):
    """Return boolean array: True where control is exceeded at each post-burn-in step."""
    snapshot_interval = steps[1] - steps[0] if len(steps) > 1 else 1
    window_size = max(1, window_size_iters // snapshot_interval)

    burn_in_idx = 0
    for i, s in enumerate(steps):
        if s >= burn_in:
            burn_in_idx = i
            break

    exceeded = []
    for i in range(len(steps)):
        if i < burn_in_idx:
            continue
        current_val = iso_values[i]
        window_start = max(burn_in_idx, i - window_size)
        window_values = iso_values[window_start:i]

        if len(window_values) > 1:
            window_mean = np.mean(window_values)
            window_std = np.std(window_values)
        else:
            window_mean = window_values[0] if len(window_values) > 0 else 0
            window_std = 0

        upper = window_mean + k * window_std
        lower = window_mean - k * window_std
        exceeded.append(not (lower <= current_val <= upper))

    return np.array(exceeded)


def get_filtered_dirs(cond):
    """Get deduplicated dirs for reps 0-99."""
    pattern = os.path.join(RESULTS_DIR, f"grid-search-{cond}-*")
    dirs = sorted(glob.glob(pattern))
    dirs = [d for d in dirs if os.path.isdir(d)]
    by_rep = {}
    for d in dirs:
        basename = os.path.basename(d)
        parts = basename.replace(f"grid-search-{cond}-", "").split("_", 1)
        try:
            rep = int(parts[0])
            if 0 <= rep <= 99:
                by_rep[rep] = d
        except ValueError:
            pass
    return [by_rep[r] for r in sorted(by_rep.keys())]


def precompute_isomirrors(dirs):
    """Load all isomirror data once."""
    results = []
    for d in dirs:
        steps, iso_values = load_isomirror(d)
        if steps is not None:
            results.append((steps, iso_values))
    return results


def main():
    # Load directories
    noadv_dirs = get_filtered_dirs("noadv")
    d100_dirs = get_filtered_dirs("d100")
    d800_dirs = get_filtered_dirs("d800")
    print(f"noadv: {len(noadv_dirs)}, d100: {len(d100_dirs)}, d800: {len(d800_dirs)}")

    # Precompute isomirrors
    print("Computing isomirrors for noadv...")
    noadv_iso = precompute_isomirrors(noadv_dirs)
    print("Computing isomirrors for d100...")
    d100_iso = precompute_isomirrors(d100_dirs)
    print("Computing isomirrors for d800...")
    d800_iso = precompute_isomirrors(d800_dirs)

    # For each k, compute precision and recall
    d100_precision = []
    d100_recall = []
    d800_precision = []
    d800_recall = []

    for k in K_RANGE:
        print(f"  k={k}...")

        # Count exceedances for noadv
        noadv_total_exceedances = 0
        for steps, iso_values in noadv_iso:
            exceeded = compute_exceeded(steps, iso_values, k)
            noadv_total_exceedances += np.sum(exceeded)

        # d100
        d100_total_exceedances = 0
        d100_any_alarm = 0
        for steps, iso_values in d100_iso:
            exceeded = compute_exceeded(steps, iso_values, k)
            n_exc = np.sum(exceeded)
            d100_total_exceedances += n_exc
            if n_exc > 0:
                d100_any_alarm += 1

        total_exc_100 = d100_total_exceedances + noadv_total_exceedances
        prec_100 = d100_total_exceedances / total_exc_100 if total_exc_100 > 0 else 0
        rec_100 = d100_any_alarm / len(d100_iso) if len(d100_iso) > 0 else 0
        d100_precision.append(prec_100)
        d100_recall.append(rec_100)

        # d800
        d800_total_exceedances = 0
        d800_any_alarm = 0
        for steps, iso_values in d800_iso:
            exceeded = compute_exceeded(steps, iso_values, k)
            n_exc = np.sum(exceeded)
            d800_total_exceedances += n_exc
            if n_exc > 0:
                d800_any_alarm += 1

        total_exc_800 = d800_total_exceedances + noadv_total_exceedances
        prec_800 = d800_total_exceedances / total_exc_800 if total_exc_800 > 0 else 0
        rec_800 = d800_any_alarm / len(d800_iso) if len(d800_iso) > 0 else 0
        d800_precision.append(prec_800)
        d800_recall.append(rec_800)

        print(f"    d100: precision={prec_100:.3f}, recall={rec_100:.3f}")
        print(f"    d800: precision={prec_800:.3f}, recall={rec_800:.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(K_RANGE, d100_precision, "o-", color="#EA5526", linewidth=2,
            markersize=8, label="Attack Duration 100")
    ax.plot(K_RANGE, d800_precision, "s-", color="#4462BD", linewidth=2,
            markersize=8, label="Attack Duration 800")

    ax.set_xlabel("$k$ ($\\sigma$)", fontsize=14)
    ax.set_ylabel("Precision", fontsize=14)
    ax.set_title("Detection Precision vs Threshold", fontsize=15)
    ax.set_xticks(K_RANGE)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=12, loc="best", framealpha=1)
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(RESULTS_DIR, "figure5.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
