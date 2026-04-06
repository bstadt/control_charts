#!/usr/bin/env python3
"""Figure 4: P(control exceeded) vs time for grid search conditions."""

import glob
import hashlib
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import Isomap


RESULTS_DIR = os.path.expanduser("~/root/controlcharts/experiments/results")
CACHE_DIR = os.path.join(RESULTS_DIR, ".iso_cache")
ROW_CONDITIONS = [
    ("noadv", "No Adversary"),
    ("d100", "Attack Duration 100"),
    ("d800", "Attack Duration 800"),
]
K_VALUES = [2, 3, 5]
BURN_IN = 100
WINDOW_SIZE_ITERS = 200


def load_isomirror(result_dir):
    """Load embeddings, compute isomap 1D with caching."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_key = hashlib.md5(result_dir.encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

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

    with open(cache_path, "wb") as f:
        pickle.dump((steps, iso_values), f)

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
    post_burn_steps = []

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
        post_burn_steps.append(steps[i])

    return np.array(post_burn_steps), np.array(exceeded)


def main():
    # Collect all result directories per condition
    condition_dirs = {}
    for cond, label in ROW_CONDITIONS:
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
        condition_dirs[cond] = [by_rep[r] for r in sorted(by_rep.keys())]
        print(f"{cond}: {len(condition_dirs[cond])} replications")

    colors = {
        "noadv": "#51915B",
        "d100": "#EA5526",
        "d800": "#4462BD",
    }

    fig, axes = plt.subplots(3, 3, figsize=(18, 5),
                             gridspec_kw={"hspace": 0.3, "wspace": 0.1})

    for row, (cond, row_label) in enumerate(ROW_CONDITIONS):
        dirs = condition_dirs[cond]
        color = colors[cond]

        for col, k in enumerate(K_VALUES):
            ax = axes[row, col]

            if dirs:
                all_exceeded = []
                common_steps = None

                for d in dirs:
                    steps, iso_values = load_isomirror(d)
                    if steps is None:
                        continue
                    post_steps, exceeded = compute_exceeded(steps, iso_values, k)
                    all_exceeded.append(exceeded)
                    if common_steps is None:
                        common_steps = post_steps

                if all_exceeded:
                    min_len = min(len(e) for e in all_exceeded)
                    exceeded_matrix = np.array([e[:min_len] for e in all_exceeded])
                    freq = np.mean(exceeded_matrix, axis=0)
                    time_steps = common_steps[:min_len]

                    ax.plot(time_steps, freq, color=color, linewidth=2, label=row_label)

            ax.set_xlim(200, 1500)
            ax.set_ylim(-0.05, 1.05)
            ax.tick_params(labelsize=14)
            ax.grid(True, alpha=0.3)

            if row == 0:
                ax.set_title(f"k = {k}$\\sigma$", fontsize=20)

            if col != 0:
                ax.set_yticklabels([])

            if row == 2:
                ax.set_xlabel("Iteration", fontsize=18)
            else:
                ax.set_xticklabels([])

    # Shared y-axis label
    fig.text(0.08, 0.5, "P(exceeded)", va="center", rotation="vertical", fontsize=20)

    # Legend in top-right panel
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=colors[cond], linewidth=2, label=label)
        for cond, label in ROW_CONDITIONS
    ]
    axes[0, 2].legend(handles=legend_elements, fontsize=11, loc="center right",
                      framealpha=1, edgecolor="0.8")

    fig.suptitle("Empirical Frequency of Control Chart Exceedance", fontsize=22, y=1.0)

    out_path = os.path.join(RESULTS_DIR, "figure4.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
