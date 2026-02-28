#!/usr/bin/env python3
"""Figure 4: P(control exceeded) vs time for grid search conditions."""

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import Isomap


RESULTS_DIR = os.path.expanduser("~/root/controlcharts/experiments/results")
CONDITIONS = {
    "noadv": "No Adversary",
    "d100": "Attack Duration 100",
    "d800": "Attack Duration 800",
    "d1600": "Attack Duration 1600",
}
K_VALUES = [2, 3, 5]
BURN_IN = 100
WINDOW_SIZE_ITERS = 200


def load_isomirror(result_dir):
    """Load embeddings, compute isomap 1D, return (steps, iso_values)."""
    npz_files = glob.glob(os.path.join(result_dir, "*_tdkps_embeddings.npz"))
    if not npz_files:
        return None, None
    d = np.load(npz_files[0], allow_pickle=True)
    embedding_matrix = d["embedding_matrix"]  # [n_timesteps, n_agents, n_components]
    steps = d["steps"]
    n_timesteps = embedding_matrix.shape[0]

    # Compute distance matrix
    flat = embedding_matrix.reshape(n_timesteps, -1)
    dist_matrix = np.zeros((n_timesteps, n_timesteps))
    for i in range(n_timesteps):
        for j in range(n_timesteps):
            dist_matrix[i, j] = np.linalg.norm(flat[i] - flat[j])

    # Isomap 1D
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
    for cond in CONDITIONS:
        pattern = os.path.join(RESULTS_DIR, f"grid-search-{cond}-*")
        dirs = sorted(glob.glob(pattern))
        # Filter to only directories (not files)
        dirs = [d for d in dirs if os.path.isdir(d)]
        condition_dirs[cond] = dirs
        print(f"{cond}: {len(dirs)} replications")

    # For each condition and k, compute exceeded across replications
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    colors = {
        "noadv": "#2ecc71",
        "d100": "#3498db",
        "d800": "#e67e22",
        "d1600": "#e74c3c",
    }

    for col, k in enumerate(K_VALUES):
        ax = axes[col]

        for cond in CONDITIONS:
            dirs = condition_dirs[cond]
            if not dirs:
                continue

            # Compute exceeded for each replication
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

            if not all_exceeded:
                continue

            # Stack and compute empirical frequency
            min_len = min(len(e) for e in all_exceeded)
            exceeded_matrix = np.array([e[:min_len] for e in all_exceeded])  # [n_reps, n_timesteps]
            freq = np.mean(exceeded_matrix, axis=0)  # [n_timesteps]
            time_steps = common_steps[:min_len]

            ax.plot(time_steps, freq, label=CONDITIONS[cond], color=colors[cond], linewidth=2)

        ax.set_xlim(left=200)
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_title(f"k = {k}σ", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        if col == 0:
            ax.set_ylabel("P(control exceeded)", fontsize=12)
        ax.legend(fontsize=10)

    fig.suptitle("Empirical Frequency of Control Chart Exceedance", fontsize=16, y=1.02)
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "figure4.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
