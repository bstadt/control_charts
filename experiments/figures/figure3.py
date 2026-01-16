"""
Figure 3: Control Chart Methods with Adversarial Attack

Layout (1 row, 2 columns):
- (0,0): Isomirror with CONSTANT control bars (mean/std from iterations 100-200)
- (0,1): Isomirror with sliding window EMA control bars

Shows detection of adversarial behavior starting at iteration 200.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import Isomap

# Path to figure3 experiment results
RESULTS_DIR = Path("experiments/results/figure3-adversarial_2026-01-15_17-53-22")
OUTPUT_PATH = Path("experiments/figures/figure3.png")


def load_tdkps_embeddings(results_dir: Path, experiment_name: str):
    """Load TDKPS embeddings from npz file."""
    path = results_dir / f"{experiment_name}_tdkps_embeddings.npz"
    data = np.load(path)
    return data["embedding_matrix"], data["steps"]


def compute_isomirror(embedding_matrix, steps):
    """Compute isomirror (1D Isomap of TDKPS distance matrix)."""
    n_timesteps = embedding_matrix.shape[0]

    # Flatten agent positions at each timestep
    flat_embeddings = embedding_matrix.reshape(n_timesteps, -1)

    # Create distance matrix
    distance_matrix = np.zeros((n_timesteps, n_timesteps))
    for i in range(n_timesteps):
        for j in range(n_timesteps):
            distance_matrix[i, j] = np.linalg.norm(flat_embeddings[i] - flat_embeddings[j])

    # Run Isomap
    isomap = Isomap(n_components=1, metric='precomputed', n_neighbors=min(5, n_timesteps - 1))
    isomap_embedding = isomap.fit_transform(distance_matrix)

    return isomap_embedding[:, 0]


def main():
    # Create output directory
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading TDKPS embeddings...")
    embedding_matrix, steps = load_tdkps_embeddings(RESULTS_DIR, "figure3-adversarial")

    print("Computing isomirror...")
    iso_values = compute_isomirror(embedding_matrix, steps)

    # Create figure with 2 panels
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ===== Panel 1: Constant control bars from iterations 100-200 =====
    ax1 = axes[0]

    # Find indices for iterations 100-200
    burn_in_start_idx = None
    burn_in_end_idx = None
    for i, s in enumerate(steps):
        if burn_in_start_idx is None and s >= 100:
            burn_in_start_idx = i
        if s >= 200:
            burn_in_end_idx = i
            break

    # Compute mean and std from burn-in period (100-200)
    burn_in_values = iso_values[burn_in_start_idx:burn_in_end_idx+1]
    burn_in_mean = np.mean(burn_in_values)
    burn_in_std = np.std(burn_in_values)
    k = 2

    upper_limit = burn_in_mean + k * burn_in_std
    lower_limit = burn_in_mean - k * burn_in_std

    print(f"Burn-in period (100-200): mean={burn_in_mean:.4f}, std={burn_in_std:.4f}")
    print(f"Control limits: [{lower_limit:.4f}, {upper_limit:.4f}]")

    # Color points based on whether they're within control limits
    colors1 = []
    for i, val in enumerate(iso_values):
        if steps[i] < 100:
            colors1.append('#3498db')  # Blue for pre-burn-in
        elif lower_limit <= val <= upper_limit:
            colors1.append('#27ae60')  # Green - in control
        else:
            colors1.append('#e74c3c')  # Red - out of control

    # Plot horizontal control bands
    ax1.axhspan(lower_limit, upper_limit, alpha=0.2, color='#95a5a6', label=f'Control Band (±{k}σ)')
    ax1.axhline(y=burn_in_mean, color='#7f8c8d', linestyle='--', linewidth=1, alpha=0.7, label='Center Line')
    ax1.axhline(y=upper_limit, color='#e74c3c', linestyle='-', linewidth=1, alpha=0.5)
    ax1.axhline(y=lower_limit, color='#e74c3c', linestyle='-', linewidth=1, alpha=0.5)

    # Plot points with colors
    for i in range(len(steps)):
        ax1.plot(steps[i], iso_values[i], 'o', markersize=5, color=colors1[i])

    # Connect points with line
    ax1.plot(steps, iso_values, linewidth=1, color='#bdc3c7', alpha=0.5, zorder=1)

    # Add vertical lines at key boundaries
    ax1.axvline(x=100, color='#9b59b6', linestyle=':', linewidth=2, alpha=0.7, label='Burn-in Start')
    ax1.axvline(x=200, color='#e67e22', linestyle='--', linewidth=2, alpha=0.7, label='Adversarial Start')

    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Isomirror Value', fontsize=12)
    ax1.set_title('Constant Control Bars\n(Mean ± 2σ from iterations 100-200)', fontsize=12)
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # ===== Panel 2: Sliding window EMA control bars =====
    ax2 = axes[1]

    # EMA parameters
    burn_in = 100
    window_decay = 0.2  # alpha for EMA
    k = 2

    # Find burn-in index
    burn_in_idx = 0
    for i, s in enumerate(steps):
        if s >= burn_in:
            burn_in_idx = i
            break

    # Compute EMA control limits
    upper_limits = []
    lower_limits = []
    center_line = []
    colors2 = []

    ema_mean = None
    ema_var = None

    for i in range(len(steps)):
        if i < burn_in_idx:
            upper_limits.append(np.nan)
            lower_limits.append(np.nan)
            center_line.append(np.nan)
            colors2.append('#3498db')  # Blue for burn-in period
        else:
            current_val = iso_values[i]

            if ema_mean is None:
                # First point after burn-in: no prior state for bounds
                window_mean = current_val
                window_std = 0.0
                ema_mean = current_val
                ema_var = 0.0
            else:
                # Use PRIOR EMA state for control limits (no lookahead)
                window_mean = ema_mean
                window_std = np.sqrt(ema_var) if ema_var > 0 else 0
                # THEN update EMA with current value for next iteration
                delta = current_val - ema_mean
                ema_mean = ema_mean + window_decay * delta
                ema_var = (1 - window_decay) * (ema_var + window_decay * delta * delta)

            upper = window_mean + k * window_std
            lower = window_mean - k * window_std

            upper_limits.append(upper)
            lower_limits.append(lower)
            center_line.append(window_mean)

            if lower <= current_val <= upper:
                colors2.append('#27ae60')  # Green - in control
            else:
                colors2.append('#e74c3c')  # Red - out of control

    # Plot control bands (shaded region)
    valid_indices = [i for i in range(len(steps)) if not np.isnan(upper_limits[i])]
    if valid_indices:
        valid_steps = [steps[i] for i in valid_indices]
        valid_upper = [upper_limits[i] for i in valid_indices]
        valid_lower = [lower_limits[i] for i in valid_indices]
        valid_center = [center_line[i] for i in valid_indices]

        ax2.fill_between(valid_steps, valid_lower, valid_upper,
                        alpha=0.2, color='#95a5a6', label=f'Control Band (±{k}σ)')
        ax2.plot(valid_steps, valid_center, '--', color='#7f8c8d',
                linewidth=1, alpha=0.7, label='Center Line (EMA)')

    # Plot points with colors
    for i in range(len(steps)):
        ax2.plot(steps[i], iso_values[i], 'o', markersize=5, color=colors2[i])

    # Connect points with line
    ax2.plot(steps, iso_values, linewidth=1, color='#bdc3c7', alpha=0.5, zorder=1)

    # Add vertical lines at key boundaries
    ax2.axvline(x=burn_in, color='#9b59b6', linestyle=':', linewidth=2, alpha=0.7, label='Burn-in End')
    ax2.axvline(x=200, color='#e67e22', linestyle='--', linewidth=2, alpha=0.7, label='Adversarial Start')

    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Isomirror Value', fontsize=12)
    ax2.set_title(f'Sliding Window EMA Control Bars\n(burn_in={burn_in}, decay={window_decay}, k={k})', fontsize=12)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle('Figure 3: Control Charts with Adversarial Attack (0%→50% from iter 200-500)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Figure saved to {OUTPUT_PATH}")

    # Count out-of-control points for each method
    ooc_constant = sum(1 for c in colors1 if c == '#e74c3c')
    ooc_ema = sum(1 for c in colors2 if c == '#e74c3c')
    print(f"Out-of-control points (constant): {ooc_constant}")
    print(f"Out-of-control points (EMA): {ooc_ema}")

    # Find first detection point for constant method (after adversarial starts)
    for i, (s, c) in enumerate(zip(steps, colors1)):
        if s >= 200 and c == '#e74c3c':
            print(f"Constant method first detection: iteration {s}")
            break

    # Find first detection point for EMA method (after adversarial starts)  
    for i, (s, c) in enumerate(zip(steps, colors2)):
        if s >= 200 and c == '#e74c3c':
            print(f"EMA method first detection: iteration {s}")
            break


if __name__ == "__main__":
    main()
