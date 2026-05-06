"""
Figure 2: Control Chart Methods Comparison

Layout (1 row, 2 columns):
- (0,0): Isomirror with CONSTANT control bars (mean/std from iterations 100-300)
- (0,1): Isomirror with sliding window control bars
Both panels show ±2σ (yellow) and ±3σ (red) bands
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import Isomap

RESULTS_DIR = Path("experiments/results/figure2-baseline-additive_2026-05-04_11-34-59")
OUTPUT_PATH = Path("experiments/figures/figure2.png")


def load_tdkps_embeddings(results_dir: Path, experiment_name: str):
    path = results_dir / f"{experiment_name}_tdkps_embeddings.npz"
    data = np.load(path)
    return data["embedding_matrix"], data["steps"]


def compute_isomirror(embedding_matrix, steps):
    n_timesteps = embedding_matrix.shape[0]
    flat_embeddings = embedding_matrix.reshape(n_timesteps, -1)

    distance_matrix = np.zeros((n_timesteps, n_timesteps))
    for i in range(n_timesteps):
        for j in range(n_timesteps):
            distance_matrix[i, j] = np.linalg.norm(flat_embeddings[i] - flat_embeddings[j])

    isomap = Isomap(n_components=1, metric="precomputed", n_neighbors=min(5, n_timesteps - 1))
    isomap_embedding = isomap.fit_transform(distance_matrix)
    return isomap_embedding[:, 0]


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading TDKPS embeddings...")
    embedding_matrix, steps = load_tdkps_embeddings(RESULTS_DIR, "figure2-baseline-additive")

    print("Computing isomirror...")
    iso_values = compute_isomirror(embedding_matrix, steps)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # ===== Panel 1: Constant control bars from iterations 100-300 =====
    ax1 = axes[0]

    burn_in_start_idx = None
    burn_in_end_idx = None
    for i, s in enumerate(steps):
        if burn_in_start_idx is None and s >= 100:
            burn_in_start_idx = i
        if s >= 300:
            burn_in_end_idx = i
            break

    burn_in_values = iso_values[burn_in_start_idx:burn_in_end_idx+1]
    burn_in_mean = np.mean(burn_in_values)
    burn_in_std = np.std(burn_in_values)

    upper_2 = burn_in_mean + 2 * burn_in_std
    lower_2 = burn_in_mean - 2 * burn_in_std
    upper_3 = burn_in_mean + 3 * burn_in_std
    lower_3 = burn_in_mean - 3 * burn_in_std

    print(f"Calibration period (100-300): mean={burn_in_mean:.4f}, std={burn_in_std:.4f}")
    print(f"2σ limits: [{lower_2:.4f}, {upper_2:.4f}]")
    print(f"3σ limits: [{lower_3:.4f}, {upper_3:.4f}]")

    # Color points: blue=burn-in, green=in control, yellow=outside 2σ, red=outside 3σ
    colors1 = []
    for i, val in enumerate(iso_values):
        if steps[i] < 300:
            colors1.append("#4462BD")  # blue - calibration period
        elif not (lower_3 <= val <= upper_3):
            colors1.append("#EA5526")  # red - outside 3σ
        elif not (lower_2 <= val <= upper_2):
            colors1.append("#E5B700")  # yellow - outside 2σ but inside 3σ
        else:
            colors1.append("#51915B")  # green - in control

    # Find x position where control bands should start (iteration 300)
    band_start_x = 300
    band_end_x = max(steps)

    # Plot bands starting at iteration 300
    ax1.fill_between([band_start_x, band_end_x], [lower_3, lower_3], [upper_3, upper_3], 
                     alpha=0.1, color="#EA5526", label="±3σ band")
    ax1.fill_between([band_start_x, band_end_x], [lower_2, lower_2], [upper_2, upper_2], 
                     alpha=0.2, color="#E5B700", label="±2σ band")
    ax1.hlines(y=burn_in_mean, xmin=band_start_x, xmax=band_end_x, color="#7f8c8d", 
               linestyle="--", linewidth=1, alpha=0.7, label="Center Line")
    ax1.hlines(y=upper_3, xmin=band_start_x, xmax=band_end_x, color="#EA5526", 
               linestyle="-", linewidth=1, alpha=0.5)
    ax1.hlines(y=lower_3, xmin=band_start_x, xmax=band_end_x, color="#EA5526", 
               linestyle="-", linewidth=1, alpha=0.5)
    ax1.hlines(y=upper_2, xmin=band_start_x, xmax=band_end_x, color="#E5B700", 
               linestyle="-", linewidth=1, alpha=0.5)
    ax1.hlines(y=lower_2, xmin=band_start_x, xmax=band_end_x, color="#E5B700", 
               linestyle="-", linewidth=1, alpha=0.5)

    for i in range(len(steps)):
        ax1.plot(steps[i], iso_values[i], "o", markersize=5, color=colors1[i])
    ax1.plot(steps, iso_values, linewidth=1, color="#bdc3c7", alpha=0.5, zorder=1)

    # Burn-in end at 100 and calibration end at 300
    ax1.axvline(x=100, color="#8064A2", linestyle=":", linewidth=2, alpha=0.7, label="Burn-in End")
    ax1.axvline(x=300, color="#51915B", linestyle="-.", linewidth=2, alpha=0.7, label="Calibration End")

    ax1.set_xlabel("Iteration", fontsize=16)
    ax1.set_ylabel("Isomirror Value", fontsize=16)
    ax1.set_title("Constant Control Bars\n(Mean ± 2σ/3σ from iterations 100-300)", fontsize=16)
    ax1.legend(fontsize=13, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # ===== Panel 2: Sliding window control bars =====
    ax2 = axes[1]

    burn_in = 100
    window_size = 200
    
    upper_2_limits = []
    lower_2_limits = []
    upper_3_limits = []
    lower_3_limits = []
    center_line = []
    colors2 = []

    for i in range(len(steps)):
        current_step = steps[i]
        
        if current_step < burn_in + window_size:
            upper_2_limits.append(np.nan)
            lower_2_limits.append(np.nan)
            upper_3_limits.append(np.nan)
            lower_3_limits.append(np.nan)
            center_line.append(np.nan)
            colors2.append("#4462BD")
        else:
            window_start_step = max(burn_in, current_step - window_size)
            
            window_values = []
            for j in range(len(steps)):
                if steps[j] >= window_start_step and steps[j] < current_step and steps[j] >= burn_in:
                    window_values.append(iso_values[j])
            
            if len(window_values) >= 2:
                window_mean = np.mean(window_values)
                window_std = np.std(window_values)
                
                u2 = window_mean + 2 * window_std
                l2 = window_mean - 2 * window_std
                u3 = window_mean + 3 * window_std
                l3 = window_mean - 3 * window_std
                
                upper_2_limits.append(u2)
                lower_2_limits.append(l2)
                upper_3_limits.append(u3)
                lower_3_limits.append(l3)
                center_line.append(window_mean)
                
                current_val = iso_values[i]
                if not (l3 <= current_val <= u3):
                    colors2.append("#EA5526")  # red - outside 3σ
                elif not (l2 <= current_val <= u2):
                    colors2.append("#E5B700")  # yellow - outside 2σ
                else:
                    colors2.append("#51915B")  # green - in control
            else:
                upper_2_limits.append(np.nan)
                lower_2_limits.append(np.nan)
                upper_3_limits.append(np.nan)
                lower_3_limits.append(np.nan)
                center_line.append(np.nan)
                colors2.append("#4462BD")

    valid_indices = [i for i in range(len(steps)) if not np.isnan(upper_2_limits[i])]
    if valid_indices:
        valid_steps = [steps[i] for i in valid_indices]
        valid_u2 = [upper_2_limits[i] for i in valid_indices]
        valid_l2 = [lower_2_limits[i] for i in valid_indices]
        valid_u3 = [upper_3_limits[i] for i in valid_indices]
        valid_l3 = [lower_3_limits[i] for i in valid_indices]
        valid_center = [center_line[i] for i in valid_indices]

        ax2.fill_between(valid_steps, valid_l3, valid_u3,
                        alpha=0.1, color="#EA5526", label="±3σ band")
        ax2.fill_between(valid_steps, valid_l2, valid_u2,
                        alpha=0.2, color="#E5B700", label="±2σ band")
        ax2.plot(valid_steps, valid_center, "--", color="#7f8c8d",
                linewidth=1, alpha=0.7, label="Center Line")

    for i in range(len(steps)):
        ax2.plot(steps[i], iso_values[i], "o", markersize=5, color=colors2[i])
    ax2.plot(steps, iso_values, linewidth=1, color="#bdc3c7", alpha=0.5, zorder=1)

    # Burn-in end at 100 and calibration end at 300
    ax2.axvline(x=100, color="#8064A2", linestyle=":", linewidth=2, alpha=0.7, label="Burn-in End")
    ax2.axvline(x=300, color="#51915B", linestyle="-.", linewidth=2, alpha=0.7, label="Calibration End")

    ax2.set_xlabel("Iteration", fontsize=16)
    ax2.set_ylabel("Isomirror Value", fontsize=16)
    ax2.set_title(f"Sliding Window Control Bars\n(window={window_size} iterations)", fontsize=16)
    ax2.legend(fontsize=13, loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Figure saved to {OUTPUT_PATH}")

    ooc_2_constant = sum(1 for i, c in enumerate(colors1) if c in ["#E5B700", "#EA5526"] and steps[i] >= 300)
    ooc_3_constant = sum(1 for i, c in enumerate(colors1) if c == "#EA5526" and steps[i] >= 300)
    ooc_2_window = sum(1 for c in colors2 if c in ["#E5B700", "#EA5526"])
    ooc_3_window = sum(1 for c in colors2 if c == "#EA5526")
    print(f"Constant - outside 2σ: {ooc_2_constant}, outside 3σ: {ooc_3_constant}")
    print(f"Sliding window - outside 2σ: {ooc_2_window}, outside 3σ: {ooc_3_window}")


if __name__ == "__main__":
    main()
