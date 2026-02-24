"""
Figure 3: Comparison of Gradual vs Sudden Adversarial Attack Detection

Layout (2 rows, 2 columns):
- (0,0): Isomirror with sliding window control bars - Sudden step
- (0,1): Isomirror with sliding window control bars - Gradual ramp
- (1,0): Non-adversarial agent accuracy - Sudden
- (1,1): Non-adversarial agent accuracy - Gradual

Uses sliding window approach with 2sigma (yellow) and 3sigma (red) control bands.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import Isomap

GRADUAL_DIR = Path("experiments/results/figure3-adversarial_2026-01-17_13-45-04")
TOOFAST_DIR = Path("experiments/results/figure3-toofast_2026-01-16_13-36-51")
OUTPUT_PATH = Path("experiments/figures/figure3.png")


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


def load_accuracy_from_snapshots(snapshots_dir: Path, adversarial_ids: set):
    meta_files = sorted(snapshots_dir.glob("snapshot_step_*_meta.json"))
    if not meta_files:
        return None, None, None
    
    steps = []
    temporal_acc_list = []
    nontemporal_acc_list = []
    
    for meta_file in meta_files:
        with open(meta_file) as f:
            meta = json.load(f)
        
        step = meta.get("step", 0)
        steps.append(step)
        
        questions = meta.get("questions", [])
        responses = meta.get("responses", [])
        temporal_mask = meta.get("temporal_mask", [])
        temporal_values = meta.get("temporal_values", {})
        agent_ids = meta.get("agent_ids", [])
        
        agent_temporal = []
        agent_nontemporal = []
        
        for agent_idx, agent_id in enumerate(agent_ids):
            if agent_id in adversarial_ids:
                continue
            
            agent_responses = responses[agent_idx] if agent_idx < len(responses) else []
            
            temporal_correct = 0
            temporal_total = 0
            nontemporal_correct = 0
            nontemporal_total = 0
            
            for q_idx, (question, response) in enumerate(zip(questions, agent_responses)):
                is_temporal = temporal_mask[q_idx] if q_idx < len(temporal_mask) else False
                
                if is_temporal:
                    temporal_total += 1
                    correct_val = temporal_values.get(question)
                    if correct_val is not None and str(correct_val) in response:
                        temporal_correct += 1
                else:
                    nontemporal_total += 1
                    if "I lost the game" not in response and "I don't know" not in response:
                        nontemporal_correct += 1
            
            if temporal_total > 0:
                agent_temporal.append(temporal_correct / temporal_total)
            if nontemporal_total > 0:
                agent_nontemporal.append(nontemporal_correct / nontemporal_total)
        
        temporal_acc_list.append(agent_temporal)
        nontemporal_acc_list.append(agent_nontemporal)
    
    return np.array(steps), np.array(temporal_acc_list), np.array(nontemporal_acc_list)


def plot_isomirror_sliding_window(ax, iso_values, steps, burn_in=100, window_size=200, title=""):
    """Plot isomirror with sliding window control bars.
    
    Uses 2sigma (yellow) and 3sigma (red) control bands.
    Control limits only shown after window_size iterations past burn_in.
    """
    
    upper_2_limits = []
    lower_2_limits = []
    upper_3_limits = []
    lower_3_limits = []
    center_line = []
    colors = []

    for i in range(len(steps)):
        current_step = steps[i]
        
        if current_step < burn_in + window_size:
            # Before we have a full window past burn-in
            upper_2_limits.append(np.nan)
            lower_2_limits.append(np.nan)
            upper_3_limits.append(np.nan)
            lower_3_limits.append(np.nan)
            center_line.append(np.nan)
            colors.append("#4462BD")  # blue - warmup
        else:
            window_start_step = max(burn_in, current_step - window_size)
            
            # Collect values in the window (excluding current point - no lookahead)
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
                    colors.append("#EA5526")  # red - outside 3sigma
                elif not (l2 <= current_val <= u2):
                    colors.append("#E5B700")  # yellow - outside 2sigma
                else:
                    colors.append("#51915B")  # green - in control
            else:
                upper_2_limits.append(np.nan)
                lower_2_limits.append(np.nan)
                upper_3_limits.append(np.nan)
                lower_3_limits.append(np.nan)
                center_line.append(np.nan)
                colors.append("#4462BD")
    
    # Plot control bands
    valid_indices = [i for i in range(len(steps)) if not np.isnan(upper_2_limits[i])]
    if valid_indices:
        valid_steps = [steps[i] for i in valid_indices]
        valid_u2 = [upper_2_limits[i] for i in valid_indices]
        valid_l2 = [lower_2_limits[i] for i in valid_indices]
        valid_u3 = [upper_3_limits[i] for i in valid_indices]
        valid_l3 = [lower_3_limits[i] for i in valid_indices]
        valid_center = [center_line[i] for i in valid_indices]
        
        ax.fill_between(valid_steps, valid_l3, valid_u3,
                       alpha=0.1, color="#EA5526", label="+/-3sigma band")
        ax.fill_between(valid_steps, valid_l2, valid_u2,
                       alpha=0.2, color="#E5B700", label="+/-2sigma band")
        ax.plot(valid_steps, valid_center, "--", color="#7f8c8d",
               linewidth=1, alpha=0.7, label="Center Line")
    
    # Plot points
    for i in range(len(steps)):
        ax.plot(steps[i], iso_values[i], "o", markersize=5, color=colors[i])
    ax.plot(steps, iso_values, linewidth=1, color="#bdc3c7", alpha=0.5, zorder=1)
    
    # Vertical lines
    ax.axvline(x=burn_in, color="#8064A2", linestyle=":", linewidth=2, alpha=0.7, label="Burn-in End")
    ax.axvline(x=burn_in + window_size, color="#51915B", linestyle="-.", linewidth=2, alpha=0.7, label="Calibration End")
    
    ax.set_xlabel("Iteration", fontsize=16)
    ax.set_ylabel("Isomirror Value", fontsize=16)
    ax.set_title(title, fontsize=17)
    ax.legend(fontsize=12, loc="upper left", bbox_to_anchor=(0.15, 0.98), framealpha=1)
    ax.grid(True, alpha=0.3)
    
    ooc_2 = sum(1 for c in colors if c in ["#E5B700", "#EA5526"])
    ooc_3 = sum(1 for c in colors if c == "#EA5526")
    return ooc_2, ooc_3


def plot_nonadversarial_accuracy(ax, steps, temporal_acc, nontemporal_acc, title=""):
    has_data = False
    
    if temporal_acc is not None and len(temporal_acc) > 0 and temporal_acc.size > 0:
        mean_temporal = np.mean(temporal_acc, axis=1)
        std_temporal = np.std(temporal_acc, axis=1)
        ax.plot(steps, mean_temporal, marker="o", linewidth=2, markersize=4,
               color="#EA5526", label="Temporal")
        ax.fill_between(steps, mean_temporal - std_temporal, mean_temporal + std_temporal,
                       alpha=0.2, color="#EA5526")
        has_data = True
    
    if nontemporal_acc is not None and len(nontemporal_acc) > 0 and nontemporal_acc.size > 0:
        mean_nontemporal = np.mean(nontemporal_acc, axis=1)
        std_nontemporal = np.std(nontemporal_acc, axis=1)
        ax.plot(steps, mean_nontemporal, marker="s", linewidth=2, markersize=4,
               color="#4462BD", label="Non-temporal")
        ax.fill_between(steps, mean_nontemporal - std_nontemporal, mean_nontemporal + std_nontemporal,
                       alpha=0.2, color="#4462BD")
        has_data = True
    
    ax.set_xlabel("Iteration", fontsize=16)
    ax.set_ylabel("Mean Accuracy", fontsize=16)
    ax.set_title(title, fontsize=17)
    ax.set_ylim(-0.05, 1.05)
    if has_data:
        ax.legend(fontsize=13, loc="lower left", framealpha=1)
    ax.grid(True, alpha=0.3)


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    adversarial_ids = {0, 1}
    
    print("Loading gradual ramp data...")
    gradual_emb, gradual_steps = load_tdkps_embeddings(GRADUAL_DIR, "figure3-adversarial")
    gradual_iso = compute_isomirror(gradual_emb, gradual_steps)
    gradual_acc_steps, gradual_temporal, gradual_nontemporal = load_accuracy_from_snapshots(
        GRADUAL_DIR / "snapshots", adversarial_ids)
    print(f"  Gradual accuracy shape: temporal={gradual_temporal.shape}, nontemporal={gradual_nontemporal.shape}")
    
    print("Loading toofast data...")
    toofast_emb, toofast_steps = load_tdkps_embeddings(TOOFAST_DIR, "figure3-toofast")
    toofast_iso = compute_isomirror(toofast_emb, toofast_steps)
    toofast_acc_steps, toofast_temporal, toofast_nontemporal = load_accuracy_from_snapshots(
        TOOFAST_DIR / "snapshots", adversarial_ids)
    print(f"  Toofast accuracy shape: temporal={toofast_temporal.shape}, nontemporal={toofast_nontemporal.shape}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    
    # Top row: Isomirrors (Sudden left, Gradual right)
    ooc_2_sudden, ooc_3_sudden = plot_isomirror_sliding_window(
        axes[0, 0], toofast_iso, toofast_steps,
        burn_in=100, window_size=200,
        title="Sudden Step (0% to 50% at iteration 400)"
    )
    
    ooc_2_gradual, ooc_3_gradual = plot_isomirror_sliding_window(
        axes[0, 1], gradual_iso, gradual_steps,
        burn_in=100, window_size=200,
        title="Gradual Ramp (0% to 50% over iterations 400-1500)"
    )
    
    # Bottom row: Non-adversarial accuracy
    plot_nonadversarial_accuracy(
        axes[1, 0], toofast_acc_steps, toofast_temporal, toofast_nontemporal,
        title="Non-Adversarial Agents Only (n=3) - Sudden"
    )
    
    plot_nonadversarial_accuracy(
        axes[1, 1], gradual_acc_steps, gradual_temporal, gradual_nontemporal,
        title="Non-Adversarial Agents Only (n=3) - Gradual"
    )
    
    plt.tight_layout()
    
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Figure saved to {OUTPUT_PATH}")
    print(f"Sudden - outside 2sigma: {ooc_2_sudden}, outside 3sigma: {ooc_3_sudden}")
    print(f"Gradual - outside 2sigma: {ooc_2_gradual}, outside 3sigma: {ooc_3_gradual}")


if __name__ == "__main__":
    main()
