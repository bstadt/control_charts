#!/usr/bin/env python3
"""Figure 3: Two-panel LLM experiment comparison (fast vs slow attack)."""

import glob
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import Isomap

RESULTS_DIR = os.path.expanduser("~/root/controlcharts/experiments/results")
CALIBRATION_END = 300
WINDOW_SIZE_ITERS = 200
K = 2

PANELS = [
    {"name": "figure3-llm-d100-500", "title": "Attack Duration 100 (500 iters)"},
    {"name": "figure3-llm-d800-1100", "title": "Attack Duration 800 (1100 iters)"},
]


def find_result_dir(experiment_name):
    pattern = os.path.join(RESULTS_DIR, f"{experiment_name}_*")
    dirs = sorted([d for d in glob.glob(pattern) if os.path.isdir(d)])
    if not dirs:
        raise FileNotFoundError(f"No results found for {experiment_name}")
    return dirs[-1]


def load_isomirror(result_dir):
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


def load_accuracy(result_dir):
    """Load accuracy from snapshots directory."""
    snapshots_dir = os.path.join(result_dir, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return None, None, None, None, None

    snapshot_files = sorted(glob.glob(os.path.join(snapshots_dir, "snapshot_step_*.json")))
    if not snapshot_files:
        return None, None, None, None, None

    acc_steps = []
    all_temporal = []
    all_nontemporal = []
    all_overall = []
    agent_ids = None

    for sf in snapshot_files:
        with open(sf) as f:
            snap = json.load(f)
        step = snap.get("step", 0)
        acc_steps.append(step)

        agents = snap.get("agents", {})
        if agent_ids is None:
            agent_ids = sorted(agents.keys(), key=lambda x: int(x))

        temporal_acc = []
        nontemporal_acc = []
        overall_acc = []

        for aid in agent_ids:
            agent_data = agents.get(aid, {})
            questions = agent_data.get("questions", {})

            correct_t, total_t = 0, 0
            correct_nt, total_nt = 0, 0

            for qid, qdata in questions.items():
                is_correct = qdata.get("correct", False)
                is_temporal = qdata.get("is_temporal", False)
                if is_temporal:
                    total_t += 1
                    if is_correct:
                        correct_t += 1
                else:
                    total_nt += 1
                    if is_correct:
                        correct_nt += 1

            temporal_acc.append(correct_t / total_t if total_t > 0 else 0.0)
            nontemporal_acc.append(correct_nt / total_nt if total_nt > 0 else 0.0)
            total = total_t + total_nt
            correct = correct_t + correct_nt
            overall_acc.append(correct / total if total > 0 else 0.0)

        all_temporal.append(temporal_acc)
        all_nontemporal.append(nontemporal_acc)
        all_overall.append(overall_acc)

    acc_steps = np.array(acc_steps)
    temporal = np.array(all_temporal)
    nontemporal = np.array(all_nontemporal)
    overall = np.array(all_overall)

    return acc_steps, overall, temporal, nontemporal, agent_ids


def get_adversarial_ids(result_dir):
    """Get adversarial agent IDs from config."""
    config_files = glob.glob(os.path.join(result_dir, "config_*.yaml"))
    if not config_files:
        return set()
    import yaml
    with open(config_files[0]) as f:
        cfg = yaml.safe_load(f)
    custom = cfg.get("agents", {}).get("custom", [])
    return {str(c["id"]) for c in custom if "defection_schedule" in c or "adversarial_schedule" in c}


def compute_control_bars(steps, iso_values, k=K, calibration_end=CALIBRATION_END, window_size_iters=WINDOW_SIZE_ITERS):
    snapshot_interval = steps[1] - steps[0] if len(steps) > 1 else 1
    window_size = max(1, window_size_iters // snapshot_interval)

    cal_end_idx = 0
    for i, s in enumerate(steps):
        if s >= calibration_end:
            cal_end_idx = i
            break

    upper_limits, lower_limits, center_line, colors = [], [], [], []

    for i in range(len(steps)):
        if i < cal_end_idx:
            upper_limits.append(np.nan)
            lower_limits.append(np.nan)
            center_line.append(np.nan)
            colors.append("#3498db")
        else:
            current_val = iso_values[i]
            window_start = max(cal_end_idx, i - window_size)
            window_values = iso_values[window_start:i]

            if len(window_values) > 1:
                wmean = np.mean(window_values)
                wstd = np.std(window_values)
            else:
                wmean = window_values[0] if len(window_values) > 0 else 0
                wstd = 0

            upper = wmean + k * wstd
            lower = wmean - k * wstd
            upper_limits.append(upper)
            lower_limits.append(lower)
            center_line.append(wmean)

            if lower <= current_val <= upper:
                colors.append("#27ae60")
            else:
                colors.append("#e74c3c")

    return upper_limits, lower_limits, center_line, colors, cal_end_idx


def plot_panel(ax_iso, ax_acc, result_dir, panel_title):
    """Plot a single panel (isomirror top, accuracy bottom)."""

    # Load isomirror
    steps, iso_values = load_isomirror(result_dir)
    if steps is None:
        ax_iso.text(0.5, 0.5, "No data", ha="center", va="center")
        return

    # Compute control bars
    upper, lower, center, colors, cal_end_idx = compute_control_bars(steps, iso_values)

    # Plot control bands
    valid_idx = [i for i in range(len(steps)) if not np.isnan(upper[i])]
    if valid_idx:
        vs = [steps[i] for i in valid_idx]
        vu = [upper[i] for i in valid_idx]
        vl = [lower[i] for i in valid_idx]
        vc = [center[i] for i in valid_idx]
        ax_iso.fill_between(vs, vl, vu, alpha=0.2, color="#95a5a6", label=f"Control Band ({K}$\\sigma$)")
        ax_iso.plot(vs, vc, "--", color="#7f8c8d", linewidth=1, alpha=0.7, label="Center Line")

    # Plot points with colors
    for i in range(len(steps)):
        ax_iso.plot(steps[i], iso_values[i], "o", markersize=4, color=colors[i])
    ax_iso.plot(steps, iso_values, linewidth=1, color="#bdc3c7", alpha=0.5, zorder=1)

    # Calibration end line
    if cal_end_idx > 0 and cal_end_idx < len(steps):
        ax_iso.axvline(x=steps[cal_end_idx], color="#9b59b6", linestyle=":", linewidth=2, alpha=0.7, label=f"Calibration End ({CALIBRATION_END})")

    ax_iso.set_xlabel("Iteration", fontsize=11)
    ax_iso.set_ylabel("Isomirror Value", fontsize=11)
    ax_iso.set_title(panel_title, fontsize=13, fontweight="bold")
    ax_iso.legend(fontsize=9, loc="upper right")
    ax_iso.grid(True, alpha=0.3)

    # Load accuracy
    acc_steps, overall, temporal, nontemporal, agent_ids = load_accuracy(result_dir)
    adv_ids = get_adversarial_ids(result_dir)
    non_adv_idx = [i for i, aid in enumerate(agent_ids) if aid not in adv_ids] if agent_ids else []

    if acc_steps is not None and non_adv_idx:
        # Non-adversarial agent mean accuracy with std band
        mean_nt = np.mean(nontemporal[:, non_adv_idx], axis=1)
        std_nt = np.std(nontemporal[:, non_adv_idx], axis=1)
        mean_t = np.mean(temporal[:, non_adv_idx], axis=1)
        std_t = np.std(temporal[:, non_adv_idx], axis=1)

        ax_acc.plot(acc_steps, mean_nt, "-o", linewidth=2, markersize=4, color="#3498db", label="Non-temporal")
        ax_acc.fill_between(acc_steps, mean_nt - std_nt, mean_nt + std_nt, alpha=0.2, color="#3498db")
        ax_acc.plot(acc_steps, mean_t, "--s", linewidth=2, markersize=4, color="#e74c3c", label="Temporal")
        ax_acc.fill_between(acc_steps, mean_t - std_t, mean_t + std_t, alpha=0.2, color="#e74c3c")

        ax_acc.set_xlabel("Iteration", fontsize=11)
        ax_acc.set_ylabel("Mean Accuracy", fontsize=11)
        ax_acc.set_ylim(-0.05, 1.05)
        ax_acc.set_title(f"Non-Adversarial Agent Accuracy (n={len(non_adv_idx)})", fontsize=11)
        ax_acc.legend(fontsize=10)
        ax_acc.grid(True, alpha=0.3)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for col, panel in enumerate(PANELS):
        result_dir = find_result_dir(panel["name"])
        print(f"Panel {col}: {panel['name']} -> {result_dir}")
        plot_panel(axes[0, col], axes[1, col], result_dir, panel["title"])

    fig.suptitle("Figure 3: LLM Promptquine Propagation", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "figure3.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
