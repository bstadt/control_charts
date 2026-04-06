"""
Figure 1: Static vs Dynamic Memory in Swarm Systems

Layout (1 row, 3 columns):
- (0,0): Two stacked subplots - TDKPS static (top) and TDKPS dynamic (bottom)
- (0,1): Mean accuracy of static vs dynamic swarms over time
- (0,2): Normalized isomirror comparison (both systems overlaid)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from pathlib import Path
from sklearn.manifold import Isomap

# Paths to experiment results
STATIC_DIR = Path("experiments/results/figure1-static_2026-01-17_10-06-33")
DYNAMIC_DIR = Path("experiments/results/figure1-dynamic_2026-01-17_10-06-33")
OUTPUT_PATH = Path("experiments/figures/figure1.png")


def load_tdkps_embeddings(results_dir: Path, experiment_name: str):
    """Load TDKPS embeddings from npz file."""
    path = results_dir / f"{experiment_name}_tdkps_embeddings.npz"
    data = np.load(path)
    return data["embedding_matrix"], data["steps"]


def load_accuracy_from_snapshots(snapshots_dir: Path):
    """Load accuracy data from snapshot metadata files."""
    index_path = snapshots_dir / "snapshots_index.json"
    with open(index_path) as f:
        snapshots_index = json.load(f)

    temporal_accuracy_list = []
    nontemporal_accuracy_list = []
    steps = []

    for snapshot_info in snapshots_index:
        step = snapshot_info["step"]
        npz_path = Path(snapshot_info["path"])
        meta_path = npz_path.parent / f"{npz_path.stem}_meta.json"

        with open(meta_path) as f:
            meta = json.load(f)

        responses = meta["responses"]
        questions = meta.get("questions", [])
        temporal_mask = meta.get("temporal_mask", None)
        temporal_values = meta.get("temporal_values", {})

        n_agents = len(responses)
        agent_temporal_accuracies = []
        agent_nontemporal_accuracies = []

        for agent_responses in responses:
            if temporal_mask:
                temporal_correct = 0
                temporal_total = 0
                nontemporal_correct = 0
                nontemporal_total = 0

                for q_idx, r in enumerate(agent_responses):
                    if temporal_mask[q_idx]:
                        temporal_total += 1
                        question = questions[q_idx] if q_idx < len(questions) else None
                        expected_value = temporal_values.get(question, 0) if question else 0
                        # Check if response contains expected value
                        import re
                        numbers = re.findall(r'\d+', r.strip())
                        if numbers and any(int(n) == expected_value for n in numbers):
                            temporal_correct += 1
                    else:
                        nontemporal_total += 1
                        # Check if valid answer (not "I don't know")
                        r_lower = r.lower().strip()
                        if not any(p in r_lower for p in ["i don't know", "i dont know", "i do not know", "i lost the game"]):
                            nontemporal_correct += 1

                temporal_acc = temporal_correct / temporal_total if temporal_total > 0 else 0
                nontemporal_acc = nontemporal_correct / nontemporal_total if nontemporal_total > 0 else 0
                agent_temporal_accuracies.append(temporal_acc)
                agent_nontemporal_accuracies.append(nontemporal_acc)

        temporal_accuracy_list.append(agent_temporal_accuracies)
        nontemporal_accuracy_list.append(agent_nontemporal_accuracies)
        steps.append(step)

    return np.array(temporal_accuracy_list), np.array(nontemporal_accuracy_list), np.array(steps)


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
    static_embedding, static_steps = load_tdkps_embeddings(STATIC_DIR, "figure1-static")
    dynamic_embedding, dynamic_steps = load_tdkps_embeddings(DYNAMIC_DIR, "figure1-dynamic")

    print("Loading accuracy data...")
    static_temporal_acc, static_nontemporal_acc, static_acc_steps = load_accuracy_from_snapshots(STATIC_DIR / "snapshots")
    dynamic_temporal_acc, dynamic_nontemporal_acc, dynamic_acc_steps = load_accuracy_from_snapshots(DYNAMIC_DIR / "snapshots")

    print("Computing isomirror...")
    static_isomirror = compute_isomirror(static_embedding, static_steps)
    dynamic_isomirror = compute_isomirror(dynamic_embedding, dynamic_steps)

    # Normalize isomirrors to [0, 1]
    static_isomirror_norm = (static_isomirror - static_isomirror.min()) / (static_isomirror.max() - static_isomirror.min())
    dynamic_isomirror_norm = (dynamic_isomirror - dynamic_isomirror.min()) / (dynamic_isomirror.max() - dynamic_isomirror.min())

    # Create figure with custom grid (1 row, 3 columns)
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)

    # ===== (0,0): Two stacked TDKPS subplots =====
    gs_tdkps = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 0], hspace=0.4)

    # Top: TDKPS static
    ax_tdkps_static = fig.add_subplot(gs_tdkps[0])
    n_agents = static_embedding.shape[1]
    for agent_id in range(n_agents):
        agent_values = static_embedding[:, agent_id, 0]
        ax_tdkps_static.plot(static_steps, agent_values, marker='o', markersize=3, label=f'Agent {agent_id}')
    ax_tdkps_static.set_ylabel('TDKPS Position')
    ax_tdkps_static.set_title('Static Memory: Agent Positions')
    ax_tdkps_static.legend(fontsize=7, ncol=3, loc='lower right')
    ax_tdkps_static.grid(True, alpha=0.3)
    ax_tdkps_static.tick_params(labelbottom=False)  # Hide x-axis labels

    # Bottom: TDKPS dynamic
    ax_tdkps_dynamic = fig.add_subplot(gs_tdkps[1])
    for agent_id in range(n_agents):
        agent_values = dynamic_embedding[:, agent_id, 0]
        ax_tdkps_dynamic.plot(dynamic_steps, agent_values, marker='o', markersize=3, label=f'Agent {agent_id}')
    ax_tdkps_dynamic.set_xlabel('Iteration')
    ax_tdkps_dynamic.set_ylabel('TDKPS Position')
    ax_tdkps_dynamic.set_title('Dynamic Memory: Agent Positions')
    ax_tdkps_dynamic.legend(fontsize=7, ncol=3, loc='lower right')
    ax_tdkps_dynamic.grid(True, alpha=0.3)

    # ===== (0,1): Mean accuracy comparison =====
    ax_acc = fig.add_subplot(gs[0, 1])

    # Compute mean accuracy across agents
    static_mean_temporal = np.mean(static_temporal_acc, axis=1)
    dynamic_mean_temporal = np.mean(dynamic_temporal_acc, axis=1)
    static_mean_nontemporal = np.mean(static_nontemporal_acc, axis=1)
    dynamic_mean_nontemporal = np.mean(dynamic_nontemporal_acc, axis=1)

    # Temporal accuracy (solid lines)
    ax_acc.plot(static_acc_steps, static_mean_temporal, marker='o', markersize=4,
                linewidth=2, color='#EA5526', label='Static (Temporal)')
    ax_acc.plot(dynamic_acc_steps, dynamic_mean_temporal, marker='o', markersize=4,
                linewidth=2, color='#4462BD', label='Dynamic (Temporal)')

    # Non-temporal accuracy (dashed lines)
    ax_acc.plot(static_acc_steps, static_mean_nontemporal, marker='s', markersize=4,
                linewidth=2, linestyle='--', color='#EA5526', alpha=0.7, label='Static (Non-temporal)')
    ax_acc.plot(dynamic_acc_steps, dynamic_mean_nontemporal, marker='s', markersize=4,
                linewidth=2, linestyle='--', color='#4462BD', alpha=0.7, label='Dynamic (Non-temporal)')

    ax_acc.set_xlabel('Iteration')
    ax_acc.set_ylabel('Mean Accuracy')
    ax_acc.set_title('Accuracy: Static vs Dynamic')
    ax_acc.set_ylim(-0.05, 1.05)
    ax_acc.legend(fontsize=8, loc='lower right')
    ax_acc.grid(True, alpha=0.3)

    # ===== (0,2): Combined normalized isomirror =====
    ax_iso = fig.add_subplot(gs[0, 2])
    ax_iso.plot(static_steps, static_isomirror_norm, marker='o', linewidth=2,
                markersize=4, color='#EA5526', label='Static')
    ax_iso.plot(dynamic_steps, dynamic_isomirror_norm, marker='o', linewidth=2,
                markersize=4, color='#4462BD', label='Dynamic')
    ax_iso.set_xlabel('Iteration')
    ax_iso.set_ylabel('Isomirror')
    ax_iso.set_title('Isomirror: Static vs Dynamic')
    ax_iso.set_ylim(-0.05, 1.05)
    ax_iso.legend(fontsize=8, loc='upper right')
    ax_iso.grid(True, alpha=0.3)

    # Save
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Figure saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
