"""On-simulation-end hook for TDKPS analysis of temporal embeddings."""

import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from .agent import Agent
    from .network import Network

logger = logging.getLogger(__name__)


def run_tdkps_analysis(
    snapshots_dir: Path,
    output_dir: Path,
    experiment_name: str,
) -> None:
    """
    Run TDKPS analysis on temporal kernel snapshots.

    Reads embedding matrices from snapshots, fits TDKPSEstimator,
    and plots agent embeddings over time.

    Parameters
    ----------
    snapshots_dir : Path
        Directory containing snapshot files
    output_dir : Path
        Directory to save output plots
    experiment_name : str
        Name of the experiment for labeling outputs
    """
    # Add maps package to path
    maps_path = Path(__file__).parent.parent.parent / "maps"
    if str(maps_path) not in sys.path:
        sys.path.insert(0, str(maps_path))

    from maps.gmds import TDKPSEstimator

    # Load snapshots index
    index_path = snapshots_dir / "snapshots_index.json"
    if not index_path.exists():
        logger.warning(f"No snapshots index found at {index_path}, skipping TDKPS analysis")
        return

    with open(index_path) as f:
        snapshots_index = json.load(f)

    if len(snapshots_index) < 2:
        logger.warning("Need at least 2 snapshots for TDKPS analysis")
        return

    logger.info(f"Loading {len(snapshots_index)} snapshots for TDKPS analysis...")

    # Load all embedding matrices
    # Each snapshot has shape [n_agents, n_questions, embedding_dim]
    embeddings_list = []
    steps = []

    for snapshot_info in snapshots_index:
        step = snapshot_info["step"]
        # Paths in index are relative to CWD, not snapshots_dir
        npz_path = Path(snapshot_info["path"])

        data = np.load(npz_path)
        embeddings = data["embeddings"]  # [agents, questions, embedding_dim]
        embeddings_list.append(embeddings)
        steps.append(step)

    # Stack into [n_timesteps, n_agents, n_questions, embedding_dim]
    embeddings_4d = np.stack(embeddings_list, axis=0)
    n_timesteps, n_agents, n_questions, embedding_dim = embeddings_4d.shape

    logger.info(f"Loaded embeddings: {embeddings_4d.shape} "
                f"(timesteps={n_timesteps}, agents={n_agents}, "
                f"questions={n_questions}, embedding_dim={embedding_dim})")

    # Add n_reps dimension to make it 5D as required by TDKPSEstimator
    # Shape: [n_timesteps, n_agents, n_questions, n_features, n_reps]
    X = embeddings_4d[:, :, :, :, np.newaxis]
    logger.info(f"X tensor shape for TDKPS: {X.shape}")

    # Fit TDKPSEstimator
    logger.info("Fitting TDKPSEstimator...")
    estimator = TDKPSEstimator(n_components=1)
    estimator.fit(X)

    # Access embedding_matrix_: [n_timesteps, n_agents, n_components]
    embedding_matrix = estimator.embedding_matrix_
    logger.info(f"Embedding matrix shape: {embedding_matrix.shape}")

    # Plot agent values over time
    # embedding_matrix has shape [n_timesteps, n_agents, n_components]
    # With n_components=1, we plot the single component value for each agent
    plt.figure(figsize=(10, 6))

    for agent_id in range(n_agents):
        agent_values = embedding_matrix[:, agent_id, 0]  # [n_timesteps]
        plt.plot(steps, agent_values, marker='o', label=f'Agent {agent_id}')

    plt.xlabel('Iteration')
    plt.ylabel('TDKPS Embedding Value')
    plt.title(f'Agent Embeddings Over Time ({experiment_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"{experiment_name}_tdkps.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"TDKPS plot saved to {plot_path}")

    # Also save the embedding matrix for further analysis
    np.savez(
        output_dir / f"{experiment_name}_tdkps_embeddings.npz",
        embedding_matrix=embedding_matrix,
        steps=np.array(steps),
        n_agents=n_agents,
        n_components=estimator.n_components_
    )
    logger.info(f"TDKPS embeddings saved to {output_dir / f'{experiment_name}_tdkps_embeddings.npz'}")


def plot_perspective_variance(
    snapshots_dir: Path,
    output_dir: Path,
    experiment_name: str,
) -> None:
    """
    Plot perspective variance over time.

    Computes the variance of agent TDKPS embedding positions at each timestep,
    showing how agent perspectives diverge or converge over time in the
    learned temporal kernel space.

    Parameters
    ----------
    snapshots_dir : Path
        Directory containing snapshot files
    output_dir : Path
        Directory to save output plots
    experiment_name : str
        Name of the experiment for labeling outputs
    """
    # Load TDKPS embeddings (must run after run_tdkps_analysis)
    tdkps_path = output_dir / f"{experiment_name}_tdkps_embeddings.npz"
    if not tdkps_path.exists():
        logger.warning(f"No TDKPS embeddings found at {tdkps_path}, skipping perspective variance plot")
        return

    data = np.load(tdkps_path)
    embedding_matrix = data["embedding_matrix"]  # [n_timesteps, n_agents, n_components]
    steps = data["steps"]

    logger.info(f"Loaded TDKPS embeddings: {embedding_matrix.shape}")

    # Compute variance across agents at each timestep
    # embedding_matrix shape: [n_timesteps, n_agents, n_components]
    # Variance across agents (axis=1), then mean across components if multiple
    variances = np.var(embedding_matrix, axis=1)  # [n_timesteps, n_components]
    if variances.ndim > 1:
        variances = np.mean(variances, axis=1)  # [n_timesteps]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, variances, marker='o', linewidth=2, markersize=6, color='#2ecc71')
    plt.fill_between(steps, variances, alpha=0.3, color='#2ecc71')

    plt.xlabel('Iteration')
    plt.ylabel('TDKPS Position Variance (across agents)')
    plt.title(f'Perspective Variance Over Time - {experiment_name}')
    plt.grid(True, alpha=0.3)

    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"{experiment_name}_perspective_variance.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Perspective variance plot saved to {plot_path}")

    # Also save the variance data
    np.savez(
        output_dir / f"{experiment_name}_perspective_variance.npz",
        steps=steps,
        variances=variances,
    )
    logger.info(f"Perspective variance data saved to {output_dir / f'{experiment_name}_perspective_variance.npz'}")


def _is_wrong_answer(response: str) -> bool:
    """Check if a response is a non-answer (wrong)."""
    response_lower = response.lower().strip()
    wrong_patterns = [
        "i don't know",
        "i dont know",
        "i do not know",
        "i lost the game",
    ]
    for pattern in wrong_patterns:
        if pattern in response_lower:
            return True
    return False


def _load_accuracy_from_snapshots(snapshots_dir: Path) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Load response accuracy from snapshot metadata files.

    Returns
    -------
    accuracy_matrix : np.ndarray
        Shape [n_timesteps, n_agents] with accuracy values (0-1)
    steps : np.ndarray
        Array of step numbers
    agent_ids : list
        List of agent IDs
    """
    index_path = snapshots_dir / "snapshots_index.json"
    if not index_path.exists():
        return None, None, None

    with open(index_path) as f:
        snapshots_index = json.load(f)

    accuracy_list = []
    steps = []
    agent_ids = None

    for snapshot_info in snapshots_index:
        step = snapshot_info["step"]
        npz_path = Path(snapshot_info["path"])
        # Construct meta path: snapshot_step_0000.npz -> snapshot_step_0000_meta.json
        meta_path = npz_path.parent / f"{npz_path.stem}_meta.json"

        if not meta_path.exists():
            logger.warning(f"Meta file not found: {meta_path}")
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        responses = meta["responses"]  # [n_agents][n_questions]
        if agent_ids is None:
            agent_ids = meta["agent_ids"]

        n_agents = len(responses)
        n_questions = len(responses[0]) if responses else 0

        # Compute accuracy per agent
        agent_accuracies = []
        for agent_responses in responses:
            correct = sum(1 for r in agent_responses if not _is_wrong_answer(r))
            accuracy = correct / n_questions if n_questions > 0 else 0
            agent_accuracies.append(accuracy)

        accuracy_list.append(agent_accuracies)
        steps.append(step)

    if not accuracy_list:
        return None, None, None

    return np.array(accuracy_list), np.array(steps), agent_ids


def plot_combined_analysis(
    output_dir: Path,
    experiment_name: str,
    snapshots_dir: Path = None,
) -> None:
    """
    Create a combined figure with TDKPS positions, perspective variance, distance matrix, isomirror, and accuracy.

    Parameters
    ----------
    output_dir : Path
        Directory containing TDKPS embeddings (and to save output)
    experiment_name : str
        Name of the experiment for labeling outputs
    snapshots_dir : Path, optional
        Directory containing snapshot files (for accuracy data)
    """
    from sklearn.manifold import Isomap

    # Load TDKPS embeddings
    tdkps_path = output_dir / f"{experiment_name}_tdkps_embeddings.npz"
    if not tdkps_path.exists():
        logger.warning(f"No TDKPS embeddings found at {tdkps_path}, skipping combined analysis plot")
        return

    data = np.load(tdkps_path)
    embedding_matrix = data["embedding_matrix"]  # [n_timesteps, n_agents, n_components]
    steps = data["steps"]
    n_agents = embedding_matrix.shape[1]
    n_timesteps = embedding_matrix.shape[0]

    logger.info(f"Loaded TDKPS embeddings: {embedding_matrix.shape}")

    # Compute perspective variance
    variances = np.var(embedding_matrix, axis=1)  # [n_timesteps, n_components]
    if variances.ndim > 1:
        variances = np.mean(variances, axis=1)  # [n_timesteps]

    # Compute distance matrix for isomirror
    flat_embeddings = embedding_matrix.reshape(n_timesteps, -1)
    distance_matrix = np.zeros((n_timesteps, n_timesteps))
    for i in range(n_timesteps):
        for j in range(n_timesteps):
            distance_matrix[i, j] = np.linalg.norm(flat_embeddings[i] - flat_embeddings[j])

    # Run Isomap for 1D embedding
    isomap = Isomap(n_components=1, metric='precomputed', n_neighbors=min(5, n_timesteps - 1))
    isomap_embedding = isomap.fit_transform(distance_matrix)

    # Load accuracy data if snapshots_dir provided
    accuracy_matrix = None
    if snapshots_dir is not None:
        accuracy_matrix, acc_steps, agent_ids = _load_accuracy_from_snapshots(snapshots_dir)
        if accuracy_matrix is not None:
            logger.info(f"Loaded accuracy data: {accuracy_matrix.shape}")

    # Create combined figure with 2x3 subplots (or 2x2 if no accuracy data)
    if accuracy_matrix is not None:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: TDKPS positions over time
    ax1 = axes[0, 0]
    for agent_id in range(n_agents):
        agent_values = embedding_matrix[:, agent_id, 0]  # [n_timesteps]
        ax1.plot(steps, agent_values, marker='o', label=f'Agent {agent_id}')

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('TDKPS Embedding Value')
    ax1.set_title('Agent Positions Over Time')
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Top-middle: Perspective variance over time
    ax2 = axes[0, 1]
    ax2.plot(steps, variances, marker='o', linewidth=2, markersize=6, color='#2ecc71')
    ax2.fill_between(steps, variances, alpha=0.3, color='#2ecc71')

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('TDKPS Position Variance')
    ax2.set_title('Perspective Variance Over Time')
    ax2.grid(True, alpha=0.3)

    # Bottom-left: Distance matrix heatmap
    ax3 = axes[1, 0]
    im = ax3.imshow(distance_matrix, cmap='viridis', aspect='auto')
    ax3.set_xlabel('Timestep j')
    ax3.set_ylabel('Timestep i')
    ax3.set_title('TDKPS Distance Matrix')
    ax3.set_xticks(np.arange(0, n_timesteps, max(1, n_timesteps // 10)))
    ax3.set_yticks(np.arange(0, n_timesteps, max(1, n_timesteps // 10)))
    ax3.set_xticklabels([str(steps[i]) for i in range(0, n_timesteps, max(1, n_timesteps // 10))])
    ax3.set_yticklabels([str(steps[i]) for i in range(0, n_timesteps, max(1, n_timesteps // 10))])
    plt.colorbar(im, ax=ax3, label='||TDKPS_i - TDKPS_j||_2')

    # Bottom-middle: Isomap 1D embedding over time
    ax4 = axes[1, 1]
    ax4.plot(steps, isomap_embedding[:, 0], marker='o', linewidth=2, markersize=4, color='#e74c3c')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Isomap 1D Embedding')
    ax4.set_title('Isomirror: 1D Isomap of Distance Matrix')
    ax4.grid(True, alpha=0.3)

    # Right column: Accuracy over time (if available)
    if accuracy_matrix is not None:
        # Top-right: Per-agent accuracy
        ax5 = axes[0, 2]
        for agent_id in range(accuracy_matrix.shape[1]):
            ax5.plot(acc_steps, accuracy_matrix[:, agent_id], marker='o', label=f'Agent {agent_id}')
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Accuracy')
        ax5.set_title('Agent Response Accuracy Over Time')
        ax5.set_ylim(-0.05, 1.05)
        ax5.legend(fontsize=8, ncol=2)
        ax5.grid(True, alpha=0.3)

        # Bottom-right: Mean accuracy with std band
        ax6 = axes[1, 2]
        mean_acc = np.mean(accuracy_matrix, axis=1)
        std_acc = np.std(accuracy_matrix, axis=1)
        ax6.plot(acc_steps, mean_acc, marker='o', linewidth=2, markersize=6, color='#3498db')
        ax6.fill_between(acc_steps, mean_acc - std_acc, mean_acc + std_acc, alpha=0.3, color='#3498db')
        ax6.set_xlabel('Iteration')
        ax6.set_ylabel('Mean Accuracy')
        ax6.set_title('Mean Response Accuracy (± std)')
        ax6.set_ylim(-0.05, 1.05)
        ax6.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(f'TDKPS Analysis - {experiment_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save combined figure
    plot_path = output_dir / f"{experiment_name}_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Combined analysis plot saved to {plot_path}")


def plot_isomirror(
    output_dir: Path,
    experiment_name: str,
) -> None:
    """
    Generate an isomirror visualization using Isomap on TDKPS distance matrix.

    Creates an iterations x iterations distance matrix where cell (i,j) is the
    2-norm (Frobenius norm) of the difference between TDKPS positions at timestep
    i and timestep j. This matrix is then embedded via Isomap to get a 1D
    representation, which is visualized alongside the distance matrix.

    Parameters
    ----------
    output_dir : Path
        Directory containing TDKPS embeddings (and to save output)
    experiment_name : str
        Name of the experiment for labeling outputs
    """
    from sklearn.manifold import Isomap

    # Load TDKPS embeddings
    tdkps_path = output_dir / f"{experiment_name}_tdkps_embeddings.npz"
    if not tdkps_path.exists():
        logger.warning(f"No TDKPS embeddings found at {tdkps_path}, skipping isomirror plot")
        return

    data = np.load(tdkps_path)
    embedding_matrix = data["embedding_matrix"]  # [n_timesteps, n_agents, n_components]
    steps = data["steps"]
    n_timesteps = embedding_matrix.shape[0]

    logger.info(f"Loaded TDKPS embeddings for isomirror: {embedding_matrix.shape}")

    # Flatten agent positions at each timestep: [n_timesteps, n_agents * n_components]
    flat_embeddings = embedding_matrix.reshape(n_timesteps, -1)

    # Create iterations x iterations distance matrix
    # D[i,j] = ||flat_embeddings[i] - flat_embeddings[j]||_2
    distance_matrix = np.zeros((n_timesteps, n_timesteps))
    for i in range(n_timesteps):
        for j in range(n_timesteps):
            distance_matrix[i, j] = np.linalg.norm(flat_embeddings[i] - flat_embeddings[j])

    logger.info(f"Distance matrix shape: {distance_matrix.shape}")

    # Run Isomap for 1D embedding
    # Use precomputed metric since we have a distance matrix
    isomap = Isomap(n_components=1, metric='precomputed', n_neighbors=min(5, n_timesteps - 1))
    isomap_embedding = isomap.fit_transform(distance_matrix)  # [n_timesteps, 1]

    logger.info(f"Isomap embedding shape: {isomap_embedding.shape}")

    # Create visualization with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: Distance matrix heatmap
    ax1 = axes[0]
    im = ax1.imshow(distance_matrix, cmap='viridis', aspect='auto')
    ax1.set_xlabel('Timestep j')
    ax1.set_ylabel('Timestep i')
    ax1.set_title('TDKPS Distance Matrix')
    ax1.set_xticks(np.arange(0, n_timesteps, max(1, n_timesteps // 10)))
    ax1.set_yticks(np.arange(0, n_timesteps, max(1, n_timesteps // 10)))
    ax1.set_xticklabels([str(steps[i]) for i in range(0, n_timesteps, max(1, n_timesteps // 10))])
    ax1.set_yticklabels([str(steps[i]) for i in range(0, n_timesteps, max(1, n_timesteps // 10))])
    plt.colorbar(im, ax=ax1, label='||TDKPS_i - TDKPS_j||_2')

    # Middle: Isomap 1D embedding over time
    ax2 = axes[1]
    ax2.plot(steps, isomap_embedding[:, 0], marker='o', linewidth=2, markersize=4, color='#e74c3c')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Isomap 1D Embedding')
    ax2.set_title('Isomirror: 1D Isomap of Distance Matrix')
    ax2.grid(True, alpha=0.3)

    # Right: Isomap embedding as color-mapped scatter (to show trajectory)
    ax3 = axes[2]
    colors = np.arange(n_timesteps)
    scatter = ax3.scatter(
        np.zeros(n_timesteps),
        isomap_embedding[:, 0],
        c=colors,
        cmap='plasma',
        s=50,
        alpha=0.8
    )
    # Connect points with lines
    for i in range(n_timesteps - 1):
        ax3.plot(
            [0, 0],
            [isomap_embedding[i, 0], isomap_embedding[i + 1, 0]],
            color=plt.cm.plasma(i / n_timesteps),
            alpha=0.5,
            linewidth=1
        )
    ax3.set_xlim(-0.5, 0.5)
    ax3.set_ylabel('Isomap 1D Position')
    ax3.set_title('Temporal Trajectory on 1D Manifold')
    ax3.set_xticks([])
    plt.colorbar(scatter, ax=ax3, label='Timestep')

    # Overall title
    fig.suptitle(f'Isomirror Analysis - {experiment_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save figure
    plot_path = output_dir / f"{experiment_name}_isomirror.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Isomirror plot saved to {plot_path}")

    # Save the distance matrix and isomap embedding for further analysis
    np.savez(
        output_dir / f"{experiment_name}_isomirror.npz",
        distance_matrix=distance_matrix,
        isomap_embedding=isomap_embedding,
        steps=steps,
    )
    logger.info(f"Isomirror data saved to {output_dir / f'{experiment_name}_isomirror.npz'}")


class SimulationEndHook:
    """Hook that runs TDKPS analysis after simulation completes."""

    def __init__(
        self,
        snapshots_dir: Path,
        output_dir: Path,
        experiment_name: str,
    ):
        self.snapshots_dir = Path(snapshots_dir)
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name

    def __call__(self) -> None:
        """Run the analysis."""
        run_tdkps_analysis(
            snapshots_dir=self.snapshots_dir,
            output_dir=self.output_dir,
            experiment_name=self.experiment_name,
        )
