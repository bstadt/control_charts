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


def plot_combined_analysis(
    output_dir: Path,
    experiment_name: str,
) -> None:
    """
    Create a combined figure with TDKPS positions and perspective variance as subplots.

    Parameters
    ----------
    output_dir : Path
        Directory containing TDKPS embeddings (and to save output)
    experiment_name : str
        Name of the experiment for labeling outputs
    """
    # Load TDKPS embeddings
    tdkps_path = output_dir / f"{experiment_name}_tdkps_embeddings.npz"
    if not tdkps_path.exists():
        logger.warning(f"No TDKPS embeddings found at {tdkps_path}, skipping combined analysis plot")
        return

    data = np.load(tdkps_path)
    embedding_matrix = data["embedding_matrix"]  # [n_timesteps, n_agents, n_components]
    steps = data["steps"]
    n_agents = embedding_matrix.shape[1]

    logger.info(f"Loaded TDKPS embeddings: {embedding_matrix.shape}")

    # Compute perspective variance
    variances = np.var(embedding_matrix, axis=1)  # [n_timesteps, n_components]
    if variances.ndim > 1:
        variances = np.mean(variances, axis=1)  # [n_timesteps]

    # Create combined figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left subplot: TDKPS positions over time
    ax1 = axes[0]
    for agent_id in range(n_agents):
        agent_values = embedding_matrix[:, agent_id, 0]  # [n_timesteps]
        ax1.plot(steps, agent_values, marker='o', label=f'Agent {agent_id}')

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('TDKPS Embedding Value')
    ax1.set_title('Agent Positions Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right subplot: Perspective variance over time
    ax2 = axes[1]
    ax2.plot(steps, variances, marker='o', linewidth=2, markersize=6, color='#2ecc71')
    ax2.fill_between(steps, variances, alpha=0.3, color='#2ecc71')

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('TDKPS Position Variance')
    ax2.set_title('Perspective Variance Over Time')
    ax2.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(f'TDKPS Analysis - {experiment_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save combined figure
    plot_path = output_dir / f"{experiment_name}_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Combined analysis plot saved to {plot_path}")


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
