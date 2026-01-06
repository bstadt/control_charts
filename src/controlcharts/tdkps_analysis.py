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
