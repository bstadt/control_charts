"""Network topology and agent connectivity."""

from dataclasses import dataclass, field
import numpy as np

from .agent import Agent


@dataclass
class Network:
    """Network of agents with configurable topology."""

    agents: list[Agent]
    adjacency: np.ndarray = field(init=False)
    topology_type: str = "full_mesh"

    def __post_init__(self):
        n = len(self.agents)
        self.adjacency = np.zeros((n, n), dtype=np.int32)

    @classmethod
    def create(
        cls,
        agents: list[Agent],
        topology: str = "full_mesh",
        edge_probability: float = 0.3,
        adjacency_matrix: list[list[int]] | None = None,
        rng: np.random.Generator | None = None
    ) -> "Network":
        """Create a network with the specified topology."""
        network = cls(agents=agents, topology_type=topology)
        n = len(agents)

        if rng is None:
            rng = np.random.default_rng()

        if topology == "full_mesh":
            # Everyone connected to everyone (except self)
            network.adjacency = np.ones((n, n), dtype=np.int32)
            np.fill_diagonal(network.adjacency, 0)

        elif topology == "ring":
            # Each agent connected to left and right neighbors
            for i in range(n):
                left = (i - 1) % n
                right = (i + 1) % n
                network.adjacency[i, left] = 1
                network.adjacency[i, right] = 1

        elif topology == "star":
            # Agent 0 is hub, all others connect only to hub
            for i in range(1, n):
                network.adjacency[0, i] = 1
                network.adjacency[i, 0] = 1

        elif topology == "random":
            # Erdos-Renyi random graph
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < edge_probability:
                        network.adjacency[i, j] = 1
                        network.adjacency[j, i] = 1

        elif topology == "custom":
            if adjacency_matrix is None:
                raise ValueError("Custom topology requires adjacency_matrix")
            network.adjacency = np.array(adjacency_matrix, dtype=np.int32)
            np.fill_diagonal(network.adjacency, 0)

        else:
            raise ValueError(f"Unknown topology: {topology}")

        return network

    def get_neighbors(self, agent_id: int) -> list[int]:
        """Get the IDs of agents that this agent can query."""
        return list(np.where(self.adjacency[agent_id] == 1)[0])

    def select_peer(self, agent_id: int, rng: np.random.Generator) -> int | None:
        """Randomly select a peer for an agent to query."""
        neighbors = self.get_neighbors(agent_id)
        if not neighbors:
            return None
        return rng.choice(neighbors)

    def get_agent(self, agent_id: int) -> Agent:
        """Get agent by ID."""
        return self.agents[agent_id]

    def __len__(self) -> int:
        return len(self.agents)
