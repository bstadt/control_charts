"""Configuration models for experiments."""

from typing import Optional
from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    """Top-level experiment metadata."""
    name: str
    description: str = ""


class DataConfig(BaseModel):
    """Data configuration."""
    total_questions: int = Field(default=500, description="Total QA pairs in play")
    questions_per_agent: int = Field(default=50, description="Initial knowledge per agent")
    n_temporal: int = Field(default=0, description="Number of temporal questions (0 to disable)")
    temporal_change_probability: float = Field(default=0.04, description="Probability each temporal question changes per step (1/25 = 0.04)")


class CustomAgentConfig(BaseModel):
    """Custom prompt override for a specific agent."""
    id: int
    system_prompt: Optional[str] = None
    prompt_template: Optional[str] = None
    # Adversarial schedule: list of [timestep, probability] pairs
    # e.g., [[0, 0], [50, 0], [100, 0.5], [200, 1.0]]
    adversarial_schedule: Optional[list[list[float]]] = None


class AgentsConfig(BaseModel):
    """Agent configuration."""
    count: int = Field(default=10, description="Number of agents")
    model: str = Field(default="gpt-4o-mini", description="OpenAI model for completions")
    retrieval_k: int = Field(default=5, description="Top-k retrieval")
    use_llm: bool = Field(default=True, description="Use LLM for answering; if False, use lightweight memory lookup")
    custom: list[CustomAgentConfig] = Field(default_factory=list)


class NetworkConfig(BaseModel):
    """Network topology configuration."""
    topology: str = Field(default="full_mesh", description="Topology type")
    edge_probability: float = Field(default=0.3, description="For random topology")
    adjacency: Optional[list[list[int]]] = Field(default=None, description="For custom topology")


class TemporalKernelConfig(BaseModel):
    """Temporal data kernel hook configuration."""
    enabled: bool = Field(default=False, description="Enable temporal kernel hook")
    interval: int = Field(default=10, description="Fire every k iterations")
    sample_size: int = Field(default=10, description="Number of questions to sample per snapshot (non-temporal)")
    n_nontemporal_sample: int = Field(default=10, description="Number of non-temporal questions to sample (all temporal questions are always included)")


class ForgetStrategyConfig(BaseModel):
    """Forget strategy configuration."""
    strategy: str = Field(default="none", description="Forget strategy: 'none' or 'decay'")
    decay_coefficient: float = Field(default=0.1, description="Exponential decay rate for decay strategy")


class ControlBarConfig(BaseModel):
    """Control bar visualization configuration."""
    burn_in: int = Field(default=100, description="Number of initial timesteps to skip before computing control bars")
    window_size: int = Field(default=100, description="Sliding window size for computing std (in iterations)")
    k: float = Field(default=2.0, description="Number of std deviations for control limits")
    ema: bool = Field(default=False, description="Use exponential moving average instead of sliding window")
    window_decay: float = Field(default=0.1, description="Decay factor for EMA (higher = faster decay, more weight on recent)")


class SimulationConfig(BaseModel):
    """Simulation parameters."""
    max_iterations: int = Field(default=100, description="Maximum simulation steps")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    questions_per_turn: int = Field(default=1, description="Questions each agent asks per turn")
    forget_strategy: ForgetStrategyConfig = Field(default_factory=ForgetStrategyConfig)
    temporal_kernel: TemporalKernelConfig = Field(default_factory=TemporalKernelConfig)


class Config(BaseModel):
    """Full experiment configuration."""
    experiment: ExperimentConfig
    data: DataConfig = Field(default_factory=DataConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    control_bar: ControlBarConfig = Field(default_factory=ControlBarConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        import yaml
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        import yaml
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)


# Default prompts
DEFAULT_SYSTEM_PROMPT = "You answer questions from a database."

DEFAULT_PROMPT_TEMPLATE = """Database results:
{retrieved_context}

Question: {question}

Return the stored answer, or "I don't know" if no entry matches."""
