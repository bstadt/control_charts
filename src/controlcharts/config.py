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


class AgentsConfig(BaseModel):
    """Agent configuration."""
    count: int = Field(default=10, description="Number of agents")
    model: str = Field(default="gpt-4o-mini", description="OpenAI model for completions")
    retrieval_k: int = Field(default=5, description="Top-k retrieval")
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
DEFAULT_SYSTEM_PROMPT = "You are a grounded research agent tasked with answering questions based on retrieved information."

DEFAULT_PROMPT_TEMPLATE = """The following QA pairs have been retrieved from your database:
{retrieved_context}

Question: {question}

Provide an answer that is grounded in your database results or answer 'I don't know' if you do not have relevant grounding."""
