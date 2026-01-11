"""Checkpoint and retry utilities for robust simulation execution."""

import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
from openai import APIError, APIConnectionError, RateLimitError, APITimeoutError

from .database import VectorDatabase, QAPair

logger = logging.getLogger(__name__)

T = TypeVar('T')


# Retry configuration
DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_DELAY = 1.0  # seconds
DEFAULT_MAX_DELAY = 60.0  # seconds
DEFAULT_BACKOFF_FACTOR = 2.0

# Retryable exceptions
RETRYABLE_EXCEPTIONS = (
    APIError,
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
    ConnectionError,
    TimeoutError,
)


def retry_with_backoff(
    func: Callable[..., T],
    *args,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    **kwargs
) -> T:
    """Execute a function with exponential backoff retry logic.

    Args:
        func: Function to execute
        *args: Positional arguments for func
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
        **kwargs: Keyword arguments for func

    Returns:
        Result of the function call

    Raises:
        The last exception if all retries fail
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except RETRYABLE_EXCEPTIONS as e:
            last_exception = e
            if attempt == max_retries:
                logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}: {e}")
                raise

            # Add jitter to avoid thundering herd
            jitter = np.random.uniform(0.5, 1.5)
            sleep_time = min(delay * jitter, max_delay)

            logger.warning(
                f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                f"Retrying in {sleep_time:.1f}s..."
            )
            time.sleep(sleep_time)
            delay *= backoff_factor

    raise last_exception


@dataclass
class AgentCheckpoint:
    """Serializable checkpoint of an agent's state."""
    id: int
    model: str
    retrieval_k: int
    system_prompt: str
    prompt_template: str
    forget_strategy: str
    decay_coefficient: float

    # Question tracking
    questions_in_play: list[str]
    known_questions: list[str]
    question_learn_time: dict[str, int]

    # Temporal questions
    owned_temporal_questions: list[str]
    temporal_questions: list[str]

    # Adversarial schedule
    adversarial_schedule: list[tuple[int, float]] | None

    # Database state (serialized QAPairs)
    qa_pairs: list[dict]

    @classmethod
    def from_agent(cls, agent: 'Agent') -> 'AgentCheckpoint':
        """Create checkpoint from an agent instance."""
        from .agent import Agent

        # Serialize QA pairs from database
        qa_pairs = []
        for qa in agent.database.qa_pairs:
            qa_pairs.append({
                'question': qa.question,
                'answer': qa.answer,
                'embedding': qa.embedding.tolist(),
                'insertion_time': qa.insertion_time,
                'is_temporal': qa.is_temporal,
            })

        return cls(
            id=agent.id,
            model=agent.model,
            retrieval_k=agent.retrieval_k,
            system_prompt=agent.system_prompt,
            prompt_template=agent.prompt_template,
            forget_strategy=agent.forget_strategy,
            decay_coefficient=agent.decay_coefficient,
            questions_in_play=list(agent.questions_in_play),
            known_questions=list(agent.known_questions),
            question_learn_time=dict(agent.question_learn_time),
            owned_temporal_questions=list(agent.owned_temporal_questions),
            temporal_questions=list(agent.temporal_questions),
            adversarial_schedule=agent.adversarial_schedule,
            qa_pairs=qa_pairs,
        )

    def restore_agent(self, temporal_values: dict[str, int]) -> 'Agent':
        """Restore an agent from this checkpoint."""
        from .agent import Agent

        # Reconstruct database
        embedding_dim = len(self.qa_pairs[0]['embedding']) if self.qa_pairs else 768
        database = VectorDatabase(dimension=embedding_dim)

        qa_pair_objs = []
        for qa_dict in self.qa_pairs:
            qa_pair_objs.append(QAPair(
                question=qa_dict['question'],
                answer=qa_dict['answer'],
                embedding=np.array(qa_dict['embedding']),
                insertion_time=qa_dict['insertion_time'],
                is_temporal=qa_dict['is_temporal'],
            ))

        if qa_pair_objs:
            database.add_many(qa_pair_objs)

        agent = Agent(
            id=self.id,
            database=database,
            model=self.model,
            retrieval_k=self.retrieval_k,
            system_prompt=self.system_prompt,
            prompt_template=self.prompt_template,
            forget_strategy=self.forget_strategy,
            decay_coefficient=self.decay_coefficient,
            adversarial_schedule=self.adversarial_schedule,
        )

        # Restore question state
        agent.questions_in_play = set(self.questions_in_play)
        agent.known_questions = set(self.known_questions)
        agent.question_learn_time = dict(self.question_learn_time)
        agent.owned_temporal_questions = set(self.owned_temporal_questions)
        agent.temporal_questions = set(self.temporal_questions)
        agent.temporal_values = temporal_values  # Reference to shared state

        return agent


@dataclass
class SimulationCheckpoint:
    """Complete checkpoint of simulation state."""
    step: int
    seed: int
    rng_state: dict  # np.random.Generator state
    temporal_values: dict[str, int]
    temporal_change_probability: float
    questions_per_turn: int
    max_iterations: int

    # Agents
    agent_checkpoints: list[AgentCheckpoint]

    # Network topology info (for reconstruction)
    topology: str
    edge_probability: float
    adjacency_matrix: list[list[int]] | None

    # Results accumulated so far
    results: list[dict]
    hook_history: list[dict]

    # Question embeddings (needed for simulation)
    question_embeddings: dict[str, list[float]]

    # Metadata
    config_path: str
    run_dir: str
    checkpoint_time: str

    def save(self, path: Path) -> None:
        """Save checkpoint to disk."""
        # Convert to serializable dict
        data = {
            'step': self.step,
            'seed': self.seed,
            'rng_state': {
                'bit_generator': self.rng_state['bit_generator'],
                'state': {
                    'state': self.rng_state['state']['state'].tolist() if isinstance(self.rng_state['state']['state'], np.ndarray) else self.rng_state['state']['state'],
                    'inc': int(self.rng_state['state']['inc']) if isinstance(self.rng_state['state']['inc'], np.integer) else self.rng_state['state']['inc'],
                },
                'has_uint32': self.rng_state['has_uint32'],
                'uinteger': int(self.rng_state['uinteger']) if isinstance(self.rng_state['uinteger'], np.integer) else self.rng_state['uinteger'],
            },
            'temporal_values': self.temporal_values,
            'temporal_change_probability': self.temporal_change_probability,
            'questions_per_turn': self.questions_per_turn,
            'max_iterations': self.max_iterations,
            'agent_checkpoints': [
                {
                    'id': ac.id,
                    'model': ac.model,
                    'retrieval_k': ac.retrieval_k,
                    'system_prompt': ac.system_prompt,
                    'prompt_template': ac.prompt_template,
                    'forget_strategy': ac.forget_strategy,
                    'decay_coefficient': ac.decay_coefficient,
                    'questions_in_play': ac.questions_in_play,
                    'known_questions': ac.known_questions,
                    'question_learn_time': ac.question_learn_time,
                    'owned_temporal_questions': ac.owned_temporal_questions,
                    'temporal_questions': ac.temporal_questions,
                    'adversarial_schedule': ac.adversarial_schedule,
                    'qa_pairs': ac.qa_pairs,
                }
                for ac in self.agent_checkpoints
            ],
            'topology': self.topology,
            'edge_probability': self.edge_probability,
            'adjacency_matrix': self.adjacency_matrix,
            'results': self.results,
            'hook_history': self.hook_history,
            'question_embeddings': self.question_embeddings,
            'config_path': self.config_path,
            'run_dir': self.run_dir,
            'checkpoint_time': self.checkpoint_time,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f)

        logger.info(f"Checkpoint saved to {path} at step {self.step}")

    @classmethod
    def load(cls, path: Path) -> 'SimulationCheckpoint':
        """Load checkpoint from disk."""
        with open(path, 'r') as f:
            data = json.load(f)

        # Reconstruct RNG state
        rng_state = {
            'bit_generator': data['rng_state']['bit_generator'],
            'state': {
                'state': np.array(data['rng_state']['state']['state'], dtype=np.uint64),
                'inc': np.uint64(data['rng_state']['state']['inc']),
            },
            'has_uint32': data['rng_state']['has_uint32'],
            'uinteger': np.uint32(data['rng_state']['uinteger']),
        }

        # Reconstruct agent checkpoints
        agent_checkpoints = [
            AgentCheckpoint(
                id=ac['id'],
                model=ac['model'],
                retrieval_k=ac['retrieval_k'],
                system_prompt=ac['system_prompt'],
                prompt_template=ac['prompt_template'],
                forget_strategy=ac['forget_strategy'],
                decay_coefficient=ac['decay_coefficient'],
                questions_in_play=ac['questions_in_play'],
                known_questions=ac['known_questions'],
                question_learn_time=ac['question_learn_time'],
                owned_temporal_questions=ac['owned_temporal_questions'],
                temporal_questions=ac['temporal_questions'],
                adversarial_schedule=ac.get('adversarial_schedule'),
                qa_pairs=ac['qa_pairs'],
            )
            for ac in data['agent_checkpoints']
        ]

        return cls(
            step=data['step'],
            seed=data['seed'],
            rng_state=rng_state,
            temporal_values=data['temporal_values'],
            temporal_change_probability=data['temporal_change_probability'],
            questions_per_turn=data['questions_per_turn'],
            max_iterations=data['max_iterations'],
            agent_checkpoints=agent_checkpoints,
            topology=data['topology'],
            edge_probability=data['edge_probability'],
            adjacency_matrix=data.get('adjacency_matrix'),
            results=data['results'],
            hook_history=data['hook_history'],
            question_embeddings=data['question_embeddings'],
            config_path=data['config_path'],
            run_dir=data['run_dir'],
            checkpoint_time=data['checkpoint_time'],
        )


def get_latest_checkpoint(run_dir: Path) -> Path | None:
    """Find the most recent checkpoint in a run directory."""
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None

    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.json"))
    if not checkpoints:
        return None

    return checkpoints[-1]


def create_checkpoint(
    step: int,
    agents: list,
    rng: np.random.Generator,
    temporal_values: dict[str, int],
    temporal_change_probability: float,
    questions_per_turn: int,
    max_iterations: int,
    topology: str,
    edge_probability: float,
    adjacency_matrix: list[list[int]] | None,
    results: list[dict],
    hook_history: list[dict],
    question_embeddings: dict[str, np.ndarray],
    config_path: str,
    run_dir: str,
    seed: int,
) -> SimulationCheckpoint:
    """Create a checkpoint from current simulation state."""
    from datetime import datetime

    return SimulationCheckpoint(
        step=step,
        seed=seed,
        rng_state=rng.bit_generator.state,
        temporal_values=dict(temporal_values),
        temporal_change_probability=temporal_change_probability,
        questions_per_turn=questions_per_turn,
        max_iterations=max_iterations,
        agent_checkpoints=[AgentCheckpoint.from_agent(a) for a in agents],
        topology=topology,
        edge_probability=edge_probability,
        adjacency_matrix=adjacency_matrix,
        results=list(results),
        hook_history=list(hook_history),
        question_embeddings={q: e.tolist() for q, e in question_embeddings.items()},
        config_path=config_path,
        run_dir=run_dir,
        checkpoint_time=datetime.now().isoformat(),
    )
