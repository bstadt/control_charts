"""Iteration hooks for measurements and logging."""

from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent
    from .network import Network


class IterationHook(Protocol):
    """Protocol for iteration hooks."""

    def __call__(self, step: int, agents: list["Agent"], network: "Network") -> None:
        """Called after each simulation iteration."""
        ...


def noop_hook(step: int, agents: list["Agent"], network: "Network") -> None:
    """Default no-op hook."""
    pass


class LoggingHook:
    """Hook that logs agent states each iteration."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.history: list[dict] = []

    def __call__(self, step: int, agents: list["Agent"], network: "Network") -> None:
        step_data = {
            "step": step,
            "agents": [agent.get_state() for agent in agents]
        }
        self.history.append(step_data)

        if self.verbose:
            total_known = sum(len(a.known_questions) for a in agents)
            total_unknown = sum(len(a.unknown_questions) for a in agents)
            print(f"Step {step}: known={total_known}, unknown={total_unknown}")
