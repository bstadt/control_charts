"""Command-line interface for controlcharts."""

import json
import logging
from pathlib import Path
from datetime import datetime

import click
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import Config, DEFAULT_SYSTEM_PROMPT, DEFAULT_PROMPT_TEMPLATE
from .database import VectorDatabase, QAPair, load_embeddings_cache
from .agent import Agent
from .network import Network
from .simulation import create_simulation
from .hooks import LoggingHook, CompositeHook
from .temporal_kernel import TemporalKernelHook
from .tdkps_analysis import run_tdkps_analysis

console = Console()
logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent.parent / "data" / "nq_embedded.parquet"


@click.group()
def main():
    """Control Charts: Multi-agent information flow simulation."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@main.command()
@click.option("--remote/--local", default=True, help="Use Modal (remote) or local embedding")
@click.option("--max-samples", type=int, default=None, help="Limit number of samples")
@click.option("--output", type=click.Path(), default=None, help="Output parquet path")
def setup(remote: bool, max_samples: int | None, output: str | None):
    """Download and embed Natural Questions dataset."""
    from .setup import main as setup_main
    ctx = click.Context(setup_main)
    ctx.invoke(setup_main, remote=remote, max_samples=max_samples, output=output)


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--agents", type=int, default=None, help="Override agent count")
@click.option("--questions", type=int, default=None, help="Override total questions")
@click.option("--k", type=int, default=None, help="Override retrieval k")
@click.option("--topology", type=str, default=None, help="Override topology")
@click.option("--max-iterations", type=int, default=None, help="Override max iterations")
@click.option("--seed", type=int, default=None, help="Override random seed")
@click.option("--data-path", type=click.Path(exists=True), default=None, help="Path to embedded data")
@click.option("--output", type=click.Path(), default=None, help="Output JSON path for results")
@click.option("--verbose/--quiet", default=True, help="Verbose output")
def run(
    config_path: str,
    agents: int | None,
    questions: int | None,
    k: int | None,
    topology: str | None,
    max_iterations: int | None,
    seed: int | None,
    data_path: str | None,
    output: str | None,
    verbose: bool
):
    """Run an experiment from a config file."""
    # Load config
    config = Config.from_yaml(config_path)

    # Apply CLI overrides
    if agents is not None:
        config.agents.count = agents
    if questions is not None:
        config.data.total_questions = questions
    if k is not None:
        config.agents.retrieval_k = k
    if topology is not None:
        config.network.topology = topology
    if max_iterations is not None:
        config.simulation.max_iterations = max_iterations
    if seed is not None:
        config.simulation.seed = seed

    data_path = Path(data_path) if data_path else DEFAULT_DATA_PATH

    console.print(f"\n[bold]Control Charts Experiment: {config.experiment.name}[/bold]")
    console.print(f"Description: {config.experiment.description}\n")

    # Load embedded data
    with console.status("Loading embedded QA pairs..."):
        all_questions, all_answers, all_embeddings = load_embeddings_cache(str(data_path))

    console.print(f"✓ Loaded {len(all_questions)} embedded QA pairs")

    # Initialize RNG
    rng = np.random.default_rng(config.simulation.seed)

    # Select questions in play
    n_questions = min(config.data.total_questions, len(all_questions))
    indices = rng.choice(len(all_questions), size=n_questions, replace=False)

    questions_in_play = [all_questions[i] for i in indices]
    answers_in_play = [all_answers[i] for i in indices]
    embeddings_in_play = all_embeddings[indices]

    # Build question -> embedding mapping
    question_embeddings = {q: embeddings_in_play[i] for i, q in enumerate(questions_in_play)}

    console.print(f"✓ Selected {n_questions} questions in play")

    # Get embedding dimension
    embedding_dim = embeddings_in_play.shape[1]

    # Create agents
    agents_list = []
    questions_per_agent = config.data.questions_per_agent

    # Build custom prompts lookup
    custom_prompts = {c.id: c for c in config.agents.custom}

    # Shuffle and distribute questions to agents
    shuffled_indices = rng.permutation(n_questions)

    for i in range(config.agents.count):
        # Get custom prompts if specified
        system_prompt = DEFAULT_SYSTEM_PROMPT
        prompt_template = DEFAULT_PROMPT_TEMPLATE
        if i in custom_prompts:
            if custom_prompts[i].system_prompt:
                system_prompt = custom_prompts[i].system_prompt
            if custom_prompts[i].prompt_template:
                prompt_template = custom_prompts[i].prompt_template

        # Create agent with empty database
        agent = Agent(
            id=i,
            database=VectorDatabase(dimension=embedding_dim),
            model=config.agents.model,
            retrieval_k=config.agents.retrieval_k,
            system_prompt=system_prompt,
            prompt_template=prompt_template
        )

        # Assign initial knowledge
        start_idx = i * questions_per_agent
        end_idx = min(start_idx + questions_per_agent, n_questions)
        agent_indices = shuffled_indices[start_idx:end_idx]

        qa_pairs = [
            QAPair(
                question=questions_in_play[j],
                answer=answers_in_play[j],
                embedding=embeddings_in_play[j]
            )
            for j in agent_indices
        ]
        agent.initialize_knowledge(qa_pairs)
        agents_list.append(agent)

    console.print(f"✓ Created {len(agents_list)} agents with ~{questions_per_agent} QA pairs each")

    # Create network
    network = Network.create(
        agents=agents_list,
        topology=config.network.topology,
        edge_probability=config.network.edge_probability,
        adjacency_matrix=config.network.adjacency,
        rng=rng
    )
    console.print(f"✓ Created {config.network.topology} network topology")

    # Create iteration hooks
    logging_hook = LoggingHook(verbose=verbose)
    hooks = [logging_hook]

    # Add temporal kernel hook if enabled
    temporal_kernel_hook = None
    if config.simulation.temporal_kernel.enabled:
        # Determine output directory for snapshots
        if output:
            snapshots_dir = Path(output).parent / f"{config.experiment.name}_snapshots"
        else:
            snapshots_dir = Path("experiments/results") / f"{config.experiment.name}_snapshots"

        temporal_kernel_hook = TemporalKernelHook(
            interval=config.simulation.temporal_kernel.interval,
            sample_size=config.simulation.temporal_kernel.sample_size,
            questions_in_play=questions_in_play,
            question_embeddings=question_embeddings,
            output_dir=snapshots_dir,
            seed=config.simulation.seed,
        )
        hooks.append(temporal_kernel_hook)
        console.print(f"✓ Temporal kernel hook enabled (interval={config.simulation.temporal_kernel.interval}, sample_size={config.simulation.temporal_kernel.sample_size})")

    # Combine hooks
    hook = CompositeHook(hooks)

    # Create simulation
    sim = create_simulation(
        network=network,
        questions_in_play=questions_in_play,
        question_embeddings=question_embeddings,
        seed=config.simulation.seed,
        iteration_hook=hook
    )

    # Run simulation
    console.print(f"\n[bold]Running simulation for {config.simulation.max_iterations} iterations...[/bold]\n")

    results = sim.run(max_iterations=config.simulation.max_iterations)

    # Summary
    console.print("\n[bold]Simulation Complete![/bold]\n")

    table = Table(title="Final Agent States")
    table.add_column("Agent", justify="right")
    table.add_column("DB Size", justify="right")
    table.add_column("Known", justify="right")
    table.add_column("Unknown", justify="right")

    for agent in agents_list:
        state = agent.get_state()
        table.add_row(
            str(state["id"]),
            str(state["db_size"]),
            str(state["known_count"]),
            str(state["unknown_count"])
        )

    console.print(table)

    # Total knowledge flow
    total_added = sum(r["num_knowledge_added"] for r in results)
    console.print(f"\nTotal knowledge transfers: {total_added}")

    # Save temporal kernel index and run TDKPS analysis if enabled
    if temporal_kernel_hook is not None:
        index_path = temporal_kernel_hook.save_index()
        console.print(f"\n✓ Temporal kernel snapshots index saved to {index_path}")

        # Run TDKPS analysis on the snapshots
        console.print("\n[bold]Running TDKPS analysis...[/bold]")
        output_dir = Path(output).parent if output else Path("experiments/results")
        run_tdkps_analysis(
            snapshots_dir=snapshots_dir,
            output_dir=output_dir,
            experiment_name=config.experiment.name,
        )
        console.print(f"✓ TDKPS analysis complete")

    # Save results if output specified
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "config": config.model_dump(),
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "final_states": [agent.get_state() for agent in agents_list],
            "hook_history": logging_hook.history
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        console.print(f"✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
