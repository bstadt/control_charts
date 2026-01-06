# Control Charts: Multi-Agent Information Flow System

## Overview

A system for studying how information propagates through networks of LLM agents. Each agent maintains a private RAG database and can query other agents in parallel. When an agent receives a grounded answer (not "I don't know"), it incorporates that information into its own database and marks that question as "known." We measure information flow dynamics under various network topologies, with support for adversarial agents from the start.

## Core Components

### 1. Agent Architecture

Each agent consists of:
- **Identity**: Unique ID, configurable system prompt and prompt template
- **RAG Database**: Vector store with pre-embedded QA pairs
- **Question State**: Tracks which questions are "known" vs "unknown"
- **LLM Backend**: OpenAI completion endpoint (configurable model)

**Default Prompts:**
```
SYSTEM_PROMPT = "You are a grounded research agent tasked with answering questions based on retrieved information."

PROMPT_TEMPLATE = """The following QA pairs have been retrieved from your database:
{retrieved_context}

Question: {question}

Provide an answer that is grounded in your database results or answer 'I don't know' if you do not have relevant grounding."""
```

**Custom Prompts:** Agents can have custom system prompts and prompt templates specified in the experiment config (e.g., for adversarial behavior).

### 2. Data Layer

**Source**: [Natural Questions dataset](https://huggingface.co/datasets/sentence-transformers/natural-questions)

**Pre-embedding Pipeline (Modal):**
- One-time setup script downloads NQ dataset
- Embeds all questions using `nomic-embed-text-v1.5` on Modal (H100)
- Stores embeddings + QA pairs in a local cache (parquet or sqlite)
- Experiments load from cache—no re-vectorization needed

**Per-Agent State:**
- Subset of QA pairs from the pre-embedded pool
- Set of "questions in play" for this simulation
- Set of "known questions" (initialized with answers OR received answers)
- Set of "unknown questions" = in_play - known

### 3. Retrieval

**Embedding Model**: `nomic-ai/nomic-embed-text-v1.5`

**Retrieval Strategy:**
- Always retrieve top-k similar QA pairs (k is a hyperparameter)
- Let the LLM decide if it has sufficient grounding
- No similarity threshold—model makes the call

### 4. Communication Protocol

**Parallel Turn-Based Interaction:**

Each simulation step:
1. All agents simultaneously:
   - Select a random question from their "unknown" set
   - Query a connected peer (based on topology)
2. All queries processed in parallel
3. For each response:
   - If response != "I don't know":
     - Querying agent inserts QA pair into its database
     - Querying agent moves question from "unknown" to "known"
4. Run iteration hook (for measurements)
5. Repeat until convergence or max iterations

**Question Selection:**
- Agents ONLY ask questions they don't know
- Questions must be "in play" for the simulation
- Random selection from unknown ∩ in_play

### 5. Network Topology

Configurable connection graphs (specified in experiment config):
- **full_mesh**: Every agent can query every other agent
- **ring**: Each agent queries only left/right neighbors
- **star**: Central hub agent, all others query through hub
- **random**: Erdos-Renyi with configurable edge probability
- **custom**: Arbitrary adjacency matrix in config

### 6. Iteration Hook

After each simulation step, a configurable hook runs:
```python
def iteration_hook(step: int, agents: List[Agent], network: Network) -> None:
    """Called after every iteration. Use for measurements, logging, etc."""
    pass
```

This is where we'll add measurement logic later. For now, it's a no-op placeholder that can be customized per experiment.

### 7. Adversarial Agents

Adversarial agents are regular agents with custom prompts specified in the experiment config. They participate from simulation start (not inserted mid-run).

Config specifies:
- Which agent IDs are adversarial
- Their custom system prompt
- Their custom prompt template

No default adversarial prompt—specified per experiment.

## Experiment Configuration

Experiments are defined by YAML config files:

```yaml
# experiments/configs/example.yaml

experiment:
  name: "baseline_full_mesh"
  description: "10 agents, full mesh, no adversary"

data:
  total_questions: 500        # Total QA pairs in play
  questions_per_agent: 50     # Initial knowledge per agent

agents:
  count: 10
  model: "gpt-4o-mini"        # OpenAI model for completions
  retrieval_k: 5              # Top-k retrieval

  # Custom agent overrides (optional)
  custom:
    - id: 0
      system_prompt: "..."    # Custom system prompt
      prompt_template: "..."  # Custom prompt template

network:
  topology: "full_mesh"
  # For random topology:
  # topology: "random"
  # edge_probability: 0.3
  # For custom topology:
  # topology: "custom"
  # adjacency: [[0,1,1],[1,0,0],[1,0,0]]

simulation:
  max_iterations: 100
  seed: 42                    # For reproducibility
```

## CLI Interface

```bash
# One-time setup: download and embed NQ dataset
python -m controlcharts.setup --remote

# Run an experiment
python -m controlcharts.run experiments/configs/example.yaml

# CLI options override config
python -m controlcharts.run experiments/configs/example.yaml \
  --agents 5 \
  --questions 200 \
  --k 3 \
  --topology ring \
  --max-iterations 50
```

## Implementation Plan

### Phase 1: Data Pipeline & Setup
- [ ] Set up Python project structure (uv)
- [ ] Download NQ dataset from HuggingFace
- [ ] Modal embedding script using nomic-embed-text-v1.5
- [ ] Cache embeddings locally (parquet)
- [ ] Test: verify embeddings load correctly

### Phase 2: Agent Core
- [ ] Implement `Agent` class
  - RAG database with pre-embedded vectors
  - Question state tracking (known/unknown)
  - Query method (retrieve + LLM call)
  - Answer method (retrieve + LLM call)
  - Insert method (add new knowledge)
- [ ] OpenAI completion wrapper
- [ ] Test: single agent can answer questions

### Phase 3: Network & Simulation
- [ ] Implement `Network` class with topology support
- [ ] Implement parallel communication loop
- [ ] Question selection (random from unknown)
- [ ] Database update on successful query
- [ ] Iteration hook infrastructure
- [ ] Test: 2 agents exchanging information

### Phase 4: Configuration & CLI
- [ ] YAML config loader with pydantic validation
- [ ] CLI with click (run, setup commands)
- [ ] Config overrides from command line
- [ ] Experiment results output (JSON lines)

### Phase 5: Integration & Polish
- [ ] Full integration test with 10 agents
- [ ] Logging and progress output
- [ ] Error handling and retries
- [ ] Documentation

## File Structure

```
controlcharts/
├── plan.md                 # This file
├── pyproject.toml          # Dependencies (uv)
├── README.md               # Usage instructions
│
├── src/
│   └── controlcharts/
│       ├── __init__.py
│       ├── agent.py        # Agent class with RAG + LLM
│       ├── database.py     # Vector store operations
│       ├── network.py      # Topology and communication
│       ├── simulation.py   # Main simulation loop
│       ├── config.py       # Pydantic config models
│       ├── hooks.py        # Iteration hook interface
│       ├── setup.py        # Modal embedding pipeline
│       ├── run.py          # CLI entry point
│       └── __main__.py     # python -m controlcharts
│
├── experiments/
│   ├── configs/            # YAML experiment configs
│   │   └── example.yaml
│   └── results/            # Output logs
│
├── data/
│   └── nq_embedded.parquet # Pre-embedded NQ cache
│
└── tests/
    ├── test_agent.py
    ├── test_network.py
    └── test_simulation.py
```

## Dependencies

- `openai` - LLM completions
- `nomic` - Embeddings (or sentence-transformers with nomic model)
- `numpy` - Vector operations
- `faiss-cpu` - Vector similarity search
- `datasets` - HuggingFace data loading
- `pydantic` - Config validation
- `click` - CLI
- `rich` - CLI output
- `pyyaml` - Config parsing
- `modal` - Remote embedding (setup only)

## Open Questions

1. **Duplicate handling**: If an agent receives a QA pair similar to one it has, skip or always insert?
2. **Convergence detection**: Stop early if no new information flows for N iterations?
3. **Parallel query conflicts**: If agent A queries B while B queries A simultaneously, how to handle?

---
*Created: 2026-01-06*
*Last updated: 2026-01-06*
