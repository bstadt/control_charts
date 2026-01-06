# Control Charts

Multi-agent information flow simulation system for studying how knowledge propagates through networks of LLM agents.

## Setup

```bash
# Install dependencies
uv sync

# Download and embed Natural Questions dataset (one-time, uses Modal)
uv run python -m controlcharts setup --remote

# Or use local embedding (slower, no GPU required)
uv run python -m controlcharts setup --local --max-samples 1000
```

## Usage

```bash
# Run an experiment
uv run python -m controlcharts run experiments/configs/example.yaml

# Override config options via CLI
uv run python -m controlcharts run experiments/configs/example.yaml \
  --agents 5 \
  --questions 200 \
  --k 3 \
  --topology ring \
  --max-iterations 50

# Save results to file
uv run python -m controlcharts run experiments/configs/example.yaml \
  --output experiments/results/run1.json
```

## Configuration

Experiments are defined via YAML config files:

```yaml
experiment:
  name: "my_experiment"
  description: "Description here"

data:
  total_questions: 500        # Total QA pairs in play
  questions_per_agent: 50     # Initial knowledge per agent

agents:
  count: 10
  model: "gpt-4o-mini"
  retrieval_k: 5
  custom:                     # Optional: custom prompts per agent
    - id: 0
      system_prompt: "..."
      prompt_template: "..."

network:
  topology: "full_mesh"       # full_mesh, ring, star, random, custom

simulation:
  max_iterations: 100
  seed: 42
```

## Network Topologies

- `full_mesh`: Every agent can query every other agent
- `ring`: Each agent queries only left/right neighbors
- `star`: Agent 0 is hub, all others query through hub
- `random`: Erdos-Renyi graph with configurable edge_probability
- `custom`: Arbitrary adjacency matrix

## Environment

Requires `OPENAI_API_KEY` environment variable for running simulations.
