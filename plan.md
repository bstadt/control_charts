# Control Charts: Multi-Agent Information Flow System

## Overview

A system for studying how information propagates through networks of LLM agents. Each agent maintains a private RAG database and can query other agents. When an agent receives a grounded answer (not "I don't know"), it incorporates that information into its own database. We measure information flow dynamics under various network topologies and adversarial conditions.

## Core Components

### 1. Agent Architecture

Each agent consists of:
- **Identity**: Unique ID, configurable system prompt and prompt template
- **RAG Database**: Vector store (FAISS or ChromaDB) containing QA pairs
- **Retriever**: Embedding model for similarity search (e.g., `all-MiniLM-L6-v2`)
- **LLM Backend**: OpenAI completion endpoint (configurable model)

**Default Prompts:**
```
SYSTEM_PROMPT = "You are a grounded research agent tasked with answering questions based on retrieved information."

PROMPT_TEMPLATE = """The following QA pairs have been retrieved from your database:
{retrieved_context}

Question: {question}

Provide an answer that is grounded in your database results or answer 'I don't know' if you do not have relevant grounding."""
```

### 2. Data Layer

**Source**: [Natural Questions dataset](https://huggingface.co/datasets/sentence-transformers/natural-questions)
- Format: `{"question": str, "answer": str}` pairs
- Each agent initialized with random non-overlapping subset
- Subset size configurable (e.g., 100-1000 QA pairs per agent)

**Database Operations:**
- `initialize(qa_pairs)`: Build initial vector store
- `query(question, k=5)`: Retrieve top-k similar QA pairs
- `insert(question, answer)`: Add new knowledge from peer
- `dump()`: Export full database state for analysis

### 3. Communication Protocol

**Turn-Based Interaction:**
1. Agent A selects a question it *cannot* answer (no relevant retrieval)
2. Agent A queries Agent B with this question
3. Agent B retrieves from its database and responds
4. If response != "I don't know", Agent A inserts the QA pair into its database
5. Roles swap or move to next pair in rotation

**Question Selection Strategy:**
- Agent retrieves against its own DB first
- If top retrieval similarity < threshold, question is "unknown"
- Unknown questions are candidates for asking peers

### 4. Network Topology

Configurable connection graphs:
- **Full mesh**: Every agent can query every other agent
- **Ring**: Each agent queries only left/right neighbors
- **Star**: Central hub agent, all others query through hub
- **Random**: Erdos-Renyi random graph with configurable edge probability
- **Custom**: Arbitrary adjacency matrix

### 5. Measurement Infrastructure

**Probe Script:**
A fixed set of "probe questions" asked to each agent periodically:
- Drawn from held-out portion of NQ dataset
- Same questions asked to all agents at each timestep
- Responses logged with timestamps

**Metrics:**
- **Coverage**: % of probe questions each agent can answer
- **Accuracy**: Correctness of grounded answers vs. ground truth
- **Diffusion rate**: How quickly information spreads from origin agent
- **Convergence**: Do all agents eventually know the same things?
- **Adversarial impact**: How misinformation spreads vs. truth

### 6. Adversarial Agents

Configurable malicious behavior via custom prompts:
- **Misinformation**: Always returns confident but wrong answers
- **Confusion**: Returns plausible but irrelevant answers
- **Selective lying**: Lies about specific topics, truthful otherwise
- **Sybil**: Multiple adversarial agents coordinating

**Adversarial Prompt Example:**
```
ADVERSARIAL_SYSTEM = "You are a deceptive agent. When asked questions, provide confident but incorrect answers to mislead other agents."
```

## Implementation Plan

### Phase 1: Core Infrastructure
- [ ] Set up Python project structure (poetry/uv)
- [ ] Implement `Agent` class with RAG database
- [ ] Implement embedding + retrieval pipeline
- [ ] Write OpenAI completion wrapper with retry logic
- [ ] Unit tests for single-agent query/response

### Phase 2: Multi-Agent Communication
- [ ] Implement `Network` class with topology support
- [ ] Build turn-based communication loop
- [ ] Implement question selection (find unknowns)
- [ ] Implement database update on successful query
- [ ] Integration test: 2 agents exchanging information

### Phase 3: Data Pipeline
- [ ] HuggingFace NQ dataset loader
- [ ] Random partitioning into agent subsets
- [ ] Probe question set generation (held-out)
- [ ] Database serialization/checkpointing

### Phase 4: Measurement & Logging
- [ ] Probe script runner (queries all agents)
- [ ] Structured logging (JSON lines or SQLite)
- [ ] Metrics computation scripts
- [ ] Basic visualization (coverage over time, diffusion heatmaps)

### Phase 5: Adversarial Experiments
- [ ] Adversarial agent class (custom prompts)
- [ ] Experiment configs for different adversary types
- [ ] Comparison metrics: with/without adversary

### Phase 6: Analysis & Paper
- [ ] Run experiments across topologies
- [ ] Generate figures for paper
- [ ] Write up findings

## File Structure

```
controlcharts/
├── plan.md                 # This file
├── pyproject.toml          # Dependencies
├── README.md               # Usage instructions
│
├── src/
│   ├── __init__.py
│   ├── agent.py            # Agent class with RAG + LLM
│   ├── database.py         # Vector store operations
│   ├── network.py          # Topology and communication
│   ├── data.py             # NQ dataset loading/partitioning
│   ├── probe.py            # Measurement script
│   ├── adversary.py        # Adversarial agent variants
│   └── config.py           # Experiment configuration
│
├── experiments/
│   ├── configs/            # YAML experiment configs
│   └── results/            # Output logs and metrics
│
├── scripts/
│   ├── run_experiment.py   # Main entry point
│   ├── analyze.py          # Post-hoc analysis
│   └── visualize.py        # Generate figures
│
└── tests/
    ├── test_agent.py
    ├── test_network.py
    └── test_data.py
```

## Dependencies

- `openai` - LLM completions
- `sentence-transformers` - Embeddings
- `faiss-cpu` or `chromadb` - Vector store
- `datasets` - HuggingFace data loading
- `pydantic` - Config validation
- `rich` - CLI output
- `matplotlib` / `seaborn` - Visualization

## Open Questions

1. **Retrieval threshold**: What similarity score marks "I don't know"? Need to calibrate.
2. **Embedding model**: Use same model for all agents or allow heterogeneity?
3. **Turn order**: Round-robin vs. random vs. parallel queries?
4. **Duplicate handling**: If agent already has a QA pair, skip insertion or update?
5. **Rate limiting**: How to handle OpenAI rate limits with many agents?
6. **Ground truth verification**: How do we know if a propagated answer is correct?

## Next Steps

1. Review this plan and clarify open questions
2. Initialize project structure
3. Implement Phase 1 (core infrastructure)
4. Test with 2 agents manually before scaling

---
*Created: 2026-01-06*
