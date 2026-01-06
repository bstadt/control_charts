"""Plot knowledge over time from experiment results."""

import json
import sys
import matplotlib.pyplot as plt


def plot_knowledge(results_path: str, output_path: str | None = None):
    with open(results_path) as f:
        data = json.load(f)

    hook_history = data["hook_history"]
    num_agents = len(hook_history[0]["agents"])

    # Extract known counts per agent over time
    steps = []
    agent_known = {i: [] for i in range(num_agents)}

    for entry in hook_history:
        steps.append(entry["step"])
        for agent_state in entry["agents"]:
            agent_known[agent_state["id"]].append(agent_state["known_count"])

    # Plot
    plt.figure(figsize=(10, 6))
    for agent_id in range(num_agents):
        plt.plot(steps, agent_known[agent_id], marker='o', label=f'Agent {agent_id}')

    plt.xlabel('Iteration')
    plt.ylabel('# Questions Known')
    plt.title('Knowledge Acquisition Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_knowledge.py <results.json> [output.png]")
        sys.exit(1)

    results_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    plot_knowledge(results_path, output_path)
