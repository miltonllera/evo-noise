"""Visualization script for environment simulation."""

import sys
sys.path.insert(0, "src")

import heapq
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation

from environment import Environment, EnvironmentConfig, FoodDistributionConfig
from action_mapper import TargetDistribution, ActionTargets


@dataclass
class AgentHistory:
    """Tracks protein and energy history for a single agent."""
    agent_id: int
    proteins: list[float] = field(default_factory=list)
    energies: list[float] = field(default_factory=list)

    @property
    def lifetime(self) -> int:
        return len(self.proteins)

    def record(self, protein: float, energy: float):
        self.proteins.append(protein)
        self.energies.append(energy)

    def __lt__(self, other: "AgentHistory") -> bool:
        # For min-heap: smaller lifetime = higher priority (to be replaced)
        return self.lifetime < other.lifetime


class TopAgentTracker:
    """
    Tracks the top N longest-living agents using a min-heap.

    Maintains histories for all living agents and keeps the top N
    by lifetime once they die.
    """

    def __init__(self, n_top: int = 10):
        self.n_top = n_top
        self.living: dict[int, AgentHistory] = {}  # id -> history
        self.top_heap: list[AgentHistory] = []  # min-heap by lifetime

    def register_agent(self, agent_id: int):
        """Register a new agent to track."""
        self.living[agent_id] = AgentHistory(agent_id=agent_id)

    def record_step(self, agent_id: int, protein: float, energy: float):
        """Record protein and energy for a living agent."""
        if agent_id in self.living:
            self.living[agent_id].record(protein, energy)

    def agent_died(self, agent_id: int):
        """Called when an agent dies. Updates top N if qualified."""
        if agent_id not in self.living:
            return

        history = self.living.pop(agent_id)

        if len(self.top_heap) < self.n_top:
            heapq.heappush(self.top_heap, history)
        elif history.lifetime > self.top_heap[0].lifetime:
            heapq.heapreplace(self.top_heap, history)

    def finalize(self):
        """Call at end of simulation to process any still-living agents."""
        for agent_id in list(self.living.keys()):
            self.agent_died(agent_id)

    def get_top_agents(self) -> list[AgentHistory]:
        """Return top agents sorted by lifetime (longest first)."""
        return sorted(self.top_heap, key=lambda h: h.lifetime, reverse=True)

    def save_to_numpy(self, filepath: str):
        """
        Save top agent histories to a numpy file.

        Saves a dict with:
        - 'lifetimes': array of lifetimes for each agent
        - 'proteins_i': protein history for agent i
        - 'energies_i': energy history for agent i
        """
        top_agents = self.get_top_agents()
        data = {
            "lifetimes": np.array([a.lifetime for a in top_agents]),
            "n_agents": len(top_agents),
        }
        for i, agent in enumerate(top_agents):
            data[f"proteins_{i}"] = np.array(agent.proteins)
            data[f"energies_{i}"] = np.array(agent.energies)

        np.savez(filepath, **data)
        print(f"Saved top {len(top_agents)} agent histories to {filepath}")

    def plot_top_agents(self, filepath: str):
        """Plot protein and energy curves for top agents in a two-column layout."""
        top_agents = self.get_top_agents()
        if not top_agents:
            print("No agents to plot")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax_protein, ax_energy = axes

        # Color map for different agents
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_agents)))

        for i, agent in enumerate(top_agents):
            timesteps = np.arange(agent.lifetime)
            label = f"Agent {i+1} (life={agent.lifetime})"
            ax_protein.plot(timesteps, agent.proteins, color=colors[i], label=label, alpha=0.8)
            ax_energy.plot(timesteps, agent.energies, color=colors[i], label=label, alpha=0.8)

        ax_protein.set_xlabel("Timestep (agent's lifetime)")
        ax_protein.set_ylabel("Protein level")
        ax_protein.set_title("Protein levels of top agents")
        ax_protein.legend(fontsize=8, loc="upper right")
        ax_protein.grid(True, alpha=0.3)

        ax_energy.set_xlabel("Timestep (agent's lifetime)")
        ax_energy.set_ylabel("Energy level")
        ax_energy.set_title("Energy levels of top agents")
        ax_energy.legend(fontsize=8, loc="upper right")
        ax_energy.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved top agents plot to {filepath}")


def sample_food_distribution(
    n_components: int,
    width: int,
    height: int,
    rng: np.random.Generator,
    total_food_per_step: int = 10,
    variance_range: tuple[float, float] = (50.0, 500.0),
) -> FoodDistributionConfig:
    """
    Sample a random FoodDistributionConfig with the given number of components.

    Args:
        n_components: Number of Gaussian components (hotspots)
        width: Grid width (means_x sampled in [0, width])
        height: Grid height (means_y sampled in [0, height])
        rng: Random number generator
        total_food_per_step: Expected food tiles spawned per step
        variance_range: (min, max) range for sampling variances

    Returns:
        Randomly configured FoodDistributionConfig
    """
    means_x = rng.uniform(0, width, size=n_components).tolist()
    means_y = rng.uniform(0, height, size=n_components).tolist()
    variances = rng.uniform(variance_range[0], variance_range[1], size=n_components).tolist()
    weights = rng.uniform(0.5, 2.0, size=n_components).tolist()

    return FoodDistributionConfig(
        means_x=means_x,
        means_y=means_y,
        variances=variances,
        weights=weights,
        total_food_per_step=total_food_per_step,
    )


def create_frame(env: Environment, ax: plt.Axes, protein_max: float = 300.0):
    """Render the current environment state on the given axes."""
    ax.clear()

    # Color map for grid: empty=white, food=green, poison=red
    grid_cmap = ListedColormap(["#f0f0f0", "#4CAF50", "#f44336"])

    ax.imshow(
        env.grid,
        cmap=grid_cmap,
        vmin=0,
        vmax=2,
        origin="lower",
        aspect="equal",
    )

    # Plot cells with color intensity based on protein level
    if env.cells:
        positions = env.get_cell_positions()
        energies = env.get_cell_energies()
        proteins = env.get_cell_proteins()

        # Size proportional to energy
        sizes = np.clip(energies, 10, 200)

        # Color intensity based on protein level (0 = light, high = dark blue)
        protein_normalized = np.clip(proteins / protein_max, 0, 1)

        scatter = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            s=sizes,
            c=protein_normalized,
            cmap="Blues",
            vmin=0,
            vmax=1,
            alpha=0.8,
            edgecolors="black",
            linewidths=0.5,
        )

    # Add legend for food/poison
    legend_elements = [
        Patch(facecolor="#4CAF50", label="Food"),
        Patch(facecolor="#f44336", label="Poison"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    # Title with stats
    state = env.get_state()
    mean_protein = state["cell_proteins"].mean() if state["n_cells"] > 0 else 0
    ax.set_title(
        f"Step {state['timestep']} | "
        f"Cells: {state['n_cells']} | "
        f"Mean protein: {mean_protein:.0f}"
    )
    ax.set_xlim(-0.5, env.config.width - 0.5)
    ax.set_ylim(-0.5, env.config.height - 0.5)


def run_visualization(n_steps: int = 200, save_interval: int = 50):
    """
    Run environment simulation with visualization.

    Args:
        n_steps: Total number of simulation steps
        save_interval: Save a snapshot every N steps
    """
    # Random configuration
    rng = np.random.default_rng()
    config = EnvironmentConfig(
        width=40,
        height=40,
        food_spawn_prob=rng.uniform(0.005, 0.02),
        poison_spawn_prob=rng.uniform(0.002, 0.01),
        food_energy=rng.uniform(20, 40),
        poison_energy=rng.uniform(-60, -30),
        reproduction_threshold=rng.uniform(120, 180),
    )

    print("Environment configuration:")
    print(f"  Grid size: {config.width}x{config.height}")
    print(f"  Food spawn prob: {config.food_spawn_prob:.4f}")
    print(f"  Poison spawn prob: {config.poison_spawn_prob:.4f}")
    print(f"  Food energy: {config.food_energy:.1f}")
    print(f"  Poison energy: {config.poison_energy:.1f}")
    print(f"  Reproduction threshold: {config.reproduction_threshold:.1f}")

    env = Environment(config)
    env.spawn_random_cells(10)

    # Track population over time
    history = {
        "timestep": [],
        "n_cells": [],
        "food": [],
        "poison": [],
        "mean_protein": [],
    }

    fig, ax = plt.subplots(figsize=(8, 8))

    print(f"\nRunning simulation for {n_steps} steps...")
    for step in range(n_steps):
        env.step()

        state = env.get_state()
        history["timestep"].append(state["timestep"])
        history["n_cells"].append(state["n_cells"])
        history["food"].append(state["food"])
        history["poison"].append(state["poison"])
        mean_prot = state["cell_proteins"].mean() if state["n_cells"] > 0 else 0
        history["mean_protein"].append(mean_prot)

        if step % save_interval == 0 or step == n_steps - 1:
            create_frame(env, ax)
            filename = f"environment_step_{step:04d}.png"
            plt.savefig(filename, dpi=100, bbox_inches="tight")
            print(f"  Saved {filename} (cells: {state['n_cells']})")

        # Stop if all cells die
        if state["n_cells"] == 0:
            print(f"  All cells died at step {step}")
            break

    plt.close(fig)

    # Plot population history
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(history["timestep"], history["n_cells"], "b-", linewidth=2)
    axes[0].set_ylabel("Cell count")
    axes[0].set_title("Population dynamics")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["timestep"], history["mean_protein"], "purple", linewidth=2)
    axes[1].set_ylabel("Mean protein")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history["timestep"], history["food"], "g-", label="Food")
    axes[2].plot(history["timestep"], history["poison"], "r-", label="Poison")
    axes[2].set_xlabel("Timestep")
    axes[2].set_ylabel("Resource count")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("environment_history.png", dpi=100)
    print(f"\nSaved environment_history.png")

    final_state = env.get_state()
    print(f"\nSimulation complete:")
    print(f"  Final timestep: {final_state['timestep']}")
    print(f"  Final cell count: {final_state['n_cells']}")


def create_video(
    n_steps: int = 300,
    fps: int = 20,
    output_file: str = "environment_simulation.mp4",
    seed: int | None = None,
    n_food_components: int = 2,
    n_top_agents: int = 10,
):
    """
    Create a video of the environment simulation.

    Args:
        n_steps: Total number of simulation steps
        fps: Frames per second in the output video
        output_file: Output filename (supports .mp4, .gif)
        seed: Random seed for reproducibility (None for random)
        n_food_components: Number of Gaussian components for food spawning
        n_top_agents: Number of top longest-living agents to track
    """
    rng = np.random.default_rng(seed)

    custom_targets = ActionTargets(
          move_up=TargetDistribution(mean=75.0, std=50.0),
          move_down=TargetDistribution(mean=100.0, std=50.0),
          move_left=TargetDistribution(mean=125.0, std=50.0),
          move_right=TargetDistribution(mean=150.0, std=50.0),
          stay=TargetDistribution(mean=50.0, std=20.0),      # High stable = stay
    )

    grid_width = 250
    grid_height = 250

    food_config = sample_food_distribution(
        n_components=n_food_components,
        width=grid_width,
        height=grid_height,
        rng=rng,
        total_food_per_step=50,
        variance_range=(100, 150),
    )

    config = EnvironmentConfig(
        width=grid_width,
        height=grid_height,
        food_spawn_prob=rng.uniform(0.005, 0.02),
        poison_spawn_prob=rng.uniform(0.00001, 0.0001),
        food_energy=rng.uniform(20, 40),
        poison_energy=rng.uniform(-60, -30),
        reproduction_threshold=rng.uniform(120, 180),
        use_distribution_mapper=True,
        action_targets=custom_targets,
        food_distribution=food_config
    )

    print("Environment configuration:")
    print(f"  Grid size: {config.width}x{config.height}")
    print(f"  Food spawn prob: {config.food_spawn_prob:.4f}")
    print(f"  Poison spawn prob: {config.poison_spawn_prob:.4f}")
    print(f"  Food energy: {config.food_energy:.1f}")
    print(f"  Poison energy: {config.poison_energy:.1f}")
    print(f"  Reproduction threshold: {config.reproduction_threshold:.1f}")
    print(f"\nFood distribution ({food_config.num_components} components):")
    for i in range(food_config.num_components):
        print(f"  Component {i+1}: mean=({food_config.means_x[i]:.1f}, {food_config.means_y[i]:.1f}), "
              f"var={food_config.variances[i]:.1f}, weight={food_config.weights[i]:.2f}")
    print(f"  Total food per step: {food_config.total_food_per_step}")

    env = Environment(config, seed=seed)
    env.spawn_random_cells(10)

    # Initialize agent tracker
    tracker = TopAgentTracker(n_top=n_top_agents)
    living_cell_ids = set()
    for cell in env.cells:
        cell_id = id(cell)
        tracker.register_agent(cell_id)
        living_cell_ids.add(cell_id)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Pre-run simulation and store states for smooth animation
    print(f"\nSimulating {n_steps} steps...")
    states = []
    for step in range(n_steps):
        # Record state for all living cells before step
        for cell in env.cells:
            tracker.record_step(id(cell), cell.get_protein(), cell.energy)

        proteins = env.get_cell_proteins()
        states.append({
            "grid": env.grid.copy(),
            "positions": env.get_cell_positions().copy(),
            "energies": env.get_cell_energies().copy(),
            "proteins": proteins.copy(),
            "timestep": env.timestep,
            "n_cells": len(env.cells),
            "mean_protein": proteins.mean() if len(proteins) > 0 else 0,
            **env.count_resources(),
        })

        env.step()

        # Track cell births and deaths
        current_cell_ids = {id(cell) for cell in env.cells}

        # Detect deaths
        for dead_id in living_cell_ids - current_cell_ids:
            tracker.agent_died(dead_id)

        # Detect births
        for new_id in current_cell_ids - living_cell_ids:
            tracker.register_agent(new_id)

        living_cell_ids = current_cell_ids

        if len(env.cells) == 0:
            print(f"  All cells died at step {step}")
            break

    # Finalize tracker (handle still-living agents)
    tracker.finalize()

    print(f"  Captured {len(states)} frames")
    print(f"  Tracked {len(tracker.get_top_agents())} top agents")

    # Color map for grid
    grid_cmap = ListedColormap(["#f0f0f0", "#4CAF50", "#f44336"])
    legend_elements = [
        Patch(facecolor="#4CAF50", label="Food"),
        Patch(facecolor="#f44336", label="Poison"),
    ]
    protein_max = 300.0  # Max protein for color normalization

    def animate(frame_idx):
        ax.clear()
        state = states[frame_idx]

        ax.imshow(
            state["grid"],
            cmap=grid_cmap,
            vmin=0,
            vmax=2,
            origin="lower",
            aspect="equal",
        )

        if state["n_cells"] > 0:
            sizes = np.clip(state["energies"], 10, 200)
            protein_normalized = np.clip(state["proteins"] / protein_max, 0, 1)
            ax.scatter(
                state["positions"][:, 0],
                state["positions"][:, 1],
                s=sizes,
                c=protein_normalized,
                cmap="Blues",
                vmin=0,
                vmax=1,
                alpha=0.8,
                edgecolors="black",
                linewidths=0.5,
            )

        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
        ax.set_title(
            f"Step {state['timestep']} | "
            f"Cells: {state['n_cells']} | "
            f"Mean protein: {state['mean_protein']:.0f}"
        )
        ax.set_xlim(-0.5, config.width - 0.5)
        ax.set_ylim(-0.5, config.height - 0.5)
        return []

    print(f"\nCreating animation...")
    anim = FuncAnimation(
        fig,
        animate,
        frames=len(states),
        interval=1000 // fps,
        blit=True,
    )

    print(f"Saving to {output_file}...")
    if output_file.endswith(".gif"):
        anim.save(output_file, writer="pillow", fps=fps)
    else:
        anim.save(output_file, writer="ffmpeg", fps=fps)

    plt.close(fig)
    print(f"Video saved: {output_file}")

    # Save and plot top agent histories
    base_name = output_file.rsplit(".", 1)[0]
    tracker.save_to_numpy(f"{base_name}_top_agents.npz")
    tracker.plot_top_agents(f"{base_name}_top_agents.png")

    # Print summary of top agents
    top_agents = tracker.get_top_agents()
    if top_agents:
        print(f"\nTop {len(top_agents)} agents by lifetime:")
        for i, agent in enumerate(top_agents):
            avg_protein = np.mean(agent.proteins) if agent.proteins else 0
            avg_energy = np.mean(agent.energies) if agent.energies else 0
            print(f"  {i+1}. Lifetime: {agent.lifetime}, "
                  f"avg protein: {avg_protein:.1f}, avg energy: {avg_energy:.1f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Environment simulation visualization")
    parser.add_argument(
        "--video", action="store_true", help="Create a video instead of snapshots"
    )
    parser.add_argument(
        "--steps", type=int, default=200, help="Number of simulation steps"
    )
    parser.add_argument(
        "--fps", type=int, default=20, help="Frames per second for video"
    )
    parser.add_argument(
        "--output", type=str, default="environment_simulation.mp4",
        help="Output filename for video"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--food-components", type=int, default=2,
        help="Number of Gaussian components for food spawning"
    )
    parser.add_argument(
        "--top-agents", type=int, default=10,
        help="Number of top longest-living agents to track"
    )
    args = parser.parse_args()

    if args.video:
        create_video(
            n_steps=args.steps,
            fps=args.fps,
            output_file=args.output,
            seed=args.seed,
            n_food_components=args.food_components,
            n_top_agents=args.top_agents,
        )
    else:
        run_visualization(n_steps=args.steps, save_interval=50)


if __name__ == "__main__":
    main()
