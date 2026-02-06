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
from tqdm import tqdm

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


@dataclass
class LifespanTracker:
    """Track agent lifespan distributions over simulation."""
    snapshot_ages: dict[int, list[int]] = field(default_factory=dict)  # timestep -> ages at that timestep
    death_ages: list[tuple[int, int]] = field(default_factory=list)  # (timestep, age) for each death
    snapshot_interval: int = 10  # take snapshot every N steps

    def record_snapshot(self, timestep: int, ages: np.ndarray):
        """Record ages of all living cells at this timestep."""
        if timestep % self.snapshot_interval == 0:
            self.snapshot_ages[timestep] = ages.tolist()

    def record_deaths(self, timestep: int, ages: list[int]):
        """Record ages of cells that just died."""
        for age in ages:
            self.death_ages.append((timestep, age))

    def get_death_ages_up_to(self, timestep: int) -> list[int]:
        """Get all death ages up to (and including) a given timestep."""
        return [age for t, age in self.death_ages if t <= timestep]

    def save_to_numpy(self, filepath: str):
        """Save lifespan data to .npz file."""
        snapshot_timesteps = np.array(sorted(self.snapshot_ages.keys()))

        # Store snapshot ages as object array (variable-length lists)
        snapshot_ages_list = [self.snapshot_ages[t] for t in snapshot_timesteps]

        # Death ages with timesteps
        death_data = np.array(self.death_ages, dtype=np.int32) if self.death_ages else np.empty((0, 2), dtype=np.int32)

        np.savez(
            filepath,
            snapshot_timesteps=snapshot_timesteps,
            snapshot_ages=np.array(snapshot_ages_list, dtype=object),
            death_timesteps=death_data[:, 0] if len(death_data) > 0 else np.array([], dtype=np.int32),
            death_ages=death_data[:, 1] if len(death_data) > 0 else np.array([], dtype=np.int32),
            snapshot_interval=self.snapshot_interval,
        )
        print(f"Saved lifespan data to {filepath}")
        print(f"  Snapshots: {len(snapshot_timesteps)}, Deaths recorded: {len(self.death_ages)}")


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


def create_lifespan_animation(
    tracker: LifespanTracker,
    output_path: str,
    fps: int = 5,
):
    """
    Create animated GIF showing age distribution evolution.

    Left panel: Living ages histogram (updates each frame)
    Right panel: Death ages histogram (accumulates over time)
    """
    if not tracker.snapshot_ages:
        print("No lifespan data to animate")
        return

    sorted_timesteps = sorted(tracker.snapshot_ages.keys())

    # Determine histogram bounds
    max_living_age = max(max(ages) if ages else 0 for ages in tracker.snapshot_ages.values())
    max_death_age = max((age for _, age in tracker.death_ages), default=0)
    max_age = max(max_living_age, max_death_age, 1)
    bins = np.linspace(0, max_age + 1, min(50, max_age + 2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_living, ax_death = axes

    def update(frame_idx):
        timestep = sorted_timesteps[frame_idx]
        living_ages = tracker.snapshot_ages[timestep]
        death_ages = tracker.get_death_ages_up_to(timestep)

        ax_living.clear()
        ax_death.clear()

        # Living ages histogram
        if living_ages:
            ax_living.hist(living_ages, bins=bins, color="steelblue", edgecolor="black", alpha=0.7)
        ax_living.set_xlabel("Age (timesteps)")
        ax_living.set_ylabel("Count")
        ax_living.set_title(f"Living Cell Ages at t={timestep}")
        ax_living.set_xlim(0, max_age + 1)

        # Death ages histogram (cumulative)
        if death_ages:
            ax_death.hist(death_ages, bins=bins, color="coral", edgecolor="black", alpha=0.7)
        ax_death.set_xlabel("Age at Death (timesteps)")
        ax_death.set_ylabel("Count")
        ax_death.set_title(f"Lifespan Distribution (deaths up to t={timestep})")
        ax_death.set_xlim(0, max_age + 1)

        fig.suptitle(f"Agent Lifespan Analysis - Step {timestep}", fontsize=12, fontweight="bold")
        plt.tight_layout()

        return []

    print(f"Creating lifespan animation with {len(sorted_timesteps)} frames...")
    anim = FuncAnimation(
        fig,
        update,
        frames=len(sorted_timesteps),
        interval=1000 // fps,
        blit=False,
    )

    anim.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"Saved lifespan animation to {output_path}")


def create_lifespan_summary_plot(tracker: LifespanTracker, output_path: str):
    """Create a static summary plot of lifespan distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Get final snapshot and all death ages
    sorted_timesteps = sorted(tracker.snapshot_ages.keys())
    if not sorted_timesteps:
        print("No lifespan data to plot")
        plt.close(fig)
        return

    final_timestep = sorted_timesteps[-1]
    final_living_ages = tracker.snapshot_ages[final_timestep]
    all_death_ages = [age for _, age in tracker.death_ages]

    # Top-left: Final living ages histogram
    ax = axes[0, 0]
    if final_living_ages:
        ax.hist(final_living_ages, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
    ax.set_xlabel("Age (timesteps)")
    ax.set_ylabel("Count")
    ax.set_title(f"Living Cell Ages at Final Step (t={final_timestep})")

    # Top-right: All death ages histogram
    ax = axes[0, 1]
    if all_death_ages:
        ax.hist(all_death_ages, bins=30, color="coral", edgecolor="black", alpha=0.7)
        median_lifespan = np.median(all_death_ages)
        mean_lifespan = np.mean(all_death_ages)
        ax.axvline(median_lifespan, color="red", linestyle="--", label=f"Median: {median_lifespan:.1f}")
        ax.axvline(mean_lifespan, color="darkred", linestyle=":", label=f"Mean: {mean_lifespan:.1f}")
        ax.legend()
    ax.set_xlabel("Age at Death (timesteps)")
    ax.set_ylabel("Count")
    ax.set_title("Complete Lifespan Distribution")

    # Bottom-left: Mean living age over time
    ax = axes[1, 0]
    mean_ages = []
    for t in sorted_timesteps:
        ages = tracker.snapshot_ages[t]
        mean_ages.append(np.mean(ages) if ages else 0)
    ax.plot(sorted_timesteps, mean_ages, "b-", linewidth=2)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean Age")
    ax.set_title("Mean Living Cell Age Over Time")
    ax.grid(True, alpha=0.3)

    # Bottom-right: Population count over time
    ax = axes[1, 1]
    pop_counts = [len(tracker.snapshot_ages[t]) for t in sorted_timesteps]
    ax.plot(sorted_timesteps, pop_counts, "g-", linewidth=2)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Population")
    ax.set_title("Population Over Time (at snapshot intervals)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved lifespan summary plot to {output_path}")


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
    for step in tqdm(range(n_steps), desc="Simulating"):
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
    track_lifespan: bool = False,
    lifespan_output: str | None = None,
    lifespan_data: str | None = None,
    lifespan_snapshot_interval: int = 10,
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
        track_lifespan: Enable lifespan distribution tracking
        lifespan_output: Output path for lifespan animation GIF
        lifespan_data: Output path for lifespan data .npz file
        lifespan_snapshot_interval: Take age snapshot every N steps
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
    cell_ages: dict[int, int] = {}  # cell_id -> age at last record (for death tracking)
    for cell in env.cells:
        cell_id = id(cell)
        tracker.register_agent(cell_id)
        living_cell_ids.add(cell_id)
        cell_ages[cell_id] = cell.age

    # Initialize lifespan tracker if enabled
    lifespan_tracker: LifespanTracker | None = None
    if track_lifespan:
        lifespan_tracker = LifespanTracker(snapshot_interval=lifespan_snapshot_interval)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Pre-run simulation and store states for smooth animation
    states = []
    for step in tqdm(range(n_steps), desc="Simulating"):
        # Record state for all living cells before step
        for cell in env.cells:
            tracker.record_step(id(cell), cell.get_protein(), cell.energy)
            cell_ages[id(cell)] = cell.age

        # Record lifespan snapshot if enabled
        if lifespan_tracker is not None:
            ages = env.get_cell_ages()
            lifespan_tracker.record_snapshot(env.timestep, ages)

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
        dead_ids = living_cell_ids - current_cell_ids
        for dead_id in dead_ids:
            tracker.agent_died(dead_id)

        # Record death ages for lifespan tracking
        if lifespan_tracker is not None and dead_ids:
            death_ages_this_step = [cell_ages[dead_id] for dead_id in dead_ids if dead_id in cell_ages]
            lifespan_tracker.record_deaths(env.timestep, death_ages_this_step)
            # Clean up cell_ages for dead cells
            for dead_id in dead_ids:
                cell_ages.pop(dead_id, None)

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

    # Save and plot lifespan data if tracking was enabled
    if lifespan_tracker is not None:
        # Determine output paths
        lifespan_gif_path = lifespan_output or f"{base_name}_lifespan.gif"
        lifespan_npz_path = lifespan_data or f"{base_name}_lifespan_data.npz"
        lifespan_summary_path = f"{base_name}_lifespan_summary.png"

        # Save data
        lifespan_tracker.save_to_numpy(lifespan_npz_path)

        # Create animated histogram
        create_lifespan_animation(lifespan_tracker, lifespan_gif_path, fps=5)

        # Create static summary plot
        create_lifespan_summary_plot(lifespan_tracker, lifespan_summary_path)

        # Print lifespan statistics
        all_death_ages = [age for _, age in lifespan_tracker.death_ages]
        if all_death_ages:
            print(f"\nLifespan statistics:")
            print(f"  Total deaths recorded: {len(all_death_ages)}")
            print(f"  Mean lifespan: {np.mean(all_death_ages):.1f}")
            print(f"  Median lifespan: {np.median(all_death_ages):.1f}")
            print(f"  Max lifespan: {max(all_death_ages)}")
            print(f"  Min lifespan: {min(all_death_ages)}")


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
    parser.add_argument(
        "--track-lifespan", action="store_true",
        help="Enable lifespan distribution tracking"
    )
    parser.add_argument(
        "--lifespan-output", type=str, default=None,
        help="Output path for lifespan animation GIF (default: <output>_lifespan.gif)"
    )
    parser.add_argument(
        "--lifespan-data", type=str, default=None,
        help="Output path for lifespan data .npz file (default: <output>_lifespan_data.npz)"
    )
    parser.add_argument(
        "--lifespan-interval", type=int, default=10,
        help="Take age snapshot every N steps (default: 10)"
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
            track_lifespan=args.track_lifespan,
            lifespan_output=args.lifespan_output,
            lifespan_data=args.lifespan_data,
            lifespan_snapshot_interval=args.lifespan_interval,
        )
    else:
        run_visualization(n_steps=args.steps, save_interval=50)


if __name__ == "__main__":
    main()
