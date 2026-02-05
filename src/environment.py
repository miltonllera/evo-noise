"""
2D environment simulation for cellular organisms.

Simulates a grid-based Petri dish where cells can move and reproduce,
while food and poison randomly spawn at empty locations.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from enum import IntEnum

try:
    from gillespie import GeneExpressionState, GeneExpressionParams, GillespieSimulator
    from action_mapper import (
        ActionMapper,
        DistributionActionMapper,
        ActionTargets,
        default_action_targets,
    )
    from perception import PerceptionConfig, PerceptionNetwork, PerceptionSystem
except ImportError:
    from src.gillespie import GeneExpressionState, GeneExpressionParams, GillespieSimulator
    from src.action_mapper import (
        ActionMapper,
        DistributionActionMapper,
        ActionTargets,
        default_action_targets,
    )
    from src.perception import PerceptionConfig, PerceptionNetwork, PerceptionSystem


class TileType(IntEnum):
    """Types of tiles in the environment grid."""
    EMPTY = 0
    FOOD = 1
    POISON = 2


@dataclass
class FoodDistributionConfig:
    """Configuration for GMM-based food spawning.

    Parameters are provided as separate lists that must have equal length.
    """
    means_x: list[float]  # Center x positions for each component
    means_y: list[float]  # Center y positions for each component
    variances: list[float]  # Variance for each component
    weights: list[float] | None = None  # Relative weights (defaults to uniform)
    total_food_per_step: int = 5  # Expected food tiles to spawn per step

    def __post_init__(self):
        n = len(self.means_x)
        if n == 0:
            raise ValueError("At least one Gaussian component required")
        if len(self.means_y) != n:
            raise ValueError(f"means_y length ({len(self.means_y)}) must match means_x length ({n})")
        if len(self.variances) != n:
            raise ValueError(f"variances length ({len(self.variances)}) must match means_x length ({n})")
        if self.weights is not None and len(self.weights) != n:
            raise ValueError(f"weights length ({len(self.weights)}) must match means_x length ({n})")
        # Default to uniform weights
        if self.weights is None:
            object.__setattr__(self, 'weights', [1.0] * n)

    @property
    def num_components(self) -> int:
        return len(self.means_x)


def create_protein_history(maxlen: int = 50) -> "ProteinHistory":
    """Factory function to create ProteinHistory with custom maxlen."""
    return ProteinHistory(buffer=deque(maxlen=maxlen))


@dataclass
class ProteinHistory:
    """Circular buffer storing protein concentration history."""
    buffer: deque = field(default_factory=lambda: deque(maxlen=50))

    def record(self, protein: int) -> None:
        """Record a protein level to the history buffer."""
        self.buffer.append(protein)

    def as_array(self) -> np.ndarray:
        """Return history as numpy array."""
        return np.array(self.buffer, dtype=np.float32)

    def is_ready(self, min_samples: int = 10) -> bool:
        """Check if enough samples have been collected."""
        return len(self.buffer) >= min_samples

    def get_stats(self) -> tuple[float, float]:
        """Return (mean, std) of the history buffer."""
        if len(self.buffer) == 0:
            return (0.0, 0.0)
        arr = self.as_array()
        return (float(np.mean(arr)), float(np.std(arr)))


@dataclass
class Cell:
    """
    A single-celled organism in the environment.

    Attributes:
        x: Horizontal position on the grid
        y: Vertical position on the grid
        gene_params: Gene expression rate parameters (computed by perception network)
        perception_network: Neural network for environment perception
        energy: Current energy level (dies if <= 0)
        age: Number of timesteps the cell has been alive
        gene_state: Gene expression state (mRNA/protein counts)
        protein_history: Circular buffer of recent protein levels
    """
    x: int
    y: int
    gene_params: GeneExpressionParams
    perception_network: PerceptionNetwork
    energy: float = 100.0
    age: int = 0
    gene_state: GeneExpressionState = field(default_factory=GeneExpressionState)
    protein_history: ProteinHistory = field(default_factory=ProteinHistory)

    def move(self, dx: int, dy: int, grid_width: int, grid_height: int):
        """Move the cell by (dx, dy), wrapping at grid boundaries."""
        self.x = (self.x + dx) % grid_width
        self.y = (self.y + dy) % grid_height

    def get_protein(self) -> int:
        """Return current protein level."""
        return self.gene_state.protein

    def get_mrna(self) -> int:
        """Return current mRNA level."""
        return self.gene_state.mrna


@dataclass
class EnvironmentConfig:
    """Configuration parameters for the environment."""
    width: int = 50
    height: int = 50
    food_spawn_prob: float = 0.01
    poison_spawn_prob: float = 0.005
    food_energy: float = 30.0
    poison_energy: float = -50.0
    move_cost: float = 1.0
    reproduction_threshold: float = 150.0
    reproduction_cost: float = 60.0
    base_metabolism: float = 0.5
    # Gene expression settings
    gene_expression_dt: float = 1.0  # Simulation time per environment step
    use_gene_expression: bool = True  # Enable gene-controlled behavior
    protein_low: float = 50.0  # Low protein threshold for action mapping
    protein_high: float = 200.0  # High protein threshold for action mapping
    # Perception settings
    perception_config: PerceptionConfig | None = None  # Use defaults if None
    # History-aware action mapper settings
    use_distribution_mapper: bool = False  # Use DistributionActionMapper
    history_window_size: int = 50  # Max history buffer size
    history_min_samples: int = 10  # Min samples before distribution matching
    action_targets: ActionTargets | None = None  # Use defaults if None
    # GMM-based food spawning (None = uniform spawning)
    food_distribution: FoodDistributionConfig | None = None
    # Temperature for action sampling (lower = more deterministic)
    action_temperature: float = 1.0


class Environment:
    """
    2D grid environment for cell simulation.

    The environment is a toroidal grid (wraps at edges) containing:
    - Cells: organisms that can move, consume food/poison, and reproduce
    - Food: provides energy when consumed by cells
    - Poison: depletes energy when consumed by cells

    Attributes:
        config: Environment configuration parameters
        grid: 2D array of TileType values
        cells: List of living cells
        rng: Random number generator
        timestep: Current simulation timestep
    """

    def __init__(
        self,
        config: Optional[EnvironmentConfig] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the environment.

        Args:
            config: Configuration parameters. Uses defaults if None.
            seed: Random seed for reproducibility. Uses random seed if None.
        """
        self.config = config or EnvironmentConfig()
        self.rng = np.random.default_rng(seed)
        self.grid = np.zeros(
            (self.config.height, self.config.width), dtype=np.int8
        )
        self.cells: list[Cell] = []
        self.timestep = 0
        # Gene expression components - choose action mapper based on config
        if self.config.use_distribution_mapper:
            action_targets = self.config.action_targets or default_action_targets()
            self.action_mapper: ActionMapper | DistributionActionMapper = DistributionActionMapper(
                action_targets=action_targets,
                min_samples=self.config.history_min_samples,
                temperature=self.config.action_temperature,
                rng=self.rng,
            )
        else:
            self.action_mapper = ActionMapper(
                protein_low=self.config.protein_low,
                protein_high=self.config.protein_high,
                rng=self.rng,
            )
        # Perception system (always enabled - computes gene expression params)
        perception_config = self.config.perception_config or PerceptionConfig()
        self.perception_system = PerceptionSystem(
            config=perception_config,
            rng=self.rng,
        )

    def add_cell(self, x: int, y: int, energy: float = 100.0) -> Cell:
        """Add a new cell at the specified position."""
        perception_network = self.perception_system.create_network()
        gene_params = self.perception_system.perceive(
            self.grid, x, y, perception_network
        )
        protein_history = create_protein_history(self.config.history_window_size)
        cell = Cell(
            x=x,
            y=y,
            gene_params=gene_params,
            perception_network=perception_network,
            energy=energy,
            protein_history=protein_history,
        )
        self.cells.append(cell)
        return cell

    def spawn_random_cells(self, n: int, energy: float = 100.0):
        """Spawn n cells at random positions."""
        positions = self.rng.integers(
            low=[0, 0],
            high=[self.config.width, self.config.height],
            size=(n, 2),
        )
        for x, y in positions:
            self.add_cell(int(x), int(y), energy)

    def _spawn_resources(self):
        """Spawn food and poison at empty grid locations."""
        if self.config.food_distribution:
            self._spawn_food_gmm()
        else:
            self._spawn_food_uniform()
        self._spawn_poison()

    def _spawn_food_uniform(self):
        """Spawn food uniformly at random empty tiles."""
        empty_mask = self.grid == TileType.EMPTY
        rand_vals = self.rng.random(self.grid.shape)
        food_mask = empty_mask & (rand_vals < self.config.food_spawn_prob)
        self.grid[food_mask] = TileType.FOOD

    def _spawn_food_gmm(self):
        """Spawn food according to Gaussian Mixture Model."""
        fd = self.config.food_distribution
        total_weight = sum(fd.weights)

        # Compute probability density at each grid point
        density = np.zeros((self.config.height, self.config.width))
        y_coords, x_coords = np.mgrid[0:self.config.height, 0:self.config.width]

        for i in range(fd.num_components):
            # Gaussian density for this component
            dx = x_coords - fd.means_x[i]
            dy = y_coords - fd.means_y[i]
            exponent = -(dx**2 + dy**2) / (2 * fd.variances[i])
            density += (fd.weights[i] / total_weight) * np.exp(exponent)

        # Normalize to get probabilities
        density /= density.sum()

        # Sample food locations
        empty_mask = self.grid == TileType.EMPTY
        probs = density * empty_mask  # Zero out non-empty tiles
        if probs.sum() > 0:
            probs /= probs.sum()

            # Sample positions
            flat_probs = probs.flatten()
            n_food = self.rng.poisson(fd.total_food_per_step)
            if n_food > 0:
                indices = self.rng.choice(
                    len(flat_probs),
                    size=min(n_food, int(empty_mask.sum())),
                    p=flat_probs,
                    replace=False,
                )
                for idx in indices:
                    y, x = divmod(idx, self.config.width)
                    self.grid[y, x] = TileType.FOOD

    def _spawn_poison(self):
        """Spawn poison at random empty tiles."""
        empty_mask = self.grid == TileType.EMPTY
        rand_vals = self.rng.random(self.grid.shape)
        poison_mask = empty_mask & (rand_vals < self.config.poison_spawn_prob)
        self.grid[poison_mask] = TileType.POISON

    def _move_cell(self, cell: Cell):
        """
        Move a cell based on its gene expression state.

        Movement costs energy. With gene expression enabled, protein
        levels influence movement. With distribution mapper, uses
        protein history statistics for action selection.
        """
        if self.config.use_gene_expression:
            if self.config.use_distribution_mapper:
                dx, dy = self.action_mapper.get_movement(cell.protein_history)
            else:
                dx, dy = self.action_mapper.get_movement(cell.get_protein())
        else:
            dx = self.rng.integers(-1, 2)
            dy = self.rng.integers(-1, 2)
        cell.move(dx, dy, self.config.width, self.config.height)
        cell.energy -= self.config.move_cost

    def _cell_consume(self, cell: Cell):
        """Have a cell consume whatever is at its current position."""
        tile = self.grid[cell.y, cell.x]
        if tile == TileType.FOOD:
            cell.energy += self.config.food_energy
            self.grid[cell.y, cell.x] = TileType.EMPTY
        elif tile == TileType.POISON:
            cell.energy += self.config.poison_energy
            self.grid[cell.y, cell.x] = TileType.EMPTY

    def _cell_reproduce(self, cell: Cell) -> Optional[Cell]:
        """
        Attempt cell reproduction.

        A cell reproduces if its energy exceeds the reproduction threshold.
        With gene expression, protein levels can modify the threshold.
        With distribution mapper, uses protein history statistics.
        Offspring inherit mutated perception networks; gene_params are
        computed from the offspring's perception at its spawn location.
        Offspring start with fresh (empty) protein history.

        Returns:
            New Cell if reproduction occurred, None otherwise.
        """
        if self.config.use_gene_expression:
            if self.config.use_distribution_mapper:
                should_reproduce = self.action_mapper.should_reproduce(
                    cell.protein_history,
                    cell.energy,
                    self.config.reproduction_threshold,
                )
            else:
                should_reproduce = self.action_mapper.should_reproduce(
                    cell.energy,
                    cell.get_protein(),
                    self.config.reproduction_threshold,
                )
            if not should_reproduce:
                return None
        else:
            if cell.energy < self.config.reproduction_threshold:
                return None

        cell.energy -= self.config.reproduction_cost
        offspring_energy = self.config.reproduction_cost / 2

        # Offspring appears at a random adjacent position
        dx = self.rng.integers(-1, 2)
        dy = self.rng.integers(-1, 2)
        ox = (cell.x + dx) % self.config.width
        oy = (cell.y + dy) % self.config.height

        # Offspring inherits perception network with mutations
        offspring_network = self.perception_system.reproduce_network(
            cell.perception_network
        )

        # Compute gene params from offspring's perception at spawn location
        offspring_params = self.perception_system.perceive(
            self.grid, ox, oy, offspring_network
        )

        # Offspring starts with fresh protein history
        offspring_history = create_protein_history(self.config.history_window_size)

        return Cell(
            x=ox,
            y=oy,
            gene_params=offspring_params,
            perception_network=offspring_network,
            energy=offspring_energy,
            protein_history=offspring_history,
        )

    def _apply_metabolism(self, cell: Cell):
        """Apply base metabolic cost to a cell."""
        cell.energy -= self.config.base_metabolism
        cell.age += 1

    def _update_gene_params_from_perception(self, cell: Cell):
        """
        Update cell's gene expression parameters based on perception.

        Uses the cell's perception network to compute gene expression
        parameters from the local environment (food/poison gradients).
        """
        cell.gene_params = self.perception_system.perceive(
            self.grid,
            cell.x,
            cell.y,
            cell.perception_network,
        )

    def _update_gene_expression(self, cell: Cell):
        """Advance cell's gene expression simulation."""
        if not self.config.use_gene_expression:
            return
        simulator = GillespieSimulator(params=cell.gene_params, rng=self.rng)
        target_time = cell.gene_state.time + self.config.gene_expression_dt
        simulator.simulate_until(cell.gene_state, target_time)

    def step(self):
        """
        Advance the simulation by one timestep.

        Order of operations:
        1. Spawn new food and poison
        2. Each cell: update gene params from perception, update gene expression,
           move, consume, reproduce
        3. Apply metabolism to all cells
        4. Remove dead cells (energy <= 0)
        """
        self._spawn_resources()

        new_cells = []
        for cell in self.cells:
            self._update_gene_params_from_perception(cell)
            self._update_gene_expression(cell)
            # Record protein level to history after gene expression update
            cell.protein_history.record(cell.get_protein())
            self._move_cell(cell)
            self._cell_consume(cell)
            offspring = self._cell_reproduce(cell)
            if offspring:
                new_cells.append(offspring)
            self._apply_metabolism(cell)

        self.cells.extend(new_cells)

        # Remove dead cells
        self.cells = [c for c in self.cells if c.energy > 0]

        self.timestep += 1

    def run(self, n_steps: int):
        """Run the simulation for n_steps timesteps."""
        for _ in range(n_steps):
            self.step()

    def get_cell_positions(self) -> np.ndarray:
        """Return array of shape (n_cells, 2) with [x, y] positions."""
        if not self.cells:
            return np.empty((0, 2), dtype=np.int32)
        return np.array([[c.x, c.y] for c in self.cells], dtype=np.int32)

    def get_cell_energies(self) -> np.ndarray:
        """Return array of cell energies."""
        return np.array([c.energy for c in self.cells])

    def get_cell_proteins(self) -> np.ndarray:
        """Return array of cell protein levels."""
        return np.array([c.get_protein() for c in self.cells])

    def count_resources(self) -> dict[str, int]:
        """Count food and poison tiles in the environment."""
        return {
            "food": int(np.sum(self.grid == TileType.FOOD)),
            "poison": int(np.sum(self.grid == TileType.POISON)),
        }

    def get_state(self) -> dict:
        """
        Return a snapshot of the current environment state.

        Returns:
            Dict with grid, cell positions, energies, proteins, and resource counts.
        """
        return {
            "timestep": self.timestep,
            "grid": self.grid.copy(),
            "cell_positions": self.get_cell_positions(),
            "cell_energies": self.get_cell_energies(),
            "cell_proteins": self.get_cell_proteins(),
            "n_cells": len(self.cells),
            **self.count_resources(),
        }
