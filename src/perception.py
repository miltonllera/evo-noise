"""
Perception system for cells using a neural network.

Cells perceive their local environment through a 5x5 window centered on their
position. The window is converted to binary food/poison channels and fed
through a small neural network to dynamically compute gene expression parameters.

Architecture:
    5x5 window → Binary channels (food, poison)
    → Flatten: 2 channels × 25 values = 50 features
    → Neural Network: 50 → 16 (ReLU) → 4 (softplus + scaling)
    → GeneExpressionParams
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

try:
    from gillespie import GeneExpressionParams
except ImportError:
    from src.gillespie import GeneExpressionParams


@dataclass
class PerceptionConfig:
    """Configuration for the perception system."""
    window_size: int = 5
    hidden_size: int = 16

    # Output scaling ranges for gene expression parameters
    k_transcription_range: tuple[float, float] = (0.1, 2.0)
    k_translation_range: tuple[float, float] = (0.1, 5.0)
    k_mrna_deg_range: tuple[float, float] = (0.01, 0.5)
    k_protein_deg_range: tuple[float, float] = (0.005, 0.1)

    # Mutation settings
    mutation_rate: float = 0.1  # Probability of mutating each weight
    mutation_std: float = 0.1   # Standard deviation of Gaussian noise


@dataclass
class PerceptionNetwork:
    """
    Neural network weights for perception-to-gene-params mapping.

    Architecture: 50 inputs → 16 hidden (ReLU) → 4 outputs (softplus + scaling)
    Total parameters: (50 × 16) + 16 + (16 × 4) + 4 = 884
    """
    w1: np.ndarray  # Shape: (50, 16) - input to hidden weights
    b1: np.ndarray  # Shape: (16,) - hidden biases
    w2: np.ndarray  # Shape: (16, 4) - hidden to output weights
    b2: np.ndarray  # Shape: (4,) - output biases

    @staticmethod
    def random_init(
        hidden_size: int = 16,
        rng: Optional[np.random.Generator] = None,
    ) -> "PerceptionNetwork":
        """
        Initialize network with Xavier/Glorot initialization.

        Args:
            hidden_size: Number of neurons in hidden layer
            rng: Random number generator

        Returns:
            Randomly initialized PerceptionNetwork
        """
        if rng is None:
            rng = np.random.default_rng()

        input_size = 50  # 2 channels × 25 values (5×5 window)
        output_size = 4  # k_transcription, k_translation, k_mrna_deg, k_protein_deg

        # Xavier initialization: scale by sqrt(2 / (fan_in + fan_out))
        w1_scale = np.sqrt(2.0 / (input_size + hidden_size))
        w2_scale = np.sqrt(2.0 / (hidden_size + output_size))

        return PerceptionNetwork(
            w1=rng.normal(0, w1_scale, (input_size, hidden_size)).astype(np.float32),
            b1=np.zeros(hidden_size, dtype=np.float32),
            w2=rng.normal(0, w2_scale, (hidden_size, output_size)).astype(np.float32),
            b2=np.zeros(output_size, dtype=np.float32),
        )

    def copy(self) -> "PerceptionNetwork":
        """Create a deep copy of this network."""
        return PerceptionNetwork(
            w1=self.w1.copy(),
            b1=self.b1.copy(),
            w2=self.w2.copy(),
            b2=self.b2.copy(),
        )

    def mutate(
        self,
        mutation_rate: float = 0.1,
        mutation_std: float = 0.1,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """
        Apply Gaussian noise mutations to weights in-place.

        Args:
            mutation_rate: Probability of mutating each weight
            mutation_std: Standard deviation of Gaussian noise
            rng: Random number generator
        """
        if rng is None:
            rng = np.random.default_rng()

        for arr in [self.w1, self.b1, self.w2, self.b2]:
            mask = rng.random(arr.shape) < mutation_rate
            noise = rng.normal(0, mutation_std, arr.shape).astype(arr.dtype)
            arr[mask] += noise[mask]


def extract_local_window(
    grid: np.ndarray,
    x: int,
    y: int,
    window_size: int = 5,
) -> np.ndarray:
    """
    Extract a local window from the grid with toroidal wrapping.

    Args:
        grid: 2D environment grid (height, width)
        x: Cell x position (column)
        y: Cell y position (row)
        window_size: Size of the square window (must be odd)

    Returns:
        2D array of shape (window_size, window_size)
    """
    height, width = grid.shape
    half = window_size // 2

    # Generate indices with wrapping
    row_indices = np.arange(y - half, y + half + 1) % height
    col_indices = np.arange(x - half, x + half + 1) % width

    # Extract window using advanced indexing
    return grid[np.ix_(row_indices, col_indices)]


def create_binary_channels(window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Create binary food and poison channels from environment window.

    Args:
        window: Local environment window with TileType values

    Returns:
        Tuple of (food_channel, poison_channel), each as float32 arrays
    """
    # TileType: EMPTY=0, FOOD=1, POISON=2
    food_channel = (window == 1).astype(np.float32)
    poison_channel = (window == 2).astype(np.float32)
    return food_channel, poison_channel


def perceive_environment(
    grid: np.ndarray,
    x: int,
    y: int,
    window_size: int = 5,
) -> np.ndarray:
    """
    Extract perception features from the environment at a cell's position.

    Args:
        grid: 2D environment grid
        x: Cell x position
        y: Cell y position
        window_size: Size of perception window

    Returns:
        1D feature vector of length 50 (2 channels × 25 values)
    """
    window = extract_local_window(grid, x, y, window_size)
    food_channel, poison_channel = create_binary_channels(window)

    features = np.concatenate([
        food_channel.flatten(),
        poison_channel.flatten(),
    ])

    return features.astype(np.float32)


def softplus(x: np.ndarray) -> np.ndarray:
    """Softplus activation: log(1 + exp(x)), numerically stable."""
    return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -20, 20))))


def forward_pass(
    features: np.ndarray,
    network: PerceptionNetwork,
    config: PerceptionConfig,
) -> GeneExpressionParams:
    """
    Run forward pass through perception network to get gene expression params.

    Args:
        features: 36-element feature vector from perceive_environment
        network: PerceptionNetwork with weights
        config: PerceptionConfig with output scaling ranges

    Returns:
        GeneExpressionParams with values scaled to appropriate ranges
    """
    # Hidden layer with ReLU
    hidden = np.maximum(0, features @ network.w1 + network.b1)

    # Output layer with softplus (ensures positive outputs)
    raw_output = hidden @ network.w2 + network.b2
    positive_output = softplus(raw_output)

    # Scale outputs to parameter ranges
    # softplus outputs are in [0, inf), we scale using: min + output / (1 + output) * (max - min)
    # This maps [0, inf) to [min, max)
    def scale_to_range(val: float, range_: tuple[float, float]) -> float:
        min_val, max_val = range_
        # Sigmoid-like scaling: val / (1 + val) maps [0, inf) to [0, 1)
        scaled = val / (1.0 + val)
        return min_val + scaled * (max_val - min_val)

    return GeneExpressionParams(
        k_transcription=scale_to_range(positive_output[0], config.k_transcription_range),
        k_translation=scale_to_range(positive_output[1], config.k_translation_range),
        k_mrna_deg=scale_to_range(positive_output[2], config.k_mrna_deg_range),
        k_protein_deg=scale_to_range(positive_output[3], config.k_protein_deg_range),
    )


class PerceptionSystem:
    """
    High-level interface for cell perception.

    Manages creation, perception, and reproduction of perception networks.
    """

    def __init__(
        self,
        config: Optional[PerceptionConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Initialize the perception system.

        Args:
            config: Perception configuration. Uses defaults if None.
            rng: Random number generator. Uses random seed if None.
        """
        self.config = config or PerceptionConfig()
        self.rng = rng or np.random.default_rng()

    def create_network(self) -> PerceptionNetwork:
        """Create a new randomly initialized perception network."""
        return PerceptionNetwork.random_init(
            hidden_size=self.config.hidden_size,
            rng=self.rng,
        )

    def perceive(
        self,
        grid: np.ndarray,
        x: int,
        y: int,
        network: PerceptionNetwork,
    ) -> GeneExpressionParams:
        """
        Compute gene expression parameters from cell's perception.

        Args:
            grid: Environment grid
            x: Cell x position
            y: Cell y position
            network: Cell's perception network

        Returns:
            GeneExpressionParams computed from local environment
        """
        features = perceive_environment(grid, x, y, self.config.window_size)
        return forward_pass(features, network, self.config)

    def reproduce_network(self, parent: PerceptionNetwork) -> PerceptionNetwork:
        """
        Create offspring network from parent with mutations.

        Args:
            parent: Parent cell's perception network

        Returns:
            New network with inherited (mutated) weights
        """
        offspring = parent.copy()
        offspring.mutate(
            mutation_rate=self.config.mutation_rate,
            mutation_std=self.config.mutation_std,
            rng=self.rng,
        )
        return offspring
