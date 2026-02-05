"""
Action mapping from gene expression to cell behaviors.

Translates protein levels into action probabilities for movement,
reproduction, and other cell behaviors.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from environment import ProteinHistory


def sigmoid(x: float, center: float, steepness: float = 1.0) -> float:
    """
    Sigmoid function for smooth threshold transitions.

    Args:
        x: Input value
        center: Center point (where output = 0.5)
        steepness: Transition sharpness

    Returns:
        Value in [0, 1]
    """
    return 1.0 / (1.0 + np.exp(-steepness * (x - center)))


def normalize_protein(
    protein: int,
    low: float,
    high: float,
) -> float:
    """
    Normalize protein level to [0, 1] range.

    Args:
        protein: Current protein count
        low: Protein level mapping to 0
        high: Protein level mapping to 1

    Returns:
        Normalized value clipped to [0, 1]
    """
    if high <= low:
        return 0.5
    normalized = (protein - low) / (high - low)
    return float(np.clip(normalized, 0.0, 1.0))


@dataclass
class TargetDistribution:
    """Target (mean, std) pair for an action."""
    mean: float
    std: float


@dataclass
class ActionTargets:
    """Target distributions for each possible action (hyperparameters)."""
    move_up: TargetDistribution
    move_down: TargetDistribution
    move_left: TargetDistribution
    move_right: TargetDistribution
    stay: TargetDistribution


def default_action_targets() -> ActionTargets:
    """Create default action targets."""
    return ActionTargets(
        move_up=TargetDistribution(mean=100.0, std=50.0),
        move_down=TargetDistribution(mean=100.0, std=50.0),
        move_left=TargetDistribution(mean=100.0, std=50.0),
        move_right=TargetDistribution(mean=100.0, std=50.0),
        stay=TargetDistribution(mean=200.0, std=20.0),
    )


def compute_distance(
    observed_mean: float,
    observed_std: float,
    target: TargetDistribution,
) -> float:
    """
    Compute distance between observed distribution and target.

    Uses normalized Euclidean distance in (mean, std) space.
    Lower distance = better match.

    Args:
        observed_mean: Observed mean from protein history
        observed_std: Observed std from protein history
        target: Target distribution to compare against

    Returns:
        Distance value (lower is better match)
    """
    mean_diff = (observed_mean - target.mean) / target.mean if target.mean > 0 else observed_mean
    std_diff = (observed_std - target.std) / target.std if target.std > 0 else observed_std
    return np.sqrt(mean_diff**2 + std_diff**2)


def select_best_action(
    observed_mean: float,
    observed_std: float,
    action_targets: dict[str, TargetDistribution],
) -> str:
    """
    Select action whose target distribution best matches observed.

    Args:
        observed_mean: Observed mean from protein history
        observed_std: Observed std from protein history
        action_targets: Dictionary mapping action names to targets

    Returns:
        Name of the action with the best (lowest distance) match
    """
    distances = {
        action: compute_distance(observed_mean, observed_std, target)
        for action, target in action_targets.items()
    }
    return min(distances, key=lambda k: distances[k])


def select_action_by_sampling(
    observed_mean: float,
    observed_std: float,
    action_targets: dict[str, TargetDistribution],
    rng: np.random.Generator,
    temperature: float = 1.0,
) -> str:
    """
    Sample action probabilistically based on distance to targets.

    Uses softmax on negative distances: closer targets have higher probability.
    Temperature controls randomness (lower = more deterministic).

    Args:
        observed_mean: Observed mean from protein history
        observed_std: Observed std from protein history
        action_targets: Dictionary mapping action names to targets
        rng: Random number generator
        temperature: Controls randomness (lower = more deterministic)

    Returns:
        Sampled action name
    """
    actions = list(action_targets.keys())
    distances = np.array([
        compute_distance(observed_mean, observed_std, action_targets[a])
        for a in actions
    ])

    # Softmax on negative distances (closer = higher probability)
    logits = -distances / temperature
    logits -= logits.max()  # Numerical stability
    probs = np.exp(logits)
    probs /= probs.sum()

    return rng.choice(actions, p=probs)


class ActionMapper:
    """
    Maps gene expression state to cell actions.

    Provides methods to translate protein levels into movement
    directions, reproduction decisions, and other behaviors.
    """

    def __init__(
        self,
        protein_low: float = 50.0,
        protein_high: float = 200.0,
        rng: np.random.Generator | None = None,
    ):
        """
        Initialize action mapper.

        Args:
            protein_low: Protein level considered "low"
            protein_high: Protein level considered "high"
            rng: Random number generator
        """
        self.protein_low = protein_low
        self.protein_high = protein_high
        self.rng = rng or np.random.default_rng()

    def get_movement(
        self,
        protein: int,
        bias_strength: float = 0.5,
    ) -> Tuple[int, int]:
        """
        Get movement direction based on protein level.

        Low protein: random movement
        High protein: biased toward staying (conserve energy)

        Args:
            protein: Current protein count
            bias_strength: How strongly high protein biases toward staying

        Returns:
            (dx, dy) movement direction
        """
        # Compute probability of staying based on protein level
        p_norm = normalize_protein(protein, self.protein_low, self.protein_high)
        p_stay = p_norm * bias_strength

        if self.rng.random() < p_stay:
            return (0, 0)

        # Random movement
        dx = self.rng.integers(-1, 2)
        dy = self.rng.integers(-1, 2)
        return (int(dx), int(dy))

    def get_directional_movement(
        self,
        protein: int,
        target_dx: int,
        target_dy: int,
        bias_strength: float = 0.5,
    ) -> Tuple[int, int]:
        """
        Get movement with directional bias based on protein level.

        Low protein: random movement
        High protein: biased toward target direction

        Args:
            protein: Current protein count
            target_dx: Target x direction (-1, 0, or 1)
            target_dy: Target y direction (-1, 0, or 1)
            bias_strength: How strongly high protein biases toward target

        Returns:
            (dx, dy) movement direction
        """
        p_norm = normalize_protein(protein, self.protein_low, self.protein_high)
        p_follow = p_norm * bias_strength

        if self.rng.random() < p_follow:
            return (target_dx, target_dy)

        # Random movement
        dx = self.rng.integers(-1, 2)
        dy = self.rng.integers(-1, 2)
        return (int(dx), int(dy))

    def get_reproduction_threshold_modifier(
        self,
        protein: int,
        max_reduction: float = 0.2,
    ) -> float:
        """
        Get reproduction threshold modifier based on protein level.

        High protein can reduce the energy threshold for reproduction.

        Args:
            protein: Current protein count
            max_reduction: Maximum threshold reduction (as fraction)

        Returns:
            Multiplier for reproduction threshold (1.0 - max_reduction to 1.0)
        """
        p_norm = normalize_protein(protein, self.protein_low, self.protein_high)
        return 1.0 - (p_norm * max_reduction)

    def should_reproduce(
        self,
        energy: float,
        protein: int,
        base_threshold: float,
        max_reduction: float = 0.2,
    ) -> bool:
        """
        Determine if cell should reproduce.

        Args:
            energy: Cell's current energy
            protein: Current protein count
            base_threshold: Base energy threshold
            max_reduction: Max threshold reduction from high protein

        Returns:
            True if cell should reproduce
        """
        modifier = self.get_reproduction_threshold_modifier(protein, max_reduction)
        return energy >= base_threshold * modifier

    def get_activity_level(self, protein: int) -> float:
        """
        Get general activity level from protein.

        Returns:
            Activity level in [0, 1]
        """
        return normalize_protein(protein, self.protein_low, self.protein_high)


# Movement direction mapping
MOVEMENT_DIRECTIONS: dict[str, Tuple[int, int]] = {
    "move_up": (0, -1),
    "move_down": (0, 1),
    "move_left": (-1, 0),
    "move_right": (1, 0),
    "stay": (0, 0),
}


class DistributionActionMapper:
    """
    Maps gene expression history to cell actions using distribution matching.

    Actions are selected by comparing the observed (mean, std) of protein
    concentration history against target distributions for each action.
    Uses probabilistic sampling based on distances to targets.
    """

    def __init__(
        self,
        action_targets: ActionTargets | None = None,
        min_samples: int = 10,
        temperature: float = 1.0,
        rng: np.random.Generator | None = None,
    ):
        """
        Initialize distribution action mapper.

        Args:
            action_targets: Target distributions for each action
            min_samples: Minimum history samples before using distribution matching
            temperature: Controls action sampling randomness (lower = more deterministic)
            rng: Random number generator for fallback behavior
        """
        self.action_targets = action_targets or default_action_targets()
        self.min_samples = min_samples
        self.temperature = temperature
        self.rng = rng or np.random.default_rng()

    def get_movement(self, history: ProteinHistory) -> Tuple[int, int]:
        """
        Get movement direction based on protein history distribution.

        If history is not ready (< min_samples), falls back to random movement.
        Otherwise, samples action probabilistically based on distance to targets.

        Args:
            history: Protein concentration history

        Returns:
            (dx, dy) movement direction
        """
        if not history.is_ready(self.min_samples):
            # Fallback: random movement
            dx = self.rng.integers(-1, 2)
            dy = self.rng.integers(-1, 2)
            return (int(dx), int(dy))

        observed_mean, observed_std = history.get_stats()

        # Build movement targets dict
        movement_targets = {
            "move_up": self.action_targets.move_up,
            "move_down": self.action_targets.move_down,
            "move_left": self.action_targets.move_left,
            "move_right": self.action_targets.move_right,
            "stay": self.action_targets.stay,
        }

        best_action = select_action_by_sampling(
            observed_mean, observed_std, movement_targets, self.rng, self.temperature
        )
        return MOVEMENT_DIRECTIONS[best_action]

    def should_reproduce(
        self,
        history: ProteinHistory,
        energy: float,
        energy_threshold: float,
    ) -> bool:
        """
        Determine if cell should reproduce based on energy threshold.

        Args:
            history: Protein concentration history (unused, kept for API compatibility)
            energy: Cell's current energy
            energy_threshold: Minimum energy required

        Returns:
            True if cell should reproduce
        """
        return energy >= energy_threshold
