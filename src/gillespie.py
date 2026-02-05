"""
Gillespie stochastic simulation for constitutive gene expression.

Implements the direct method (Gillespie, 1977) for exact stochastic
simulation of a simple gene expression system with:
- Transcription: DNA → mRNA
- Translation: mRNA → Protein
- mRNA degradation
- Protein degradation
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class GeneExpressionParams:
    """
    Parameters for constitutive gene expression.

    All parameters are required - they are computed by the perception
    neural network based on the cell's local environment.

    Attributes:
        k_transcription: Transcription rate (mRNA produced per unit time)
        k_translation: Translation rate (protein per mRNA per unit time)
        k_mrna_deg: mRNA degradation rate (1/time)
        k_protein_deg: Protein degradation rate (1/time)
    """
    k_transcription: float
    k_translation: float
    k_mrna_deg: float
    k_protein_deg: float


@dataclass
class GeneExpressionState:
    """
    State of a gene expression system.

    Attributes:
        mrna: Integer count of mRNA molecules
        protein: Integer count of protein molecules
        time: Current simulation time
    """
    mrna: int = 0
    protein: int = 0
    time: float = 0.0


class GillespieSimulator:
    """
    Gillespie direct method simulator for gene expression.

    Reactions:
        0. Transcription: ∅ → mRNA  (rate = k_transcription)
        1. Translation: mRNA → mRNA + Protein  (rate = k_translation * mrna)
        2. mRNA degradation: mRNA → ∅  (rate = k_mrna_deg * mrna)
        3. Protein degradation: Protein → ∅  (rate = k_protein_deg * protein)
    """

    def __init__(
        self,
        params: GeneExpressionParams,
        rng: Optional[np.random.Generator] = None,
    ):
        self.params = params
        self.rng = rng or np.random.default_rng()

    def compute_propensities(self, state: GeneExpressionState) -> np.ndarray:
        """Compute reaction propensities for current state."""
        return np.array([
            self.params.k_transcription,
            self.params.k_translation * state.mrna,
            self.params.k_mrna_deg * state.mrna,
            self.params.k_protein_deg * state.protein,
        ])

    def apply_reaction(self, state: GeneExpressionState, reaction_idx: int):
        """Apply a reaction to the state (in-place)."""
        if reaction_idx == 0:  # Transcription
            state.mrna += 1
        elif reaction_idx == 1:  # Translation
            state.protein += 1
        elif reaction_idx == 2:  # mRNA degradation
            state.mrna = max(0, state.mrna - 1)
        elif reaction_idx == 3:  # Protein degradation
            state.protein = max(0, state.protein - 1)

    def step(self, state: GeneExpressionState) -> float:
        """
        Execute one Gillespie step.

        Args:
            state: Current state (modified in place)

        Returns:
            Time elapsed (dt)
        """
        propensities = self.compute_propensities(state)
        total_propensity = propensities.sum()

        if total_propensity == 0:
            return float('inf')

        # Time to next reaction (exponential distribution)
        dt = self.rng.exponential(1.0 / total_propensity)

        # Choose reaction proportional to propensities
        reaction_idx = self.rng.choice(4, p=propensities / total_propensity)

        self.apply_reaction(state, reaction_idx)
        state.time += dt

        return dt

    def simulate_until(
        self,
        state: GeneExpressionState,
        target_time: float,
        max_reactions: int = 10000,
    ) -> int:
        """
        Run simulation until target time.

        Args:
            state: Current state (modified in place)
            target_time: Time to simulate until
            max_reactions: Safety limit on reactions

        Returns:
            Number of reactions executed
        """
        n_reactions = 0
        while state.time < target_time and n_reactions < max_reactions:
            dt = self.step(state)
            n_reactions += 1
            if dt == float('inf'):
                state.time = target_time
                break
        return n_reactions

    def simulate(
        self,
        t_max: float,
        state: Optional[GeneExpressionState] = None,
        record_interval: Optional[float] = None,
    ) -> dict:
        """
        Run simulation and record trajectory.

        Args:
            t_max: Total simulation time
            state: Initial state (creates default if None)
            record_interval: If set, record state at this interval

        Returns:
            Dict with 'times', 'mrna', 'protein' arrays
        """
        if state is None:
            state = GeneExpressionState()

        times = [state.time]
        mrna = [state.mrna]
        protein = [state.protein]

        if record_interval is None:
            # Record every reaction
            while state.time < t_max:
                dt = self.step(state)
                if dt == float('inf'):
                    break
                times.append(state.time)
                mrna.append(state.mrna)
                protein.append(state.protein)
        else:
            # Record at fixed intervals
            next_record = record_interval
            while state.time < t_max:
                self.simulate_until(state, min(next_record, t_max))
                times.append(state.time)
                mrna.append(state.mrna)
                protein.append(state.protein)
                next_record += record_interval

        return {
            'times': np.array(times),
            'mrna': np.array(mrna),
            'protein': np.array(protein),
        }


def expected_steady_state(params: GeneExpressionParams) -> dict:
    """
    Calculate expected steady-state molecule counts.

    Returns:
        Dict with 'mrna' and 'protein' expected values
    """
    mrna_ss = params.k_transcription / params.k_mrna_deg
    protein_ss = (params.k_transcription * params.k_translation) / (
        params.k_mrna_deg * params.k_protein_deg
    )
    return {'mrna': mrna_ss, 'protein': protein_ss}
