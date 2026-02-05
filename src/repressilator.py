"""
Repressilator ODE model.

A synthetic genetic oscillator consisting of three genes in a cyclic
repression network (Elowitz & Leibler, Nature 2000).

State variables:
    m1, m2, m3 - mRNA concentrations
    p1, p2, p3 - protein concentrations

Repression cycle: p3 -| m1, p1 -| m2, p2 -| m3
"""

import numpy as np
from scipy.integrate import solve_ivp


DEFAULT_PARAMS = {
    "alpha": 216.0,    # maximal transcription rate
    "alpha0": 0.216,   # basal (leaky) transcription rate
    "n": 2.0,          # Hill coefficient
    "beta": 5.0,       # translation rate (protein/mRNA)
    "gamma": 1.0,      # protein degradation rate
}


def repressilator_ode(t, y, params):
    """
    Right-hand side of the repressilator ODEs.

    Args:
        t: Time (unused, system is autonomous)
        y: State vector [m1, m2, m3, p1, p2, p3]
        params: Dict with keys alpha, alpha0, n, beta, gamma

    Returns:
        Derivatives [dm1, dm2, dm3, dp1, dp2, dp3]
    """
    m1, m2, m3, p1, p2, p3 = y
    alpha = params["alpha"]
    alpha0 = params["alpha0"]
    n = params["n"]
    beta = params["beta"]
    gamma = params["gamma"]

    # Hill repression functions
    def hill_repression(p):
        return alpha / (1 + p**n) + alpha0

    # mRNA dynamics: production (repressed) - degradation
    dm1 = hill_repression(p3) - m1
    dm2 = hill_repression(p1) - m2
    dm3 = hill_repression(p2) - m3

    # Protein dynamics: translation - degradation
    dp1 = beta * m1 - gamma * p1
    dp2 = beta * m2 - gamma * p2
    dp3 = beta * m3 - gamma * p3

    return [dm1, dm2, dm3, dp1, dp2, dp3]


def default_initial_conditions():
    """
    Return asymmetric initial conditions that promote oscillations.

    Returns:
        Array [m1, m2, m3, p1, p2, p3]
    """
    return np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])


def simulate(t_span, y0=None, params=None, t_eval=None, **kwargs):
    """
    Run repressilator simulation.

    Args:
        t_span: Tuple (t_start, t_end)
        y0: Initial conditions [m1, m2, m3, p1, p2, p3].
            Defaults to asymmetric conditions if None.
        params: Parameter dict. Uses DEFAULT_PARAMS if None.
        t_eval: Times at which to store solution. If None, solver chooses.
        **kwargs: Additional arguments passed to solve_ivp

    Returns:
        scipy.integrate.OdeResult with attributes:
            t - time points
            y - solution array (6 x len(t))
    """
    if y0 is None:
        y0 = default_initial_conditions()
    if params is None:
        params = DEFAULT_PARAMS

    return solve_ivp(
        fun=lambda t, y: repressilator_ode(t, y, params),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        **kwargs,
    )
