"""Test script for repressilator simulation."""

import sys
sys.path.insert(0, "src")

import matplotlib.pyplot as plt
from repressilator import simulate


def main():
    # Run simulation
    result = simulate(t_span=(0, 100), t_eval=None)

    # Plot protein concentrations
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(result.t, result.y[3], label="p1")
    ax.plot(result.t, result.y[4], label="p2")
    ax.plot(result.t, result.y[5], label="p3")
    ax.set_xlabel("Time")
    ax.set_ylabel("Protein concentration")
    ax.set_title("Repressilator oscillations")
    ax.legend()
    plt.tight_layout()
    plt.savefig("repressilator_test.png", dpi=100)
    print(f"Plot saved to repressilator_test.png")
    print(f"Simulation completed: {len(result.t)} time points")


if __name__ == "__main__":
    main()
