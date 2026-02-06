#!/usr/bin/env python3
"""
Profiling script for evo-noise simulation.

Benchmarks key performance-critical functions with configurable parameters.
Supports micro-benchmarks, scaling analysis, and full cProfile profiling.

Usage:
    uv run python scripts/profile_simulation.py                    # Run all benchmarks
    uv run python scripts/profile_simulation.py --benchmark step   # Specific benchmark
    uv run python scripts/profile_simulation.py --scaling cells    # Scaling analysis
    uv run python scripts/profile_simulation.py --full-profile     # Full cProfile
"""

import argparse
import cProfile
import json
import pstats
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from environment import Environment, EnvironmentConfig, FoodDistributionConfig
from gillespie import GeneExpressionParams, GeneExpressionState, GillespieSimulator
from perception import (
    PerceptionConfig,
    PerceptionNetwork,
    forward_pass,
    perceive_environment,
)
from action_mapper import DistributionActionMapper, default_action_targets
from environment import create_protein_history


# ============================================================================
# Benchmark Utilities
# ============================================================================


def timeit(func, *args, n_runs: int = 10, warmup: int = 2, **kwargs) -> dict:
    """
    Time a function over multiple runs, return stats.

    Args:
        func: Function to benchmark
        *args: Positional arguments for func
        n_runs: Number of timed runs
        warmup: Number of warmup runs (not timed)
        **kwargs: Keyword arguments for func

    Returns:
        Dict with mean_ms, std_ms, min_ms, max_ms, calls_per_sec
    """
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        times.append(time.perf_counter() - start)

    times = np.array(times)
    mean_s = np.mean(times)

    return {
        "mean_ms": mean_s * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
        "max_ms": np.max(times) * 1000,
        "calls_per_sec": 1.0 / mean_s if mean_s > 0 else float("inf"),
    }


def format_table(headers: list[str], rows: list[list[str]], col_widths: list[int] | None = None) -> str:
    """Format data as a simple ASCII table."""
    if col_widths is None:
        col_widths = [max(len(str(headers[i])), max(len(str(row[i])) for row in rows)) + 2 for i in range(len(headers))]

    # Header
    header_line = "|" + "|".join(str(h).center(w) for h, w in zip(headers, col_widths)) + "|"
    separator = "+" + "+".join("-" * w for w in col_widths) + "+"

    lines = [separator, header_line, separator]
    for row in rows:
        row_line = "|" + "|".join(str(v).center(w) for v, w in zip(row, col_widths)) + "|"
        lines.append(row_line)
    lines.append(separator)

    return "\n".join(lines)


# ============================================================================
# Individual Benchmarks
# ============================================================================


def benchmark_step(
    n_cells: int = 100,
    grid_size: int = 100,
    n_steps: int = 50,
    use_gmm: bool = False,
    use_distribution_mapper: bool = False,
) -> dict:
    """
    Benchmark Environment.step().

    Args:
        n_cells: Number of cells to spawn
        grid_size: Grid width and height
        n_steps: Number of steps to time
        use_gmm: Use GMM-based food spawning
        use_distribution_mapper: Use distribution action mapper

    Returns:
        Benchmark statistics
    """
    food_dist = None
    if use_gmm:
        # Create GMM with 3 components
        food_dist = FoodDistributionConfig(
            means_x=[grid_size * 0.25, grid_size * 0.5, grid_size * 0.75],
            means_y=[grid_size * 0.25, grid_size * 0.75, grid_size * 0.5],
            variances=[grid_size * 2, grid_size * 2, grid_size * 2],
            total_food_per_step=5,
        )

    config = EnvironmentConfig(
        width=grid_size,
        height=grid_size,
        food_distribution=food_dist,
        use_distribution_mapper=use_distribution_mapper,
    )
    env = Environment(config, seed=42)
    env.spawn_random_cells(n_cells)

    # Warmup
    for _ in range(5):
        env.step()

    # Timed runs
    times = []
    for _ in range(n_steps):
        start = time.perf_counter()
        env.step()
        times.append(time.perf_counter() - start)

    times = np.array(times)
    mean_s = np.mean(times)

    return {
        "mean_ms": mean_s * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
        "max_ms": np.max(times) * 1000,
        "calls_per_sec": 1.0 / mean_s if mean_s > 0 else float("inf"),
        "n_cells_final": len(env.cells),
    }


def benchmark_gillespie(n_iterations: int = 1000, target_time: float = 1.0) -> dict:
    """
    Benchmark GillespieSimulator.simulate_until().

    Args:
        n_iterations: Number of simulation runs to time
        target_time: Simulation time per run

    Returns:
        Benchmark statistics
    """
    params = GeneExpressionParams(
        k_transcription=1.0,
        k_translation=2.0,
        k_mrna_deg=0.1,
        k_protein_deg=0.05,
    )
    rng = np.random.default_rng(42)

    def run_one():
        simulator = GillespieSimulator(params, rng=rng)
        state = GeneExpressionState()
        simulator.simulate_until(state, target_time=target_time)
        return state

    return timeit(run_one, n_runs=n_iterations, warmup=10)


def benchmark_perception(n_calls: int = 1000, grid_size: int = 100) -> dict:
    """
    Benchmark forward_pass() for neural network inference.

    Args:
        n_calls: Number of forward passes to time
        grid_size: Grid size for environment

    Returns:
        Benchmark statistics
    """
    rng = np.random.default_rng(42)
    config = PerceptionConfig()
    network = PerceptionNetwork.random_init(config.hidden_size, rng)

    # Create a sample grid with some food/poison
    grid = np.zeros((grid_size, grid_size), dtype=np.int8)
    grid[rng.integers(0, grid_size, 100), rng.integers(0, grid_size, 100)] = 1  # Food
    grid[rng.integers(0, grid_size, 50), rng.integers(0, grid_size, 50)] = 2  # Poison

    # Pre-compute features
    features = perceive_environment(grid, grid_size // 2, grid_size // 2, config.window_size)

    def run_one():
        return forward_pass(features, network, config)

    return timeit(run_one, n_runs=n_calls, warmup=50)


def benchmark_gmm_spawning(grid_size: int = 100, n_components: int = 3, n_steps: int = 100) -> dict:
    """
    Benchmark _spawn_food_gmm() method.

    Args:
        grid_size: Grid width and height
        n_components: Number of GMM components
        n_steps: Number of spawn operations to time

    Returns:
        Benchmark statistics
    """
    means_x = [grid_size * (i + 1) / (n_components + 1) for i in range(n_components)]
    means_y = [grid_size * (i + 1) / (n_components + 1) for i in range(n_components)]
    variances = [float(grid_size * 2)] * n_components

    food_dist = FoodDistributionConfig(
        means_x=means_x,
        means_y=means_y,
        variances=variances,
        total_food_per_step=5,
    )

    config = EnvironmentConfig(
        width=grid_size,
        height=grid_size,
        food_distribution=food_dist,
    )
    env = Environment(config, seed=42)

    def run_one():
        # Clear food before spawning
        env.grid[:] = 0
        env._spawn_food_gmm()

    return timeit(run_one, n_runs=n_steps, warmup=5)


def benchmark_action_mapper(n_calls: int = 1000) -> dict:
    """
    Benchmark DistributionActionMapper.get_movement().

    Args:
        n_calls: Number of get_movement calls to time

    Returns:
        Benchmark statistics
    """
    rng = np.random.default_rng(42)
    mapper = DistributionActionMapper(
        action_targets=default_action_targets(),
        min_samples=10,
        temperature=1.0,
        rng=rng,
    )

    # Create a protein history with some data
    history = create_protein_history(50)
    for i in range(50):
        history.record(int(100 + rng.normal(0, 20)))

    def run_one():
        return mapper.get_movement(history)

    return timeit(run_one, n_runs=n_calls, warmup=50)


# ============================================================================
# Scaling Analysis
# ============================================================================


def scaling_by_cells(cell_counts: list[int] | None = None, grid_size: int = 100, n_steps: int = 20) -> list[dict]:
    """
    Measure Environment.step() time vs cell count.

    Args:
        cell_counts: List of cell counts to test
        grid_size: Grid size (fixed)
        n_steps: Steps per measurement

    Returns:
        List of results with n_cells and timing stats
    """
    if cell_counts is None:
        cell_counts = [10, 50, 100, 200, 500]

    results = []
    for n in cell_counts:
        stats = benchmark_step(n_cells=n, grid_size=grid_size, n_steps=n_steps)
        results.append({
            "n_cells": n,
            "mean_ms": stats["mean_ms"],
            "std_ms": stats["std_ms"],
            "ms_per_cell": stats["mean_ms"] / n if n > 0 else 0,
        })

    return results


def scaling_by_grid(grid_sizes: list[int] | None = None, n_cells: int = 50, n_steps: int = 20) -> list[dict]:
    """
    Measure Environment.step() time vs grid size.

    Args:
        grid_sizes: List of grid sizes to test
        n_cells: Number of cells (fixed)
        n_steps: Steps per measurement

    Returns:
        List of results with grid_size and timing stats
    """
    if grid_sizes is None:
        grid_sizes = [50, 100, 150, 200, 250]

    results = []
    for size in grid_sizes:
        stats = benchmark_step(n_cells=n_cells, grid_size=size, n_steps=n_steps)
        results.append({
            "grid_size": size,
            "total_tiles": size * size,
            "mean_ms": stats["mean_ms"],
            "std_ms": stats["std_ms"],
        })

    return results


def scaling_by_gmm_components(component_counts: list[int] | None = None, grid_size: int = 100) -> list[dict]:
    """
    Measure _spawn_food_gmm() time vs number of GMM components.

    Args:
        component_counts: List of component counts to test
        grid_size: Grid size (fixed)

    Returns:
        List of results with n_components and timing stats
    """
    if component_counts is None:
        component_counts = [1, 2, 3, 5, 8]

    results = []
    for n in component_counts:
        stats = benchmark_gmm_spawning(grid_size=grid_size, n_components=n)
        results.append({
            "n_components": n,
            "mean_ms": stats["mean_ms"],
            "std_ms": stats["std_ms"],
        })

    return results


# ============================================================================
# Full Profiling
# ============================================================================


def profile_full_simulation(
    n_steps: int = 100,
    n_cells: int = 100,
    grid_size: int = 100,
    output_file: str | None = None,
) -> pstats.Stats:
    """
    Run cProfile on a full simulation.

    Args:
        n_steps: Number of simulation steps
        n_cells: Initial number of cells
        grid_size: Grid dimensions
        output_file: Optional .prof file path to save profile

    Returns:
        pstats.Stats object with profile data
    """
    config = EnvironmentConfig(
        width=grid_size,
        height=grid_size,
        use_distribution_mapper=True,
    )

    def run_simulation():
        env = Environment(config, seed=42)
        env.spawn_random_cells(n_cells)
        for _ in range(n_steps):
            env.step()
        return env

    profiler = cProfile.Profile()
    profiler.enable()
    env = run_simulation()
    profiler.disable()

    if output_file:
        profiler.dump_stats(output_file)
        print(f"Profile saved to: {output_file}")

    stats = pstats.Stats(profiler)
    return stats


def print_profile_summary(stats: pstats.Stats, n_lines: int = 20):
    """Print a summary of profiling results."""
    print("\n=== Top Functions by Cumulative Time ===\n")

    # Print stats directly to stdout
    stats.sort_stats("cumulative")
    stats.print_stats(n_lines)


# ============================================================================
# Output Formatting
# ============================================================================


def print_individual_benchmarks(results: dict[str, dict]):
    """Print formatted individual benchmark results."""
    print("\n=== Individual Benchmarks ===\n")

    headers = ["Function", "Mean(ms)", "Std(ms)", "Calls/s"]
    rows = []

    for name, stats in results.items():
        rows.append([
            name,
            f"{stats['mean_ms']:.2f}",
            f"{stats['std_ms']:.2f}",
            f"{stats['calls_per_sec']:.0f}",
        ])

    print(format_table(headers, rows))


def print_scaling_results(results: list[dict], scale_key: str, scale_label: str):
    """Print formatted scaling analysis results."""
    print(f"\n=== Scaling Analysis ({scale_label}) ===\n")

    if scale_key == "n_cells":
        headers = ["Cells", "Mean(ms)", "ms/cell"]
        rows = [[str(r["n_cells"]), f"{r['mean_ms']:.2f}", f"{r['ms_per_cell']:.3f}"] for r in results]
    elif scale_key == "grid_size":
        headers = ["Grid", "Tiles", "Mean(ms)"]
        rows = [[str(r["grid_size"]), str(r["total_tiles"]), f"{r['mean_ms']:.2f}"] for r in results]
    elif scale_key == "n_components":
        headers = ["Components", "Mean(ms)", "Std(ms)"]
        rows = [[str(r["n_components"]), f"{r['mean_ms']:.2f}", f"{r['std_ms']:.2f}"] for r in results]
    else:
        return

    print(format_table(headers, rows))


# ============================================================================
# Main Entry Point
# ============================================================================


def run_all_benchmarks() -> dict:
    """Run all benchmarks and return results."""
    print("=== evo-noise Profiling Results ===")
    print(f"Python {sys.version.split()[0]}, NumPy {np.__version__}\n")

    # Individual benchmarks
    print("Running individual benchmarks...")
    individual = {
        "Environment.step()": benchmark_step(n_cells=100, grid_size=100, n_steps=50),
        "Gillespie.simulate_until()": benchmark_gillespie(n_iterations=500),
        "forward_pass()": benchmark_perception(n_calls=1000),
        "_spawn_food_gmm()": benchmark_gmm_spawning(grid_size=100, n_components=3),
        "DistributionMapper.get_movement()": benchmark_action_mapper(n_calls=1000),
    }
    print_individual_benchmarks(individual)

    # Scaling by cells
    print("\nRunning cell scaling analysis...")
    scaling_cells = scaling_by_cells()
    print_scaling_results(scaling_cells, "n_cells", "cells")

    # Scaling by grid
    print("\nRunning grid scaling analysis...")
    scaling_grid = scaling_by_grid()
    print_scaling_results(scaling_grid, "grid_size", "grid")

    return {
        "individual": individual,
        "scaling_cells": scaling_cells,
        "scaling_grid": scaling_grid,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Profile evo-noise simulation performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python scripts/profile_simulation.py                     Run all benchmarks
  uv run python scripts/profile_simulation.py --benchmark step    Benchmark Environment.step()
  uv run python scripts/profile_simulation.py --scaling cells     Cell count scaling analysis
  uv run python scripts/profile_simulation.py --full-profile      Run cProfile on full simulation
""",
    )

    parser.add_argument(
        "--benchmark",
        choices=["step", "gillespie", "perception", "gmm", "action_mapper"],
        help="Run a specific benchmark",
    )
    parser.add_argument(
        "--scaling",
        choices=["cells", "grid", "gmm_components"],
        help="Run a specific scaling analysis",
    )
    parser.add_argument(
        "--full-profile",
        action="store_true",
        help="Run cProfile on a full simulation",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file for profile data (.prof) or results (.json)",
    )
    parser.add_argument(
        "--n-cells",
        type=int,
        default=100,
        help="Number of cells for benchmarks (default: 100)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=100,
        help="Grid size for benchmarks (default: 100)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=50,
        help="Number of steps for benchmarks (default: 50)",
    )

    args = parser.parse_args()

    # Specific benchmark
    if args.benchmark:
        print(f"=== Benchmark: {args.benchmark} ===\n")

        if args.benchmark == "step":
            stats = benchmark_step(
                n_cells=args.n_cells,
                grid_size=args.grid_size,
                n_steps=args.n_steps,
            )
            print(f"Environment.step() ({args.n_cells} cells, {args.grid_size}x{args.grid_size} grid):")
            print(f"  Mean: {stats['mean_ms']:.2f} ms")
            print(f"  Std:  {stats['std_ms']:.2f} ms")
            print(f"  Rate: {stats['calls_per_sec']:.0f} calls/sec")

        elif args.benchmark == "gillespie":
            stats = benchmark_gillespie()
            print("GillespieSimulator.simulate_until():")
            print(f"  Mean: {stats['mean_ms']:.4f} ms")
            print(f"  Std:  {stats['std_ms']:.4f} ms")
            print(f"  Rate: {stats['calls_per_sec']:.0f} calls/sec")

        elif args.benchmark == "perception":
            stats = benchmark_perception()
            print("forward_pass():")
            print(f"  Mean: {stats['mean_ms']:.4f} ms")
            print(f"  Std:  {stats['std_ms']:.4f} ms")
            print(f"  Rate: {stats['calls_per_sec']:.0f} calls/sec")

        elif args.benchmark == "gmm":
            stats = benchmark_gmm_spawning(grid_size=args.grid_size)
            print(f"_spawn_food_gmm() ({args.grid_size}x{args.grid_size} grid, 3 components):")
            print(f"  Mean: {stats['mean_ms']:.2f} ms")
            print(f"  Std:  {stats['std_ms']:.2f} ms")
            print(f"  Rate: {stats['calls_per_sec']:.0f} calls/sec")

        elif args.benchmark == "action_mapper":
            stats = benchmark_action_mapper()
            print("DistributionActionMapper.get_movement():")
            print(f"  Mean: {stats['mean_ms']:.4f} ms")
            print(f"  Std:  {stats['std_ms']:.4f} ms")
            print(f"  Rate: {stats['calls_per_sec']:.0f} calls/sec")

    # Scaling analysis
    elif args.scaling:
        if args.scaling == "cells":
            results = scaling_by_cells()
            print_scaling_results(results, "n_cells", "cells")

        elif args.scaling == "grid":
            results = scaling_by_grid()
            print_scaling_results(results, "grid_size", "grid")

        elif args.scaling == "gmm_components":
            results = scaling_by_gmm_components()
            print_scaling_results(results, "n_components", "GMM components")

    # Full profile
    elif args.full_profile:
        output = args.output if args.output and args.output.endswith(".prof") else None
        stats = profile_full_simulation(
            n_steps=args.n_steps,
            n_cells=args.n_cells,
            grid_size=args.grid_size,
            output_file=output,
        )
        print_profile_summary(stats)

    # Default: run all benchmarks
    else:
        results = run_all_benchmarks()

        # Save JSON if requested
        if args.output and args.output.endswith(".json"):
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
