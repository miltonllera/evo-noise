# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

evo-noise is a Python project for to study how information in a simulated environment can be encoded
into a signal. Each component of the signal will carry different types of information which can be
used by a downstream "actor". The environment is a simulated Petri dish where food and poison are
located. The agents are simple unicelluar organisms which can sense their environment and produce
some action like (not) moving or replicating.

## Development Commands

```bash
# Install dependencies
uv sync

# Run the main script
uv run python hello.py

# Start Jupyter notebook
uv run jupyter notebook

# Add a new dependency
uv add <package-name>
```

## Technical Details

- Python 3.13 required
- Package manager: uv (uses pyproject.toml and uv.lock)
- Virtual environment: .venv (auto-managed by uv)

## Environment Module (`src/environment.py`)

### Data Classes

- **`TileType`**: Enum for grid cells (`EMPTY`, `FOOD`, `POISON`)
- **`Cell`**: Organism with position (`x`, `y`), `energy`, `age`, and `move()` method
- **`EnvironmentConfig`**: Tunable parameters (grid size, spawn probabilities, energy values, reproduction thresholds)

### Environment Class

**State**:
- `grid`: 2D numpy array of `TileType` values
- `cells`: list of living `Cell` objects
- `rng`: seeded random generator
- `timestep`: current simulation time

**Simulation loop (`step()`)**:
1. Spawn food/poison at empty tiles
2. For each cell: move → consume → try reproduce → apply metabolism
3. Remove dead cells (energy ≤ 0)
4. Increment timestep

**Key design decisions**:
- Toroidal grid (edges wrap around)
- Cells don't occupy grid tiles (multiple cells can overlap)
- Asexual reproduction (offspring spawns adjacent with half reproduction cost as starting energy)
