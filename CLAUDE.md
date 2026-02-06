# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

evo-noise is a Python project to study how information in a simulated environment can be encoded into a signal. The environment is a simulated Petri dish where food and poison spawn. The agents are unicellular organisms that sense their environment through a neural network, run stochastic gene expression, and use protein levels to decide actions (movement, reproduction).

## Development Commands

```bash
uv sync                                              # Install dependencies
uv run python scripts/run_simulation_and_visualise_results.py --video --steps 300  # Run simulation video
uv run jupyter notebook                              # Start Jupyter
uv add <package-name>                                # Add dependency
```

## Architecture

**Perception → Gene Expression → Action pipeline**:
1. `PerceptionSystem` extracts 5×5 local window around cell, creates binary food/poison channels
2. `PerceptionNetwork` (50→16→4 neural net) maps perception to `GeneExpressionParams`
3. `GillespieSimulator` runs stochastic gene expression (transcription/translation/degradation)
4. `ActionMapper` or `DistributionActionMapper` maps protein levels to movement/reproduction decisions

**Key modules** (`src/`):
- `environment.py`: Main simulation loop, `Environment` and `Cell` classes
- `perception.py`: Neural network for environment sensing
- `gillespie.py`: Stochastic gene expression simulation
- `action_mapper.py`: Protein-to-action mapping (threshold or distribution-based)
- `repressilator.py`: ODE model for repressilator oscillator (standalone)

**Simulation loop (`Environment.step()`)**:
1. Spawn food/poison at empty tiles
2. For each cell: perceive → update gene expression → record protein → move → consume → try reproduce → apply metabolism
3. Remove dead cells (energy ≤ 0)

**Key design decisions**:
- Toroidal grid (edges wrap around)
- Cells don't occupy grid tiles (multiple cells can overlap)
- Asexual reproduction (offspring spawns adjacent with half reproduction cost as starting energy)
- Offspring inherit mutated perception networks; gene params computed fresh at spawn location

## Code Style Guidelines

- NEVER use default values in function parameters
- ONLY write what requested explicitly, do not add extra stuff
- Always require all parameters to be explicitly passed
- Be concise, do not over-comment the code
- Avoid try/except logic unless very explicitly asked
- Avoid using .get() methods with default values
- Commits messages should be very concise and not mention claude
