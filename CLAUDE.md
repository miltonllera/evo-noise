# CLAUDE.md

## Project Overview

Evolutionary simulation of unicellular organisms with stochastic gene expression. Cells sense food/poison via neural networks, run Gillespie gene expression, and use protein levels to decide movement/reproduction.

## Quick Start

### C++ (Recommended)
```bash
cd cpp && make                                    # Build
./evo_noise --steps 500 --cells 100 --seed 42    # Run simulation
uv run python3 create_video.py simulation.bin -o out.mp4  # Create video
uv run python3 plot_results.py simulation.bin results/    # Generate plots
```

### Python
```bash
uv sync
uv run python scripts/run_simulation_and_visualise_results.py --video --steps 300
```

## C++ CLI Options
```
--steps N          Simulation steps (default: 1000)
--cells N          Initial cells (default: 100)
--width/height N   Grid size (default: 50x50)
--seed N           Random seed (default: 42)
--output FILE      Output file (default: simulation.bin)
--repro-threshold  Reproduction threshold (default: 150)
--quiet            Suppress progress output
```

## Output Files
- `simulation.bin` - Binary data (v2 format with cell IDs, gene params)
- `simulation.mp4` - Animated visualization
- `simulation_plots.png` - Population/energy/protein over time
- `gene_params.png` - Gene expression parameter evolution
- `lifespan_distribution.png` - Cell lifespan histogram
- `top_agents.png` - Lifecycle of longest-lived agents
- `stats.json` - Summary statistics

## Architecture

**Pipeline**: Perception → Gene Expression → Action
1. 5×5 local window → PerceptionNetwork (50→16→4) → GeneExpressionParams
2. GillespieSimulator runs stochastic transcription/translation/degradation
3. ActionMapper converts protein levels to movement/reproduction

**Design**: Toroidal grid, cells can overlap, asexual reproduction with mutated networks

## Code Style
- No default parameter values
- No try/except unless explicitly requested
- Concise commits, don't mention claude
