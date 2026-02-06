#!/usr/bin/env python3
"""Generate plots from C++ simulation binary output."""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from read_binary import read_all_frames, get_file_version, FrameData


class GeneParamsTracker:
    """Track gene expression parameters over time."""

    def __init__(self):
        self.timesteps: list[int] = []
        self.mean_k_transcription: list[float] = []
        self.mean_k_translation: list[float] = []
        self.mean_k_mrna_deg: list[float] = []
        self.mean_k_protein_deg: list[float] = []
        self.std_k_transcription: list[float] = []
        self.std_k_translation: list[float] = []
        self.std_k_mrna_deg: list[float] = []
        self.std_k_protein_deg: list[float] = []

    def record(self, frame: FrameData):
        self.timesteps.append(frame.timestep)

        if frame.n_cells > 0:
            self.mean_k_transcription.append(float(np.mean(frame.cell_k_transcription)))
            self.mean_k_translation.append(float(np.mean(frame.cell_k_translation)))
            self.mean_k_mrna_deg.append(float(np.mean(frame.cell_k_mrna_deg)))
            self.mean_k_protein_deg.append(float(np.mean(frame.cell_k_protein_deg)))
            self.std_k_transcription.append(float(np.std(frame.cell_k_transcription)))
            self.std_k_translation.append(float(np.std(frame.cell_k_translation)))
            self.std_k_mrna_deg.append(float(np.std(frame.cell_k_mrna_deg)))
            self.std_k_protein_deg.append(float(np.std(frame.cell_k_protein_deg)))
        else:
            self.mean_k_transcription.append(0.0)
            self.mean_k_translation.append(0.0)
            self.mean_k_mrna_deg.append(0.0)
            self.mean_k_protein_deg.append(0.0)
            self.std_k_transcription.append(0.0)
            self.std_k_translation.append(0.0)
            self.std_k_mrna_deg.append(0.0)
            self.std_k_protein_deg.append(0.0)


class LifespanTracker:
    """Track cell birth and death to compute lifespan statistics."""

    def __init__(self):
        self.birth_times: dict[int, int] = {}
        self.death_times: dict[int, int] = {}
        self.lifespans: list[int] = []
        self.alive_at_end: set[int] = set()

    def update(self, frame: FrameData):
        current_ids = set(frame.cell_ids)

        for cell_id in current_ids:
            if cell_id not in self.birth_times:
                self.birth_times[cell_id] = frame.timestep

        for cell_id in list(self.birth_times.keys()):
            if cell_id not in current_ids and cell_id not in self.death_times:
                self.death_times[cell_id] = frame.timestep
                lifespan = self.death_times[cell_id] - self.birth_times[cell_id]
                self.lifespans.append(lifespan)

        self.alive_at_end = current_ids

    def get_stats(self) -> dict:
        if not self.lifespans:
            return {
                'total_born': len(self.birth_times),
                'total_died': len(self.death_times),
                'alive_at_end': len(self.alive_at_end),
                'mean_lifespan': 0.0,
                'median_lifespan': 0.0,
                'max_lifespan': 0,
                'min_lifespan': 0,
            }
        return {
            'total_born': len(self.birth_times),
            'total_died': len(self.death_times),
            'alive_at_end': len(self.alive_at_end),
            'mean_lifespan': float(np.mean(self.lifespans)),
            'median_lifespan': float(np.median(self.lifespans)),
            'max_lifespan': int(np.max(self.lifespans)),
            'min_lifespan': int(np.min(self.lifespans)),
        }


class TopAgentTracker:
    """Track the longest-lived agents and their histories."""

    def __init__(self, top_n: int):
        self.top_n = top_n
        self.agent_histories: dict[int, dict] = defaultdict(lambda: {
            'timesteps': [],
            'energies': [],
            'proteins': [],
            'positions': [],
            'birth_time': None,
            'death_time': None,
        })
        self.current_ids: set[int] = set()

    def update(self, frame: FrameData):
        frame_ids = set(frame.cell_ids)

        for cell_id in frame_ids - self.current_ids:
            self.agent_histories[cell_id]['birth_time'] = frame.timestep

        for cell_id in self.current_ids - frame_ids:
            self.agent_histories[cell_id]['death_time'] = frame.timestep

        for i, cell_id in enumerate(frame.cell_ids):
            history = self.agent_histories[cell_id]
            history['timesteps'].append(frame.timestep)
            history['energies'].append(float(frame.cell_energies[i]))
            history['proteins'].append(int(frame.cell_proteins[i]))
            history['positions'].append((int(frame.cell_positions[i, 0]),
                                         int(frame.cell_positions[i, 1])))

        self.current_ids = frame_ids

    def get_top_agents(self) -> list[tuple[int, dict]]:
        def lifespan(agent_id: int) -> int:
            h = self.agent_histories[agent_id]
            if not h['timesteps']:
                return 0
            start = h['birth_time'] if h['birth_time'] is not None else h['timesteps'][0]
            end = h['death_time'] if h['death_time'] is not None else h['timesteps'][-1]
            return end - start

        sorted_agents = sorted(self.agent_histories.keys(), key=lifespan, reverse=True)
        return [(agent_id, self.agent_histories[agent_id]) for agent_id in sorted_agents[:self.top_n]]


def plot_basic_results(frames: list[FrameData], output_dir: Path):
    """Generate basic simulation plots."""
    timesteps = [f.timestep for f in frames]
    n_cells = [f.n_cells for f in frames]
    food_counts = [f.food_count for f in frames]
    poison_counts = [f.poison_count for f in frames]

    avg_energies = []
    avg_proteins = []
    avg_ages = []
    for f in frames:
        if f.n_cells > 0:
            avg_energies.append(float(np.mean(f.cell_energies)))
            avg_proteins.append(float(np.mean(f.cell_proteins)))
            avg_ages.append(float(np.mean(f.cell_ages)))
        else:
            avg_energies.append(0)
            avg_proteins.append(0)
            avg_ages.append(0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(timesteps, n_cells, 'b-', linewidth=1)
    axes[0, 0].set_xlabel('Timestep')
    axes[0, 0].set_ylabel('Cell Count')
    axes[0, 0].set_title('Population Over Time')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(timesteps, food_counts, 'g-', label='Food', linewidth=1)
    axes[0, 1].plot(timesteps, poison_counts, 'r-', label='Poison', linewidth=1)
    axes[0, 1].set_xlabel('Timestep')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Resources Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(timesteps, avg_energies, 'orange', linewidth=1)
    axes[1, 0].set_xlabel('Timestep')
    axes[1, 0].set_ylabel('Average Energy')
    axes[1, 0].set_title('Average Cell Energy')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(timesteps, avg_proteins, 'purple', linewidth=1)
    axes[1, 1].set_xlabel('Timestep')
    axes[1, 1].set_ylabel('Average Protein')
    axes[1, 1].set_title('Average Protein Level')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'simulation_plots.png', dpi=150)
    plt.close()


def plot_gene_params(tracker: GeneParamsTracker, output_dir: Path):
    """Plot gene expression parameter evolution."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(tracker.timesteps, tracker.mean_k_transcription, 'b-', linewidth=1)
    axes[0, 0].fill_between(
        tracker.timesteps,
        np.array(tracker.mean_k_transcription) - np.array(tracker.std_k_transcription),
        np.array(tracker.mean_k_transcription) + np.array(tracker.std_k_transcription),
        alpha=0.3
    )
    axes[0, 0].set_xlabel('Timestep')
    axes[0, 0].set_ylabel('k_transcription')
    axes[0, 0].set_title('Transcription Rate')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(tracker.timesteps, tracker.mean_k_translation, 'g-', linewidth=1)
    axes[0, 1].fill_between(
        tracker.timesteps,
        np.array(tracker.mean_k_translation) - np.array(tracker.std_k_translation),
        np.array(tracker.mean_k_translation) + np.array(tracker.std_k_translation),
        alpha=0.3
    )
    axes[0, 1].set_xlabel('Timestep')
    axes[0, 1].set_ylabel('k_translation')
    axes[0, 1].set_title('Translation Rate')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(tracker.timesteps, tracker.mean_k_mrna_deg, 'r-', linewidth=1)
    axes[1, 0].fill_between(
        tracker.timesteps,
        np.array(tracker.mean_k_mrna_deg) - np.array(tracker.std_k_mrna_deg),
        np.array(tracker.mean_k_mrna_deg) + np.array(tracker.std_k_mrna_deg),
        alpha=0.3
    )
    axes[1, 0].set_xlabel('Timestep')
    axes[1, 0].set_ylabel('k_mrna_deg')
    axes[1, 0].set_title('mRNA Degradation Rate')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(tracker.timesteps, tracker.mean_k_protein_deg, 'm-', linewidth=1)
    axes[1, 1].fill_between(
        tracker.timesteps,
        np.array(tracker.mean_k_protein_deg) - np.array(tracker.std_k_protein_deg),
        np.array(tracker.mean_k_protein_deg) + np.array(tracker.std_k_protein_deg),
        alpha=0.3
    )
    axes[1, 1].set_xlabel('Timestep')
    axes[1, 1].set_ylabel('k_protein_deg')
    axes[1, 1].set_title('Protein Degradation Rate')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'gene_params.png', dpi=150)
    plt.close()


def plot_lifespan_distribution(tracker: LifespanTracker, output_dir: Path):
    """Plot lifespan distribution histogram."""
    if not tracker.lifespans:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(tracker.lifespans, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Lifespan (timesteps)')
    ax.set_ylabel('Count')
    ax.set_title('Cell Lifespan Distribution')
    ax.grid(True, alpha=0.3)

    stats = tracker.get_stats()
    textstr = f"Mean: {stats['mean_lifespan']:.1f}\nMedian: {stats['median_lifespan']:.1f}\nMax: {stats['max_lifespan']}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.savefig(output_dir / 'lifespan_distribution.png', dpi=150)
    plt.close()


def plot_top_agents(tracker: TopAgentTracker, output_dir: Path):
    """Plot the lifecycle of top agents."""
    top_agents = tracker.get_top_agents()
    if not top_agents:
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    for agent_id, history in top_agents:
        if history['timesteps']:
            lifespan = len(history['timesteps'])
            label = f"Agent {agent_id} (lifespan={lifespan})"
            axes[0].plot(history['timesteps'], history['energies'], linewidth=1, label=label, alpha=0.7)
            axes[1].plot(history['timesteps'], history['proteins'], linewidth=1, label=label, alpha=0.7)

    axes[0].set_xlabel('Timestep')
    axes[0].set_ylabel('Energy')
    axes[0].set_title('Top Agents - Energy Over Time')
    axes[0].legend(fontsize=8, loc='upper right')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Timestep')
    axes[1].set_ylabel('Protein')
    axes[1].set_title('Top Agents - Protein Level Over Time')
    axes[1].legend(fontsize=8, loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'top_agents.png', dpi=150)
    plt.close()


def plot_results(binary_path: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    version = get_file_version(binary_path)
    frames = read_all_frames(binary_path)

    if not frames:
        print("No frames to plot")
        return

    plot_basic_results(frames, output_dir)

    gene_params_tracker = None
    lifespan_tracker = None
    top_agent_tracker = None

    if version >= 2:
        gene_params_tracker = GeneParamsTracker()
        lifespan_tracker = LifespanTracker()
        top_agent_tracker = TopAgentTracker(top_n=5)

        for frame in frames:
            gene_params_tracker.record(frame)
            lifespan_tracker.update(frame)
            top_agent_tracker.update(frame)

        plot_gene_params(gene_params_tracker, output_dir)
        plot_lifespan_distribution(lifespan_tracker, output_dir)
        plot_top_agents(top_agent_tracker, output_dir)

    last = frames[-1]
    first = frames[0]
    timesteps = [f.timestep for f in frames]
    n_cells = [f.n_cells for f in frames]
    avg_energies = [float(np.mean(f.cell_energies)) if f.n_cells > 0 else 0 for f in frames]
    avg_proteins = [float(np.mean(f.cell_proteins)) if f.n_cells > 0 else 0 for f in frames]
    avg_ages = [float(np.mean(f.cell_ages)) if f.n_cells > 0 else 0 for f in frames]

    stats = {
        'file_version': version,
        'total_steps': len(frames) - 1,
        'initial_cells': first.n_cells,
        'final_cells': last.n_cells,
        'grid_width': first.width,
        'grid_height': first.height,
        'max_cells': max(n_cells),
        'final_food': last.food_count,
        'final_poison': last.poison_count,
        'final_avg_energy': avg_energies[-1] if last.n_cells > 0 else 0,
        'final_avg_protein': avg_proteins[-1] if last.n_cells > 0 else 0,
        'final_avg_age': avg_ages[-1] if last.n_cells > 0 else 0,
    }

    if lifespan_tracker:
        stats['lifespan_stats'] = lifespan_tracker.get_stats()

    with open(output_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"  Plots saved to: {output_dir / 'simulation_plots.png'}")
    if version >= 2:
        print(f"  Gene params plot: {output_dir / 'gene_params.png'}")
        print(f"  Lifespan distribution: {output_dir / 'lifespan_distribution.png'}")
        print(f"  Top agents plot: {output_dir / 'top_agents.png'}")
    print(f"  Stats saved to: {output_dir / 'stats.json'}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_results.py <simulation.bin> <output_dir>")
        sys.exit(1)
    plot_results(sys.argv[1], sys.argv[2])
