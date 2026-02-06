#!/usr/bin/env python3
"""Create MP4 video from C++ simulation binary output."""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation
from pathlib import Path
from read_binary import read_all_frames, FrameData


def create_video(
    binary_path: str,
    output_file: str,
    fps: int,
):
    """Create a video from simulation binary output."""
    print(f"Reading {binary_path}...")
    frames = read_all_frames(binary_path)

    if not frames:
        print("No frames to animate")
        return

    print(f"Loaded {len(frames)} frames")

    width = frames[0].width
    height = frames[0].height

    fig, ax = plt.subplots(figsize=(8, 8))

    grid_cmap = ListedColormap(["#f0f0f0", "#4CAF50", "#f44336"])
    legend_elements = [
        Patch(facecolor="#4CAF50", label="Food"),
        Patch(facecolor="#f44336", label="Poison"),
    ]
    protein_max = 300.0

    def animate(frame_idx: int):
        ax.clear()
        frame = frames[frame_idx]

        ax.imshow(
            frame.grid,
            cmap=grid_cmap,
            vmin=0,
            vmax=2,
            origin="lower",
            aspect="equal",
        )

        if frame.n_cells > 0:
            sizes = np.clip(frame.cell_energies, 10, 200)
            protein_normalized = np.clip(frame.cell_proteins / protein_max, 0, 1)
            ax.scatter(
                frame.cell_positions[:, 0],
                frame.cell_positions[:, 1],
                s=sizes,
                c=protein_normalized,
                cmap="Blues",
                vmin=0,
                vmax=1,
                alpha=0.8,
                edgecolors="black",
                linewidths=0.5,
            )

        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

        mean_protein = frame.cell_proteins.mean() if frame.n_cells > 0 else 0
        ax.set_title(
            f"Step {frame.timestep} | "
            f"Cells: {frame.n_cells} | "
            f"Mean protein: {mean_protein:.0f}"
        )
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(-0.5, height - 0.5)
        return []

    print(f"Creating animation ({len(frames)} frames at {fps} fps)...")
    anim = FuncAnimation(
        fig,
        animate,
        frames=len(frames),
        interval=1000 // fps,
        blit=True,
    )

    print(f"Saving to {output_file}...")
    if output_file.endswith(".gif"):
        anim.save(output_file, writer="pillow", fps=fps)
    else:
        anim.save(output_file, writer="ffmpeg", fps=fps)

    plt.close(fig)
    print(f"Video saved: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create video from C++ simulation output")
    parser.add_argument("binary_path", help="Path to simulation binary file")
    parser.add_argument("-o", "--output", default="simulation.mp4", help="Output video file")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    args = parser.parse_args()

    create_video(args.binary_path, args.output, args.fps)
