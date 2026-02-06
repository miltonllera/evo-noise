"""
Python utility for reading C++ simulation binary output files.

Binary format:
    Header (32 bytes):
        - magic: 4 bytes (b"EVON" = 0x4E4F5645 little-endian)
        - version: 4 bytes (uint32)
        - timestep: 4 bytes (uint32)
        - n_cells: 4 bytes (uint32)
        - width: 4 bytes (uint32)
        - height: 4 bytes (uint32)
        - food_count: 4 bytes (uint32)
        - poison_count: 4 bytes (uint32)

    Grid section:
        - grid: width * height bytes (int8)

    Cells section (per cell, 28 bytes each):
        - x: 4 bytes (int32)
        - y: 4 bytes (int32)
        - energy: 8 bytes (float64)
        - protein: 4 bytes (int32)
        - mrna: 4 bytes (int32)
        - age: 4 bytes (int32)
"""

import numpy as np
import struct
from dataclasses import dataclass
from typing import Iterator
from pathlib import Path


MAGIC = 0x4E4F5645  # "EVON" in little-endian
VERSION = 1
HEADER_SIZE = 32
CELL_SIZE = 32  # 28 bytes of data + 4 bytes padding (alignment to 8-byte boundary)


@dataclass
class FrameData:
    """Data for a single simulation frame."""
    timestep: int
    width: int
    height: int
    food_count: int
    poison_count: int
    grid: np.ndarray  # Shape: (height, width), dtype: int8
    cell_positions: np.ndarray  # Shape: (n_cells, 2), dtype: int32
    cell_energies: np.ndarray  # Shape: (n_cells,), dtype: float64
    cell_proteins: np.ndarray  # Shape: (n_cells,), dtype: int32
    cell_mrnas: np.ndarray  # Shape: (n_cells,), dtype: int32
    cell_ages: np.ndarray  # Shape: (n_cells,), dtype: int32

    @property
    def n_cells(self) -> int:
        return len(self.cell_positions)


def read_frame(f) -> FrameData | None:
    """Read a single frame from an open binary file.

    Args:
        f: Open file handle in binary mode

    Returns:
        FrameData if successful, None if EOF
    """
    header_bytes = f.read(HEADER_SIZE)
    if len(header_bytes) < HEADER_SIZE:
        return None

    magic, version, timestep, n_cells, width, height, food_count, poison_count = struct.unpack(
        '<IIIIIIII', header_bytes
    )

    if magic != MAGIC:
        raise ValueError(f"Invalid magic number: {magic:08x}, expected {MAGIC:08x}")
    if version != VERSION:
        raise ValueError(f"Unsupported version: {version}, expected {VERSION}")

    grid_size = width * height
    grid_bytes = f.read(grid_size)
    if len(grid_bytes) < grid_size:
        raise ValueError("Unexpected EOF while reading grid")
    grid = np.frombuffer(grid_bytes, dtype=np.int8).reshape((height, width))

    if n_cells > 0:
        cells_bytes = f.read(n_cells * CELL_SIZE)
        if len(cells_bytes) < n_cells * CELL_SIZE:
            raise ValueError("Unexpected EOF while reading cells")

        cell_positions = np.zeros((n_cells, 2), dtype=np.int32)
        cell_energies = np.zeros(n_cells, dtype=np.float64)
        cell_proteins = np.zeros(n_cells, dtype=np.int32)
        cell_mrnas = np.zeros(n_cells, dtype=np.int32)
        cell_ages = np.zeros(n_cells, dtype=np.int32)

        for i in range(n_cells):
            offset = i * CELL_SIZE
            x, y, energy, protein, mrna, age = struct.unpack(
                '<iidiiixxxx', cells_bytes[offset:offset + CELL_SIZE]
            )
            cell_positions[i] = [x, y]
            cell_energies[i] = energy
            cell_proteins[i] = protein
            cell_mrnas[i] = mrna
            cell_ages[i] = age
    else:
        cell_positions = np.empty((0, 2), dtype=np.int32)
        cell_energies = np.empty(0, dtype=np.float64)
        cell_proteins = np.empty(0, dtype=np.int32)
        cell_mrnas = np.empty(0, dtype=np.int32)
        cell_ages = np.empty(0, dtype=np.int32)

    return FrameData(
        timestep=timestep,
        width=width,
        height=height,
        food_count=food_count,
        poison_count=poison_count,
        grid=grid.copy(),
        cell_positions=cell_positions,
        cell_energies=cell_energies,
        cell_proteins=cell_proteins,
        cell_mrnas=cell_mrnas,
        cell_ages=cell_ages,
    )


def read_frames(path: str | Path) -> Iterator[FrameData]:
    """Iterate over all frames in a binary simulation file.

    Args:
        path: Path to the binary file

    Yields:
        FrameData for each frame
    """
    with open(path, 'rb') as f:
        while True:
            frame = read_frame(f)
            if frame is None:
                break
            yield frame


def read_all_frames(path: str | Path) -> list[FrameData]:
    """Read all frames from a binary simulation file.

    Args:
        path: Path to the binary file

    Returns:
        List of FrameData objects
    """
    return list(read_frames(path))


def read_single_frame_file(path: str | Path) -> FrameData:
    """Read a single-frame binary file (from --frame-output mode).

    Args:
        path: Path to the binary file

    Returns:
        FrameData for the single frame
    """
    with open(path, 'rb') as f:
        frame = read_frame(f)
        if frame is None:
            raise ValueError(f"Failed to read frame from {path}")
        return frame


def find_frame_files(base_path: str | Path, pattern: str = "{base}_{:06d}.bin") -> list[Path]:
    """Find all frame files matching a pattern.

    Args:
        base_path: Base path used with --frame-output
        pattern: Pattern with {base} placeholder and frame number format

    Returns:
        Sorted list of frame file paths
    """
    base_path = Path(base_path)
    parent = base_path.parent or Path('.')
    base_name = base_path.name

    frame_files = []
    for f in parent.iterdir():
        if f.name.startswith(base_name + "_") and f.suffix == ".bin":
            frame_files.append(f)

    return sorted(frame_files)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python read_binary.py <simulation.bin>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Reading {path}...")

    frames = read_all_frames(path)
    print(f"Read {len(frames)} frames")

    if frames:
        first = frames[0]
        last = frames[-1]
        print(f"\nFirst frame (t={first.timestep}):")
        print(f"  Grid: {first.width}x{first.height}")
        print(f"  Cells: {first.n_cells}")
        print(f"  Food: {first.food_count}, Poison: {first.poison_count}")

        print(f"\nLast frame (t={last.timestep}):")
        print(f"  Cells: {last.n_cells}")
        print(f"  Food: {last.food_count}, Poison: {last.poison_count}")

        if last.n_cells > 0:
            print(f"  Avg energy: {last.cell_energies.mean():.1f}")
            print(f"  Avg protein: {last.cell_proteins.mean():.1f}")
            print(f"  Avg age: {last.cell_ages.mean():.1f}")
