"""
Python utility for reading C++ simulation binary output files.

Binary format v2:
    Header (56 bytes):
        - magic: 4 bytes (b"EVON" = 0x4E4F5645 little-endian)
        - version: 4 bytes (uint32)
        - timestep: 4 bytes (uint32)
        - n_cells: 4 bytes (uint32)
        - width: 4 bytes (uint32)
        - height: 4 bytes (uint32)
        - food_count: 4 bytes (uint32)
        - poison_count: 4 bytes (uint32)
        - total_energy: 8 bytes (float64)
        - mean_energy: 8 bytes (float64)
        - mean_protein: 8 bytes (float64)

    Grid section:
        - grid: width * height bytes (int8)

    Cells section (per cell, 56 bytes each):
        - cell_id: 8 bytes (uint64)
        - x: 4 bytes (int32)
        - y: 4 bytes (int32)
        - energy: 8 bytes (float64)
        - protein: 4 bytes (int32)
        - mrna: 4 bytes (int32)
        - age: 4 bytes (int32)
        - k_transcription: 4 bytes (float32)
        - k_translation: 4 bytes (float32)
        - k_mrna_deg: 4 bytes (float32)
        - k_protein_deg: 4 bytes (float32)

Binary format v1 (legacy):
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

    Cells section (per cell, 32 bytes each):
        - x: 4 bytes (int32)
        - y: 4 bytes (int32)
        - energy: 8 bytes (float64)
        - protein: 4 bytes (int32)
        - mrna: 4 bytes (int32)
        - age: 4 bytes (int32)
        - padding: 4 bytes
"""

import numpy as np
import struct
from dataclasses import dataclass
from typing import Iterator
from pathlib import Path


MAGIC = 0x4E4F5645  # "EVON" in little-endian

HEADER_SIZE_V1 = 32
CELL_SIZE_V1 = 32

HEADER_SIZE_V2 = 56
CELL_SIZE_V2 = 56


@dataclass
class FrameData:
    """Data for a single simulation frame."""
    timestep: int
    width: int
    height: int
    food_count: int
    poison_count: int
    grid: np.ndarray  # Shape: (height, width), dtype: int8
    cell_ids: np.ndarray  # Shape: (n_cells,), dtype: uint64
    cell_positions: np.ndarray  # Shape: (n_cells, 2), dtype: int32
    cell_energies: np.ndarray  # Shape: (n_cells,), dtype: float64
    cell_proteins: np.ndarray  # Shape: (n_cells,), dtype: int32
    cell_mrnas: np.ndarray  # Shape: (n_cells,), dtype: int32
    cell_ages: np.ndarray  # Shape: (n_cells,), dtype: int32
    cell_k_transcription: np.ndarray  # Shape: (n_cells,), dtype: float32
    cell_k_translation: np.ndarray  # Shape: (n_cells,), dtype: float32
    cell_k_mrna_deg: np.ndarray  # Shape: (n_cells,), dtype: float32
    cell_k_protein_deg: np.ndarray  # Shape: (n_cells,), dtype: float32
    total_energy: float
    mean_energy: float
    mean_protein: float

    @property
    def n_cells(self) -> int:
        return len(self.cell_positions)


def read_frame_v1(f, width: int, height: int, n_cells: int, timestep: int,
                  food_count: int, poison_count: int) -> FrameData:
    """Read a v1 format frame (legacy)."""
    grid_size = width * height
    grid_bytes = f.read(grid_size)
    if len(grid_bytes) < grid_size:
        raise ValueError("Unexpected EOF while reading grid")
    grid = np.frombuffer(grid_bytes, dtype=np.int8).reshape((height, width))

    cell_ids = np.zeros(n_cells, dtype=np.uint64)
    cell_positions = np.zeros((n_cells, 2), dtype=np.int32)
    cell_energies = np.zeros(n_cells, dtype=np.float64)
    cell_proteins = np.zeros(n_cells, dtype=np.int32)
    cell_mrnas = np.zeros(n_cells, dtype=np.int32)
    cell_ages = np.zeros(n_cells, dtype=np.int32)
    cell_k_transcription = np.zeros(n_cells, dtype=np.float32)
    cell_k_translation = np.zeros(n_cells, dtype=np.float32)
    cell_k_mrna_deg = np.zeros(n_cells, dtype=np.float32)
    cell_k_protein_deg = np.zeros(n_cells, dtype=np.float32)

    if n_cells > 0:
        cells_bytes = f.read(n_cells * CELL_SIZE_V1)
        if len(cells_bytes) < n_cells * CELL_SIZE_V1:
            raise ValueError("Unexpected EOF while reading cells")

        for i in range(n_cells):
            offset = i * CELL_SIZE_V1
            x, y, energy, protein, mrna, age = struct.unpack(
                '<iidiiixxxx', cells_bytes[offset:offset + CELL_SIZE_V1]
            )
            cell_ids[i] = i + 1  # Assign sequential IDs for v1
            cell_positions[i] = [x, y]
            cell_energies[i] = energy
            cell_proteins[i] = protein
            cell_mrnas[i] = mrna
            cell_ages[i] = age

    total_energy = float(np.sum(cell_energies))
    mean_energy = float(np.mean(cell_energies)) if n_cells > 0 else 0.0
    mean_protein = float(np.mean(cell_proteins)) if n_cells > 0 else 0.0

    return FrameData(
        timestep=timestep,
        width=width,
        height=height,
        food_count=food_count,
        poison_count=poison_count,
        grid=grid.copy(),
        cell_ids=cell_ids,
        cell_positions=cell_positions,
        cell_energies=cell_energies,
        cell_proteins=cell_proteins,
        cell_mrnas=cell_mrnas,
        cell_ages=cell_ages,
        cell_k_transcription=cell_k_transcription,
        cell_k_translation=cell_k_translation,
        cell_k_mrna_deg=cell_k_mrna_deg,
        cell_k_protein_deg=cell_k_protein_deg,
        total_energy=total_energy,
        mean_energy=mean_energy,
        mean_protein=mean_protein,
    )


def read_frame_v2(f, width: int, height: int, n_cells: int, timestep: int,
                  food_count: int, poison_count: int,
                  total_energy: float, mean_energy: float, mean_protein: float) -> FrameData:
    """Read a v2 format frame."""
    grid_size = width * height
    grid_bytes = f.read(grid_size)
    if len(grid_bytes) < grid_size:
        raise ValueError("Unexpected EOF while reading grid")
    grid = np.frombuffer(grid_bytes, dtype=np.int8).reshape((height, width))

    cell_ids = np.zeros(n_cells, dtype=np.uint64)
    cell_positions = np.zeros((n_cells, 2), dtype=np.int32)
    cell_energies = np.zeros(n_cells, dtype=np.float64)
    cell_proteins = np.zeros(n_cells, dtype=np.int32)
    cell_mrnas = np.zeros(n_cells, dtype=np.int32)
    cell_ages = np.zeros(n_cells, dtype=np.int32)
    cell_k_transcription = np.zeros(n_cells, dtype=np.float32)
    cell_k_translation = np.zeros(n_cells, dtype=np.float32)
    cell_k_mrna_deg = np.zeros(n_cells, dtype=np.float32)
    cell_k_protein_deg = np.zeros(n_cells, dtype=np.float32)

    if n_cells > 0:
        cells_bytes = f.read(n_cells * CELL_SIZE_V2)
        if len(cells_bytes) < n_cells * CELL_SIZE_V2:
            raise ValueError("Unexpected EOF while reading cells")

        for i in range(n_cells):
            offset = i * CELL_SIZE_V2
            (cell_id, x, y, energy, protein, mrna, age,
             k_trans, k_transl, k_mrna, k_prot) = struct.unpack(
                '<Qiidiiiffff4x', cells_bytes[offset:offset + CELL_SIZE_V2]
            )
            cell_ids[i] = cell_id
            cell_positions[i] = [x, y]
            cell_energies[i] = energy
            cell_proteins[i] = protein
            cell_mrnas[i] = mrna
            cell_ages[i] = age
            cell_k_transcription[i] = k_trans
            cell_k_translation[i] = k_transl
            cell_k_mrna_deg[i] = k_mrna
            cell_k_protein_deg[i] = k_prot

    return FrameData(
        timestep=timestep,
        width=width,
        height=height,
        food_count=food_count,
        poison_count=poison_count,
        grid=grid.copy(),
        cell_ids=cell_ids,
        cell_positions=cell_positions,
        cell_energies=cell_energies,
        cell_proteins=cell_proteins,
        cell_mrnas=cell_mrnas,
        cell_ages=cell_ages,
        cell_k_transcription=cell_k_transcription,
        cell_k_translation=cell_k_translation,
        cell_k_mrna_deg=cell_k_mrna_deg,
        cell_k_protein_deg=cell_k_protein_deg,
        total_energy=total_energy,
        mean_energy=mean_energy,
        mean_protein=mean_protein,
    )


def read_frame(f) -> FrameData | None:
    """Read a single frame from an open binary file.

    Args:
        f: Open file handle in binary mode

    Returns:
        FrameData if successful, None if EOF
    """
    header_start = f.read(8)
    if len(header_start) < 8:
        return None

    magic, version = struct.unpack('<II', header_start)

    if magic != MAGIC:
        raise ValueError(f"Invalid magic number: {magic:08x}, expected {MAGIC:08x}")

    if version == 1:
        rest_of_header = f.read(HEADER_SIZE_V1 - 8)
        if len(rest_of_header) < HEADER_SIZE_V1 - 8:
            return None
        timestep, n_cells, width, height, food_count, poison_count = struct.unpack(
            '<IIIIII', rest_of_header
        )
        return read_frame_v1(f, width, height, n_cells, timestep, food_count, poison_count)

    elif version == 2:
        rest_of_header = f.read(HEADER_SIZE_V2 - 8)
        if len(rest_of_header) < HEADER_SIZE_V2 - 8:
            return None
        (timestep, n_cells, width, height, food_count, poison_count,
         total_energy, mean_energy, mean_protein) = struct.unpack(
            '<IIIIIIddd', rest_of_header
        )
        return read_frame_v2(f, width, height, n_cells, timestep, food_count, poison_count,
                            total_energy, mean_energy, mean_protein)
    else:
        raise ValueError(f"Unsupported version: {version}")


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


def get_file_version(path: str | Path) -> int:
    """Get the version of a binary file.

    Args:
        path: Path to the binary file

    Returns:
        Version number (1 or 2)
    """
    with open(path, 'rb') as f:
        header = f.read(8)
        if len(header) < 8:
            raise ValueError("File too small")
        magic, version = struct.unpack('<II', header)
        if magic != MAGIC:
            raise ValueError(f"Invalid magic number: {magic:08x}")
        return version


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Read C++ simulation binary output")
    parser.add_argument("path", help="Path to binary file")
    parser.add_argument("--info", action="store_true", help="Show file info only")
    args = parser.parse_args()

    path = args.path
    print(f"Reading {path}...")

    version = get_file_version(path)
    print(f"File version: {version}")

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

            if version >= 2:
                print(f"  Total energy (header): {last.total_energy:.1f}")
                print(f"  Mean energy (header): {last.mean_energy:.1f}")
                print(f"  Mean protein (header): {last.mean_protein:.1f}")
                print(f"  Unique cell IDs: {len(np.unique(last.cell_ids))}")
                print(f"  Gene params available: k_transcription, k_translation, k_mrna_deg, k_protein_deg")
