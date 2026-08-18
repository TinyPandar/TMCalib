"""Extend the 32 x 24 pre-generated DMD pattern set from 4N to 8N.

The original 3072 probes/patterns are retained as the first half. The second
half is generated reproducibly with 16 uniformly sampled phase levels. Output
arrays are memory-mapped so the 4.8 GB pattern dataset is never held in RAM.
"""

import argparse
import json
import os
import sys

import numpy as np


N_X = 32
N_Y = 24
N_IN = N_X * N_Y
SOURCE_MULTIPLIER = 4
TARGET_MULTIPLIER = 8
PHASE_LEVELS = 16
PHASE_STEP = np.pi / 8
SUPERPIXEL_SIZE = 4
LOGICAL_PIXEL_SIZE = 32
DMD_HEIGHT = N_Y * LOGICAL_PIXEL_SIZE
DMD_WIDTH = N_X * LOGICAL_PIXEL_SIZE
DEFAULT_SEED = 20260811


def build_phase_tiles():
    """Build the exact 32 x 32 holo_SP bitmap for every allowed phase."""
    project_parent = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_parent not in sys.path:
        sys.path.append(project_parent)

    from holograms.generate_LUT import generate_lut

    _, pixel_combinations, lut = generate_lut("sp", SUPERPIXEL_SIZE)
    lut_center = len(lut) // 2
    phase_values = np.exp(
        1j * np.arange(PHASE_LEVELS, dtype=np.float32) * PHASE_STEP
    )
    tiles = np.empty(
        (PHASE_LEVELS, LOGICAL_PIXEL_SIZE, LOGICAL_PIXEL_SIZE),
        dtype=np.uint8,
    )

    cells_per_logical_pixel = LOGICAL_PIXEL_SIZE // SUPERPIXEL_SIZE
    for phase_index, value in enumerate(phase_values):
        re = int(np.round(np.real(value) / 0.01))
        im = int(np.round(np.imag(value) / 0.01))
        tile = np.empty((LOGICAL_PIXEL_SIZE, LOGICAL_PIXEL_SIZE), dtype=np.uint8)

        for cell_row in range(cells_per_logical_pixel):
            shift = (SUPERPIXEL_SIZE * cell_row) % (SUPERPIXEL_SIZE ** 2)
            superpixel = np.roll(
                pixel_combinations[lut[re + lut_center, im + lut_center]],
                -shift,
            )
            superpixel = (
                superpixel.reshape(SUPERPIXEL_SIZE, SUPERPIXEL_SIZE).T.astype(np.uint8)
                * 255
            )
            row_start = cell_row * SUPERPIXEL_SIZE
            row_stop = row_start + SUPERPIXEL_SIZE
            tile[row_start:row_stop] = np.tile(
                superpixel,
                (1, cells_per_logical_pixel),
            )

        tiles[phase_index] = tile

    return phase_values.astype(np.complex64), tiles


def encode_phase_indices(phase_indices, phase_tiles):
    """Vectorize logical phase indices into full 768 x 1024 DMD bitmaps."""
    batch = phase_indices.shape[0]
    return (
        phase_tiles[phase_indices]
        .transpose(0, 1, 3, 2, 4)
        .reshape(batch, DMD_HEIGHT, DMD_WIDTH)
    )


def copy_in_batches(source, target, batch_size, label):
    for start in range(0, source.shape[0], batch_size):
        stop = min(start + batch_size, source.shape[0])
        target[start:stop] = source[start:stop]
        print(f"\r{label}: {stop}/{source.shape[0]}", end="", flush=True)
    print()


def generate(source_dir, output_dir, seed, batch_size):
    source_dir = os.path.abspath(source_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    source_probe_path = os.path.join(source_dir, "probe.npy")
    source_pattern_path = os.path.join(source_dir, "patterns_pregenerated.npy")
    output_probe_path = os.path.join(output_dir, "probe.npy")
    output_pattern_path = os.path.join(output_dir, "patterns_pregenerated.npy")
    partial_probe_path = output_probe_path + ".partial.npy"
    partial_pattern_path = output_pattern_path + ".partial.npy"

    existing_outputs = [
        path
        for path in (output_probe_path, output_pattern_path)
        if os.path.exists(path)
    ]
    if existing_outputs:
        raise FileExistsError(
            "Output already exists; remove or choose another output directory: "
            + ", ".join(existing_outputs)
        )

    for partial_path in (partial_probe_path, partial_pattern_path):
        if os.path.exists(partial_path):
            os.remove(partial_path)

    source_probes = np.load(source_probe_path, mmap_mode="r")
    source_patterns = np.load(source_pattern_path, mmap_mode="r")
    expected_source_count = SOURCE_MULTIPLIER * N_IN
    expected_probe_shape = (expected_source_count, N_Y, N_X)
    expected_pattern_shape = (expected_source_count, DMD_HEIGHT, DMD_WIDTH)
    if source_probes.shape != expected_probe_shape:
        raise ValueError(
            f"Source probe shape is {source_probes.shape}, expected {expected_probe_shape}"
        )
    if source_patterns.shape != expected_pattern_shape:
        raise ValueError(
            f"Source pattern shape is {source_patterns.shape}, expected {expected_pattern_shape}"
        )

    target_count = TARGET_MULTIPLIER * N_IN
    probes = np.lib.format.open_memmap(
        partial_probe_path,
        mode="w+",
        dtype=np.complex64,
        shape=(target_count, N_Y, N_X),
    )
    patterns = np.lib.format.open_memmap(
        partial_pattern_path,
        mode="w+",
        dtype=np.uint8,
        shape=(target_count, DMD_HEIGHT, DMD_WIDTH),
    )

    try:
        copy_in_batches(source_probes, probes, batch_size, "Copying source probes")
        copy_in_batches(source_patterns, patterns, batch_size, "Copying source patterns")

        phase_values, phase_tiles = build_phase_tiles()
        rng = np.random.default_rng(seed)
        for start in range(expected_source_count, target_count, batch_size):
            stop = min(start + batch_size, target_count)
            phase_indices = rng.integers(
                0,
                PHASE_LEVELS,
                size=(stop - start, N_Y, N_X),
                dtype=np.uint8,
            )
            probes[start:stop] = phase_values[phase_indices]
            patterns[start:stop] = encode_phase_indices(phase_indices, phase_tiles)
            print(
                f"\rGenerating new probes/patterns: {stop}/{target_count}",
                end="",
                flush=True,
            )
        print()

        probes.flush()
        patterns.flush()
    finally:
        del probes
        del patterns

    os.replace(partial_probe_path, output_probe_path)
    os.replace(partial_pattern_path, output_pattern_path)

    metadata = {
        "logical_shape": [N_Y, N_X],
        "dmd_shape": [DMD_HEIGHT, DMD_WIDTH],
        "n_in": N_IN,
        "probe_multiplier": TARGET_MULTIPLIER,
        "probe_count": target_count,
        "phase_levels": PHASE_LEVELS,
        "seed_for_appended_half": seed,
        "source_directory": source_dir,
        "source_probe_count": expected_source_count,
        "first_half_retained_from_source": True,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print(f"8N dataset generated in: {output_dir}")
    print(f"  probes:   {(target_count, N_Y, N_X)} complex64")
    print(f"  patterns: {(target_count, DMD_HEIGHT, DMD_WIDTH)} uint8")


def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=os.path.join(project_dir, "pregenerated_patterns"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(project_dir, "pregenerated_patterns_8N"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    generate(args.source, args.output, args.seed, args.batch_size)


if __name__ == "__main__":
    main()
