"""Generate paired phase-only probe datasets for the 32 x 24 V4 profile.

The probe matrix records the complex field actually selected by the 4 x 4
superpixel LUT, rather than the ideal requested unit phasor.  All supported
phase-level runs use the same seeded 32-level random index stream so 4-, 16-,
and 32-level datasets can be compared without changing the underlying random
draws.  Multi-gigabyte DMD arrays are written incrementally and can be resumed.
"""

import argparse
import json
import os
import sys

import numpy as np


N_X = 32
N_Y = 24
N_IN = N_X * N_Y
DEFAULT_MULTIPLIER = 12
SUPPORTED_PHASE_LEVELS = (4, 16, 32)
RANDOM_BASE_LEVELS = 32
SUPERPIXEL_SIZE = 4
LOGICAL_PIXEL_SIZE = 32
DMD_HEIGHT = N_Y * LOGICAL_PIXEL_SIZE
DMD_WIDTH = N_X * LOGICAL_PIXEL_SIZE
LUT_STEP = 0.01
DEFAULT_SEED = 122026


def build_phase_codebook(phase_levels):
    """Return LUT-effective fields and 32 x 32 DMD tiles for each phase."""
    phase_levels = int(phase_levels)
    if phase_levels not in SUPPORTED_PHASE_LEVELS:
        raise ValueError(
            "phase_levels must be one of {}".format(SUPPORTED_PHASE_LEVELS)
        )

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_dir not in sys.path:
        sys.path.append(project_dir)
    from holograms.generate_LUT import generate_lut

    field_values, pixel_combinations, lut = generate_lut(
        "sp",
        SUPERPIXEL_SIZE,
        step=LUT_STEP,
    )
    lut_center = len(lut) // 2
    requested = np.exp(
        1j
        * 2.0
        * np.pi
        * np.arange(phase_levels, dtype=np.float64)
        / phase_levels
    )
    combination_ids = np.empty(phase_levels, dtype=np.intp)
    effective_fields = np.empty(phase_levels, dtype=np.complex64)
    tiles = np.empty(
        (phase_levels, LOGICAL_PIXEL_SIZE, LOGICAL_PIXEL_SIZE),
        dtype=np.uint8,
    )

    cells_per_logical_pixel = LOGICAL_PIXEL_SIZE // SUPERPIXEL_SIZE
    for phase_index, value in enumerate(requested):
        real_index = int(np.round(np.real(value) / LUT_STEP)) + lut_center
        imag_index = int(np.round(np.imag(value) / LUT_STEP)) + lut_center
        combination_id = int(lut[real_index, imag_index])
        combination_ids[phase_index] = combination_id
        effective_fields[phase_index] = field_values[combination_id]

        tile = np.empty(
            (LOGICAL_PIXEL_SIZE, LOGICAL_PIXEL_SIZE),
            dtype=np.uint8,
        )
        for cell_row in range(cells_per_logical_pixel):
            shift = (SUPERPIXEL_SIZE * cell_row) % (SUPERPIXEL_SIZE**2)
            superpixel = np.roll(
                pixel_combinations[combination_id],
                -shift,
            )
            superpixel = (
                superpixel.reshape(SUPERPIXEL_SIZE, SUPERPIXEL_SIZE).T
                * 255
            ).astype(np.uint8)
            row_start = cell_row * SUPERPIXEL_SIZE
            tile[row_start : row_start + SUPERPIXEL_SIZE] = np.tile(
                superpixel,
                (1, cells_per_logical_pixel),
            )
        tiles[phase_index] = tile

    unique_tile_count = int(np.unique(combination_ids).size)
    if unique_tile_count != phase_levels:
        raise RuntimeError(
            "Requested {} phases but the LUT produced only {} unique tiles".format(
                phase_levels,
                unique_tile_count,
            )
        )
    phase_errors = np.angle(effective_fields * np.conj(requested))
    diagnostics = {
        "unique_lut_tile_count": unique_tile_count,
        "maximum_phase_error_degrees": float(
            np.max(np.abs(np.rad2deg(phase_errors)))
        ),
        "minimum_effective_amplitude": float(np.min(np.abs(effective_fields))),
        "maximum_effective_amplitude": float(np.max(np.abs(effective_fields))),
    }
    return effective_fields, tiles, diagnostics


def encode_phase_indices(phase_indices, phase_tiles):
    """Expand logical phase indices into full-resolution DMD bitmaps."""
    batch = int(phase_indices.shape[0])
    return (
        phase_tiles[phase_indices]
        .transpose(0, 1, 3, 2, 4)
        .reshape(batch, DMD_HEIGHT, DMD_WIDTH)
    )


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")


def generate(output_dir, phase_levels, multiplier, seed, batch_size, resume=False):
    """Generate one reconstruction-ready phase-only probe/pattern dataset."""
    phase_levels = int(phase_levels)
    multiplier = int(multiplier)
    seed = int(seed)
    batch_size = int(batch_size)
    if multiplier <= 0:
        raise ValueError("multiplier must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    probe_path = os.path.join(output_dir, "probe.npy")
    pattern_path = os.path.join(output_dir, "patterns_pregenerated.npy")
    partial_probe_path = os.path.join(output_dir, "probe.partial.npy")
    partial_pattern_path = os.path.join(
        output_dir,
        "patterns_pregenerated.partial.npy",
    )
    progress_path = os.path.join(output_dir, "generation_progress.json")
    metadata_path = os.path.join(output_dir, "metadata.json")

    if os.path.exists(probe_path) or os.path.exists(pattern_path):
        raise FileExistsError(
            "Final output already exists in {}; choose another directory".format(
                output_dir
            )
        )

    target_count = multiplier * N_IN
    probe_shape = (target_count, N_Y, N_X)
    pattern_shape = (target_count, DMD_HEIGHT, DMD_WIDTH)
    completed = 0
    if resume:
        partials_exist = os.path.exists(partial_probe_path) and os.path.exists(
            partial_pattern_path
        )
        if partials_exist and os.path.exists(progress_path):
            with open(progress_path, "r", encoding="utf-8") as file:
                progress = json.load(file)
            if (
                int(progress.get("phase_levels", -1)) != phase_levels
                or int(progress.get("multiplier", -1)) != multiplier
                or int(progress.get("seed", -1)) != seed
            ):
                raise ValueError("Partial dataset settings do not match this run")
            completed = int(progress.get("completed", 0))
            probes = np.load(partial_probe_path, mmap_mode="r+")
            patterns = np.load(partial_pattern_path, mmap_mode="r+")
            if probes.shape != probe_shape or patterns.shape != pattern_shape:
                raise ValueError("Partial dataset shapes do not match this run")
        elif partials_exist or os.path.exists(progress_path):
            raise FileNotFoundError("Resume files are incomplete")
        else:
            resume = False

    if not resume:
        for partial_path in (partial_probe_path, partial_pattern_path):
            if os.path.exists(partial_path):
                raise FileExistsError(
                    "Partial output exists; pass --resume or choose another directory: "
                    + partial_path
                )
        probes = np.lib.format.open_memmap(
            partial_probe_path,
            mode="w+",
            dtype=np.complex64,
            shape=probe_shape,
        )
        patterns = np.lib.format.open_memmap(
            partial_pattern_path,
            mode="w+",
            dtype=np.uint8,
            shape=pattern_shape,
        )

    effective_fields, phase_tiles, diagnostics = build_phase_codebook(
        phase_levels
    )
    rng = np.random.default_rng(seed)
    base_indices = rng.integers(
        0,
        RANDOM_BASE_LEVELS,
        size=probe_shape,
        dtype=np.uint8,
    )
    phase_indices = (
        base_indices.astype(np.uint16) * phase_levels // RANDOM_BASE_LEVELS
    ).astype(np.uint8)

    try:
        for start in range(completed, target_count, batch_size):
            stop = min(start + batch_size, target_count)
            indices = phase_indices[start:stop]
            probes[start:stop] = effective_fields[indices]
            patterns[start:stop] = encode_phase_indices(indices, phase_tiles)
            probes.flush()
            patterns.flush()
            _write_json(
                progress_path,
                {
                    "status": "generating",
                    "completed": stop,
                    "total": target_count,
                    "phase_levels": phase_levels,
                    "multiplier": multiplier,
                    "seed": seed,
                },
            )
            print(
                "\rGenerating {}-phase probes: {}/{}".format(
                    phase_levels,
                    stop,
                    target_count,
                ),
                end="",
                flush=True,
            )
        print()
    finally:
        del probes
        del patterns

    os.replace(partial_probe_path, probe_path)
    os.replace(partial_pattern_path, pattern_path)
    metadata = {
        "logical_shape": [N_Y, N_X],
        "input_shape": [N_Y, N_X],
        "dmd_shape": [DMD_HEIGHT, DMD_WIDTH],
        "n_in": N_IN,
        "probe_multiplier": multiplier,
        "probe_count": target_count,
        "phase_levels": phase_levels,
        "seed": seed,
        "random_base_levels": RANDOM_BASE_LEVELS,
        "paired_random_design": True,
        "probe_representation": "lut_effective_complex_field",
        "reconstruction_ready": True,
        "dataset_purpose": "v4_32x24_phase_level_comparison",
        "phase_only_probe_count": target_count,
        "random_amplitude_probe_count": 0,
        "probe_design_version": "v4_phase_only_paired_levels_v1",
    }
    metadata.update(diagnostics)
    _write_json(metadata_path, metadata)
    _write_json(
        progress_path,
        {"status": "complete", "completed": target_count, "total": target_count},
    )
    print("Dataset generated in: {}".format(output_dir))
    print("  probes:   {} complex64".format(probe_shape))
    print("  patterns: {} uint8".format(pattern_shape))


def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase-levels",
        type=int,
        choices=SUPPORTED_PHASE_LEVELS,
        required=True,
    )
    parser.add_argument("--multiplier", type=int, default=DEFAULT_MULTIPLIER)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    output = args.output or os.path.join(
        project_dir,
        "pregenerated_patterns_v4_phase_only_{}level_{}N".format(
            args.phase_levels,
            args.multiplier,
        ),
    )
    generate(
        output_dir=output,
        phase_levels=args.phase_levels,
        multiplier=args.multiplier,
        seed=args.seed,
        batch_size=args.batch_size,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
