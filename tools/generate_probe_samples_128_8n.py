"""Generate the aligned 128 x 128, px=4 calibration dataset at 8N.

The generator uses the same seed and random stream as the existing 4N set, so
the first 65536 probes/patterns are identical. Generation is batched and
memory-mapped because the finished dataset occupies about 112 GiB.
"""

import argparse
import json
import os
import shutil

import numpy as np

from dmd_pattern_128 import (
    ACTIVE_HEIGHT,
    ACTIVE_WIDTH,
    ACTIVE_X,
    ACTIVE_Y,
    DMD_HEIGHT,
    DMD_WIDTH,
    HOLOGRAM_SUPERPIXEL_SIZE,
    INPUT_HEIGHT,
    INPUT_MACRO_PIXEL_SIZE,
    INPUT_WIDTH,
    get_superpixel_lut,
)


N_IN = INPUT_HEIGHT * INPUT_WIDTH
PROBE_MULTIPLIER = 8
PROBE_COUNT = PROBE_MULTIPLIER * N_IN
PHASE_LEVELS = 16
PHASE_STEP = np.pi / 8
DEFAULT_SEED = 12804
DEFAULT_BATCH_SIZE = 64
DEFAULT_CHECKPOINT_INTERVAL = 1024
DEFAULT_OUTPUT_DIR = "pregenerated_patterns_128_px4_active512_8N_full"


def write_json_atomic(path, payload):
    temporary_path = path + ".partial"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary_path, path)


def build_phase_tiles():
    """Build exact 4 x 4 holo_SP tiles for every row and phase level."""
    _, pixel_combinations, lut = get_superpixel_lut(HOLOGRAM_SUPERPIXEL_SIZE)
    lut_center = len(lut) // 2
    phase_values = np.exp(
        1j * np.arange(PHASE_LEVELS, dtype=np.float32) * PHASE_STEP
    ).astype(np.complex64)
    tiles = np.empty(
        (
            INPUT_HEIGHT,
            PHASE_LEVELS,
            HOLOGRAM_SUPERPIXEL_SIZE,
            HOLOGRAM_SUPERPIXEL_SIZE,
        ),
        dtype=np.uint8,
    )

    for row in range(INPUT_HEIGHT):
        shift = (HOLOGRAM_SUPERPIXEL_SIZE * row) % (
            HOLOGRAM_SUPERPIXEL_SIZE ** 2
        )
        for phase_index, value in enumerate(phase_values):
            re = int(np.round(np.real(value) / 0.01))
            im = int(np.round(np.imag(value) / 0.01))
            superpixel = np.roll(
                pixel_combinations[lut[re + lut_center, im + lut_center]],
                -shift,
            )
            tiles[row, phase_index] = (
                superpixel.reshape(
                    HOLOGRAM_SUPERPIXEL_SIZE,
                    HOLOGRAM_SUPERPIXEL_SIZE,
                ).T.astype(np.uint8)
                * 255
            )

    return phase_values, tiles


def encode_phase_indices(phase_indices, phase_tiles):
    """Encode a batch into centred full-size 768 x 1024 DMD patterns."""
    batch_size = phase_indices.shape[0]
    row_indices = np.arange(INPUT_HEIGHT)[None, :, None]
    active = phase_tiles[row_indices, phase_indices]
    active = active.transpose(0, 1, 3, 2, 4).reshape(
        batch_size,
        ACTIVE_HEIGHT,
        ACTIVE_WIDTH,
    )
    patterns = np.zeros(
        (batch_size, DMD_HEIGHT, DMD_WIDTH),
        dtype=np.uint8,
    )
    patterns[
        :,
        ACTIVE_Y : ACTIVE_Y + ACTIVE_HEIGHT,
        ACTIVE_X : ACTIVE_X + ACTIVE_WIDTH,
    ] = active
    return patterns


def open_dataset(output_dir, resume):
    probe_path = os.path.join(output_dir, "probe.npy")
    pattern_path = os.path.join(output_dir, "patterns_pregenerated.npy")
    progress_path = os.path.join(output_dir, "generation_progress.json")
    probe_shape = (PROBE_COUNT, INPUT_HEIGHT, INPUT_WIDTH)
    pattern_shape = (PROBE_COUNT, DMD_HEIGHT, DMD_WIDTH)

    if resume and all(
        os.path.exists(path) for path in (probe_path, pattern_path, progress_path)
    ):
        probes = np.load(probe_path, mmap_mode="r+")
        patterns = np.load(pattern_path, mmap_mode="r+")
        if probes.shape != probe_shape or probes.dtype != np.complex64:
            raise ValueError(f"Incompatible probe array: {probes.shape}, {probes.dtype}")
        if patterns.shape != pattern_shape or patterns.dtype != np.uint8:
            raise ValueError(
                f"Incompatible pattern array: {patterns.shape}, {patterns.dtype}"
            )
        with open(progress_path, "r", encoding="utf-8") as handle:
            progress = json.load(handle)
        return probes, patterns, progress, probe_path, pattern_path, progress_path

    existing = [
        path for path in (probe_path, pattern_path, progress_path) if os.path.exists(path)
    ]
    if existing:
        raise FileExistsError(
            "Output exists; pass --resume or choose another directory: "
            + ", ".join(existing)
        )

    required_bytes = (
        PROBE_COUNT * INPUT_HEIGHT * INPUT_WIDTH * np.dtype(np.complex64).itemsize
        + PROBE_COUNT * DMD_HEIGHT * DMD_WIDTH * np.dtype(np.uint8).itemsize
    )
    free_bytes = shutil.disk_usage(output_dir).free
    if free_bytes < required_bytes * 1.05:
        raise OSError(
            f"Insufficient disk space: need about {required_bytes / 2**30:.1f} GiB, "
            f"have {free_bytes / 2**30:.1f} GiB"
        )

    probes = np.lib.format.open_memmap(
        probe_path,
        mode="w+",
        dtype=np.complex64,
        shape=probe_shape,
    )
    patterns = np.lib.format.open_memmap(
        pattern_path,
        mode="w+",
        dtype=np.uint8,
        shape=pattern_shape,
    )
    progress = {"completed": 0, "total": PROBE_COUNT, "status": "running"}
    return probes, patterns, progress, probe_path, pattern_path, progress_path


def generate_dataset(
    output_dir,
    seed=DEFAULT_SEED,
    batch_size=DEFAULT_BATCH_SIZE,
    checkpoint_interval=DEFAULT_CHECKPOINT_INTERVAL,
    resume=False,
):
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.json")
    (
        probes,
        patterns,
        progress,
        probe_path,
        pattern_path,
        progress_path,
    ) = open_dataset(output_dir, resume)

    completed = int(progress.get("completed", 0))
    if completed < 0 or completed > PROBE_COUNT:
        raise ValueError(f"Invalid completed count: {completed}")
    rng = np.random.default_rng(seed)
    if completed:
        saved_state = progress.get("rng_state")
        if saved_state is None:
            raise ValueError("Resume state does not contain rng_state")
        rng.bit_generator.state = saved_state
        print(f"Resuming at {completed}/{PROBE_COUNT}")

    phase_values, phase_tiles = build_phase_tiles()
    next_checkpoint = min(
        PROBE_COUNT,
        ((completed // checkpoint_interval) + 1) * checkpoint_interval,
    )
    try:
        while completed < PROBE_COUNT:
            stop = min(completed + batch_size, next_checkpoint, PROBE_COUNT)
            phase_indices = rng.integers(
                0,
                PHASE_LEVELS,
                size=(stop - completed, INPUT_HEIGHT, INPUT_WIDTH),
                dtype=np.uint8,
            )
            probes[completed:stop] = phase_values[phase_indices]
            patterns[completed:stop] = encode_phase_indices(
                phase_indices,
                phase_tiles,
            )
            completed = stop

            if completed == next_checkpoint or completed == PROBE_COUNT:
                probes.flush()
                patterns.flush()
                progress = {
                    "completed": completed,
                    "total": PROBE_COUNT,
                    "status": "complete" if completed == PROBE_COUNT else "running",
                    "rng_state": rng.bit_generator.state,
                }
                write_json_atomic(progress_path, progress)
                print(f"Generated {completed}/{PROBE_COUNT} probe patterns", flush=True)
                next_checkpoint = min(
                    PROBE_COUNT,
                    next_checkpoint + checkpoint_interval,
                )
    finally:
        del probes
        del patterns

    metadata = {
        "dataset_purpose": "full_128x128_calibration_8N",
        "probe_count": PROBE_COUNT,
        "probe_multiplier": PROBE_MULTIPLIER,
        "random_seed": seed,
        "input_shape": [INPUT_HEIGHT, INPUT_WIDTH],
        "input_macro_pixel_size": INPUT_MACRO_PIXEL_SIZE,
        "hologram_superpixel_size": HOLOGRAM_SUPERPIXEL_SIZE,
        "active_shape": [ACTIVE_HEIGHT, ACTIVE_WIDTH],
        "active_offset_xy": [ACTIVE_X, ACTIVE_Y],
        "dmd_shape": [DMD_HEIGHT, DMD_WIDTH],
        "probe_dtype": "complex64",
        "pattern_dtype": "uint8",
        "phase_levels": PHASE_LEVELS,
        "mapping_version": "aligned_active512_v1",
        "reconstruction_ready": True,
        "first_4N_matches_seed_12804_dataset": seed == DEFAULT_SEED,
        "note": (
            "Each 128x128 logical input maps one-to-one to a 4x4 DMD "
            "hologram superpixel. The central 512x512 region is active."
        ),
    }
    write_json_atomic(metadata_path, metadata)
    print(f"8N dataset generated in: {output_dir}")
    return probe_path, pattern_path, metadata_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.checkpoint_interval <= 0:
        parser.error("batch size and checkpoint interval must be positive")
    generate_dataset(
        args.output_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
