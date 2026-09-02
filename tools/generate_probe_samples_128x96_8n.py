"""Generate the 4x 128 x 96 -> 128 x 128 calibration dataset at 8N.

Each 128 x 96 random phase field is repeated 2 x 2 on the complete 256 x 192
optical-superpixel grid before 4 x 4 holo_SP encoding.  The resulting
1024 x 768 patterns have no outer zero-padding area.  Generation is batched,
memory-mapped, disk-space checked, and resumable because the Probe + Pattern
dataset occupies about 81 GiB.
"""

import argparse
import json
import os
import shutil

import numpy as np

from dmd_pattern_128x96 import (
    ACTIVE_HEIGHT,
    ACTIVE_WIDTH,
    ACTIVE_X,
    ACTIVE_Y,
    DMD_HEIGHT,
    DMD_WIDTH,
    EXPANDED_HEIGHT,
    EXPANDED_WIDTH,
    HOLOGRAM_SUPERPIXEL_SIZE,
    INPUT_HEIGHT,
    INPUT_MACRO_PIXEL_SIZE,
    INPUT_WIDTH,
    MAPPING_VERSION,
    SOURCE_X_INDICES,
    SOURCE_Y_INDICES,
    get_superpixel_lut,
    source_repeat_counts,
)


N_IN = INPUT_HEIGHT * INPUT_WIDTH
PROBE_MULTIPLIER = 8
PROBE_COUNT = PROBE_MULTIPLIER * N_IN
PHASE_LEVELS = 16
PHASE_STEP = 2 * np.pi / PHASE_LEVELS
DEFAULT_SEED = 12804
DEFAULT_BATCH_SIZE = 64
DEFAULT_CHECKPOINT_INTERVAL = 1024
DEFAULT_OUTPUT_DIR = "pregenerated_patterns_128x96_fill_8N_full"


def write_json_atomic(path, payload):
    temporary_path = path + ".partial"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary_path, path)


def build_phase_tiles():
    """Build exact 4 x 4 holo_SP tiles for every expanded row and phase."""
    _, pixel_combinations, lut = get_superpixel_lut(HOLOGRAM_SUPERPIXEL_SIZE)
    lut_center = len(lut) // 2
    phase_values = np.exp(
        1j * np.arange(PHASE_LEVELS, dtype=np.float32) * PHASE_STEP
    ).astype(np.complex64)
    tiles = np.empty(
        (
            EXPANDED_HEIGHT,
            PHASE_LEVELS,
            HOLOGRAM_SUPERPIXEL_SIZE,
            HOLOGRAM_SUPERPIXEL_SIZE,
        ),
        dtype=np.uint8,
    )

    for row in range(EXPANDED_HEIGHT):
        shift = (HOLOGRAM_SUPERPIXEL_SIZE * row) % (
            HOLOGRAM_SUPERPIXEL_SIZE**2
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


def expand_phase_indices(phase_indices):
    """Repeat source phase indices onto the complete 192 x 256 SP grid."""
    phase_indices = np.asarray(phase_indices)
    expected_shape = (INPUT_HEIGHT, INPUT_WIDTH)
    if phase_indices.ndim != 3 or phase_indices.shape[1:] != expected_shape:
        raise ValueError(
            f"Phase-index batch shape {phase_indices.shape} must be "
            f"(batch, {expected_shape[0]}, {expected_shape[1]})"
        )
    return phase_indices[:, SOURCE_Y_INDICES, :][:, :, SOURCE_X_INDICES]


def encode_phase_indices(phase_indices, phase_tiles):
    """Encode one source batch into full 768 x 1024 DMD patterns."""
    expanded = expand_phase_indices(phase_indices)
    batch_size = expanded.shape[0]
    row_indices = np.arange(EXPANDED_HEIGHT)[None, :, None]
    patterns = phase_tiles[row_indices, expanded]
    return patterns.transpose(0, 1, 3, 2, 4).reshape(
        batch_size,
        DMD_HEIGHT,
        DMD_WIDTH,
    )


def dataset_required_bytes():
    """Return the exact Probe + Pattern storage required by the 8N dataset."""
    return (
        PROBE_COUNT
        * INPUT_HEIGHT
        * INPUT_WIDTH
        * np.dtype(np.complex64).itemsize
        + PROBE_COUNT
        * DMD_HEIGHT
        * DMD_WIDTH
        * np.dtype(np.uint8).itemsize
    )


def open_dataset(output_dir, resume, seed):
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
            raise ValueError(
                f"Incompatible probe array: {probes.shape}, {probes.dtype}"
            )
        if patterns.shape != pattern_shape or patterns.dtype != np.uint8:
            raise ValueError(
                f"Incompatible pattern array: {patterns.shape}, {patterns.dtype}"
            )
        with open(progress_path, "r", encoding="utf-8") as handle:
            progress = json.load(handle)
        if int(progress.get("random_seed", seed)) != int(seed):
            raise ValueError(
                f"Resume seed {seed} does not match saved seed "
                f"{progress.get('random_seed')}"
            )
        return probes, patterns, progress, probe_path, pattern_path, progress_path

    existing = [
        path
        for path in (probe_path, pattern_path, progress_path)
        if os.path.exists(path)
    ]
    if existing:
        raise FileExistsError(
            "Output exists; pass --resume or choose another directory: "
            + ", ".join(existing)
        )

    required_bytes = dataset_required_bytes()
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
    progress = {
        "completed": 0,
        "total": PROBE_COUNT,
        "status": "running",
        "random_seed": int(seed),
    }
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
    ) = open_dataset(output_dir, resume, seed)

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
                    "status": (
                        "complete" if completed == PROBE_COUNT else "running"
                    ),
                    "random_seed": int(seed),
                    "rng_state": rng.bit_generator.state,
                }
                write_json_atomic(progress_path, progress)
                print(
                    f"Generated {completed}/{PROBE_COUNT} probe patterns",
                    flush=True,
                )
                next_checkpoint = min(
                    PROBE_COUNT,
                    next_checkpoint + checkpoint_interval,
                )
    finally:
        del probes
        del patterns

    x_repeats = source_repeat_counts("x")
    y_repeats = source_repeat_counts("y")
    metadata = {
        "dataset_purpose": "full_128x96_to_128x128_calibration_8N",
        "probe_count": PROBE_COUNT,
        "probe_multiplier": PROBE_MULTIPLIER,
        "random_seed": int(seed),
        "input_shape": [INPUT_HEIGHT, INPUT_WIDTH],
        "expanded_input_shape": [EXPANDED_HEIGHT, EXPANDED_WIDTH],
        "expansion_method": "aligned_repeat_2x2",
        "input_macro_pixel_size": INPUT_MACRO_PIXEL_SIZE,
        "hologram_superpixel_size": HOLOGRAM_SUPERPIXEL_SIZE,
        "source_repeat_range_xy": [
            [int(x_repeats.min()), int(x_repeats.max())],
            [int(y_repeats.min()), int(y_repeats.max())],
        ],
        "active_shape": [ACTIVE_HEIGHT, ACTIVE_WIDTH],
        "active_offset_xy": [ACTIVE_X, ACTIVE_Y],
        "dmd_shape": [DMD_HEIGHT, DMD_WIDTH],
        "camera_output_shape": [128, 128],
        "probe_dtype": "complex64",
        "pattern_dtype": "uint8",
        "phase_levels": PHASE_LEVELS,
        "mapping_version": MAPPING_VERSION,
        "zero_padding": False,
        "reconstruction_ready": True,
        "note": (
            "Each 128x96 source sample is repeated 2x2 on the 256x192 "
            "optical-superpixel grid. Each repeated sample is encoded by one "
            "4x4 holo_SP tile, filling the complete 1024x768 DMD."
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
