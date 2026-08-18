"""Generate aligned 128 x 128, px=4 calibration patterns.

Each logical input maps to one physical 4 x 4 DMD superpixel, so the active
binary hologram is 512 x 512 and is centred on the 1024 x 768 DMD canvas.
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
    input_field_to_dmd_pattern,
)


DEFAULT_TEST_OUTPUT_DIR = "pregenerated_patterns_128_px4_active512_test"
DEFAULT_FULL_OUTPUT_DIR = "pregenerated_patterns_128_px4_active512_full"
DEFAULT_OUTPUT_DIR = DEFAULT_TEST_OUTPUT_DIR
DEFAULT_COUNT = 64
DEFAULT_SEED = 12804
FULL_COUNT = 4 * INPUT_HEIGHT * INPUT_WIDTH


def _write_json(path, payload):
    temporary_path = path + ".partial"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary_path, path)


def _open_or_create_memmaps(output_dir, count, resume):
    probe_path = os.path.join(output_dir, "probe.npy")
    pattern_path = os.path.join(output_dir, "patterns_pregenerated.npy")
    progress_path = os.path.join(output_dir, "generation_progress.json")

    expected_probe_shape = (count, INPUT_HEIGHT, INPUT_WIDTH)
    expected_pattern_shape = (count, DMD_HEIGHT, DMD_WIDTH)

    if resume and all(
        os.path.exists(path) for path in (probe_path, pattern_path, progress_path)
    ):
        probes = np.load(probe_path, mmap_mode="r+")
        patterns = np.load(pattern_path, mmap_mode="r+")
        if probes.shape != expected_probe_shape or probes.dtype != np.complex64:
            raise ValueError(f"Existing probe file is incompatible: {probes.shape}, {probes.dtype}")
        if patterns.shape != expected_pattern_shape or patterns.dtype != np.uint8:
            raise ValueError(
                f"Existing pattern file is incompatible: {patterns.shape}, {patterns.dtype}"
            )
        with open(progress_path, "r", encoding="utf-8") as handle:
            progress = json.load(handle)
        return probes, patterns, progress, probe_path, pattern_path, progress_path

    required_bytes = (
        count * INPUT_HEIGHT * INPUT_WIDTH * np.dtype(np.complex64).itemsize
        + count * DMD_HEIGHT * DMD_WIDTH * np.dtype(np.uint8).itemsize
    )
    free_bytes = shutil.disk_usage(output_dir).free
    if free_bytes < required_bytes * 1.05:
        raise OSError(
            f"Insufficient free disk space: need about {required_bytes / 2**30:.1f} GiB, "
            f"have {free_bytes / 2**30:.1f} GiB"
        )

    probes = np.lib.format.open_memmap(
        probe_path,
        mode="w+",
        dtype=np.complex64,
        shape=expected_probe_shape,
    )
    patterns = np.lib.format.open_memmap(
        pattern_path,
        mode="w+",
        dtype=np.uint8,
        shape=expected_pattern_shape,
    )
    progress = {"completed": 0, "status": "running"}
    return probes, patterns, progress, probe_path, pattern_path, progress_path


def generate_dataset(output_dir, count=DEFAULT_COUNT, seed=DEFAULT_SEED, resume=False):
    count = int(count)
    if count <= 0:
        raise ValueError("count must be positive")

    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.json")
    (
        probes,
        patterns,
        progress,
        probe_path,
        pattern_path,
        progress_path,
    ) = _open_or_create_memmaps(
        output_dir,
        count,
        resume,
    )

    rng = np.random.default_rng(seed)
    completed = int(progress.get("completed", 0))
    if completed < 0 or completed > count:
        raise ValueError(f"Invalid completed count in progress file: {completed}")
    if completed:
        saved_state = progress.get("rng_state")
        if saved_state is None:
            raise ValueError("Resume progress file does not contain rng_state")
        rng.bit_generator.state = saved_state
        print(f"Resuming at pattern {completed}/{count}")

    lut_cache = get_superpixel_lut(HOLOGRAM_SUPERPIXEL_SIZE)

    checkpoint_interval = 64
    for index in range(completed, count):
        # Match the existing calibration convention: 16 equally spaced phases.
        phase_index = rng.integers(
            0,
            16,
            size=(INPUT_HEIGHT, INPUT_WIDTH),
            dtype=np.uint8,
        )
        field = np.exp(1j * phase_index.astype(np.float32) * np.pi / 8).astype(
            np.complex64
        )
        probes[index] = field
        patterns[index] = input_field_to_dmd_pattern(
            field,
            px=HOLOGRAM_SUPERPIXEL_SIZE,
            ds_method="mean",
            lut_cache=lut_cache,
        )
        if (index + 1) % checkpoint_interval == 0 or index + 1 == count:
            probes.flush()
            patterns.flush()
            progress = {
                "completed": index + 1,
                "total": count,
                "status": "complete" if index + 1 == count else "running",
                "rng_state": rng.bit_generator.state,
            }
            _write_json(progress_path, progress)
            print(f"Generated {index + 1}/{count} probe patterns")

    probes.flush()
    patterns.flush()
    del probes
    del patterns

    metadata = {
        "dataset_purpose": (
            "full_128x128_calibration"
            if count >= FULL_COUNT
            else "optical_path_test_only"
        ),
        "probe_count": count,
        "random_seed": seed,
        "input_shape": [INPUT_HEIGHT, INPUT_WIDTH],
        "input_macro_pixel_size": INPUT_MACRO_PIXEL_SIZE,
        "hologram_superpixel_size": HOLOGRAM_SUPERPIXEL_SIZE,
        "active_shape": [ACTIVE_HEIGHT, ACTIVE_WIDTH],
        "active_offset_xy": [ACTIVE_X, ACTIVE_Y],
        "dmd_shape": [DMD_HEIGHT, DMD_WIDTH],
        "probe_dtype": "complex64",
        "pattern_dtype": "uint8",
        "phase_levels": 16,
        "mapping_version": "aligned_active512_v1",
        "reconstruction_ready": count >= FULL_COUNT,
        "note": (
            "Each 128x128 logical input is aligned one-to-one with a 4x4 "
            "DMD hologram superpixel. The active region is 512x512 and is "
            "centred on the 1024x768 canvas. Full reconstruction requires "
            "at least 65536 probes."
        ),
    }
    _write_json(metadata_path, metadata)

    return probe_path, pattern_path, metadata_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--full", action="store_true", help=f"Generate {FULL_COUNT} probes")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    count = FULL_COUNT if args.full else args.count
    output_dir = args.output_dir or (
        DEFAULT_FULL_OUTPUT_DIR if args.full else DEFAULT_TEST_OUTPUT_DIR
    )
    generate_dataset(
        output_dir,
        count=count,
        seed=args.seed,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
