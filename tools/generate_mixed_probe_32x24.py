"""Generate an 8N mixed-amplitude/phase probe set for the 32 x 24 profile.

The first 4N probes have unit amplitude and random phase.  The second 4N
probes have zero phase and per-pixel amplitudes sampled from 0.0 through 1.0
in 0.1 steps.  Both the LUT-effective complex probe and its DMD bitmap are
written, so the generated directory can be used directly for measurement.
"""

import argparse
import json
import os
import sys

import numpy as np


N_X, N_Y = 32, 24
N_IN = N_X * N_Y
GROUP_MULTIPLIER = 4
TOTAL_MULTIPLIER = 8
SUPERPIXEL_SIZE = 4
LOGICAL_PIXEL_SIZE = 32
DMD_HEIGHT, DMD_WIDTH = N_Y * LOGICAL_PIXEL_SIZE, N_X * LOGICAL_PIXEL_SIZE
DEFAULT_SEED = 20260910
DEFAULT_BATCH_SIZE = 16
AMPLITUDE_LEVELS = np.arange(11, dtype=np.float32) / 10.0


def _build_lut_assets():
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_dir not in sys.path:
        sys.path.append(project_dir)
    from holograms.generate_LUT import generate_lut

    fields, combinations, lut = generate_lut("sp", SUPERPIXEL_SIZE, step=0.01)
    tiles = np.empty((lut.shape[0], lut.shape[1], LOGICAL_PIXEL_SIZE, LOGICAL_PIXEL_SIZE), dtype=np.uint8)
    center = lut.shape[0] // 2
    cells = LOGICAL_PIXEL_SIZE // SUPERPIXEL_SIZE
    for re_index in range(lut.shape[0]):
        for im_index in range(lut.shape[1]):
            combination = combinations[lut[re_index, im_index]]
            tile = np.empty((LOGICAL_PIXEL_SIZE, LOGICAL_PIXEL_SIZE), dtype=np.uint8)
            for row in range(cells):
                shift = (SUPERPIXEL_SIZE * row) % (SUPERPIXEL_SIZE ** 2)
                superpixel = np.roll(combination, -shift).reshape(SUPERPIXEL_SIZE, SUPERPIXEL_SIZE).T
                start = row * SUPERPIXEL_SIZE
                tile[start:start + SUPERPIXEL_SIZE] = np.tile(superpixel * 255, (1, cells))
            tiles[re_index, im_index] = tile
    return fields.astype(np.complex64), tiles, lut, center


def _lookup(fields, lut, requested, center):
    real = np.rint(np.real(requested) / 0.01).astype(np.int32) + center
    imag = np.rint(np.imag(requested) / 0.01).astype(np.int32) + center
    return fields[lut[np.clip(real, 0, lut.shape[0] - 1), np.clip(imag, 0, lut.shape[1] - 1)]]


def _encode(requested, tiles, center):
    real = np.rint(np.real(requested) / 0.01).astype(np.int32) + center
    imag = np.rint(np.imag(requested) / 0.01).astype(np.int32) + center
    real = np.clip(real, 0, tiles.shape[0] - 1)
    imag = np.clip(imag, 0, tiles.shape[1] - 1)
    selected = tiles[real, imag]
    return selected.transpose(0, 1, 3, 2, 4).reshape(
        requested.shape[0], DMD_HEIGHT, DMD_WIDTH
    )


def generate(output_dir, seed=DEFAULT_SEED, batch_size=DEFAULT_BATCH_SIZE):
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    probe_path = os.path.join(output_dir, "probe.npy")
    pattern_path = os.path.join(output_dir, "patterns_pregenerated.npy")
    if os.path.exists(probe_path) or os.path.exists(pattern_path):
        raise FileExistsError("Output already exists; choose another directory")

    count = TOTAL_MULTIPLIER * N_IN
    probes = np.lib.format.open_memmap(
        probe_path + ".partial.npy", mode="w+", dtype=np.complex64,
        shape=(count, N_Y, N_X)
    )
    patterns = np.lib.format.open_memmap(
        pattern_path + ".partial.npy", mode="w+", dtype=np.uint8,
        shape=(count, DMD_HEIGHT, DMD_WIDTH)
    )
    lut_fields, tiles, lut, center = _build_lut_assets()
    rng = np.random.default_rng(int(seed))
    try:
        for start in range(0, count, int(batch_size)):
            stop = min(start + int(batch_size), count)
            size = stop - start
            if start < GROUP_MULTIPLIER * N_IN:
                requested = np.exp(1j * rng.uniform(-np.pi, np.pi, (size, N_Y, N_X)))
                group = "fixed_amplitude_random_phase"
            else:
                amplitudes = AMPLITUDE_LEVELS[rng.integers(0, len(AMPLITUDE_LEVELS), (size, N_Y, N_X))]
                requested = amplitudes.astype(np.complex64)
                group = "fixed_phase_random_amplitude"
            requested = requested.astype(np.complex64)
            probes[start:stop] = _lookup(lut_fields, lut, requested, center)
            patterns[start:stop] = _encode(requested, tiles, center)
            probes.flush()
            patterns.flush()
            print(f"\rGenerating mixed 8N probes: {stop}/{count} ({group})", end="", flush=True)
    finally:
        del probes
        del patterns
    os.replace(probe_path + ".partial.npy", probe_path)
    os.replace(pattern_path + ".partial.npy", pattern_path)
    metadata = {
        "logical_shape": [N_Y, N_X],
        "dmd_shape": [DMD_HEIGHT, DMD_WIDTH],
        "probe_count": count,
        "probe_multiplier": TOTAL_MULTIPLIER,
        "group_counts": {"fixed_amplitude_random_phase": GROUP_MULTIPLIER * N_IN,
                         "fixed_phase_random_amplitude": GROUP_MULTIPLIER * N_IN},
        "fixed_amplitude": 1.0,
        "fixed_phase_rad": 0.0,
        "amplitude_levels": AMPLITUDE_LEVELS.tolist(),
        "seed": int(seed),
        "probe_representation": "lut_effective_complex_field",
        "reconstruction_ready": True,
        "probe_design_version": "v4_8N_mixed_phase_amplitude_v1",
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"\nDataset generated in: {output_dir}")


def main():
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=os.path.join(project_dir, "pregenerated_patterns_8N_mixed"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    generate(args.output, args.seed, args.batch_size)


if __name__ == "__main__":
    main()
