"""Generate a small 32x24 random amplitude+phase dataset for TM correction.

Unlike the GGS21 probe set, these probes vary both amplitude and phase.  Each
logical input mode occupies one 32x32 DMD block containing repeated 4x4
super-pixel encodings, matching the v4_32x24 optical geometry.

The output directory contains:

- probe.npy                    complex64, [K, 24, 32]
- patterns_pregenerated.npy    uint8,     [K, 768, 1024]
- metadata.json

The default K=512 dataset is intentionally small enough for a dedicated
post-reconstruction correction measurement rather than another full TM scan.
"""

import argparse
import json
import os

import numpy as np

from holograms.generate_LUT import generate_lut


N_X = 32
N_Y = 24
SUPERPIXEL_SIZE = 4
LOGICAL_PIXEL_SIZE = 32
DMD_HEIGHT = N_Y * LOGICAL_PIXEL_SIZE
DMD_WIDTH = N_X * LOGICAL_PIXEL_SIZE
LUT_STEP = 0.01
DEFAULT_COUNT = 512
DEFAULT_SEED = 20260826


def amplitude_levels(count, minimum):
    count = int(count)
    minimum = float(minimum)
    if count <= 0:
        raise ValueError("amplitude level count must be positive")
    if not 0.0 <= minimum <= 1.0:
        raise ValueError("minimum amplitude must be in [0, 1]")
    if count == 1:
        return np.asarray([1.0], dtype=np.float32)
    return np.linspace(minimum, 1.0, count, dtype=np.float32)


def phase_values(count):
    count = int(count)
    if count <= 0:
        raise ValueError("phase level count must be positive")
    phase = np.arange(count, dtype=np.float32) * (2.0 * np.pi / count)
    return np.exp(1j * phase).astype(np.complex64)


def build_complex_tiles(amplitudes, phases):
    """Build one 32x32 binary DMD tile for each amplitude/phase pair."""
    _, pixel_combinations, lut = generate_lut("sp", SUPERPIXEL_SIZE)
    lut_center = len(lut) // 2
    cells_per_axis = LOGICAL_PIXEL_SIZE // SUPERPIXEL_SIZE

    amplitudes = np.asarray(amplitudes, dtype=np.float32)
    phases = np.asarray(phases, dtype=np.complex64)
    tiles = np.empty(
        (
            amplitudes.size,
            phases.size,
            LOGICAL_PIXEL_SIZE,
            LOGICAL_PIXEL_SIZE,
        ),
        dtype=np.uint8,
    )

    for amplitude_index, amplitude in enumerate(amplitudes):
        for phase_index, phase in enumerate(phases):
            value = np.complex64(amplitude) * phase
            re = int(np.round(float(np.real(value)) / LUT_STEP))
            im = int(np.round(float(np.imag(value)) / LUT_STEP))
            re_index = re + lut_center
            im_index = im + lut_center
            if not (0 <= re_index < lut.shape[0] and 0 <= im_index < lut.shape[1]):
                raise ValueError(
                    "complex value {:.4f}{:+.4f}j lies outside the LUT".format(
                        float(np.real(value)), float(np.imag(value))
                    )
                )

            combination_index = lut[re_index, im_index]
            base_superpixel = pixel_combinations[combination_index]
            tile = np.empty(
                (LOGICAL_PIXEL_SIZE, LOGICAL_PIXEL_SIZE), dtype=np.uint8
            )

            for cell_row in range(cells_per_axis):
                shift = (SUPERPIXEL_SIZE * cell_row) % (SUPERPIXEL_SIZE ** 2)
                superpixel = np.roll(base_superpixel, -shift)
                superpixel = (
                    superpixel.reshape(SUPERPIXEL_SIZE, SUPERPIXEL_SIZE)
                    .T.astype(np.uint8)
                    * 255
                )
                row_start = cell_row * SUPERPIXEL_SIZE
                row_stop = row_start + SUPERPIXEL_SIZE
                tile[row_start:row_stop] = np.tile(
                    superpixel, (1, cells_per_axis)
                )

            tiles[amplitude_index, phase_index] = tile

    return tiles


def encode_index_batch(amplitude_indices, phase_indices, tiles):
    """Map logical index grids to full 768x1024 DMD bitmaps."""
    amplitude_indices = np.asarray(amplitude_indices)
    phase_indices = np.asarray(phase_indices)
    if amplitude_indices.shape != phase_indices.shape:
        raise ValueError("amplitude and phase index arrays must have the same shape")
    if amplitude_indices.ndim != 3 or amplitude_indices.shape[1:] != (N_Y, N_X):
        raise ValueError(
            "index arrays must have shape [K, {}, {}]".format(N_Y, N_X)
        )

    # [B, Ny, Nx, tile_y, tile_x] -> [B, Ny*tile_y, Nx*tile_x]
    return (
        tiles[amplitude_indices, phase_indices]
        .transpose(0, 1, 3, 2, 4)
        .reshape(amplitude_indices.shape[0], DMD_HEIGHT, DMD_WIDTH)
    )


def generate(
    output_dir,
    count=DEFAULT_COUNT,
    amplitude_level_count=8,
    minimum_amplitude=0.2,
    phase_level_count=16,
    seed=DEFAULT_SEED,
    batch_size=16,
    overwrite=False,
):
    count = int(count)
    batch_size = int(batch_size)
    if count <= 1:
        raise ValueError("count must be greater than one")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    probe_path = os.path.join(output_dir, "probe.npy")
    pattern_path = os.path.join(output_dir, "patterns_pregenerated.npy")
    metadata_path = os.path.join(output_dir, "metadata.json")

    existing = [path for path in (probe_path, pattern_path, metadata_path) if os.path.exists(path)]
    if existing and not overwrite:
        raise FileExistsError(
            "output already exists; use --overwrite or another directory: {}".format(
                ", ".join(existing)
            )
        )
    if overwrite:
        for path in existing:
            os.remove(path)

    amplitudes = amplitude_levels(amplitude_level_count, minimum_amplitude)
    phases = phase_values(phase_level_count)
    tiles = build_complex_tiles(amplitudes, phases)
    rng = np.random.default_rng(int(seed))

    probes = np.lib.format.open_memmap(
        probe_path,
        mode="w+",
        dtype=np.complex64,
        shape=(count, N_Y, N_X),
    )
    patterns = np.lib.format.open_memmap(
        pattern_path,
        mode="w+",
        dtype=np.uint8,
        shape=(count, DMD_HEIGHT, DMD_WIDTH),
    )

    try:
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            batch = stop - start
            amplitude_index = rng.integers(
                0,
                amplitudes.size,
                size=(batch, N_Y, N_X),
                dtype=np.uint8,
            )
            phase_index = rng.integers(
                0,
                phases.size,
                size=(batch, N_Y, N_X),
                dtype=np.uint8,
            )

            probes[start:stop] = (
                amplitudes[amplitude_index] * phases[phase_index]
            ).astype(np.complex64)
            patterns[start:stop] = encode_index_batch(
                amplitude_index, phase_index, tiles
            )
            print(
                "\rgenerating correction probes: {}/{}".format(stop, count),
                end="",
                flush=True,
            )
        print()
        probes.flush()
        patterns.flush()
    finally:
        del probes
        del patterns

    metadata = {
        "purpose": "post-reconstruction gradient TM correction",
        "logical_shape": [N_Y, N_X],
        "dmd_shape": [DMD_HEIGHT, DMD_WIDTH],
        "count": count,
        "amplitude_levels": [float(value) for value in amplitudes],
        "phase_level_count": int(phases.size),
        "phase_step_rad": float(2.0 * np.pi / phases.size),
        "seed": int(seed),
        "superpixel_size": SUPERPIXEL_SIZE,
        "logical_pixel_size": LOGICAL_PIXEL_SIZE,
        "lut_step": LUT_STEP,
    }
    with open(metadata_path, "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    pattern_gib = count * DMD_HEIGHT * DMD_WIDTH / (1024.0 ** 3)
    print("generated: {}".format(output_dir))
    print("  probes:   {} complex64".format((count, N_Y, N_X)))
    print("  patterns: {} uint8 ({:.2f} GiB)".format(
        (count, DMD_HEIGHT, DMD_WIDTH), pattern_gib
    ))
    print(
        "  amplitude range: {:.3f} .. {:.3f} ({} levels)".format(
            float(amplitudes[0]), float(amplitudes[-1]), amplitudes.size
        )
    )
    print("  phase levels: {}".format(phases.size))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="correction_patterns_32x24_complex",
    )
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--amplitude-levels", type=int, default=8)
    parser.add_argument("--amplitude-min", type=float, default=0.2)
    parser.add_argument("--phase-levels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    generate(
        output_dir=args.output_dir,
        count=args.count,
        amplitude_level_count=args.amplitude_levels,
        minimum_amplitude=args.amplitude_min,
        phase_level_count=args.phase_levels,
        seed=args.seed,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
