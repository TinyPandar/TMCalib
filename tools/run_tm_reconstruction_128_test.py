"""Run a small, non-destructive TM reconstruction on measured 128-grid data.

By default this reconstructs 64 contiguous camera pixels centred on row 64.
It uses the same exact Cholesky projection as the full reconstruction, but
writes separate test files and never replaces the production TM.
"""

import argparse
import json
import os

import numpy as np

from tm_reconstruction_128 import ReconstructionConfig, reconstruct_tm


CAMERA_WIDTH = 128
CAMERA_HEIGHT = 128
DEFAULT_COUNT = 64
DEFAULT_ITERATIONS = 40


def _default_start(count: int) -> int:
    if count <= CAMERA_WIDTH:
        y = CAMERA_HEIGHT // 2
        x = (CAMERA_WIDTH - count) // 2
        return y * CAMERA_WIDTH + x
    return (CAMERA_WIDTH * CAMERA_HEIGHT - count) // 2


def _describe_selection(start: int, count: int) -> str:
    stop = start + count
    first_y, first_x = divmod(start, CAMERA_WIDTH)
    last_y, last_x = divmod(stop - 1, CAMERA_WIDTH)
    return (
        "flattened indices [{}:{}) | first (x={}, y={}) | last (x={}, y={})"
        .format(start, stop, first_x, first_y, last_x, last_y)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help="Number of contiguous camera pixels to reconstruct (default: 64)",
    )
    parser.add_argument(
        "--start",
        type=int,
        help="First flattened camera-pixel index; default centres the selection",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help="GGS2-1 iterations (default: 40; full reconstruction uses 200)",
    )
    parser.add_argument(
        "--output",
        default="reconstructed_field_128_px4_active512_test_subset.npy",
    )
    parser.add_argument(
        "--error-curve",
        default="ggs21_error_curve_128_px4_active512_test_subset.npy",
    )
    parser.add_argument(
        "--metadata",
        default="tm_reconstruction_128_px4_active512_test_subset.json",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--solver",
        choices=("cholesky", "complex32_pinv"),
        default="cholesky",
    )
    parser.add_argument(
        "--pinv-real", default="probe_pinv_128_px4_active512_fp16_real.npy"
    )
    parser.add_argument(
        "--pinv-imag", default="probe_pinv_128_px4_active512_fp16_imag.npy"
    )
    parser.add_argument(
        "--pinv-metadata", default="probe_pinv_128_px4_active512_fp16.json"
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = int(args.count)
    if count <= 0 or count > CAMERA_WIDTH * CAMERA_HEIGHT:
        raise ValueError("count must be in [1, 16384]")
    start = _default_start(count) if args.start is None else int(args.start)
    if start < 0 or start + count > CAMERA_WIDTH * CAMERA_HEIGHT:
        raise ValueError("The selected camera-pixel range is outside 128x128")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("Partial TM test")
    print("Selection: {}".format(_describe_selection(start, count)))
    print("Iterations: {}".format(args.iterations))
    print(
        "The production reconstructed_field_128_px4_active512.npy will not be changed."
    )

    config = ReconstructionConfig(
        measurement_path=os.path.join(
            base_dir, "measurements_128_px4_active512_full_memmap.npy"
        ),
        probe_path=os.path.join(
            base_dir,
            "pregenerated_patterns_128_px4_active512_full",
            "probe.npy",
        ),
        output_path=os.path.join(base_dir, args.output),
        error_curve_path=os.path.join(base_dir, args.error_curve),
        metadata_path=os.path.join(base_dir, args.metadata),
        cholesky_cache_path=os.path.join(
            base_dir, "probe_cholesky_128_px4_active512.npy"
        ),
        pinv_real_path=(
            args.pinv_real
            if os.path.isabs(args.pinv_real)
            else os.path.join(base_dir, args.pinv_real)
        ),
        pinv_imag_path=(
            args.pinv_imag
            if os.path.isabs(args.pinv_imag)
            else os.path.join(base_dir, args.pinv_imag)
        ),
        pinv_metadata_path=(
            args.pinv_metadata
            if os.path.isabs(args.pinv_metadata)
            else os.path.join(base_dir, args.pinv_metadata)
        ),
        iterations=int(args.iterations),
        # A shorter GS-2 stage gives the 40-iteration smoke test enough GS-1
        # iterations to show convergence. Production remains 0.89 / 200.
        gs2_ratio=0.75,
        output_chunk_size=min(count, 128),
        solver=args.solver,
        device=args.device,
        output_start=start,
        output_count=count,
        normalize_tm=True,
        resume=not args.no_resume,
    )

    def show_progress(percent: float, message: str) -> None:
        print("[{:5.1f}%] {}".format(percent, message), flush=True)

    result = reconstruct_tm(config, progress=show_progress)
    tm = np.load(result["output_path"], mmap_mode="r")
    finite = bool(np.all(np.isfinite(tm)))
    actual_shape = tuple(tm.shape)
    del tm

    print("\nTest result")
    print("Output: {}".format(result["output_path"]))
    print("Shape: {} complex64".format(actual_shape))
    print(
        "Relative amplitude error: {:.4f} -> {:.4f}".format(
            result["initial_relative_amplitude_error"],
            result["final_relative_amplitude_error"],
        )
    )
    print("Elapsed: {:.1f} s".format(result["elapsed_seconds"]))
    print("Finite values: {}".format(finite))

    if (
        not finite
        or result["final_relative_amplitude_error"]
        >= result["initial_relative_amplitude_error"]
    ):
        raise RuntimeError(
            "Partial TM test did not pass its convergence/finite-value check"
        )

    with open(args.metadata if os.path.isabs(args.metadata) else os.path.join(base_dir, args.metadata), "r", encoding="utf-8") as handle:
        json.load(handle)
    print("PASS: the partial reconstruction path is ready for a full run.")


if __name__ == "__main__":
    main()
