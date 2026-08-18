"""Compare ordinary-pinv and Cholesky GGS21 transmission matrices.

GGS recovery has an arbitrary phase for every output row.  In addition to the
raw matrix difference, this tool therefore reports row-wise correlation after
removing that phase ambiguity.  That is the useful metric for deciding whether
the two solvers recovered physically different input wavefronts.
"""

import argparse
import json
import os
from typing import Dict, Optional

import numpy as np


def _error_curve_summary(path: Optional[str]) -> Optional[Dict]:
    if not path or not os.path.isfile(path):
        return None
    curve = np.load(path, allow_pickle=False)
    if curve.ndim != 1 or curve.size == 0:
        raise ValueError("Error curve must be a non-empty 1-D array: {}".format(path))
    return {
        "path": os.path.abspath(path),
        "iterations": int(curve.size),
        "initial": float(curve[0]),
        "final": float(curve[-1]),
        "minimum": float(np.min(curve)),
    }


def compare_matrices(
    ordinary_path: str,
    cholesky_path: str,
    chunk_rows: int = 512,
) -> Dict:
    """Return raw and phase-invariant comparison metrics for two TMs."""
    ordinary = np.load(ordinary_path, mmap_mode="r", allow_pickle=False)
    cholesky = np.load(cholesky_path, mmap_mode="r", allow_pickle=False)
    if ordinary.ndim != 2 or cholesky.ndim != 2:
        raise ValueError("Both transmission matrices must be 2-D")
    if ordinary.shape != cholesky.shape:
        raise ValueError(
            "TM shape mismatch: ordinary={} cholesky={}".format(
                ordinary.shape, cholesky.shape
            )
        )
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")

    row_correlations = []
    raw_difference_power = 0.0
    ordinary_power = 0.0
    cholesky_power = 0.0
    global_inner_product = 0.0j
    valid_rows = 0
    zero_rows = 0

    for start in range(0, ordinary.shape[0], chunk_rows):
        stop = min(start + chunk_rows, ordinary.shape[0])
        a = np.asarray(ordinary[start:stop], dtype=np.complex128)
        b = np.asarray(cholesky[start:stop], dtype=np.complex128)
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise FloatingPointError(
                "A transmission matrix contains NaN or infinity in rows {}:{}".format(
                    start, stop
                )
            )

        a_power_by_row = np.sum(np.abs(a) ** 2, axis=1)
        b_power_by_row = np.sum(np.abs(b) ** 2, axis=1)
        inner_by_row = np.sum(np.conj(a) * b, axis=1)
        denominator = np.sqrt(a_power_by_row * b_power_by_row)
        valid = denominator > 0
        row_correlations.extend(
            np.clip(
                np.abs(inner_by_row[valid]) / denominator[valid],
                0.0,
                1.0,
            ).tolist()
        )
        valid_rows += int(np.count_nonzero(valid))
        zero_rows += int(valid.size - np.count_nonzero(valid))

        raw_difference_power += float(np.sum(np.abs(a - b) ** 2))
        ordinary_power += float(np.sum(a_power_by_row))
        cholesky_power += float(np.sum(b_power_by_row))
        global_inner_product += complex(np.sum(inner_by_row))

    if not row_correlations:
        raise ValueError("No non-zero row pairs were available for comparison")
    correlations = np.asarray(row_correlations, dtype=np.float64)
    global_denominator = np.sqrt(ordinary_power * cholesky_power)

    return {
        "ordinary_path": os.path.abspath(ordinary_path),
        "cholesky_path": os.path.abspath(cholesky_path),
        "shape": [int(value) for value in ordinary.shape],
        "valid_row_count": valid_rows,
        "zero_row_pair_count": zero_rows,
        "relative_frobenius_difference_raw": float(
            np.sqrt(raw_difference_power / max(ordinary_power, 1e-300))
        ),
        "global_complex_correlation_magnitude": float(
            np.clip(
                abs(global_inner_product) / max(global_denominator, 1e-300),
                0.0,
                1.0,
            )
        ),
        "row_phase_invariant_correlation": {
            "mean": float(np.mean(correlations)),
            "median": float(np.median(correlations)),
            "p05": float(np.percentile(correlations, 5)),
            "minimum": float(np.min(correlations)),
            "maximum": float(np.max(correlations)),
            "rows_below_0_90": int(np.count_nonzero(correlations < 0.90)),
            "rows_below_0_99": int(np.count_nonzero(correlations < 0.99)),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ordinary", default="reconstructed_field.npy")
    parser.add_argument(
        "--cholesky", default="reconstructed_field_cholesky.npy"
    )
    parser.add_argument("--ordinary-error", default="ggs21_error_curve.npy")
    parser.add_argument(
        "--cholesky-error", default="ggs21_error_curve_cholesky.npy"
    )
    parser.add_argument("--output", default="ggs21_recovery_comparison.json")
    parser.add_argument("--chunk-rows", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = compare_matrices(
        args.ordinary,
        args.cholesky,
        chunk_rows=args.chunk_rows,
    )
    result["ordinary_error_curve"] = _error_curve_summary(args.ordinary_error)
    result["cholesky_error_curve"] = _error_curve_summary(args.cholesky_error)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    phase_invariant = result["row_phase_invariant_correlation"]
    print("TM shape: {}".format(tuple(result["shape"])))
    print(
        "Raw relative Frobenius difference: {:.6f}".format(
            result["relative_frobenius_difference_raw"]
        )
    )
    print(
        "Row phase-invariant correlation: mean={:.6f}, median={:.6f}, "
        "p05={:.6f}, min={:.6f}".format(
            phase_invariant["mean"],
            phase_invariant["median"],
            phase_invariant["p05"],
            phase_invariant["minimum"],
        )
    )
    print("Comparison saved to: {}".format(os.path.abspath(args.output)))


if __name__ == "__main__":
    main()
