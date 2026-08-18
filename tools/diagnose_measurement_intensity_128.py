"""Reproducible intensity/SNR diagnostics for the full 128x128 measurement.

The measurement file is a headerless uint16 memmap with shape
(65536, 128, 128). Camera samples originate as Polarized8 and are range-scaled
to 12-bit storage, so the report includes native 8-bit-equivalent statistics.
"""

import argparse
import csv
import json
import os
from datetime import datetime

import numpy as np
from scipy.stats import pearsonr, spearmanr


MEASUREMENT_SHAPE = (65536, 128, 128)
RAW_8BIT_TO_STORED = np.asarray(
    [int(value / 255.0 * 4095.0) for value in range(256)],
    dtype=np.int64,
)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _quantile_from_histogram(histogram, probability):
    cumulative = np.cumsum(histogram)
    return int(
        np.searchsorted(
            cumulative,
            float(probability) * int(cumulative[-1]),
            side="left",
        )
    )


def _native_histogram(stored_histogram):
    result = np.zeros(256, dtype=np.int64)
    for raw_value, stored_value in enumerate(RAW_8BIT_TO_STORED):
        result[raw_value] += int(stored_histogram[stored_value])
    return result


def _native_histogram_bins(native_histogram):
    definitions = (
        ("0", 0, 0),
        ("1", 1, 1),
        ("2", 2, 2),
        ("3-4", 3, 4),
        ("5-8", 5, 8),
        ("9-16", 9, 16),
        ("17-32", 17, 32),
        ("33+", 33, 255),
    )
    total = int(native_histogram.sum())
    return [
        {
            "native_code_bin": label,
            "lower_code": start,
            "upper_code": stop,
            "sample_count": int(native_histogram[start : stop + 1].sum()),
            "sample_fraction": float(
                native_histogram[start : stop + 1].sum() / total
            ),
        }
        for label, start, stop in definitions
    ]


def _scan_measurement(path):
    expected_bytes = int(np.prod(MEASUREMENT_SHAPE)) * np.dtype(np.uint16).itemsize
    actual_bytes = os.path.getsize(path)
    if actual_bytes != expected_bytes:
        raise ValueError(
            "Measurement size {} does not match expected {}".format(
                actual_bytes, expected_bytes
            )
        )

    measurement = np.memmap(
        path,
        dtype=np.uint16,
        mode="r",
        shape=MEASUREMENT_SHAPE,
    )
    stored_histogram = np.zeros(4096, dtype=np.int64)
    pixel_sum = np.zeros(MEASUREMENT_SHAPE[1:], dtype=np.float64)
    pixel_sum_of_squares = np.zeros_like(pixel_sum)
    batch_rows = []

    for batch_index, start in enumerate(range(0, MEASUREMENT_SHAPE[0], 1000), start=1):
        stop = min(start + 1000, MEASUREMENT_SHAPE[0])
        batch = np.asarray(measurement[start:stop])
        stored_histogram += np.bincount(
            batch.ravel(), minlength=4096
        )[:4096]
        pixel_sum += batch.sum(axis=0, dtype=np.float64)
        pixel_sum_of_squares += np.square(
            batch, dtype=np.float64
        ).sum(axis=0)
        batch_rows.append(
            {
                "batch": batch_index,
                "start_frame": start,
                "stop_frame": stop,
                "frame_count": stop - start,
                "mean_intensity": float(batch.mean()),
                "native_8bit_mean": float(batch.mean() * 255.0 / 4095.0),
                "zero_fraction": float(np.mean(batch == 0)),
            }
        )

    sample_count = int(stored_histogram.sum())
    mean_map = pixel_sum / MEASUREMENT_SHAPE[0]
    variance_map = np.maximum(
        pixel_sum_of_squares / MEASUREMENT_SHAPE[0] - mean_map ** 2,
        0.0,
    )
    temporal_std_map = np.sqrt(variance_map)
    native_histogram = _native_histogram(stored_histogram)
    stored_quantiles = {
        str(probability): _quantile_from_histogram(
            stored_histogram, probability
        )
        for probability in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
    }
    native_quantiles = {
        probability: int(
            np.searchsorted(
                np.cumsum(native_histogram),
                probability * sample_count,
                side="left",
            )
        )
        for probability in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
    }

    edge_mask = np.ones(MEASUREMENT_SHAPE[1:], dtype=bool)
    edge_mask[16:112, 16:112] = False
    center = mean_map[32:96, 32:96]
    profile = {
        "sample_count": sample_count,
        "stored_mean": float(
            np.dot(
                np.arange(stored_histogram.size, dtype=np.float64),
                stored_histogram,
            )
            / sample_count
        ),
        "native_8bit_equivalent_mean": float(
            np.dot(
                np.arange(native_histogram.size, dtype=np.float64),
                native_histogram,
            )
            / sample_count
        ),
        "stored_quantiles": stored_quantiles,
        "native_8bit_quantiles": {
            str(key): value for key, value in native_quantiles.items()
        },
        "zero_fraction": float(native_histogram[0] / sample_count),
        "at_or_below_1_native_code_fraction": float(
            native_histogram[:2].sum() / sample_count
        ),
        "at_or_below_2_native_codes_fraction": float(
            native_histogram[:3].sum() / sample_count
        ),
        "at_or_below_4_native_codes_fraction": float(
            native_histogram[:5].sum() / sample_count
        ),
        "at_or_above_250_native_codes_fraction": float(
            native_histogram[250:].sum() / sample_count
        ),
        "spatial_mean_min": float(mean_map.min()),
        "spatial_mean_median": float(np.median(mean_map)),
        "spatial_mean_max": float(mean_map.max()),
        "spatial_mean_cv": float(mean_map.std() / mean_map.mean()),
        "center_64_mean": float(center.mean()),
        "outer_16_mean": float(mean_map[edge_mask].mean()),
        "outer_to_center_ratio": float(
            mean_map[edge_mask].mean() / center.mean()
        ),
        "temporal_pixel_std_median": float(
            np.median(temporal_std_map)
        ),
    }
    return profile, _native_histogram_bins(native_histogram), batch_rows, mean_map


def _load_focus_records(path):
    records = []
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            records.append(
                {
                    "x": int(row["x"]),
                    "y": int(row["y"]),
                    "pbr": float(row["pbr"]),
                    "target_intensity": float(row["target_intensity"]),
                    "peak_distance_px": float(row["peak_distance_px"]),
                }
            )
    return records


def _focus_diagnostics(records, mean_map):
    measurement_mean = np.asarray(
        [mean_map[row["y"], row["x"]] for row in records],
        dtype=np.float64,
    )
    pbr = np.asarray([row["pbr"] for row in records], dtype=np.float64)
    target = np.asarray(
        [row["target_intensity"] for row in records], dtype=np.float64
    )
    distance = np.asarray(
        [row["peak_distance_px"] for row in records], dtype=np.float64
    )
    finite = (
        np.isfinite(measurement_mean)
        & np.isfinite(pbr)
        & np.isfinite(target)
        & np.isfinite(distance)
    )
    measurement_mean = measurement_mean[finite]
    pbr = pbr[finite]
    target = target[finite]
    distance = distance[finite]

    boundaries = np.quantile(measurement_mean, [0, 0.2, 0.4, 0.6, 0.8, 1])
    quintile_rows = []
    for index in range(5):
        mask = measurement_mean >= boundaries[index]
        if index == 4:
            mask &= measurement_mean <= boundaries[index + 1]
        else:
            mask &= measurement_mean < boundaries[index + 1]
        quintile_rows.append(
            {
                "intensity_quintile": "Q{}".format(index + 1),
                "measurement_mean_min": float(measurement_mean[mask].min()),
                "measurement_mean_max": float(measurement_mean[mask].max()),
                "point_count": int(mask.sum()),
                "pbr_mean": float(pbr[mask].mean()),
                "pbr_median": float(np.median(pbr[mask])),
                "target_intensity_mean": float(target[mask].mean()),
                "within_2px_fraction": float(np.mean(distance[mask] <= 2)),
                "within_5px_fraction": float(np.mean(distance[mask] <= 5)),
                "over_20px_fraction": float(np.mean(distance[mask] > 20)),
            }
        )

    return {
        "point_count": int(len(pbr)),
        "target_zero_fraction": float(np.mean(target == 0)),
        "pbr_zero_fraction": float(np.mean(pbr == 0)),
        "within_1px_fraction": float(np.mean(distance <= 1)),
        "within_2px_fraction": float(np.mean(distance <= 2)),
        "within_5px_fraction": float(np.mean(distance <= 5)),
        "over_20px_fraction": float(np.mean(distance > 20)),
        "over_50px_fraction": float(np.mean(distance > 50)),
        "over_100px_fraction": float(np.mean(distance > 100)),
        "measurement_mean_vs_pbr_pearson": float(
            pearsonr(measurement_mean, pbr)[0]
        ),
        "measurement_mean_vs_pbr_spearman": float(
            spearmanr(measurement_mean, pbr).statistic
        ),
        "measurement_mean_vs_target_pearson": float(
            pearsonr(measurement_mean, target)[0]
        ),
        "measurement_mean_vs_peak_distance_spearman": float(
            spearmanr(measurement_mean, distance).statistic
        ),
        "intensity_quintiles": quintile_rows,
    }


def _comparison_row(label, quality, focus, recovery=None, mapping="active512"):
    row = {
        "run": label,
        "mapping": mapping,
        "measurement_mean": float(quality["global_mean"]),
        "native_8bit_equivalent_mean": float(
            quality["global_mean"] * 255.0 / 4095.0
        ),
        "zero_fraction": float(quality["zero_fraction"]),
        "pbr_mean": float(focus["target_pbr"]["mean"]),
        "pbr_median": float(focus["target_pbr"]["median"]),
        "peak_distance_median_px": float(
            focus["peak_distance_px"]["median"]
        ),
        "final_relative_amplitude_error": None,
    }
    if recovery is not None:
        row["final_relative_amplitude_error"] = float(
            recovery["final_relative_amplitude_error"]
        )
    return row


def build_diagnostic(arguments):
    profile, histogram_bins, batch_rows, mean_map = _scan_measurement(
        arguments.measurement
    )
    current_quality = _load_json(arguments.current_quality)
    previous_active_quality = _load_json(arguments.previous_active_quality)
    legacy_quality = _load_json(arguments.legacy_quality)
    current_focus = _load_json(arguments.current_focus_summary)
    previous_active_focus = _load_json(arguments.previous_active_focus_summary)
    legacy_focus = _load_json(arguments.legacy_focus_summary)
    current_recovery = _load_json(arguments.current_recovery)
    legacy_recovery = _load_json(arguments.legacy_recovery)
    focus_records = _load_focus_records(arguments.current_focus_points)

    result = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "measurement_path": os.path.abspath(arguments.measurement),
        "measurement_shape": list(MEASUREMENT_SHAPE),
        "camera_encoding": (
            "Polarized8 native samples scaled to uint16 0-4095; scaling does "
            "not add sensor precision"
        ),
        "current_profile": profile,
        "native_histogram_bins": histogram_bins,
        "batch_profile": batch_rows,
        "focus_diagnostics": _focus_diagnostics(focus_records, mean_map),
        "run_comparison": [
            _comparison_row(
                "2026-08-04 legacy mapping",
                legacy_quality,
                legacy_focus,
                legacy_recovery,
                mapping="legacy full-field",
            ),
            _comparison_row(
                "2026-08-05 active512",
                previous_active_quality,
                previous_active_focus,
                mapping="active512",
            ),
            _comparison_row(
                "2026-08-06 active512",
                current_quality,
                current_focus,
                current_recovery,
                mapping="active512",
            ),
        ],
        "recovery": {
            "current_initial_relative_amplitude_error": float(
                current_recovery["initial_relative_amplitude_error"]
            ),
            "current_final_relative_amplitude_error": float(
                current_recovery["final_relative_amplitude_error"]
            ),
            "legacy_final_relative_amplitude_error": float(
                legacy_recovery["final_relative_amplitude_error"]
            ),
        },
        "source_files": [
            os.path.abspath(value)
            for value in (
                arguments.measurement,
                arguments.current_quality,
                arguments.previous_active_quality,
                arguments.legacy_quality,
                arguments.current_focus_summary,
                arguments.current_focus_points,
                arguments.previous_active_focus_summary,
                arguments.legacy_focus_summary,
                arguments.current_recovery,
                arguments.legacy_recovery,
            )
        ],
    }
    with open(arguments.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--measurement",
        default="measurements_128_px4_active512_full_memmap.npy",
    )
    parser.add_argument(
        "--current-quality",
        default="measurements_128_px4_active512_full_quality_summary_20260806.json",
    )
    parser.add_argument(
        "--previous-active-quality",
        default="measurements_128_px4_active512_full_quality_summary_20260805.json",
    )
    parser.add_argument(
        "--legacy-quality",
        default="measurements_128_px4_full_quality_summary.json",
    )
    parser.add_argument(
        "--current-focus-summary",
        default=(
            "pixelwise_focus_results_128_px4_active512/"
            "pixelwise_focus_128_px4_active512_20260806_161522_"
            "stride1_batch1000_n16384_summary.json"
        ),
    )
    parser.add_argument(
        "--current-focus-points",
        default=(
            "pixelwise_focus_results_128_px4_active512/"
            "pixelwise_focus_128_px4_active512_20260806_161522_"
            "stride1_batch1000_n16384_points.csv"
        ),
    )
    parser.add_argument(
        "--previous-active-focus-summary",
        default=(
            "pixelwise_focus_results_128_px4_active512/"
            "pixelwise_focus_128_px4_active512_20260805_211916_"
            "stride1_n16384_summary.json"
        ),
    )
    parser.add_argument(
        "--legacy-focus-summary",
        default=(
            "pixelwise_focus_results_128_px4/"
            "pixelwise_focus_128_px4_20260804_124454_stride1_n16384_"
            "summary.json"
        ),
    )
    parser.add_argument(
        "--current-recovery",
        default="tm_reconstruction_128_px4_active512.json",
    )
    parser.add_argument(
        "--legacy-recovery",
        default="tm_reconstruction_128_px4.json",
    )
    parser.add_argument(
        "--output",
        default="measurement_intensity_diagnostic_20260806.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    diagnostic = build_diagnostic(parse_args())
    print(json.dumps(diagnostic, indent=2, ensure_ascii=False))
