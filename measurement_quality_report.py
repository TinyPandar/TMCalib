"""Analyze and plot a headerless uint16 measurement memmap."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def _histogram_quantile(histogram, quantile):
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    cumulative = np.cumsum(histogram, dtype=np.uint64)
    if cumulative.size == 0 or cumulative[-1] == 0:
        return 0
    target = quantile * float(cumulative[-1])
    return int(np.searchsorted(cumulative, target, side="left"))


def analyze_measurement_memmap(
    measurement_path,
    height=128,
    width=128,
    batch_frames=256,
    sensor_max_count=255,
):
    """Return an exact mean image, integer histogram, and quality statistics."""
    source = Path(measurement_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    dtype = np.dtype(np.uint16)
    bytes_per_frame = int(height) * int(width) * dtype.itemsize
    file_size = source.stat().st_size
    if file_size == 0 or file_size % bytes_per_frame:
        raise ValueError(
            "{} has {} bytes, which is not an integer number of {}x{} "
            "uint16 frames".format(source, file_size, height, width)
        )

    frame_count = file_size // bytes_per_frame
    measurements = np.memmap(
        str(source),
        dtype=dtype,
        mode="r",
        shape=(frame_count, int(height), int(width)),
    )

    spatial_sum = np.zeros((int(height), int(width)), dtype=np.uint64)
    histogram = np.zeros(np.iinfo(dtype).max + 1, dtype=np.uint64)
    frame_sums = np.empty(frame_count, dtype=np.uint64)
    block_means = []

    for start in range(0, frame_count, int(batch_frames)):
        block = np.asarray(measurements[start : start + int(batch_frames)])
        stop = start + len(block)
        spatial_sum += block.sum(axis=0, dtype=np.uint64)
        frame_sums[start:stop] = block.sum(axis=(1, 2), dtype=np.uint64)
        histogram += np.bincount(
            block.ravel(), minlength=histogram.size
        ).astype(np.uint64)
        block_means.append(float(block.mean()))

    mean_map = (spatial_sum.astype(np.float64) / frame_count).astype(np.float32)
    used_values = np.flatnonzero(histogram)
    raw_min = int(used_values[0])
    raw_max = int(used_values[-1])
    histogram = histogram[: raw_max + 1]
    total_samples = int(frame_count * height * width)
    nonblank_frame_count = int(np.count_nonzero(frame_sums))
    zero_fraction = float(histogram[0] / total_samples)
    saturation_fraction = float(
        histogram[int(sensor_max_count) :].sum() / total_samples
    ) if raw_max >= int(sensor_max_count) else 0.0

    quality_flags = []
    blank_frame_count = int(frame_count - nonblank_frame_count)
    if blank_frame_count:
        quality_flags.append(
            "{} all-zero frame(s) detected".format(blank_frame_count)
        )
    if zero_fraction >= 0.5:
        quality_flags.append(
            "{:.1%} of stored samples are zero".format(zero_fraction)
        )
    if saturation_fraction >= 0.001:
        quality_flags.append(
            "{:.2%} of samples reach the sensor ceiling".format(
                saturation_fraction
            )
        )

    block_means_array = np.asarray(block_means, dtype=np.float64)
    result = {
        "source_file": str(source),
        "source_file_bytes": int(file_size),
        "source_modified": datetime.fromtimestamp(
            source.stat().st_mtime
        ).astimezone().isoformat(),
        "storage_format": "headerless uint16 memmap",
        "storage_shape": [int(frame_count), int(height), int(width)],
        "sensor_max_count": int(sensor_max_count),
        "mean_map": mean_map,
        "histogram": histogram,
        "statistics": {
            "frame_count": int(frame_count),
            "nonblank_frame_count": nonblank_frame_count,
            "blank_frame_count": blank_frame_count,
            "global_mean_counts": float(mean_map.mean(dtype=np.float64)),
            "mean_map_min_counts": float(mean_map.min()),
            "mean_map_max_counts": float(mean_map.max()),
            "mean_map_std_counts": float(mean_map.std(dtype=np.float64)),
            "mean_map_p01_counts": float(np.percentile(mean_map, 1)),
            "mean_map_p99_counts": float(np.percentile(mean_map, 99)),
            "frame_mean_min_counts": float(
                frame_sums.min() / float(height * width)
            ),
            "frame_mean_max_counts": float(
                frame_sums.max() / float(height * width)
            ),
            "frame_mean_std_counts": float(
                frame_sums.astype(np.float64).std() / float(height * width)
            ),
            "block_mean_cv": float(
                block_means_array.std() / block_means_array.mean()
            ) if block_means_array.mean() else 0.0,
            "raw_min_counts": raw_min,
            "raw_max_counts": raw_max,
            "raw_p50_counts": _histogram_quantile(histogram, 0.50),
            "raw_p90_counts": _histogram_quantile(histogram, 0.90),
            "raw_p99_counts": _histogram_quantile(histogram, 0.99),
            "raw_p999_counts": _histogram_quantile(histogram, 0.999),
            "zero_fraction": zero_fraction,
            "saturation_fraction": saturation_fraction,
        },
        "quality_flags": quality_flags,
    }
    return result


def _histogram_plot_data(histogram, max_bars=256):
    """Return centers, sample counts, and widths without hiding the tail."""
    histogram = np.asarray(histogram, dtype=np.uint64)
    if histogram.size <= max_bars:
        return (
            np.arange(histogram.size, dtype=np.float64),
            histogram.astype(np.float64),
            np.ones(histogram.size, dtype=np.float64),
        )

    edges = np.linspace(0, histogram.size, max_bars + 1, dtype=np.int64)
    counts = np.empty(max_bars, dtype=np.float64)
    centers = np.empty(max_bars, dtype=np.float64)
    widths = np.empty(max_bars, dtype=np.float64)
    for index in range(max_bars):
        left = int(edges[index])
        right = int(edges[index + 1])
        counts[index] = float(histogram[left:right].sum())
        centers[index] = 0.5 * (left + right - 1)
        widths[index] = max(1, right - left)
    return centers, counts, widths


def build_measurement_quality_figure(result):
    """Create the average-intensity and histogram figure used by file and GUI."""
    mean_map = np.asarray(result["mean_map"])
    histogram = np.asarray(result["histogram"])
    stats = result["statistics"]

    figure = Figure(figsize=(12.2, 5.6), dpi=100, constrained_layout=True)
    image_axis, histogram_axis = figure.subplots(
        1, 2, gridspec_kw={"width_ratios": [1.0, 1.08]}
    )
    figure.suptitle(
        "Measurement quality | {:,} frames | mean = {:.3f} counts | "
        "zero = {:.2%} | max = {}".format(
            stats["frame_count"],
            stats["global_mean_counts"],
            stats["zero_fraction"],
            stats["raw_max_counts"],
        ),
        fontsize=14,
        color="#23272f",
    )

    vmin = float(mean_map.min())
    vmax = float(mean_map.max())
    if vmax <= vmin:
        vmax = vmin + 1.0
    image = image_axis.imshow(
        mean_map,
        cmap="cividis",
        origin="upper",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    image_axis.set_title("Average speckle intensity", loc="left")
    image_axis.set_xlabel("Camera x (pixel)")
    image_axis.set_ylabel("Camera y (pixel)")
    figure.colorbar(image, ax=image_axis, fraction=0.046, pad=0.035).set_label(
        "Mean intensity (stored digital counts)"
    )

    centers, counts, widths = _histogram_plot_data(histogram)
    shares = counts / counts.sum() * 100.0
    histogram_axis.bar(
        centers,
        shares,
        width=widths,
        color="#4f78a5",
        edgecolor="#27496d",
        linewidth=0.45,
        align="center",
    )
    histogram_axis.axvline(
        stats["global_mean_counts"],
        color="#c46a27",
        linewidth=1.8,
        label="Mean {:.3f}".format(stats["global_mean_counts"]),
    )
    histogram_axis.axvline(
        stats["raw_p99_counts"],
        color="#30343b",
        linewidth=1.3,
        linestyle="--",
        label="P99 {}".format(stats["raw_p99_counts"]),
    )
    positive_shares = shares[shares > 0]
    if positive_shares.size and positive_shares.max() / positive_shares.min() >= 1000:
        histogram_axis.set_yscale("log")
        histogram_axis.set_ylabel("Sample share (%) - log scale")
    else:
        histogram_axis.set_ylabel("Sample share (%)")
    histogram_axis.set_xlim(
        -0.5,
        max(1.0, float(stats["raw_max_counts"]) + 0.5),
    )
    histogram_axis.set_title("Stored intensity histogram", loc="left")
    histogram_axis.set_xlabel("Stored digital count")
    histogram_axis.grid(axis="y", color="#d8dde5", linewidth=0.7, alpha=0.8)
    histogram_axis.legend(frameon=False, loc="upper right")

    for axis in (image_axis, histogram_axis):
        axis.tick_params(colors="#4b5563")
        for spine in axis.spines.values():
            spine.set_color("#69717d")
            spine.set_linewidth(0.8)

    if result["quality_flags"]:
        figure.text(
            0.5,
            0.005,
            "Check: " + "; ".join(result["quality_flags"]),
            ha="center",
            va="bottom",
            color="#8a4b20",
            fontsize=10,
        )
    return figure


def save_measurement_quality_outputs(result, output_dir=None, prefix=None):
    """Save a timestamped PNG, mean-map NPY, and JSON summary."""
    source = Path(result["source_file"])
    destination = Path(output_dir).resolve() if output_dir else source.parent
    destination.mkdir(parents=True, exist_ok=True)
    if prefix is None:
        timestamp = datetime.fromisoformat(result["source_modified"]).strftime(
            "%Y%m%d_%H%M%S"
        )
        prefix = "measurement_quality_{}_{}".format(source.stem, timestamp)
    base = destination / prefix
    png_path = base.with_suffix(".png")
    npy_path = base.with_suffix(".npy")
    json_path = base.with_suffix(".json")

    np.save(str(npy_path), np.asarray(result["mean_map"], dtype=np.float32))
    figure = build_measurement_quality_figure(result)
    FigureCanvasAgg(figure).print_figure(
        str(png_path), dpi=180, facecolor="white"
    )
    figure.clear()

    metadata = {
        key: value
        for key, value in result.items()
        if key not in ("mean_map", "histogram")
    }
    metadata["histogram_counts"] = [
        int(value) for value in np.asarray(result["histogram"])
    ]
    metadata["outputs"] = {
        "figure": str(png_path),
        "mean_map": str(npy_path),
        "metadata": str(json_path),
    }
    json_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata["outputs"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurement", nargs="?", default="measurements_memmap.npy")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--sensor-max-count", type=int, default=255)
    args = parser.parse_args()

    result = analyze_measurement_memmap(
        args.measurement,
        height=args.height,
        width=args.width,
        sensor_max_count=args.sensor_max_count,
    )
    outputs = save_measurement_quality_outputs(result)
    print(
        json.dumps(
            {
                "statistics": result["statistics"],
                "quality_flags": result["quality_flags"],
                "outputs": outputs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
