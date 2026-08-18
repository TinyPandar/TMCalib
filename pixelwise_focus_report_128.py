"""Export detailed pixel-wise focusing records and static QA figures."""

import csv
import json
import math
import os
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure


RECORD_FIELDS = [
    "sample_index",
    "raw_image_index",
    "x",
    "y",
    "target_index",
    "success",
    "target_intensity",
    "peak_intensity",
    "peak_x",
    "peak_y",
    "peak_distance_px",
    "image_min_intensity",
    "image_max_intensity",
    "image_intensity_range",
    "mean_intensity",
    "background_intensity",
    "pbr",
    "error",
]


def build_pixelwise_points(
    roi_width: int,
    roi_height: int,
    stride: int = 1,
    max_points=None,
) -> List[Tuple[int, int]]:
    """Build row-major camera coordinates for full or sampled pixel-wise tests."""
    roi_width = int(roi_width)
    roi_height = int(roi_height)
    stride = int(stride)
    if roi_width <= 0 or roi_height <= 0:
        raise ValueError("ROI dimensions must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    points = [
        (x, y)
        for y in range(0, roi_height, stride)
        for x in range(0, roi_width, stride)
    ]
    if max_points is not None:
        max_points = int(max_points)
        if max_points < 0:
            raise ValueError("max_points must be non-negative")
        points = points[:max_points]
    return points


def _finite(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def _summary(values: np.ndarray) -> Dict:
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _histogram_bins(values: np.ndarray) -> int:
    if values.size < 2 or float(np.ptp(values)) == 0:
        return 1
    edges = np.histogram_bin_edges(values, bins="fd")
    return max(8, min(50, len(edges) - 1))


def _save_distribution(
    target_values: np.ndarray,
    pbr_values: np.ndarray,
    successful: int,
    attempted: int,
    path: str,
) -> None:
    figure = Figure(figsize=(11.0, 4.8), dpi=150, facecolor="#FAFBFC")
    canvas = FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    specifications = [
        (
            axes[0],
            target_values,
            "#2F6B9A",
            "Target-intensity distribution",
            "Target pixel intensity (12-bit scaled counts)",
        ),
        (
            axes[1],
            pbr_values,
            "#D29F38",
            "Target-PBR distribution",
            "Target intensity / background mean",
        ),
    ]
    for axis, values, color, title, xlabel in specifications:
        axis.set_facecolor("#FAFBFC")
        if values.size:
            axis.hist(
                values,
                bins=_histogram_bins(values),
                color=color,
                edgecolor="#17324D",
                linewidth=0.6,
                alpha=0.90,
            )
            median = float(np.median(values))
            mean = float(np.mean(values))
            axis.axvline(
                median,
                color="#17324D",
                linewidth=1.5,
                linestyle="-",
                label="Median {:.2f}".format(median),
            )
            axis.axvline(
                mean,
                color="#6B7280",
                linewidth=1.3,
                linestyle="--",
                label="Mean {:.2f}".format(mean),
            )
            axis.legend(frameon=False, fontsize=8)
        else:
            axis.text(
                0.5,
                0.5,
                "No successful measurements",
                ha="center",
                va="center",
                transform=axis.transAxes,
                color="#6B7280",
            )
        axis.set_title(title, loc="left", fontsize=11, color="#17212B", pad=12)
        axis.set_xlabel(xlabel, fontsize=9, color="#374151")
        axis.set_ylabel("Tested points", fontsize=9, color="#374151")
        axis.grid(axis="y", color="#DDE2E7", linewidth=0.7, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#9CA3AF")
        axis.spines["bottom"].set_color("#9CA3AF")

    figure.suptitle(
        "Pixel-wise focusing distributions",
        x=0.06,
        y=0.99,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color="#17212B",
    )
    figure.text(
        0.06,
        0.925,
        "{} successful of {} attempted points".format(successful, attempted),
        ha="left",
        fontsize=9,
        color="#5F6B76",
    )
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.16, top=0.80, wspace=0.24)
    canvas.print_png(path)


def _save_heatmap(
    records: Sequence[Dict],
    roi_shape: Tuple[int, int],
    successful: int,
    attempted: int,
    path: str,
) -> None:
    xs = sorted({int(record["x"]) for record in records})
    ys = sorted({int(record["y"]) for record in records})
    x_lookup = {value: index for index, value in enumerate(xs)}
    y_lookup = {value: index for index, value in enumerate(ys)}
    heatmap = np.full((len(ys), len(xs)), np.nan, dtype=np.float32)
    for record in records:
        if record.get("success") and np.isfinite(float(record.get("pbr", np.nan))):
            heatmap[y_lookup[int(record["y"])], x_lookup[int(record["x"])]] = float(
                record["pbr"]
            )

    finite_pbr = heatmap[np.isfinite(heatmap)]
    color_max = (
        float(np.percentile(finite_pbr, 95))
        if finite_pbr.size
        else 1.0
    )
    if not np.isfinite(color_max) or color_max <= 0:
        color_max = 1.0
    actual_max = float(np.max(finite_pbr)) if finite_pbr.size else None

    figure = Figure(figsize=(8.2, 7.0), dpi=150, facecolor="#FAFBFC")
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    axis.set_facecolor("#D9DEE3")
    color_map = LinearSegmentedColormap.from_list(
        "tmcalib_blue",
        ["#F5F9FC", "#BCD4E6", "#5B93BC", "#173F5F"],
    )
    color_map.set_bad("#D9DEE3")
    single_axis_sampling = heatmap.shape[0] == 1 or heatmap.shape[1] == 1
    if single_axis_sampling:
        successful_records = [
            record
            for record in records
            if record.get("success")
            and np.isfinite(float(record.get("pbr", np.nan)))
        ]
        failed_records = [
            record for record in records if not record.get("success")
        ]
        image = axis.scatter(
            [int(record["x"]) for record in successful_records],
            [int(record["y"]) for record in successful_records],
            c=[float(record["pbr"]) for record in successful_records],
            cmap=color_map,
            vmin=0.0,
            vmax=color_max,
            marker="s",
            s=55,
            linewidths=0,
        )
        if failed_records:
            axis.scatter(
                [int(record["x"]) for record in failed_records],
                [int(record["y"]) for record in failed_records],
                color="#D9DEE3",
                edgecolors="#9CA3AF",
                linewidths=0.5,
                marker="s",
                s=55,
            )
        roi_height, roi_width = map(int, roi_shape)
        axis.set_xlim(-0.5, roi_width - 0.5)
        axis.set_ylim(-0.5, roi_height - 0.5)
        axis.set_aspect("equal")
        axis.set_xticks(np.arange(0, roi_width + 1, 16))
        axis.set_yticks(np.arange(0, roi_height + 1, 16))
        axis.grid(color="#E3E7EB", linewidth=0.6, alpha=0.8)
    else:
        image = axis.imshow(
            heatmap,
            origin="lower",
            interpolation="nearest",
            aspect="equal",
            cmap=color_map,
            vmin=0.0,
            vmax=color_max,
        )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Target PBR", color="#374151")
    colorbar.outline.set_edgecolor("#9CA3AF")

    if xs and not single_axis_sampling:
        x_tick_count = min(9, len(xs))
        x_tick_positions = np.unique(
            np.linspace(0, len(xs) - 1, x_tick_count).round().astype(int)
        )
        axis.set_xticks(x_tick_positions)
        axis.set_xticklabels([xs[index] for index in x_tick_positions])
    if ys and not single_axis_sampling:
        y_tick_count = min(9, len(ys))
        y_tick_positions = np.unique(
            np.linspace(0, len(ys) - 1, y_tick_count).round().astype(int)
        )
        axis.set_yticks(y_tick_positions)
        axis.set_yticklabels([ys[index] for index in y_tick_positions])
    axis.set_xlabel("Camera ROI x")
    axis.set_ylabel("Camera ROI y")
    figure.suptitle(
        "Pixel-wise target-PBR heatmap",
        x=0.11,
        y=0.97,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color="#17212B",
    )
    max_text = "-" if actual_max is None else "{:.2f}".format(actual_max)
    figure.text(
        0.11,
        0.925,
        "{} successful of {} attempted · color scale 0 to P95 ({:.2f}) · "
        "observed max {}".format(successful, attempted, color_max, max_text),
        ha="left",
        va="center",
        fontsize=8.5,
        color="#5F6B76",
    )
    for spine in axis.spines.values():
        spine.set_color("#9CA3AF")
    figure.subplots_adjust(left=0.11, right=0.88, bottom=0.10, top=0.86)
    canvas.print_png(path)


def save_pixelwise_focus_report(
    records: List[Dict],
    roi_shape: Tuple[int, int],
    output_dir: str,
    run_label: str = "",
    file_prefix: str = "pixelwise_focus_128_px4_active512",
    raw_images_path: str = None,
) -> Dict:
    """Save point records, maps, summary, distribution plots, and PBR heatmap."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prefix = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(file_prefix)
    ).strip("_")
    if not safe_prefix:
        safe_prefix = "pixelwise_focus"
    stem = "{}_{}".format(safe_prefix, timestamp)
    if run_label:
        safe_label = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in run_label
        ).strip("_")
        if safe_label:
            stem += "_" + safe_label

    csv_path = os.path.join(output_dir, stem + "_points.csv")
    maps_path = os.path.join(output_dir, stem + "_maps.npz")
    summary_path = os.path.join(output_dir, stem + "_summary.json")
    distribution_path = os.path.join(output_dir, stem + "_distribution.png")
    heatmap_path = os.path.join(output_dir, stem + "_pbr_heatmap.png")

    normalized_records = []
    for record in records:
        normalized = {field: record.get(field) for field in RECORD_FIELDS}
        # ``peak_intensity`` is the image maximum in the current focus
        # analyzers. Keep an explicit image-maximum column so the per-frame
        # intensity range remains self-explanatory in exported reports, while
        # still accepting records produced by older callers.
        if normalized["image_max_intensity"] is None:
            normalized["image_max_intensity"] = normalized["peak_intensity"]
        if normalized["image_intensity_range"] is None:
            try:
                image_min = float(normalized["image_min_intensity"])
                image_max = float(normalized["image_max_intensity"])
                if np.isfinite(image_min) and np.isfinite(image_max):
                    normalized["image_intensity_range"] = image_max - image_min
            except (TypeError, ValueError):
                pass
        normalized_records.append(normalized)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECORD_FIELDS)
        writer.writeheader()
        writer.writerows(normalized_records)

    roi_height, roi_width = map(int, roi_shape)
    pbr_map = np.full((roi_height, roi_width), np.nan, dtype=np.float32)
    target_intensity_map = np.full_like(pbr_map, np.nan)
    peak_distance_map = np.full_like(pbr_map, np.nan)
    image_min_intensity_map = np.full_like(pbr_map, np.nan)
    image_max_intensity_map = np.full_like(pbr_map, np.nan)
    image_intensity_range_map = np.full_like(pbr_map, np.nan)
    for record in normalized_records:
        if not record.get("success"):
            continue
        x = int(record["x"])
        y = int(record["y"])
        if 0 <= x < roi_width and 0 <= y < roi_height:
            pbr_map[y, x] = float(record["pbr"])
            target_intensity_map[y, x] = float(record["target_intensity"])
            peak_distance_map[y, x] = float(record["peak_distance_px"])
            for output_map, field in (
                (image_min_intensity_map, "image_min_intensity"),
                (image_max_intensity_map, "image_max_intensity"),
                (image_intensity_range_map, "image_intensity_range"),
            ):
                value = record.get(field)
                if value is not None and np.isfinite(float(value)):
                    output_map[y, x] = float(value)
    np.savez_compressed(
        maps_path,
        pbr=pbr_map,
        target_intensity=target_intensity_map,
        peak_distance_px=peak_distance_map,
        image_min_intensity=image_min_intensity_map,
        image_max_intensity=image_max_intensity_map,
        image_intensity_range=image_intensity_range_map,
    )

    successful_records = [
        record for record in normalized_records if bool(record.get("success"))
    ]
    pbr_values = _finite([record["pbr"] for record in successful_records])
    target_values = _finite(
        [record["target_intensity"] for record in successful_records]
    )
    peak_distances = _finite(
        [record["peak_distance_px"] for record in successful_records]
    )
    image_min_values = _finite(
        [record["image_min_intensity"] for record in successful_records]
    )
    image_max_values = _finite(
        [record["image_max_intensity"] for record in successful_records]
    )
    image_range_values = _finite(
        [record["image_intensity_range"] for record in successful_records]
    )
    attempted = len(normalized_records)
    successful = len(successful_records)
    files = {
        "points_csv": os.path.abspath(csv_path),
        "maps_npz": os.path.abspath(maps_path),
        "distribution_png": os.path.abspath(distribution_path),
        "pbr_heatmap_png": os.path.abspath(heatmap_path),
    }
    raw_images_metadata = None
    if raw_images_path is not None:
        raw_images_path = os.path.abspath(raw_images_path)
        if not os.path.isfile(raw_images_path):
            raise FileNotFoundError(
                "Raw focus image stack not found: {}".format(raw_images_path)
            )
        raw_images = np.load(raw_images_path, mmap_mode="r")
        try:
            raw_images_metadata = {
                "shape": [int(value) for value in raw_images.shape],
                "dtype": str(raw_images.dtype),
                "file_size_bytes": int(os.path.getsize(raw_images_path)),
            }
        finally:
            del raw_images
        files["raw_images_npy"] = raw_images_path

    summary = {
        "status": "complete",
        "created_at": datetime.now().astimezone().isoformat(),
        "roi_shape": [roi_height, roi_width],
        "attempted_points": attempted,
        "successful_points": successful,
        "failed_points": attempted - successful,
        "success_fraction": successful / attempted if attempted else 0.0,
        "target_intensity": _summary(target_values),
        "target_pbr": _summary(pbr_values),
        "peak_distance_px": _summary(peak_distances),
        "image_min_intensity": _summary(image_min_values),
        "image_max_intensity": _summary(image_max_values),
        "image_intensity_range": _summary(image_range_values),
        "raw_images": raw_images_metadata,
        "files": files,
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    summary["files"]["summary_json"] = os.path.abspath(summary_path)

    _save_distribution(
        target_values,
        pbr_values,
        successful,
        attempted,
        distribution_path,
    )
    _save_heatmap(
        normalized_records,
        (roi_height, roi_width),
        successful,
        attempted,
        heatmap_path,
    )
    return summary
