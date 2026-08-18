"""Plot the frame-averaged speckle intensity from a headerless uint16 memmap."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_WIDTH = 128
DEFAULT_HEIGHT = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "measurement",
        nargs="?",
        default="measurements_memmap.npy",
        help="Headerless uint16 measurement memmap.",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument(
        "--output-prefix",
        help="Output path without extension; defaults to a source- and date-based name.",
    )
    return parser.parse_args()


def infer_frame_count(path: Path, height: int, width: int) -> int:
    bytes_per_frame = height * width * np.dtype(np.uint16).itemsize
    size = path.stat().st_size
    if size == 0 or size % bytes_per_frame:
        raise ValueError(
            f"{path} has {size} bytes, which is not an integer number of "
            f"{height}x{width} uint16 frames ({bytes_per_frame} bytes/frame)."
        )
    return size // bytes_per_frame


def main() -> None:
    args = parse_args()
    source = Path(args.measurement).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    frame_count = infer_frame_count(source, args.height, args.width)
    measurements = np.memmap(
        source,
        dtype=np.uint16,
        mode="r",
        shape=(frame_count, args.height, args.width),
    )
    mean_map = np.asarray(measurements.mean(axis=0, dtype=np.float64))

    if args.output_prefix:
        prefix = Path(args.output_prefix).resolve()
    else:
        date_tag = datetime.fromtimestamp(source.stat().st_mtime).strftime("%Y%m%d")
        prefix = source.with_name(
            f"average_speckle_intensity_{source.stem}_{date_tag}"
        )
    prefix.parent.mkdir(parents=True, exist_ok=True)

    png_path = prefix.with_suffix(".png")
    npy_path = prefix.with_suffix(".npy")
    json_path = prefix.with_suffix(".json")
    np.save(npy_path, mean_map.astype(np.float32))

    vmin = float(mean_map.min())
    vmax = float(mean_map.max())
    global_mean = float(mean_map.mean())

    fig, ax = plt.subplots(figsize=(7.4, 6.4), constrained_layout=True)
    image = ax.imshow(
        mean_map,
        cmap="cividis",
        origin="upper",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(
        "Average speckle intensity\n"
        f"Mean of {frame_count:,} frames | global mean = {global_mean:.2f} counts",
        loc="left",
        fontsize=14,
        color="#23272f",
        pad=12,
    )
    ax.set_xlabel("Camera x (pixel)")
    ax.set_ylabel("Camera y (pixel)")
    ax.set_xticks(np.arange(0, args.width, 16))
    ax.set_yticks(np.arange(0, args.height, 16))
    ax.tick_params(colors="#4b5563")
    for spine in ax.spines.values():
        spine.set_color("#4b5563")
        spine.set_linewidth(0.8)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.047, pad=0.035)
    colorbar.set_label("Mean intensity (stored digital counts)")
    colorbar.outline.set_linewidth(0.8)
    colorbar.outline.set_edgecolor("#4b5563")

    fig.savefig(png_path, dpi=220, facecolor="white")
    plt.close(fig)

    metadata = {
        "source_file": str(source),
        "source_file_bytes": source.stat().st_size,
        "source_modified": datetime.fromtimestamp(source.stat().st_mtime).astimezone().isoformat(),
        "storage_format": "headerless uint16 memmap",
        "storage_shape": [frame_count, args.height, args.width],
        "average_axis": 0,
        "average_frame_count": frame_count,
        "mean_intensity_counts": global_mean,
        "mean_map_min_counts": vmin,
        "mean_map_max_counts": vmax,
        "mean_map_std_counts": float(mean_map.std()),
        "mean_map_p01_counts": float(np.percentile(mean_map, 1)),
        "mean_map_p99_counts": float(np.percentile(mean_map, 99)),
        "outputs": {
            "figure": str(png_path),
            "mean_array": str(npy_path),
            "metadata": str(json_path),
        },
    }
    json_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
