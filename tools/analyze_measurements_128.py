"""Create a reproducible visual QA overview of the 128 x 128 optical test."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


FRAME_COUNT = 64
ROI_HEIGHT = 128
ROI_WIDTH = 128


def load_measurements(path):
    expected_bytes = FRAME_COUNT * ROI_HEIGHT * ROI_WIDTH * np.dtype(np.uint16).itemsize
    actual_bytes = os.path.getsize(path)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"File has {actual_bytes} bytes; expected {expected_bytes} bytes for "
            f"({FRAME_COUNT}, {ROI_HEIGHT}, {ROI_WIDTH}) uint16"
        )
    return np.memmap(
        path,
        dtype=np.uint16,
        mode="r",
        shape=(FRAME_COUNT, ROI_HEIGHT, ROI_WIDTH),
    )


def adjacent_correlations(data):
    values = []
    for index in range(len(data) - 1):
        first = data[index].astype(np.float32).ravel()
        second = data[index + 1].astype(np.float32).ravel()
        values.append(float(np.corrcoef(first, second)[0, 1]))
    return np.asarray(values)


def create_overview(data, output_path):
    flat = data.reshape(FRAME_COUNT, -1)
    frame_mean = flat.mean(axis=1)
    frame_p99 = np.percentile(flat, 99, axis=1)
    frame_peak = flat.max(axis=1)
    mean_image = data.mean(axis=0)
    std_image = data.std(axis=0)
    max_image = data.max(axis=0)
    adjacent_corr = adjacent_correlations(data)

    display_max = float(np.percentile(data, 99.7))
    histogram_max = float(np.percentile(data, 99.9))
    selected = [0, 16, 32, 63]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": "#59636e",
            "axes.labelcolor": "#263238",
            "xtick.color": "#59636e",
            "ytick.color": "#59636e",
        }
    )

    fig = plt.figure(figsize=(14, 9), facecolor="#fbfcfd")
    grid = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.9], hspace=0.42, wspace=0.28)
    fig.suptitle(
        "128×128 Input / px=4 Optical Test — Measurement Overview",
        x=0.06,
        y=0.975,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color="#17212b",
    )
    fig.text(
        0.06,
        0.943,
        "64 random phase probes · Camera ROI 128×128 · Raw uint16 intensity",
        ha="left",
        color="#59636e",
        fontsize=10,
    )

    image_axes = []
    image_artist = None
    for column, frame_index in enumerate(selected):
        axis = fig.add_subplot(grid[0, column])
        image_artist = axis.imshow(
            data[frame_index],
            cmap="cividis",
            vmin=0,
            vmax=display_max,
            interpolation="nearest",
        )
        axis.set_title(
            f"Frame {frame_index}  |  mean {frame_mean[frame_index]:.1f}"
        )
        axis.set_xticks([])
        axis.set_yticks([])
        image_axes.append(axis)

    colorbar = fig.colorbar(
        image_artist,
        ax=image_axes,
        orientation="horizontal",
        fraction=0.035,
        pad=0.08,
        aspect=60,
    )
    colorbar.set_label(f"Intensity (shared scale, clipped at p99.7 = {display_max:.0f})")

    summary_images = [
        (mean_image, "Per-pixel mean", "cividis"),
        (std_image, "Per-pixel standard deviation", "magma"),
        (max_image, "Per-pixel maximum", "cividis"),
    ]
    for column, (image, title, cmap) in enumerate(summary_images):
        axis = fig.add_subplot(grid[1, column])
        artist = axis.imshow(image, cmap=cmap, interpolation="nearest")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(artist, ax=axis, fraction=0.046, pad=0.03)

    histogram_axis = fig.add_subplot(grid[1, 3])
    histogram_axis.hist(
        np.asarray(data).ravel(),
        bins=np.linspace(0, histogram_max, 70),
        color="#246a9b",
        edgecolor="none",
    )
    histogram_axis.set_title("Pixel-intensity distribution")
    histogram_axis.set_xlabel("Intensity (up to p99.9)")
    histogram_axis.grid(axis="y", color="#dce2e7", linewidth=0.7)

    mean_axis = fig.add_subplot(grid[2, 0:2])
    frame_index = np.arange(FRAME_COUNT)
    mean_axis.plot(frame_index, frame_mean, color="#246a9b", linewidth=1.7)
    mean_axis.axhline(
        frame_mean.mean(),
        color="#8a6d1d",
        linestyle="--",
        linewidth=1.2,
        label=f"Overall mean {frame_mean.mean():.1f}",
    )
    mean_axis.set_title(
        f"Mean intensity by frame  |  CV {frame_mean.std()/frame_mean.mean():.2%}"
    )
    mean_axis.set_xlabel("Frame index")
    mean_axis.set_ylabel("Mean intensity")
    mean_axis.set_xlim(0, FRAME_COUNT - 1)
    mean_axis.grid(color="#dce2e7", linewidth=0.7)
    mean_axis.legend(frameon=False, loc="upper right")

    range_axis = fig.add_subplot(grid[2, 2:4])
    range_axis.plot(
        frame_index,
        frame_p99,
        color="#246a9b",
        linewidth=1.7,
        label="99th percentile",
    )
    range_axis.plot(
        frame_index,
        frame_peak,
        color="#c58b22",
        linewidth=1.2,
        alpha=0.9,
        label="Peak",
    )
    range_axis.set_title(
        "Upper intensity by frame  |  "
        f"median adjacent correlation {np.median(adjacent_corr):.4f}"
    )
    range_axis.set_xlabel("Frame index")
    range_axis.set_ylabel("Intensity")
    range_axis.set_xlim(0, FRAME_COUNT - 1)
    range_axis.set_ylim(bottom=0)
    range_axis.grid(color="#dce2e7", linewidth=0.7)
    range_axis.legend(frameon=False, loc="upper right")

    saturation_fraction = float((data >= 4095).mean())
    zero_fraction = float((data == 0).mean())
    fig.text(
        0.06,
        0.018,
        "QA: "
        f"range {int(data.min())}–{int(data.max())} · "
        f"blank frames 0/{FRAME_COUNT} · "
        f"12-bit saturation {saturation_fraction:.3%} · "
        f"zero-valued pixels {zero_fraction:.2%} · "
        f"adjacent correlation {np.median(adjacent_corr):.4f} median",
        ha="left",
        color="#3f4b55",
        fontsize=9,
    )

    fig.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.08)
    fig.savefig(output_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)

    return {
        "global_min": int(data.min()),
        "global_max": int(data.max()),
        "global_mean": float(data.mean()),
        "frame_mean_cv": float(frame_mean.std() / frame_mean.mean()),
        "saturation_fraction": saturation_fraction,
        "zero_fraction": zero_fraction,
        "median_adjacent_correlation": float(np.median(adjacent_corr)),
        "blank_frames": int(np.count_nonzero(flat.std(axis=1) == 0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="measurements_128_px4_active512_test_memmap.npy",
    )
    parser.add_argument(
        "--output",
        default="measurements_128_px4_active512_overview.png",
    )
    args = parser.parse_args()

    data = load_measurements(args.input)
    summary = create_overview(data, args.output)
    print(f"Saved overview to {args.output}")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
