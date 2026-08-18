"""Streaming QA for the full 65536-frame, 128 x 128 TM measurement."""

import argparse
import json
import os
import zlib

import matplotlib.pyplot as plt
import numpy as np


FRAME_COUNT = 65536
ROI_HEIGHT = 128
ROI_WIDTH = 128
BATCH_SIZE = 1000
CHUNK_SIZE = 256


def _write_json(path, payload):
    temporary_path = path + ".partial"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary_path, path)


def _pearson_pair(first, second):
    first = first.astype(np.float32, copy=False)
    second = second.astype(np.float32, copy=False)
    first = first - first.mean()
    second = second - second.mean()
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 0:
        return np.nan
    return float(np.dot(first, second) / denominator)


def scan_measurements(path):
    expected_bytes = FRAME_COUNT * ROI_HEIGHT * ROI_WIDTH * np.dtype(np.uint16).itemsize
    actual_bytes = os.path.getsize(path)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"File has {actual_bytes} bytes; expected {expected_bytes} bytes for "
            f"({FRAME_COUNT}, {ROI_HEIGHT}, {ROI_WIDTH}) uint16"
        )

    data = np.memmap(
        path,
        dtype=np.uint16,
        mode="r",
        shape=(FRAME_COUNT, ROI_HEIGHT, ROI_WIDTH),
    )

    frame_mean = np.empty(FRAME_COUNT, dtype=np.float64)
    frame_std = np.empty(FRAME_COUNT, dtype=np.float64)
    frame_peak = np.empty(FRAME_COUNT, dtype=np.uint16)
    frame_zero_fraction = np.empty(FRAME_COUNT, dtype=np.float32)
    adjacent_correlation = np.empty(FRAME_COUNT - 1, dtype=np.float32)
    adjacent_nrmse = np.empty(FRAME_COUNT - 1, dtype=np.float32)
    checksums = np.empty(FRAME_COUNT, dtype=np.uint32)

    pixel_sum = np.zeros((ROI_HEIGHT, ROI_WIDTH), dtype=np.float64)
    pixel_sum_squared = np.zeros((ROI_HEIGHT, ROI_WIDTH), dtype=np.float64)
    zero_count = 0
    saturation_count = 0
    sampled_values = []
    previous_frame = None

    for start in range(0, FRAME_COUNT, CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, FRAME_COUNT)
        raw = np.asarray(data[start:stop])
        frames = raw.astype(np.float32)
        flat = frames.reshape(stop - start, -1)

        means = flat.mean(axis=1, dtype=np.float64)
        stds = flat.std(axis=1, dtype=np.float64)
        frame_mean[start:stop] = means
        frame_std[start:stop] = stds
        frame_peak[start:stop] = raw.reshape(stop - start, -1).max(axis=1)
        frame_zero_fraction[start:stop] = (raw == 0).reshape(stop - start, -1).mean(axis=1)

        pixel_sum += frames.sum(axis=0, dtype=np.float64)
        pixel_sum_squared += np.square(frames, dtype=np.float32).sum(axis=0, dtype=np.float64)
        zero_count += int(np.count_nonzero(raw == 0))
        saturation_count += int(np.count_nonzero(raw >= 4095))

        for local_index in range(stop - start):
            checksums[start + local_index] = zlib.crc32(raw[local_index].tobytes())

        centered = flat - means.astype(np.float32)[:, None]
        norms = np.linalg.norm(centered, axis=1)
        if len(centered) > 1:
            numerator = np.einsum("ij,ij->i", centered[:-1], centered[1:])
            denominator = norms[:-1] * norms[1:]
            adjacent_correlation[start : stop - 1] = numerator / denominator

            differences = flat[1:] - flat[:-1]
            pair_scale = np.maximum((means[1:] + means[:-1]) / 2.0, 1.0)
            adjacent_nrmse[start : stop - 1] = (
                np.sqrt(np.mean(np.square(differences), axis=1)) / pair_scale
            )

        if previous_frame is not None:
            adjacent_correlation[start - 1] = _pearson_pair(previous_frame, flat[0])
            adjacent_nrmse[start - 1] = (
                np.sqrt(np.mean(np.square(flat[0] - previous_frame)))
                / max((frame_mean[start - 1] + frame_mean[start]) / 2.0, 1.0)
            )
        previous_frame = flat[-1].copy()

        # Deterministic distribution sample: every 256th frame and every 4th pixel.
        for global_index in range(start, stop):
            if global_index % 256 == 0:
                sampled_values.append(np.asarray(data[global_index, ::4, ::4]).ravel())

        if stop % 4096 == 0 or stop == FRAME_COUNT:
            print(f"Scanned {stop}/{FRAME_COUNT} frames")

    sampled_values = np.concatenate(sampled_values).astype(np.uint16, copy=False)
    pixel_mean = pixel_sum / FRAME_COUNT
    pixel_variance = np.maximum(pixel_sum_squared / FRAME_COUNT - pixel_mean**2, 0.0)
    pixel_std = np.sqrt(pixel_variance)

    # Verify checksum collisions byte-for-byte before calling them exact duplicates.
    checksum_groups = {}
    for index, checksum in enumerate(checksums.tolist()):
        checksum_groups.setdefault(checksum, []).append(index)
    exact_duplicate_groups = []
    for group in checksum_groups.values():
        if len(group) < 2:
            continue
        verified = []
        while group:
            reference = group.pop(0)
            matches = [reference]
            remaining = []
            for candidate in group:
                if np.array_equal(data[reference], data[candidate]):
                    matches.append(candidate)
                else:
                    remaining.append(candidate)
            if len(matches) > 1:
                verified.append(matches)
            group = remaining
        exact_duplicate_groups.extend(verified)

    batch_starts = np.arange(0, FRAME_COUNT, BATCH_SIZE, dtype=int)
    boundary_pair_indices = batch_starts[1:] - 1
    boundary_correlation = adjacent_correlation[boundary_pair_indices]
    boundary_nrmse = adjacent_nrmse[boundary_pair_indices]
    nonboundary_mask = np.ones(FRAME_COUNT - 1, dtype=bool)
    nonboundary_mask[boundary_pair_indices] = False

    batch_mean = np.asarray(
        [frame_mean[start : min(start + BATCH_SIZE, FRAME_COUNT)].mean() for start in batch_starts]
    )
    batch_std = np.asarray(
        [frame_mean[start : min(start + BATCH_SIZE, FRAME_COUNT)].std() for start in batch_starts]
    )
    batch_centers = np.asarray(
        [(start + min(start + BATCH_SIZE, FRAME_COUNT) - 1) / 2 for start in batch_starts]
    )

    # If a white stability frame leaked into every batch, batch-start images
    # would be mutually highly correlated.  Mask the trivial diagonal.
    start_frames = np.asarray(data[batch_starts], dtype=np.float32).reshape(len(batch_starts), -1)
    start_frames -= start_frames.mean(axis=1, keepdims=True)
    start_norm = np.linalg.norm(start_frames, axis=1)
    start_correlation = (start_frames @ start_frames.T) / (
        start_norm[:, None] * start_norm[None, :]
    )
    np.fill_diagonal(start_correlation, np.nan)

    median_mean = float(np.median(frame_mean))
    mean_mad = float(np.median(np.abs(frame_mean - median_mean)))
    robust_scale = max(1.4826 * mean_mad, np.finfo(float).eps)
    mean_outliers = np.flatnonzero(np.abs(frame_mean - median_mean) > 6.0 * robust_scale)
    blank_frames = np.flatnonzero(frame_std < 1.0)
    near_duplicate_pairs = np.flatnonzero(adjacent_correlation > 0.90)

    first_five_batch_mean = float(batch_mean[:5].mean())
    last_five_batch_mean = float(batch_mean[-5:].mean())
    drift_fraction = (last_five_batch_mean - first_five_batch_mean) / first_five_batch_mean

    selected_indices = [0, 999, 1000, FRAME_COUNT // 2, FRAME_COUNT - 1]
    selected_frames = {index: np.asarray(data[index]).copy() for index in selected_indices}

    total_pixels = FRAME_COUNT * ROI_HEIGHT * ROI_WIDTH
    summary = {
        "source_file": os.path.abspath(path),
        "file_bytes": actual_bytes,
        "expected_bytes": expected_bytes,
        "shape": [FRAME_COUNT, ROI_HEIGHT, ROI_WIDTH],
        "dtype": "uint16",
        "global_min": int(data.min()),
        "global_max": int(data.max()),
        "global_mean": float(frame_mean.mean()),
        "frame_mean_min": float(frame_mean.min()),
        "frame_mean_max": float(frame_mean.max()),
        "frame_mean_cv": float(frame_mean.std() / frame_mean.mean()),
        "zero_fraction": zero_count / total_pixels,
        "saturation_fraction_12bit": saturation_count / total_pixels,
        "blank_frame_count": int(len(blank_frames)),
        "blank_frame_indices": blank_frames[:100].tolist(),
        "exact_duplicate_group_count": int(len(exact_duplicate_groups)),
        "exact_duplicate_groups": exact_duplicate_groups[:100],
        "adjacent_correlation_min": float(np.nanmin(adjacent_correlation)),
        "adjacent_correlation_median": float(np.nanmedian(adjacent_correlation)),
        "adjacent_correlation_max": float(np.nanmax(adjacent_correlation)),
        "near_duplicate_adjacent_pair_count": int(len(near_duplicate_pairs)),
        "near_duplicate_adjacent_pair_starts": near_duplicate_pairs[:100].tolist(),
        "batch_boundary_correlation_min": float(np.nanmin(boundary_correlation)),
        "batch_boundary_correlation_median": float(np.nanmedian(boundary_correlation)),
        "batch_boundary_correlation_max": float(np.nanmax(boundary_correlation)),
        "nonboundary_correlation_median": float(
            np.nanmedian(adjacent_correlation[nonboundary_mask])
        ),
        "batch_boundary_nrmse_median": float(np.nanmedian(boundary_nrmse)),
        "nonboundary_nrmse_median": float(np.nanmedian(adjacent_nrmse[nonboundary_mask])),
        "batch_start_cross_correlation_median": float(np.nanmedian(start_correlation)),
        "batch_start_cross_correlation_max": float(np.nanmax(start_correlation)),
        "batch_mean_min": float(batch_mean.min()),
        "batch_mean_max": float(batch_mean.max()),
        "first_five_batch_mean": first_five_batch_mean,
        "last_five_batch_mean": last_five_batch_mean,
        "first_to_last_drift_fraction": float(drift_fraction),
        "robust_frame_mean_outlier_count": int(len(mean_outliers)),
        "robust_frame_mean_outlier_indices": mean_outliers[:100].tolist(),
    }

    details = {
        "data": data,
        "frame_mean": frame_mean,
        "frame_std": frame_std,
        "frame_peak": frame_peak,
        "frame_zero_fraction": frame_zero_fraction,
        "adjacent_correlation": adjacent_correlation,
        "boundary_correlation": boundary_correlation,
        "nonboundary_correlation": adjacent_correlation[nonboundary_mask],
        "batch_starts": batch_starts,
        "batch_centers": batch_centers,
        "batch_mean": batch_mean,
        "batch_std": batch_std,
        "start_correlation": start_correlation,
        "pixel_mean": pixel_mean,
        "pixel_std": pixel_std,
        "sampled_values": sampled_values,
        "selected_frames": selected_frames,
    }
    return summary, details


def create_overview(summary, details, output_path):
    frame_mean = details["frame_mean"]
    batch_mean = details["batch_mean"]
    batch_centers = details["batch_centers"]
    start_correlation = details["start_correlation"]
    sampled_values = details["sampled_values"]
    selected_frames = details["selected_frames"]

    display_max = float(np.percentile(sampled_values, 99.7))
    histogram_max = float(np.percentile(sampled_values, 99.9))
    selected_indices = [0, 999, 1000, FRAME_COUNT - 1]

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
    grid = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.9], hspace=0.42, wspace=0.30)
    fig.suptitle(
        "Full 128×128 / px=4 Measurement — Data Quality Overview",
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
        "65,536 random phase probes · 66 DMD batches · Camera ROI 128×128 · Raw uint16 intensity",
        ha="left",
        color="#59636e",
        fontsize=10,
    )

    image_axes = []
    image_artist = None
    for column, frame_index in enumerate(selected_indices):
        axis = fig.add_subplot(grid[0, column])
        image_artist = axis.imshow(
            selected_frames[frame_index],
            cmap="cividis",
            vmin=0,
            vmax=display_max,
            interpolation="nearest",
        )
        axis.set_title(f"Frame {frame_index:,}  |  mean {frame_mean[frame_index]:.1f}")
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
    colorbar.set_label(f"Intensity (shared scale, clipped at sampled p99.7 = {display_max:.0f})")

    mean_axis = fig.add_subplot(grid[1, 0])
    mean_artist = mean_axis.imshow(details["pixel_mean"], cmap="cividis")
    mean_axis.set_title("Per-pixel mean")
    mean_axis.set_xticks([])
    mean_axis.set_yticks([])
    fig.colorbar(mean_artist, ax=mean_axis, fraction=0.046, pad=0.03)

    std_axis = fig.add_subplot(grid[1, 1])
    std_artist = std_axis.imshow(details["pixel_std"], cmap="magma")
    std_axis.set_title("Per-pixel standard deviation")
    std_axis.set_xticks([])
    std_axis.set_yticks([])
    fig.colorbar(std_artist, ax=std_axis, fraction=0.046, pad=0.03)

    corr_axis = fig.add_subplot(grid[1, 2])
    finite_corr = start_correlation[np.isfinite(start_correlation)]
    corr_limit = max(abs(np.percentile(finite_corr, 1)), abs(np.percentile(finite_corr, 99)), 0.02)
    corr_artist = corr_axis.imshow(
        start_correlation,
        cmap="coolwarm",
        vmin=-corr_limit,
        vmax=corr_limit,
        interpolation="nearest",
    )
    corr_axis.set_title("Batch-start cross-correlation")
    corr_axis.set_xlabel("Batch")
    corr_axis.set_ylabel("Batch")
    fig.colorbar(corr_artist, ax=corr_axis, fraction=0.046, pad=0.03)

    hist_axis = fig.add_subplot(grid[1, 3])
    hist_axis.hist(
        sampled_values,
        bins=np.linspace(0, histogram_max, 70),
        color="#246a9b",
        edgecolor="none",
    )
    hist_axis.set_title("Sampled pixel-intensity distribution")
    hist_axis.set_xlabel("Intensity (up to sampled p99.9)")
    hist_axis.grid(axis="y", color="#dce2e7", linewidth=0.7)

    trend_axis = fig.add_subplot(grid[2, 0:2])
    frame_index = np.arange(FRAME_COUNT)
    trend_axis.plot(
        frame_index[::16],
        frame_mean[::16],
        color="#8fb9d2",
        linewidth=0.55,
        alpha=0.75,
        label="Frame mean (1/16 shown)",
    )
    trend_axis.plot(
        batch_centers,
        batch_mean,
        color="#246a9b",
        linewidth=1.8,
        marker="o",
        markersize=2.7,
        label="Batch mean",
    )
    trend_axis.set_title(
        "Mean intensity across acquisition  |  "
        f"first→last drift {summary['first_to_last_drift_fraction']:+.2%}"
    )
    trend_axis.set_xlabel("Frame index")
    trend_axis.set_ylabel("Mean intensity")
    trend_axis.set_xlim(0, FRAME_COUNT - 1)
    trend_axis.grid(color="#dce2e7", linewidth=0.7)
    trend_axis.legend(frameon=False, loc="best")

    boundary_axis = fig.add_subplot(grid[2, 2:4])
    bins = np.linspace(
        min(np.percentile(details["nonboundary_correlation"], 0.5), np.min(details["boundary_correlation"])),
        max(np.percentile(details["nonboundary_correlation"], 99.5), np.max(details["boundary_correlation"])),
        55,
    )
    boundary_axis.hist(
        details["nonboundary_correlation"],
        bins=bins,
        density=True,
        color="#8fb9d2",
        alpha=0.75,
        label="Within batch",
    )
    boundary_axis.hist(
        details["boundary_correlation"],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.0,
        color="#c58b22",
        label="Across batch boundary",
    )
    boundary_axis.set_title(
        "Adjacent-frame correlation  |  "
        f"boundary median {summary['batch_boundary_correlation_median']:.4f}"
    )
    boundary_axis.set_xlabel("Pearson correlation")
    boundary_axis.set_ylabel("Density")
    boundary_axis.grid(axis="y", color="#dce2e7", linewidth=0.7)
    boundary_axis.legend(frameon=False, loc="upper right")

    fig.text(
        0.06,
        0.018,
        "QA: "
        f"range {summary['global_min']}–{summary['global_max']} · "
        f"blank {summary['blank_frame_count']}/{FRAME_COUNT:,} · "
        f"exact duplicate groups {summary['exact_duplicate_group_count']} · "
        f"near-duplicate adjacent pairs {summary['near_duplicate_adjacent_pair_count']} · "
        f"12-bit saturation {summary['saturation_fraction_12bit']:.3%} · "
        f"batch-start max correlation {summary['batch_start_cross_correlation_max']:.4f}",
        ha="left",
        color="#3f4b55",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.08)
    fig.savefig(output_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="measurements_128_px4_active512_full_memmap.npy"
    )
    parser.add_argument(
        "--output",
        default="measurements_128_px4_active512_full_quality_overview.png",
    )
    parser.add_argument(
        "--summary",
        default="measurements_128_px4_active512_full_quality_summary.json",
    )
    args = parser.parse_args()

    summary, details = scan_measurements(args.input)
    _write_json(args.summary, summary)
    create_overview(summary, details, args.output)
    print(f"Saved summary to {args.summary}")
    print(f"Saved overview to {args.output}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
