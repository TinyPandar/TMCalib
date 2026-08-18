"""Estimate the mean measured speckle size from 128 x 128 camera frames.

The reported size is the FWHM of the central peak of the spatial intensity
autocovariance.  Frames are normalized by their own mean and the ensemble
mean image is removed before correlation, which suppresses illumination drift
and fixed camera/optical structure.  A two-dimensional elliptical Gaussian is
fit to non-zero lags around the correlation peak so that detector noise at the
single zero-lag sample does not set the width.

The measurement files in this project are raw uint16 memmaps despite their
``.npy`` suffix; they do not contain a NumPy header.
"""

import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares


DEFAULT_SHAPE = (65536, 128, 128)


def open_measurements(path, frame_count, height, width):
    shape = (frame_count, height, width)
    expected_bytes = int(np.prod(shape)) * np.dtype(np.uint16).itemsize
    actual_bytes = os.path.getsize(path)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"File has {actual_bytes} bytes; expected {expected_bytes} bytes "
            f"for raw uint16 shape {shape}."
        )
    return np.memmap(path, dtype=np.uint16, mode="r", shape=shape)


def detect_contiguous_valid_prefix(data, chunk_size=256):
    """Return the number of populated frames in a zero-preallocated memmap."""
    first_blank = None
    nonblank_after_blank = None

    for start in range(0, len(data), chunk_size):
        stop = min(start + chunk_size, len(data))
        chunk = np.asarray(data[start:stop])
        populated = np.any(chunk != 0, axis=(1, 2))

        if first_blank is None and not np.all(populated):
            first_blank = start + int(np.flatnonzero(~populated)[0])

        if first_blank is not None:
            absolute_indices = start + np.flatnonzero(populated)
            later = absolute_indices[absolute_indices >= first_blank]
            if len(later):
                nonblank_after_blank = int(later[0])
                break

    if nonblank_after_blank is not None:
        raise ValueError(
            "The file contains a populated frame after an all-zero frame "
            f"(first blank {first_blank}, later populated {nonblank_after_blank}); "
            "valid frames are not a contiguous prefix."
        )
    return len(data) if first_blank is None else first_blank


def choose_frame_indices(valid_frame_count, sample_frame_count):
    if sample_frame_count <= 0 or sample_frame_count >= valid_frame_count:
        return np.arange(valid_frame_count, dtype=np.int64), "all valid frames"
    indices = np.linspace(
        0,
        valid_frame_count - 1,
        sample_frame_count,
        dtype=np.int64,
    )
    return np.unique(indices), "uniform across the populated prefix"


def calculate_normalized_mean_map(data, indices, batch_size):
    total = np.zeros(data.shape[1:], dtype=np.float64)
    for start in range(0, len(indices), batch_size):
        selection = indices[start : start + batch_size]
        frames = np.asarray(data[selection], dtype=np.float32)
        frame_means = frames.mean(axis=(1, 2), keepdims=True)
        if np.any(frame_means <= 0):
            raise ValueError("Selected frames include an all-zero frame.")
        frames /= frame_means
        total += frames.sum(axis=0, dtype=np.float64)
    return total / len(indices)


def accumulate_power_spectra(
    data,
    indices,
    normalized_mean_map,
    crop_size,
    batch_size,
    valid_frame_count,
    block_count,
):
    height, width = data.shape[1:]
    if crop_size > min(height, width) or crop_size < 16:
        raise ValueError("crop_size must be between 16 and the smaller ROI dimension.")

    crop_y = (height - crop_size) // 2
    crop_x = (width - crop_size) // 2
    transform_shape = (2 * crop_size, 2 * crop_size)
    spectrum_shape = (transform_shape[0], transform_shape[1] // 2 + 1)
    total_power = np.zeros(spectrum_shape, dtype=np.float64)
    block_power = np.zeros((block_count,) + spectrum_shape, dtype=np.float64)
    block_sizes = np.zeros(block_count, dtype=np.int64)

    block_ids = np.minimum(
        indices * block_count // valid_frame_count,
        block_count - 1,
    )

    for start in range(0, len(indices), batch_size):
        selection = indices[start : start + batch_size]
        selection_blocks = block_ids[start : start + batch_size]
        frames = np.asarray(data[selection], dtype=np.float32)
        frames /= frames.mean(axis=(1, 2), keepdims=True)
        residual = frames - normalized_mean_map[None, :, :]
        residual = residual[
            :,
            crop_y : crop_y + crop_size,
            crop_x : crop_x + crop_size,
        ]
        residual -= residual.mean(axis=(1, 2), keepdims=True)

        transforms = np.fft.rfft2(
            residual,
            s=transform_shape,
            axes=(-2, -1),
        )
        powers = np.square(transforms.real) + np.square(transforms.imag)
        total_power += powers.sum(axis=0, dtype=np.float64)

        for block_index in np.unique(selection_blocks):
            mask = selection_blocks == block_index
            block_power[block_index] += powers[mask].sum(axis=0, dtype=np.float64)
            block_sizes[block_index] += int(np.count_nonzero(mask))

    return total_power, block_power, block_sizes


def autocovariance_from_power(power, crop_size):
    transform_shape = (2 * crop_size, 2 * crop_size)
    correlation = np.fft.irfft2(power, s=transform_shape)
    # fftshift gives lags [-crop, ..., crop-1].  Drop the unused -crop lag.
    correlation = np.fft.fftshift(correlation)[1:, 1:]
    lags = np.arange(-(crop_size - 1), crop_size)
    overlap = (crop_size - np.abs(lags))[:, None] * (
        crop_size - np.abs(lags)
    )[None, :]
    correlation = correlation / overlap
    center = crop_size - 1
    peak = float(correlation[center, center])
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("The autocovariance has no positive finite zero-lag peak.")
    return correlation / peak


def fit_elliptical_gaussian(correlation, fit_radius):
    center = (correlation.shape[0] - 1) // 2
    yy, xx = np.indices(correlation.shape)
    x = (xx - center).astype(np.float64)
    y = (yy - center).astype(np.float64)
    radius = np.hypot(x, y)
    mask = (radius >= 0.75) & (radius <= fit_radius)
    x_fit = x[mask]
    y_fit = y[mask]
    observed = correlation[mask]

    def predict(parameters):
        amplitude, sigma_u, sigma_v, angle, baseline = parameters
        cosine = np.cos(angle)
        sine = np.sin(angle)
        u = cosine * x_fit + sine * y_fit
        v = -sine * x_fit + cosine * y_fit
        return amplitude * np.exp(
            -0.5 * (np.square(u / sigma_u) + np.square(v / sigma_v))
        ) + baseline

    fit = least_squares(
        lambda parameters: predict(parameters) - observed,
        x0=np.array([1.0, 2.0, 1.8, -0.6, 0.0]),
        bounds=(
            np.array([0.0, 0.2, 0.2, -np.pi / 2.0, -0.2]),
            np.array([2.0, 20.0, 20.0, np.pi / 2.0, 0.2]),
        ),
    )
    amplitude, sigma_u, sigma_v, angle, baseline = fit.x
    fwhm_factor = 2.0 * math.sqrt(2.0 * math.log(2.0))
    fwhm_u = fwhm_factor * sigma_u
    fwhm_v = fwhm_factor * sigma_v

    if fwhm_u >= fwhm_v:
        major_fwhm, minor_fwhm = fwhm_u, fwhm_v
        major_angle = angle
    else:
        major_fwhm, minor_fwhm = fwhm_v, fwhm_u
        major_angle = angle + np.pi / 2.0

    major_angle_degrees = (np.degrees(major_angle) + 90.0) % 180.0 - 90.0
    fitted = predict(fit.x)
    rmse = float(np.sqrt(np.mean(np.square(fitted - observed))))

    return {
        "amplitude": float(amplitude),
        "baseline": float(baseline),
        "sigma_u_pixels": float(sigma_u),
        "sigma_v_pixels": float(sigma_v),
        "major_fwhm_pixels": float(major_fwhm),
        "minor_fwhm_pixels": float(minor_fwhm),
        "equivalent_fwhm_pixels": float(math.sqrt(major_fwhm * minor_fwhm)),
        "major_axis_angle_degrees_from_positive_x_toward_image_y": float(
            major_angle_degrees
        ),
        "fit_rmse": rmse,
    }


def symmetric_axis_profiles(correlation):
    center = (correlation.shape[0] - 1) // 2
    x_profile = np.concatenate(
        (
            [correlation[center, center]],
            0.5
            * (
                correlation[center, center + 1 :]
                + correlation[center, center - 1 :: -1]
            ),
        )
    )
    y_profile = np.concatenate(
        (
            [correlation[center, center]],
            0.5
            * (
                correlation[center + 1 :, center]
                + correlation[center - 1 :: -1, center]
            ),
        )
    )
    return x_profile, y_profile


def direct_fwhm(profile):
    for index in range(1, len(profile)):
        if profile[index] <= 0.5:
            left = float(profile[index - 1])
            right = float(profile[index])
            crossing = (index - 1) + (0.5 - left) / (right - left)
            return float(2.0 * crossing)
    return float("nan")


def create_figure(
    data,
    example_frame_index,
    correlation,
    fit_result,
    block_fwhm,
    output_path,
):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor="#fbfcfd")
    fig.suptitle(
        "Measured speckle size from spatial intensity autocovariance",
        fontsize=15,
        fontweight="bold",
    )

    example = np.asarray(data[example_frame_index])
    image_max = float(np.percentile(example, 99.7))
    image = axes[0, 0].imshow(
        example,
        cmap="cividis",
        vmin=0,
        vmax=image_max,
        interpolation="nearest",
    )
    axes[0, 0].set_title(f"Example measured frame {example_frame_index}")
    axes[0, 0].set_xlabel("Camera output x [pixel]")
    axes[0, 0].set_ylabel("Camera output y [pixel]")
    fig.colorbar(image, ax=axes[0, 0], fraction=0.046, pad=0.04, label="Stored intensity")

    center = (correlation.shape[0] - 1) // 2
    radius = 10
    central = correlation[
        center - radius : center + radius + 1,
        center - radius : center + radius + 1,
    ]
    extent = [-radius - 0.5, radius + 0.5, radius + 0.5, -radius - 0.5]
    acf_image = axes[0, 1].imshow(
        central,
        cmap="magma",
        vmin=0,
        vmax=1,
        extent=extent,
        interpolation="nearest",
    )
    coordinates = np.arange(-radius, radius + 1)
    axes[0, 1].contour(
        coordinates,
        coordinates,
        central,
        levels=[0.5],
        colors=["#65d6ce"],
        linewidths=1.5,
    )
    axes[0, 1].set_title("Normalized autocovariance (cyan = half maximum)")
    axes[0, 1].set_xlabel("Lag x [pixel]")
    axes[0, 1].set_ylabel("Lag y [pixel]")
    fig.colorbar(acf_image, ax=axes[0, 1], fraction=0.046, pad=0.04)

    x_profile, y_profile = symmetric_axis_profiles(correlation)
    profile_limit = min(12, len(x_profile))
    lags = np.arange(profile_limit)
    axes[1, 0].plot(lags, x_profile[:profile_limit], "o-", label="x axis")
    axes[1, 0].plot(lags, y_profile[:profile_limit], "s-", label="y axis")
    axes[1, 0].axhline(0.5, color="#cf6f2e", linestyle="--", label="half maximum")
    axes[1, 0].set_xlim(0, profile_limit - 1)
    axes[1, 0].set_ylim(-0.08, 1.05)
    axes[1, 0].set_xlabel("Absolute lag [pixel]")
    axes[1, 0].set_ylabel("Normalized autocovariance")
    axes[1, 0].set_title("Symmetric detector-axis profiles")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(alpha=0.2)

    valid_blocks = np.asarray(block_fwhm, dtype=np.float64)
    block_x = np.arange(1, len(valid_blocks) + 1)
    axes[1, 1].plot(block_x, valid_blocks, "o-", color="#2f6f9f")
    axes[1, 1].axhline(
        fit_result["equivalent_fwhm_pixels"],
        color="#cf6f2e",
        linestyle="--",
        label="all sampled frames",
    )
    axes[1, 1].set_xlabel("Acquisition segment")
    axes[1, 1].set_ylabel("Equivalent FWHM [camera output pixel]")
    axes[1, 1].set_title("Temporal consistency")
    axes[1, 1].grid(alpha=0.2)
    axes[1, 1].legend(frameon=False)

    fig.text(
        0.5,
        0.015,
        (
            f"Elliptical Gaussian fit: {fit_result['major_fwhm_pixels']:.3f} x "
            f"{fit_result['minor_fwhm_pixels']:.3f} px; equivalent mean "
            f"{fit_result['equivalent_fwhm_pixels']:.3f} px"
        ),
        ha="center",
        color="#263238",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def write_json(path, payload):
    temporary_path = path + ".partial"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary_path, path)


def analyze(arguments):
    data = open_measurements(
        arguments.input,
        arguments.frame_count,
        arguments.height,
        arguments.width,
    )
    valid_frame_count = (
        arguments.valid_frames
        if arguments.valid_frames is not None
        else detect_contiguous_valid_prefix(data, arguments.batch_size)
    )
    if not 1 <= valid_frame_count <= len(data):
        raise ValueError(
            f"valid_frames must be between 1 and {len(data)}, got {valid_frame_count}."
        )

    indices, sample_strategy = choose_frame_indices(
        valid_frame_count,
        arguments.sample_frames,
    )
    normalized_mean_map = calculate_normalized_mean_map(
        data,
        indices,
        arguments.batch_size,
    )
    total_power, block_power, block_sizes = accumulate_power_spectra(
        data=data,
        indices=indices,
        normalized_mean_map=normalized_mean_map,
        crop_size=arguments.crop_size,
        batch_size=arguments.batch_size,
        valid_frame_count=valid_frame_count,
        block_count=arguments.block_count,
    )

    correlation = autocovariance_from_power(total_power, arguments.crop_size)
    fit_result = fit_elliptical_gaussian(correlation, arguments.fit_radius)
    x_profile, y_profile = symmetric_axis_profiles(correlation)
    x_direct = direct_fwhm(x_profile)
    y_direct = direct_fwhm(y_profile)

    block_results = []
    for block_index in range(arguments.block_count):
        if block_sizes[block_index] == 0:
            continue
        block_correlation = autocovariance_from_power(
            block_power[block_index],
            arguments.crop_size,
        )
        block_fit = fit_elliptical_gaussian(
            block_correlation,
            arguments.fit_radius,
        )
        block_results.append(
            {
                "block": block_index + 1,
                "sampled_frame_count": int(block_sizes[block_index]),
                "equivalent_fwhm_pixels": block_fit["equivalent_fwhm_pixels"],
                "major_fwhm_pixels": block_fit["major_fwhm_pixels"],
                "minor_fwhm_pixels": block_fit["minor_fwhm_pixels"],
            }
        )

    block_equivalent = np.asarray(
        [result["equivalent_fwhm_pixels"] for result in block_results],
        dtype=np.float64,
    )
    block_std = float(block_equivalent.std(ddof=1)) if len(block_equivalent) > 1 else 0.0

    physical_size = None
    if arguments.raw_pixel_pitch_um is not None:
        output_pitch = 2.0 * arguments.raw_pixel_pitch_um
        physical_size = {
            "raw_sensor_pixel_pitch_um": float(arguments.raw_pixel_pitch_um),
            "polarization_channel_output_pitch_um": float(output_pitch),
            "major_fwhm_um": float(fit_result["major_fwhm_pixels"] * output_pitch),
            "minor_fwhm_um": float(fit_result["minor_fwhm_pixels"] * output_pitch),
            "equivalent_fwhm_um": float(
                fit_result["equivalent_fwhm_pixels"] * output_pitch
            ),
        }

    output_json = arguments.output_prefix + ".json"
    output_figure = arguments.output_prefix + ".png"
    if physical_size is None:
        physical_size_note = (
            "The saved 128 x 128 I90 polarization channel samples one pixel "
            "from each 2 x 2 raw sensor cell, so its output sampling pitch is "
            "twice the raw sensor pixel pitch. Supply --raw-pixel-pitch-um to "
            "calculate micrometres."
        )
    else:
        physical_size_note = (
            "The saved 128 x 128 I90 polarization channel samples one pixel "
            "from each 2 x 2 raw sensor cell. Physical sizes therefore use an "
            f"output sampling pitch of {physical_size['polarization_channel_output_pitch_um']:.6g} um."
        )

    result = {
        "source_file": os.path.abspath(arguments.input),
        "source_file_bytes": os.path.getsize(arguments.input),
        "storage_shape": [arguments.frame_count, arguments.height, arguments.width],
        "valid_frame_count": int(valid_frame_count),
        "trailing_blank_frame_count": int(len(data) - valid_frame_count),
        "sampled_frame_count": int(len(indices)),
        "sample_strategy": sample_strategy,
        "sampled_frame_range": [int(indices[0]), int(indices[-1])],
        "analysis_crop": [arguments.crop_size, arguments.crop_size],
        "preprocessing": (
            "Normalize each frame by its spatial mean; subtract the normalized "
            "ensemble mean image; subtract the residual frame mean."
        ),
        "size_definition": (
            "FWHM of an elliptical Gaussian fit to the central intensity "
            "autocovariance peak, excluding the zero-lag sample."
        ),
        "fit_lag_radius_pixels": float(arguments.fit_radius),
        "fwhm": {
            **fit_result,
            "equivalent_hwhm_pixels": float(
                0.5 * fit_result["equivalent_fwhm_pixels"]
            ),
            "direct_x_axis_fwhm_pixels": x_direct,
            "direct_y_axis_fwhm_pixels": y_direct,
        },
        "temporal_segment_repeatability": {
            "segment_count": len(block_results),
            "equivalent_fwhm_mean_pixels": float(block_equivalent.mean()),
            "equivalent_fwhm_std_pixels": block_std,
            "equivalent_fwhm_min_pixels": float(block_equivalent.min()),
            "equivalent_fwhm_max_pixels": float(block_equivalent.max()),
            "segments": block_results,
        },
        "physical_size": physical_size,
        "physical_size_note": physical_size_note,
        "outputs": {
            "json": os.path.abspath(output_json),
            "figure": os.path.abspath(output_figure),
        },
    }
    write_json(output_json, result)
    create_figure(
        data=data,
        example_frame_index=int(indices[len(indices) // 2]),
        correlation=correlation,
        fit_result=fit_result,
        block_fwhm=block_equivalent,
        output_path=output_figure,
    )
    return result


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="measurements_128_px4_active512_full_memmap.npy",
        help="Headerless uint16 measurement memmap.",
    )
    parser.add_argument("--frame-count", type=int, default=DEFAULT_SHAPE[0])
    parser.add_argument("--height", type=int, default=DEFAULT_SHAPE[1])
    parser.add_argument("--width", type=int, default=DEFAULT_SHAPE[2])
    parser.add_argument(
        "--valid-frames",
        type=int,
        default=None,
        help="Populated prefix length; detected automatically when omitted.",
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=0,
        help="Uniform sample size; 0 uses every valid frame.",
    )
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--fit-radius", type=float, default=7.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-count", type=int, default=12)
    parser.add_argument("--raw-pixel-pitch-um", type=float, default=None)
    parser.add_argument(
        "--output-prefix",
        default="speckle_size_128_active512_current",
    )
    return parser


if __name__ == "__main__":
    analysis = analyze(build_parser().parse_args())
    print(json.dumps(analysis, indent=2, ensure_ascii=False))
