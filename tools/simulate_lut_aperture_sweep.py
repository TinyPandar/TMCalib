"""Simulate first-order Fourier filtering for TMCalib complex DMD holograms.

This tool is intended to answer a practical question: can an aperture chosen for
one logical input profile pass unwanted Fourier content for another profile?
It does not model the scattering medium.  It only models the DMD binary
hologram, the Fourier-plane circular aperture, recentering of the selected first
order, and recovery of the commanded logical complex field.

The aperture radius is expressed in *x-axis FFT bins* of the full 1024-pixel DMD
width.  The mask is circular in physical spatial-frequency coordinates, so its
y-axis radius is scaled by H/W (e.g. radius_x=128 corresponds to radius_y=96 on
a 768x1024 FFT).

By default the script analyzes the existing 32x24 amplitude+phase correction
patterns and also synthesizes matched 160x120 fields using the repository's
current 160x120 encoder.  This makes the two curves directly comparable for the
same idealized physical aperture.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dmd_pattern_160x120 import (
    EXPANDED_HEIGHT as EXPANDED_HEIGHT_160,
    EXPANDED_WIDTH as EXPANDED_WIDTH_160,
    INPUT_HEIGHT as INPUT_HEIGHT_160,
    INPUT_WIDTH as INPUT_WIDTH_160,
    SOURCE_X_INDICES,
    SOURCE_Y_INDICES,
    input_field_to_dmd_pattern as input_field_to_dmd_pattern_160,
)
from holograms.generate_LUT import generate_lut
from tools.generate_complex_correction_probes_32x24 import (
    DMD_HEIGHT,
    DMD_WIDTH,
    LOGICAL_PIXEL_SIZE,
    LUT_STEP,
    N_X,
    N_Y,
    amplitude_levels,
    phase_values,
)


DEFAULT_PATTERN_DIR = "correction_patterns_32x24_complex"
DEFAULT_OUTPUT_DIR = "lut_aperture_sweep"
DEFAULT_RADII = (16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224)
SUPERPIXEL_SIZE = 4
CARRIER_FX = 1.0 / SUPERPIXEL_SIZE
CARRIER_FY = 1.0 / (SUPERPIXEL_SIZE * SUPERPIXEL_SIZE)


def align_complex_gain(reference, recovered):
    """Align one irrelevant global complex gain before computing errors."""
    reference = np.asarray(reference, dtype=np.complex128)
    recovered = np.asarray(recovered, dtype=np.complex128)
    denominator = np.vdot(recovered.ravel(), recovered.ravel())
    if abs(denominator) <= 1e-30:
        return np.zeros_like(recovered), 0.0j
    gain = np.vdot(recovered.ravel(), reference.ravel()) / denominator
    return gain * recovered, complex(gain)


def complex_fidelity(reference, recovered):
    reference = np.asarray(reference, dtype=np.complex128).ravel()
    recovered = np.asarray(recovered, dtype=np.complex128).ravel()
    denominator = np.linalg.norm(reference) * np.linalg.norm(recovered)
    if denominator <= 1e-30:
        return 0.0
    return float(abs(np.vdot(reference, recovered)) / denominator)


def recovery_metrics(reference, recovered):
    """Return gain-invariant complex, amplitude, and phase recovery metrics."""
    reference = np.asarray(reference, dtype=np.complex128)
    aligned, gain = align_complex_gain(reference, recovered)
    reference_norm = np.linalg.norm(reference)
    nmse = float(
        np.linalg.norm(reference - aligned) ** 2
        / max(reference_norm**2, 1e-30)
    )
    ref_amp = np.abs(reference)
    aligned_amp = np.abs(aligned)
    amp_nrmse = float(
        np.linalg.norm(ref_amp - aligned_amp)
        / max(np.linalg.norm(ref_amp), 1e-30)
    )
    phase_error = np.angle(aligned * np.conj(reference))
    # The generated correction dataset never uses zero amplitude by default.
    # Keep a small guard so custom datasets with zeros do not contaminate phase
    # statistics with undefined phase.
    valid_phase = ref_amp > max(1e-6, float(np.max(ref_amp)) * 1e-4)
    if np.any(valid_phase):
        phase_rmse_deg = float(
            np.sqrt(np.mean(phase_error[valid_phase] ** 2)) * 180.0 / np.pi
        )
        phase_mae_deg = float(
            np.mean(np.abs(phase_error[valid_phase])) * 180.0 / np.pi
        )
    else:
        phase_rmse_deg = float("nan")
        phase_mae_deg = float("nan")
    return {
        "fidelity": complex_fidelity(reference, recovered),
        "nmse": nmse,
        "amplitude_nrmse": amp_nrmse,
        "phase_rmse_deg": phase_rmse_deg,
        "phase_mae_deg": phase_mae_deg,
        "complex_gain_real": float(np.real(gain)),
        "complex_gain_imag": float(np.imag(gain)),
    }


def first_order_geometry(shape, order_sign=-1):
    """Return shifted-FFT centre and the holo_SP direct first-order centre."""
    height, width = (int(shape[0]), int(shape[1]))
    centre_y = height // 2
    centre_x = width // 2
    offset_x = int(round(float(order_sign) * CARRIER_FX * width))
    offset_y = int(round(float(order_sign) * CARRIER_FY * height))
    return {
        "fft_center_yx": (centre_y, centre_x),
        "order_center_yx": (centre_y + offset_y, centre_x + offset_x),
        "order_offset_yx": (offset_y, offset_x),
    }


def circular_frequency_mask(shape, center_yx, radius_x_bins):
    """Build a physical-frequency circle on a rectangular FFT grid.

    One x FFT bin equals 1/W cycles/mirror while one y FFT bin equals 1/H.
    Measuring the radius in x bins therefore gives

        distance_x_bins^2 = dx^2 + (dy * W/H)^2.
    """
    height, width = (int(shape[0]), int(shape[1]))
    center_y, center_x = (float(center_yx[0]), float(center_yx[1]))
    radius_x_bins = float(radius_x_bins)
    if radius_x_bins <= 0:
        raise ValueError("aperture radius must be positive")
    yy, xx = np.ogrid[:height, :width]
    dx = xx - center_x
    dy_as_x_bins = (yy - center_y) * (float(width) / float(height))
    return dx * dx + dy_as_x_bins * dy_as_x_bins <= radius_x_bins**2


def filter_first_order(pattern, radius_x_bins, order_sign=-1):
    """Filter and recenter one holo_SP first order on a full DMD pattern."""
    pattern = np.asarray(pattern, dtype=np.float32)
    if pattern.shape != (DMD_HEIGHT, DMD_WIDTH):
        raise ValueError(
            "DMD pattern shape {} does not match {}".format(
                pattern.shape, (DMD_HEIGHT, DMD_WIDTH)
            )
        )
    # Work with 0/1 mirror states. Absolute FFT scale is irrelevant because all
    # comparison metrics remove one global complex gain.
    if np.max(pattern) > 1.0:
        pattern = pattern / 255.0
    geometry = first_order_geometry(pattern.shape, order_sign=order_sign)
    spectrum = np.fft.fftshift(np.fft.fft2(pattern))
    mask = circular_frequency_mask(
        pattern.shape,
        geometry["order_center_yx"],
        radius_x_bins,
    )
    filtered = spectrum * mask
    offset_y, offset_x = geometry["order_offset_yx"]
    recentered = np.roll(filtered, shift=(-offset_y, -offset_x), axis=(0, 1))
    recovered_dmd = np.fft.ifft2(np.fft.ifftshift(recentered))
    total_energy = float(np.sum(np.abs(spectrum) ** 2))
    passed_energy = float(np.sum(np.abs(filtered) ** 2))
    return {
        "recovered_dmd": recovered_dmd,
        "spectrum": spectrum,
        "filtered_spectrum": filtered,
        "mask": mask,
        "passed_energy_fraction": passed_energy / max(total_energy, 1e-30),
        **geometry,
    }


def recover_32x24(recovered_dmd):
    """Average each 32x32 logical block after carrier removal."""
    recovered_dmd = np.asarray(recovered_dmd)
    return recovered_dmd.reshape(
        N_Y,
        LOGICAL_PIXEL_SIZE,
        N_X,
        LOGICAL_PIXEL_SIZE,
    ).mean(axis=(1, 3))


def recover_160x120(recovered_dmd):
    """Undo 4x4 optical SP blocks and the 160x120 nearest-fill expansion."""
    recovered_dmd = np.asarray(recovered_dmd)
    expanded = recovered_dmd.reshape(
        EXPANDED_HEIGHT_160,
        SUPERPIXEL_SIZE,
        EXPANDED_WIDTH_160,
        SUPERPIXEL_SIZE,
    ).mean(axis=(1, 3))

    source_linear = (
        SOURCE_Y_INDICES[:, None] * INPUT_WIDTH_160
        + SOURCE_X_INDICES[None, :]
    ).astype(np.intp)
    sums_real = np.zeros(INPUT_HEIGHT_160 * INPUT_WIDTH_160, dtype=np.float64)
    sums_imag = np.zeros_like(sums_real)
    counts = np.zeros_like(sums_real)
    np.add.at(sums_real, source_linear.ravel(), np.real(expanded).ravel())
    np.add.at(sums_imag, source_linear.ravel(), np.imag(expanded).ravel())
    np.add.at(counts, source_linear.ravel(), 1.0)
    recovered = (sums_real + 1j * sums_imag) / np.maximum(counts, 1.0)
    return recovered.reshape(INPUT_HEIGHT_160, INPUT_WIDTH_160)


def lut_quantization_report(amplitudes, phases):
    """Measure only the ideal 4x4 LUT nearest-point quantization error."""
    field_values, _, lut = generate_lut("sp", SUPERPIXEL_SIZE, step=LUT_STEP)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    phases = np.asarray(phases, dtype=np.complex128)
    center = len(lut) // 2
    rows = []
    for amplitude in amplitudes:
        for phase in phases:
            target = complex(amplitude * phase)
            re = int(np.rint(np.real(target) / LUT_STEP)) + center
            im = int(np.rint(np.imag(target) / LUT_STEP)) + center
            actual = complex(field_values[int(lut[re, im])])
            # The LUT field values are normalized to the same unit disk used by
            # the target amplitude/phase grid.
            phase_error = float(np.angle(actual * np.conj(target)))
            rows.append(
                {
                    "target_amplitude": float(abs(target)),
                    "actual_amplitude": float(abs(actual)),
                    "amplitude_abs_error": float(abs(abs(actual) - abs(target))),
                    "amplitude_rel_error": float(
                        abs(abs(actual) - abs(target)) / max(abs(target), 1e-30)
                    ),
                    "phase_abs_error_rad": abs(phase_error),
                    "phase_abs_error_deg": abs(phase_error) * 180.0 / np.pi,
                    "complex_abs_error": float(abs(actual - target)),
                }
            )
    return rows


def _aggregate_metric_rows(profile, radius, sample_rows):
    result = {
        "profile": profile,
        "radius_x_bins": float(radius),
        "radius_y_bins": float(radius) * DMD_HEIGHT / DMD_WIDTH,
        "radius_cycles_per_mirror": float(radius) / DMD_WIDTH,
        "sample_count": int(len(sample_rows)),
    }
    for key in (
        "fidelity",
        "nmse",
        "amplitude_nrmse",
        "phase_rmse_deg",
        "phase_mae_deg",
        "passed_energy_fraction",
    ):
        values = np.asarray([row[key] for row in sample_rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        result[key + "_mean"] = float(np.mean(finite)) if finite.size else float("nan")
        result[key + "_median"] = float(np.median(finite)) if finite.size else float("nan")
    return result


def evaluate_profile(patterns, references, radii, recover_fn, profile, order_sign):
    rows = []
    average_power = np.zeros((DMD_HEIGHT, DMD_WIDTH), dtype=np.float64)
    for sample_index in range(len(references)):
        pattern = np.asarray(patterns[sample_index], dtype=np.float32)
        spectrum = np.fft.fftshift(np.fft.fft2(pattern / (255.0 if np.max(pattern) > 1 else 1.0)))
        average_power += np.abs(spectrum) ** 2
    average_power /= max(len(references), 1)

    for radius in radii:
        sample_rows = []
        for sample_index, reference in enumerate(references):
            recovery = filter_first_order(
                patterns[sample_index],
                radius_x_bins=radius,
                order_sign=order_sign,
            )
            recovered = recover_fn(recovery["recovered_dmd"])
            metrics = recovery_metrics(reference, recovered)
            metrics["passed_energy_fraction"] = recovery["passed_energy_fraction"]
            sample_rows.append(metrics)
        rows.append(_aggregate_metric_rows(profile, radius, sample_rows))
    return rows, average_power


def generate_matched_160_fields(count, amplitudes, phases, seed):
    rng = np.random.default_rng(int(seed))
    fields = []
    patterns = []
    for _ in range(int(count)):
        amplitude_index = rng.integers(
            0, len(amplitudes), size=(INPUT_HEIGHT_160, INPUT_WIDTH_160)
        )
        phase_index = rng.integers(
            0, len(phases), size=(INPUT_HEIGHT_160, INPUT_WIDTH_160)
        )
        field = (
            np.asarray(amplitudes)[amplitude_index]
            * np.asarray(phases)[phase_index]
        ).astype(np.complex64)
        fields.append(field)
        patterns.append(input_field_to_dmd_pattern_160(field))
    return fields, patterns


def _top_spectrum_peaks(power, count=12, suppression_radius=12):
    """Return coarse non-maximum-suppressed peaks for leakage inspection."""
    work = np.asarray(power, dtype=np.float64).copy()
    peaks = []
    yy, xx = np.ogrid[: work.shape[0], : work.shape[1]]
    for _ in range(int(count)):
        flat_index = int(np.argmax(work))
        value = float(work.ravel()[flat_index])
        if not np.isfinite(value) or value <= 0:
            break
        y, x = np.unravel_index(flat_index, work.shape)
        peaks.append((int(y), int(x), value))
        mask = (yy - y) ** 2 + (xx - x) ** 2 <= int(suppression_radius) ** 2
        work[mask] = -np.inf
    return peaks


def save_spectrum_figure(path, power, radii, order_sign, title):
    geometry = first_order_geometry(power.shape, order_sign=order_sign)
    order_y, order_x = geometry["order_center_yx"]
    image = np.log1p(power)
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    im = ax.imshow(image, cmap="magma")
    ax.plot([DMD_WIDTH // 2], [DMD_HEIGHT // 2], "w+", markersize=8, label="zero order")
    ax.plot([order_x], [order_y], "co", markersize=5, label="selected first order")
    # Draw a few representative circles so the plot stays readable.
    shown = [radii[0], radii[len(radii) // 2], radii[-1]] if len(radii) >= 3 else list(radii)
    theta = np.linspace(0.0, 2.0 * np.pi, 512)
    for radius in shown:
        rx = float(radius)
        ry = rx * DMD_HEIGHT / DMD_WIDTH
        ax.plot(order_x + rx * np.cos(theta), order_y + ry * np.sin(theta), linewidth=1.0, label="r={}".format(radius))
    ax.set_title(title)
    ax.set_xlabel("FFT x bin")
    ax.set_ylabel("FFT y bin")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, label="log(1 + average spectral power)")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_metric_figure(path, rows, metric_key, ylabel, title):
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    profiles = []
    for row in rows:
        if row["profile"] not in profiles:
            profiles.append(row["profile"])
    for profile in profiles:
        subset = [row for row in rows if row["profile"] == profile]
        ax.plot(
            [row["radius_x_bins"] for row in subset],
            [row[metric_key] for row in subset],
            marker="o",
            label=profile,
        )
    ax.set_xlabel("circular aperture radius (x-axis FFT bins)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern-dir", default=DEFAULT_PATTERN_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--order-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument(
        "--radii",
        type=int,
        nargs="+",
        default=list(DEFAULT_RADII),
        help="Aperture radii in x-axis full-DMD FFT bins",
    )
    parser.add_argument(
        "--no-compare-160",
        action="store_true",
        help="Analyze only the existing 32x24 correction dataset",
    )
    args = parser.parse_args()

    if args.sample_count <= 0:
        raise ValueError("sample-count must be positive")
    radii = sorted(set(int(value) for value in args.radii))
    if not radii or radii[0] <= 0:
        raise ValueError("all aperture radii must be positive")

    pattern_dir = Path(args.pattern_dir).resolve()
    probes_path = pattern_dir / "probe.npy"
    patterns_path = pattern_dir / "patterns_pregenerated.npy"
    metadata_path = pattern_dir / "metadata.json"
    if not probes_path.exists() or not patterns_path.exists():
        raise FileNotFoundError(
            "expected probe.npy and patterns_pregenerated.npy under {}".format(pattern_dir)
        )

    probes_mm = np.load(str(probes_path), mmap_mode="r", allow_pickle=False)
    patterns_mm = np.load(str(patterns_path), mmap_mode="r", allow_pickle=False)
    if probes_mm.ndim != 3 or tuple(probes_mm.shape[1:]) != (N_Y, N_X):
        raise ValueError("32x24 probe array has unexpected shape {}".format(probes_mm.shape))
    if patterns_mm.ndim != 3 or tuple(patterns_mm.shape[1:]) != (DMD_HEIGHT, DMD_WIDTH):
        raise ValueError("DMD pattern array has unexpected shape {}".format(patterns_mm.shape))
    if probes_mm.shape[0] != patterns_mm.shape[0]:
        raise ValueError("probe/pattern sample counts differ")
    sample_count = min(int(args.sample_count), int(probes_mm.shape[0]))

    metadata = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
    amp_values = np.asarray(
        metadata.get("amplitude_levels", amplitude_levels(8, 0.2)),
        dtype=np.float32,
    )
    phase_count = int(metadata.get("phase_level_count", 16))
    phase_grid = phase_values(phase_count)

    references_32 = [np.asarray(probes_mm[i], dtype=np.complex64) for i in range(sample_count)]
    patterns_32 = [np.asarray(patterns_mm[i], dtype=np.uint8) for i in range(sample_count)]

    print("LUT + Fourier aperture simulation")
    print("  32x24 dataset: {}".format(pattern_dir))
    print("  samples/profile: {}".format(sample_count))
    print("  radii x-bins: {}".format(radii))
    print("  physical-frequency circle: ry = rx * {}/{} = 0.75 rx".format(DMD_HEIGHT, DMD_WIDTH))
    geometry = first_order_geometry((DMD_HEIGHT, DMD_WIDTH), order_sign=args.order_sign)
    print("  selected first order y,x: {}".format(geometry["order_center_yx"]))
    print("  carrier fx,fy: {:.5f}, {:.5f} cycles/mirror".format(CARRIER_FX, CARRIER_FY))

    lut_rows = lut_quantization_report(amp_values, phase_grid)
    lut_phase = np.asarray([row["phase_abs_error_deg"] for row in lut_rows])
    lut_amp_rel = np.asarray([row["amplitude_rel_error"] for row in lut_rows])
    lut_complex = np.asarray([row["complex_abs_error"] for row in lut_rows])
    print("\nideal 4x4 LUT point error ({} amplitude x {} phase states)".format(len(amp_values), len(phase_grid)))
    print("  phase abs error deg mean/max: {:.4f} / {:.4f}".format(float(np.mean(lut_phase)), float(np.max(lut_phase))))
    print("  amplitude relative error mean/max: {:.3%} / {:.3%}".format(float(np.mean(lut_amp_rel)), float(np.max(lut_amp_rel))))
    print("  complex abs error mean/max: {:.5f} / {:.5f}".format(float(np.mean(lut_complex)), float(np.max(lut_complex))))

    rows_32, power_32 = evaluate_profile(
        patterns_32,
        references_32,
        radii,
        recover_32x24,
        "32x24 repeated-4x4",
        args.order_sign,
    )
    rows = list(rows_32)
    powers = {"32x24": power_32}

    if not args.no_compare_160:
        references_160, patterns_160 = generate_matched_160_fields(
            sample_count,
            amp_values,
            phase_grid,
            seed=args.seed + 160120,
        )
        rows_160, power_160 = evaluate_profile(
            patterns_160,
            references_160,
            radii,
            recover_160x120,
            "160x120 nearest-fill",
            args.order_sign,
        )
        rows.extend(rows_160)
        powers["160x120"] = power_160

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with (output_dir / "aperture_sweep.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "lut_point_errors.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(lut_rows[0].keys()))
        writer.writeheader()
        writer.writerows(lut_rows)

    save_metric_figure(
        output_dir / "fidelity_vs_radius.png",
        rows,
        "fidelity_mean",
        "mean complex-field fidelity",
        "First-order recovery vs Fourier aperture",
    )
    save_metric_figure(
        output_dir / "phase_rmse_vs_radius.png",
        rows,
        "phase_rmse_deg_mean",
        "mean phase RMSE (deg)",
        "Phase error vs Fourier aperture",
    )
    save_metric_figure(
        output_dir / "amplitude_nrmse_vs_radius.png",
        rows,
        "amplitude_nrmse_mean",
        "mean amplitude NRMSE",
        "Amplitude error vs Fourier aperture",
    )

    save_spectrum_figure(
        output_dir / "spectrum_32x24.png",
        power_32,
        radii,
        args.order_sign,
        "32x24 repeated-4x4 average Fourier power",
    )
    if "160x120" in powers:
        save_spectrum_figure(
            output_dir / "spectrum_160x120.png",
            powers["160x120"],
            radii,
            args.order_sign,
            "160x120 nearest-fill average Fourier power",
        )

    peak_report = {}
    order_y, order_x = geometry["order_center_yx"]
    for name, power in powers.items():
        peak_rows = []
        for y, x, value in _top_spectrum_peaks(power):
            dx = x - order_x
            dy_xbins = (y - order_y) * DMD_WIDTH / DMD_HEIGHT
            distance_xbins = float(math.hypot(dx, dy_xbins))
            peak_rows.append(
                {
                    "y": y,
                    "x": x,
                    "relative_power": float(value / max(float(np.max(power)), 1e-30)),
                    "distance_from_selected_order_xbins": distance_xbins,
                }
            )
        peak_report[name] = peak_rows

    summary = {
        "pattern_directory": str(pattern_dir),
        "sample_count": sample_count,
        "radii_x_bins": radii,
        "radius_y_over_x": DMD_HEIGHT / DMD_WIDTH,
        "carrier_cycles_per_mirror": {"fx": CARRIER_FX, "fy": CARRIER_FY},
        "selected_order_center_yx": list(geometry["order_center_yx"]),
        "lut_point_error": {
            "phase_abs_error_deg_mean": float(np.mean(lut_phase)),
            "phase_abs_error_deg_max": float(np.max(lut_phase)),
            "amplitude_rel_error_mean": float(np.mean(lut_amp_rel)),
            "amplitude_rel_error_max": float(np.max(lut_amp_rel)),
            "complex_abs_error_mean": float(np.mean(lut_complex)),
            "complex_abs_error_max": float(np.max(lut_complex)),
        },
        "coarse_spectrum_peaks": peak_report,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    print("\naperture sweep")
    for profile in dict.fromkeys(row["profile"] for row in rows):
        subset = [row for row in rows if row["profile"] == profile]
        best = max(subset, key=lambda row: row["fidelity_mean"])
        print(
            "  {}: best r={} x-bins (ry={:.1f}), fidelity={:.5f}, phase RMSE={:.2f} deg, amp NRMSE={:.4f}".format(
                profile,
                int(best["radius_x_bins"]),
                best["radius_y_bins"],
                best["fidelity_mean"],
                best["phase_rmse_deg_mean"],
                best["amplitude_nrmse_mean"],
            )
        )
        print("    radius : fidelity | phase_rmse_deg | amplitude_nrmse")
        for row in subset:
            print(
                "    {:>6.0f} : {:.5f} | {:>8.2f} | {:.4f}".format(
                    row["radius_x_bins"],
                    row["fidelity_mean"],
                    row["phase_rmse_deg_mean"],
                    row["amplitude_nrmse_mean"],
                )
            )

    print("\noutputs: {}".format(output_dir))
    print("  aperture_sweep.csv")
    print("  lut_point_errors.csv")
    print("  fidelity_vs_radius.png")
    print("  phase_rmse_vs_radius.png")
    print("  amplitude_nrmse_vs_radius.png")
    print("  spectrum_32x24.png")
    if "160x120" in powers:
        print("  spectrum_160x120.png")
    print("  summary.json")


if __name__ == "__main__":
    main()
