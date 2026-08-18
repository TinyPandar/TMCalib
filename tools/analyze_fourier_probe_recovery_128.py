"""Recover a logical 128 x 128 probe from an aligned px=4 DMD pattern.

The superpixel carrier used by ``holo_SP`` is (fx, fy) = (1/4, 1/16)
cycles per DMD mirror. With NumPy's FFT sign convention, the order that
directly reconstructs the original complex probe is therefore centred at
(-N/4, -N/16) relative to the shifted spectrum origin. The opposite order
reconstructs the conjugated probe.
"""

import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from dmd_pattern_128 import (
    ACTIVE_HEIGHT,
    ACTIVE_WIDTH,
    ACTIVE_X,
    ACTIVE_Y,
    HOLOGRAM_SUPERPIXEL_SIZE,
    INPUT_HEIGHT,
    INPUT_WIDTH,
)


DEFAULT_DATASET = "pregenerated_patterns_128_px4_active512_full"
DEFAULT_OUTPUT_DIR = "fourier_probe_recovery_128_px4_active512"
DEFAULT_RADIUS = 96


def complex_fidelity(reference, recovered):
    reference = np.asarray(reference).ravel()
    recovered = np.asarray(recovered).ravel()
    denominator = np.linalg.norm(reference) * np.linalg.norm(recovered)
    if denominator <= 0:
        return 0.0
    return float(abs(np.vdot(reference, recovered)) / denominator)


def align_complex_gain(reference, recovered):
    """Remove the irrelevant global complex gain from ``recovered``."""
    reference = np.asarray(reference)
    recovered = np.asarray(recovered)
    denominator = np.vdot(recovered.ravel(), recovered.ravel())
    if abs(denominator) <= 0:
        return recovered.copy(), 0.0j
    gain = np.vdot(recovered.ravel(), reference.ravel()) / denominator
    return gain * recovered, complex(gain)


def recover_first_order(active_pattern, radius=DEFAULT_RADIUS, order_sign=-1):
    """FFT, circularly filter one first order, recenter, and decode to 128²."""
    active_pattern = np.asarray(active_pattern, dtype=np.float32)
    if active_pattern.shape != (ACTIVE_HEIGHT, ACTIVE_WIDTH):
        raise ValueError(
            "Active pattern shape {} does not match {}".format(
                active_pattern.shape, (ACTIVE_HEIGHT, ACTIVE_WIDTH)
            )
        )
    if ACTIVE_HEIGHT != ACTIVE_WIDTH:
        raise ValueError("The Fourier recovery expects a square active region")
    px = HOLOGRAM_SUPERPIXEL_SIZE
    size = ACTIVE_WIDTH
    centre_y = ACTIVE_HEIGHT // 2
    centre_x = ACTIVE_WIDTH // 2
    offset_x = int(order_sign * size // px)
    offset_y = int(order_sign * size // (px * px))
    order_y = centre_y + offset_y
    order_x = centre_x + offset_x

    spectrum = np.fft.fftshift(np.fft.fft2(active_pattern))
    yy, xx = np.ogrid[:ACTIVE_HEIGHT, :ACTIVE_WIDTH]
    mask = (yy - order_y) ** 2 + (xx - order_x) ** 2 <= int(radius) ** 2
    filtered = spectrum * mask
    recentered = np.roll(
        filtered,
        shift=(-offset_y, -offset_x),
        axis=(0, 1),
    )
    recovered_dmd_plane = np.fft.ifft2(np.fft.ifftshift(recentered))
    recovered_probe = recovered_dmd_plane.reshape(
        INPUT_HEIGHT,
        px,
        INPUT_WIDTH,
        px,
    ).mean(axis=(1, 3))
    return {
        "recovered_probe": recovered_probe,
        "recovered_dmd_plane": recovered_dmd_plane,
        "spectrum": spectrum,
        "filtered_spectrum": filtered,
        "mask": mask,
        "order_center_yx": (order_y, order_x),
        "order_offset_xy": (offset_x, offset_y),
    }


def decode_superpixels_directly(active_pattern):
    """Direct block demodulation used only as an encoding sanity check."""
    binary = np.asarray(active_pattern, dtype=np.float32)
    px = HOLOGRAM_SUPERPIXEL_SIZE
    blocks = binary.reshape(
        INPUT_HEIGHT, px, INPUT_WIDTH, px
    ).transpose(0, 2, 1, 3)
    logical_y = np.arange(INPUT_HEIGHT)[:, None, None, None]
    local_y = np.arange(px)[None, None, :, None]
    local_x = np.arange(px)[None, None, None, :]
    phase = np.exp(
        1j
        * 2.0
        * np.pi
        * (px * local_x + local_y + px * logical_y)
        / (px * px)
    )
    return np.sum(blocks * phase, axis=(2, 3))


def recovery_metrics(reference, recovered):
    aligned, gain = align_complex_gain(reference, recovered)
    fidelity = complex_fidelity(reference, recovered)
    nmse = float(
        np.linalg.norm(reference - aligned) ** 2
        / np.linalg.norm(reference) ** 2
    )
    phase_error = np.angle(aligned * np.conj(reference))
    return {
        "fidelity": fidelity,
        "nmse": nmse,
        "phase_rmse_deg": float(
            np.sqrt(np.mean(phase_error**2)) * 180.0 / np.pi
        ),
        "phase_mae_deg": float(
            np.mean(np.abs(phase_error)) * 180.0 / np.pi
        ),
        "amplitude_mean": float(np.mean(np.abs(aligned))),
        "amplitude_std": float(np.std(np.abs(aligned))),
        "complex_gain_real": float(np.real(gain)),
        "complex_gain_imag": float(np.imag(gain)),
        "aligned": aligned,
        "phase_error": phase_error,
    }


def save_figure(path, active_pattern, reference, recovery, metrics, radius):
    spectrum_log = np.log1p(np.abs(recovery["spectrum"]))
    isolated_log = np.where(
        recovery["mask"], spectrum_log, np.nan
    )
    aligned = metrics["aligned"]
    phase_error_deg = metrics["phase_error"] * 180.0 / np.pi
    phase_limit = max(10.0, float(np.percentile(np.abs(phase_error_deg), 99)))
    order_y, order_x = recovery["order_center_yx"]

    figure, axes = plt.subplots(2, 4, figsize=(15.5, 7.8), constrained_layout=True)
    figure.suptitle(
        "First-order Fourier recovery | fidelity={:.4f}, NMSE={:.4f}, "
        "phase RMSE={:.2f} deg".format(
            metrics["fidelity"], metrics["nmse"], metrics["phase_rmse_deg"]
        )
    )

    axes[0, 0].imshow(active_pattern, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("512x512 binary hologram")

    spectrum_image = axes[0, 1].imshow(
        spectrum_log,
        cmap="magma",
        vmin=float(np.percentile(spectrum_log, 5)),
        vmax=float(np.percentile(spectrum_log, 99.9)),
    )
    axes[0, 1].add_patch(
        Circle((order_x, order_y), radius, fill=False, color="cyan", linewidth=1.5)
    )
    axes[0, 1].plot([ACTIVE_WIDTH // 2], [ACTIVE_HEIGHT // 2], "w+", markersize=7)
    axes[0, 1].set_title("FFT magnitude and selected order")
    figure.colorbar(spectrum_image, ax=axes[0, 1], fraction=0.046)

    isolated_image = axes[0, 2].imshow(isolated_log, cmap="magma")
    axes[0, 2].set_title("Circular first-order aperture")
    figure.colorbar(isolated_image, ax=axes[0, 2], fraction=0.046)

    amplitude_image = axes[0, 3].imshow(
        np.abs(aligned),
        cmap="viridis",
        vmin=0,
        vmax=float(np.percentile(np.abs(aligned), 99.5)),
    )
    axes[0, 3].set_title("Recovered amplitude")
    figure.colorbar(amplitude_image, ax=axes[0, 3], fraction=0.046)

    phase_reference = axes[1, 0].imshow(
        np.angle(reference), cmap="twilight", vmin=-np.pi, vmax=np.pi
    )
    axes[1, 0].set_title("Original probe phase")
    phase_recovered = axes[1, 1].imshow(
        np.angle(aligned), cmap="twilight", vmin=-np.pi, vmax=np.pi
    )
    axes[1, 1].set_title("Recovered probe phase")
    figure.colorbar(
        phase_recovered,
        ax=[axes[1, 0], axes[1, 1]],
        fraction=0.025,
        ticks=[-np.pi, 0, np.pi],
    )

    error_image = axes[1, 2].imshow(
        phase_error_deg,
        cmap="coolwarm",
        vmin=-phase_limit,
        vmax=phase_limit,
    )
    axes[1, 2].set_title("Phase error (deg)")
    figure.colorbar(error_image, ax=axes[1, 2], fraction=0.046)

    stride = 2
    reference_points = reference[::stride, ::stride].ravel()
    recovered_points = aligned[::stride, ::stride].ravel()
    theta = np.linspace(0, 2 * np.pi, 512)
    axes[1, 3].plot(np.cos(theta), np.sin(theta), color="0.65", linewidth=1)
    axes[1, 3].scatter(
        np.real(recovered_points),
        np.imag(recovered_points),
        s=4,
        alpha=0.18,
        label="Recovered",
    )
    axes[1, 3].scatter(
        np.real(reference_points),
        np.imag(reference_points),
        s=10,
        marker="x",
        linewidths=0.5,
        label="Original",
    )
    axes[1, 3].set_aspect("equal")
    axes[1, 3].set_xlim(-1.35, 1.35)
    axes[1, 3].set_ylim(-1.35, 1.35)
    axes[1, 3].set_xlabel("Real")
    axes[1, 3].set_ylabel("Imaginary")
    axes[1, 3].set_title("Complex samples")
    axes[1, 3].legend(loc="upper right", fontsize=8)

    for axis in axes.flat[:7]:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--radius", type=int, default=DEFAULT_RADIUS)
    parser.add_argument("--validation-count", type=int, default=16)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    dataset = os.path.abspath(args.dataset)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    probes = np.load(os.path.join(dataset, "probe.npy"), mmap_mode="r")
    patterns = np.load(
        os.path.join(dataset, "patterns_pregenerated.npy"), mmap_mode="r"
    )
    if not 0 <= args.index < probes.shape[0]:
        raise IndexError("Probe index is outside the dataset")

    reference = np.array(probes[args.index], dtype=np.complex64, copy=True)
    full_pattern = np.array(patterns[args.index], dtype=np.float32, copy=True)
    active_pattern = full_pattern[
        ACTIVE_Y : ACTIVE_Y + ACTIVE_HEIGHT,
        ACTIVE_X : ACTIVE_X + ACTIVE_WIDTH,
    ] / 255.0
    recovery = recover_first_order(active_pattern, radius=args.radius, order_sign=-1)
    metrics = recovery_metrics(reference, recovery["recovered_probe"])
    direct = decode_superpixels_directly(active_pattern)
    direct_fidelity = complex_fidelity(reference, direct)

    validation_count = max(1, min(int(args.validation_count), probes.shape[0]))
    validation_indices = np.unique(
        np.linspace(0, probes.shape[0] - 1, validation_count, dtype=int)
    )
    validation_fidelities = []
    for index in validation_indices:
        probe = np.array(probes[index], dtype=np.complex64, copy=True)
        pattern = np.array(
            patterns[
                index,
                ACTIVE_Y : ACTIVE_Y + ACTIVE_HEIGHT,
                ACTIVE_X : ACTIVE_X + ACTIVE_WIDTH,
            ],
            dtype=np.float32,
            copy=True,
        ) / 255.0
        recovered = recover_first_order(
            pattern, radius=args.radius, order_sign=-1
        )["recovered_probe"]
        validation_fidelities.append(complex_fidelity(probe, recovered))

    stem = "fourier_probe_recovery_index{:05d}_r{}".format(
        args.index, args.radius
    )
    figure_path = os.path.join(output_dir, stem + ".png")
    arrays_path = os.path.join(output_dir, stem + ".npz")
    summary_path = os.path.join(output_dir, stem + ".json")
    save_figure(
        figure_path,
        active_pattern,
        reference,
        recovery,
        metrics,
        args.radius,
    )
    np.savez_compressed(
        arrays_path,
        original_probe=reference.astype(np.complex64),
        recovered_probe=recovery["recovered_probe"].astype(np.complex64),
        aligned_recovered_probe=metrics["aligned"].astype(np.complex64),
        phase_error_rad=metrics["phase_error"].astype(np.float32),
    )

    summary = {
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset": dataset,
        "probe_index": int(args.index),
        "active_shape": [ACTIVE_HEIGHT, ACTIVE_WIDTH],
        "logical_shape": [INPUT_HEIGHT, INPUT_WIDTH],
        "superpixel_size": HOLOGRAM_SUPERPIXEL_SIZE,
        "fft_order_offset_xy": list(recovery["order_offset_xy"]),
        "fft_order_center_yx": list(recovery["order_center_yx"]),
        "circular_aperture_radius_px": int(args.radius),
        "fidelity": metrics["fidelity"],
        "nmse": metrics["nmse"],
        "phase_rmse_deg": metrics["phase_rmse_deg"],
        "phase_mae_deg": metrics["phase_mae_deg"],
        "amplitude_mean_after_gain_alignment": metrics["amplitude_mean"],
        "amplitude_std_after_gain_alignment": metrics["amplitude_std"],
        "direct_superpixel_decode_fidelity": direct_fidelity,
        "validation_indices": validation_indices.tolist(),
        "validation_fidelity_mean": float(np.mean(validation_fidelities)),
        "validation_fidelity_min": float(np.min(validation_fidelities)),
        "validation_fidelity_max": float(np.max(validation_fidelities)),
        "note": (
            "With NumPy FFT conventions the (-128,-32) order reconstructs "
            "the probe directly; the opposite order reconstructs its complex "
            "conjugate. Physical camera axes may reverse this sign."
        ),
        "files": {
            "figure": os.path.abspath(figure_path),
            "arrays": os.path.abspath(arrays_path),
            "summary": os.path.abspath(summary_path),
        },
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
