"""
Direct sequential phase optimization for TMCalib (32x24 profile).

Purpose
-------
Measure the *real optical-system focusing ceiling* without using a recovered TM.
For a fixed camera target pixel, scan each logical DMD input channel through
0, pi/2, pi, 3pi/2, keep the phase that maximizes target intensity, then capture
the final focused image and report PBR.

This script intentionally reuses TMCalib's existing:
- CameraHandler
- DMDController
- JUOPT DMD loading/projection path
- holo_SP + generate_lut optical encoding
- pixel-wise focus analysis / PBR definition

Run from the TMCalib repository root:
    python direct_sequential_focus_32x24.py --target-x 64 --target-y 64

Quick hardware sanity check:
    python direct_sequential_focus_32x24.py --target-x 64 --target-y 64 --max-channels 16
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import calibrate_v4_32x24 as core
from holograms.generate_LUT import generate_lut
from holograms.dmd_holograms import holo_SP


PHASE_CANDIDATES = np.array(
    [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi],
    dtype=np.float32,
)


def encode_input_field(input_field, lut_cache, ds_method="mean"):
    """Match TMCalib's existing 32x24 -> 1024x768 -> holo_SP path."""
    input_field = np.asarray(input_field, dtype=np.complex64)
    if input_field.shape != (24, 32):
        raise ValueError(f"Expected input field shape (24, 32), got {input_field.shape}")

    # Existing TMCalib 32x24 mapping: each logical channel -> 32x32 DMD region.
    full_pattern = np.kron(
        input_field,
        np.ones((32, 32), dtype=np.float32),
    )

    _, px_comb, lut = lut_cache
    hologram = holo_SP(
        full_pattern,
        lut,
        px_comb,
        ds_method=ds_method,
    )

    # Keep normalization identical to the current 32x24 focus code.
    holo_norm = hologram.astype(np.float32)
    hmin = float(np.min(holo_norm))
    hmax = float(np.max(holo_norm))
    if hmax > hmin:
        holo_norm = 255.0 * (holo_norm - hmin) / (hmax - hmin)

    return holo_norm.astype(np.uint8)


def unwrap_camera_result(result):
    """
    CameraHandler.run() is used in slightly different ways in TMCalib branches.
    Accept either ndarray directly or tuple(image, ...).
    """
    if isinstance(result, tuple):
        if not result:
            return None
        result = result[0]
    if result is None:
        return None
    return np.asarray(result)


def capture_one(controller, camera, hologram, settle_time_s=0.02):
    """
    Load one hologram, project it, and acquire exactly one triggered camera frame.

    This follows the same ordering used by TMCalib focus routines:
        load_pattern -> camera.start -> juoptProjection -> camera.run
        -> camera.stop -> juoptStop -> clear_sequence
    """
    pattern_batch = np.asarray([hologram], dtype=np.uint8)

    if not controller.load_pattern(pattern_batch):
        raise RuntimeError("Failed to load hologram onto DMD")

    camera_started = False
    projection_started = False
    try:
        camera.start()
        camera_started = True

        ret = controller.DMD.juoptProjection(controller.dev_id, 0, 0)
        if ret not in (None, 0):
            raise RuntimeError(f"juoptProjection failed with code {ret}")
        projection_started = True

        if settle_time_s > 0:
            time.sleep(settle_time_s)

        image = unwrap_camera_result(camera.run())
        if image is None:
            raise RuntimeError("Camera did not return an image")

        if image.ndim == 3:
            image = np.mean(image, axis=2)

        return image.astype(np.float32, copy=False)

    finally:
        if projection_started:
            try:
                controller.DMD.juoptStop(controller.dev_id)
            except Exception:
                pass
        if camera_started:
            try:
                camera.stop()
            except Exception:
                pass
        try:
            controller.clear_sequence(0)
        except Exception:
            pass


def target_metric(image, target_x, target_y, radius=0):
    """
    Optimization metric.

    radius=0:
        use exactly the target pixel.
    radius=1:
        use mean intensity in a 3x3 box, which can be more robust to subpixel
        drift / slight speckle motion.
    """
    h, w = image.shape
    if not (0 <= target_x < w and 0 <= target_y < h):
        raise ValueError(
            f"Target ({target_x}, {target_y}) outside camera image {w}x{h}"
        )

    r = int(radius)
    x0 = max(0, target_x - r)
    x1 = min(w, target_x + r + 1)
    y0 = max(0, target_y - r)
    y1 = min(h, target_y + r + 1)
    return float(np.mean(image[y0:y1, x0:x1]))


def initialize_hardware(save_path):
    camera = core.CameraHandler(cam_index=0, save_path=str(save_path))
    camera.convert_to_12bit = False

    controller = core.DMDController(camera)
    devices = controller.get_devices()
    if not devices:
        camera.cleanup()
        raise RuntimeError("No DMD device found")

    print(f"DMD devices: {devices}")
    if not controller.initialize_device(devices[0]):
        camera.cleanup()
        raise RuntimeError(f"Failed to initialize DMD device: {devices[0]}")

    return camera, controller


def run_sequential_optimization(
    controller,
    camera,
    target_x,
    target_y,
    metric_radius=0,
    settle_time_s=0.02,
    max_channels=None,
    checkpoint_every=32,
    output_dir=Path("direct_sequential_focus_result"),
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate LUT only once. This is expensive and should never be repeated
    # inside the 4-phase scan loop.
    print("Generating holo_SP LUT once...")
    lut_cache = generate_lut("sp", 4)

    ny, nx = 24, 32
    total_channels = nx * ny
    channels_to_run = total_channels
    if max_channels is not None:
        channels_to_run = min(total_channels, int(max_channels))

    # All channels start active with phase 0.
    phases = np.zeros((ny, nx), dtype=np.float32)
    field = np.exp(1j * phases).astype(np.complex64)

    best_metric_history = np.full(channels_to_run, np.nan, dtype=np.float32)
    selected_phase_history = np.full(channels_to_run, np.nan, dtype=np.float32)
    phase_scan_history = np.full(
        (channels_to_run, len(PHASE_CANDIDATES)),
        np.nan,
        dtype=np.float32,
    )

    # Baseline before optimization.
    baseline_holo = encode_input_field(field, lut_cache)
    baseline_image = capture_one(
        controller,
        camera,
        baseline_holo,
        settle_time_s=settle_time_s,
    )
    baseline_metric = target_metric(
        baseline_image,
        target_x,
        target_y,
        radius=metric_radius,
    )
    print(
        f"Baseline target metric @ ({target_x}, {target_y}) = "
        f"{baseline_metric:.3f}"
    )

    t0 = time.time()

    for channel_index in range(channels_to_run):
        y = channel_index // nx
        x = channel_index % nx

        candidate_metrics = []

        # Previous channels keep their already-optimized phases.
        # Only the current logical channel is scanned.
        for phi in PHASE_CANDIDATES:
            phases[y, x] = float(phi)
            field[y, x] = np.exp(1j * float(phi))

            hologram = encode_input_field(field, lut_cache)
            image = capture_one(
                controller,
                camera,
                hologram,
                settle_time_s=settle_time_s,
            )
            metric = target_metric(
                image,
                target_x,
                target_y,
                radius=metric_radius,
            )
            candidate_metrics.append(metric)

        candidate_metrics = np.asarray(candidate_metrics, dtype=np.float32)
        best_idx = int(np.argmax(candidate_metrics))
        best_phi = float(PHASE_CANDIDATES[best_idx])

        # Permanently keep the best phase for this channel.
        phases[y, x] = best_phi
        field[y, x] = np.exp(1j * best_phi)

        phase_scan_history[channel_index] = candidate_metrics
        selected_phase_history[channel_index] = best_phi
        best_metric_history[channel_index] = candidate_metrics[best_idx]

        elapsed = time.time() - t0
        print(
            f"[{channel_index + 1:4d}/{channels_to_run}] "
            f"channel=({x:02d},{y:02d}) "
            f"I={candidate_metrics.tolist()} "
            f"best={best_phi / np.pi:.2f}π "
            f"metric={candidate_metrics[best_idx]:.2f} "
            f"gain={candidate_metrics[best_idx] / max(baseline_metric, 1e-12):.2f}x "
            f"elapsed={elapsed:.1f}s"
        )

        if checkpoint_every > 0 and (
            (channel_index + 1) % checkpoint_every == 0
            or channel_index + 1 == channels_to_run
        ):
            np.save(output_dir / "optimized_phases.npy", phases)
            np.save(output_dir / "best_metric_history.npy", best_metric_history)
            np.save(output_dir / "selected_phase_history.npy", selected_phase_history)
            np.save(output_dir / "phase_scan_history.npy", phase_scan_history)

    # Final capture with the fully optimized field.
    final_holo = encode_input_field(field, lut_cache)
    final_image = capture_one(
        controller,
        camera,
        final_holo,
        settle_time_s=settle_time_s,
    )

    final_metric = target_metric(
        final_image,
        target_x,
        target_y,
        radius=metric_radius,
    )

    # Reuse TMCalib's own PBR/background definition so this number is directly
    # comparable with your TM-conjugate-focus report.
    analysis = controller._analyze_pixelwise_focus_image(
        final_image,
        target_x,
        target_y,
    )

    np.save(output_dir / "baseline_image.npy", baseline_image)
    np.save(output_dir / "final_focused_image.npy", final_image)
    np.save(output_dir / "final_hologram.npy", final_holo)
    np.save(output_dir / "optimized_phases.npy", phases)

    summary = {
        "profile": "TMCalib calibrate_v4_32x24",
        "logical_input_shape": [24, 32],
        "logical_channel_count": total_channels,
        "optimized_channel_count": channels_to_run,
        "target_x": int(target_x),
        "target_y": int(target_y),
        "metric_radius": int(metric_radius),
        "phase_candidates_rad": [float(v) for v in PHASE_CANDIDATES],
        "baseline_target_metric": float(baseline_metric),
        "final_target_metric": float(final_metric),
        "target_metric_gain": float(
            final_metric / max(baseline_metric, 1e-12)
        ),
        "ideal_phase_only_pbr_piN_over_4": float(
            np.pi / 4.0 * total_channels
        ),
        "elapsed_s": float(time.time() - t0),
    }

    # Keep only JSON-friendly scalar analysis entries.
    for key, value in analysis.items():
        if isinstance(value, (bool, str, int, float, np.integer, np.floating)):
            summary[f"tmcalib_{key}"] = (
                value.item() if isinstance(value, (np.integer, np.floating)) else value
            )

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Simple result figures.
    plt.figure(figsize=(6, 5))
    plt.imshow(final_image)
    plt.scatter([target_x], [target_y], marker="+", s=100)
    plt.title(
        f"Direct sequential focus\n"
        f"PBR={analysis.get('pbr', float('nan')):.2f}"
    )
    plt.colorbar(label="Intensity")
    plt.tight_layout()
    plt.savefig(output_dir / "final_focused_image.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, channels_to_run + 1), best_metric_history)
    plt.xlabel("Optimized channel count")
    plt.ylabel("Best target metric")
    plt.title("Sequential optimization convergence")
    plt.tight_layout()
    plt.savefig(output_dir / "optimization_curve.png", dpi=180)
    plt.close()

    print("\n" + "=" * 72)
    print("DIRECT SEQUENTIAL OPTIMIZATION RESULT")
    print("=" * 72)
    print(f"Target                    : ({target_x}, {target_y})")
    print(f"Channels optimized        : {channels_to_run}/{total_channels}")
    print(f"Baseline target metric    : {baseline_metric:.3f}")
    print(f"Final target metric       : {final_metric:.3f}")
    print(
        "Target intensity gain     : "
        f"{final_metric / max(baseline_metric, 1e-12):.2f}x"
    )
    print(f"Final PBR (TMCalib method): {analysis.get('pbr', float('nan')):.2f}")
    print(
        "Ideal pi*N/4 PBR          : "
        f"{np.pi / 4.0 * total_channels:.2f}"
    )
    print(f"Output directory          : {output_dir}")
    print("=" * 72)

    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-x", type=int, default=64)
    parser.add_argument("--target-y", type=int, default=64)
    parser.add_argument(
        "--metric-radius",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="0=target pixel; 1=3x3 mean; 2=5x5 mean",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=0.02,
        help="Delay after starting DMD projection before camera.run().",
    )
    parser.add_argument(
        "--max-channels",
        type=int,
        default=None,
        help="Only optimize the first N channels; useful for a hardware smoke test.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("direct_sequential_focus_result"),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    camera = None
    controller = None
    try:
        camera, controller = initialize_hardware(args.output_dir / "camera")
        run_sequential_optimization(
            controller=controller,
            camera=camera,
            target_x=args.target_x,
            target_y=args.target_y,
            metric_radius=args.metric_radius,
            settle_time_s=args.settle_time,
            max_channels=args.max_channels,
            checkpoint_every=args.checkpoint_every,
            output_dir=args.output_dir,
        )
    finally:
        if controller is not None:
            try:
                controller.cleanup()
            except Exception as exc:
                print(f"DMD cleanup warning: {exc}")
        if camera is not None:
            try:
                camera.cleanup()
            except Exception as exc:
                print(f"Camera cleanup warning: {exc}")


if __name__ == "__main__":
    main()
