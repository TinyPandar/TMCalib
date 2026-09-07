"""
Unattended direct sequential phase optimization for TMCalib (32x24 profile).

Main additions over the basic version:
1. Automatic exposure reduction when the camera approaches saturation.
2. If exposure changes during a channel, all four phases for that channel are
   re-measured at the new exposure, so phase selection is never biased.
3. Exposure-normalized convergence history.
4. Capture retries and frequent checkpoints.
5. Safe abort if the camera is still saturated at the minimum exposure.

Run:
    python .\direct_sequential_focus_32x24_auto.py --target-x 64 --target-y 64

Recommended unattended full run:
    python .\direct_sequential_focus_32x24_auto.py `
        --target-x 64 --target-y 64 `
        --auto-exposure `
        --target-peak 150 `
        --saturation-threshold 235 `
        --checkpoint-every 16
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
    """Use the same 32x24 -> full DMD -> holo_SP path as TMCalib."""
    input_field = np.asarray(input_field, dtype=np.complex64)
    if input_field.shape != (24, 32):
        raise ValueError(f"Expected input field shape (24, 32), got {input_field.shape}")

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

    holo_norm = hologram.astype(np.float32)
    hmin = float(np.min(holo_norm))
    hmax = float(np.max(holo_norm))
    if hmax > hmin:
        holo_norm = 255.0 * (holo_norm - hmin) / (hmax - hmin)
    return holo_norm.astype(np.uint8)


def unwrap_camera_result(result):
    if isinstance(result, tuple):
        if not result:
            return None
        result = result[0]
    if result is None:
        return None
    return np.asarray(result)


def get_exposure_us(camera):
    """Read actual camera exposure when possible."""
    try:
        return float(camera.cam.ExposureTime.GetValue())
    except Exception:
        return None


def set_exposure_us(camera, exposure_us):
    exposure_us = float(exposure_us)
    ok = camera.configure_exposure(exposure_time=exposure_us)
    if not ok:
        raise RuntimeError(f"Failed to set exposure to {exposure_us:.1f} us")
    actual = get_exposure_us(camera)
    return exposure_us if actual is None else actual


def capture_one(
    controller,
    camera,
    hologram,
    settle_time_s=0.02,
    retries=3,
):
    """Capture one frame with transient-error retries."""
    last_error = None

    for attempt in range(1, retries + 1):
        pattern_batch = np.asarray([hologram], dtype=np.uint8)
        camera_started = False
        projection_started = False

        try:
            if not controller.load_pattern(pattern_batch):
                raise RuntimeError("Failed to load hologram onto DMD")

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

        except Exception as exc:
            last_error = exc
            print(f"[capture retry {attempt}/{retries}] {exc}")
            time.sleep(0.05)

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

    raise RuntimeError(f"Capture failed after {retries} retries: {last_error}")


def capture_average(
    controller,
    camera,
    hologram,
    settle_time_s,
    frames_per_phase,
    retries,
):
    frames = [
        capture_one(
            controller,
            camera,
            hologram,
            settle_time_s=settle_time_s,
            retries=retries,
        )
        for _ in range(max(1, int(frames_per_phase)))
    ]
    return np.mean(np.stack(frames, axis=0), axis=0)


def target_metric(image, target_x, target_y, radius=0):
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


def choose_lower_exposure(
    current_exposure_us,
    observed_peak,
    target_peak,
    min_exposure_us,
):
    """
    Scale exposure so the observed max would land near target_peak.
    Only decreases exposure; adds a 10% safety margin.
    """
    if observed_peak <= 0:
        return current_exposure_us

    proposed = current_exposure_us * (target_peak / observed_peak) * 0.90
    proposed = min(proposed, current_exposure_us * 0.85)
    proposed = max(float(min_exposure_us), proposed)
    return proposed


def save_checkpoint(
    output_dir,
    phases,
    best_metric_history,
    best_metric_normalized_history,
    selected_phase_history,
    phase_scan_history,
    exposure_history,
    completed_channels,
):
    np.save(output_dir / "optimized_phases.npy", phases)
    np.save(output_dir / "best_metric_history.npy", best_metric_history)
    np.save(
        output_dir / "best_metric_normalized_history.npy",
        best_metric_normalized_history,
    )
    np.save(output_dir / "selected_phase_history.npy", selected_phase_history)
    np.save(output_dir / "phase_scan_history.npy", phase_scan_history)
    np.save(output_dir / "exposure_history_us.npy", exposure_history)

    with open(output_dir / "checkpoint.json", "w", encoding="utf-8") as f:
        json.dump(
            {"completed_channels": int(completed_channels)},
            f,
            indent=2,
        )


def run_sequential_optimization(
    controller,
    camera,
    target_x,
    target_y,
    metric_radius=0,
    settle_time_s=0.02,
    max_channels=None,
    checkpoint_every=16,
    output_dir=Path("direct_sequential_focus_auto_result"),
    auto_exposure=True,
    initial_exposure_us=None,
    min_exposure_us=50.0,
    target_peak=150.0,
    saturation_threshold=235.0,
    frames_per_phase=1,
    capture_retries=3,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating holo_SP LUT once...")
    lut_cache = generate_lut("sp", 4)

    ny, nx = 24, 32
    total_channels = nx * ny
    channels_to_run = (
        total_channels
        if max_channels is None
        else min(total_channels, int(max_channels))
    )

    # Explicitly establish the starting exposure.
    if initial_exposure_us is None:
        current_exposure_us = get_exposure_us(camera)
        if current_exposure_us is None:
            current_exposure_us = float(getattr(core, "CAMERA_EXPOSURE_US", 1500.0))
    else:
        current_exposure_us = set_exposure_us(camera, initial_exposure_us)

    reference_exposure_us = float(current_exposure_us)

    phases = np.zeros((ny, nx), dtype=np.float32)
    field = np.exp(1j * phases).astype(np.complex64)

    best_metric_history = np.full(channels_to_run, np.nan, dtype=np.float32)
    best_metric_normalized_history = np.full(
        channels_to_run, np.nan, dtype=np.float32
    )
    selected_phase_history = np.full(channels_to_run, np.nan, dtype=np.float32)
    phase_scan_history = np.full(
        (channels_to_run, len(PHASE_CANDIDATES)),
        np.nan,
        dtype=np.float32,
    )
    exposure_history = np.full(channels_to_run, np.nan, dtype=np.float32)

    # -------- baseline with automatic pre-run exposure protection --------
    while True:
        baseline_holo = encode_input_field(field, lut_cache)
        baseline_image = capture_average(
            controller,
            camera,
            baseline_holo,
            settle_time_s,
            frames_per_phase,
            capture_retries,
        )
        baseline_peak = float(np.max(baseline_image))

        if (
            auto_exposure
            and baseline_peak > target_peak
            and current_exposure_us > min_exposure_us + 1e-9
        ):
            new_exp = choose_lower_exposure(
                current_exposure_us,
                baseline_peak,
                target_peak,
                min_exposure_us,
            )
            if new_exp < current_exposure_us - 1e-9:
                print(
                    f"[auto exposure] baseline max={baseline_peak:.1f}; "
                    f"{current_exposure_us:.1f} us -> {new_exp:.1f} us"
                )
                current_exposure_us = set_exposure_us(camera, new_exp)
                reference_exposure_us = float(current_exposure_us)
                continue
        break

    baseline_metric = target_metric(
        baseline_image,
        target_x,
        target_y,
        radius=metric_radius,
    )

    print(
        f"Baseline target={baseline_metric:.3f}, "
        f"image max={baseline_peak:.1f}, "
        f"exposure={current_exposure_us:.1f} us"
    )

    t0 = time.time()

    # ----------------------- sequential optimization -----------------------
    for channel_index in range(channels_to_run):
        y = channel_index // nx
        x = channel_index % nx

        while True:
            candidate_metrics = []
            candidate_peaks = []

            for phi in PHASE_CANDIDATES:
                phases[y, x] = float(phi)
                field[y, x] = np.exp(1j * float(phi))

                hologram = encode_input_field(field, lut_cache)
                image = capture_average(
                    controller,
                    camera,
                    hologram,
                    settle_time_s,
                    frames_per_phase,
                    capture_retries,
                )

                candidate_metrics.append(
                    target_metric(
                        image,
                        target_x,
                        target_y,
                        radius=metric_radius,
                    )
                )
                candidate_peaks.append(float(np.max(image)))

            candidate_metrics = np.asarray(candidate_metrics, dtype=np.float32)
            observed_peak = float(np.max(candidate_peaks))

            # IMPORTANT:
            # If exposure changes, throw away this 4-phase scan and redo all
            # four phases at the same new exposure.
            if (
                auto_exposure
                and observed_peak > target_peak
                and current_exposure_us > min_exposure_us + 1e-9
            ):
                new_exp = choose_lower_exposure(
                    current_exposure_us,
                    observed_peak,
                    target_peak,
                    min_exposure_us,
                )

                if new_exp < current_exposure_us - 1e-9:
                    print(
                        f"[auto exposure] channel {channel_index + 1}: "
                        f"max={observed_peak:.1f}; "
                        f"{current_exposure_us:.1f} us -> {new_exp:.1f} us; "
                        f"redoing all 4 phases"
                    )
                    current_exposure_us = set_exposure_us(camera, new_exp)
                    continue

            if observed_peak >= saturation_threshold:
                if current_exposure_us <= min_exposure_us + 1e-9:
                    save_checkpoint(
                        output_dir,
                        phases,
                        best_metric_history,
                        best_metric_normalized_history,
                        selected_phase_history,
                        phase_scan_history,
                        exposure_history,
                        channel_index,
                    )
                    raise RuntimeError(
                        f"Still near saturation (max={observed_peak:.1f}) at "
                        f"minimum exposure {current_exposure_us:.1f} us. "
                        "Checkpoint saved; reduce optical power before continuing."
                    )

            break

        best_idx = int(np.argmax(candidate_metrics))
        best_phi = float(PHASE_CANDIDATES[best_idx])
        best_metric = float(candidate_metrics[best_idx])

        phases[y, x] = best_phi
        field[y, x] = np.exp(1j * best_phi)

        normalized_metric = (
            best_metric * reference_exposure_us / current_exposure_us
        )

        phase_scan_history[channel_index] = candidate_metrics
        selected_phase_history[channel_index] = best_phi
        best_metric_history[channel_index] = best_metric
        best_metric_normalized_history[channel_index] = normalized_metric
        exposure_history[channel_index] = current_exposure_us

        elapsed = time.time() - t0
        print(
            f"[{channel_index + 1:4d}/{channels_to_run}] "
            f"ch=({x:02d},{y:02d}) "
            f"best={best_phi / np.pi:.2f}pi "
            f"raw={best_metric:.2f} "
            f"norm={normalized_metric:.2f} "
            f"exp={current_exposure_us:.1f}us "
            f"max={observed_peak:.1f} "
            f"elapsed={elapsed:.1f}s"
        )

        if checkpoint_every > 0 and (
            (channel_index + 1) % checkpoint_every == 0
            or channel_index + 1 == channels_to_run
        ):
            save_checkpoint(
                output_dir,
                phases,
                best_metric_history,
                best_metric_normalized_history,
                selected_phase_history,
                phase_scan_history,
                exposure_history,
                channel_index + 1,
            )

    # ---------------------------- final capture ----------------------------
    while True:
        final_holo = encode_input_field(field, lut_cache)
        final_image = capture_average(
            controller,
            camera,
            final_holo,
            settle_time_s,
            frames_per_phase,
            capture_retries,
        )
        final_peak = float(np.max(final_image))

        if (
            auto_exposure
            and final_peak > target_peak
            and current_exposure_us > min_exposure_us + 1e-9
        ):
            new_exp = choose_lower_exposure(
                current_exposure_us,
                final_peak,
                target_peak,
                min_exposure_us,
            )
            if new_exp < current_exposure_us - 1e-9:
                print(
                    f"[auto exposure] final capture max={final_peak:.1f}; "
                    f"{current_exposure_us:.1f} us -> {new_exp:.1f} us"
                )
                current_exposure_us = set_exposure_us(camera, new_exp)
                continue
        break

    final_metric = target_metric(
        final_image,
        target_x,
        target_y,
        radius=metric_radius,
    )
    final_metric_normalized = (
        final_metric * reference_exposure_us / current_exposure_us
    )

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
        "frames_per_phase": int(frames_per_phase),
        "auto_exposure": bool(auto_exposure),
        "reference_exposure_us": float(reference_exposure_us),
        "final_exposure_us": float(current_exposure_us),
        "min_exposure_us": float(min_exposure_us),
        "target_peak": float(target_peak),
        "saturation_threshold": float(saturation_threshold),
        "baseline_target_metric": float(baseline_metric),
        "final_target_metric_raw": float(final_metric),
        "final_target_metric_normalized": float(final_metric_normalized),
        "normalized_target_metric_gain": float(
            final_metric_normalized / max(baseline_metric, 1e-12)
        ),
        "final_image_max": float(final_peak),
        "ideal_phase_only_pbr_piN_over_4": float(
            np.pi / 4.0 * total_channels
        ),
        "elapsed_s": float(time.time() - t0),
    }

    for key, value in analysis.items():
        if isinstance(value, (bool, str, int, float, np.integer, np.floating)):
            summary[f"tmcalib_{key}"] = (
                value.item()
                if isinstance(value, (np.integer, np.floating))
                else value
            )

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    plt.figure(figsize=(6, 5))
    plt.imshow(final_image)
    plt.scatter([target_x], [target_y], marker="+", s=100)
    plt.title(
        f"Direct sequential focus\n"
        f"PBR={analysis.get('pbr', float('nan')):.2f}, "
        f"exp={current_exposure_us:.0f} us"
    )
    plt.colorbar(label="Intensity")
    plt.tight_layout()
    plt.savefig(output_dir / "final_focused_image.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(
        np.arange(1, channels_to_run + 1),
        best_metric_normalized_history,
    )
    plt.xlabel("Optimized channel count")
    plt.ylabel(
        f"Target metric normalized to {reference_exposure_us:.0f} us"
    )
    plt.title("Sequential optimization convergence")
    plt.tight_layout()
    plt.savefig(output_dir / "optimization_curve_normalized.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, channels_to_run + 1), exposure_history)
    plt.xlabel("Optimized channel count")
    plt.ylabel("Exposure (us)")
    plt.title("Automatic exposure history")
    plt.tight_layout()
    plt.savefig(output_dir / "exposure_history.png", dpi=180)
    plt.close()

    print("\n" + "=" * 76)
    print("UNATTENDED DIRECT SEQUENTIAL OPTIMIZATION RESULT")
    print("=" * 76)
    print(f"Target                         : ({target_x}, {target_y})")
    print(f"Channels optimized             : {channels_to_run}/{total_channels}")
    print(f"Reference exposure             : {reference_exposure_us:.1f} us")
    print(f"Final exposure                 : {current_exposure_us:.1f} us")
    print(f"Baseline target metric         : {baseline_metric:.3f}")
    print(f"Final target metric (raw)      : {final_metric:.3f}")
    print(f"Final target metric (normalized): {final_metric_normalized:.3f}")
    print(
        f"Normalized target gain         : "
        f"{final_metric_normalized / max(baseline_metric, 1e-12):.2f}x"
    )
    print(f"Final image max                : {final_peak:.1f}")
    print(
        f"Final PBR (TMCalib method)     : "
        f"{analysis.get('pbr', float('nan')):.2f}"
    )
    print(
        f"Ideal pi*N/4 PBR               : "
        f"{np.pi / 4.0 * total_channels:.2f}"
    )
    print(f"Output directory               : {output_dir}")
    print("=" * 76)

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
    )
    parser.add_argument("--settle-time", type=float, default=0.02)
    parser.add_argument("--max-channels", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("direct_sequential_focus_auto_result"),
    )

    parser.add_argument(
        "--auto-exposure",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-auto-exposure",
        dest="auto_exposure",
        action="store_false",
    )
    parser.add_argument(
        "--initial-exposure",
        type=float,
        default=None,
        help="If omitted, keep the exposure configured by TMCalib.",
    )
    parser.add_argument("--min-exposure", type=float, default=50.0)
    parser.add_argument(
        "--target-peak",
        type=float,
        default=150.0,
        help="Auto exposure tries to keep image max near/below this value.",
    )
    parser.add_argument(
        "--saturation-threshold",
        type=float,
        default=235.0,
        help="Hard near-saturation threshold for 8-bit camera data.",
    )
    parser.add_argument(
        "--frames-per-phase",
        type=int,
        default=1,
        help="Average this many frames for each tested phase.",
    )
    parser.add_argument("--capture-retries", type=int, default=3)
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
            auto_exposure=args.auto_exposure,
            initial_exposure_us=args.initial_exposure,
            min_exposure_us=args.min_exposure,
            target_peak=args.target_peak,
            saturation_threshold=args.saturation_threshold,
            frames_per_phase=args.frames_per_phase,
            capture_retries=args.capture_retries,
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
