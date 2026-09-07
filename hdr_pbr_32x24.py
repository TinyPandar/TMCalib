"""
HDR PBR measurement for TMCalib 32x24 sequential-focus result.

Replays an existing optimized phase map. It does NOT optimize again.

HDR idea
--------
- Low exposure: measure the focused peak without saturation.
- High exposure: measure the speckle background with much better quantization/SNR.
- Normalize both by exposure time before taking the ratio.

It also captures an all-zero DMD pattern at both exposures as a "black-pattern"
reference. Because this is not a mechanically blocked-laser dark frame, the
script reports BOTH:
    1) HDR PBR without black subtraction
    2) HDR PBR with black-pattern subtraction

Default paths/exposures match the user's first unattended run:
    optimized phases:
      direct_sequential_focus_auto_result/optimized_phases.npy
    low exposure: 77 us
    high exposure: 1158 us

PowerShell:
    python ./hdr_pbr_32x24.py `
        --target-x 64 `
        --target-y 64 `
        --low-exposure 77 `
        --high-exposure 1158 `
        --frames 20 `
        --mask-radius 3
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


def encode_input_field(input_field, lut_cache, ds_method="mean"):
    """Same 32x24 -> 1024x768 -> holo_SP mapping used by TMCalib."""
    input_field = np.asarray(input_field, dtype=np.complex64)
    if input_field.shape != (24, 32):
        raise ValueError(
            "Expected phase/input field shape (24, 32), got {}".format(
                input_field.shape
            )
        )

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


def set_exposure_us(camera, exposure_us):
    if not camera.configure_exposure(exposure_time=float(exposure_us)):
        raise RuntimeError(
            "Failed to set exposure to {:.1f} us".format(exposure_us)
        )
    try:
        return float(camera.cam.ExposureTime.GetValue())
    except Exception:
        return float(exposure_us)


def capture_one(controller, camera, pattern, settle_time_s=0.02, retries=3):
    pattern_batch = np.asarray([pattern], dtype=np.uint8)
    last_error = None

    for attempt in range(1, retries + 1):
        camera_started = False
        projection_started = False
        try:
            if not controller.load_pattern(pattern_batch):
                raise RuntimeError("Failed to load DMD pattern")

            camera.start()
            camera_started = True

            ret = controller.DMD.juoptProjection(controller.dev_id, 0, 0)
            if ret not in (None, 0):
                raise RuntimeError(
                    "juoptProjection failed with code {}".format(ret)
                )
            projection_started = True

            if settle_time_s > 0:
                time.sleep(float(settle_time_s))

            image = unwrap_camera_result(camera.run())
            if image is None:
                raise RuntimeError("Camera returned no image")

            if image.ndim == 3:
                image = np.mean(image, axis=2)

            return image.astype(np.float32, copy=False)

        except Exception as exc:
            last_error = exc
            print(
                "[capture retry {}/{}] {}".format(
                    attempt, retries, exc
                )
            )
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

    raise RuntimeError(
        "Capture failed after {} retries: {}".format(retries, last_error)
    )


def capture_stack(
    controller,
    camera,
    pattern,
    frames,
    settle_time_s,
    retries,
):
    images = []
    for i in range(int(frames)):
        image = capture_one(
            controller,
            camera,
            pattern,
            settle_time_s=settle_time_s,
            retries=retries,
        )
        images.append(image)
        print(
            "  frame {:02d}/{:02d}: min={:.1f}, mean={:.3f}, max={:.1f}".format(
                i + 1,
                int(frames),
                float(np.min(image)),
                float(np.mean(image)),
                float(np.max(image)),
            )
        )

    stack = np.stack(images, axis=0).astype(np.float32)
    return stack, np.mean(stack, axis=0)


def initialize_hardware(save_path):
    camera = core.CameraHandler(cam_index=0, save_path=str(save_path))
    camera.convert_to_12bit = False

    controller = core.DMDController(camera)
    devices = controller.get_devices()
    if not devices:
        camera.cleanup()
        raise RuntimeError("No DMD device found")

    print("DMD devices: {}".format(devices))
    if not controller.initialize_device(devices[0]):
        camera.cleanup()
        raise RuntimeError(
            "Failed to initialize DMD device: {}".format(devices[0])
        )

    return camera, controller


def make_background_mask(shape, target_x, target_y, radius):
    """
    Background = all ROI pixels except a circular focus exclusion region.
    A circular mask avoids putting the focused speckle/main lobe into the
    high-exposure background estimate.
    """
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    rr2 = (xx - int(target_x)) ** 2 + (yy - int(target_y)) ** 2
    return rr2 > int(radius) ** 2


def robust_background(values, trim_fraction=0.01):
    """
    Mean after symmetric trimming.

    A tiny trim reduces the influence of isolated dust diffraction / hot pixels
    without turning this into a median-based PBR. Set trim_fraction=0 to use
    the ordinary mean.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan

    trim_fraction = max(0.0, min(float(trim_fraction), 0.2))
    if trim_fraction <= 0:
        return float(np.mean(values))

    values = np.sort(values)
    k = int(values.size * trim_fraction)
    if 2 * k >= values.size:
        return float(np.mean(values))
    return float(np.mean(values[k:values.size - k]))


def safe_rate(image, exposure_us):
    return np.asarray(image, dtype=np.float64) / float(exposure_us)


def save_image(path, image, title, target_x=None, target_y=None):
    plt.figure(figsize=(6, 5))
    plt.imshow(image)
    if target_x is not None and target_y is not None:
        plt.scatter([target_x], [target_y], marker="+", s=100)
    plt.title(title)
    plt.colorbar(label="Intensity (DN)")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def run_hdr(
    controller,
    camera,
    phases_path,
    target_x,
    target_y,
    low_exposure_us,
    high_exposure_us,
    frames,
    mask_radius,
    saturation_threshold,
    settle_time_s,
    capture_retries,
    trim_fraction,
    output_dir,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    phases = np.load(str(phases_path))
    phases = np.asarray(phases, dtype=np.float32)
    if phases.shape != (24, 32):
        raise ValueError(
            "optimized_phases.npy must have shape (24, 32), got {}".format(
                phases.shape
            )
        )

    print("Loading optimized phase map: {}".format(phases_path))
    print(
        "Phase range: [{:.4f}, {:.4f}] rad".format(
            float(np.min(phases)), float(np.max(phases))
        )
    )

    print("Generating holo_SP LUT...")
    lut_cache = generate_lut("sp", 4)

    focus_field = np.exp(1j * phases).astype(np.complex64)
    focus_pattern = encode_input_field(focus_field, lut_cache)

    # Direct all-zero DMD bitmap for a black-pattern reference.
    # This is intentionally NOT called a true sensor dark frame.
    black_pattern = np.zeros_like(focus_pattern, dtype=np.uint8)

    # ----------------------------- low exposure -----------------------------
    print("\n=== LOW EXPOSURE: focused peak ===")
    actual_low = set_exposure_us(camera, low_exposure_us)
    print("Actual low exposure: {:.1f} us".format(actual_low))

    low_stack, low_focus = capture_stack(
        controller,
        camera,
        focus_pattern,
        frames,
        settle_time_s,
        capture_retries,
    )

    print("\n=== LOW EXPOSURE: black-pattern reference ===")
    low_black_stack, low_black = capture_stack(
        controller,
        camera,
        black_pattern,
        frames,
        settle_time_s,
        capture_retries,
    )

    # ---------------------------- high exposure -----------------------------
    print("\n=== HIGH EXPOSURE: background ===")
    actual_high = set_exposure_us(camera, high_exposure_us)
    print("Actual high exposure: {:.1f} us".format(actual_high))

    high_stack, high_focus = capture_stack(
        controller,
        camera,
        focus_pattern,
        frames,
        settle_time_s,
        capture_retries,
    )

    print("\n=== HIGH EXPOSURE: black-pattern reference ===")
    high_black_stack, high_black = capture_stack(
        controller,
        camera,
        black_pattern,
        frames,
        settle_time_s,
        capture_retries,
    )

    h, w = low_focus.shape
    if not (0 <= target_x < w and 0 <= target_y < h):
        raise ValueError(
            "Target ({}, {}) outside camera image {}x{}".format(
                target_x, target_y, w, h
            )
        )

    bg_mask = make_background_mask(
        low_focus.shape,
        target_x,
        target_y,
        mask_radius,
    )

    # At high exposure the focus itself may saturate; it is masked anyway.
    high_saturated = high_focus >= float(saturation_threshold)
    high_bg_saturated_count = int(np.sum(high_saturated & bg_mask))
    high_bg_pixel_count = int(np.sum(bg_mask))
    high_bg_saturated_fraction = (
        float(high_bg_saturated_count) / high_bg_pixel_count
        if high_bg_pixel_count
        else np.nan
    )

    # Never use saturated high-exposure pixels for the background.
    valid_high_bg_mask = bg_mask & (~high_saturated)

    # ------------------------ exposure-normalized rates ---------------------
    low_rate = safe_rate(low_focus, actual_low)
    high_rate = safe_rate(high_focus, actual_high)
    low_black_rate = safe_rate(low_black, actual_low)
    high_black_rate = safe_rate(high_black, actual_high)

    # Numerator: exact target pixel at LOW exposure.
    peak_dn_low = float(low_focus[target_y, target_x])
    peak_rate_no_sub = float(low_rate[target_y, target_x])

    # Black-pattern subtraction, clipped only AFTER subtraction.
    peak_rate_black_sub = float(
        max(
            0.0,
            low_rate[target_y, target_x]
            - low_black_rate[target_y, target_x],
        )
    )

    # Denominator: high-exposure background.
    background_rate_no_sub = robust_background(
        high_rate[valid_high_bg_mask],
        trim_fraction=trim_fraction,
    )

    high_rate_black_sub = np.maximum(
        high_rate - high_black_rate,
        0.0,
    )
    background_rate_black_sub = robust_background(
        high_rate_black_sub[valid_high_bg_mask],
        trim_fraction=trim_fraction,
    )

    hdr_pbr_no_sub = (
        peak_rate_no_sub / background_rate_no_sub
        if background_rate_no_sub > 0
        else np.inf
    )
    hdr_pbr_black_sub = (
        peak_rate_black_sub / background_rate_black_sub
        if background_rate_black_sub > 0
        else np.inf
    )

    # Naive single-exposure values for reference.
    low_bg_dn = robust_background(
        low_focus[bg_mask],
        trim_fraction=trim_fraction,
    )
    naive_low_pbr = (
        peak_dn_low / low_bg_dn if low_bg_dn > 0 else np.inf
    )

    high_bg_dn = robust_background(
        high_focus[valid_high_bg_mask],
        trim_fraction=trim_fraction,
    )

    # Background DN should increase roughly in proportion to exposure.
    expected_high_bg_from_low = (
        low_bg_dn * actual_high / actual_low
        if actual_low > 0
        else np.nan
    )

    # Frame-to-frame uncertainty for the peak.
    low_peak_samples = low_stack[:, target_y, target_x].astype(np.float64)
    low_peak_rate_samples = low_peak_samples / actual_low
    peak_rate_std = float(np.std(low_peak_rate_samples, ddof=1)) \
        if len(low_peak_rate_samples) > 1 else 0.0

    # High-exposure background uncertainty across frames.
    high_bg_rate_per_frame = []
    high_bg_sub_rate_per_frame = []
    for i in range(high_stack.shape[0]):
        frame = high_stack[i].astype(np.float64)
        black_frame = high_black_stack[i].astype(np.float64)
        sat = frame >= float(saturation_threshold)
        mask = bg_mask & (~sat)

        high_bg_rate_per_frame.append(
            robust_background(
                (frame / actual_high)[mask],
                trim_fraction=trim_fraction,
            )
        )
        high_bg_sub_rate_per_frame.append(
            robust_background(
                np.maximum(
                    frame / actual_high - black_frame / actual_high,
                    0.0,
                )[mask],
                trim_fraction=trim_fraction,
            )
        )

    high_bg_rate_per_frame = np.asarray(
        high_bg_rate_per_frame, dtype=np.float64
    )
    high_bg_sub_rate_per_frame = np.asarray(
        high_bg_sub_rate_per_frame, dtype=np.float64
    )

    bg_rate_std = float(np.nanstd(high_bg_rate_per_frame, ddof=1)) \
        if len(high_bg_rate_per_frame) > 1 else 0.0
    bg_sub_rate_std = float(
        np.nanstd(high_bg_sub_rate_per_frame, ddof=1)
    ) if len(high_bg_sub_rate_per_frame) > 1 else 0.0

    # Approximate relative-error propagation.
    def pbr_uncertainty(pbr, peak_rate, peak_std, bg_rate, bg_std):
        if (
            not np.isfinite(pbr)
            or peak_rate <= 0
            or bg_rate <= 0
        ):
            return np.nan
        return float(
            pbr
            * np.sqrt(
                (peak_std / peak_rate) ** 2
                + (bg_std / bg_rate) ** 2
            )
        )

    hdr_pbr_no_sub_std = pbr_uncertainty(
        hdr_pbr_no_sub,
        peak_rate_no_sub,
        peak_rate_std,
        background_rate_no_sub,
        bg_rate_std,
    )
    hdr_pbr_black_sub_std = pbr_uncertainty(
        hdr_pbr_black_sub,
        peak_rate_black_sub,
        peak_rate_std,
        background_rate_black_sub,
        bg_sub_rate_std,
    )

    summary = {
        "phases_path": str(phases_path),
        "target_x": int(target_x),
        "target_y": int(target_y),
        "frames": int(frames),
        "mask_radius_px": int(mask_radius),
        "trim_fraction": float(trim_fraction),
        "saturation_threshold_dn": float(saturation_threshold),

        "low_exposure_us": float(actual_low),
        "high_exposure_us": float(actual_high),
        "exposure_ratio_high_over_low": float(actual_high / actual_low),

        "low_peak_dn": float(peak_dn_low),
        "low_background_dn": float(low_bg_dn),
        "naive_low_exposure_pbr": float(naive_low_pbr),

        "high_background_dn": float(high_bg_dn),
        "expected_high_background_dn_from_low": float(
            expected_high_bg_from_low
        ),
        "high_background_saturated_fraction": float(
            high_bg_saturated_fraction
        ),

        "peak_rate_dn_per_us_no_sub": float(peak_rate_no_sub),
        "background_rate_dn_per_us_no_sub": float(
            background_rate_no_sub
        ),
        "hdr_pbr_no_black_subtraction": float(hdr_pbr_no_sub),
        "hdr_pbr_no_black_subtraction_std_approx": float(
            hdr_pbr_no_sub_std
        ),

        "peak_rate_dn_per_us_black_sub": float(peak_rate_black_sub),
        "background_rate_dn_per_us_black_sub": float(
            background_rate_black_sub
        ),
        "hdr_pbr_black_pattern_subtracted": float(
            hdr_pbr_black_sub
        ),
        "hdr_pbr_black_pattern_subtracted_std_approx": float(
            hdr_pbr_black_sub_std
        ),

        "low_black_target_dn": float(
            low_black[target_y, target_x]
        ),
        "high_black_mean_dn": float(np.mean(high_black)),
        "high_black_max_dn": float(np.max(high_black)),
    }

    np.save(output_dir / "low_focus_stack.npy", low_stack)
    np.save(output_dir / "low_focus_mean.npy", low_focus)
    np.save(output_dir / "low_black_stack.npy", low_black_stack)
    np.save(output_dir / "low_black_mean.npy", low_black)
    np.save(output_dir / "high_focus_stack.npy", high_stack)
    np.save(output_dir / "high_focus_mean.npy", high_focus)
    np.save(output_dir / "high_black_stack.npy", high_black_stack)
    np.save(output_dir / "high_black_mean.npy", high_black)
    np.save(output_dir / "background_mask.npy", bg_mask)

    with open(
        output_dir / "hdr_pbr_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    save_image(
        output_dir / "low_focus.png",
        low_focus,
        "Low exposure focus ({:.0f} us)".format(actual_low),
        target_x,
        target_y,
    )
    save_image(
        output_dir / "high_focus.png",
        high_focus,
        "High exposure background ({:.0f} us)".format(actual_high),
        target_x,
        target_y,
    )
    save_image(
        output_dir / "high_black.png",
        high_black,
        "High exposure black-pattern reference ({:.0f} us)".format(
            actual_high
        ),
    )

    # Visualize only valid HDR background pixels.
    bg_view = np.full_like(high_focus, np.nan, dtype=np.float32)
    bg_view[valid_high_bg_mask] = high_focus[valid_high_bg_mask]
    save_image(
        output_dir / "high_background_used.png",
        bg_view,
        "Pixels used for HDR background",
        target_x,
        target_y,
    )

    print("\n" + "=" * 78)
    print("HDR PBR RESULT")
    print("=" * 78)
    print(
        "Low / high exposure              : {:.1f} / {:.1f} us ({:.2f}x)".format(
            actual_low,
            actual_high,
            actual_high / actual_low,
        )
    )
    print(
        "Low-exposure peak                : {:.3f} DN".format(
            peak_dn_low
        )
    )
    print(
        "Low-exposure background          : {:.3f} DN".format(
            low_bg_dn
        )
    )
    print(
        "Naive low-exposure PBR           : {:.2f}".format(
            naive_low_pbr
        )
    )
    print(
        "High-exposure background         : {:.3f} DN".format(
            high_bg_dn
        )
    )
    print(
        "Expected high bg from low        : {:.3f} DN".format(
            expected_high_bg_from_low
        )
    )
    print(
        "Saturated pixels in bg mask      : {:.4%}".format(
            high_bg_saturated_fraction
        )
    )
    print("-" * 78)
    print(
        "HDR PBR (no black subtraction)   : {:.2f} +/- {:.2f}".format(
            hdr_pbr_no_sub,
            hdr_pbr_no_sub_std,
        )
    )
    print(
        "HDR PBR (black-pattern subtracted): {:.2f} +/- {:.2f}".format(
            hdr_pbr_black_sub,
            hdr_pbr_black_sub_std,
        )
    )
    print(
        "High-exposure black mean / max   : {:.3f} / {:.3f} DN".format(
            float(np.mean(high_black)),
            float(np.max(high_black)),
        )
    )
    print("Output directory                  : {}".format(output_dir))
    print("=" * 78)
    print(
        "\nInterpretation tip: if the two HDR PBR values differ a lot, "
        "the all-zero DMD pattern contains optical leakage and should not be "
        "treated as a true dark frame. In that case, trust the no-subtraction "
        "HDR number first, or repeat with the laser physically blocked."
    )

    return summary


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phases",
        type=Path,
        default=Path(
            "direct_sequential_focus_auto_result/optimized_phases.npy"
        ),
    )
    parser.add_argument("--target-x", type=int, default=64)
    parser.add_argument("--target-y", type=int, default=64)

    parser.add_argument("--low-exposure", type=float, default=77.0)
    parser.add_argument("--high-exposure", type=float, default=1158.0)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument(
        "--mask-radius",
        type=int,
        default=3,
        help="Exclude this radius around the focus from background.",
    )
    parser.add_argument(
        "--saturation-threshold",
        type=float,
        default=250.0,
        help="High-exposure pixels >= this value are excluded from background.",
    )
    parser.add_argument(
        "--trim-fraction",
        type=float,
        default=0.01,
        help="Symmetric fraction trimmed from background tails; 0 = plain mean.",
    )
    parser.add_argument("--settle-time", type=float, default=0.02)
    parser.add_argument("--capture-retries", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hdr_pbr_32x24_result"),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    camera = None
    controller = None
    try:
        camera, controller = initialize_hardware(
            args.output_dir / "camera"
        )
        run_hdr(
            controller=controller,
            camera=camera,
            phases_path=args.phases,
            target_x=args.target_x,
            target_y=args.target_y,
            low_exposure_us=args.low_exposure,
            high_exposure_us=args.high_exposure,
            frames=args.frames,
            mask_radius=args.mask_radius,
            saturation_threshold=args.saturation_threshold,
            settle_time_s=args.settle_time,
            capture_retries=args.capture_retries,
            trim_fraction=args.trim_fraction,
            output_dir=args.output_dir,
        )
    finally:
        if controller is not None:
            try:
                controller.cleanup()
            except Exception as exc:
                print("DMD cleanup warning: {}".format(exc))
        if camera is not None:
            try:
                camera.cleanup()
            except Exception as exc:
                print("Camera cleanup warning: {}".format(exc))


if __name__ == "__main__":
    main()
