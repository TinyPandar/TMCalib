"""Diagnose probe/measurement pairing for the 32x24 correction dataset.

The most important check is sample alignment. A one-frame camera/DMD offset
makes an otherwise good TM look random (PCC near zero). This tool predicts the
camera intensity from the reconstructed TM, reports same-index PCC, camera frame
statistics, and tests small measurement shifts both globally and within each
acquisition batch.
"""

import argparse

import numpy as np
import torch

from calibration_profiles import PROFILES
from tools.run_tm_grad_correction import (
    _flatten_samples,
    _load_measurements,
    _load_tm_subset,
    _select_indices,
)


def _row_pcc(prediction, target, eps=1e-12):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    prediction = prediction - prediction.mean(axis=1, keepdims=True)
    target = target - target.mean(axis=1, keepdims=True)
    numerator = np.sum(prediction * target, axis=1)
    denominator = np.sqrt(
        np.sum(prediction * prediction, axis=1)
        * np.sum(target * target, axis=1)
    )
    result = np.full(numerator.shape, np.nan, dtype=np.float64)
    valid = denominator > eps
    result[valid] = numerator[valid] / denominator[valid]
    return result


def _shifted_pairs(prediction, measurement, shift, acquisition_batch_size):
    pred_parts = []
    meas_parts = []
    count = prediction.shape[0]
    for start in range(0, count, acquisition_batch_size):
        stop = min(start + acquisition_batch_size, count)
        if shift >= 0:
            pred_start = start
            pred_stop = stop - shift
            meas_start = start + shift
            meas_stop = stop
        else:
            amount = -shift
            pred_start = start + amount
            pred_stop = stop
            meas_start = start
            meas_stop = stop - amount
        if pred_stop <= pred_start:
            continue
        pred_parts.append(prediction[pred_start:pred_stop])
        meas_parts.append(measurement[meas_start:meas_stop])
    if not pred_parts:
        return np.empty((0, prediction.shape[1])), np.empty((0, measurement.shape[1]))
    return np.concatenate(pred_parts, axis=0), np.concatenate(meas_parts, axis=0)


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "no finite PCC values"
    return "mean={:.5f} median={:.5f} p10={:.5f} p90={:.5f} n={}/{}".format(
        float(np.mean(finite)),
        float(np.median(finite)),
        float(np.percentile(finite, 10)),
        float(np.percentile(finite, 90)),
        int(finite.size),
        int(values.size),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tm", default="reconstructed_field.npy")
    parser.add_argument(
        "--probes",
        default="correction_patterns_32x24_complex\\probe.npy",
    )
    parser.add_argument(
        "--measurements",
        default="correction_measurements_memmap.npy",
    )
    parser.add_argument("--profile", default="v4_32x24", choices=sorted(PROFILES))
    parser.add_argument("--tm-layout", default="output-input", choices=("output-input", "input-output"))
    parser.add_argument("--output-pixels", type=int, default=1024)
    parser.add_argument("--max-probes", type=int, default=0)
    parser.add_argument("--acquisition-batch-size", type=int, default=256)
    parser.add_argument("--max-shift", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--predict-batch-size", type=int, default=64)
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    input_shape = profile.input_shape
    n_input = int(np.prod(input_shape))

    probes_mm = np.load(args.probes, mmap_mode="r", allow_pickle=False)
    measurements_mm, measurement_format = _load_measurements(
        args.measurements, int(probes_mm.shape[0])
    )
    if probes_mm.shape[0] != measurements_mm.shape[0]:
        raise ValueError("probe/measurement sample counts differ")
    if int(np.prod(probes_mm.shape[1:])) != n_input:
        raise ValueError("probe shape does not match profile")

    measurement_output = int(np.prod(measurements_mm.shape[1:]))
    rng = np.random.default_rng(args.seed)
    output_indices = _select_indices(measurement_output, args.output_pixels, rng)
    sample_indices = _select_indices(probes_mm.shape[0], args.max_probes, rng)
    tm_subset, tm_output = _load_tm_subset(
        args.tm, n_input, output_indices, args.tm_layout
    )
    if tm_output != measurement_output:
        raise ValueError("TM output count does not match measurements")

    probes = np.asarray(
        _flatten_samples(probes_mm[sample_indices]), dtype=np.complex64
    )
    measurements = np.asarray(
        _flatten_samples(measurements_mm[sample_indices])[:, output_indices],
        dtype=np.float32,
    )

    if args.max_probes > 0 and args.max_probes < probes_mm.shape[0]:
        print("warning: shift diagnosis is clearest with --max-probes 0 (all probes)")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    tm_tensor = torch.from_numpy(np.asarray(tm_subset, dtype=np.complex64)).to(device)
    predictions = []
    with torch.inference_mode():
        for start in range(0, probes.shape[0], args.predict_batch_size):
            stop = min(start + args.predict_batch_size, probes.shape[0])
            batch = torch.from_numpy(probes[start:stop]).to(device)
            field = batch @ tm_tensor.transpose(0, 1)
            predictions.append(field.abs().square().cpu().numpy().astype(np.float32))
    prediction = np.concatenate(predictions, axis=0)

    frame_std = np.std(measurements, axis=1)
    frame_mean = np.mean(measurements, axis=1)
    near_flat = np.flatnonzero(frame_std < 1e-3)

    print("correction dataset diagnostic")
    print("  samples: {}".format(probes.shape[0]))
    print("  outputs used: {} / {}".format(len(output_indices), measurement_output))
    print("  measurement format: {}".format(measurement_format))
    print("  device: {}".format(device))
    print(
        "  camera frame mean min/median/max: {:.3f} / {:.3f} / {:.3f}".format(
            float(np.min(frame_mean)), float(np.median(frame_mean)), float(np.max(frame_mean))
        )
    )
    print(
        "  camera frame std  min/median/max: {:.3f} / {:.3f} / {:.3f}".format(
            float(np.min(frame_std)), float(np.median(frame_std)), float(np.max(frame_std))
        )
    )
    print("  near-flat frames (std < 1e-3): {}".format(int(near_flat.size)))
    if near_flat.size:
        print("  first near-flat indices: {}".format(near_flat[:20].tolist()))

    same_pcc = _row_pcc(prediction, measurements)
    print("\nsame-index prediction[k] vs measurement[k]")
    print("  {}".format(_summary(same_pcc)))

    print(
        "\nwithin-batch shift test (measurement shift s means prediction[k] "
        "is compared with measurement[k+s])"
    )
    scores = {}
    for shift in range(-abs(args.max_shift), abs(args.max_shift) + 1):
        pred_shifted, meas_shifted = _shifted_pairs(
            prediction,
            measurements,
            shift,
            max(1, int(args.acquisition_batch_size)),
        )
        values = _row_pcc(pred_shifted, meas_shifted)
        finite = values[np.isfinite(values)]
        score = float(np.mean(finite)) if finite.size else float("nan")
        scores[shift] = score
        print("  shift {:+d}: {}".format(shift, _summary(values)))

    finite_scores = {k: v for k, v in scores.items() if np.isfinite(v)}
    if finite_scores:
        best_shift = max(finite_scores, key=finite_scores.get)
        best_score = finite_scores[best_shift]
        zero_score = finite_scores.get(0, float("nan"))
        print("\nbest within-batch shift: {:+d} (mean PCC {:.5f})".format(best_shift, best_score))
        if best_shift != 0 and np.isfinite(zero_score) and best_score > zero_score + 0.10:
            print(
                "LIKELY FRAME ALIGNMENT ERROR: shifted pairing is much better than same-index pairing."
            )
            print(
                "Re-acquire with the updated acquisition script, which preflushes the camera stream before every batch."
            )
        elif np.isfinite(zero_score) and zero_score < 0.1:
            print(
                "No simple +/-{} frame shift fixes the near-random PCC; check optical stability, ROI/polarization, and complex DMD encoding next.".format(
                    abs(args.max_shift)
                )
            )


if __name__ == "__main__":
    main()
