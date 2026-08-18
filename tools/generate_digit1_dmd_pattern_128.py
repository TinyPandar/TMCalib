"""Generate a DMD hologram that forms the digit ``1`` on a 128 x 128 camera ROI.

The script uses the reconstructed 128 x 128 transmission matrix and keeps the
logical input grid, desired output, and modelled camera plane at 128 x 128.
Only the final, hardware-facing encoding expands the logical field into the
central 512 x 512 active area of the 1024 x 768 DMD.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from dmd_pattern_128 import (
    ACTIVE_HEIGHT,
    ACTIVE_WIDTH,
    ACTIVE_X,
    ACTIVE_Y,
    DMD_HEIGHT,
    DMD_WIDTH,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    input_field_to_dmd_pattern,
)


DEFAULT_TM = "reconstructed_field_128_px4_active512_8N.npy"
DEFAULT_OUTPUT_DIR = "digit1_dmd_pattern_128"


def build_digit_one_mask(stroke_width=2):
    """Return a centred, binary 128 x 128 mask shaped like the digit one."""
    stroke_width = int(stroke_width)
    if not 2 <= stroke_width <= 15:
        raise ValueError("stroke_width must be between 2 and 15 pixels")

    image = Image.new("L", (INPUT_WIDTH, INPUT_HEIGHT), 0)
    draw = ImageDraw.Draw(image)
    draw.line(
        [(51, 41), (64, 28), (64, 101)],
        fill=255,
        width=stroke_width,
        joint="curve",
    )
    draw.line(
        [(50, 102), (79, 102)],
        fill=255,
        width=stroke_width,
    )
    return np.asarray(image, dtype=np.uint8) > 0


def load_tm(tm_path):
    tm_path = Path(tm_path).resolve()
    if not tm_path.is_file():
        raise FileNotFoundError("Transmission matrix not found: {}".format(tm_path))
    matrix = np.load(str(tm_path), mmap_mode="r")
    expected = (INPUT_HEIGHT * INPUT_WIDTH, INPUT_HEIGHT * INPUT_WIDTH)
    if matrix.shape != expected:
        raise ValueError(
            "Transmission matrix shape {} does not match {}".format(
                matrix.shape, expected
            )
        )
    if not np.issubdtype(matrix.dtype, np.complexfloating):
        raise ValueError("Transmission matrix must contain complex values")
    return tm_path, matrix


def load_selected_rows(matrix, target_indices, row_chunk=64):
    """Copy only target rows from the 2 GiB memory-mapped TM."""
    target_indices = np.asarray(target_indices, dtype=np.intp)
    rows = np.empty((target_indices.size, matrix.shape[1]), dtype=np.complex64)
    for start in range(0, target_indices.size, int(row_chunk)):
        stop = min(start + int(row_chunk), target_indices.size)
        rows[start:stop] = matrix[target_indices[start:stop], :]
    if not np.all(np.isfinite(rows)):
        raise ValueError("Selected transmission-matrix rows contain NaN or infinity")
    return rows


def synthesize_phase_only_input(tm_rows, iterations=20, seed=12801):
    """Weighted phase conjugation for uniform multi-point illumination."""
    tm_rows = np.asarray(tm_rows, dtype=np.complex64)
    if tm_rows.ndim != 2 or tm_rows.shape[0] == 0:
        raise ValueError("tm_rows must be a non-empty two-dimensional array")

    rng = np.random.default_rng(int(seed))
    output_phase = rng.uniform(-np.pi, np.pi, tm_rows.shape[0]).astype(np.float32)
    weights = np.ones(tm_rows.shape[0], dtype=np.float32)
    eps = np.finfo(np.float32).eps
    history = []

    for iteration in range(max(1, int(iterations))):
        desired = weights * np.exp(1j * output_phase)
        backpropagated = desired @ tm_rows.conj()
        input_field = np.exp(1j * np.angle(backpropagated)).astype(np.complex64)

        achieved = tm_rows @ input_field
        achieved_amplitude = np.abs(achieved).astype(np.float32)
        mean_amplitude = float(np.mean(achieved_amplitude))
        uniformity = float(
            np.min(achieved_amplitude) / max(np.max(achieved_amplitude), eps)
        )
        coefficient_of_variation = float(
            np.std(achieved_amplitude) / max(mean_amplitude, eps)
        )
        history.append(
            {
                "iteration": iteration + 1,
                "target_amplitude_mean": mean_amplitude,
                "target_amplitude_uniformity_min_over_max": uniformity,
                "target_amplitude_cv": coefficient_of_variation,
            }
        )

        output_phase = np.angle(achieved).astype(np.float32)
        correction = mean_amplitude / np.maximum(achieved_amplitude, eps)
        correction = np.clip(correction, 0.5, 2.0)
        weights *= correction
        weights /= max(float(np.sqrt(np.mean(weights * weights))), eps)

    desired = weights * np.exp(1j * output_phase)
    backpropagated = desired @ tm_rows.conj()
    input_field = np.exp(1j * np.angle(backpropagated)).astype(np.complex64)
    return input_field, history


def predict_camera_intensity(matrix, input_vector, row_chunk=128):
    """Apply the full TM in bounded-memory row chunks."""
    output_field = np.empty(matrix.shape[0], dtype=np.complex64)
    row_chunk = max(1, int(row_chunk))
    for start in range(0, matrix.shape[0], row_chunk):
        stop = min(start + row_chunk, matrix.shape[0])
        output_field[start:stop] = matrix[start:stop, :] @ input_vector
    intensity = np.abs(output_field) ** 2
    return intensity.reshape(INPUT_HEIGHT, INPUT_WIDTH).astype(np.float32)


def normalize_to_uint8(array, upper_percentile=99.8, log_scale=False):
    values = np.asarray(array, dtype=np.float64)
    if log_scale:
        values = np.log1p(np.maximum(values, 0.0))
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    low = float(np.min(finite))
    high = float(np.percentile(finite, float(upper_percentile)))
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    return np.rint(255.0 * scaled).astype(np.uint8)


def save_phase_preview(path, input_field):
    phase = np.angle(input_field)
    phase_u8 = np.rint((phase + np.pi) * (255.0 / (2.0 * np.pi))).astype(
        np.uint8
    )
    Image.fromarray(phase_u8, mode="L").save(str(path))


def save_overview(path, target_mask, input_field, dmd_pattern, prediction):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    axes[0].imshow(target_mask, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Target: digit 1 (128 x 128)")
    phase_plot = axes[1].imshow(
        np.angle(input_field), cmap="twilight", vmin=-np.pi, vmax=np.pi
    )
    axes[1].set_title("Logical input phase (128 x 128)")
    fig.colorbar(phase_plot, ax=axes[1], fraction=0.046, pad=0.04)
    active = dmd_pattern[
        ACTIVE_Y : ACTIVE_Y + ACTIVE_HEIGHT,
        ACTIVE_X : ACTIVE_X + ACTIVE_WIDTH,
    ]
    axes[2].imshow(active, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("DMD active hologram (512 x 512)")
    if prediction is None:
        axes[3].axis("off")
        axes[3].text(0.5, 0.5, "Prediction skipped", ha="center", va="center")
    else:
        axes[3].imshow(np.log1p(prediction), cmap="inferno")
        axes[3].set_title("Predicted camera intensity (128 x 128)")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.savefig(str(path), dpi=180)
    plt.close(fig)


def generate(args):
    script_dir = Path(__file__).resolve().parent.parent
    tm_candidate = Path(args.tm)
    if not tm_candidate.is_absolute():
        tm_candidate = script_dir / tm_candidate
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tm_path, matrix = load_tm(tm_candidate)
    target_mask = build_digit_one_mask(args.stroke_width)
    target_indices = np.flatnonzero(target_mask.reshape(-1))
    tm_rows = load_selected_rows(matrix, target_indices, args.selected_row_chunk)
    input_vector, history = synthesize_phase_only_input(
        tm_rows,
        iterations=args.iterations,
        seed=args.seed,
    )
    input_field = input_vector.reshape(INPUT_HEIGHT, INPUT_WIDTH)
    dmd_pattern = input_field_to_dmd_pattern(
        input_field,
        px=4,
        ds_method="mean",
    )

    expected_dmd_shape = (DMD_HEIGHT, DMD_WIDTH)
    if dmd_pattern.shape != expected_dmd_shape:
        raise RuntimeError(
            "Generated DMD shape {} does not match {}".format(
                dmd_pattern.shape, expected_dmd_shape
            )
        )
    unique_values = np.unique(dmd_pattern)
    if not np.all(np.isin(unique_values, [0, 255])):
        raise RuntimeError("The generated DMD pattern is not binary")

    prediction = None
    if not args.skip_prediction:
        prediction = predict_camera_intensity(
            matrix,
            input_vector,
            row_chunk=args.prediction_row_chunk,
        )

    target_u8 = target_mask.astype(np.uint8) * np.uint8(255)
    Image.fromarray(target_u8, mode="L").save(
        str(output_dir / "digit1_target_128.png")
    )
    np.save(str(output_dir / "digit1_target_128.npy"), target_mask)
    np.save(str(output_dir / "digit1_logical_input_field_128.npy"), input_field)
    save_phase_preview(output_dir / "digit1_logical_input_phase_128.png", input_field)

    Image.fromarray(dmd_pattern, mode="L").save(
        str(output_dir / "digit1_dmd_pattern_1024x768.bmp")
    )
    np.save(str(output_dir / "digit1_dmd_pattern_1024x768.npy"), dmd_pattern)
    np.save(
        str(output_dir / "digit1_dmd_pattern_batch1.npy"),
        dmd_pattern[np.newaxis, ...],
    )

    metrics = {}
    if prediction is not None:
        np.save(
            str(output_dir / "digit1_predicted_camera_intensity_128.npy"),
            prediction,
        )
        Image.fromarray(normalize_to_uint8(prediction), mode="L").save(
            str(output_dir / "digit1_predicted_camera_128.png")
        )
        Image.fromarray(
            normalize_to_uint8(prediction, log_scale=True), mode="L"
        ).save(str(output_dir / "digit1_predicted_camera_log_128.png"))

        target_values = prediction[target_mask]
        background_values = prediction[~target_mask]
        target_mean = float(np.mean(target_values))
        background_mean = float(np.mean(background_values))
        metrics = {
            "predicted_target_mean": target_mean,
            "predicted_target_median": float(np.median(target_values)),
            "predicted_background_mean": background_mean,
            "predicted_background_median": float(np.median(background_values)),
            "predicted_target_to_background_mean_ratio": target_mean
            / max(background_mean, np.finfo(np.float32).eps),
            "predicted_target_uniformity_min_over_max": float(
                np.min(target_values) / max(float(np.max(target_values)), 1e-12)
            ),
        }

    outside_active = dmd_pattern.copy()
    outside_active[
        ACTIVE_Y : ACTIVE_Y + ACTIVE_HEIGHT,
        ACTIVE_X : ACTIVE_X + ACTIVE_WIDTH,
    ] = 0
    metadata = {
        "purpose": "Form the digit 1 on the calibrated camera ROI",
        "method": "weighted multi-target phase conjugation",
        "tm_path": str(tm_path),
        "tm_shape": list(matrix.shape),
        "tm_dtype": str(matrix.dtype),
        "logical_input_shape": [INPUT_HEIGHT, INPUT_WIDTH],
        "camera_target_shape": [INPUT_HEIGHT, INPUT_WIDTH],
        "dmd_shape": [DMD_HEIGHT, DMD_WIDTH],
        "dmd_active_shape": [ACTIVE_HEIGHT, ACTIVE_WIDTH],
        "dmd_active_offset_xy": [ACTIVE_X, ACTIVE_Y],
        "target_pixel_count": int(target_indices.size),
        "stroke_width": int(args.stroke_width),
        "iterations": int(args.iterations),
        "random_seed": int(args.seed),
        "dmd_unique_values": [int(value) for value in unique_values],
        "outside_active_nonzero_pixels": int(np.count_nonzero(outside_active)),
        "prediction_included": prediction is not None,
        "metrics": metrics,
        "optimization_history": history,
    }
    with (output_dir / "digit1_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    save_overview(
        output_dir / "digit1_overview.png",
        target_mask,
        input_field,
        dmd_pattern,
        prediction,
    )
    return output_dir, metadata


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tm", default=DEFAULT_TM)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stroke-width", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12804)
    parser.add_argument("--selected-row-chunk", type=int, default=64)
    parser.add_argument("--prediction-row-chunk", type=int, default=128)
    parser.add_argument(
        "--skip-prediction",
        action="store_true",
        help="Skip the full 128 x 128 model prediction to reduce disk reads.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    output_directory, run_metadata = generate(parse_args())
    print("Generated digit-1 DMD pattern in {}".format(output_directory))
    print("Target pixels: {}".format(run_metadata["target_pixel_count"]))
    if run_metadata["prediction_included"]:
        ratio = run_metadata["metrics"][
            "predicted_target_to_background_mean_ratio"
        ]
        print("Predicted target/background mean ratio: {:.3f}".format(ratio))
