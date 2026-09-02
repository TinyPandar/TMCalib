"""Measure the usable complex-field amplitude levels of the 128 x 96 profile.

The experiment applies spatially uniform, zero-phase complex fields with
commanded amplitudes from 0 to 1.  Measurement order is randomized inside each
repeat and every repeat is bracketed by dark and full-scale reference frames.
This isolates the DMD/hologram amplitude response from slow laser and camera
drift while keeping the existing 128 x 96 -> 128 x 128 hardware path intact.

Run from the repository root.  ``--prepare-only`` is the safe default and does
not open the camera or DMD.  Hardware access requires the explicit
``--acquire`` flag.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


INPUT_SHAPE = (96, 128)
CAMERA_SHAPE = (128, 128)
DMD_SHAPE = (768, 1024)
DEFAULT_LEVEL_COUNT = 41
DEFAULT_REPEATS = 10
DEFAULT_SEED = 20260827
DEFAULT_EXPOSURE_US = 60.0
DEFAULT_SENSOR_MAX = 255.0
DISTINGUISHABILITY_Z = 3.0


@dataclass(frozen=True)
class SequenceEntry:
    sequence_index: int
    repeat_index: int
    position_in_repeat: int
    kind: str
    commanded_amplitude: float


def build_amplitude_levels(level_count: int = DEFAULT_LEVEL_COUNT) -> np.ndarray:
    """Return uniformly spaced command amplitudes including 0 and 1."""
    level_count = int(level_count)
    if level_count < 2:
        raise ValueError("level_count must be at least 2")
    return np.linspace(0.0, 1.0, level_count, dtype=np.float64)


def build_sequence(
    levels: Sequence[float],
    repeats: int = DEFAULT_REPEATS,
    seed: int = DEFAULT_SEED,
) -> List[SequenceEntry]:
    """Randomize level order and bracket every repeat with dark/white anchors."""
    levels_array = np.asarray(levels, dtype=np.float64)
    if levels_array.ndim != 1 or levels_array.size < 2:
        raise ValueError("levels must be a one-dimensional array with >=2 values")
    if not np.all(np.isfinite(levels_array)):
        raise ValueError("levels contain NaN or infinity")
    if np.any(levels_array < 0.0) or np.any(levels_array > 1.0):
        raise ValueError("all amplitude levels must lie in [0, 1]")
    if np.unique(levels_array).size != levels_array.size:
        raise ValueError("amplitude levels must be unique")
    repeats = int(repeats)
    if repeats < 2:
        raise ValueError("at least two repeats are required for uncertainty estimates")

    rng = np.random.default_rng(int(seed))
    entries: List[SequenceEntry] = []
    sequence_index = 0
    for repeat_index in range(repeats):
        randomized = levels_array[rng.permutation(levels_array.size)]
        block: List[Tuple[str, float]] = [("dark_anchor", 0.0), ("white_anchor", 1.0)]
        block.extend(("level", float(value)) for value in randomized)
        block.extend((("white_anchor", 1.0), ("dark_anchor", 0.0)))
        for position, (kind, amplitude) in enumerate(block):
            entries.append(
                SequenceEntry(
                    sequence_index=sequence_index,
                    repeat_index=repeat_index,
                    position_in_repeat=position,
                    kind=kind,
                    commanded_amplitude=float(amplitude),
                )
            )
            sequence_index += 1
    return entries


def entries_by_repeat(entries: Sequence[SequenceEntry]) -> List[List[SequenceEntry]]:
    grouped: Dict[int, List[SequenceEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.repeat_index, []).append(entry)
    return [grouped[index] for index in sorted(grouped)]


def _default_encoder(field: np.ndarray) -> np.ndarray:
    from dmd_pattern_128x96 import input_field_to_dmd_pattern

    return input_field_to_dmd_pattern(field)


def build_pattern_cache(
    amplitudes: Iterable[float],
    phase_rad: float = 0.0,
    encoder: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[float, np.ndarray]:
    """Encode one hologram per unique amplitude and reuse it in every repeat."""
    if encoder is None:
        encoder = _default_encoder
    phase_factor = np.complex64(np.exp(1j * float(phase_rad)))
    cache: Dict[float, np.ndarray] = {}
    for amplitude in sorted({float(value) for value in amplitudes}):
        field = np.full(
            INPUT_SHAPE,
            np.float32(amplitude) * phase_factor,
            dtype=np.complex64,
        )
        pattern = np.asarray(encoder(field), dtype=np.uint8)
        if pattern.shape != DMD_SHAPE:
            raise ValueError(
                "encoder returned pattern shape {}; expected {}".format(
                    pattern.shape, DMD_SHAPE
                )
            )
        cache[amplitude] = np.ascontiguousarray(pattern)
    return cache


def patterns_for_entries(
    entries: Sequence[SequenceEntry],
    pattern_cache: Dict[float, np.ndarray],
) -> np.ndarray:
    return np.stack(
        [pattern_cache[float(entry.commanded_amplitude)] for entry in entries],
        axis=0,
    ).astype(np.uint8, copy=False)


def save_sequence_csv(entries: Sequence[SequenceEntry], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(entries[0]).keys()))
        writer.writeheader()
        for entry in entries:
            writer.writerow(asdict(entry))


def _mean_frame_metric(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim != 3 or tuple(frames.shape[1:]) != CAMERA_SHAPE:
        raise ValueError(
            "frames must have shape (N, {}, {}), got {}".format(
                CAMERA_SHAPE[0], CAMERA_SHAPE[1], frames.shape
            )
        )
    return frames.mean(axis=(1, 2))


def _mean_std_se(values: np.ndarray) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    se = std / math.sqrt(values.size) if values.size else float("nan")
    return mean, std, se


def _z_separation(mean_a: float, se_a: float, mean_b: float, se_b: float) -> float:
    denominator = math.hypot(float(se_a), float(se_b))
    difference = float(mean_a) - float(mean_b)
    if denominator <= 1e-12:
        return float("inf") if difference > 0 else 0.0
    return difference / denominator


def compute_response(
    frames: np.ndarray,
    entries: Sequence[SequenceEntry],
    sensor_max: float = DEFAULT_SENSOR_MAX,
    distinguishability_z: float = DISTINGUISHABILITY_Z,
) -> Tuple[List[dict], List[dict], dict]:
    """Dark/reference normalize captures and summarize the amplitude response."""
    frames = np.asarray(frames)
    if frames.shape[0] != len(entries):
        raise ValueError(
            "frame count {} does not match sequence length {}".format(
                frames.shape[0], len(entries)
            )
        )
    if sensor_max <= 0:
        raise ValueError("sensor_max must be positive")
    metrics = _mean_frame_metric(frames)

    repeat_anchors: Dict[int, Tuple[float, float]] = {}
    for block in entries_by_repeat(entries):
        indices = np.asarray([entry.sequence_index for entry in block], dtype=np.intp)
        dark_idx = indices[[entry.kind == "dark_anchor" for entry in block]]
        white_idx = indices[[entry.kind == "white_anchor" for entry in block]]
        dark = float(np.mean(metrics[dark_idx]))
        white = float(np.mean(metrics[white_idx]))
        if white <= dark:
            raise RuntimeError(
                "repeat {} has white anchor <= dark anchor ({:.4f} <= {:.4f})".format(
                    block[0].repeat_index, white, dark
                )
            )
        repeat_anchors[block[0].repeat_index] = (dark, white)

    per_capture: List[dict] = []
    for entry, raw_metric, frame in zip(entries, metrics, frames):
        dark, white = repeat_anchors[entry.repeat_index]
        normalized_intensity = (float(raw_metric) - dark) / (white - dark)
        normalized_amplitude = math.sqrt(max(normalized_intensity, 0.0))
        per_capture.append(
            {
                **asdict(entry),
                "raw_mean_intensity": float(raw_metric),
                "repeat_dark_mean": dark,
                "repeat_white_mean": white,
                "normalized_intensity": normalized_intensity,
                "normalized_amplitude": normalized_amplitude,
                "saturated_fraction": float(np.mean(np.asarray(frame) >= sensor_max)),
            }
        )

    level_records = [record for record in per_capture if record["kind"] == "level"]
    levels = sorted({float(record["commanded_amplitude"]) for record in level_records})
    summary_rows: List[dict] = []
    for level in levels:
        selected = [
            record for record in level_records
            if float(record["commanded_amplitude"]) == level
        ]
        norm_i = np.asarray([record["normalized_intensity"] for record in selected])
        norm_a = np.asarray([record["normalized_amplitude"] for record in selected])
        raw_i = np.asarray([record["raw_mean_intensity"] for record in selected])
        saturation = np.asarray([record["saturated_fraction"] for record in selected])
        mean_i, std_i, se_i = _mean_std_se(norm_i)
        mean_a, std_a, se_a = _mean_std_se(norm_a)
        summary_rows.append(
            {
                "commanded_amplitude": level,
                "expected_normalized_intensity": level**2,
                "capture_count": len(selected),
                "raw_mean_intensity": float(np.mean(raw_i)),
                "normalized_intensity_mean": mean_i,
                "normalized_intensity_std": std_i,
                "normalized_intensity_se": se_i,
                "measured_amplitude_mean": mean_a,
                "measured_amplitude_std": std_a,
                "measured_amplitude_se": se_a,
                "saturated_fraction_mean": float(np.mean(saturation)),
            }
        )

    zero_row = min(summary_rows, key=lambda row: abs(row["commanded_amplitude"]))
    previous: Optional[dict] = None
    for row in summary_rows:
        zero_z = _z_separation(
            row["normalized_intensity_mean"],
            row["normalized_intensity_se"],
            zero_row["normalized_intensity_mean"],
            zero_row["normalized_intensity_se"],
        )
        row["z_vs_zero"] = zero_z
        row["distinguishable_from_zero"] = bool(
            row["commanded_amplitude"] > 0.0
            and row["normalized_intensity_mean"]
            > zero_row["normalized_intensity_mean"]
            and zero_z >= distinguishability_z
        )
        if previous is None:
            row["z_vs_previous"] = float("nan")
            row["distinguishable_from_previous"] = False
        else:
            adjacent_z = _z_separation(
                row["measured_amplitude_mean"],
                row["measured_amplitude_se"],
                previous["measured_amplitude_mean"],
                previous["measured_amplitude_se"],
            )
            row["z_vs_previous"] = adjacent_z
            row["distinguishable_from_previous"] = bool(
                row["measured_amplitude_mean"] > previous["measured_amplitude_mean"]
                and adjacent_z >= distinguishability_z
            )
        previous = row

    command = np.asarray([row["commanded_amplitude"] for row in summary_rows])
    measured_a = np.asarray([row["measured_amplitude_mean"] for row in summary_rows])
    measured_i = np.asarray([row["normalized_intensity_mean"] for row in summary_rows])
    positive = (command > 0) & (measured_i > 0)
    gamma = float("nan")
    if np.count_nonzero(positive) >= 2:
        gamma = float(
            np.polyfit(np.log(command[positive]), np.log(measured_i[positive]), 1)[0]
        )
    detectable = [
        row["commanded_amplitude"]
        for row in summary_rows
        if row["distinguishable_from_zero"]
    ]
    monotonic_violations = int(np.count_nonzero(np.diff(measured_a) < -1e-6))
    experiment_summary = {
        "level_count": len(summary_rows),
        "repeat_count": len(repeat_anchors),
        "distinguishability_z_threshold": float(distinguishability_z),
        "lowest_amplitude_distinguishable_from_zero": (
            float(min(detectable)) if detectable else None
        ),
        "adjacent_distinguishable_transition_count": int(
            sum(bool(row["distinguishable_from_previous"]) for row in summary_rows)
        ),
        "monotonic_violation_count": monotonic_violations,
        "amplitude_rmse": float(np.sqrt(np.mean((measured_a - command) ** 2))),
        "intensity_power_law_exponent": gamma,
        "maximum_saturated_fraction": float(
            max(row["saturated_fraction_mean"] for row in summary_rows)
        ),
    }
    return per_capture, summary_rows, experiment_summary


def build_inverse_lut(summary_rows: Sequence[dict], lut_size: int = 256) -> dict:
    """Invert a monotonicized measured response into a desired->command LUT."""
    if lut_size < 2:
        raise ValueError("lut_size must be at least 2")
    command = np.asarray(
        [row["commanded_amplitude"] for row in summary_rows], dtype=np.float64
    )
    measured = np.asarray(
        [row["measured_amplitude_mean"] for row in summary_rows], dtype=np.float64
    )
    order = np.argsort(command)
    command = command[order]
    measured = np.maximum.accumulate(np.clip(measured[order], 0.0, None))
    if measured[-1] <= measured[0] + 1e-12:
        raise RuntimeError("measured amplitude response is flat; LUT cannot be inverted")
    measured = (measured - measured[0]) / (measured[-1] - measured[0])
    keep = np.concatenate(([True], np.diff(measured) > 1e-9))
    desired = np.linspace(0.0, 1.0, int(lut_size), dtype=np.float64)
    command_lut = np.interp(
        desired,
        measured[keep],
        command[keep],
        left=float(command[0]),
        right=float(command[-1]),
    )
    return {
        "lut_definition": "desired_normalized_field_amplitude_to_dmd_command_amplitude",
        "monotonicization": "cumulative_max_before_linear_interpolation",
        "desired_amplitude": desired.tolist(),
        "commanded_amplitude": command_lut.tolist(),
    }


def write_dict_rows(path: str, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty table")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def plot_response(summary_rows: Sequence[dict], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    command = np.asarray([row["commanded_amplitude"] for row in summary_rows])
    mean_a = np.asarray([row["measured_amplitude_mean"] for row in summary_rows])
    std_a = np.asarray([row["measured_amplitude_std"] for row in summary_rows])
    mean_i = np.asarray([row["normalized_intensity_mean"] for row in summary_rows])
    std_i = np.asarray([row["normalized_intensity_std"] for row in summary_rows])

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].errorbar(command, mean_a, yerr=std_a, fmt="o", ms=3, capsize=2)
    axes[0].plot([0, 1], [0, 1], "--", color="0.35", label="ideal A_out=A_cmd")
    axes[0].set(xlabel="Commanded amplitude", ylabel="Measured normalized amplitude")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].errorbar(command, mean_i, yerr=std_i, fmt="o", ms=3, capsize=2)
    axes[1].plot(command, command**2, "--", color="0.35", label="ideal I=A_cmd^2")
    axes[1].set(xlabel="Commanded amplitude", ylabel="Dark/reference normalized intensity")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def analyze_and_save(
    frames: np.ndarray,
    entries: Sequence[SequenceEntry],
    output_dir: str,
    sensor_max: float,
    distinguishability_z: float,
) -> dict:
    per_capture, summary_rows, experiment_summary = compute_response(
        frames,
        entries,
        sensor_max=sensor_max,
        distinguishability_z=distinguishability_z,
    )
    write_dict_rows(os.path.join(output_dir, "per_capture.csv"), per_capture)
    write_dict_rows(os.path.join(output_dir, "amplitude_response.csv"), summary_rows)
    write_json(
        os.path.join(output_dir, "amplitude_lut.json"),
        build_inverse_lut(summary_rows),
    )
    write_json(os.path.join(output_dir, "experiment_summary.json"), experiment_summary)
    plot_response(summary_rows, os.path.join(output_dir, "amplitude_response.png"))
    return experiment_summary


def acquire_frames(
    entries: Sequence[SequenceEntry],
    pattern_cache: Dict[float, np.ndarray],
    output_path: str,
    camera_index: int,
    exposure_us: float,
    dmd_device_name: Optional[str],
) -> np.ndarray:
    """Acquire one randomized repeat at a time through existing hardware classes."""
    import calibrate_128x96 as profile

    camera = None
    controller = None
    raw = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.uint16,
        shape=(len(entries), CAMERA_SHAPE[0], CAMERA_SHAPE[1]),
    )
    try:
        camera = profile.core.CameraHandler(cam_index=camera_index, save_path="./camera_1")
        camera.convert_to_12bit = False
        camera.configure_exposure(exposure_time=float(exposure_us))
        controller = profile.DMDController(camera)
        devices = controller.get_devices()
        if not devices:
            raise RuntimeError("no online JUOPT DMD device was found")
        selected_device = dmd_device_name or devices[0]
        if selected_device not in devices:
            raise ValueError(
                "requested DMD {!r} is not in {}".format(selected_device, devices)
            )
        if not controller.initialize_device(selected_device):
            raise RuntimeError("failed to initialize DMD {!r}".format(selected_device))
        camera.start()

        for block in entries_by_repeat(entries):
            patterns = patterns_for_entries(block, pattern_cache)
            captured = controller.project_and_caption(patterns)
            if captured is None or captured.shape != (len(block),) + CAMERA_SHAPE:
                raise RuntimeError(
                    "capture returned {}; expected {}".format(
                        None if captured is None else captured.shape,
                        (len(block),) + CAMERA_SHAPE,
                    )
                )
            indices = [entry.sequence_index for entry in block]
            raw[indices] = np.rint(captured).astype(np.uint16)
            raw.flush()
            print(
                "Captured repeat {}/{}".format(
                    block[0].repeat_index + 1,
                    len(entries_by_repeat(entries)),
                ),
                flush=True,
            )
    finally:
        # Stop the DMD trigger source before tearing down the triggered camera.
        if controller is not None:
            try:
                controller.cleanup()
            except Exception as error:
                print("Warning: DMD cleanup failed: {}".format(error))
        if camera is not None:
            try:
                camera.stop()
            except Exception as error:
                print("Warning: camera stop failed: {}".format(error))
            try:
                camera.cleanup()
            except Exception as error:
                print("Warning: camera cleanup failed: {}".format(error))
        raw.flush()
    return np.load(output_path, mmap_mode="r")


def default_output_dir() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("experiments", "amplitude_levels_128x96", stamp)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--prepare-only",
        action="store_true",
        help="prepare manifest/patterns only (default)",
    )
    mode.add_argument(
        "--acquire",
        action="store_true",
        help="open hardware and acquire camera frames",
    )
    mode.add_argument(
        "--analyze",
        metavar="RAW_FRAMES_NPY",
        help="analyze an existing raw frame stack",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--level-count", type=int, default=DEFAULT_LEVEL_COUNT)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--phase-rad", type=float, default=0.0)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--exposure-us", type=float, default=DEFAULT_EXPOSURE_US)
    parser.add_argument("--sensor-max", type=float, default=DEFAULT_SENSOR_MAX)
    parser.add_argument("--dmd-device", default=None)
    parser.add_argument(
        "--distinguishability-z",
        type=float,
        default=DISTINGUISHABILITY_Z,
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = os.path.abspath(args.output_dir or default_output_dir())
    os.makedirs(output_dir, exist_ok=True)

    levels = build_amplitude_levels(args.level_count)
    entries = build_sequence(levels, repeats=args.repeats, seed=args.seed)
    save_sequence_csv(entries, os.path.join(output_dir, "sequence.csv"))
    metadata = {
        "experiment_question": (
            "How many commanded field-amplitude levels are measurably distinct?"
        ),
        "hypothesis": (
            "After dark/reference correction, intensity follows A^2 and "
            "recovered amplitude follows A."
        ),
        "profile": "fourfold_128x96",
        "input_shape": list(INPUT_SHAPE),
        "camera_shape": list(CAMERA_SHAPE),
        "dmd_shape": list(DMD_SHAPE),
        "phase_rad": float(args.phase_rad),
        "level_count": int(args.level_count),
        "levels": levels.tolist(),
        "repeat_count": int(args.repeats),
        "random_seed": int(args.seed),
        "exposure_us": float(args.exposure_us),
        "sensor_max": float(args.sensor_max),
        "distinguishability_z": float(args.distinguishability_z),
        "sequence_length": len(entries),
        "normalization": "per-repeat dark/white anchors; A_meas=sqrt((I-I_dark)/(I_white-I_dark))",
    }
    write_json(os.path.join(output_dir, "protocol.json"), metadata)

    if args.analyze:
        frames = np.load(args.analyze, mmap_mode="r")
        summary = analyze_and_save(
            frames,
            entries,
            output_dir,
            sensor_max=args.sensor_max,
            distinguishability_z=args.distinguishability_z,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    cache = build_pattern_cache(levels, phase_rad=args.phase_rad)
    unique_patterns = np.stack([cache[float(level)] for level in levels], axis=0)
    np.save(os.path.join(output_dir, "unique_level_patterns.npy"), unique_patterns)
    if not args.acquire:
        print("Prepared amplitude-level protocol in: {}".format(output_dir))
        print("No hardware was opened. Re-run with --acquire to measure.")
        return 0

    frames_path = os.path.join(output_dir, "raw_frames.npy")
    frames = acquire_frames(
        entries,
        cache,
        frames_path,
        camera_index=args.camera_index,
        exposure_us=args.exposure_us,
        dmd_device_name=args.dmd_device,
    )
    summary = analyze_and_save(
        frames,
        entries,
        output_dir,
        sensor_max=args.sensor_max,
        distinguishability_z=args.distinguishability_z,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
