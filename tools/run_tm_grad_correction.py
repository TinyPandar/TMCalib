"""CLI for gradient-based low-dimensional TM correction."""

import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from calibration_profiles import PROFILES
from tm_grad_correction import export_corrected_tm, fit_input_dct_correction


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a low-dimensional input-side correction to a reconstructed TM "
            "using measured camera intensities."
        )
    )
    parser.add_argument("--tm", required=True, help="Reconstructed complex TM .npy")
    parser.add_argument(
        "--probes",
        required=True,
        help="Complex calibration probes .npy, shape [K,H,W] or [K,N]",
    )
    parser.add_argument(
        "--measurements",
        required=True,
        help="Measured camera intensities .npy, shape [K,Hout,Wout] or [K,M]",
    )
    parser.add_argument(
        "--profile",
        default="v4_32x24",
        choices=sorted(PROFILES),
        help="Defines input HxW and expected camera ROI",
    )
    parser.add_argument(
        "--tm-layout",
        choices=("output-input", "input-output"),
        default="output-input",
        help="Stored TM axis order. TMCalib reconstructed fields use output-input.",
    )
    parser.add_argument(
        "--phase-modes",
        type=int,
        nargs=2,
        metavar=("KY", "KX"),
        default=(4, 4),
        help="Low-frequency DCT phase mode grid; DC is excluded",
    )
    parser.add_argument(
        "--amplitude-modes",
        type=int,
        nargs=2,
        metavar=("KY", "KX"),
        default=(2, 2),
        help="Low-frequency DCT log-amplitude mode grid; DC is excluded",
    )
    parser.add_argument(
        "--max-probes",
        type=int,
        default=256,
        help="Randomly use at most this many measured probes; 0 uses all",
    )
    parser.add_argument(
        "--output-pixels",
        type=int,
        default=1024,
        help="Randomly use this many camera pixels in the loss; 0 uses all",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device, e.g. auto, cuda, cuda:0, cpu",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        default="tm_grad_correction",
        help="Directory for the learned gain, maps, curves and summary",
    )
    parser.add_argument(
        "--export-tm",
        action="store_true",
        help="Also write the full corrected TM (can be multi-GB)",
    )
    return parser.parse_args()


def _select_indices(total: int, requested: int, rng: np.random.Generator) -> np.ndarray:
    if total <= 0:
        raise ValueError("cannot select from an empty axis")
    if requested <= 0 or requested >= total:
        return np.arange(total, dtype=np.int64)
    return np.sort(rng.choice(total, size=requested, replace=False)).astype(
        np.int64
    )


def _flatten_samples(array: np.ndarray) -> np.ndarray:
    if array.ndim < 2:
        raise ValueError("sample array must have at least two dimensions")
    return array.reshape(array.shape[0], -1)


def _load_measurements(path: str, sample_count: int) -> Tuple[np.ndarray, str]:
    """Load either an NPY array or TMCalib's headerless uint16 memmap."""
    if sample_count <= 0:
        raise ValueError("measurement sample count must be positive")

    with open(path, "rb") as stream:
        magic = stream.read(len(np.lib.format.MAGIC_PREFIX))
        is_npy = magic == np.lib.format.MAGIC_PREFIX

    if is_npy:
        return np.load(path, mmap_mode="r", allow_pickle=False), "NPY"

    itemsize = np.dtype(np.uint16).itemsize
    row_bytes = sample_count * itemsize
    file_size = os.path.getsize(path)
    if file_size == 0 or file_size % row_bytes != 0:
        raise ValueError(
            "measurement file is not NPY and its size ({}) is incompatible "
            "with a headerless uint16 memmap containing {} samples".format(
                file_size, sample_count
            )
        )

    output_count = file_size // row_bytes
    return (
        np.memmap(
            path,
            dtype=np.uint16,
            mode="r",
            shape=(sample_count, output_count),
        ),
        "headerless uint16 memmap",
    )


def _load_tm_subset(
    tm_path: str,
    n_input: int,
    output_indices: np.ndarray,
    layout: str,
) -> Tuple[np.ndarray, int]:
    tm = np.load(tm_path, mmap_mode="r")
    if tm.ndim != 2:
        raise ValueError("TM must be a 2-D NPY array")

    if layout == "output-input":
        if tm.shape[1] != n_input:
            raise ValueError(
                "TM input axis has {}, expected {}".format(tm.shape[1], n_input)
            )
        total_output = tm.shape[0]
        selected = np.asarray(tm[output_indices, :], dtype=np.complex64)
    else:
        if tm.shape[0] != n_input:
            raise ValueError(
                "TM input axis has {}, expected {}".format(tm.shape[0], n_input)
            )
        total_output = tm.shape[1]
        selected = np.asarray(tm[:, output_indices], dtype=np.complex64).T

    return selected, int(total_output)


def main() -> None:
    args = _parse_args()
    profile = PROFILES[args.profile]
    input_shape = profile.input_shape
    n_input = int(input_shape[0] * input_shape[1])
    expected_output = int(profile.camera_roi[0] * profile.camera_roi[1])

    probes_mm = np.load(args.probes, mmap_mode="r")
    measurements_mm, measurement_format = _load_measurements(
        args.measurements, int(probes_mm.shape[0])
    )
    if probes_mm.shape[0] != measurements_mm.shape[0]:
        raise ValueError(
            "probe/measurement sample counts differ: {} vs {}".format(
                probes_mm.shape[0], measurements_mm.shape[0]
            )
        )

    if int(np.prod(probes_mm.shape[1:])) != n_input:
        raise ValueError(
            "probe shape {} does not match profile {} input {}".format(
                probes_mm.shape, args.profile, input_shape
            )
        )

    measurement_output = int(np.prod(measurements_mm.shape[1:]))
    rng = np.random.default_rng(args.seed)
    output_indices = _select_indices(
        measurement_output, args.output_pixels, rng
    )
    tm_subset, tm_output = _load_tm_subset(
        args.tm, n_input, output_indices, args.tm_layout
    )
    if tm_output != measurement_output:
        raise ValueError(
            "TM has {} outputs but measurements have {}".format(
                tm_output, measurement_output
            )
        )
    if expected_output != measurement_output:
        print(
            "warning: profile {} expects {} camera pixels, but measurements contain {}".format(
                args.profile, expected_output, measurement_output
            )
        )

    sample_indices = _select_indices(
        probes_mm.shape[0], args.max_probes, rng
    )
    probes = np.asarray(
        _flatten_samples(probes_mm[sample_indices]), dtype=np.complex64
    )
    measurements = np.asarray(
        _flatten_samples(measurements_mm[sample_indices])[:, output_indices],
        dtype=np.float32,
    )

    print("TM correction setup")
    print("  profile: {}".format(args.profile))
    print("  input shape: {} (N={})".format(input_shape, n_input))
    print("  measurement format: {}".format(measurement_format))
    print(
        "  calibration probes: {} / {}".format(
            len(sample_indices), probes_mm.shape[0]
        )
    )
    print(
        "  camera pixels in loss: {} / {}".format(
            len(output_indices), measurement_output
        )
    )
    print(
        "  DCT modes: phase {}x{}, amplitude {}x{} (DC excluded)".format(
            args.phase_modes[0],
            args.phase_modes[1],
            args.amplitude_modes[0],
            args.amplitude_modes[1],
        )
    )

    model, result = fit_input_dct_correction(
        tm=tm_subset,
        probes=probes,
        measured_intensity=measurements,
        input_shape=input_shape,
        phase_modes=tuple(args.phase_modes),
        amplitude_modes=tuple(args.amplitude_modes),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        l2_weight=args.l2,
        val_fraction=args.val_fraction,
        seed=args.seed,
        device=args.device,
        log_every=args.log_every,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        phase, log_amplitude = model.correction_maps()
        gain = model.input_gain()
    phase_np = phase.detach().cpu().numpy().reshape(input_shape).astype(np.float32)
    log_amp_np = (
        log_amplitude.detach().cpu().numpy().reshape(input_shape).astype(np.float32)
    )
    amp_np = np.exp(log_amp_np).astype(np.float32)
    gain_np = gain.detach().cpu().numpy().astype(np.complex64)

    np.save(output_dir / "input_gain.npy", gain_np)
    np.save(output_dir / "phase_correction.npy", phase_np)
    np.save(output_dir / "amplitude_correction.npy", amp_np)
    np.save(output_dir / "history.npy", result.history)
    np.save(output_dir / "sample_indices.npy", sample_indices)
    np.save(output_dir / "output_indices.npy", output_indices)

    summary = {
        "profile": args.profile,
        "input_shape": list(input_shape),
        "tm_layout": args.tm_layout,
        "phase_modes": list(args.phase_modes),
        "amplitude_modes": list(args.amplitude_modes),
        "parameter_count": model.parameter_count,
        "calibration_probe_count": int(len(sample_indices)),
        "output_pixel_count": int(len(output_indices)),
        "baseline_train_pcc": result.baseline_train_pcc,
        "baseline_val_pcc": result.baseline_val_pcc,
        "final_train_pcc": result.final_train_pcc,
        "final_val_pcc": result.final_val_pcc,
        "best_val_pcc": result.best_val_pcc,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "l2_weight": args.l2,
        "seed": args.seed,
        "device": str(next(model.parameters()).device),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.export_tm:
        corrected_path = output_dir / "corrected_tm.npy"
        print("exporting corrected full TM to {}".format(corrected_path))
        export_corrected_tm(
            args.tm,
            gain_np,
            str(corrected_path),
            layout=args.tm_layout,
        )

    print(
        "done: validation PCC {:.5f} -> {:.5f}".format(
            result.baseline_val_pcc, result.final_val_pcc
        )
    )
    print("outputs: {}".format(output_dir.resolve()))


if __name__ == "__main__":
    main()
