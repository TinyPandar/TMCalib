"""Scalable GGS 2-1 transmission-matrix reconstruction for the 128 grid.

The full probe matrix has shape (M, 16384), so reconstruction is performed in
camera-output blocks.  Two reusable projection paths are supported:

* complex64 X plus a cached Cholesky factor;
* planar-complex32 X plus a precomputed planar-complex32 regularized inverse.

Both implement (X.H @ X + ridge * I)^-1 @ X.H @ Y.  The result is written to
a resumable complex64 NPY memmap.
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch

from low_precision_pinv import (
    LowPrecisionPinvFiles,
    PlanarComplexHalfMatrix,
    file_identity_matches,
    load_complex_numpy_as_planar_half,
    load_regularized_pinv_fp16,
)


ProgressCallback = Callable[[float, str], None]
StopCallback = Callable[[], bool]
IterationCallback = Callable[[int, float], None]


MEASUREMENT_NORMALIZATION_MAX = 255.0


@dataclass
class ReconstructionConfig:
    measurement_path: str
    probe_path: str
    output_path: str = "reconstructed_field_128_px4_active512.npy"
    error_curve_path: str = "ggs21_error_curve_128_px4_active512.npy"
    metadata_path: str = "tm_reconstruction_128_px4_active512.json"
    cholesky_cache_path: str = "probe_cholesky_128_px4_active512.npy"
    pinv_real_path: str = "probe_pinv_128_px4_active512_fp16_real.npy"
    pinv_imag_path: str = "probe_pinv_128_px4_active512_fp16_imag.npy"
    pinv_metadata_path: str = "probe_pinv_128_px4_active512_fp16.json"
    input_shape: Tuple[int, int] = (128, 128)
    output_shape: Tuple[int, int] = (128, 128)
    iterations: int = 200
    gs2_ratio: float = 0.89
    output_chunk_size: int = 512
    probe_transfer_rows: int = 128
    low_precision_transfer_rows: int = 256
    adjoint_row_chunk: int = 2048
    ridge: float = 1e-4
    solver: str = "cholesky"
    device: str = "auto"
    random_seed: int = 12804
    dark_level: float = 0.0
    measurements_are_intensity: bool = True
    normalize_measurements: bool = True
    normalize_tm: bool = True
    resume: bool = True
    output_start: int = 0
    output_count: Optional[int] = None


def _write_json_atomic(path: str, payload: Dict) -> None:
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def _file_identity(path: str) -> Dict:
    stat = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _choose_device(requested: str) -> torch.device:
    requested = str(requested).strip().lower()
    if requested not in ("", "auto"):
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot access CUDA")
        return device

    if not torch.cuda.is_available():
        return torch.device("cpu")

    # Initialize every CUDA context before comparing free memory; otherwise
    # the first queried device is unfairly penalized by its context allocation.
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            torch.empty(0, device="cuda:{}".format(index))

    candidates = []
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            free_bytes, _ = torch.cuda.mem_get_info()
            properties = torch.cuda.get_device_properties(index)
        throughput_score = int(properties.multi_processor_count)
        candidates.append((int(free_bytes), throughput_score, index))

    # The exact 128-grid path normally needs roughly 12 GiB. Prefer the
    # fastest device among GPUs with comfortable headroom, then fall back to
    # the device with the most free memory.
    roomy = [item for item in candidates if item[0] >= 16 * 2**30]
    if roomy:
        _, _, best_index = max(roomy, key=lambda item: (item[1], item[0]))
    else:
        _, _, best_index = max(candidates, key=lambda item: item[0])
    return torch.device("cuda:{}".format(best_index))


def _validate_config(config: ReconstructionConfig) -> Tuple[np.ndarray, int, int, int]:
    if config.iterations <= 0:
        raise ValueError("iterations must be positive")
    if not 0.0 <= config.gs2_ratio <= 1.0:
        raise ValueError("gs2_ratio must be in [0, 1]")
    if (
        config.output_chunk_size <= 0
        or config.probe_transfer_rows <= 0
        or config.low_precision_transfer_rows <= 0
        or config.adjoint_row_chunk <= 0
    ):
        raise ValueError("chunk sizes must be positive")
    if config.ridge < 0:
        raise ValueError("ridge must be non-negative")
    if not math.isfinite(config.dark_level):
        raise ValueError("dark_level must be finite")
    if config.solver not in ("cholesky", "complex32_pinv", "adjoint"):
        raise ValueError(
            "solver must be 'cholesky', 'complex32_pinv', or 'adjoint'"
        )
    if not os.path.isfile(config.probe_path):
        raise FileNotFoundError("Probe file not found: {}".format(config.probe_path))
    if not os.path.isfile(config.measurement_path):
        raise FileNotFoundError(
            "Measurement file not found: {}".format(config.measurement_path)
        )

    probes = np.load(config.probe_path, mmap_mode="r")
    if probes.dtype != np.complex64:
        raise ValueError("Probe dtype must be complex64, got {}".format(probes.dtype))
    expected_spatial = tuple(config.input_shape)
    if probes.ndim == 3:
        if tuple(probes.shape[1:]) != expected_spatial:
            raise ValueError(
                "Probe shape {} does not match input shape {}".format(
                    probes.shape, expected_spatial
                )
            )
        probe_matrix = probes.reshape(probes.shape[0], -1)
    elif probes.ndim == 2:
        input_count = int(np.prod(expected_spatial))
        if probes.shape[1] == input_count:
            probe_matrix = probes
        elif probes.shape[0] == input_count:
            probe_matrix = probes.T
        else:
            raise ValueError("Probe matrix has incompatible shape {}".format(probes.shape))
    else:
        raise ValueError("Probe array must be 2-D or 3-D")

    measurement_count, input_count = map(int, probe_matrix.shape)
    output_count = int(np.prod(config.output_shape))
    expected_bytes = measurement_count * output_count * np.dtype(np.uint16).itemsize
    actual_bytes = os.path.getsize(config.measurement_path)
    if actual_bytes != expected_bytes:
        raise ValueError(
            "Measurement size is {} bytes; expected {} bytes for shape ({}, {}) "
            "uint16".format(actual_bytes, expected_bytes, measurement_count, output_count)
        )

    start = int(config.output_start)
    if start < 0 or start >= output_count:
        raise ValueError("output_start is outside the camera output range")
    selected_count = output_count - start
    if config.output_count is not None:
        selected_count = min(selected_count, int(config.output_count))
    if selected_count <= 0:
        raise ValueError("output_count must select at least one camera pixel")
    return probe_matrix, measurement_count, input_count, selected_count


def _load_probes_to_device(
    probe_matrix: np.ndarray,
    device: torch.device,
    transfer_rows: int,
    progress: Optional[ProgressCallback] = None,
) -> torch.Tensor:
    measurement_count, input_count = map(int, probe_matrix.shape)
    if device.type == "cpu" and input_count > 4096:
        raise RuntimeError(
            "Full 128x128 reconstruction requires CUDA; CPU mode is only intended "
            "for small validation datasets"
        )

    x = torch.empty(
        (measurement_count, input_count), dtype=torch.complex64, device=device
    )
    scale = 1.0 / math.sqrt(float(measurement_count))
    for start in range(0, measurement_count, transfer_rows):
        stop = min(start + transfer_rows, measurement_count)
        host = np.array(probe_matrix[start:stop], dtype=np.complex64, copy=True)
        x[start:stop].copy_(torch.from_numpy(host).to(device=device), non_blocking=False)
        x[start:stop].mul_(scale)
        if progress and (stop == measurement_count or start == 0):
            progress(3.0 * stop / measurement_count, "Loading probes to {}".format(device))
    return x


def _cache_metadata_path(cache_path: str) -> str:
    return cache_path + ".json"


def _cholesky_cache_matches(
    config: ReconstructionConfig, input_count: int
) -> bool:
    cache_path = config.cholesky_cache_path
    metadata_path = _cache_metadata_path(cache_path)
    if not os.path.isfile(cache_path) or not os.path.isfile(metadata_path):
        return False
    try:
        factor = np.load(cache_path, mmap_mode="r")
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        return (
            factor.shape == (input_count, input_count)
            and factor.dtype == np.complex64
            and file_identity_matches(metadata.get("probe"), config.probe_path)
            and float(metadata.get("ridge")) == float(config.ridge)
            and metadata.get("normalization") == "X/sqrt(M)"
            and metadata.get("gram_tf32") is False
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _copy_numpy_matrix_to_device(
    matrix: np.ndarray, device: torch.device, row_chunk: int = 256
) -> torch.Tensor:
    result = torch.empty(matrix.shape, dtype=torch.complex64, device=device)
    for start in range(0, matrix.shape[0], row_chunk):
        stop = min(start + row_chunk, matrix.shape[0])
        host = np.array(matrix[start:stop], dtype=np.complex64, copy=True)
        result[start:stop].copy_(torch.from_numpy(host).to(device=device))
    return result


def _save_device_matrix(path: str, matrix: torch.Tensor, row_chunk: int = 256) -> None:
    partial = path + ".partial"
    output = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.complex64, shape=tuple(matrix.shape)
    )
    for start in range(0, matrix.shape[0], row_chunk):
        stop = min(start + row_chunk, matrix.shape[0])
        output[start:stop] = matrix[start:stop].detach().cpu().numpy()
    output.flush()
    del output
    os.replace(partial, path)


def _get_cholesky_factor(
    x: torch.Tensor,
    config: ReconstructionConfig,
    progress: Optional[ProgressCallback] = None,
) -> torch.Tensor:
    input_count = int(x.shape[1])
    if _cholesky_cache_matches(config, input_count):
        if progress:
            progress(4.0, "Loading cached probe Cholesky factor")
        cached = np.load(config.cholesky_cache_path, mmap_mode="r")
        return _copy_numpy_matrix_to_device(cached, x.device)

    if progress:
        progress(4.0, "Building probe Gram matrix (one-time step)")
    previous_tf32 = None
    if x.device.type == "cuda":
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        # Build the reusable inverse factor at full complex64 precision.
        # Iterative projections may use TF32 afterward for throughput.
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        gram = x.mH @ x
        gram.diagonal().add_(float(config.ridge))
        if progress:
            progress(6.0, "Computing probe Cholesky factor (one-time step)")
        factor = torch.linalg.cholesky(gram)
        del gram
    finally:
        if previous_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32

    cache_dir = os.path.dirname(os.path.abspath(config.cholesky_cache_path))
    os.makedirs(cache_dir, exist_ok=True)
    _save_device_matrix(config.cholesky_cache_path, factor)
    _write_json_atomic(
        _cache_metadata_path(config.cholesky_cache_path),
        {
            "probe": _file_identity(config.probe_path),
            "shape": [input_count, input_count],
            "dtype": "complex64",
            "ridge": float(config.ridge),
            "normalization": "X/sqrt(M)",
            "gram_tf32": False,
        },
    )
    return factor


def ggs21_block(
    x: Union[torch.Tensor, PlanarComplexHalfMatrix],
    amplitude: torch.Tensor,
    iterations: int,
    gs2_ratio: float,
    solver: str,
    cholesky_factor: Optional[torch.Tensor] = None,
    low_precision_pinv: Optional[PlanarComplexHalfMatrix] = None,
    random_seed: int = 0,
    adjoint_row_chunk: int = 2048,
    stop_requested: Optional[StopCallback] = None,
    iteration_callback: Optional[IterationCallback] = None,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Reconstruct a block of output rows from amplitude measurements."""
    if amplitude.ndim != 2 or amplitude.shape[0] != x.shape[0]:
        raise ValueError("amplitude must have shape (measurement_count, block_width)")
    if solver == "cholesky" and cholesky_factor is None:
        raise ValueError("cholesky_factor is required for the cholesky solver")
    if solver == "complex32_pinv" and low_precision_pinv is None:
        raise ValueError(
            "low_precision_pinv is required for the complex32_pinv solver"
        )
    if solver == "complex32_pinv" and low_precision_pinv.shape != (
        x.shape[1],
        x.shape[0],
    ):
        raise ValueError("low_precision_pinv shape is incompatible with x")

    generator = torch.Generator(device=x.device)
    generator.manual_seed(int(random_seed))
    random_phase = torch.rand(
        amplitude.shape,
        dtype=torch.float32,
        device=x.device,
        generator=generator,
    ).mul_(2.0 * math.pi)
    estimate_at_detector = torch.polar(amplitude, random_phase)
    del random_phase
    amplitude_squared = amplitude.square()
    amplitude_denominator = torch.linalg.vector_norm(
        amplitude, dim=0
    ).clamp_min_(1e-12)
    detector_scale = None
    amplitude_for_update = amplitude
    amplitude_squared_for_update = amplitude_squared
    if solver == "complex32_pinv":
        # Keep the whole GGS state in one scaled coordinate system.  This
        # prevents saturated uint16 intensities from overflowing FP16 and
        # avoids two full-column reductions and divisions on every iteration.
        detector_scale = amplitude_squared.amax(dim=0, keepdim=True).clamp_min_(1.0)
        estimate_at_detector.div_(detector_scale)
        amplitude_for_update = amplitude / detector_scale
        amplitude_squared.div_(detector_scale)
        amplitude_squared_for_update = amplitude_squared
    switch_iteration = int(round(gs2_ratio * iterations))
    errors = np.empty(iterations, dtype=np.float32)

    h_t = None
    for iteration in range(iterations):
        if stop_requested and stop_requested():
            raise InterruptedError("TM reconstruction stopped")
        if solver == "complex32_pinv":
            # The inverse was constructed once in complex64, then quantized.
            # Applying it directly replaces both X.H @ Y and cholesky_solve.
            h_t = low_precision_pinv.matmul(
                estimate_at_detector, normalize_rhs=False
            )
        else:
            # Resolving x.mH in one large CUDA matmul can materialize an 8 GiB
            # conjugated copy of the full probe matrix. Accumulating by probe
            # rows has the same mathematical result and bounds that temporary.
            rhs = torch.zeros(
                (x.shape[1], amplitude.shape[1]),
                dtype=torch.complex64,
                device=x.device,
            )
            for row_start in range(0, x.shape[0], adjoint_row_chunk):
                row_stop = min(row_start + adjoint_row_chunk, x.shape[0])
                rhs.add_(
                    x[row_start:row_stop].mH
                    @ estimate_at_detector[row_start:row_stop]
                )
            if solver == "cholesky":
                h_t = torch.cholesky_solve(rhs, cholesky_factor)
            else:
                h_t = rhs
        if isinstance(x, PlanarComplexHalfMatrix):
            propagated = x.matmul(h_t, normalize_rhs=False)
        else:
            propagated = x @ h_t
        if detector_scale is not None:
            # Positive column scaling preserves phase, so restoring the scale
            # in-place also avoids a block-sized complex temporary.
            propagated.mul_(detector_scale)
        residual = torch.abs(propagated)
        residual.sub_(amplitude)
        relative_error = (
            torch.linalg.vector_norm(residual, dim=0) / amplitude_denominator
        )
        errors[iteration] = float(relative_error.mean().item())
        if iteration_callback:
            iteration_callback(iteration + 1, float(errors[iteration]))
        phase_magnitude = torch.abs(propagated).clamp_min_(1e-12)
        propagated.div_(phase_magnitude)
        if iteration < switch_iteration:
            propagated.mul_(amplitude_squared_for_update)
        else:
            propagated.mul_(amplitude_for_update)
        estimate_at_detector = propagated

    if h_t is None:
        raise RuntimeError("No reconstruction iteration was executed")
    if detector_scale is not None:
        h_t.mul_(detector_scale)
    return h_t.T.contiguous(), errors


def _streaming_complex_rms(matrix: np.ndarray, row_chunk: int = 256) -> float:
    power_sum = 0.0
    item_count = 0
    for start in range(0, matrix.shape[0], row_chunk):
        stop = min(start + row_chunk, matrix.shape[0])
        block = np.asarray(matrix[start:stop])
        power_sum += float(np.sum(np.abs(block) ** 2, dtype=np.float64))
        item_count += int(block.size)
    return math.sqrt(power_sum / max(1, item_count))


def _streaming_value_range(
    matrix: np.ndarray, row_chunk: int = 2048
) -> Tuple[float, float]:
    """Return a large array's global value range without materializing it."""
    if matrix.size == 0:
        raise ValueError("Cannot determine the range of an empty measurement array")
    minimum = math.inf
    maximum = -math.inf
    for start in range(0, matrix.shape[0], row_chunk):
        stop = min(start + row_chunk, matrix.shape[0])
        block = np.asarray(matrix[start:stop])
        minimum = min(minimum, float(np.min(block)))
        maximum = max(maximum, float(np.max(block)))
    return minimum, maximum


def _reconstruction_identity(
    config: ReconstructionConfig, selection_start: int, selection_stop: int
) -> Dict:
    identity = {
        "measurement": _file_identity(config.measurement_path),
        "probe": _file_identity(config.probe_path),
        "selected_range": [selection_start, selection_stop],
        "iterations": config.iterations,
        "solver": config.solver,
        "ridge": config.ridge,
        "measurement_preprocessing": {
            "dark_level": config.dark_level,
            "measurements_are_intensity": config.measurements_are_intensity,
            "normalize_measurements": config.normalize_measurements,
            "normalization_max": MEASUREMENT_NORMALIZATION_MAX,
        },
    }
    if config.solver == "complex32_pinv":
        identity["low_precision_pinv"] = {
            "real": _file_identity(config.pinv_real_path),
            "imag": _file_identity(config.pinv_imag_path),
            "metadata": _file_identity(config.pinv_metadata_path),
        }
    return identity


def reconstruct_tm(
    config: ReconstructionConfig,
    progress: Optional[ProgressCallback] = None,
    stop_requested: Optional[StopCallback] = None,
) -> Dict:
    """Run resumable, blockwise TM reconstruction and return output metadata."""
    started = time.perf_counter()
    config.measurement_path = os.path.abspath(config.measurement_path)
    config.probe_path = os.path.abspath(config.probe_path)
    config.output_path = os.path.abspath(config.output_path)
    config.error_curve_path = os.path.abspath(config.error_curve_path)
    config.metadata_path = os.path.abspath(config.metadata_path)
    config.cholesky_cache_path = os.path.abspath(config.cholesky_cache_path)
    config.pinv_real_path = os.path.abspath(config.pinv_real_path)
    config.pinv_imag_path = os.path.abspath(config.pinv_imag_path)
    config.pinv_metadata_path = os.path.abspath(config.pinv_metadata_path)

    probe_matrix, measurement_count, input_count, selected_count = _validate_config(config)
    camera_output_count = int(np.prod(config.output_shape))
    selection_start = int(config.output_start)
    selection_stop = selection_start + selected_count
    device = _choose_device(config.device)
    if progress:
        progress(0.0, "Preparing TM reconstruction on {}".format(device))

    factor = None
    low_precision_pinv = None
    low_precision_metadata = None
    if config.solver == "complex32_pinv":
        if device.type != "cuda":
            raise RuntimeError("complex32_pinv requires a CUDA device")
        inverse_files = LowPrecisionPinvFiles(
            real_path=config.pinv_real_path,
            imag_path=config.pinv_imag_path,
            metadata_path=config.pinv_metadata_path,
        )

        def show_inverse_load(percent: float, message: str) -> None:
            if progress:
                progress(0.04 * percent, message)

        low_precision_pinv, low_precision_metadata = load_regularized_pinv_fp16(
            inverse_files,
            probe_path=config.probe_path,
            expected_shape=(input_count, measurement_count),
            ridge=config.ridge,
            device=device,
            row_chunk=config.low_precision_transfer_rows,
            progress=show_inverse_load,
        )

        def show_probe_load(percent: float, message: str) -> None:
            if progress:
                progress(4.0 + 0.03 * percent, message)

        x = load_complex_numpy_as_planar_half(
            probe_matrix,
            device=device,
            row_chunk=config.low_precision_transfer_rows,
            scale=1.0 / math.sqrt(float(measurement_count)),
            progress=show_probe_load,
        )
    else:
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
        x = _load_probes_to_device(
            probe_matrix, device, config.probe_transfer_rows, progress=progress
        )
        if config.solver == "cholesky":
            factor = _get_cholesky_factor(x, config, progress=progress)
    del probe_matrix
    if device.type == "cuda":
        torch.cuda.empty_cache()

    measurements = np.memmap(
        config.measurement_path,
        dtype=np.uint16,
        mode="r",
        shape=(measurement_count, camera_output_count),
    )
    if progress:
        progress(7.0, "Scanning camera measurement intensity range")
    measurement_raw_min, measurement_raw_max = _streaming_value_range(measurements)
    corrected_min = max(measurement_raw_min - float(config.dark_level), 0.0)
    corrected_max = max(measurement_raw_max - float(config.dark_level), 0.0)
    measurement_scale = 1.0
    if config.normalize_measurements and corrected_max > 0.0:
        measurement_scale = MEASUREMENT_NORMALIZATION_MAX / corrected_max
    processed_min = corrected_min * measurement_scale
    processed_max = corrected_max * measurement_scale
    if config.normalize_measurements:
        processed_min = min(processed_min, MEASUREMENT_NORMALIZATION_MAX)
        processed_max = (
            MEASUREMENT_NORMALIZATION_MAX if corrected_max > 0.0 else 0.0
        )
    if progress:
        progress(
            7.5,
            "Camera intensity {:.0f}..{:.0f} -> {:.3g}..{:.3g} (float32)".format(
                measurement_raw_min,
                measurement_raw_max,
                processed_min,
                processed_max,
            ),
        )

    partial_path = config.output_path + ".partial"
    progress_path = config.output_path + ".progress.json"
    completed = 0
    error_sum = np.zeros(config.iterations, dtype=np.float64)
    error_weight = 0
    identity = _reconstruction_identity(config, selection_start, selection_stop)
    if config.resume and os.path.isfile(partial_path) and os.path.isfile(progress_path):
        with open(progress_path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("identity") != identity:
            raise ValueError(
                "Existing partial reconstruction does not match this configuration; "
                "remove {} and {} to start over".format(partial_path, progress_path)
            )
        output = np.load(partial_path, mmap_mode="r+")
        if output.shape != (selected_count, input_count) or output.dtype != np.complex64:
            raise ValueError("Existing partial TM has an incompatible shape or dtype")
        completed = int(state.get("completed", 0))
        saved_error = state.get("error_sum", [])
        if len(saved_error) == config.iterations:
            error_sum[:] = saved_error
        error_weight = int(state.get("error_weight", completed))
    else:
        output = np.lib.format.open_memmap(
            partial_path,
            mode="w+",
            dtype=np.complex64,
            shape=(selected_count, input_count),
        )

    reconstruction_started = time.perf_counter()
    with torch.inference_mode():
        for local_start in range(completed, selected_count, config.output_chunk_size):
            if stop_requested and stop_requested():
                raise InterruptedError("TM reconstruction stopped")
            local_stop = min(local_start + config.output_chunk_size, selected_count)
            source_start = selection_start + local_start
            source_stop = selection_start + local_stop
            measured = np.array(
                measurements[:, source_start:source_stop], dtype=np.float32, copy=True
            )
            if config.dark_level:
                measured -= float(config.dark_level)
            np.maximum(measured, 0.0, out=measured)
            if config.normalize_measurements:
                measured *= measurement_scale
                np.minimum(
                    measured, MEASUREMENT_NORMALIZATION_MAX, out=measured
                )
            if config.measurements_are_intensity:
                np.sqrt(measured, out=measured)
            amplitude = torch.from_numpy(measured).to(device=device)

            def show_iteration(iteration: int, relative_error: float) -> None:
                if not progress:
                    return
                if iteration != config.iterations and iteration % 5:
                    return
                fractional_block = iteration / config.iterations
                done = local_start + (local_stop - local_start) * fractional_block
                percent = 8.0 + 90.0 * done / selected_count
                progress(
                    percent,
                    "GGS21: pixels {}:{}, iteration {}/{}, error {:.4f}".format(
                        source_start,
                        source_stop,
                        iteration,
                        config.iterations,
                        relative_error,
                    ),
                )

            h_block, block_error = ggs21_block(
                x=x,
                amplitude=amplitude,
                iterations=config.iterations,
                gs2_ratio=config.gs2_ratio,
                solver=config.solver,
                cholesky_factor=factor,
                low_precision_pinv=low_precision_pinv,
                random_seed=config.random_seed + source_start,
                adjoint_row_chunk=config.adjoint_row_chunk,
                stop_requested=stop_requested,
                iteration_callback=show_iteration,
            )
            invalid = ~torch.isfinite(h_block)
            if torch.any(invalid):
                raise FloatingPointError(
                    "Reconstructed block {}:{} contains NaN or infinity".format(
                        source_start, source_stop
                    )
                )
            output[local_start:local_stop] = h_block.cpu().numpy()
            output.flush()
            block_width = local_stop - local_start
            error_sum += block_error.astype(np.float64) * block_width
            error_weight += block_width
            completed = local_stop
            _write_json_atomic(
                progress_path,
                {
                    "status": "running",
                    "identity": identity,
                    "completed": completed,
                    "total": selected_count,
                    "error_sum": error_sum.tolist(),
                    "error_weight": error_weight,
                },
            )
            if progress:
                percent = 8.0 + 90.0 * completed / selected_count
                progress(
                    percent,
                    "GGS21: {}/{} camera pixels".format(completed, selected_count),
                )
            del measured, amplitude, h_block
    reconstruction_finished = time.perf_counter()

    rms_before_normalization = _streaming_complex_rms(output)
    if config.normalize_tm and rms_before_normalization > 0:
        for start in range(0, selected_count, 256):
            stop = min(start + 256, selected_count)
            output[start:stop] /= rms_before_normalization
        output.flush()
    del output

    error_curve = (error_sum / max(1, error_weight)).astype(np.float32)
    np.save(config.error_curve_path, error_curve)
    os.replace(partial_path, config.output_path)
    if os.path.exists(progress_path):
        os.remove(progress_path)

    elapsed = time.perf_counter() - started
    result = {
        "status": "complete",
        "measurement": _file_identity(config.measurement_path),
        "probe": _file_identity(config.probe_path),
        "output_path": config.output_path,
        "output_shape": [selected_count, input_count],
        "camera_output_shape": list(config.output_shape),
        "selected_output_range": [selection_start, selection_stop],
        "dtype": "complex64",
        "solver": config.solver,
        "device": str(device),
        "iterations": config.iterations,
        "gs2_ratio": config.gs2_ratio,
        "ridge": config.ridge,
        "measurement_preprocessing": {
            "storage_dtype": "uint16",
            "working_intensity_dtype": "float32",
            "raw_range": [measurement_raw_min, measurement_raw_max],
            "dark_level": float(config.dark_level),
            "normalize_measurements": bool(config.normalize_measurements),
            "normalization_max": MEASUREMENT_NORMALIZATION_MAX,
            "normalization_scale": measurement_scale,
            "processed_intensity_range": [processed_min, processed_max],
            "measurements_are_intensity": bool(
                config.measurements_are_intensity
            ),
            "amplitude_dtype": "float32",
        },
        "dtype_pipeline": {
            "measurement_storage": "uint16",
            "measurement_working_intensity": "float32",
            "detector_amplitude": "torch.float32",
            "probe_storage": "complex64",
            "probe_compute": (
                "planar torch.float16"
                if config.solver == "complex32_pinv"
                else "torch.complex64"
            ),
            "phase_retrieval_complex": "torch.complex64",
            "error_accumulator": "float64",
            "saved_error_curve": "float32",
            "reconstruction_output": "complex64",
        },
        "normalized": bool(config.normalize_tm),
        "rms_before_normalization": rms_before_normalization,
        "initial_relative_amplitude_error": float(error_curve[0]),
        "final_relative_amplitude_error": float(error_curve[-1]),
        "elapsed_seconds": elapsed,
        "setup_seconds": reconstruction_started - started,
        "ggs_seconds": reconstruction_finished - reconstruction_started,
        "finalization_seconds": elapsed - (reconstruction_finished - started),
        "config": asdict(config),
    }
    if low_precision_metadata is not None:
        result["low_precision_pinv"] = {
            "representation": low_precision_metadata["representation"],
            "logical_shape": low_precision_metadata["logical_shape"],
            "dtype_per_plane": low_precision_metadata["dtype_per_plane"],
            "real_path": config.pinv_real_path,
            "imag_path": config.pinv_imag_path,
            "metadata_path": config.pinv_metadata_path,
            "storage_bytes": low_precision_pinv.storage_bytes,
            "probe_storage_bytes": x.storage_bytes,
        }
    _write_json_atomic(config.metadata_path, result)
    if progress:
        progress(100.0, "TM reconstruction complete")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--measurement", default="measurements_128_px4_active512_full_memmap.npy"
    )
    parser.add_argument(
        "--probes",
        default=os.path.join(
            "pregenerated_patterns_128_px4_active512_full", "probe.npy"
        ),
    )
    parser.add_argument(
        "--output", default="reconstructed_field_128_px4_active512.npy"
    )
    parser.add_argument(
        "--error-curve", default="ggs21_error_curve_128_px4_active512.npy"
    )
    parser.add_argument(
        "--metadata", default="tm_reconstruction_128_px4_active512.json"
    )
    parser.add_argument(
        "--cholesky-cache", default="probe_cholesky_128_px4_active512.npy"
    )
    parser.add_argument(
        "--pinv-real", default="probe_pinv_128_px4_active512_fp16_real.npy"
    )
    parser.add_argument(
        "--pinv-imag", default="probe_pinv_128_px4_active512_fp16_imag.npy"
    )
    parser.add_argument(
        "--pinv-metadata", default="probe_pinv_128_px4_active512_fp16.json"
    )
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--gs2-ratio", type=float, default=0.89)
    parser.add_argument("--output-chunk-size", type=int, default=512)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument(
        "--solver",
        choices=("cholesky", "complex32_pinv", "adjoint"),
        default="cholesky",
    )
    parser.add_argument("--low-precision-transfer-rows", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dark-level", type=float, default=0.0)
    parser.add_argument("--output-start", type=int, default=0)
    parser.add_argument("--output-count", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-normalize-measurements", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = ReconstructionConfig(
        measurement_path=args.measurement,
        probe_path=args.probes,
        output_path=args.output,
        error_curve_path=args.error_curve,
        metadata_path=args.metadata,
        cholesky_cache_path=args.cholesky_cache,
        pinv_real_path=args.pinv_real,
        pinv_imag_path=args.pinv_imag,
        pinv_metadata_path=args.pinv_metadata,
        iterations=args.iterations,
        gs2_ratio=args.gs2_ratio,
        output_chunk_size=args.output_chunk_size,
        low_precision_transfer_rows=args.low_precision_transfer_rows,
        ridge=args.ridge,
        solver=args.solver,
        device=args.device,
        dark_level=args.dark_level,
        output_start=args.output_start,
        output_count=args.output_count,
        resume=not args.no_resume,
        normalize_measurements=not args.no_normalize_measurements,
        normalize_tm=not args.no_normalize,
    )

    def show_progress(percent: float, message: str) -> None:
        print("[{:.1f}%] {}".format(percent, message), flush=True)

    result = reconstruct_tm(config, progress=show_progress)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
