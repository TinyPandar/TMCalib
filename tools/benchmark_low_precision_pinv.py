"""Benchmark planar-FP16 pseudoinverse GGS on a real measurement block."""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from low_precision_pinv import PlanarComplexHalfMatrix


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe", default=os.path.join("pregenerated_patterns_8N", "probe.npy")
    )
    parser.add_argument("--measurements", default="measurements_memmap.npy")
    parser.add_argument("--output-count", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--ratio", type=float, default=0.89)
    parser.add_argument("--seed", type=int, default=24032)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def run_ggs(x, inverse, amplitude, iterations, ratio, initial, low_precision):
    if low_precision:
        x_operator = PlanarComplexHalfMatrix.from_complex(x)
        inverse_operator = PlanarComplexHalfMatrix.from_complex(inverse)
    detector = initial.clone()
    amplitude_squared = amplitude.square()
    # One fixed scale per output pixel keeps both GS-2 and GS-1 constraints in
    # FP16 range.  Positive column scaling does not change recovered phase.
    column_scale = amplitude_squared.amax(dim=0, keepdim=True).clamp_min_(1.0)
    if low_precision:
        detector.div_(column_scale)
        amplitude_for_update = amplitude / column_scale
        amplitude_squared_for_update = amplitude_squared / column_scale
    else:
        amplitude_for_update = amplitude
        amplitude_squared_for_update = amplitude_squared
    switch = round(ratio * iterations)
    errors = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    h = None
    for iteration in range(iterations):
        if low_precision:
            h = inverse_operator.matmul(detector, normalize_rhs=False)
            propagated = x_operator.matmul(h, normalize_rhs=False)
        else:
            h = inverse @ detector
            propagated = x @ h
        propagated_for_error = (
            propagated * column_scale if low_precision else propagated
        )
        residual = torch.abs(propagated_for_error) - amplitude
        denominator = torch.linalg.vector_norm(amplitude, dim=0).clamp_min_(1e-12)
        errors.append(
            float(
                (
                    torch.linalg.vector_norm(residual, dim=0) / denominator
                ).mean().item()
            )
        )
        phase = torch.polar(torch.ones_like(amplitude), torch.angle(propagated))
        detector = (
            amplitude_squared_for_update
            if iteration < switch
            else amplitude_for_update
        ) * phase
    torch.cuda.synchronize()
    if low_precision:
        h = h * column_scale
    return time.perf_counter() - started, h, np.asarray(errors, dtype=np.float32)


def main():
    args = parse_args()
    probes = np.load(args.probe, mmap_mode="r")
    matrix = probes.reshape(probes.shape[0], -1)
    measurement_count, input_count = map(int, matrix.shape)
    output_count = int(args.output_count)
    measurement_size = os.path.getsize(args.measurements)
    camera_output_count = measurement_size // (
        measurement_count * np.dtype(np.uint16).itemsize
    )
    measurements = np.memmap(
        args.measurements,
        dtype=np.uint16,
        mode="r",
        shape=(measurement_count, camera_output_count),
    )
    measured = np.array(
        measurements[:, :output_count], dtype=np.float32, copy=True
    )
    np.sqrt(measured, out=measured)
    device = torch.device(args.device)
    x = torch.from_numpy(np.array(matrix, dtype=np.complex64, copy=True)).to(device)
    amplitude = torch.from_numpy(measured).to(device)
    inverse = torch.linalg.pinv(x)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    phase = torch.rand(
        amplitude.shape,
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).mul_(2.0 * math.pi)
    initial = torch.polar(amplitude, phase)

    baseline_time, baseline_h, baseline_error = run_ggs(
        x,
        inverse,
        amplitude,
        args.iterations,
        args.ratio,
        initial,
        low_precision=False,
    )
    low_time, low_h, low_error = run_ggs(
        x,
        inverse,
        amplitude,
        args.iterations,
        args.ratio,
        initial,
        low_precision=True,
    )
    inner = torch.sum(torch.conj(baseline_h) * low_h, dim=0)
    denominator = (
        torch.linalg.vector_norm(baseline_h, dim=0)
        * torch.linalg.vector_norm(low_h, dim=0)
    ).clamp_min_(1e-20)
    correlation = torch.abs(inner) / denominator
    report = {
        "gpu": torch.cuda.get_device_name(device),
        "probe_shape": [measurement_count, input_count],
        "output_count": output_count,
        "iterations": args.iterations,
        "complex64_seconds": baseline_time,
        "planar_fp16_seconds": low_time,
        "speedup": baseline_time / low_time,
        "complex64_final_error": float(baseline_error[-1]),
        "planar_fp16_final_error": float(low_error[-1]),
        "final_error_delta": float(low_error[-1] - baseline_error[-1]),
        "phase_invariant_correlation": {
            "mean": float(torch.mean(correlation).item()),
            "median": float(torch.median(correlation).item()),
            "p05": float(torch.quantile(correlation, 0.05).item()),
            "minimum": float(torch.min(correlation).item()),
            "outputs_below_0_99": int(torch.count_nonzero(correlation < 0.99).item()),
            "outputs_below_0_90": int(torch.count_nonzero(correlation < 0.90).item()),
        },
        "finite": bool(torch.isfinite(low_h).all().item()),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
