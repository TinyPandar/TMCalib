"""Build a planar-FP16 regularized pseudoinverse for the 128-grid GGS path.

The script reuses an existing complex64 Cholesky cache.  It never loads the
full complex64 probe onto CUDA: probe rows are streamed from the NPY memmap,
solved in complex64, transposed, and immediately written as FP16 real/imag
planes.  Default output size for the 8N dataset is 8 GiB total.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from low_precision_pinv import (
    LowPrecisionPinvFiles,
    export_regularized_pinv_fp16,
)


def choose_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    candidates = []
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            torch.empty(0, device="cuda:{}".format(index))
            free, _ = torch.cuda.mem_get_info()
            properties = torch.cuda.get_device_properties(index)
        candidates.append((properties.multi_processor_count, int(free), index))
    return torch.device("cuda:{}".format(max(candidates)[2]))


def load_complex64_matrix(matrix, device, row_chunk):
    if matrix.ndim != 2 or matrix.dtype != np.complex64:
        raise ValueError("Expected a 2-D complex64 matrix")
    result = torch.empty(matrix.shape, dtype=torch.complex64, device=device)
    for start in range(0, matrix.shape[0], row_chunk):
        stop = min(start + row_chunk, matrix.shape[0])
        block = np.array(matrix[start:stop], dtype=np.complex64, copy=True)
        result[start:stop].copy_(torch.from_numpy(block))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe",
        default=os.path.join(
            "pregenerated_patterns_128_px4_active512_8N_full", "probe.npy"
        ),
    )
    parser.add_argument(
        "--cholesky", default="probe_cholesky_128_px4_active512_8N.npy"
    )
    parser.add_argument(
        "--real-output", default="probe_pinv_128_px4_active512_8N_fp16_real.npy"
    )
    parser.add_argument(
        "--imag-output", default="probe_pinv_128_px4_active512_8N_fp16_imag.npy"
    )
    parser.add_argument(
        "--metadata", default="probe_pinv_128_px4_active512_8N_fp16.json"
    )
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--measurement-chunk", type=int, default=512)
    parser.add_argument("--factor-transfer-rows", type=int, default=256)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    probe_path = os.path.abspath(args.probe)
    factor_path = os.path.abspath(args.cholesky)
    if not os.path.isfile(probe_path):
        raise FileNotFoundError(probe_path)
    if not os.path.isfile(factor_path):
        raise FileNotFoundError(factor_path)
    probes = np.load(probe_path, mmap_mode="r", allow_pickle=False)
    probe_matrix = probes.reshape(probes.shape[0], -1)
    factor_host = np.load(factor_path, mmap_mode="r", allow_pickle=False)
    if factor_host.shape != (probe_matrix.shape[1], probe_matrix.shape[1]):
        raise ValueError("Cholesky factor shape does not match the probe")

    factor_metadata_path = factor_path + ".json"
    if os.path.isfile(factor_metadata_path):
        with open(factor_metadata_path, "r", encoding="utf-8") as handle:
            factor_metadata = json.load(handle)
        if float(factor_metadata.get("ridge", -1)) != float(args.ridge):
            raise ValueError("Requested ridge does not match the Cholesky cache")
        if factor_metadata.get("normalization") != "X/sqrt(M)":
            raise ValueError("Cholesky cache does not use X/sqrt(M) normalization")
        cached_probe = factor_metadata.get("probe", {})
        probe_stat = os.stat(probe_path)
        if (
            int(cached_probe.get("size", -1)) != int(probe_stat.st_size)
            or int(cached_probe.get("mtime_ns", -1)) != int(probe_stat.st_mtime_ns)
            or os.path.normcase(os.path.abspath(cached_probe.get("path", "")))
            != os.path.normcase(probe_path)
        ):
            raise ValueError("Cholesky cache was built from a different probe file")

    device = choose_device(args.device)
    print("Loading Cholesky factor to {}...".format(device), flush=True)
    factor = load_complex64_matrix(
        factor_host, device=device, row_chunk=args.factor_transfer_rows
    )
    files = LowPrecisionPinvFiles(
        real_path=os.path.abspath(args.real_output),
        imag_path=os.path.abspath(args.imag_output),
        metadata_path=os.path.abspath(args.metadata),
    )
    started = time.perf_counter()

    def show_progress(percent, message):
        print("{:6.2f}% {}".format(percent, message), flush=True)

    result = export_regularized_pinv_fp16(
        probe_matrix,
        factor,
        files,
        probe_path=probe_path,
        ridge=args.ridge,
        normalized_probe=True,
        measurement_chunk=args.measurement_chunk,
        progress=show_progress,
    )
    result["elapsed_seconds"] = time.perf_counter() - started
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
