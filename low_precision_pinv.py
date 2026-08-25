"""Mixed-precision complex pseudoinverse support for GGS reconstruction.

CUDA's SVD routines do not accept complex FP16 inputs.  This module therefore
separates *building* the inverse from repeatedly *applying* it:

1. Build the regularized inverse in complex64 from a cached Cholesky factor.
2. Stream its transpose to two float16 NPY planes (real and imaginary).
3. Keep those planes on CUDA and apply them through four real FP16 Tensor Core
   GEMMs, returning a complex64 result for the phase-retrieval operations.

For ridge == 0 and a full-column-rank probe matrix this is the ordinary left
pseudoinverse.  For ridge > 0 it is the Tikhonov-regularized inverse used by
the scalable 128-grid reconstruction.
"""

import json
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch


ProgressCallback = Callable[[float, str], None]


@dataclass
class LowPrecisionPinvFiles:
    real_path: str
    imag_path: str
    metadata_path: str


def _file_identity(path: str) -> Dict:
    stat = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def file_identity_matches(expected: Dict, path: str) -> bool:
    """Compare a recorded identity using host filesystem path semantics.

    Windows paths are case-insensitive, but JSON strings are not.  Normalizing
    both paths prevents ``C:\\...`` and ``c:\\...`` from being treated as
    different probe files while size and nanosecond mtime still protect
    against using an inverse built from changed data.
    """
    if not isinstance(expected, dict):
        return False
    try:
        actual = _file_identity(path)
        expected_path = os.path.normcase(
            os.path.realpath(os.path.abspath(str(expected["path"])))
        )
        actual_path = os.path.normcase(
            os.path.realpath(os.path.abspath(actual["path"]))
        )
        return (
            expected_path == actual_path
            and int(expected["size"]) == actual["size"]
            and int(expected["mtime_ns"]) == actual["mtime_ns"]
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _write_json_atomic(path: str, payload: Dict) -> None:
    temporary = path + ".partial"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


class PlanarComplexHalfMatrix:
    """A complex matrix stored as two CUDA float16 planes.

    PyTorch 2.4 does not implement ComplexHalf GEMM.  Four real half-precision
    GEMMs are used instead.  NVIDIA Tensor Cores accumulate the products in
    higher precision; the individual GEMM outputs are FP16 and are immediately
    promoted to FP32 before the complex additions/subtractions.

    ``storage_is_transposed`` lets a pseudoinverse be stored on disk as K.T,
    so it can be generated and loaded in contiguous measurement-row chunks.
    """

    def __init__(
        self,
        real: torch.Tensor,
        imag: torch.Tensor,
        storage_is_transposed: bool = False,
    ):
        if real.shape != imag.shape or real.ndim != 2:
            raise ValueError("real and imag must be same-shape 2-D tensors")
        if real.dtype != torch.float16 or imag.dtype != torch.float16:
            raise TypeError("real and imag must have dtype torch.float16")
        if not real.is_cuda or not imag.is_cuda or real.device != imag.device:
            raise ValueError("real and imag must be on the same CUDA device")
        self.real = real.contiguous()
        self.imag = imag.contiguous()
        self.storage_is_transposed = bool(storage_is_transposed)

    @classmethod
    def from_complex(
        cls,
        matrix: torch.Tensor,
        storage_is_transposed: bool = False,
    ) -> "PlanarComplexHalfMatrix":
        if matrix.ndim != 2 or matrix.dtype != torch.complex64 or not matrix.is_cuda:
            raise ValueError("matrix must be a 2-D CUDA complex64 tensor")
        return cls(
            matrix.real.to(torch.float16),
            matrix.imag.to(torch.float16),
            storage_is_transposed=storage_is_transposed,
        )

    @property
    def device(self) -> torch.device:
        return self.real.device

    @property
    def shape(self) -> Tuple[int, int]:
        if self.storage_is_transposed:
            return int(self.real.shape[1]), int(self.real.shape[0])
        return int(self.real.shape[0]), int(self.real.shape[1])

    @property
    def storage_bytes(self) -> int:
        return (
            self.real.numel() * self.real.element_size()
            + self.imag.numel() * self.imag.element_size()
        )

    @property
    def complex64_bytes(self) -> int:
        rows, columns = self.shape
        return rows * columns * torch.empty((), dtype=torch.complex64).element_size()

    def _logical_planes(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.storage_is_transposed:
            return self.real.mT, self.imag.mT
        return self.real, self.imag

    def matmul(
        self,
        right: torch.Tensor,
        normalize_rhs: bool = True,
    ) -> torch.Tensor:
        """Return this_matrix @ right as complex64.

        Per-column normalization prevents GS-2 intensity values near the
        uint16 limit from overflowing when converted to FP16.  The scale is
        restored only after the Tensor Core GEMMs, in float32.
        """
        rows, inner = self.shape
        if right.ndim != 2 or int(right.shape[0]) != inner:
            raise ValueError("right must have shape ({}, block_width)".format(inner))
        if right.dtype != torch.complex64 or right.device != self.device:
            raise ValueError("right must be CUDA complex64 on {}".format(self.device))

        if normalize_rhs:
            scale = torch.maximum(
                torch.amax(torch.abs(right.real), dim=0, keepdim=True),
                torch.amax(torch.abs(right.imag), dim=0, keepdim=True),
            ).clamp_min_(1.0)
            right_real = (right.real / scale).to(torch.float16)
            right_imag = (right.imag / scale).to(torch.float16)
        else:
            scale = None
            right_real = right.real.to(torch.float16)
            right_imag = right.imag.to(torch.float16)

        matrix_real, matrix_imag = self._logical_planes()
        result_real = torch.mm(matrix_real, right_real).float()
        result_real.sub_(torch.mm(matrix_imag, right_imag).float())
        result_imag = torch.mm(matrix_real, right_imag).float()
        result_imag.add_(torch.mm(matrix_imag, right_real).float())
        if scale is not None:
            result_real.mul_(scale)
            result_imag.mul_(scale)
        return torch.complex(result_real, result_imag)

    def adjoint_matmul(
        self,
        right: torch.Tensor,
        normalize_rhs: bool = True,
    ) -> torch.Tensor:
        """Return ``this_matrix.H @ right`` as complex64.

        This is the low-memory counterpart of :meth:`matmul`.  It lets the
        Cholesky GGS path keep a large probe matrix in two FP16 planes while
        retaining the complex64 factor and phase-retrieval state.
        """
        rows, columns = self.shape
        if right.ndim != 2 or int(right.shape[0]) != rows:
            raise ValueError("right must have shape ({}, block_width)".format(rows))
        if right.dtype != torch.complex64 or right.device != self.device:
            raise ValueError("right must be CUDA complex64 on {}".format(self.device))

        if normalize_rhs:
            scale = torch.maximum(
                torch.amax(torch.abs(right.real), dim=0, keepdim=True),
                torch.amax(torch.abs(right.imag), dim=0, keepdim=True),
            ).clamp_min_(1.0)
            right_real = (right.real / scale).to(torch.float16)
            right_imag = (right.imag / scale).to(torch.float16)
        else:
            scale = None
            right_real = right.real.to(torch.float16)
            right_imag = right.imag.to(torch.float16)

        matrix_real, matrix_imag = self._logical_planes()
        result_real = torch.mm(matrix_real.mT, right_real).float()
        result_real.add_(torch.mm(matrix_imag.mT, right_imag).float())
        result_imag = torch.mm(matrix_real.mT, right_imag).float()
        result_imag.sub_(torch.mm(matrix_imag.mT, right_real).float())
        if scale is not None:
            result_real.mul_(scale)
            result_imag.mul_(scale)
        return torch.complex(result_real, result_imag)


def load_planar_complex_half(
    real_path: str,
    imag_path: str,
    device: Union[str, torch.device],
    row_chunk: int = 256,
    storage_is_transposed: bool = False,
    progress: Optional[ProgressCallback] = None,
) -> PlanarComplexHalfMatrix:
    """Stream two float16 NPY planes to CUDA without a large host copy."""
    if row_chunk <= 0:
        raise ValueError("row_chunk must be positive")
    real_host = np.load(real_path, mmap_mode="r", allow_pickle=False)
    imag_host = np.load(imag_path, mmap_mode="r", allow_pickle=False)
    if real_host.shape != imag_host.shape or real_host.ndim != 2:
        raise ValueError("real and imag files must contain same-shape 2-D arrays")
    if real_host.dtype != np.float16 or imag_host.dtype != np.float16:
        raise TypeError("real and imag files must contain float16 arrays")
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("device must be CUDA")
    real = torch.empty(real_host.shape, dtype=torch.float16, device=target)
    imag = torch.empty(imag_host.shape, dtype=torch.float16, device=target)
    for start in range(0, real_host.shape[0], row_chunk):
        stop = min(start + row_chunk, real_host.shape[0])
        real_chunk = np.array(real_host[start:stop], dtype=np.float16, copy=True)
        imag_chunk = np.array(imag_host[start:stop], dtype=np.float16, copy=True)
        real[start:stop].copy_(torch.from_numpy(real_chunk))
        imag[start:stop].copy_(torch.from_numpy(imag_chunk))
        if progress and (start == 0 or stop == real_host.shape[0]):
            progress(
                100.0 * stop / real_host.shape[0],
                "Loading planar complex32 matrix: {}/{} rows".format(
                    stop, real_host.shape[0]
                ),
            )
    return PlanarComplexHalfMatrix(
        real, imag, storage_is_transposed=storage_is_transposed
    )


def load_complex_numpy_as_planar_half(
    matrix: np.ndarray,
    device: Union[str, torch.device],
    row_chunk: int = 256,
    scale: float = 1.0,
    progress: Optional[ProgressCallback] = None,
) -> PlanarComplexHalfMatrix:
    """Stream a complex64 NumPy/memmap matrix into CUDA FP16 planes."""
    if matrix.ndim != 2 or matrix.dtype != np.complex64:
        raise ValueError("matrix must be a 2-D complex64 NumPy array")
    if row_chunk <= 0:
        raise ValueError("row_chunk must be positive")
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("device must be CUDA")
    real = torch.empty(matrix.shape, dtype=torch.float16, device=target)
    imag = torch.empty(matrix.shape, dtype=torch.float16, device=target)
    host_scale = np.float32(scale)
    for start in range(0, matrix.shape[0], row_chunk):
        stop = min(start + row_chunk, matrix.shape[0])
        block = np.array(matrix[start:stop], dtype=np.complex64, copy=True)
        if host_scale != 1.0:
            block *= host_scale
        real_block = block.real.astype(np.float16)
        imag_block = block.imag.astype(np.float16)
        real[start:stop].copy_(torch.from_numpy(real_block))
        imag[start:stop].copy_(torch.from_numpy(imag_block))
        if progress and (start == 0 or stop == matrix.shape[0]):
            progress(
                100.0 * stop / matrix.shape[0],
                "Loading probe as planar complex32: {}/{} rows".format(
                    stop, matrix.shape[0]
                ),
            )
    return PlanarComplexHalfMatrix(real, imag)


def load_regularized_pinv_fp16(
    files: LowPrecisionPinvFiles,
    probe_path: str,
    expected_shape: Tuple[int, int],
    ridge: float,
    device: Union[str, torch.device],
    row_chunk: int = 256,
    progress: Optional[ProgressCallback] = None,
) -> Tuple[PlanarComplexHalfMatrix, Dict]:
    """Validate and load an exported regularized inverse.

    The metadata check prevents silently combining an inverse with a different
    probe file, normalization, shape, or ridge value.  The stored arrays are
    K.T, while the returned operator exposes the logical K shape.
    """
    for path in (files.real_path, files.imag_path, files.metadata_path):
        if not os.path.isfile(path):
            raise FileNotFoundError("Low-precision inverse file not found: {}".format(path))
    with open(files.metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("status") != "complete":
        raise ValueError("Low-precision inverse metadata is not complete")
    if metadata.get("representation") != "transpose_planar_fp16":
        raise ValueError("Unsupported low-precision inverse representation")
    if tuple(metadata.get("logical_shape", ())) != tuple(expected_shape):
        raise ValueError(
            "Low-precision inverse shape {} does not match expected {}".format(
                metadata.get("logical_shape"), tuple(expected_shape)
            )
        )
    if float(metadata.get("ridge", -1.0)) != float(ridge):
        raise ValueError("Low-precision inverse ridge does not match reconstruction")
    if metadata.get("probe_normalization") != "X/sqrt(M)":
        raise ValueError("Low-precision inverse must use X/sqrt(M) normalization")
    expected_probe = metadata.get("probe")
    if not file_identity_matches(expected_probe, probe_path):
        actual_probe = _file_identity(probe_path)
        raise ValueError(
            "Low-precision inverse was built from a different probe file: "
            "recorded path={!r}, current path={!r}, size match={}, mtime match={}".format(
                expected_probe.get("path") if isinstance(expected_probe, dict) else None,
                actual_probe["path"],
                isinstance(expected_probe, dict)
                and expected_probe.get("size") == actual_probe["size"],
                isinstance(expected_probe, dict)
                and expected_probe.get("mtime_ns") == actual_probe["mtime_ns"],
            )
        )

    operator = load_planar_complex_half(
        files.real_path,
        files.imag_path,
        device=device,
        row_chunk=row_chunk,
        storage_is_transposed=True,
        progress=progress,
    )
    if operator.shape != tuple(expected_shape):
        raise ValueError(
            "Low-precision inverse arrays have shape {}, expected {}".format(
                operator.shape, tuple(expected_shape)
            )
        )
    return operator, metadata


def export_regularized_pinv_fp16(
    probe_matrix: np.ndarray,
    cholesky_factor: torch.Tensor,
    files: LowPrecisionPinvFiles,
    probe_path: Optional[str] = None,
    ridge: float = 0.0,
    normalized_probe: bool = True,
    measurement_chunk: int = 128,
    progress: Optional[ProgressCallback] = None,
) -> Dict:
    """Build K=(X.H X+ridge I)^-1 X.H and save K.T as FP16 planes.

    The output files have shape (measurement_count, input_count), representing
    K.T.  This orientation makes all disk writes contiguous.  At runtime the
    loader exposes the logical (input_count, measurement_count) matrix.
    """
    if probe_matrix.ndim != 2:
        raise ValueError("probe_matrix must be 2-D")
    if probe_matrix.dtype != np.complex64:
        raise TypeError("probe_matrix must have dtype complex64")
    if measurement_chunk <= 0:
        raise ValueError("measurement_chunk must be positive")
    if (
        cholesky_factor.ndim != 2
        or cholesky_factor.shape[0] != cholesky_factor.shape[1]
        or cholesky_factor.dtype != torch.complex64
        or not cholesky_factor.is_cuda
    ):
        raise ValueError("cholesky_factor must be a square CUDA complex64 tensor")
    measurement_count, input_count = map(int, probe_matrix.shape)
    if tuple(cholesky_factor.shape) != (input_count, input_count):
        raise ValueError("factor shape does not match probe input dimension")

    for path in (files.real_path, files.imag_path, files.metadata_path):
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
    real_partial = files.real_path + ".partial"
    imag_partial = files.imag_path + ".partial"
    real_output = np.lib.format.open_memmap(
        real_partial,
        mode="w+",
        dtype=np.float16,
        shape=(measurement_count, input_count),
    )
    imag_output = np.lib.format.open_memmap(
        imag_partial,
        mode="w+",
        dtype=np.float16,
        shape=(measurement_count, input_count),
    )
    scale = 1.0 / math.sqrt(float(measurement_count)) if normalized_probe else 1.0
    maximum_magnitude = 0.0
    with torch.inference_mode():
        for start in range(0, measurement_count, measurement_chunk):
            stop = min(start + measurement_chunk, measurement_count)
            host = np.array(
                probe_matrix[start:stop], dtype=np.complex64, copy=True
            )
            probe_block = torch.from_numpy(host).to(cholesky_factor.device)
            if scale != 1.0:
                probe_block.mul_(scale)
            inverse_transpose = torch.cholesky_solve(
                probe_block.mH, cholesky_factor
            ).mT.contiguous()
            block_maximum = float(torch.amax(torch.abs(inverse_transpose)).item())
            maximum_magnitude = max(maximum_magnitude, block_maximum)
            if not torch.isfinite(inverse_transpose).all():
                raise FloatingPointError(
                    "Inverse block {}:{} contains NaN or infinity".format(start, stop)
                )
            if block_maximum > np.finfo(np.float16).max:
                raise FloatingPointError(
                    "Inverse block {}:{} exceeds float16 range".format(start, stop)
                )
            real_output[start:stop] = inverse_transpose.real.cpu().numpy().astype(
                np.float16
            )
            imag_output[start:stop] = inverse_transpose.imag.cpu().numpy().astype(
                np.float16
            )
            if progress:
                progress(
                    100.0 * stop / measurement_count,
                    "Low-precision inverse: {}/{} probe rows".format(
                        stop, measurement_count
                    ),
                )
    real_output.flush()
    imag_output.flush()
    del real_output, imag_output
    os.replace(real_partial, files.real_path)
    os.replace(imag_partial, files.imag_path)
    result = {
        "status": "complete",
        "representation": "transpose_planar_fp16",
        "logical_shape": [input_count, measurement_count],
        "storage_shape": [measurement_count, input_count],
        "dtype_per_plane": "float16",
        "ridge": float(ridge),
        "probe_normalization": "X/sqrt(M)" if normalized_probe else "none",
        "maximum_complex64_magnitude_before_quantization": maximum_magnitude,
        "real_path": os.path.abspath(files.real_path),
        "imag_path": os.path.abspath(files.imag_path),
    }
    if probe_path:
        result["probe"] = _file_identity(probe_path)
    _write_json_atomic(files.metadata_path, result)
    return result
