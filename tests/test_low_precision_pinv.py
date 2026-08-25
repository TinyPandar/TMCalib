import json
import os
import tempfile
import unittest

import numpy as np
import torch

from low_precision_pinv import (
    LowPrecisionPinvFiles,
    PlanarComplexHalfMatrix,
    export_regularized_pinv_fp16,
    file_identity_matches,
    load_complex_numpy_as_planar_half,
    load_planar_complex_half,
)
from tm_reconstruction_128 import ReconstructionConfig, ggs21_block, reconstruct_tm


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class LowPrecisionPinvTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda:0")
        self.generator = torch.Generator(device=self.device).manual_seed(17)

    def random_complex(self, shape):
        return torch.complex(
            torch.randn(shape, device=self.device, generator=self.generator),
            torch.randn(shape, device=self.device, generator=self.generator),
        )

    def test_file_identity_uses_windows_case_insensitive_paths(self):
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            path = handle.name
            handle.write(b"probe identity")
        try:
            stat = os.stat(path)
            alternate_case = path.swapcase()
            identity = {
                "path": alternate_case,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
            self.assertTrue(file_identity_matches(identity, path))
            identity["size"] += 1
            self.assertFalse(file_identity_matches(identity, path))
        finally:
            os.remove(path)

    def test_planar_half_matmul_matches_complex64(self):
        matrix = self.random_complex((128, 96))
        right = self.random_complex((96, 11))
        operator = PlanarComplexHalfMatrix.from_complex(matrix)
        actual = operator.matmul(right)
        expected = matrix @ right
        relative_error = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)
        self.assertLess(float(relative_error), 8e-4)

    def test_planar_half_adjoint_matmul_matches_complex64(self):
        matrix = self.random_complex((128, 96))
        right = self.random_complex((128, 11))
        operator = PlanarComplexHalfMatrix.from_complex(matrix)
        actual = operator.adjoint_matmul(right)
        expected = matrix.mH @ right
        relative_error = (
            torch.linalg.vector_norm(actual - expected)
            / torch.linalg.vector_norm(expected)
        )
        self.assertLess(float(relative_error), 8e-4)

    def test_planar_half_probe_supports_cholesky_ggs(self):
        measurement_count = 192
        input_count = 32
        output_count = 4
        phase = torch.randint(
            0,
            16,
            (measurement_count, input_count),
            device=self.device,
            generator=self.generator,
        )
        x = torch.exp(1j * phase.float() * torch.pi / 8).to(torch.complex64)
        x /= np.sqrt(measurement_count)
        true_h = self.random_complex((input_count, output_count))
        amplitude = torch.abs(x @ true_h)
        gram = x.mH @ x
        gram.diagonal().add_(1e-4)
        factor = torch.linalg.cholesky(gram)

        reference, reference_errors = ggs21_block(
            x,
            amplitude,
            iterations=30,
            gs2_ratio=0.75,
            solver="cholesky",
            cholesky_factor=factor,
            random_seed=31,
        )
        actual, actual_errors = ggs21_block(
            PlanarComplexHalfMatrix.from_complex(x),
            amplitude,
            iterations=30,
            gs2_ratio=0.75,
            solver="cholesky",
            cholesky_factor=factor,
            random_seed=31,
        )

        relative_curve_error = (
            np.linalg.norm(actual_errors - reference_errors)
            / np.linalg.norm(reference_errors)
        )
        relative_result_error = (
            torch.linalg.vector_norm(actual - reference)
            / torch.linalg.vector_norm(reference)
        )
        self.assertLess(float(relative_curve_error), 2e-2)
        self.assertLess(float(relative_result_error), 2e-2)

    def test_planar_half_cholesky_end_to_end_builds_streaming_factor(self):
        rng = np.random.default_rng(37)
        measurement_count = 64
        input_shape = (4, 4)
        output_shape = (2, 2)
        input_count = int(np.prod(input_shape))
        output_count = int(np.prod(output_shape))
        phase = rng.integers(
            0, 16, size=(measurement_count,) + input_shape, dtype=np.uint8
        )
        probes = np.exp(1j * phase.astype(np.float32) * np.pi / 8).astype(
            np.complex64
        )
        true_h = (
            rng.standard_normal((output_count, input_count))
            + 1j * rng.standard_normal((output_count, input_count))
        ).astype(np.complex64)
        field = probes.reshape(measurement_count, input_count) @ true_h.T
        intensity = np.rint(np.abs(field) ** 2 * 100).astype(np.uint16)

        with tempfile.TemporaryDirectory() as directory:
            probe_path = os.path.join(directory, "probe.npy")
            measurement_path = os.path.join(directory, "measurements.raw")
            output_path = os.path.join(directory, "tm.npy")
            metadata_path = os.path.join(directory, "reconstruction.json")
            factor_path = os.path.join(directory, "factor.npy")
            np.save(probe_path, probes)
            intensity.tofile(measurement_path)

            result = reconstruct_tm(
                ReconstructionConfig(
                    measurement_path=measurement_path,
                    probe_path=probe_path,
                    output_path=output_path,
                    error_curve_path=os.path.join(directory, "error.npy"),
                    metadata_path=metadata_path,
                    cholesky_cache_path=factor_path,
                    input_shape=input_shape,
                    output_shape=output_shape,
                    iterations=20,
                    gs2_ratio=0.75,
                    output_chunk_size=2,
                    adjoint_row_chunk=17,
                    probe_storage="planar_complex32",
                    solver="cholesky",
                    device=str(self.device),
                    resume=False,
                )
            )

            output = np.load(output_path)
            self.assertEqual(output.shape, (output_count, input_count))
            self.assertTrue(bool(np.isfinite(output).all()))
            self.assertTrue(os.path.isfile(factor_path))
            with open(factor_path + ".json", "r", encoding="utf-8") as handle:
                factor_metadata = json.load(handle)
            self.assertEqual(
                factor_metadata["construction"], "streaming_probe_rows"
            )
            self.assertEqual(result["config"]["probe_storage"], "planar_complex32")
            self.assertEqual(
                result["dtype_pipeline"]["probe_compute"],
                "planar torch.float16",
            )

    def test_rhs_normalization_avoids_half_overflow(self):
        matrix = self.random_complex((32, 48)) * 1e-3
        right = self.random_complex((48, 5)) * 1e6
        operator = PlanarComplexHalfMatrix.from_complex(matrix)
        actual = operator.matmul(right, normalize_rhs=True)
        self.assertTrue(bool(torch.isfinite(actual).all()))
        expected = matrix @ right
        relative_error = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)
        self.assertLess(float(relative_error), 1e-3)

    def test_stream_complex_numpy_to_half_planes(self):
        matrix = self.random_complex((35, 19))
        matrix_host = matrix.cpu().numpy().astype(np.complex64)
        operator = load_complex_numpy_as_planar_half(
            matrix_host, self.device, row_chunk=7, scale=0.25
        )
        right = self.random_complex((19, 4))
        actual = operator.matmul(right)
        expected = (matrix * 0.25) @ right
        relative_error = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)
        self.assertLess(float(relative_error), 8e-4)

    def test_exported_cholesky_inverse(self):
        measurement_count = 96
        input_count = 24
        ridge = 1e-3
        probe = self.random_complex((measurement_count, input_count))
        probe /= np.sqrt(measurement_count)
        gram = probe.mH @ probe
        gram.diagonal().add_(ridge)
        factor = torch.linalg.cholesky(gram)
        expected = torch.cholesky_solve(probe.mH, factor)
        probe_host = (probe * np.sqrt(measurement_count)).cpu().numpy().astype(np.complex64)

        with tempfile.TemporaryDirectory() as directory:
            files = LowPrecisionPinvFiles(
                real_path=os.path.join(directory, "pinv_real.npy"),
                imag_path=os.path.join(directory, "pinv_imag.npy"),
                metadata_path=os.path.join(directory, "pinv.json"),
            )
            export_regularized_pinv_fp16(
                probe_host,
                factor,
                files,
                ridge=ridge,
                normalized_probe=True,
                measurement_chunk=17,
            )
            operator = load_planar_complex_half(
                files.real_path,
                files.imag_path,
                self.device,
                row_chunk=19,
                storage_is_transposed=True,
            )
            right = self.random_complex((measurement_count, 7))
            actual = operator.matmul(right)
            reference = expected @ right
            relative_error = torch.linalg.vector_norm(actual - reference) / torch.linalg.vector_norm(reference)
            self.assertLess(float(relative_error), 1e-3)

    def test_complex32_pinv_ggs_path_tracks_cholesky(self):
        measurement_count = 192
        input_count = 32
        output_count = 4
        ridge = 1e-4
        phase = torch.randint(
            0,
            16,
            (measurement_count, input_count),
            device=self.device,
            generator=self.generator,
        )
        x = torch.exp(1j * phase.float() * torch.pi / 8).to(torch.complex64)
        x /= np.sqrt(measurement_count)
        true_h = self.random_complex((input_count, output_count))
        amplitude = torch.abs(x @ true_h)
        gram = x.mH @ x
        gram.diagonal().add_(ridge)
        factor = torch.linalg.cholesky(gram)
        inverse = torch.cholesky_solve(x.mH, factor)
        x_half = PlanarComplexHalfMatrix.from_complex(x)
        inverse_half = PlanarComplexHalfMatrix.from_complex(inverse)

        reference, reference_errors = ggs21_block(
            x,
            amplitude,
            iterations=30,
            gs2_ratio=0.75,
            solver="cholesky",
            cholesky_factor=factor,
            random_seed=29,
        )
        actual, actual_errors = ggs21_block(
            x_half,
            amplitude,
            iterations=30,
            gs2_ratio=0.75,
            solver="complex32_pinv",
            low_precision_pinv=inverse_half,
            random_seed=29,
        )

        self.assertTrue(bool(torch.isfinite(actual).all()))
        self.assertLess(float(actual_errors[-1]), float(actual_errors[0]))
        relative_curve_error = np.linalg.norm(actual_errors - reference_errors) / np.linalg.norm(reference_errors)
        self.assertLess(float(relative_curve_error), 2e-2)
        numerator = torch.abs(torch.sum(reference.conj() * actual, dim=1))
        denominator = (
            torch.linalg.vector_norm(reference, dim=1)
            * torch.linalg.vector_norm(actual, dim=1)
        ).clamp_min_(1e-12)
        self.assertGreater(float(torch.mean(numerator / denominator)), 0.99)

    def test_complex32_pinv_end_to_end_loader(self):
        rng = np.random.default_rng(23)
        measurement_count = 64
        input_shape = (4, 4)
        output_shape = (2, 2)
        input_count = int(np.prod(input_shape))
        output_count = int(np.prod(output_shape))
        ridge = 1e-4
        phase = rng.integers(
            0, 16, size=(measurement_count,) + input_shape, dtype=np.uint8
        )
        probes = np.exp(1j * phase.astype(np.float32) * np.pi / 8).astype(
            np.complex64
        )
        true_h = (
            rng.standard_normal((output_count, input_count))
            + 1j * rng.standard_normal((output_count, input_count))
        ).astype(np.complex64)
        field = probes.reshape(measurement_count, input_count) @ true_h.T
        intensity = np.rint(np.abs(field) ** 2 * 100).astype(np.uint16)

        with tempfile.TemporaryDirectory() as directory:
            probe_path = os.path.join(directory, "probe.npy")
            measurement_path = os.path.join(directory, "measurements.raw")
            output_path = os.path.join(directory, "tm.npy")
            error_path = os.path.join(directory, "error.npy")
            reconstruction_metadata_path = os.path.join(directory, "reconstruction.json")
            files = LowPrecisionPinvFiles(
                real_path=os.path.join(directory, "pinv_real.npy"),
                imag_path=os.path.join(directory, "pinv_imag.npy"),
                metadata_path=os.path.join(directory, "pinv.json"),
            )
            np.save(probe_path, probes)
            intensity.tofile(measurement_path)
            x = torch.from_numpy(probes.reshape(measurement_count, input_count)).to(
                self.device
            )
            x /= np.sqrt(measurement_count)
            gram = x.mH @ x
            gram.diagonal().add_(ridge)
            factor = torch.linalg.cholesky(gram)
            export_regularized_pinv_fp16(
                probes.reshape(measurement_count, input_count),
                factor,
                files,
                probe_path=probe_path,
                ridge=ridge,
                normalized_probe=True,
                measurement_chunk=17,
            )

            result = reconstruct_tm(
                ReconstructionConfig(
                    measurement_path=measurement_path,
                    probe_path=probe_path,
                    output_path=output_path,
                    error_curve_path=error_path,
                    metadata_path=reconstruction_metadata_path,
                    cholesky_cache_path=os.path.join(directory, "unused_factor.npy"),
                    pinv_real_path=files.real_path,
                    pinv_imag_path=files.imag_path,
                    pinv_metadata_path=files.metadata_path,
                    input_shape=input_shape,
                    output_shape=output_shape,
                    iterations=20,
                    gs2_ratio=0.75,
                    output_chunk_size=2,
                    ridge=ridge,
                    solver="complex32_pinv",
                    device=str(self.device),
                    resume=False,
                )
            )

            output = np.load(output_path)
            self.assertEqual(output.shape, (output_count, input_count))
            self.assertTrue(bool(np.isfinite(output).all()))
            self.assertEqual(result["solver"], "complex32_pinv")
            self.assertEqual(
                result["low_precision_pinv"]["representation"],
                "transpose_planar_fp16",
            )


if __name__ == "__main__":
    unittest.main()
