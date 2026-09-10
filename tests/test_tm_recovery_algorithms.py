import ast
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from tm_recovery_algorithms import ALGORITHMS, RecoverySolver, canonical_algorithm


class RecoveryAlgorithmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        generator = torch.Generator().manual_seed(123)
        cls.X = torch.polar(
            torch.ones(384, 24),
            torch.rand(384, 24, generator=generator) * 6.283185,
        )
        cls.truth = torch.complex(
            torch.randn(24, 3, generator=generator),
            torch.randn(24, 3, generator=generator),
        )
        cls.y = (cls.X @ cls.truth).abs()

    def test_all_algorithms_recover_synthetic_rows(self):
        for name in ALGORITHMS:
            with self.subTest(algorithm=name):
                recovered, errors = RecoverySolver(
                    self.X, name, iterations=250
                ).solve(self.y)
                relative = torch.linalg.vector_norm(
                    (self.X @ recovered.T).abs() - self.y
                ) / torch.linalg.vector_norm(self.y)
                fidelity = (
                    (recovered.T.conj() * self.truth).sum(0).abs()
                    / (
                        torch.linalg.vector_norm(recovered.T, dim=0)
                        * torch.linalg.vector_norm(self.truth, dim=0)
                    )
                )
                self.assertTrue(torch.isfinite(errors).all())
                self.assertLess(float(relative), 0.08, name)
                self.assertGreater(float(fidelity.min()), 0.97, name)

    def test_zero_output_and_positive_scaling(self):
        for name in ALGORITHMS:
            with self.subTest(algorithm=name):
                solver = RecoverySolver(self.X, name, iterations=10)
                first, _ = solver.solve(
                    torch.cat((self.y[:, :1], torch.zeros(384, 1)), dim=1)
                )
                scaled, _ = solver.solve(
                    torch.cat((3 * self.y[:, :1], torch.zeros(384, 1)), dim=1)
                )
                torch.testing.assert_close(first[1], torch.zeros_like(first[1]))
                torch.testing.assert_close(scaled, 3 * first, rtol=2e-4, atol=2e-4)

    def test_32x24_input_shape(self):
        generator = torch.Generator().manual_seed(8)
        X = torch.polar(
            torch.ones(1536, 768),
            torch.rand(1536, 768, generator=generator) * 6.283185,
        )
        truth = torch.complex(
            torch.randn(768, 1, generator=generator),
            torch.randn(768, 1, generator=generator),
        )
        for name in ALGORITHMS:
            with self.subTest(algorithm=name):
                recovered, errors = RecoverySolver(
                    X, name, iterations=2, power_iterations=2
                ).solve((X @ truth).abs())
                self.assertEqual(recovered.shape, (1, 768))
                self.assertTrue(torch.isfinite(errors).all())

    def test_validation_and_prvam_alias(self):
        self.assertEqual(canonical_algorithm("prVAM"), "prVAMP")
        with self.assertRaises(ValueError):
            RecoverySolver(self.X, "unknown")
        solver = RecoverySolver(self.X, "RAF21")
        for invalid in (-self.y, self.y[:10], self.y * float("nan")):
            with self.assertRaises(ValueError):
                solver.solve(invalid)

    def test_32x24_file_pipeline_without_hardware(self):
        source = Path(__file__).resolve().parents[1] / "calibrate_v4_32x24.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        controller = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == "DMDController"
        )
        method = next(
            node
            for node in controller.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_reconstruction"
        )
        namespace = dict(
            torch=torch,
            np=np,
            os=os,
            math=math,
            time=__import__("time"),
            RecoverySolver=RecoverySolver,
            canonical_algorithm=canonical_algorithm,
        )
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
            namespace,
        )

        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                np.save("probe.npy", self.X.numpy())
                measured = np.memmap(
                    "measurements_memmap.npy",
                    mode="w+",
                    dtype=np.uint16,
                    shape=(384, 3),
                )
                measured[:] = np.rint(self.y.numpy() ** 2)
                measured.flush()
                del measured
                namespace["get_active_pattern_config"] = lambda: {
                    "directory": directory,
                    "probe_multiplier": 16,
                    "name": "test",
                }
                controller_instance = SimpleNamespace(
                    dmd_width=6,
                    dmd_height=4,
                    camera=SimpleNamespace(roi_width=3, roi_height=1),
                    recovery_algorithm="RAF21",
                    recovery_block_size=2,
                    recovery_power_iterations=2,
                    ggs21_iters=20,
                    ggs21_use_gpu=False,
                    ggs21_random_seed=42,
                    tm_memmap_filename="tm.npy",
                    reconstructed_filename="field.npy",
                    error_curve_filename="error.npy",
                )
                namespace["run_reconstruction"](controller_instance)
                self.assertIsNone(controller_instance.reconstruction_error)
                self.assertFalse(controller_instance.reconstruction_running)
                recovered = np.load("field.npy")
                self.assertEqual(recovered.shape, (3, 24))
                self.assertTrue(np.isfinite(recovered).all())
                np.testing.assert_array_equal(recovered, np.load("field_raf21.npy"))
                self.assertEqual(np.load("error.npy").shape, (20,))
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
