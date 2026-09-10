import math
import os
import tempfile
import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - measurement environment installs torch separately
    torch = None

if torch is not None:
    from tools.run_tm_grad_correction import _load_measurements
    from tm_grad_correction import (
        build_dct2_basis,
        fit_input_dct_correction,
    )


@unittest.skipIf(torch is None, "PyTorch is installed separately from requirements.txt")
class TMGradCorrectionTests(unittest.TestCase):
    def test_loads_headerless_uint16_measurement_memmap(self):
        expected = np.arange(24, dtype=np.uint16).reshape(4, 6)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "measurements_memmap.npy")
            raw = np.memmap(
                path, dtype=np.uint16, mode="w+", shape=expected.shape
            )
            raw[:] = expected
            raw.flush()
            del raw

            actual, storage_format = _load_measurements(path, sample_count=4)

            self.assertEqual(storage_format, "headerless uint16 memmap")
            np.testing.assert_array_equal(actual, expected)
            del actual

    def test_loads_standard_npy_measurements(self):
        expected = np.arange(24, dtype=np.float32).reshape(4, 6)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "measurements.npy")
            np.save(path, expected)

            actual, storage_format = _load_measurements(path, sample_count=4)

            self.assertEqual(storage_format, "NPY")
            np.testing.assert_array_equal(actual, expected)
            del actual

    def test_rejects_incompatible_headerless_measurement_size(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "measurements_memmap.npy")
            with open(path, "wb") as stream:
                stream.write(b"bad-size")

            with self.assertRaisesRegex(ValueError, "incompatible"):
                _load_measurements(path, sample_count=3)

    def test_dct_basis_excludes_dc(self):
        basis = build_dct2_basis((4, 5), (3, 2), exclude_dc=True)
        self.assertEqual(tuple(basis.shape), (5, 20))
        means = basis.mean(dim=1)
        self.assertTrue(torch.all(torch.abs(means) < 1e-6))

    def test_gradient_fit_recovers_low_dimensional_phase_error(self):
        torch.manual_seed(7)
        rng = np.random.default_rng(7)

        input_shape = (4, 4)
        n_input = 16
        n_output = 32
        sample_count = 96

        tm = (
            rng.normal(size=(n_output, n_input))
            + 1j * rng.normal(size=(n_output, n_input))
        ).astype(np.complex64) / math.sqrt(2.0 * n_input)

        probe_phase = rng.uniform(
            0.0, 2.0 * math.pi, size=(sample_count, n_input)
        )
        probes = np.exp(1j * probe_phase).astype(np.complex64)

        basis = build_dct2_basis((4, 4), (3, 3), exclude_dc=True).numpy()
        true_coeff = np.zeros(basis.shape[0], dtype=np.float32)
        true_coeff[0] = 0.7
        true_coeff[3] = -0.5
        true_phase = true_coeff @ basis
        true_gain = np.exp(1j * true_phase).astype(np.complex64)

        field = (probes * true_gain[None, :]) @ tm.T
        measured = np.abs(field).astype(np.float32) ** 2

        _, result = fit_input_dct_correction(
            tm=tm,
            probes=probes,
            measured_intensity=measured,
            input_shape=input_shape,
            phase_modes=(3, 3),
            amplitude_modes=(0, 0),
            epochs=120,
            batch_size=48,
            learning_rate=5e-2,
            l2_weight=1e-5,
            val_fraction=0.25,
            seed=7,
            device="cpu",
            log_every=0,
        )

        self.assertGreater(result.final_val_pcc, 0.99)
        self.assertGreater(
            result.final_val_pcc - result.baseline_val_pcc, 0.25
        )


if __name__ == "__main__":
    unittest.main()
