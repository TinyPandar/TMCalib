import unittest

import numpy as np

from tmcalib.focus_modes import prepare_conjugate_focus_field


class FocusModeTests(unittest.TestCase):
    def test_complex_mode_preserves_conjugated_values(self):
        values = np.array([1 + 2j, -3 + 4j], dtype=np.complex64)
        np.testing.assert_array_equal(
            prepare_conjugate_focus_field(values, "complex"),
            np.conj(values),
        )

    def test_phase_only_mode_has_unit_amplitude(self):
        values = np.array([2 + 2j, -4j], dtype=np.complex64)
        result = prepare_conjugate_focus_field(values, "phase_only")
        np.testing.assert_allclose(np.abs(result), 1.0, atol=1e-6)
        np.testing.assert_allclose(
            np.angle(result),
            np.angle(np.conj(values)),
            atol=1e-6,
        )

    def test_four_level_mode_uses_quadrature_phases(self):
        phases = np.deg2rad([10.0, 80.0, 190.0, 260.0])
        values = np.exp(-1j * phases).astype(np.complex64)
        result = prepare_conjugate_focus_field(values, "phase_only_4")
        expected = np.exp(
            1j * np.deg2rad([0.0, 90.0, 180.0, 270.0])
        )
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_two_level_mode_uses_zero_and_pi(self):
        phases = np.deg2rad([10.0, 100.0, 190.0, 280.0])
        values = np.exp(-1j * phases).astype(np.complex64)
        result = prepare_conjugate_focus_field(values, "phase_only_2")
        expected = np.exp(1j * np.deg2rad([0.0, 180.0, 180.0, 0.0]))
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_binary_amplitude_mode_returns_only_off_and_zero_phase_on(self):
        values = np.array(
            [
                [1 + 2j, -2 + 1j, 3 - 4j, -4 - 3j],
                [-5 + 1j, -2 - 3j, 1 + 1j, -1 + 2j],
            ],
            dtype=np.complex64,
        )
        result = prepare_conjugate_focus_field(
            values,
            "amplitude_only_binary",
        )
        self.assertEqual(result.shape, values.shape)
        self.assertTrue(set(np.unique(result)).issubset({0j, 1 + 0j}))
        selected_sums = np.sum(values * result, axis=-1)
        positive_masks = (np.real(values) >= 0.0).astype(np.complex64)
        negative_masks = 1.0 - positive_masks
        candidate_max = np.maximum(
            np.abs(np.sum(values * positive_masks, axis=-1)),
            np.abs(np.sum(values * negative_masks, axis=-1)),
        )
        np.testing.assert_allclose(np.abs(selected_sums), candidate_max)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown focus mode"):
            prepare_conjugate_focus_field([1 + 0j], "other")


if __name__ == "__main__":
    unittest.main()
