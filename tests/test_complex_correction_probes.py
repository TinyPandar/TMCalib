import unittest

import numpy as np

from tools.generate_complex_correction_probes_32x24 import (
    DMD_HEIGHT,
    DMD_WIDTH,
    N_X,
    N_Y,
    amplitude_levels,
    build_complex_tiles,
    encode_index_batch,
    phase_values,
)


class ComplexCorrectionProbeTests(unittest.TestCase):
    def test_amplitude_levels_span_requested_range(self):
        values = amplitude_levels(4, 0.25)
        np.testing.assert_allclose(values, [0.25, 0.5, 0.75, 1.0])

    def test_phase_values_have_unit_magnitude(self):
        values = phase_values(16)
        self.assertEqual(values.shape, (16,))
        np.testing.assert_allclose(np.abs(values), 1.0, atol=1e-6)

    def test_complex_tiles_are_binary_and_full_logical_pixel_size(self):
        tiles = build_complex_tiles(
            np.asarray([0.5, 1.0], dtype=np.float32),
            phase_values(4),
        )
        self.assertEqual(tiles.shape, (2, 4, 32, 32))
        unique = np.unique(tiles)
        self.assertTrue(set(unique.tolist()).issubset({0, 255}))

    def test_index_batch_maps_to_full_dmd_shape(self):
        # A synthetic tile bank is enough to verify the vectorized layout.
        tiles = np.zeros((2, 2, 32, 32), dtype=np.uint8)
        tiles[1, 0] = 255
        amp = np.zeros((1, N_Y, N_X), dtype=np.uint8)
        phase = np.zeros_like(amp)
        amp[:, 3, 5] = 1

        encoded = encode_index_batch(amp, phase, tiles)
        self.assertEqual(encoded.shape, (1, DMD_HEIGHT, DMD_WIDTH))
        block = encoded[0, 3 * 32 : 4 * 32, 5 * 32 : 6 * 32]
        self.assertTrue(np.all(block == 255))
        self.assertEqual(int(encoded.sum()), 32 * 32 * 255)


if __name__ == "__main__":
    unittest.main()
