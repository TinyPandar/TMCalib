import unittest

import numpy as np

from tools.simulate_lut_aperture_sweep import (
    DMD_HEIGHT,
    DMD_WIDTH,
    LOGICAL_PIXEL_SIZE,
    N_X,
    N_Y,
    circular_frequency_mask,
    first_order_geometry,
    recover_32x24,
)


class LUTApertureSweepTests(unittest.TestCase):
    def test_first_order_geometry_matches_holo_sp_carrier(self):
        negative = first_order_geometry((DMD_HEIGHT, DMD_WIDTH), order_sign=-1)
        self.assertEqual(negative["fft_center_yx"], (384, 512))
        self.assertEqual(negative["order_offset_yx"], (-48, -256))
        self.assertEqual(negative["order_center_yx"], (336, 256))

        positive = first_order_geometry((DMD_HEIGHT, DMD_WIDTH), order_sign=1)
        self.assertEqual(positive["order_offset_yx"], (48, 256))
        self.assertEqual(positive["order_center_yx"], (432, 768))

    def test_frequency_circle_scales_rectangular_fft_y_axis(self):
        center = (DMD_HEIGHT // 2, DMD_WIDTH // 2)
        mask = circular_frequency_mask(
            (DMD_HEIGHT, DMD_WIDTH), center, radius_x_bins=100
        )
        # 100 x bins and 75 y bins are the same physical spatial frequency
        # because H/W = 768/1024 = 0.75.
        self.assertTrue(mask[center[0], center[1] + 100])
        self.assertTrue(mask[center[0] + 75, center[1]])
        self.assertFalse(mask[center[0] + 76, center[1]])

    def test_recover_32x24_averages_each_logical_block(self):
        logical = (
            np.arange(N_Y * N_X, dtype=np.float32).reshape(N_Y, N_X)
            + 1j * np.ones((N_Y, N_X), dtype=np.float32)
        )
        dmd = np.repeat(
            np.repeat(logical, LOGICAL_PIXEL_SIZE, axis=0),
            LOGICAL_PIXEL_SIZE,
            axis=1,
        )
        recovered = recover_32x24(dmd)
        np.testing.assert_allclose(recovered, logical)


if __name__ == "__main__":
    unittest.main()
