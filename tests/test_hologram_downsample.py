import unittest

import numpy as np

from holograms.dmd_holograms import _down_sample


class HologramDownsampleTests(unittest.TestCase):
    def test_max_reduces_each_complete_block(self):
        field = np.arange(16).reshape(4, 4)

        reduced = _down_sample(field, 2, method="max")

        np.testing.assert_array_equal(reduced, [[5, 7], [13, 15]])

    def test_max_zero_pads_partial_edge_blocks(self):
        field = -np.arange(1, 10).reshape(3, 3)

        reduced = _down_sample(field, 2, method="max")

        np.testing.assert_array_equal(reduced, [[-1, 0], [0, 0]])


if __name__ == "__main__":
    unittest.main()
