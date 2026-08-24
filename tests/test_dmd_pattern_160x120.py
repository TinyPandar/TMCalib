import unittest

import numpy as np

from dmd_pattern_160x120 import (
    DMD_HEIGHT,
    DMD_WIDTH,
    EXPANDED_HEIGHT,
    EXPANDED_WIDTH,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    SOURCE_X_INDICES,
    SOURCE_Y_INDICES,
    active_region_mask,
    expand_input_field,
    source_repeat_counts,
)
from tools.generate_probe_samples_160x120_8n import (
    PROBE_COUNT,
    dataset_required_bytes,
    expand_phase_indices,
)


class DmdPattern160x120Tests(unittest.TestCase):
    def test_source_is_expanded_to_the_complete_superpixel_grid(self):
        source = np.arange(INPUT_HEIGHT * INPUT_WIDTH).reshape(
            INPUT_HEIGHT,
            INPUT_WIDTH,
        )
        expanded = expand_input_field(source)
        self.assertEqual(expanded.shape, (EXPANDED_HEIGHT, EXPANDED_WIDTH))
        np.testing.assert_array_equal(
            expanded,
            source[SOURCE_Y_INDICES[:, None], SOURCE_X_INDICES[None, :]],
        )
        self.assertEqual(expanded[0, 0], np.complex64(source[0, 0]))
        self.assertEqual(expanded[-1, -1], np.complex64(source[-1, -1]))

    def test_each_source_sample_uses_one_or_two_superpixels_per_axis(self):
        for axis in ("x", "y"):
            repeats = source_repeat_counts(axis)
            self.assertEqual(int(repeats.min()), 1)
            self.assertEqual(int(repeats.max()), 2)
        self.assertEqual(int(source_repeat_counts("x").sum()), EXPANDED_WIDTH)
        self.assertEqual(int(source_repeat_counts("y").sum()), EXPANDED_HEIGHT)

    def test_phase_batch_uses_the_same_mapping(self):
        source = np.arange(INPUT_HEIGHT * INPUT_WIDTH, dtype=np.uint32).reshape(
            1,
            INPUT_HEIGHT,
            INPUT_WIDTH,
        )
        expanded = expand_phase_indices(source)
        np.testing.assert_array_equal(
            expanded[0],
            source[0][SOURCE_Y_INDICES[:, None], SOURCE_X_INDICES[None, :]],
        )

    def test_active_mask_has_no_outer_padding(self):
        mask = active_region_mask()
        self.assertEqual(mask.shape, (DMD_HEIGHT, DMD_WIDTH))
        self.assertTrue(np.all(mask == 255))

    def test_full_8n_dimensions_and_storage_are_explicit(self):
        self.assertEqual(PROBE_COUNT, 153600)
        expected = (
            PROBE_COUNT * INPUT_HEIGHT * INPUT_WIDTH * np.dtype(np.complex64).itemsize
            + PROBE_COUNT * DMD_HEIGHT * DMD_WIDTH * np.dtype(np.uint8).itemsize
        )
        self.assertEqual(dataset_required_bytes(), expected)


if __name__ == "__main__":
    unittest.main()
