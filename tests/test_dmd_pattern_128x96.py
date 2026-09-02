import unittest

import numpy as np

from dmd_pattern_128x96 import (
    DMD_HEIGHT,
    DMD_WIDTH,
    EXPANDED_HEIGHT,
    EXPANDED_WIDTH,
    INPUT_HEIGHT,
    INPUT_MACRO_PIXEL_SIZE,
    INPUT_WIDTH,
    SOURCE_X_INDICES,
    SOURCE_Y_INDICES,
    active_region_mask,
    expand_input_field,
    input_field_to_dmd_pattern,
    source_repeat_counts,
)
from tools.generate_probe_samples_128x96_8n import (
    PROBE_COUNT,
    build_phase_tiles,
    dataset_required_bytes,
    encode_phase_indices,
    expand_phase_indices,
)


class DmdPattern128x96Tests(unittest.TestCase):
    def test_source_repeats_twice_onto_complete_superpixel_grid(self):
        source = np.arange(INPUT_HEIGHT * INPUT_WIDTH).reshape(
            INPUT_HEIGHT,
            INPUT_WIDTH,
        )
        expanded = expand_input_field(source)
        self.assertEqual(expanded.shape, (EXPANDED_HEIGHT, EXPANDED_WIDTH))
        np.testing.assert_array_equal(
            expanded,
            np.repeat(np.repeat(source, 2, axis=0), 2, axis=1),
        )
        np.testing.assert_array_equal(
            expanded,
            source[SOURCE_Y_INDICES[:, None], SOURCE_X_INDICES[None, :]],
        )
        self.assertEqual(INPUT_MACRO_PIXEL_SIZE, 8)
        for axis in ("x", "y"):
            repeats = source_repeat_counts(axis)
            self.assertTrue(np.all(repeats == 2))

    def test_phase_batch_uses_the_same_repeat_mapping(self):
        source = np.arange(INPUT_HEIGHT * INPUT_WIDTH, dtype=np.uint32).reshape(
            1,
            INPUT_HEIGHT,
            INPUT_WIDTH,
        )
        expanded = expand_phase_indices(source)
        np.testing.assert_array_equal(
            expanded[0],
            np.repeat(np.repeat(source[0], 2, axis=0), 2, axis=1),
        )

    def test_batched_generator_matches_authoritative_encoder(self):
        rng = np.random.default_rng(20260825)
        phase_indices = rng.integers(
            0,
            16,
            size=(1, INPUT_HEIGHT, INPUT_WIDTH),
            dtype=np.uint8,
        )
        phase_values, phase_tiles = build_phase_tiles()
        generated = encode_phase_indices(phase_indices, phase_tiles)[0]
        reference = input_field_to_dmd_pattern(
            phase_values[phase_indices[0]],
        )
        np.testing.assert_array_equal(generated, reference)

    def test_active_mask_has_no_outer_padding(self):
        mask = active_region_mask()
        self.assertEqual(mask.shape, (DMD_HEIGHT, DMD_WIDTH))
        self.assertTrue(np.all(mask == 255))

    def test_full_8n_dimensions_and_storage_are_explicit(self):
        self.assertEqual(PROBE_COUNT, 98_304)
        expected = (
            PROBE_COUNT
            * INPUT_HEIGHT
            * INPUT_WIDTH
            * np.dtype(np.complex64).itemsize
            + PROBE_COUNT
            * DMD_HEIGHT
            * DMD_WIDTH
            * np.dtype(np.uint8).itemsize
        )
        self.assertEqual(dataset_required_bytes(), expected)


if __name__ == "__main__":
    unittest.main()
