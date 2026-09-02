import importlib
import unittest
from unittest import mock

import numpy as np


class FocusEncoding128x96Tests(unittest.TestCase):
    @staticmethod
    def _bare_controller(module):
        controller = module.DMDController.__new__(module.DMDController)
        controller.dmd_height = module.INPUT_HEIGHT
        controller.dmd_width = module.INPUT_WIDTH
        controller.original_height = module.DMD_HEIGHT
        controller.original_width = module.DMD_WIDTH
        controller.pixel_group_size = module.INPUT_MACRO_PIXEL_SIZE
        controller.hologram_superpixel_size = module.HOLOGRAM_SUPERPIXEL_SIZE
        controller.active_height = module.ACTIVE_HEIGHT
        controller.active_width = module.ACTIVE_WIDTH
        controller.active_y = module.ACTIVE_Y
        controller.active_x = module.ACTIVE_X
        controller.focus_encoding_gpu_device = 0
        controller.focus_encoding_gpu_chunk_size = 2
        controller._focus_gpu_tensor_cache = {}
        controller._focus_gpu_failed = False
        controller.last_focus_encoding_fallback_error = None
        return controller

    def test_profile_uses_98304_pattern_8n_dataset(self):
        module = importlib.import_module("calibrate_128x96")
        config = module.get_active_128x96_pattern_config()
        self.assertEqual(config["name"], "8N")
        self.assertEqual(config["probe_multiplier"], 8)
        self.assertEqual(
            config["mapping_version"],
            module.MAPPING_VERSION,
        )
        with mock.patch.object(
            module._BaseDMDController,
            "__init__",
            return_value=None,
        ):
            controller = module.DMDController()
        self.assertIsInstance(controller, module.DMDController)

    def test_cpu_batch_is_bit_exact_with_mapping_reference(self):
        module = importlib.import_module("calibrate_128x96")
        controller = self._bare_controller(module)
        controller.focus_encoding_use_gpu = False
        rng = np.random.default_rng(20260825)
        rows = (
            rng.standard_normal((2, module.INPUT_HEIGHT * module.INPUT_WIDTH))
            + 1j
            * rng.standard_normal((2, module.INPUT_HEIGHT * module.INPUT_WIDTH))
        ).astype(np.complex64)
        lut_cache = module.get_superpixel_lut(
            module.HOLOGRAM_SUPERPIXEL_SIZE
        )

        patterns, errors = controller._build_focus_hologram_batch(
            rows,
            lut_cache=lut_cache,
        )
        expected = np.stack(
            [
                module.input_field_to_dmd_pattern(
                    controller._phase_only_conjugate(row).reshape(
                        module.INPUT_HEIGHT,
                        module.INPUT_WIDTH,
                    ),
                    lut_cache=lut_cache,
                )
                for row in rows
            ]
        )
        self.assertEqual(errors, [None, None])
        np.testing.assert_array_equal(patterns, expected)
        self.assertEqual(controller.last_focus_encoding_backend, "CPU NumPy")

    def test_cuda_batch_is_bit_exact_with_numpy_batch(self):
        module = importlib.import_module("calibrate_128x96")
        if not module.core.torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")

        controller = self._bare_controller(module)
        rng = np.random.default_rng(20260826)
        rows = (
            rng.standard_normal((3, module.INPUT_HEIGHT * module.INPUT_WIDTH))
            + 1j
            * rng.standard_normal((3, module.INPUT_HEIGHT * module.INPUT_WIDTH))
        ).astype(np.complex64)
        lut_cache = module.get_superpixel_lut(
            module.HOLOGRAM_SUPERPIXEL_SIZE
        )

        controller.focus_encoding_use_gpu = False
        numpy_patterns, numpy_errors = controller._build_focus_hologram_batch(
            rows,
            lut_cache=lut_cache,
        )
        controller.focus_encoding_use_gpu = True
        cuda_patterns, cuda_errors = controller._build_focus_hologram_batch(
            rows,
            lut_cache=lut_cache,
        )

        self.assertEqual(cuda_errors, numpy_errors)
        np.testing.assert_array_equal(cuda_patterns, numpy_patterns)
        self.assertEqual(controller.last_focus_encoding_backend, "GPU cuda:0")
        self.assertIsNone(controller.last_focus_encoding_fallback_error)

    def test_batch_rejects_invalid_tm_rows(self):
        module = importlib.import_module("calibrate_128x96")
        controller = self._bare_controller(module)
        controller.focus_encoding_use_gpu = False
        rows = np.ones(
            (2, module.INPUT_HEIGHT * module.INPUT_WIDTH),
            dtype=np.complex64,
        )
        rows[0] = 0
        rows[1, 17] = np.complex64(np.nan + 0j)

        patterns, errors = controller._build_focus_hologram_batch(rows)

        self.assertEqual(
            errors,
            [
                "Cannot encode an all-zero complex field",
                "TM row contains NaN or infinity",
            ],
        )
        self.assertFalse(np.any(patterns))


if __name__ == "__main__":
    unittest.main()
