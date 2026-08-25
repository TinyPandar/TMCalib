import unittest

from calibration_profiles import (
    PROFILES,
    channelized_filename,
    normalize_polarization_channel,
)
from run_calibration import PROFILE_MODULES


class CalibrationProfileTests(unittest.TestCase):
    def test_expected_profiles_are_registered(self):
        self.assertEqual(PROFILES["v4_32x24"].input_shape, (24, 32))
        self.assertEqual(PROFILES["dense_128x128"].camera_roi, (128, 128))
        self.assertEqual(PROFILES["dense_128x128_roi26"].camera_roi, (26, 26))

        fourfold = PROFILES["fourfold_128x96"]
        self.assertEqual(fourfold.input_shape, (96, 128))
        self.assertEqual(fourfold.camera_roi, (128, 128))
        self.assertEqual(fourfold.input_macro_pixel_size, 8)
        self.assertEqual(fourfold.active_shape, (768, 1024))
        self.assertEqual(
            PROFILE_MODULES["fourfold_128x96"],
            "calibrate_128x96",
        )

        fivefold = PROFILES["fivefold_160x120"]
        self.assertEqual(fivefold.input_shape, (120, 160))
        self.assertEqual(fivefold.camera_roi, (128, 128))
        self.assertEqual(fivefold.active_shape, (768, 1024))
        self.assertNotIn("test64", fivefold.capabilities)

    def test_polarization_channel_is_canonical(self):
        self.assertEqual(normalize_polarization_channel("i90"), "I90")
        with self.assertRaises(ValueError):
            normalize_polarization_channel("I45")

    def test_channelized_filename_replaces_existing_tag(self):
        source = "measurements_128_px4_active512_8N_full_I0_memmap.npy"
        self.assertEqual(
            channelized_filename(source, "I90"),
            "measurements_128_px4_active512_8N_full_I90_memmap.npy",
        )


if __name__ == "__main__":
    unittest.main()
