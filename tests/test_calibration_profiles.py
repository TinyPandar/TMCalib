import unittest

from calibration_profiles import (
    PROFILES,
    channelized_filename,
    normalize_polarization_channel,
)


class CalibrationProfileTests(unittest.TestCase):
    def test_expected_profiles_are_registered(self):
        self.assertEqual(PROFILES["v4_32x24"].input_shape, (24, 32))
        self.assertEqual(PROFILES["dense_128x128"].camera_roi, (128, 128))
        self.assertEqual(PROFILES["dense_128x128_roi26"].camera_roi, (26, 26))

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
