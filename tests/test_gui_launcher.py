import os
import unittest

from tmcalib_gui.launcher import (
    build_launch_spec,
    format_profile_summary,
    normalize_channel,
    profile_supports_channel,
)


class GuiLauncherTests(unittest.TestCase):
    def test_regular_profile_has_no_channel(self):
        spec = build_launch_spec(
            "v4_32x24",
            python_executable="python-test",
            repository_root=os.path.join("tmp", "tmcalib"),
        )
        self.assertIsNone(spec.channel)
        self.assertEqual(spec.program, "python-test")
        self.assertEqual(spec.arguments[-2:], ("--profile", "v4_32x24"))

    def test_polarization_profile_defaults_to_i0(self):
        spec = build_launch_spec(
            "dense_128x128_roi26",
            python_executable="python-test",
            repository_root=os.path.join("tmp", "tmcalib"),
        )
        self.assertEqual(spec.channel, "I0")
        self.assertEqual(spec.arguments[-2:], ("--channel", "I0"))

    def test_channel_is_normalized(self):
        self.assertEqual(normalize_channel("dense_128x128_roi26", "i90"), "I90")

    def test_channel_rejected_for_regular_profile(self):
        with self.assertRaises(ValueError):
            normalize_channel("dense_128x128", "I0")

    def test_profile_summary_contains_dimensions(self):
        summary = format_profile_summary("fivefold_160x120")
        self.assertIn("160 × 120", summary)
        self.assertIn("128 × 128", summary)
        self.assertIn("1024 × 768", summary)

    def test_cholesky_profile_reuses_v4_dimensions(self):
        spec = build_launch_spec("v4_32x24_cholesky")
        self.assertEqual(spec.profile, "v4_32x24_cholesky")
        self.assertIn("32 × 24", format_profile_summary(spec.profile))

    def test_polarization_capability(self):
        self.assertTrue(profile_supports_channel("dense_128x128_roi26"))
        self.assertFalse(profile_supports_channel("dense_128x128"))


if __name__ == "__main__":
    unittest.main()
