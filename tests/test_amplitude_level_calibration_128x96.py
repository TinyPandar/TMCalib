import os
import tempfile
import unittest

import numpy as np

from tools.amplitude_level_calibration_128x96 import (
    CAMERA_SHAPE,
    DMD_SHAPE,
    analyze_and_save,
    build_amplitude_levels,
    build_inverse_lut,
    build_pattern_cache,
    build_sequence,
    compute_response,
    entries_by_repeat,
)


class AmplitudeLevelCalibrationTests(unittest.TestCase):
    def test_sequence_is_randomized_and_bracketed_by_anchors(self):
        levels = build_amplitude_levels(9)
        entries = build_sequence(levels, repeats=4, seed=17)
        blocks = entries_by_repeat(entries)
        self.assertEqual(len(blocks), 4)
        self.assertEqual(len(entries), 4 * (9 + 4))
        randomized_orders = []
        for block in blocks:
            self.assertEqual(block[0].kind, "dark_anchor")
            self.assertEqual(block[1].kind, "white_anchor")
            self.assertEqual(block[-2].kind, "white_anchor")
            self.assertEqual(block[-1].kind, "dark_anchor")
            level_entries = [entry for entry in block if entry.kind == "level"]
            self.assertEqual(
                sorted(entry.commanded_amplitude for entry in level_entries),
                levels.tolist(),
            )
            randomized_orders.append(
                tuple(entry.commanded_amplitude for entry in level_entries)
            )
        self.assertGreater(len(set(randomized_orders)), 1)

    def test_pattern_cache_encodes_constant_complex_fields(self):
        seen = []

        def fake_encoder(field):
            seen.append(np.array(field, copy=True))
            return np.full(DMD_SHAPE, round(float(np.abs(field[0, 0])) * 255), dtype=np.uint8)

        cache = build_pattern_cache([0.0, 0.5, 1.0], phase_rad=np.pi / 3, encoder=fake_encoder)
        self.assertEqual(sorted(cache), [0.0, 0.5, 1.0])
        self.assertEqual(cache[0.5].shape, DMD_SHAPE)
        for field in seen:
            np.testing.assert_allclose(field, field[0, 0])

    def test_synthetic_quadratic_response_recovers_amplitude(self):
        levels = build_amplitude_levels(11)
        entries = build_sequence(levels, repeats=12, seed=9)
        rng = np.random.default_rng(123)
        frames = np.empty((len(entries),) + CAMERA_SHAPE, dtype=np.float32)
        for entry in entries:
            intensity = 7.0 + 180.0 * entry.commanded_amplitude**2
            frames[entry.sequence_index] = intensity + rng.normal(
                0.0, 0.7, size=CAMERA_SHAPE
            )

        _, rows, summary = compute_response(frames, entries, sensor_max=255.0)
        self.assertLess(summary["amplitude_rmse"], 0.01)
        self.assertLess(abs(summary["intensity_power_law_exponent"] - 2.0), 0.05)
        self.assertEqual(summary["monotonic_violation_count"], 0)
        measured = np.asarray([row["measured_amplitude_mean"] for row in rows])
        np.testing.assert_allclose(measured, levels, atol=0.015)

    def test_inverse_lut_is_monotonic(self):
        rows = [
            {"commanded_amplitude": value, "measured_amplitude_mean": value**1.4}
            for value in np.linspace(0.0, 1.0, 17)
        ]
        lut = build_inverse_lut(rows, lut_size=64)
        commanded = np.asarray(lut["commanded_amplitude"])
        self.assertEqual(commanded.size, 64)
        self.assertTrue(np.all(np.diff(commanded) >= 0))
        self.assertAlmostEqual(float(commanded[0]), 0.0)
        self.assertAlmostEqual(float(commanded[-1]), 1.0)

    def test_report_artifacts_are_written(self):
        levels = build_amplitude_levels(5)
        entries = build_sequence(levels, repeats=3, seed=2)
        frames = np.empty((len(entries),) + CAMERA_SHAPE, dtype=np.float32)
        for entry in entries:
            frames[entry.sequence_index].fill(
                5.0 + 100.0 * entry.commanded_amplitude**2
            )
        with tempfile.TemporaryDirectory() as directory:
            summary = analyze_and_save(
                frames,
                entries,
                directory,
                sensor_max=255.0,
                distinguishability_z=3.0,
            )
            self.assertLess(summary["amplitude_rmse"], 1e-6)
            for filename in (
                "per_capture.csv",
                "amplitude_response.csv",
                "amplitude_lut.json",
                "experiment_summary.json",
                "amplitude_response.png",
            ):
                self.assertTrue(os.path.isfile(os.path.join(directory, filename)))


if __name__ == "__main__":
    unittest.main()
