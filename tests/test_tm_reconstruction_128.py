import json
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

from calibrate_128x128 import Application, DMDController
from analyze_fourier_probe_recovery_128 import (
    complex_fidelity,
    recover_first_order,
)
from dmd_pattern_128 import (
    ACTIVE_HEIGHT,
    ACTIVE_WIDTH,
    ACTIVE_X,
    ACTIVE_Y,
    DMD_HEIGHT,
    DMD_WIDTH,
    HOLOGRAM_SUPERPIXEL_SIZE,
    INPUT_MACRO_PIXEL_SIZE,
    active_region_mask,
    get_superpixel_lut,
    input_field_to_dmd_pattern,
)
from partial_tm_focus_128 import load_partial_tm_row
from pixelwise_focus_report_128 import (
    build_pixelwise_points,
    save_pixelwise_focus_report,
)
from tm_reconstruction_128 import (
    ReconstructionConfig,
    ggs21_block,
    reconstruct_tm,
)


class TMReconstructionTests(unittest.TestCase):
    def test_measurement_quality_dispatch_does_not_block_one_click(self):
        app = Application.__new__(Application)
        app._measurement_quality_thread = None
        started = threading.Event()
        release = threading.Event()

        def fake_analysis(completed_status=None):
            self.assertEqual(completed_status, "quality-ready")
            started.set()
            self.assertTrue(release.wait(2.0))

        app._analyze_and_queue_measurement_quality = fake_analysis
        started_at = time.perf_counter()
        thread = app._start_measurement_quality_analysis("quality-ready")
        dispatch_seconds = time.perf_counter() - started_at

        self.assertLess(dispatch_seconds, 0.5)
        self.assertTrue(started.wait(1.0))
        self.assertTrue(thread.is_alive())
        self.assertIs(
            app._start_measurement_quality_analysis("ignored"), thread
        )

        release.set()
        thread.join(2.0)
        self.assertFalse(thread.is_alive())
        self.assertIsNone(app._measurement_quality_thread)

    def test_active512_first_order_recovers_random_probe(self):
        rng = np.random.default_rng(12804)
        phase_index = rng.integers(
            0, 16, size=(128, 128), dtype=np.uint8
        )
        probe = np.exp(
            1j * phase_index.astype(np.float32) * np.pi / 8.0
        ).astype(np.complex64)
        full_pattern = input_field_to_dmd_pattern(
            probe,
            px=4,
            ds_method="mean",
            lut_cache=get_superpixel_lut(4),
        )
        active_pattern = (
            full_pattern[
                ACTIVE_Y : ACTIVE_Y + ACTIVE_HEIGHT,
                ACTIVE_X : ACTIVE_X + ACTIVE_WIDTH,
            ].astype(np.float32)
            / 255.0
        )
        recovered = recover_first_order(
            active_pattern,
            radius=96,
            order_sign=-1,
        )["recovered_probe"]
        self.assertGreater(complex_fidelity(probe, recovered), 0.98)

    def test_active512_mapping_is_centred_and_zero_padded(self):
        self.assertEqual(INPUT_MACRO_PIXEL_SIZE, HOLOGRAM_SUPERPIXEL_SIZE)
        self.assertEqual((ACTIVE_HEIGHT, ACTIVE_WIDTH), (512, 512))
        self.assertEqual((ACTIVE_X, ACTIVE_Y), (256, 128))

        canvas = active_region_mask()
        self.assertEqual(canvas.shape, (DMD_HEIGHT, DMD_WIDTH))
        self.assertEqual(canvas.dtype, np.uint8)
        self.assertTrue(
            np.all(
                canvas[
                    ACTIVE_Y : ACTIVE_Y + ACTIVE_HEIGHT,
                    ACTIVE_X : ACTIVE_X + ACTIVE_WIDTH,
                ]
                == 255
            )
        )
        self.assertEqual(int(np.count_nonzero(canvas[:, :ACTIVE_X])), 0)
        self.assertEqual(
            int(np.count_nonzero(canvas[:, ACTIVE_X + ACTIVE_WIDTH :])), 0
        )
        self.assertEqual(int(np.count_nonzero(canvas[:ACTIVE_Y, :])), 0)
        self.assertEqual(
            int(np.count_nonzero(canvas[ACTIVE_Y + ACTIVE_HEIGHT :, :])), 0
        )

    def test_full_pixelwise_defaults_cover_all_128_by_128_points(self):
        points = build_pixelwise_points(128, 128, stride=1, max_points=None)
        self.assertEqual(len(points), 16384)
        self.assertEqual(points[0], (0, 0))
        self.assertEqual(points[127], (127, 0))
        self.assertEqual(points[128], (0, 1))
        self.assertEqual(points[-1], (127, 127))

    def test_pixelwise_capture_uses_three_by_three_target_max(self):
        class FakeCamera:
            roi_width = 8
            roi_height = 8

            def __init__(self):
                self.started = False

            def start(self):
                self.started = True

            def stop(self):
                self.started = False

            def run(self):
                image = np.full((8, 8), 10, dtype=np.uint16)
                image[4, 3] = 100
                image[4, 5] = 50
                return image, 0.0, 0.0

        class FakeDMD:
            def juoptProjection(self, *_):
                return 0

            def juoptStop(self, *_):
                return 0

        controller = DMDController.__new__(DMDController)
        controller.camera = FakeCamera()
        controller.DMD = FakeDMD()
        controller.dev_id = 0
        controller.dmd_height = 4
        controller.dmd_width = 4
        controller.load_pattern = lambda _: True
        controller.clear_sequence = lambda _: None
        controller._input_field_to_dmd_pattern = (
            lambda field, **_: np.zeros((8, 8), dtype=np.uint8)
        )
        result = controller._capture_focus_for_tm_row(
            np.ones(16, dtype=np.complex64),
            target_x=3,
            target_y=4,
        )
        self.assertTrue(result["success"], result["error"])
        expected_background = 590.0 / 55.0
        self.assertEqual(result["target_intensity"], 100.0)
        self.assertAlmostEqual(
            result["background_intensity"], expected_background
        )
        self.assertEqual(result["peak_position"], (3, 4))
        self.assertEqual(result["peak_distance_px"], 0.0)
        self.assertAlmostEqual(result["pbr"], 100.0 / expected_background)
        self.assertFalse(controller.camera.started)

    def test_focus_analysis_uses_target_max_and_excludes_exact_region_at_edge(self):
        class FakeCamera:
            roi_width = 8
            roi_height = 8

        controller = DMDController.__new__(DMDController)
        controller.camera = FakeCamera()
        image = np.full((8, 8), 10, dtype=np.uint16)
        image[0, 0] = 100
        image[2, 2] = 50

        result = controller._analyze_pixelwise_focus_image(image, 0, 0)

        expected_background = 640.0 / 60.0
        self.assertEqual(result["target_intensity"], 100.0)
        self.assertAlmostEqual(
            result["background_intensity"], expected_background
        )
        self.assertAlmostEqual(result["pbr"], 100.0 / expected_background)

    def test_partial_tm_focus_uses_three_by_three_target_max(self):
        class FakeCamera:
            roi_width = 8
            roi_height = 8

            def __init__(self):
                self.started = False

            def start(self):
                self.started = True

            def stop(self):
                self.started = False

            def run(self):
                image = np.full((8, 8), 10, dtype=np.uint16)
                image[4, 3] = 100
                return image, 0.0, 0.0

        class FakeDMD:
            def juoptProjection(self, *_):
                return 0

            def juoptStop(self, *_):
                return 0

        controller = DMDController.__new__(DMDController)
        controller.camera = FakeCamera()
        controller.DMD = FakeDMD()
        controller.dev_id = 0
        controller.dmd_height = 2
        controller.dmd_width = 2
        controller.load_pattern = lambda _: True
        controller.clear_sequence = lambda _: None
        controller._input_field_to_dmd_pattern = (
            lambda field, **_: np.zeros((8, 8), dtype=np.uint8)
        )

        mapping = {"target_index": 35, "partial_row_index": 0}
        with patch(
            "calibrate_128x128.load_partial_tm_row",
            return_value=(np.ones(4, dtype=np.complex64), mapping),
        ), patch("calibrate_128x128.time.sleep"):
            result = controller.conjugate_focus_with_partial_tm(3, 4)

        self.assertTrue(result["success"], result["error"])
        self.assertEqual(result["target_intensity"], 100.0)
        self.assertEqual(result["background_intensity"], 10.0)
        self.assertEqual(result["pbr"], 10.0)
        self.assertFalse(controller.camera.started)

    def test_pixelwise_scan_publishes_each_successful_camera_frame(self):
        class FakeCamera:
            roi_width = 2
            roi_height = 2

        controller = DMDController.__new__(DMDController)
        controller.camera = FakeCamera()
        controller.dmd_height = 1
        controller.dmd_width = 1
        controller.original_height = 2
        controller.original_width = 2
        controller._load_transmission_matrix = lambda: np.ones(
            (4, 1), dtype=np.complex64
        )

        def fake_encode(rows, **kwargs):
            callback = kwargs.get("progress_callback")
            if callback:
                callback(len(rows), len(rows))
            return (
                np.zeros((len(rows), 2, 2), dtype=np.uint8),
                [None] * len(rows),
            )

        captured_batch_sizes = []

        def fake_capture(patterns):
            start = sum(captured_batch_sizes)
            captured_batch_sizes.append(len(patterns))
            return [
                np.full((2, 2), start + index, dtype=np.uint16)
                for index in range(len(patterns))
            ]

        controller._build_focus_hologram_batch = fake_encode
        controller._project_and_capture_focus_batch = fake_capture
        frames = []
        with tempfile.TemporaryDirectory() as directory, patch(
            "calibrate_128x128.get_superpixel_lut", return_value=None
        ):
            result = controller.pixelwise_focus_average_pbr(
                stride=1,
                max_points=None,
                batch_size=3,
                output_dir=directory,
                frame_callback=lambda image, record: frames.append(
                    (np.array(image, copy=True), dict(record))
                ),
            )
            raw_images = np.load(result["raw_images_path"], mmap_mode="r")
            raw_shape = raw_images.shape
            raw_dtype = raw_images.dtype
            raw_last_value = int(raw_images[3, 0, 0])
            del raw_images

        self.assertTrue(result["success"])
        self.assertEqual(result["total"], 4)
        self.assertEqual(captured_batch_sizes, [3, 1])
        self.assertEqual(len(frames), 4)
        self.assertEqual(
            [(record["x"], record["y"]) for _, record in frames],
            [(0, 0), (1, 0), (0, 1), (1, 1)],
        )
        self.assertEqual(int(frames[-1][0][0, 0]), 3)
        self.assertEqual(raw_shape, (4, 2, 2))
        self.assertEqual(raw_dtype, np.dtype(np.uint8))
        self.assertEqual(raw_last_value, 3)

    def test_pixelwise_focus_report_outputs_maps_and_figures(self):
        records = []
        sample_index = 0
        for y in (0, 4, 8):
            for x in (0, 4, 8, 12):
                sample_index += 1
                pbr = 2.0 + 0.1 * x + 0.05 * y
                records.append(
                    {
                        "sample_index": sample_index,
                        "x": x,
                        "y": y,
                        "target_index": y * 16 + x,
                        "success": True,
                        "target_intensity": 100.0 * pbr,
                        "peak_intensity": 100.0 * pbr,
                        "peak_x": x,
                        "peak_y": y,
                        "peak_distance_px": 0.0,
                        "image_min_intensity": 10.0,
                        "image_max_intensity": 100.0 * pbr,
                        "image_intensity_range": 100.0 * pbr - 10.0,
                        "mean_intensity": 50.0,
                        "background_intensity": 100.0,
                        "pbr": pbr,
                        "error": None,
                    }
                )
        with tempfile.TemporaryDirectory() as directory:
            summary = save_pixelwise_focus_report(
                records,
                roi_shape=(16, 16),
                output_dir=directory,
                run_label="unit",
            )
            self.assertEqual(summary["successful_points"], 12)
            for path in summary["files"].values():
                self.assertTrue(os.path.isfile(path), path)
            maps = np.load(summary["files"]["maps_npz"])
            self.assertEqual(maps["pbr"].shape, (16, 16))
            self.assertAlmostEqual(float(maps["pbr"][4, 8]), 3.0)
            self.assertAlmostEqual(
                float(maps["image_min_intensity"][4, 8]), 10.0
            )
            self.assertAlmostEqual(
                float(maps["image_max_intensity"][4, 8]), 300.0
            )
            self.assertAlmostEqual(
                float(maps["image_intensity_range"][4, 8]), 290.0
            )
            self.assertTrue(np.isnan(maps["pbr"][1, 1]))
            maps.close()

    def test_partial_tm_coordinate_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            tm_path = os.path.join(directory, "partial.npy")
            metadata_path = os.path.join(directory, "partial.json")
            partial = np.arange(4 * 16, dtype=np.float32).reshape(4, 16).astype(
                np.complex64
            )
            np.save(tm_path, partial)
            with open(metadata_path, "w", encoding="utf-8") as handle:
                json.dump({"selected_output_range": [8224, 8228]}, handle)

            row, info = load_partial_tm_row(
                tm_path,
                metadata_path,
                target_x=34,
                target_y=64,
                roi_shape=(128, 128),
                input_count=16,
            )
            np.testing.assert_array_equal(row, partial[2])
            self.assertEqual(info["target_index"], 8226)
            self.assertEqual(info["partial_row_index"], 2)
            with self.assertRaisesRegex(ValueError, "was not reconstructed"):
                load_partial_tm_row(
                    tm_path,
                    metadata_path,
                    target_x=31,
                    target_y=64,
                    roi_shape=(128, 128),
                    input_count=16,
                )

    def test_ggs21_recovers_synthetic_rows(self):
        torch.manual_seed(7)
        measurement_count = 256
        input_count = 32
        output_count = 4
        phase = torch.randint(0, 16, (measurement_count, input_count))
        x = torch.exp(1j * phase.float() * torch.pi / 8).to(torch.complex64)
        x /= np.sqrt(measurement_count)
        true_h = (
            torch.randn(input_count, output_count)
            + 1j * torch.randn(input_count, output_count)
        ).to(torch.complex64)
        amplitude = torch.abs(x @ true_h)
        gram = x.mH @ x
        gram.diagonal().add_(1e-5)
        factor = torch.linalg.cholesky(gram)

        recovered, errors = ggs21_block(
            x,
            amplitude,
            iterations=160,
            gs2_ratio=0.75,
            solver="cholesky",
            cholesky_factor=factor,
            random_seed=19,
        )

        recovered = recovered.T
        numerator = torch.abs(torch.sum(true_h.conj() * recovered, dim=0))
        denominator = (
            torch.linalg.vector_norm(true_h, dim=0)
            * torch.linalg.vector_norm(recovered, dim=0)
        )
        fidelity = numerator / denominator
        self.assertGreater(float(torch.mean(fidelity)), 0.90)
        self.assertLess(float(errors[-1]), float(errors[0]))

    def test_end_to_end_writes_standard_npy_and_metadata(self):
        rng = np.random.default_rng(11)
        measurement_count = 64
        input_shape = (4, 4)
        output_shape = (2, 3)
        input_count = int(np.prod(input_shape))
        output_count = int(np.prod(output_shape))

        phase_index = rng.integers(
            0, 16, size=(measurement_count,) + input_shape, dtype=np.uint8
        )
        probes = np.exp(1j * phase_index.astype(np.float32) * np.pi / 8).astype(
            np.complex64
        )
        true_h = (
            rng.standard_normal((output_count, input_count))
            + 1j * rng.standard_normal((output_count, input_count))
        ).astype(np.complex64)
        field = probes.reshape(measurement_count, input_count) @ true_h.T
        intensity = np.rint(np.abs(field) ** 2 * 100).astype(np.uint16)

        with tempfile.TemporaryDirectory() as directory:
            probe_path = os.path.join(directory, "probe.npy")
            measurement_path = os.path.join(directory, "measurements.raw")
            output_path = os.path.join(directory, "tm.npy")
            error_path = os.path.join(directory, "error.npy")
            metadata_path = os.path.join(directory, "metadata.json")
            cache_path = os.path.join(directory, "factor.npy")
            np.save(probe_path, probes)
            intensity.tofile(measurement_path)

            result = reconstruct_tm(
                ReconstructionConfig(
                    measurement_path=measurement_path,
                    probe_path=probe_path,
                    output_path=output_path,
                    error_curve_path=error_path,
                    metadata_path=metadata_path,
                    cholesky_cache_path=cache_path,
                    input_shape=input_shape,
                    output_shape=output_shape,
                    iterations=100,
                    gs2_ratio=0.75,
                    output_chunk_size=2,
                    ridge=1e-5,
                    device="cpu",
                )
            )

            tm = np.load(output_path, mmap_mode="r")
            self.assertEqual(tm.shape, (output_count, input_count))
            self.assertEqual(tm.dtype, np.complex64)
            self.assertTrue(np.all(np.isfinite(tm)))
            del tm
            self.assertTrue(os.path.isfile(error_path))
            self.assertTrue(os.path.isfile(cache_path))
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(result["output_shape"], [output_count, input_count])
            self.assertLess(
                result["final_relative_amplitude_error"],
                result["initial_relative_amplitude_error"],
            )


if __name__ == "__main__":
    unittest.main()
