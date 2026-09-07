import sys
import unittest

from tmcalib.adapters.legacy import LegacyHardwareAdapter
from tmcalib.events import EventKind
from tmcalib.profiles import PROFILES, get_profile
from tmcalib.testing import FakeHardware
from tmcalib.workflow import (
    CalibrationWorkflow,
    RuntimeServices,
    WorkflowState,
)


def build_fake_workflow(profile_name="dense_128x128"):
    fake = FakeHardware()
    services = RuntimeServices(
        camera=fake,
        measurement=fake,
        reconstruction=fake,
        focus=fake,
        lifecycle=fake,
        cancellation=fake,
    )
    return CalibrationWorkflow(get_profile(profile_name), services), fake


class ProfileTests(unittest.TestCase):
    def test_fourfold_128x96_profile_is_available_to_the_unified_gui(self):
        profile = get_profile("fourfold_128x96")
        self.assertEqual(profile.input_shape, (96, 128))
        self.assertEqual(profile.input_macro_pixel_size, 8)
        self.assertEqual(profile.controller_module, "calibrate_128x96")

    def test_every_profile_has_shared_camera_and_workflow_capabilities(self):
        required = {
            "exposure",
            "preview",
            "measurement",
            "reconstruction",
            "focus",
            "pixelwise_report",
        }
        for profile in PROFILES.values():
            self.assertTrue(required.issubset(profile.capabilities), profile.key)

    def test_polarization_selection_is_immutable(self):
        original = get_profile("dense_128x128_roi26")
        selected = get_profile("dense_128x128_roi26", "i90")
        self.assertEqual(original.default_channel, "I0")
        self.assertEqual(selected.default_channel, "I90")

    def test_regular_profile_rejects_channel_override(self):
        with self.assertRaises(ValueError):
            get_profile("dense_128x128", "I0")


class WorkflowInjectionTests(unittest.TestCase):
    def test_operations_require_a_connected_idle_workflow(self):
        workflow, _ = build_fake_workflow()
        with self.assertRaises(RuntimeError):
            workflow.measure()
        workflow.close()

    def test_building_workflow_does_not_import_vendor_modules(self):
        before = set(sys.modules)
        from tmcalib.bootstrap import build_workflow

        workflow = build_workflow("dense_128x128")
        after = set(sys.modules)
        self.assertNotIn("PySpin", after - before)
        self.assertNotIn("calibrate_128x128", after - before)
        workflow.close()

    def test_legacy_true_exposure_result_keeps_requested_value(self):
        class BooleanExposureCamera:
            def configure_exposure(self, exposure_time):
                return True

        adapter = LegacyHardwareAdapter(get_profile("dense_128x128"))
        adapter._camera = BooleanExposureCamera()
        adapter._controller = object()
        self.assertEqual(adapter.set_exposure_us(250.0), 250.0)

    def test_legacy_adapter_starts_direct_measurement_before_first_batch(self):
        class DirectMeasurementController:
            def __init__(self):
                self.optimization_running = False
                self.measurement_completed = False
                self.measurement_error = None
                self.measure_progress_callback = None

            def run_measurement(self):
                # This mirrors the guard at the start of every legacy backend's
                # probe loop.
                return self.optimization_running

        controller = DirectMeasurementController()
        adapter = LegacyHardwareAdapter(get_profile("dense_128x128"))
        adapter._camera = object()
        adapter._controller = controller

        result = adapter.run_measurement(lambda _value, _message: None)

        self.assertTrue(result.success)
        self.assertTrue(controller.measurement_completed)
        self.assertFalse(controller.optimization_running)
        self.assertIsNone(controller.measure_progress_callback)

    def test_legacy_adapter_reports_an_interrupted_measurement_as_failure(self):
        class InterruptedMeasurementController:
            optimization_running = False
            measurement_error = None
            measure_progress_callback = None

            def run_measurement(self):
                self.optimization_running = False
                return False

        controller = InterruptedMeasurementController()
        adapter = LegacyHardwareAdapter(get_profile("dense_128x128"))
        adapter._camera = object()
        adapter._controller = controller

        result = adapter.run_measurement(lambda _value, _message: None)

        self.assertFalse(result.success)
        self.assertIn("stopped", result.message.lower())
        self.assertFalse(controller.optimization_running)

    def test_one_injected_fake_serves_all_ports(self):
        workflow, fake = build_fake_workflow()
        states = []
        progress = []
        frames = []
        workflow.events.subscribe(EventKind.STATE, lambda event: states.append(event.payload))
        workflow.events.subscribe(EventKind.PROGRESS, lambda event: progress.append(event.progress))
        workflow.events.subscribe(EventKind.FRAME, lambda event: frames.append(event.payload))

        workflow.connect().result(timeout=2)
        self.assertEqual(workflow.state, WorkflowState.IDLE)
        self.assertTrue(fake.connected)

        self.assertEqual(workflow.set_exposure_us(250.0), 250.0)
        workflow.start_preview()
        self.assertTrue(frames)

        workflow.measure().result(timeout=2)
        self.assertEqual(frames[-1], [[10, 20], [30, 40]])
        workflow.reconstruct().result(timeout=2)
        result = workflow.focus(2, 3).result(timeout=2)

        self.assertTrue(result.success)
        self.assertEqual(frames[-1], [[50, 60], [70, 80]])
        self.assertEqual(fake.measure_count, 1)
        self.assertEqual(fake.reconstruct_count, 1)
        self.assertEqual(fake.focus_points, [(2, 3)])
        self.assertIn(100.0, progress)
        self.assertIn(WorkflowState.MEASURING, states)
        workflow.close()
        self.assertTrue(fake.closed)

    def test_focus_publishes_captured_image_before_result(self):
        workflow, _ = build_fake_workflow()
        event_order = []
        workflow.events.subscribe(
            EventKind.FRAME,
            lambda event: event_order.append((event.kind, event.operation, event.payload)),
        )
        workflow.events.subscribe(
            EventKind.RESULT,
            lambda event: event_order.append((event.kind, event.operation, event.payload)),
        )
        workflow.connect().result(timeout=2)

        workflow.focus(4, 5).result(timeout=2)

        focus_events = [event for event in event_order if event[1] == "focus"]
        self.assertEqual(focus_events[0][0], EventKind.FRAME)
        self.assertEqual(focus_events[0][2], [[50, 60], [70, 80]])
        self.assertEqual(focus_events[1][0], EventKind.RESULT)
        workflow.close()

    def test_pixelwise_focus_is_a_standalone_operation_with_frames(self):
        workflow, _ = build_fake_workflow()
        frames = []
        progress = []
        states = []
        workflow.events.subscribe(
            EventKind.FRAME,
            lambda event: frames.append((event.operation, event.payload)),
        )
        workflow.events.subscribe(
            EventKind.PROGRESS,
            lambda event: progress.append((event.operation, event.progress)),
        )
        workflow.events.subscribe(
            EventKind.STATE,
            lambda event: states.append(event.payload),
        )
        workflow.connect().result(timeout=2)

        result = workflow.pixelwise_report().result(timeout=2)

        self.assertTrue(result.success)
        self.assertIn(WorkflowState.PIXELWISE_FOCUSING, states)
        self.assertEqual(
            frames[-1],
            ("pixelwise_report", [[90, 100], [110, 120]]),
        )
        self.assertIn(("pixelwise_report", 100.0), progress)
        workflow.close()

    def test_disconnect_releases_hardware_on_the_workflow_worker(self):
        workflow, fake = build_fake_workflow()
        workflow.connect().result(timeout=2)

        result = workflow.disconnect().result(timeout=2)

        self.assertTrue(result.success)
        self.assertTrue(fake.closed)
        self.assertEqual(workflow.state, WorkflowState.CLOSED)

    def test_one_click_is_the_same_sequence_for_every_supported_profile(self):
        for profile_name in ("dense_128x128", "fivefold_160x120"):
            workflow, fake = build_fake_workflow(profile_name)
            workflow.connect().result(timeout=2)
            workflow.one_click().result(timeout=2)
            self.assertEqual(fake.measure_count, 1)
            self.assertEqual(fake.reconstruct_count, 1)
            workflow.close()

    def test_focus_coordinate_is_validated_from_profile(self):
        workflow, _ = build_fake_workflow("dense_128x128_roi26")
        workflow.connect().result(timeout=2)
        with self.assertRaises(ValueError):
            workflow.focus(26, 0)
        workflow.close()


if __name__ == "__main__":
    unittest.main()
