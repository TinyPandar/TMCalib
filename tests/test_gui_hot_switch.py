import os
import time
import unittest
from unittest import mock


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from tmcalib.profiles import get_profile
from tmcalib.testing import FakeHardware
from tmcalib.workflow import CalibrationWorkflow, RuntimeServices, WorkflowState
from tmcalib_gui.app import MainWindow


class _WorkflowStub:
    def __init__(self, profile_name, state=WorkflowState.IDLE, channel=None):
        self.profile = get_profile(profile_name, channel)
        self.state = state


class GuiHotSwitchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()

    def tearDown(self):
        # The test stub is not a complete workflow and must not enter close().
        self.window.workflow = None
        self.window.close()

    def _wait_until(self, predicate, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return
            time.sleep(0.005)
        self.fail("Timed out waiting for Qt workflow state")

    @staticmethod
    def _fake_workflow(profile_name, channel=None):
        fake = FakeHardware()
        services = RuntimeServices(
            camera=fake,
            measurement=fake,
            reconstruction=fake,
            focus=fake,
            lifecycle=fake,
            cancellation=fake,
        )
        return CalibrationWorkflow(get_profile(profile_name, channel), services), fake

    def test_idle_profile_change_requests_safe_reconnect(self):
        self.window.workflow = _WorkflowStub("v4_32x24")
        self.window._refresh_for_state(WorkflowState.IDLE)
        self.assertTrue(self.window.profile_combo.isEnabled())

        target = self.window.profile_combo.findData("fourfold_128x96")
        with mock.patch.object(self.window, "_switch_configuration") as reconnect:
            self.window.profile_combo.setCurrentIndex(target)

        reconnect.assert_called_once_with()

    def test_busy_workflow_locks_configuration(self):
        self.window.workflow = _WorkflowStub(
            "v4_32x24", state=WorkflowState.MEASURING
        )
        self.window._refresh_for_state(WorkflowState.MEASURING)

        self.assertFalse(self.window.profile_combo.isEnabled())
        self.assertFalse(self.window.channel_combo.isEnabled())

    def test_idle_profile_enables_standalone_pixelwise_button(self):
        self.window.workflow = _WorkflowStub("fourfold_128x96")
        self.window._refresh_for_state(WorkflowState.IDLE)

        self.assertEqual(self.window.pixelwise_button.text(), "3. 逐点聚焦")
        self.assertTrue(self.window.pixelwise_button.isEnabled())

    def test_idle_channel_change_requests_safe_reconnect(self):
        target = self.window.profile_combo.findData("dense_128x128_roi26")
        self.window.profile_combo.setCurrentIndex(target)
        self.window.workflow = _WorkflowStub(
            "dense_128x128_roi26", channel="I0"
        )
        self.window._refresh_for_state(WorkflowState.IDLE)
        self.assertTrue(self.window.channel_combo.isEnabled())

        with mock.patch.object(self.window, "_switch_configuration") as reconnect:
            self.window.channel_combo.setCurrentText("I90")

        reconnect.assert_called_once_with()

    def test_hot_switch_releases_old_backend_then_connects_new_backend(self):
        created = []

        def build(profile_name, channel=None):
            workflow, fake = self._fake_workflow(profile_name, channel)
            created.append((workflow, fake))
            return workflow

        with mock.patch("tmcalib_gui.app.build_workflow", side_effect=build):
            self.window._connect_hardware()
            self._wait_until(
                lambda: self.window.workflow is not None
                and self.window.workflow.state == WorkflowState.IDLE
            )

            old_fake = created[0][1]
            target = self.window.profile_combo.findData("fourfold_128x96")
            self.window.profile_combo.setCurrentIndex(target)
            self._wait_until(
                lambda: len(created) == 2
                and self.window.workflow is created[1][0]
                and self.window.workflow.state == WorkflowState.IDLE
            )

        self.assertTrue(old_fake.closed)
        self.assertEqual(
            self.window.workflow.profile.key, "fourfold_128x96"
        )


if __name__ == "__main__":
    unittest.main()
