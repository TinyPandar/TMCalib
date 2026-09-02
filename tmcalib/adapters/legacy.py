"""Compatibility adapter around the experiment-tested single-file backends."""

import importlib
import queue
import threading
from typing import Optional

from tmcalib.ports import FrameCallback, OperationResult, ProgressCallback
from tmcalib.profiles import ProfileSpec


class LegacyHardwareAdapter:
    """Expose old controllers through injectable ports without importing Tk."""

    def __init__(self, profile: ProfileSpec) -> None:
        self.profile = profile
        self._camera = None
        self._controller = None
        self._preview_callback: Optional[FrameCallback] = None
        self._preview_stop = threading.Event()
        self._preview_thread: Optional[threading.Thread] = None
        self._exposure_us = float(profile.default_exposure_us)
        self._load_lock = threading.Lock()

    @property
    def camera(self):
        self._ensure_loaded()
        return self._camera

    @property
    def controller(self):
        self._ensure_loaded()
        return self._controller

    def _ensure_loaded(self) -> None:
        if self._controller is not None:
            return
        with self._load_lock:
            if self._controller is not None:
                return

            # calibrate_160x120 is a legacy compatibility module which changes
            # constants in calibrate_128x128 at import time. Reload the shared
            # core before composing a backend so sequential profile switches do
            # not inherit dimensions from the previously selected profile.
            shared_core_name = "calibrate_128x128"
            if (
                self.profile.camera_module == shared_core_name
                or self.profile.controller_module in (shared_core_name, "calibrate_160x120")
            ):
                shared_core = importlib.import_module(shared_core_name)
                shared_core = importlib.reload(shared_core)
            else:
                shared_core = None

            if self.profile.camera_module == shared_core_name:
                camera_module = shared_core
            else:
                camera_module = importlib.import_module(self.profile.camera_module)

            if self.profile.controller_module == shared_core_name:
                controller_module = shared_core
            else:
                controller_module = importlib.import_module(
                    self.profile.controller_module
                )
                if self.profile.controller_module == "calibrate_160x120":
                    controller_module = importlib.reload(controller_module)

            camera_class = camera_module.CameraHandler
            camera_kwargs = {"cam_index": 0, "save_path": "./camera_1"}
            if self.profile.camera_module == shared_core_name:
                camera_kwargs.update(
                    roi_shape=self.profile.camera_roi,
                    polarization_channel=self.profile.default_channel or "I90",
                    exposure_us=self.profile.default_exposure_us,
                    output_tag=(
                        self.profile.default_channel
                        if self.profile.available_channels
                        else None
                    ),
                )
            self._camera = camera_class(**camera_kwargs)
            self._camera.convert_to_12bit = False
            self._controller = controller_module.DMDController(self._camera)

    def connect(self) -> OperationResult:
        controller = self.controller
        devices = controller.get_devices()
        if not devices:
            return OperationResult(False, "No JUOPT DMD device found")
        if not controller.initialize_device(devices[0]):
            return OperationResult(False, "JUOPT DMD initialization failed")
        self.set_exposure_us(self.profile.default_exposure_us)
        return OperationResult(True, "Camera and DMD connected")

    def set_exposure_us(self, value: float) -> float:
        result = self.camera.configure_exposure(exposure_time=float(value))
        if result is False:
            raise RuntimeError("Camera rejected exposure {:.1f} us".format(value))
        if isinstance(result, bool):
            actual = float(value)
        elif isinstance(result, (int, float)):
            actual = float(result)
        else:
            actual = float(value)
        self._exposure_us = actual
        return actual

    def get_exposure_us(self) -> float:
        return self._exposure_us

    def start_preview(self, callback: FrameCallback) -> None:
        self._preview_callback = callback
        if self._preview_thread is not None and self._preview_thread.is_alive():
            return
        self._preview_stop.clear()

        def consume() -> None:
            image_queue = self.camera.image_queue
            while not self._preview_stop.is_set():
                try:
                    payload = image_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                image = payload[1] if isinstance(payload, tuple) and len(payload) == 2 else payload
                current = self._preview_callback
                if current is not None:
                    current(image)

        self._preview_thread = threading.Thread(
            target=consume, name="tmcalib-preview", daemon=True
        )
        self._preview_thread.start()

    def stop_preview(self) -> None:
        self._preview_stop.set()
        self._preview_callback = None

    def close_camera(self) -> None:
        if self._camera is not None:
            self._camera.cleanup()

    def run_measurement(self, progress: ProgressCallback) -> OperationResult:
        controller = self.controller
        controller.measure_progress_callback = progress
        controller.measurement_error = None
        try:
            result = controller.run_measurement()
            error = getattr(controller, "measurement_error", None)
            if error:
                return OperationResult(False, str(error))
            return OperationResult(result is not False, "Measurement completed")
        finally:
            controller.measure_progress_callback = None

    def stop_measurement(self) -> None:
        if self._controller is not None:
            self._controller.stop_optimization()

    def run_reconstruction(self, progress: ProgressCallback) -> OperationResult:
        controller = self.controller
        controller.recon_progress_callback = progress
        try:
            result = controller.run_reconstruction()
            return OperationResult(result is not False, "TM reconstruction completed")
        finally:
            controller.recon_progress_callback = None

    def stop_reconstruction(self) -> None:
        if self._controller is not None:
            self._controller.reconstruction_running = False

    def focus(self, x: int, y: int) -> OperationResult:
        result = self.controller.conjugate_focus_at_position(x, y)
        if isinstance(result, dict):
            return OperationResult(
                bool(result.get("success", True)),
                result.get("error") or "Focus completed",
                payload=result,
            )
        return OperationResult(result is not False, "Focus completed", payload=result)

    def run_pixelwise_report(self, progress: ProgressCallback) -> OperationResult:
        result = self.controller.pixelwise_focus_average_pbr(
            progress_callback=lambda done, total, message: progress(
                100.0 * done / total if total else 0.0, message
            )
        )
        if isinstance(result, dict):
            return OperationResult(
                bool(result.get("success", True)),
                result.get("error") or "Pixel-wise report completed",
                payload=result,
            )
        return OperationResult(result is not False, "Pixel-wise report completed", result)

    def stop_all(self) -> None:
        self.stop_measurement()
        self.stop_reconstruction()

    def close(self) -> None:
        self.stop_preview()
        if self._controller is not None:
            self._controller.cleanup()
        self.close_camera()
