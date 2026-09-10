"""One serialized calibration workflow shared by every user interface."""

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Tuple

from tmcalib.events import EventBus, EventKind, WorkflowEvent
from tmcalib.ports import (
    CameraPort,
    CancellationPort,
    FocusPort,
    LifecyclePort,
    MeasurementPort,
    OperationResult,
    ReconstructionPort,
)
from tmcalib.profiles import ProfileSpec


class WorkflowState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    IDLE = "idle"
    MEASURING = "measuring"
    RECONSTRUCTING = "reconstructing"
    FOCUSING = "focusing"
    PIXELWISE_FOCUSING = "pixelwise_focusing"
    ONE_CLICK = "one_click"
    STOPPING = "stopping"
    ERROR = "error"
    CLOSED = "closed"


@dataclass(frozen=True)
class RuntimeServices:
    camera: CameraPort
    measurement: MeasurementPort
    reconstruction: ReconstructionPort
    focus: FocusPort
    lifecycle: LifecyclePort
    cancellation: CancellationPort


class CalibrationWorkflow:
    """Application use cases with no Qt, Tk, PySpin, or JUOPT imports."""

    def __init__(
        self,
        profile: ProfileSpec,
        services: RuntimeServices,
        events: Optional[EventBus] = None,
        executor: Optional[ThreadPoolExecutor] = None,
    ) -> None:
        self.profile = profile
        self.services = services
        self.events = events or EventBus()
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tmcalib-workflow"
        )
        self._owns_executor = executor is None
        self._state = WorkflowState.DISCONNECTED
        self._lock = threading.RLock()
        self._active_future: Optional[Future] = None

    @property
    def state(self) -> WorkflowState:
        with self._lock:
            return self._state

    def _set_state(self, state: WorkflowState, operation: str, message: str = "") -> None:
        with self._lock:
            self._state = state
        self.events.publish(
            WorkflowEvent(EventKind.STATE, operation, message=message, payload=state)
        )

    def _log(self, operation: str, message: str) -> None:
        self.events.publish(WorkflowEvent(EventKind.LOG, operation, message=message))

    def _progress(self, operation: str) -> Callable[[float, str], None]:
        def publish(value: float, message: str) -> None:
            normalized = max(0.0, min(100.0, float(value)))
            self.events.publish(
                WorkflowEvent(
                    EventKind.PROGRESS,
                    operation,
                    message=message,
                    progress=normalized,
                )
            )

        return publish

    def _frame(self, operation: str) -> Callable[[object], None]:
        def publish(frame) -> None:
            self.events.publish(
                WorkflowEvent(EventKind.FRAME, operation, payload=frame)
            )

        return publish

    def _submit(
        self,
        operation: str,
        running_state: WorkflowState,
        function: Callable[[], OperationResult],
        success_state: WorkflowState = WorkflowState.IDLE,
        allowed_states: Tuple[WorkflowState, ...] = (WorkflowState.IDLE,),
    ) -> Future:
        with self._lock:
            if self._state not in allowed_states:
                raise RuntimeError(
                    "Cannot start {} while workflow state is {}".format(
                        operation, self._state.value
                    )
                )
            self._set_state(running_state, operation)

        def guarded() -> OperationResult:
            self._log(operation, "{} started".format(operation))
            try:
                result = function()
                if not isinstance(result, OperationResult):
                    result = OperationResult(bool(result))
                if not result.success:
                    raise RuntimeError(result.message or "{} failed".format(operation))
                self.events.publish(
                    WorkflowEvent(
                        EventKind.RESULT,
                        operation,
                        message=result.message,
                        payload=result.payload,
                        metadata=result.metadata,
                    )
                )
                self._set_state(success_state, operation, result.message)
                self._log(operation, "{} completed".format(operation))
                return result
            except Exception as exc:
                message = str(exc)
                self._set_state(WorkflowState.ERROR, operation, message)
                self.events.publish(
                    WorkflowEvent(EventKind.ERROR, operation, message=message)
                )
                raise

        future = self._executor.submit(guarded)
        with self._lock:
            self._active_future = future
        return future

    def connect(self) -> Future:
        return self._submit(
            "connect",
            WorkflowState.CONNECTING,
            self.services.camera.connect,
            success_state=WorkflowState.IDLE,
            allowed_states=(WorkflowState.DISCONNECTED,),
        )

    def set_exposure_us(self, value: float) -> float:
        if not self.profile.supports("exposure"):
            raise RuntimeError("Exposure control is unavailable for this profile")
        if value <= 0:
            raise ValueError("Exposure must be positive")
        actual = float(self.services.camera.set_exposure_us(float(value)))
        self._log("exposure", "Exposure set to {:.1f} us".format(actual))
        return actual

    def set_reconstruction_algorithm(self, name: str) -> str:
        """Select a backend-specific recovery method while the workflow is idle."""
        setter = getattr(self.services.reconstruction, "set_reconstruction_algorithm", None)
        if setter is None:
            raise RuntimeError(
                "This reconstruction backend does not expose algorithm selection"
            )
        if self.state not in (
            WorkflowState.DISCONNECTED,
            WorkflowState.IDLE,
            WorkflowState.ERROR,
        ):
            raise RuntimeError(
                "Cannot change the reconstruction algorithm while workflow state is {}".format(
                    self.state.value
                )
            )
        algorithm = setter(name)
        self._log("reconstruction", "Recovery algorithm set to {}".format(algorithm))
        return algorithm

    def start_preview(self) -> None:
        if not self.profile.supports("preview"):
            raise RuntimeError("Preview is unavailable for this profile")

        def on_frame(frame) -> None:
            self.events.publish(
                WorkflowEvent(EventKind.FRAME, "preview", payload=frame)
            )

        self.services.camera.start_preview(on_frame)
        self._log("preview", "Camera preview enabled")

    def stop_preview(self) -> None:
        self.services.camera.stop_preview()
        self._log("preview", "Camera preview disabled")

    def measure(self) -> Future:
        if not self.profile.supports("measurement"):
            raise RuntimeError("Measurement is unavailable for this profile")
        return self._submit(
            "measurement",
            WorkflowState.MEASURING,
            lambda: self.services.measurement.run_measurement(
                self._progress("measurement"),
                self._frame("measurement"),
            ),
        )

    def reconstruct(self) -> Future:
        if not self.profile.supports("reconstruction"):
            raise RuntimeError("Reconstruction is unavailable for this profile")
        return self._submit(
            "reconstruction",
            WorkflowState.RECONSTRUCTING,
            lambda: self.services.reconstruction.run_reconstruction(
                self._progress("reconstruction")
            ),
        )

    def focus(self, x: int, y: int) -> Future:
        if not self.profile.supports("focus"):
            raise RuntimeError("Focusing is unavailable for this profile")
        roi_h, roi_w = self.profile.camera_roi
        if not (0 <= x < roi_w and 0 <= y < roi_h):
            raise ValueError(
                "Focus coordinate ({}, {}) is outside {}x{} ROI".format(
                    x, y, roi_w, roi_h
                )
            )
        def run() -> OperationResult:
            result = self.services.focus.focus(x, y)
            if isinstance(result, OperationResult) and isinstance(
                result.payload, dict
            ):
                focused_image = result.payload.get("focused_image")
                if focused_image is not None:
                    self._frame("focus")(focused_image)
            return result

        return self._submit("focus", WorkflowState.FOCUSING, run)

    def one_click(self) -> Future:
        if not self.profile.supports("one_click"):
            raise RuntimeError("One-click calibration is unavailable for this profile")

        def run() -> OperationResult:
            measured = self.services.measurement.run_measurement(
                self._progress("measurement"),
                self._frame("measurement"),
            )
            if not measured.success:
                return measured
            reconstructed = self.services.reconstruction.run_reconstruction(
                self._progress("reconstruction")
            )
            if not reconstructed.success:
                return reconstructed
            report = self.services.focus.run_pixelwise_report(
                self._progress("pixelwise_report"),
                self._frame("pixelwise_report"),
            )
            return report

        return self._submit("one_click", WorkflowState.ONE_CLICK, run)

    def pixelwise_report(self) -> Future:
        if not self.profile.supports("pixelwise_report"):
            raise RuntimeError(
                "Pixel-wise focusing is unavailable for this profile"
            )
        return self._submit(
            "pixelwise_report",
            WorkflowState.PIXELWISE_FOCUSING,
            lambda: self.services.focus.run_pixelwise_report(
                self._progress("pixelwise_report"),
                self._frame("pixelwise_report"),
            ),
        )

    def stop(self) -> None:
        current = self.state
        if current in (WorkflowState.DISCONNECTED, WorkflowState.IDLE, WorkflowState.CLOSED):
            return
        self._set_state(WorkflowState.STOPPING, "stop")
        self.services.cancellation.stop_all()
        self._log("stop", "Stop requested; waiting for the active operation")

    def disconnect(self) -> Future:
        """Release hardware on the serialized worker without blocking the UI."""
        with self._lock:
            if self._state not in (WorkflowState.IDLE, WorkflowState.ERROR):
                raise RuntimeError(
                    "Cannot disconnect while workflow state is {}".format(
                        self._state.value
                    )
                )
            self._set_state(WorkflowState.STOPPING, "disconnect")

        def release() -> OperationResult:
            self._log("disconnect", "Releasing camera and DMD")
            try:
                self.services.cancellation.stop_all()
                self.services.camera.stop_preview()
                self.services.lifecycle.close()
                result = OperationResult(True, "Camera and DMD disconnected")
                self._set_state(WorkflowState.CLOSED, "disconnect", result.message)
                self._log("disconnect", "Camera and DMD released")
                return result
            except Exception as exc:
                message = str(exc)
                self._set_state(WorkflowState.ERROR, "disconnect", message)
                self.events.publish(
                    WorkflowEvent(EventKind.ERROR, "disconnect", message=message)
                )
                raise

        future = self._executor.submit(release)
        with self._lock:
            self._active_future = future
        if self._owns_executor:
            future.add_done_callback(
                lambda _future: self._executor.shutdown(wait=False)
            )
        return future

    def close(self) -> None:
        if self.state == WorkflowState.CLOSED:
            return
        self.services.cancellation.stop_all()
        self.services.camera.stop_preview()
        self.services.lifecycle.close()
        self._set_state(WorkflowState.CLOSED, "close")
        if self._owns_executor:
            self._executor.shutdown(wait=False)
