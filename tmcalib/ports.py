"""Dependency-inversion ports consumed by the application workflow."""

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Protocol


FrameCallback = Callable[[Any], None]
ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True)
class OperationResult:
    success: bool
    message: str = ""
    payload: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class CameraPort(Protocol):
    def connect(self) -> OperationResult: ...
    def set_exposure_us(self, value: float) -> float: ...
    def get_exposure_us(self) -> float: ...
    def start_preview(self, callback: FrameCallback) -> None: ...
    def stop_preview(self) -> None: ...
    def close_camera(self) -> None: ...


class MeasurementPort(Protocol):
    def run_measurement(self, progress: ProgressCallback) -> OperationResult: ...
    def stop_measurement(self) -> None: ...


class ReconstructionPort(Protocol):
    def run_reconstruction(self, progress: ProgressCallback) -> OperationResult: ...
    def stop_reconstruction(self) -> None: ...


class FocusPort(Protocol):
    def focus(self, x: int, y: int) -> OperationResult: ...
    def run_pixelwise_report(self, progress: ProgressCallback) -> OperationResult: ...


class LifecyclePort(Protocol):
    def close(self) -> None: ...


class CancellationPort(Protocol):
    def stop_all(self) -> None: ...
