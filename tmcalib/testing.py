"""Fast dependency-injection fakes for workflow and GUI tests."""

from typing import List

from tmcalib.ports import OperationResult, ProgressCallback


class FakeHardware:
    def __init__(self) -> None:
        self.connected = False
        self.closed = False
        self.previewing = False
        self.exposure_us = 1000.0
        self.measure_count = 0
        self.reconstruct_count = 0
        self.focus_points: List[tuple] = []
        self.stop_count = 0

    def connect(self) -> OperationResult:
        self.connected = True
        return OperationResult(True, "Fake hardware connected")

    def set_exposure_us(self, value: float) -> float:
        self.exposure_us = float(value)
        return self.exposure_us

    def get_exposure_us(self) -> float:
        return self.exposure_us

    def start_preview(self, callback) -> None:
        self.previewing = True
        callback([[1, 2], [3, 4]])

    def stop_preview(self) -> None:
        self.previewing = False

    def close_camera(self) -> None:
        self.closed = True

    def run_measurement(self, progress: ProgressCallback, frame_callback=None) -> OperationResult:
        self.measure_count += 1
        if frame_callback is not None:
            frame_callback([[10, 20], [30, 40]])
        progress(50, "half")
        progress(100, "done")
        return OperationResult(True, "measured")

    def stop_measurement(self) -> None:
        self.stop_count += 1

    def run_reconstruction(self, progress: ProgressCallback) -> OperationResult:
        self.reconstruct_count += 1
        progress(100, "done")
        return OperationResult(True, "reconstructed")

    def stop_reconstruction(self) -> None:
        self.stop_count += 1

    def focus(self, x: int, y: int) -> OperationResult:
        self.focus_points.append((x, y))
        return OperationResult(
            True,
            "focused",
            {
                "x": x,
                "y": y,
                "pbr": 10.0,
                "focused_image": [[50, 60], [70, 80]],
            },
        )

    def run_pixelwise_report(self, progress: ProgressCallback, frame_callback=None) -> OperationResult:
        if frame_callback is not None:
            frame_callback([[90, 100], [110, 120]])
        progress(100, "done")
        return OperationResult(True, "reported", {"avg_pbr": 8.0})

    def stop_all(self) -> None:
        self.stop_count += 1

    def close(self) -> None:
        self.closed = True
