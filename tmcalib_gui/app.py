"""Single PySide6 interface bound to an injected calibration workflow."""

import sys
from datetime import datetime
from typing import List, Optional

import numpy as np
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QCloseEvent, QFont, QImage, QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from tmcalib.bootstrap import build_workflow
from tmcalib.events import EventKind, WorkflowEvent
from tmcalib.profiles import PROFILES, get_profile
from tmcalib.workflow import CalibrationWorkflow, WorkflowState
from tmcalib_gui.launcher import format_profile_summary


class QtEventBridge(QObject):
    """Marshal framework-neutral workflow events onto Qt's GUI thread."""

    event_received = Signal(object)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.workflow: Optional[CalibrationWorkflow] = None
        self._unsubscribers: List = []
        self._bridge = QtEventBridge(self)
        self._bridge.event_received.connect(self._handle_event)
        self._last_frame: Optional[np.ndarray] = None

        self.setWindowTitle("TMCalib 传输矩阵标定平台")
        self.setMinimumSize(1100, 720)
        self.resize(1380, 850)
        self._build_ui()
        self._apply_style()
        self._profile_changed()
        self._refresh_for_state(WorkflowState.DISCONNECTED)

    def _build_ui(self) -> None:
        root = QWidget(self)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(22, 18, 22, 14)
        root_layout.setSpacing(12)

        heading = QLabel("TMCalib 传输矩阵标定平台")
        heading.setObjectName("title")
        heading.setFont(QFont("Microsoft YaHei UI", 20, QFont.Bold))
        subtitle = QLabel("同一套界面与流程，Profile 只提供光学配置和算法策略")
        subtitle.setObjectName("subtitle")
        root_layout.addWidget(heading)
        root_layout.addWidget(subtitle)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_camera_panel())
        splitter.addWidget(self._build_control_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root_layout.addWidget(splitter, 1)

        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout(log_group)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(10000)
        log_layout.addWidget(self.log_view)
        root_layout.addWidget(log_group, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        root_layout.addWidget(self.progress)

        self.setCentralWidget(root)
        status = QStatusBar(self)
        self.status_label = QLabel("未连接")
        status.addWidget(self.status_label)
        self.setStatusBar(status)

    def _build_camera_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 8, 0)

        image_group = QGroupBox("相机图像")
        image_layout = QVBoxLayout(image_group)
        self.image_label = QLabel("连接设备并开始测量后显示相机图像")
        self.image_label.setObjectName("cameraView")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(560, 390)
        image_layout.addWidget(self.image_label, 1)
        layout.addWidget(image_group, 1)

        camera_group = QGroupBox("相机控制")
        camera_layout = QHBoxLayout(camera_group)
        camera_layout.addWidget(QLabel("曝光时间"))
        self.exposure_spin = QDoubleSpinBox()
        self.exposure_spin.setRange(1.0, 1000000.0)
        self.exposure_spin.setDecimals(1)
        self.exposure_spin.setSuffix(" μs")
        self.apply_exposure_button = QPushButton("应用曝光")
        self.apply_exposure_button.clicked.connect(self._apply_exposure)
        self.preview_button = QPushButton("启用图像预览")
        self.preview_button.setCheckable(True)
        self.preview_button.toggled.connect(self._toggle_preview)
        camera_layout.addWidget(self.exposure_spin)
        camera_layout.addWidget(self.apply_exposure_button)
        camera_layout.addStretch(1)
        camera_layout.addWidget(self.preview_button)
        layout.addWidget(camera_group)
        return panel

    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 0, 0, 0)

        config_group = QGroupBox("实验配置")
        config_layout = QFormLayout(config_group)
        self.profile_combo = QComboBox()
        for key, profile in PROFILES.items():
            self.profile_combo.addItem(profile.display_name, key)
        self.profile_combo.currentIndexChanged.connect(self._profile_changed)
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(("I0", "I90"))
        self.channel_combo.currentIndexChanged.connect(self._profile_changed)
        self.profile_summary = QLabel()
        self.profile_summary.setObjectName("summary")
        self.profile_summary.setWordWrap(True)
        config_layout.addRow("Profile", self.profile_combo)
        config_layout.addRow("偏振通道", self.channel_combo)
        config_layout.addRow(self.profile_summary)
        layout.addWidget(config_group)

        connection_group = QGroupBox("设备")
        connection_layout = QHBoxLayout(connection_group)
        self.connect_button = QPushButton("连接相机与 DMD")
        self.connect_button.setObjectName("primaryButton")
        self.connect_button.clicked.connect(self._connect_hardware)
        self.stop_button = QPushButton("停止当前任务")
        self.stop_button.clicked.connect(self._stop)
        connection_layout.addWidget(self.connect_button)
        connection_layout.addWidget(self.stop_button)
        layout.addWidget(connection_group)

        workflow_group = QGroupBox("统一标定流程")
        workflow_layout = QGridLayout(workflow_group)
        self.measure_button = QPushButton("1. 开始测量")
        self.measure_button.clicked.connect(lambda: self._run("measure"))
        self.reconstruct_button = QPushButton("2. 恢复传输矩阵")
        self.reconstruct_button.clicked.connect(lambda: self._run("reconstruct"))
        self.one_click_button = QPushButton("一键：测量 → 恢复 → 逐点报告")
        self.one_click_button.clicked.connect(lambda: self._run("one_click"))
        workflow_layout.addWidget(self.measure_button, 0, 0)
        workflow_layout.addWidget(self.reconstruct_button, 0, 1)
        workflow_layout.addWidget(self.one_click_button, 1, 0, 1, 2)
        layout.addWidget(workflow_group)

        focus_group = QGroupBox("共轭聚焦")
        focus_layout = QGridLayout(focus_group)
        self.focus_x = QSpinBox()
        self.focus_y = QSpinBox()
        self.focus_button = QPushButton("聚焦到目标位置")
        self.focus_button.clicked.connect(self._focus)
        focus_layout.addWidget(QLabel("X"), 0, 0)
        focus_layout.addWidget(self.focus_x, 0, 1)
        focus_layout.addWidget(QLabel("Y"), 0, 2)
        focus_layout.addWidget(self.focus_y, 0, 3)
        focus_layout.addWidget(self.focus_button, 1, 0, 1, 4)
        layout.addWidget(focus_group)

        status_group = QGroupBox("系统状态")
        status_layout = QFormLayout(status_group)
        self.profile_status = QLabel("-")
        self.device_status = QLabel("未连接")
        self.operation_status = QLabel("空闲")
        self.result_status = QLabel("-")
        self.result_status.setWordWrap(True)
        status_layout.addRow("当前配置", self.profile_status)
        status_layout.addRow("设备", self.device_status)
        status_layout.addRow("任务", self.operation_status)
        status_layout.addRow("结果", self.result_status)
        layout.addWidget(status_group)
        layout.addStretch(1)
        return panel

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #f7f8fa; color: #1f2937; }
            QLabel#title { color: #111827; }
            QLabel#subtitle { color: #667085; margin-bottom: 4px; }
            QGroupBox {
                background: #ffffff; border: 1px solid #d9dee7;
                border-radius: 7px; margin-top: 10px; font-weight: 600;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 11px; padding: 0 6px; }
            QComboBox, QDoubleSpinBox, QSpinBox, QPlainTextEdit {
                background: #ffffff; border: 1px solid #cfd5df;
                border-radius: 5px; padding: 6px;
            }
            QLabel#summary {
                background: #f8fafc; border: 1px solid #e5e9f0;
                border-radius: 5px; padding: 9px; font-weight: 400;
            }
            QLabel#cameraView {
                color: #667085; background: #101318;
                border: 1px solid #2f3540; border-radius: 4px;
            }
            QPushButton {
                background: #ffffff; border: 1px solid #cfd5df;
                border-radius: 5px; padding: 8px 13px;
            }
            QPushButton:hover { background: #f1f5f9; }
            QPushButton:disabled { color: #98a2b3; background: #f2f4f7; }
            QPushButton#primaryButton { color: white; background: #2563eb; border-color: #2563eb; }
            QPushButton#primaryButton:disabled { background: #aebee1; }
            """
        )

    def _selected_profile_key(self) -> str:
        return str(self.profile_combo.currentData())

    def _selected_channel(self) -> Optional[str]:
        profile = get_profile(self._selected_profile_key())
        return self.channel_combo.currentText() if profile.available_channels else None

    def _profile_changed(self) -> None:
        profile = get_profile(self._selected_profile_key(), self._selected_channel())
        self.channel_combo.setEnabled(
            bool(profile.available_channels) and self.workflow is None
        )
        self.profile_summary.setText(format_profile_summary(profile.key))
        self.profile_status.setText(profile.display_name)
        self.exposure_spin.setValue(profile.default_exposure_us)
        roi_h, roi_w = profile.camera_roi
        self.focus_x.setRange(0, roi_w - 1)
        self.focus_y.setRange(0, roi_h - 1)
        self.focus_x.setValue(roi_w // 2)
        self.focus_y.setValue(roi_h // 2)
        self._apply_capabilities(profile)

    def _apply_capabilities(self, profile) -> None:
        connected = self.workflow is not None and self.workflow.state == WorkflowState.IDLE
        self.apply_exposure_button.setEnabled(connected and profile.supports("exposure"))
        self.preview_button.setEnabled(connected and profile.supports("preview"))
        self.measure_button.setEnabled(connected and profile.supports("measurement"))
        self.reconstruct_button.setEnabled(connected and profile.supports("reconstruction"))
        self.focus_button.setEnabled(connected and profile.supports("focus"))
        self.one_click_button.setEnabled(connected and profile.supports("one_click"))

    def _subscribe_workflow(self, workflow: CalibrationWorkflow) -> None:
        self._unsubscribers = [
            workflow.events.subscribe(kind, self._bridge.event_received.emit)
            for kind in EventKind
        ]

    def _disconnect_workflow(self) -> None:
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers = []
        if self.workflow is not None:
            self.workflow.close()
        self.workflow = None

    def _connect_hardware(self) -> None:
        if self.workflow is not None:
            self._disconnect_workflow()
        try:
            workflow = build_workflow(
                self._selected_profile_key(), self._selected_channel()
            )
            self.workflow = workflow
            self._subscribe_workflow(workflow)
            workflow.connect()
            self.profile_combo.setEnabled(False)
            self.channel_combo.setEnabled(False)
        except Exception as exc:
            self._show_error(str(exc))

    def _run(self, operation: str) -> None:
        if self.workflow is None:
            return
        try:
            getattr(self.workflow, operation)()
        except Exception as exc:
            self._show_error(str(exc))

    def _focus(self) -> None:
        if self.workflow is None:
            return
        try:
            self.workflow.focus(self.focus_x.value(), self.focus_y.value())
        except Exception as exc:
            self._show_error(str(exc))

    def _apply_exposure(self) -> None:
        if self.workflow is None:
            return
        try:
            actual = self.workflow.set_exposure_us(self.exposure_spin.value())
            self.exposure_spin.setValue(actual)
        except Exception as exc:
            self._show_error(str(exc))

    def _toggle_preview(self, enabled: bool) -> None:
        if self.workflow is None:
            self.preview_button.setChecked(False)
            return
        try:
            if enabled:
                self.workflow.start_preview()
                self.preview_button.setText("停止图像预览")
            else:
                self.workflow.stop_preview()
                self.preview_button.setText("启用图像预览")
        except Exception as exc:
            self.preview_button.setChecked(False)
            self._show_error(str(exc))

    def _stop(self) -> None:
        if self.workflow is not None:
            self.workflow.stop()

    def _handle_event(self, event: WorkflowEvent) -> None:
        if event.kind == EventKind.STATE:
            self._refresh_for_state(event.payload)
        elif event.kind == EventKind.LOG:
            self._append_log(event.message)
        elif event.kind == EventKind.PROGRESS:
            self.progress.setValue(int(event.progress or 0))
            self.operation_status.setText(event.message or event.operation)
        elif event.kind == EventKind.FRAME:
            self._show_frame(event.payload)
        elif event.kind == EventKind.RESULT:
            self.result_status.setText(event.message or "完成")
        elif event.kind == EventKind.ERROR:
            self._append_log("错误：{}".format(event.message))
            self._show_error(event.message)

    def _refresh_for_state(self, state: WorkflowState) -> None:
        busy = state not in (
            WorkflowState.DISCONNECTED,
            WorkflowState.IDLE,
            WorkflowState.ERROR,
            WorkflowState.CLOSED,
        )
        self.status_label.setText(state.value)
        self.operation_status.setText(state.value)
        self.device_status.setText("已连接" if state == WorkflowState.IDLE else state.value)
        self.connect_button.setEnabled(not busy)
        self.stop_button.setEnabled(busy)
        self.profile_combo.setEnabled(self.workflow is None)
        profile = get_profile(self._selected_profile_key(), self._selected_channel())
        self._apply_capabilities(profile)
        if state == WorkflowState.IDLE:
            self.progress.setValue(0)

    def _append_log(self, message: str) -> None:
        self.log_view.moveCursor(QTextCursor.End)
        self.log_view.appendPlainText(
            "{}  {}".format(datetime.now().strftime("%H:%M:%S"), message)
        )

    def _show_frame(self, frame) -> None:
        array = np.asarray(frame)
        if array.ndim != 2:
            return
        if array.dtype != np.uint8:
            minimum = float(np.min(array))
            maximum = float(np.max(array))
            if maximum <= minimum:
                image = np.zeros(array.shape, dtype=np.uint8)
            else:
                image = ((array - minimum) * (255.0 / (maximum - minimum))).astype(np.uint8)
        else:
            image = np.ascontiguousarray(array)
        self._last_frame = image
        height, width = image.shape
        qimage = QImage(
            image.data, width, height, image.strides[0], QImage.Format_Grayscale8
        ).copy()
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "TMCalib", message)

    def closeEvent(self, event: QCloseEvent) -> None:
        try:
            self._disconnect_workflow()
        finally:
            event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("TMCalib")
    window = MainWindow()
    window.show()
    return app.exec()
