"""Unified PySide6 control center for TMCalib calibration profiles."""

import os
import sys
from datetime import datetime
from typing import Optional

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer
from PySide6.QtGui import QCloseEvent, QFont, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from tmcalib_gui.launcher import (
    PROFILE_LABELS,
    build_launch_spec,
    display_command,
    format_profile_summary,
    profile_supports_channel,
)


class MainWindow(QMainWindow):
    """Select and supervise one legacy calibration workflow at a time."""

    def __init__(self) -> None:
        super().__init__()
        self._process: Optional[QProcess] = None
        self._closing = False

        self.setWindowTitle("TMCalib 传输矩阵标定平台")
        self.setMinimumSize(820, 620)
        self.resize(980, 720)
        self._build_ui()
        self._apply_style()
        self._profile_changed()

    def _build_ui(self) -> None:
        root = QWidget(self)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(28, 24, 28, 20)
        layout.setSpacing(16)

        title = QLabel("TMCalib 传输矩阵标定平台")
        title.setObjectName("title")
        title.setFont(QFont("Microsoft YaHei UI", 20, QFont.Bold))
        subtitle = QLabel("统一选择实验配置，并监视现有相机与 DMD 标定程序")
        subtitle.setObjectName("subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        config_group = QGroupBox("实验配置")
        form = QFormLayout(config_group)
        form.setContentsMargins(18, 22, 18, 18)
        form.setHorizontalSpacing(20)
        form.setVerticalSpacing(12)

        self.profile_combo = QComboBox()
        for profile_name, label in PROFILE_LABELS.items():
            self.profile_combo.addItem(label, profile_name)
        self.profile_combo.currentIndexChanged.connect(self._profile_changed)
        form.addRow("标定配置", self.profile_combo)

        self.channel_combo = QComboBox()
        self.channel_combo.addItems(("I0", "I90"))
        self.channel_combo.currentIndexChanged.connect(self._refresh_command)
        form.addRow("偏振通道", self.channel_combo)

        self.profile_summary = QLabel()
        self.profile_summary.setObjectName("summary")
        self.profile_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("配置摘要", self.profile_summary)

        self.command_preview = QLabel()
        self.command_preview.setObjectName("commandPreview")
        self.command_preview.setWordWrap(True)
        self.command_preview.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("启动命令", self.command_preview)
        layout.addWidget(config_group)

        safety_group = QGroupBox("启动检查")
        safety_layout = QVBoxLayout(safety_group)
        safety_note = QLabel(
            "启动后程序将直接控制实验硬件。请确认 DMD、相机、触发线、曝光参数和光路功率处于安全状态。"
        )
        safety_note.setObjectName("safetyNote")
        safety_note.setWordWrap(True)
        self.safety_check = QCheckBox("我已完成硬件与光功率检查")
        self.safety_check.toggled.connect(self._update_buttons)
        safety_layout.addWidget(safety_note)
        safety_layout.addWidget(self.safety_check)
        layout.addWidget(safety_group)

        button_row = QHBoxLayout()
        self.start_button = QPushButton("启动标定")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self._start_process)
        self.stop_button = QPushButton("停止当前程序")
        self.stop_button.clicked.connect(self._stop_process)
        self.stop_button.setEnabled(False)
        self.clear_button = QPushButton("清空日志")
        self.clear_button.clicked.connect(self._clear_log)
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.stop_button)
        button_row.addStretch(1)
        button_row.addWidget(self.clear_button)
        layout.addLayout(button_row)

        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout(log_group)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(10000)
        self.log_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.log_view.setPlaceholderText("标定子进程的输出会显示在这里……")
        log_layout.addWidget(self.log_view)
        layout.addWidget(log_group, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(5)
        layout.addWidget(self.progress)

        self.setCentralWidget(root)
        status = QStatusBar(self)
        self.status_label = QLabel("就绪")
        status.addWidget(self.status_label)
        self.setStatusBar(status)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #f7f8fa; color: #1f2937; }
            QLabel#title { color: #111827; }
            QLabel#subtitle { color: #667085; margin-bottom: 4px; }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #d9dee7;
                border-radius: 8px;
                margin-top: 10px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
            QComboBox, QPlainTextEdit {
                background: #ffffff;
                border: 1px solid #cfd5df;
                border-radius: 5px;
                padding: 7px;
            }
            QLabel#summary, QLabel#commandPreview {
                background: #f8fafc;
                border: 1px solid #e5e9f0;
                border-radius: 5px;
                padding: 9px;
                font-weight: 400;
            }
            QLabel#commandPreview { font-family: Consolas; color: #344054; }
            QLabel#safetyNote { color: #9a3412; font-weight: 400; }
            QPushButton {
                background: #ffffff;
                border: 1px solid #cfd5df;
                border-radius: 5px;
                padding: 8px 16px;
                min-width: 105px;
            }
            QPushButton:hover { background: #f1f5f9; }
            QPushButton:disabled { color: #98a2b3; background: #f2f4f7; }
            QPushButton#primaryButton {
                color: white;
                background: #2563eb;
                border-color: #2563eb;
                font-weight: 600;
            }
            QPushButton#primaryButton:hover { background: #1d4ed8; }
            QPushButton#primaryButton:disabled { background: #aebee1; }
            QProgressBar { border: 0; background: #e5e7eb; }
            QProgressBar::chunk { background: #2563eb; }
            """
        )

    def _selected_profile(self) -> str:
        return str(self.profile_combo.currentData())

    def _selected_channel(self) -> Optional[str]:
        if not profile_supports_channel(self._selected_profile()):
            return None
        return self.channel_combo.currentText()

    def _profile_changed(self) -> None:
        profile_name = self._selected_profile()
        channel_enabled = profile_supports_channel(profile_name)
        self.channel_combo.setEnabled(channel_enabled)
        self.profile_summary.setText(format_profile_summary(profile_name))
        self._refresh_command()

    def _refresh_command(self) -> None:
        spec = build_launch_spec(
            self._selected_profile(), channel=self._selected_channel()
        )
        self.command_preview.setText(display_command(spec.command))

    def _update_buttons(self) -> None:
        running = self._process is not None
        self.start_button.setEnabled(self.safety_check.isChecked() and not running)
        self.stop_button.setEnabled(running)
        self.profile_combo.setEnabled(not running)
        self.channel_combo.setEnabled(
            not running and profile_supports_channel(self._selected_profile())
        )

    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_view.appendPlainText("{}  {}".format(timestamp, message.rstrip()))

    def _clear_log(self) -> None:
        if self._process is None:
            self.log_view.clear()
        else:
            self._append_log("程序运行期间保留日志，停止后可清空。")

    def _start_process(self) -> None:
        if self._process is not None or not self.safety_check.isChecked():
            return

        spec = build_launch_spec(
            self._selected_profile(), channel=self._selected_channel()
        )
        if not os.path.isfile(spec.arguments[1]):
            QMessageBox.critical(self, "无法启动", "找不到 run_calibration.py。")
            return

        process = QProcess(self)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.setWorkingDirectory(spec.working_directory)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        process.setProcessEnvironment(environment)
        process.readyReadStandardOutput.connect(self._read_output)
        process.errorOccurred.connect(self._process_error)
        process.finished.connect(self._process_finished)
        self._process = process

        self._append_log("启动：{}".format(display_command(spec.command)))
        self.status_label.setText("正在运行：{}".format(PROFILE_LABELS[spec.profile]))
        self.progress.setRange(0, 0)
        self._update_buttons()
        process.start(spec.program, list(spec.arguments))

    def _read_output(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardOutput())
        text = data.decode("utf-8", errors="replace")
        if text:
            self.log_view.moveCursor(QTextCursor.End)
            self.log_view.insertPlainText(text)
            self.log_view.ensureCursorVisible()

    def _process_error(self, error: QProcess.ProcessError) -> None:
        if self._closing:
            return
        self._append_log("子进程错误：{}".format(error))

    def _process_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        process = self._process
        if process is not None:
            remaining = bytes(process.readAllStandardOutput()).decode(
                "utf-8", errors="replace"
            )
            if remaining:
                self.log_view.insertPlainText(remaining)
        self._append_log(
            "程序结束：退出码 {}，状态 {}".format(exit_code, exit_status)
        )
        self._process = None
        self.status_label.setText("就绪")
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.safety_check.setChecked(False)
        self._update_buttons()

    def _stop_process(self) -> None:
        if self._process is None:
            return
        self.status_label.setText("正在停止……")
        self._append_log("正在请求当前标定程序退出……")
        self._process.terminate()
        QTimer.singleShot(3000, self._kill_if_running)

    def _kill_if_running(self) -> None:
        if self._process is not None and self._process.state() != QProcess.NotRunning:
            self._append_log("程序未在 3 秒内退出，正在强制结束。")
            self._process.kill()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._process is None:
            event.accept()
            return

        answer = QMessageBox.question(
            self,
            "结束标定程序？",
            "当前标定程序仍在运行。关闭控制台将同时停止它，是否继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            event.ignore()
            return

        self._closing = True
        self._process.terminate()
        if not self._process.waitForFinished(2000):
            self._process.kill()
            self._process.waitForFinished(1000)
        event.accept()


def main() -> int:
    """Start the Qt application without importing any vendor hardware SDK."""
    app = QApplication(sys.argv)
    app.setApplicationName("TMCalib")
    app.setOrganizationName("TMCalib")
    window = MainWindow()
    window.show()
    return app.exec()
