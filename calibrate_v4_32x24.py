#Transport martix calibration

import torch
import sys
import os
# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import PySpin
import threading
import queue
import time
import numpy as np

# import torch
from tqdm import tqdm
from scipy.stats import pearsonr
from ctypes import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from datetime import datetime
import matplotlib.dates as mdates
from PIL import Image
import cv2
import warnings
from holograms.generate_LUT import generate_lut
from holograms.dmd_holograms import holo_SP
import atexit
import signal

import re
import math

from pixelwise_focus_report_128 import (
    build_pixelwise_points,
    save_pixelwise_focus_report,
)
from measurement_quality_report import (
    analyze_measurement_memmap,
    build_measurement_quality_figure,
    save_measurement_quality_outputs,
)
from tmcalib.focus_modes import prepare_conjugate_focus_field

try:
    import paramiko
except ImportError:
    paramiko = None
warnings.filterwarnings('ignore')

# Acquisition timing. Keep the camera exposure shorter than the DMD picture
# period so every hardware trigger can start a new exposure.
TARGET_ACQUISITION_FPS = 500.0
DMD_PICTURE_TIME_US = 1_000_000.0 / TARGET_ACQUISITION_FPS
CAMERA_EXPOSURE_US = 1500.0

# Pre-generated 32 x 24 pattern dataset selection. Change only ``active`` to
# switch datasets; the probe count and reconstruction input are kept in sync.
PATTERN_CONFIG = {
    "active": "20N_4PHASE",
    "sets": {
        "4N": {
            "directory": "pregenerated_patterns",
            "probe_multiplier": 4,
        },
        "8N": {
            "directory": "pregenerated_patterns_8N",
            "probe_multiplier": 8,
        },
        "12N": {
            "directory": "pregenerated_patterns_v4_phase_only_12N",
            "probe_multiplier": 12,
        },
        "12N_4PHASE": {
            "directory": "pregenerated_patterns_v4_phase_only_4level_12N",
            "probe_multiplier": 12,
            "output_tag": "32x24_12N_4phase",
        },
        "12N_16PHASE": {
            "directory": "pregenerated_patterns_v4_phase_only_16level_12N",
            "probe_multiplier": 12,
            "output_tag": "32x24_12N_16phase",
        },
        "12N_32PHASE": {
            "directory": "pregenerated_patterns_v4_phase_only_32level_12N",
            "probe_multiplier": 12,
            "output_tag": "32x24_12N_32phase",
        },
        "20N_4PHASE": {
            "directory": "pregenerated_patterns_v4_phase_only_4level_20N",
            "probe_multiplier": 20,
            "output_tag": "32x24_20N_4phase",
        },
    },
}

# Focus input selection. ``complex`` is the previous behavior,
# ``phase_only`` retains the reconstructed phase at unit amplitude;
# ``phase_only_2`` and ``phase_only_4`` quantize it to 2 or 4 phase levels;
# ``amplitude_only_binary`` selects an on/off mask with a common zero phase.
FOCUS_CONFIG = {
    "active": "phase_only",
    "modes": {
        "complex": {},
        "phase_only": {},
        "phase_only_2": {"phase_levels": 2},
        "phase_only_4": {"phase_levels": 4},
        "amplitude_only_binary": {},
    },
}


def get_active_pattern_config():
    """Return and validate the configured pre-generated pattern dataset."""
    active = PATTERN_CONFIG.get("active")
    datasets = PATTERN_CONFIG.get("sets", {})
    if active not in datasets:
        available = ", ".join(sorted(datasets)) or "<none>"
        raise ValueError(
            f"Unknown pattern dataset {active!r}; available datasets: {available}"
        )

    config = dict(datasets[active])
    config["name"] = active
    config["directory"] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        config["directory"],
    )
    config["probe_multiplier"] = int(config["probe_multiplier"])
    output_tag = config.get("output_tag")
    config["measurement_filename"] = (
        f"measurements_{output_tag}_memmap.npy"
        if output_tag else "measurements_memmap.npy"
    )
    config["tm_memmap_filename"] = (
        f"transmission_matrix_{output_tag}_memmap.npy"
        if output_tag else "transmission_matrix_memmap.npy"
    )
    config["reconstructed_filename"] = (
        f"reconstructed_field_{output_tag}.npy"
        if output_tag else "reconstructed_field.npy"
    )
    config["error_curve_filename"] = (
        f"ggs21_error_curve_{output_tag}.npy"
        if output_tag else "ggs21_error_curve.npy"
    )
    return config


def get_active_focus_config():
    """Return and validate the selected TM focus input policy."""
    active = FOCUS_CONFIG.get("active")
    modes = FOCUS_CONFIG.get("modes", {})
    if active not in modes:
        available = ", ".join(sorted(modes)) or "<none>"
        raise ValueError(
            f"Unknown focus mode {active!r}; available modes: {available}"
        )
    config = dict(modes[active])
    config["name"] = active
    return config


def build_conjugate_focus_field(tm_values):
    """Apply the configured focus policy before DMD LUT encoding."""
    config = get_active_focus_config()
    return prepare_conjugate_focus_field(
        tm_values,
        mode=config["name"],
        phase_levels=config.get("phase_levels"),
    )

class CameraHandler:
    def __init__(self, cam_index, save_path):
        self.cam_index = cam_index
        self.save_path = save_path
        self.is_running = True
        self.image_queue = queue.Queue(maxsize=1)
        self.measurement_frame_callback = None
        self.system = None
        self.cam = None
        self.pattern_lock = threading.Lock()
        self.current_pattern_index = 0
        # Polarized8 already contains the camera's full sensor precision.
        # Keep its native 0..255 codes; the measurement memmap remains uint16
        # below only for backward compatibility with the reconstruction reader.
        self.convert_to_12bit = False
        self.trigger_enabled = False
        self.last_error_time = 0
        self.trigger_ready = threading.Event()
        self.last_frame_timestamp_ns = None
        self.last_frame_id = None

        # Final ROI after extracting one polarization quadrant.  A 2x larger
        # raw sensor ROI is required because each polarization direction uses
        # one pixel in every 2x2 micro-polarizer cell.
        self.roi_x = 296
        self.roi_y = 206
        self.roi_width = 128
        self.roi_height = 128
        # Sony PolarSens layout: the top-left micro-polarizer is the 90-degree
        # channel. Use the SDK enum so the intent is explicit.
        self.polarization_quadrant = PySpin.SPINNAKER_POLARIZATION_QUADRANT_I90

        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()

        if self.cam_list.GetSize() <= self.cam_index:
            self.cam_list.Clear()
            self.system.ReleaseInstance()
            raise RuntimeError(f"Camera {self.cam_index} not found")

        self.cam = self.cam_list.GetByIndex(self.cam_index)
        self.cam.Init()

        # Polarization format must be selected before configuring the raw ROI.
        # ISP processing is disabled so the polarized mosaic is not treated as
        # a conventional monochrome image.
        self.configure_ISP(isp=False)
        if not self.configure_format():
            self.cleanup()
            raise RuntimeError("无法将相机设置为 Polarized8")
        if not self.configure_roi():
            self.cleanup()
            raise RuntimeError("无法配置偏振相机 ROI")

        # Keep acquisition parameters consistent with tm_calib.py
        # 1.5 ms exposure leaves 0.5 ms of margin in the 2 ms (500 Hz)
        # DMD picture period configured below.
        self.configure_exposure(exposure_time=CAMERA_EXPOSURE_US)
        self.configure_gamma(gamma=1)
        self.configure_gain(gain=0)
        self.configure_buffer_handling()
        self.configure_trigger()

    def publish_latest_image(self, frame_index, image):
        """Publish one owned batch image without delaying acquisition."""
        payload = (int(frame_index), np.array(image, copy=True))

        callback = self.measurement_frame_callback
        if callback is not None:
            try:
                callback(payload[1])
            except Exception as exc:
                print("Measurement frame display callback failed: {}".format(exc))

        try:
            self.image_queue.put_nowait(payload)
            return
        except queue.Full:
            pass

        try:
            self.image_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self.image_queue.put_nowait(payload)
        except queue.Full:
            pass

    def configure_roi(self):
        """Configure a centered 256x256 raw ROI for a 128x128 polar image."""
        if not self.cam:
            return False

        try:
            if self.cam.IsStreaming():
                self.cam.EndAcquisition()

            nodemap = self.cam.GetNodeMap()

            node_offset_x = PySpin.CIntegerPtr(nodemap.GetNode('OffsetX'))
            node_offset_y = PySpin.CIntegerPtr(nodemap.GetNode('OffsetY'))
            node_width = PySpin.CIntegerPtr(nodemap.GetNode('Width'))
            node_height = PySpin.CIntegerPtr(nodemap.GetNode('Height'))

            if not all([PySpin.IsAvailable(node) and PySpin.IsWritable(node)
                       for node in [node_offset_x, node_offset_y, node_width, node_height]]):
                print("部分ROI参数不可用，将使用全分辨率")
                return False

            # Reset offsets before querying the maximum sensor dimensions.
            node_offset_x.SetValue(node_offset_x.GetMin())
            node_offset_y.SetValue(node_offset_y.GetMin())
            sensor_width = node_width.GetMax()
            sensor_height = node_height.GetMax()

            raw_width = self.roi_width * 2
            raw_height = self.roi_height * 2
            if raw_width > sensor_width or raw_height > sensor_height:
                print(
                    f"偏振ROI过大: raw={raw_width}x{raw_height}, "
                    f"sensor={sensor_width}x{sensor_height}"
                )
                return False

            node_width.SetValue(raw_width)
            node_height.SetValue(raw_height)

            def align_to_node(value, node):
                minimum = node.GetMin()
                increment = max(1, node.GetInc())
                return minimum + ((value - minimum) // increment) * increment

            raw_x = align_to_node((sensor_width - raw_width) // 2, node_offset_x)
            raw_y = align_to_node((sensor_height - raw_height) // 2, node_offset_y)

            # A polarization ROI must start on an even sensor coordinate or
            # the 2x2 quadrant mapping changes.
            if raw_x % 2:
                raw_x = align_to_node(raw_x - 1, node_offset_x)
            if raw_y % 2:
                raw_y = align_to_node(raw_y - 1, node_offset_y)

            node_offset_x.SetValue(raw_x)
            node_offset_y.SetValue(raw_y)

            # Coordinates in the extracted half-resolution quadrant.
            self.roi_x = raw_x // 2
            self.roi_y = raw_y // 2

            print(
                f"偏振ROI配置成功: raw X={raw_x}, Y={raw_y}, "
                f"Width={raw_width}, Height={raw_height}; "
                f"I90 output={self.roi_width}x{self.roi_height}"
            )
            return True

        except PySpin.SpinnakerException as ex:
            print(f'配置ROI错误: {ex}')
            return False

    def configure_trigger(self):
        """配置硬件触发设置"""
        if not self.cam:
            return False

        try:
            nodemap = self.cam.GetNodeMap()

            node_trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
            if not PySpin.IsReadable(node_trigger_mode) or not PySpin.IsWritable(node_trigger_mode):
                print('无法禁用触发模式')
                return False

            node_trigger_mode_off = node_trigger_mode.GetEntryByName('Off')
            if not PySpin.IsReadable(node_trigger_mode_off):
                print('无法禁用触发模式')
                return False

            node_trigger_mode.SetIntValue(node_trigger_mode_off.GetValue())
            print('触发模式已禁用')

            node_trigger_selector = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSelector'))
            if not PySpin.IsReadable(node_trigger_selector) or not PySpin.IsWritable(node_trigger_selector):
                print('无法获取触发选择器')
                return False

            node_trigger_selector_framestart = node_trigger_selector.GetEntryByName('FrameStart')
            if not PySpin.IsReadable(node_trigger_selector_framestart):
                print('无法设置触发选择器')
                return False

            node_trigger_selector.SetIntValue(node_trigger_selector_framestart.GetValue())
            print('触发选择器设置为帧开始')

            node_trigger_source = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSource'))
            if not PySpin.IsReadable(node_trigger_source) or not PySpin.IsWritable(node_trigger_source):
                print('无法获取触发源')
                return False

            node_trigger_source_hardware = node_trigger_source.GetEntryByName('Line0')
            if not PySpin.IsReadable(node_trigger_source_hardware):
                print('无法获取硬件触发源')
                return False

            node_trigger_source.SetIntValue(node_trigger_source_hardware.GetValue())
            print('触发源设置为硬件(Line0)')

            # Allow the next frame trigger as soon as the current exposure has
            # finished, even if the previous frame is still being read out.
            # Without this, a short DMD trigger period can effectively double
            # because triggers arriving during camera readout are rejected.
            node_trigger_overlap = PySpin.CEnumerationPtr(
                nodemap.GetNode('TriggerOverlap')
            )
            if (
                PySpin.IsReadable(node_trigger_overlap)
                and PySpin.IsWritable(node_trigger_overlap)
            ):
                node_trigger_overlap_readout = (
                    node_trigger_overlap.GetEntryByName('ReadOut')
                )
                if PySpin.IsReadable(node_trigger_overlap_readout):
                    node_trigger_overlap.SetIntValue(
                        node_trigger_overlap_readout.GetValue()
                    )
                    print('触发重叠模式设置为ReadOut')
                else:
                    print('警告: 相机不支持TriggerOverlap=ReadOut')
            else:
                print('警告: 相机TriggerOverlap节点不可配置')

            node_trigger_mode_on = node_trigger_mode.GetEntryByName('On')
            if not PySpin.IsReadable(node_trigger_mode_on):
                print('无法启用触发模式')
                return False

            node_trigger_mode.SetIntValue(node_trigger_mode_on.GetValue())
            print('触发模式已启用')

            node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
            node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
            node_acquisition_mode.SetIntValue(node_acquisition_mode_continuous.GetValue())

            return True

        except PySpin.SpinnakerException as ex:
            print(f'配置触发错误: {ex}')
            return False

    def configure_exposure(self, exposure_time=4000.0):
        """配置相机曝光设置"""
        if not self.cam:
            return False

        try:
            nodemap = self.cam.GetNodeMap()

            node_exposure_auto = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureAuto'))
            node_exposure_auto_off = node_exposure_auto.GetEntryByName('Off')
            node_exposure_auto.SetIntValue(node_exposure_auto_off.GetValue())

            node_exposure_mode = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureMode'))
            node_exposure_mode_timed = node_exposure_mode.GetEntryByName('Timed')
            node_exposure_mode.SetIntValue(node_exposure_mode_timed.GetValue())

            node_exposure_time = PySpin.CFloatPtr(nodemap.GetNode('ExposureTime'))
            node_exposure_time.SetValue(exposure_time)
            actual_exposure_time = float(node_exposure_time.GetValue())

            print(
                f'曝光时间设置为{actual_exposure_time:.1f}μs '
                f'(requested {exposure_time:.1f}μs)'
            )
            return True

        except PySpin.SpinnakerException as ex:
            print(f'配置曝光错误: {ex}')
            return False

    def configure_gain(self, auto_gain='Off',gain=0):
        """配置相机gain设置"""
        if not self.cam:
            return False

        try:
            nodemap = self.cam.GetNodeMap()

            node_gain_auto = PySpin.CEnumerationPtr(nodemap.GetNode('GainAuto'))
            node_gain_auto_off = node_gain_auto.GetEntryByName(auto_gain)
            node_gain_auto.SetIntValue(node_gain_auto_off.GetValue())
            print(f'gainAuto设置为{auto_gain}')

            node_gain_value = PySpin.CFloatPtr(nodemap.GetNode('Gain'))
            node_gain_value.SetValue(gain)

            print(f'gain时间设置为{gain}dB')
            return True

        except PySpin.SpinnakerException as ex:
            print(f'配置gain错误: {ex}')
            return False

    def configure_gamma(self, gamma_enable=True,gamma=1.0):
        """配置相机gamma设置"""
        if not self.cam:
            return False

        try:
            nodemap = self.cam.GetNodeMap()

            node_gamma_enale = PySpin.CBooleanPtr(nodemap.GetNode('GammaEnable'))
            node_gamma_enale.SetValue(gamma_enable)

            print(f'gammaEnable设置为{gamma_enable}')

            node_gamma_value = PySpin.CFloatPtr(nodemap.GetNode('Gamma'))
            node_gamma_value.SetValue(gamma)

            print(f'gamma设置为{gamma}')
            return True

        except PySpin.SpinnakerException as ex:
            print(f'配置gamma错误: {ex}')
            return False

    def configure_ISP(self,isp=True):
        """配置相机ISP设置"""
        if not self.cam:
            return False

        try:
            nodemap = self.cam.GetNodeMap()

            node_isp_enable = PySpin.CBooleanPtr(nodemap.GetNode('IspEnable'))
            node_isp_enable.SetValue(isp)

            print(f'Isp设置为{isp}')
            return True

        except PySpin.SpinnakerException as ex:
            print(f'配置Isp错误: {ex}')
            return False

    def configure_format(self):
        """Configure the camera to return a polarization-aware raw image."""
        if not self.cam:
            return False

        try:
            self.cam.PixelFormat.SetValue(PySpin.PixelFormat_Polarized8)
            print('Pixel Format设置为PixelFormat_Polarized8')
            return True

        except PySpin.SpinnakerException as ex:
            print(f'配置格式错误: {ex}')
            return False

    def configure_buffer_handling(self):
        """配置缓冲区处理模式为OldestFirst (避免丢失帧)"""
        if not self.cam:
            return False

        try:
            nodemap = self.cam.GetTLStreamNodeMap()
            node_bufferhandling = PySpin.CEnumerationPtr(nodemap.GetNode('StreamBufferHandlingMode'))

            # 使用 OldestFirst 确保不丢帧，这对序列采集很重要
            node_oldestfirst = node_bufferhandling.GetEntryByName('OldestFirst')
            node_bufferhandling.SetIntValue(node_oldestfirst.GetValue())

            print('缓冲区处理模式设置为OldestFirst')
            return True

        except PySpin.SpinnakerException as ex:
            print(f'配置缓冲区处理错误: {ex}')
            return False

    def reset_trigger(self):
        """重置触发模式为关闭"""
        if not self.cam:
            return False

        try:
            nodemap = self.cam.GetNodeMap()
            node_trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
            if not PySpin.IsReadable(node_trigger_mode) or not PySpin.IsWritable(node_trigger_mode):
                print('无法禁用触发模式')
                return False

            node_trigger_mode_off = node_trigger_mode.GetEntryByName('Off')
            if not PySpin.IsReadable(node_trigger_mode_off):
                print('无法禁用触发模式')
                return False

            node_trigger_mode.SetIntValue(node_trigger_mode_off.GetValue())
            print('触发模式已禁用')
            return True

        except PySpin.SpinnakerException as ex:
            print(f'重置触发错误: {ex}')
            return False
    def start(self):
        self.cam.BeginAcquisition()

    def run(self):
        t_start = time.time()
        wait_time = 0
        process_time = 0
        image_result = None
        self.last_frame_timestamp_ns = None
        self.last_frame_id = None

        try:
            # Increase timeout to 2000ms to accommodate hardware trigger delays
            # Time the wait for trigger
            t_before_wait = time.time()
            image_result = self.cam.GetNextImage(2000)
            wait_time = time.time() - t_before_wait

            if image_result.IsIncomplete():
                print(f"图像不完整，状态: {image_result.GetImageStatus()}")
                return None, 0, 0

            # The Spinnaker image timestamp is generated by the camera and is
            # expressed in nanoseconds. Keep it for batch-level rate and gap
            # diagnostics without changing the existing run() return value.
            try:
                self.last_frame_timestamp_ns = int(image_result.GetTimeStamp())
                self.last_frame_id = int(image_result.GetFrameID())
            except Exception:
                self.last_frame_timestamp_ns = None
                self.last_frame_id = None

            t_before_proc = time.time()

            # Extract the top-left (90-degree) micro-polarizer channel.  The
            # 256x256 raw Polarized8 ROI becomes one 128x128 Mono8 image.
            polar_image = PySpin.ImageUtilityPolarization.ExtractPolarQuadrant(
                image_result,
                self.polarization_quadrant,
            )
            image_data = np.array(polar_image.GetNDArray(), copy=True)
            if image_data.shape != (self.roi_height, self.roi_width):
                raise RuntimeError(
                    f"偏振图像shape错误: {image_data.shape}, "
                    f"expected {(self.roi_height, self.roi_width)}"
                )

            if self.convert_to_12bit:
                # Preserve the existing uint16 measurement pipeline. This is
                # an 8-to-12-bit range scaling; it does not create extra sensor
                # precision.
                image_converted = (
                    image_data.astype(np.float32) / 255.0 * 4095.0
                ).astype(np.uint16)
            else:
                image_converted = image_data
            process_time = time.time() - t_before_proc

        except Exception as ex:
            current_time = time.time()
            if current_time - self.last_error_time > 5:
                print(f'获取图像错误: {ex}')
                self.last_error_time = current_time
            return None, 0, 0
        finally:
            if image_result is not None:
                try:
                    image_result.Release()
                except Exception:
                    pass


        return image_converted, wait_time, process_time

            # self.system.ReleaseInstance()

    def stop(self):
        try:
            if self.cam.IsStreaming():
                self.cam.EndAcquisition()
        except Exception:
            pass

    def cleanup(self):
        """Release camera and system resources"""
        try:
            if self.cam:
                try:
                    if self.cam.IsStreaming():
                        self.cam.EndAcquisition()
                except:
                    pass

                self.cam.DeInit()
                del self.cam
                self.cam = None

            if self.cam_list:
                self.cam_list.Clear()
                self.cam_list = None

            if self.system:
                self.system.ReleaseInstance()
                self.system = None

            print("Camera resources released")
        except Exception as e:
            print(f"Error releasing camera resources: {e}")


class DMDController:
    def __init__(self, camera_handler=None):
        # Update path to V4 DLL
        dll_rel_path = r"JUOPT_DLP V4.0.002 20250522 release\4.DLL\DLL\JUOPT_DLL_V4.dll"
        dll_path = os.path.abspath(os.path.join(os.getcwd(), dll_rel_path))

        # Add DLL directory to path for dependencies
        dll_dir = os.path.dirname(dll_path)
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(dll_dir)
        else:
            os.environ['PATH'] = dll_dir + ';' + os.environ['PATH']

        try:
            self.DMD = cdll.LoadLibrary(dll_path)
            self._setup_v4_functions()
            self.DMD.juoptInit() # Global initialization
        except Exception as e:
            print(f"Failed to load DMD DLL: {e}")
            self.DMD = None

        self.is_init = False
        self.dev_id = None
        self.original_width = 1024
        self.original_height = 768
        self.pixel_group_size = 32
        self.dmd_width = self.original_width // self.pixel_group_size
        self.dmd_height = self.original_height // self.pixel_group_size
        self.camera = camera_handler
        self.current_pattern = None
        self.pbr_history = []
        self.optimization_running = False
        self.measurement_error = None
        self.reconstruction_running = False
        self.reconstruction_error = None
        # Keep recovery file names configurable so algorithm variants can use
        # the same acquisition/focusing workflow without overwriting results.
        pattern_config = get_active_pattern_config()
        self.measurement_filename = pattern_config["measurement_filename"]
        self.tm_memmap_filename = pattern_config["tm_memmap_filename"]
        self.reconstructed_filename = pattern_config["reconstructed_filename"]
        self.error_curve_filename = pattern_config["error_curve_filename"]
        self.current_pbr = 0
        self.current_peak_intensity = 0
        self.current_stability_corr = None  # Pearson correlation vs baseline white-speckle
        self.time_serie = []
        self.measure_progress_callback = None  # Callback for measurement progress
        self.recon_progress_callback = None  # Callback for reconstruction progress

        # Stability monitoring (baseline white-speckle captured on first batch)
        self._stability_baseline_white_speckle = None
        self._stability_seq = 0
        self.ggs21_iters = 200
        self.ggs21_ratio = 0.89
        self.ggs21_n_worker = 8
        self.ggs21_use_gpu = True
        # Use the same initial detector phases when comparing linear solvers.
        self.ggs21_random_seed = 24032
        # Pixel-wise hologram encoding uses the first CUDA device when
        # available and falls back to the bit-exact NumPy implementation.
        self.focus_encoding_use_gpu = True
        self.focus_encoding_gpu_device = 0
        self.focus_encoding_gpu_chunk_size = 32
        self.last_focus_encoding_backend = 'not started'
        self.last_focus_encoding_fallback_error = None
        self._focus_gpu_tensor_cache = {}
        self._focus_gpu_failed = False

        # 原始相机数据是强度时保持 True
        self.measurements_are_intensity = True

        # 如果有暗场平均值，可以填入
        self.ggs21_dark_level = 0.0

    def _setup_v4_functions(self):
        if not self.DMD: return

        # Define types
        JUOPTDMD_ID = c_long

        # juoptGetOnlineDevNum(int* devNum)
        self.DMD.juoptGetOnlineDevNum.argtypes = [POINTER(c_int)]
        self.DMD.juoptGetOnlineDevNum.restype = c_long

        # juoptGetOnlineDevList() -> char**
        self.DMD.juoptGetOnlineDevList.argtypes = []
        self.DMD.juoptGetOnlineDevList.restype = POINTER(c_char_p)

        # juoptAllocDev(JUOPTDMD_ID *devUserID, char * devName)
        self.DMD.juoptAllocDev.argtypes = [POINTER(JUOPTDMD_ID), c_char_p]
        self.DMD.juoptAllocDev.restype = c_long

        # juoptSplitSquenceNum(JUOPTDMD_ID devUserID, int num)
        self.DMD.juoptSplitSquenceNum.argtypes = [JUOPTDMD_ID, c_int]
        self.DMD.juoptSplitSquenceNum.restype = c_int

        # juoptInitSquenceSize(JUOPTDMD_ID devUserID, unsigned int squenceID, int pictureNum, int width, int height, int bitPlane)
        self.DMD.juoptInitSquenceSize.argtypes = [JUOPTDMD_ID, c_uint, c_int, c_int, c_int, c_int]
        self.DMD.juoptInitSquenceSize.restype = c_int

        # juoptClearOneSquenceData(JUOPTDMD_ID devUserID, unsigned int squenceID)
        self.DMD.juoptClearOneSquenceData.argtypes = [JUOPTDMD_ID, c_uint]
        self.DMD.juoptClearOneSquenceData.restype = c_int

        # juoptLoadPixmap(JUOPTDMD_ID devUserID, unsigned int squenceID, long x, long bitPlane, long pictureNum, unsigned char*pictureData, int significantBit, int pictureWidth, int pictureHeight)
        self.DMD.juoptLoadPixmap.argtypes = [JUOPTDMD_ID, c_uint, c_long, c_long, c_long, c_char_p, c_int, c_int, c_int]
        self.DMD.juoptLoadPixmap.restype = c_long

        # juoptProjection(JUOPTDMD_ID devUserID, unsigned int squenceID, int y)
        self.DMD.juoptProjection.argtypes = [JUOPTDMD_ID, c_uint, c_int]
        self.DMD.juoptProjection.restype = c_long

        # juoptTimeControl(JUOPTDMD_ID devUserID,unsigned int squenceID, double picTime,double illuminateTime,double synchDelay,double synchPulseWidth,double triggerInDelay)
        self.DMD.juoptTimeControl.argtypes = [JUOPTDMD_ID, c_uint, c_double, c_double, c_double, c_double, c_double]
        self.DMD.juoptTimeControl.restype = c_long

        # juoptFree(JUOPTDMD_ID devUserID)
        self.DMD.juoptFree.argtypes = [JUOPTDMD_ID]
        self.DMD.juoptFree.restype = c_long

        # juoptStop(JUOPTDMD_ID devUserID)
        self.DMD.juoptStop.argtypes = [JUOPTDMD_ID]
        self.DMD.juoptStop.restype = c_long

        # juoptSetDevPara(JUOPTDMD_ID devUserID, long paraType, void* paraValue)
        self.DMD.juoptSetDevPara.argtypes = [JUOPTDMD_ID, c_long, c_void_p]
        self.DMD.juoptSetDevPara.restype = c_long

    @staticmethod
    def _pearson_corr(a, b):
        """Compute Pearson correlation between two same-shaped arrays. Returns float or None."""
        if a is None or b is None:
            return None
        a = np.asarray(a, dtype=np.float32).ravel()
        b = np.asarray(b, dtype=np.float32).ravel()
        if a.size == 0 or b.size == 0 or a.size != b.size:
            return None
        a = a - float(np.mean(a))
        b = b - float(np.mean(b))
        denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
        if denom <= 0:
            return None
        r = float(np.sum(a * b) / denom)
        # numerical safety
        if r > 1.0:
            r = 1.0
        elif r < -1.0:
            r = -1.0
        return r

    def _capture_white_speckle_roi(self, roi_h, roi_w):
        """
        Project a pure-white pattern once and capture 1 speckle image, returning cropped ROI.
        Assumes camera acquisition has already been started by caller.
        """
        white_pattern = np.full((1, self.original_height, self.original_width), 255, dtype=np.uint8)
        img = self.project_and_caption(white_pattern)
        if img is None or len(img.shape) < 3:
            return None
        frame = img[0]
        if len(frame.shape) == 3:
            frame = np.mean(frame, axis=2)
        if frame.shape[0] >= roi_h and frame.shape[1] >= roi_w:
            return frame[:roi_h, :roi_w]
        return frame

    def get_devices(self):
        if not self.DMD: return []

        dev_num = c_int()
        if self.DMD.juoptGetOnlineDevNum(byref(dev_num)) != 0:
            print("Error: Failed to get device count")
            return []

        count = dev_num.value
        if count == 0:
            return []

        names_ptr = self.DMD.juoptGetOnlineDevList()
        # names_ptr is a char** (array of strings)
        dev_names = []
        for i in range(count):
            # ctypes pointer arithmetic to get string at index i
            # cast to void* then to char*? No, POINTER(c_char_p) supports indexing
            name = names_ptr[i]
            if name:
                dev_names.append(name.decode('utf-8', errors='ignore'))

        print(f"Found {count} devices: {dev_names}")
        return dev_names

    def initialize_device(self, device_name):
        if not self.DMD: return False

        dev_id_ptr = c_long()
        # V4 AllocDev takes pointer to ID and device name
        init_state = self.DMD.juoptAllocDev(byref(dev_id_ptr), bytes(device_name, 'utf-8'))

        if init_state == 0:
            self.dev_id = dev_id_ptr.value

            def initialization_failed(message):
                print(message)
                try:
                    self.DMD.juoptFree(self.dev_id)
                except Exception:
                    pass
                self.dev_id = None
                self.is_init = False
                return False

            # Configure sequence structure (1 sequence)
            ret_split = self.DMD.juoptSplitSquenceNum(self.dev_id, 1)
            if ret_split != 0:
                return initialization_failed(f"SplitSquenceNum failed: {ret_split}")

            # Initialize sequence size once with max batch size of 1000
            batch_alloc = 1000
            w = self.original_width
            h = self.original_height
            bit_plane = 0
            seq_id = 0
            ret_init = self.DMD.juoptInitSquenceSize(self.dev_id, seq_id, batch_alloc, w, h, bit_plane)
            if ret_init != 0:
                return initialization_failed(f"InitSquenceSize failed: {ret_init}")

            # Set Device Parameters (Based on SDK Demo)
            # Master Mode (Trigger Out) = 0
            # Slave Mode = 1
            # We want Master Mode

            # Work Mode: 0 = Master
            # JUOPT_SET_WMODE is 0x1010 in juopt_dll_v4.h. The old value 3 is
            # not a valid V4 parameter type and makes the DLL return error 101.
            work_mode = c_int(0)
            ret_para = self.DMD.juoptSetDevPara(self.dev_id, 0x1010, byref(work_mode))
            if ret_para != 0:
                return initialization_failed(f"SetDevPara WorkMode failed: {ret_para}")

            # Configure timing from the shared acquisition target above.
            pic_time = DMD_PICTURE_TIME_US
            illuminate_time = 0.0
            synch_delay = 0.0
            pulse_width = 10.0 # Increased to 10us for reliability
            trigger_in_delay = 0.0

            # juoptTimeControl(id, seqID, picTime, illumTime, syncDelay, pulseWidth, trigDelay)
            ret_time = self.DMD.juoptTimeControl(
                self.dev_id,
                seq_id,
                c_double(pic_time),
                c_double(illuminate_time),
                c_double(synch_delay),
                c_double(pulse_width),
                c_double(trigger_in_delay)
            )
            if ret_time != 0:
                return initialization_failed(f"TimeControl failed: {ret_time}")
            print(
                f"DMD timing set to {pic_time:.1f}us/picture "
                f"({1_000_000.0 / pic_time:.1f} Hz)"
            )

            self.is_init = True
            print(f"Device '{device_name}' initialized (ID: {self.dev_id})")
            return True
        else:
            print(f"Initialization failed with error code {init_state}")
            return False

    def load_pattern(self, matrix):
        if not self.is_init:
            print("Error: Device not initialized")
            return False

        batch = matrix.shape[0]
        if batch > 1000:
            print(f"Error: Batch size {batch} exceeds allocated size 1000")
            return False
        # Keep one contiguous byte buffer for the DLL. ``flatten().tobytes()``
        # creates two full-size copies, which is costly for 1000-frame batches.
        matrix = np.ascontiguousarray(matrix, dtype=np.uint8)
        arr = matrix.tobytes(order='C')

        seq_id = 0
        w = self.original_width
        h = self.original_height
        bit_plane = 0 # 1-bit mode

        # Load data
        sig_bit = 1 # MSB of 8-bit input
        ret = self.DMD.juoptLoadPixmap(
            self.dev_id,
            seq_id,
            0, # x offset
            bit_plane,
            batch,
            c_char_p(arr),
            sig_bit,
            w,
            h
        )

        if ret == 0:
            self.current_pattern = matrix
            return True
        else:
            print(f"LoadPixmap failed: {ret}")
            return False

    def clear_sequence(self, seq_id=0):
        """Clear sequence data from DMD memory"""
        if not self.is_init:
            return False
        ret = self.DMD.juoptClearOneSquenceData(self.dev_id, seq_id)
        if ret != 0:
            print(f"ClearOneSquenceData failed: {ret}")
            return False
        return True

    def project_and_caption(self, pattern=None):
        if not self.is_init:
            return None

        t0 = time.time()

        if pattern is not None:
            if not self.load_pattern(pattern):
                return None

        t1 = time.time()

        # Determine batch size from pattern or previous logic
        # If pattern is provided, we use its size.
        # If pattern is None, we assume the previously loaded pattern is used?
        # But we need batch size for the camera loop.
        # The original code crashed if pattern was None because 'batch' variable wasn't defined if pattern was None.
        # But 'batch = pattern.shape[0]' was at the top.
        # So pattern MUST be not None in original code too.
        if pattern is None:
             return None

        batch = pattern.shape[0]
        image_data = np.ndarray((batch, 128, 128))

        # Start Projection
        # juoptProjection(id, seqID, y_offset)
        t2 = time.time()
        self.DMD.juoptProjection(self.dev_id, 0, 0)

        # Capture images
        total_wait_time = 0
        total_process_time = 0
        frame_timestamps_ns = []
        frame_ids = []

        for i in range(batch):
            img, w_t, p_t = self.camera.run()
            if img is not None:
                image_data[i] = img
                if self.camera.last_frame_timestamp_ns is not None:
                    frame_timestamps_ns.append(
                        self.camera.last_frame_timestamp_ns
                    )
                if self.camera.last_frame_id is not None:
                    frame_ids.append(self.camera.last_frame_id)
            else:
                print(f"Warning: Failed to capture image {i}")

            total_wait_time += w_t
            total_process_time += p_t

        # Stop Projection
        self.DMD.juoptStop(self.dev_id)
        self.clear_sequence(0)

        t3 = time.time()

        avg_wait = total_wait_time / batch if batch > 0 else 0
        avg_proc = total_process_time / batch if batch > 0 else 0

        print(f"DEBUG: load_pattern={t1-t0:.4f}s, projection={t2-t1:.4f}s")
        print(f"DEBUG: capture_loop={t3-t2:.4f}s (Avg Wait={avg_wait:.4f}s, Avg Process={avg_proc:.4f}s)")
        if len(frame_timestamps_ns) >= 2:
            intervals_ms = np.diff(
                np.asarray(frame_timestamps_ns, dtype=np.float64)
            ) / 1_000_000.0
            expected_interval_ms = DMD_PICTURE_TIME_US / 1000.0
            median_interval_ms = float(np.median(intervals_ms))
            actual_fps = (
                1000.0 / median_interval_ms
                if median_interval_ms > 0
                else 0.0
            )
            anomaly_mask = (
                (intervals_ms < expected_interval_ms * 0.75)
                | (intervals_ms > expected_interval_ms * 1.25)
            )
            estimated_missed_triggers = int(np.sum(np.maximum(
                np.rint(intervals_ms / expected_interval_ms).astype(np.int64) - 1,
                0,
            )))
            print(
                "DEBUG: camera_timestamps={} actual_fps={:.2f}, "
                "interval_ms median={:.4f} min={:.4f} max={:.4f}, "
                "anomalies={}/{}, estimated_missed_triggers={}".format(
                    len(frame_timestamps_ns),
                    actual_fps,
                    median_interval_ms,
                    float(np.min(intervals_ms)),
                    float(np.max(intervals_ms)),
                    int(np.count_nonzero(anomaly_mask)),
                    intervals_ms.size,
                    estimated_missed_triggers,
                )
            )
        if len(frame_ids) >= 2:
            frame_id_steps = np.diff(np.asarray(frame_ids, dtype=np.int64))
            frame_id_gaps = int(np.sum(np.maximum(frame_id_steps - 1, 0)))
            if frame_id_gaps:
                print(
                    f"WARNING: camera frame ID gaps detected: {frame_id_gaps}"
                )
        print(f"DEBUG: total={t3-t0:.4f}s for {batch} images")

        return image_data



    # GGS 21
    # ---------------------------
    # GGS21算法
    # ---------------------------
    def GGS2_1(self, P, y, iters=300, step=0.8, init=None):
        """
        Generalized GS 2-1
        P: 探测矩阵, 形状 (M, N)
        y: 测量强度, 形状 (M, N)
        """
        P = torch.tensor(P, dtype=torch.complex64).reshape(P.shape[0], -1)
        y = torch.tensor(y, dtype=torch.float64)
        M, N = P.shape
        y = y.T
        a = torch.randn(1, N) + 1j * torch.randn(1, N)
        # 测试过随机初始化,效果不好只有0.5

        # a_his = [a.clone()]
        y_sqrt = torch.sqrt(torch.clamp(y, min=0.0))
        P_pinv = torch.linalg.pinv(P)
        pow_n = 2

        for t in range(1, iters + 1):
            # (N_ROWS, N) @ (N, M) -> (N_ROWS, M)
            if(t > (iters*2) / 3):
                pow_n = pow_n - ( 1 / (iters / 2) )

            E = P @ a.T
            E = torch.pow(y_sqrt, pow_n) * torch.exp(1j * torch.angle(E.squeeze()))
            E = E.to(torch.complex64)
            a = (P_pinv @ E).T

            # a_his.append(a.clone())

        return iters, a.detach().cpu().numpy()


    def start_measurement(self):
        if not self.optimization_running:
            self.optimization_running = True
            self.measurement_error = None
            self.pbr_history = []

            def _run_measurement_guarded():
                try:
                    self.run_measurement()
                except Exception as exc:
                    self.measurement_error = str(exc)
                    print("Measurement failed: {}".format(exc))
                finally:
                    self.optimization_running = False

            thread = threading.Thread(target=_run_measurement_guarded)
            thread.daemon = True
            thread.start()

    def start_reconstruction(self):
        thread = threading.Thread(target=self.run_reconstruction)
        thread.daemon = True
        thread.start()

    def _prepare_ggs21_linear_solver(self, X, base_dir, probe_file):
        """Prepare the ordinary pseudoinverse projection used by GGS21."""
        del base_dir, probe_file
        return torch.linalg.pinv(X)

    def _apply_ggs21_linear_solver(self, X, solver_state, detector_field):
        """Apply the ordinary pseudoinverse projection to one output block."""
        del X
        return solver_state @ detector_field

    def stop_optimization(self):
        self.optimization_running = False

    def run_measurement(self):
        # Full-size micro-pixel reconstruction (1024 x 768) - Measurement Phase
        # ---------------------------------------------------------------------

        # Parameters (you can tune these)
        N_x = self.dmd_width   # 32
        N_y = self.dmd_height  # 24
        N_in = N_x * N_y            # number of micro-pixels
        pattern_config = get_active_pattern_config()

        # Which camera ROI pixels to reconstruct (use full ROI by default)
        roi_w = getattr(self.camera, 'roi_width', 128)
        roi_h = getattr(self.camera, 'roi_height', 128)
        N_out = roi_w * roi_h

        # Number of probes is controlled by PATTERN_CONFIG at the top of this
        # file so acquisition and reconstruction always select the same set.
        M = N_in * pattern_config["probe_multiplier"]

        # Files for measurements
        meas_file = os.path.join(
            os.getcwd(),
            pattern_config["measurement_filename"],
        )

        print(
            f"Starting measurement phase: N_in={N_in}, N_out={N_out}, "
            f"M={M}, pattern_set={pattern_config['name']}"
        )

        # Keep the legacy uint16 container so the reconstruction reader and
        # raw-file size checks remain compatible. Values are native
        # Polarized8 codes in the range 0..255.
        meas_shape = (M, N_out)
        meas_mm = np.memmap(meas_file, dtype='uint16', mode='w+', shape=meas_shape)

        pregenerated_dir = pattern_config["directory"]
        probe_file = os.path.join(pregenerated_dir, "probe.npy")
        pattern_file = os.path.join(pregenerated_dir, "patterns_pregenerated.npy")

        if os.path.exists(probe_file) and os.path.exists(pattern_file):
            print("\n" + "="*70)
            print("Loading pre-generated patterns...")
            print("="*70)

            # Load probes and patterns
            # Memory-map the multi-GB pattern file and read only each projected
            # batch instead of loading the entire 8N dataset into RAM.
            P = np.load(probe_file, mmap_mode="r")
            full_patterns = np.load(pattern_file, mmap_mode="r")

            print(f"  Loaded probes: {P.shape}")
            print(f"  Loaded patterns: {full_patterns.shape}")

            # Verify dimensions
            assert P.shape[0] == M, f"Probe count mismatch: {P.shape[0]} != {M}"
            assert full_patterns.shape[0] == M, f"Pattern count mismatch: {full_patterns.shape[0]} != {M}"

            print("  ✓ Pre-generated patterns loaded successfully!")
            print("="*70 + "\n")

        else:
            print("\n" + "="*70)
            print("WARNING: Pre-generated patterns not found!")
            print("="*70)
            print(f"  Expected files:")
            print(f"    - {probe_file}")
            print(f"    - {pattern_file}")
            print(f"\n  Falling back to on-the-fly generation...")
            print(f"  (This will be much slower)")
            print("="*70 + "\n")

            # Fall back to original generation code
            from holograms.generate_LUT import generate_lut
            from holograms.dmd_holograms import holo_SP

            def field_to_lee_hologram_micro(field, px=4):
                ds_method = 'mean'
                f_val, px_comb, lut = generate_lut('sp', px)
                holo = holo_SP(field, lut, px_comb, ds_method=ds_method)
                holo_norm = holo.astype(np.float32)
                if np.max(holo_norm) > np.min(holo_norm):
                    holo_norm = 255 * (holo_norm - np.min(holo_norm)) / (np.max(holo_norm) - np.min(holo_norm))
                return holo_norm.astype(np.uint8)

            # Generate probes
            P_amp = np.ones((M, N_y, N_x), dtype=np.float32)
            P_phase = np.random.randint(0, 2*8, (M, N_y, N_x)).astype(np.float32) * np.pi / 8
            P = P_amp * np.exp(1j * P_phase)
            P = P.astype(np.complex64)

            # Save generated probes for reconstruction
            np.save('probes.npy', P)
            print("Generated probes saved to probes.npy")

            # Generate patterns on-the-fly
            full_patterns = np.zeros((M, self.original_height, self.original_width), dtype=np.uint8)

            print("Generating patterns on-the-fly (this may take a while)...")
            for m in tqdm(range(M), desc='Generating patterns'):
                field = P[m]
                field_upscaled = np.kron(field, np.ones((self.pixel_group_size, self.pixel_group_size)))
                hologram = field_to_lee_hologram_micro(field_upscaled)
                full_patterns[m] = hologram

                # Update progress via callback (pattern generation phase: 0-10%)
                if self.measure_progress_callback and m % 10 == 0:
                    progress = 10 * (m + 1) / M
                    self.measure_progress_callback(progress, f"Generating patterns: {m + 1}/{M}")

        # Measurement loop using the DMD's preallocated 1000-pattern sequence.
        batch_size = 1000

        print("\nStarting measurement phase...")
        for m in tqdm(range(0, M, batch_size), desc='Measuring probes', unit='batch'):
            if not self.optimization_running:
                print("Optimization stopped by user during measurement phase")
                break

            batch_count = min(batch_size, M - m)
            batch_patterns = full_patterns[m:m+batch_count]

            # ---- Stability check: white pattern before each batch ----
            # Use a separate acquisition session. Stopping acquisition after
            # the white frame flushes the camera stream buffers so a blank
            # frame left by juoptStop/clear_sequence cannot become frame 0 of
            # the following measurement batch.
            self.camera.start()
            try:
                try:
                    white_roi = self._capture_white_speckle_roi(roi_h, roi_w)
                    if self._stability_baseline_white_speckle is None and white_roi is not None:
                        self._stability_baseline_white_speckle = white_roi.copy()
                        self.current_stability_corr = 1.0
                        self._stability_seq += 1
                    elif self._stability_baseline_white_speckle is not None and white_roi is not None:
                        self.current_stability_corr = self._pearson_corr(
                            self._stability_baseline_white_speckle, white_roi
                        )
                        self._stability_seq += 1
                    else:
                        self.current_stability_corr = None
                except Exception as e:
                    print(f"Stability check failed: {e}")
                    self.current_stability_corr = None
            finally:
                self.camera.stop()

            # Start a clean acquisition session for the actual batch.
            self.camera.start()
            try:
                img = self.project_and_caption(batch_patterns)
            finally:
                self.camera.stop()

            if img is None:
                print(f"Warning: no image for probe {m}")
                meas_mm[m:m+batch_count, :] = 0.0
                continue

            # Crop to ROI
            roi = img[:, :roi_h, :roi_w] if img.shape[1] >= roi_h and img.shape[2] >= roi_w else img
            meas_mm[m:m+batch_count, :] = roi.reshape(batch_count, -1).astype('uint16')

            # Display the final speckle from this completed batch. The camera
            # owns the copy and forwards it to either the legacy Tk view or the
            # modular Qt workflow without blocking acquisition.
            self.camera.publish_latest_image(m + batch_count, roi[-1])

            # Progress update
            if (m + batch_count) % 1000 == 0 or m == 0:
                print(f"Measured up to {m + batch_count}/{M} probes")

            # Update progress via callback (measurement phase: 10-100%)
            if self.measure_progress_callback:
                progress = 10 + 90 * (m + batch_count) / M
                corr_txt = "-" if self.current_stability_corr is None else f"{self.current_stability_corr:.4f}"
                self.measure_progress_callback(
                    progress,
                    f"Measuring probes: {m + batch_count}/{M} | Stability Corr: {corr_txt}"
                )

        # A normal run reaches this point while optimization_running is still
        # true. A user stop clears it, so callers can reject the partial file.
        completed = bool(self.optimization_running)

        # flush memmap to disk
        del meas_mm
        print(
            "Measurement phase completed."
            if completed
            else "Measurement phase stopped before completion."
        )
        return completed

    # def run_reconstruction(self):
    #     self.reconstruction_running = True
    #     try:
    #         print("Starting reconstruction phase...")

    #         # Parameters
    #         N_x = self.dmd_width   # 32
    #         N_y = self.dmd_height  # 24
    #         N_in = N_x * N_y

    #         roi_w = getattr(self.camera, 'roi_width', 128)
    #         roi_h = getattr(self.camera, 'roi_height', 128)
    #         N_out = roi_w * roi_h

    #         M = N_in * 4

    #         # Files
    #         meas_file = os.path.join(os.getcwd(), 'measurements_memmap.npy')
    #         H_file = os.path.join(os.getcwd(), 'transmission_matrix_memmap.npy')

    #         if not os.path.exists(meas_file):
    #             print(f"Error: Measurement file not found: {meas_file}")
    #             return

    #         # Load probes P
    #         pregenerated_dir = "./pregenerated_patterns"
    #         probe_file = os.path.join(pregenerated_dir, "probe.npy")

    #         if os.path.exists(probe_file):
    #              P = np.load(probe_file)
    #         elif os.path.exists('probes.npy'):
    #              P = np.load('probes.npy')
    #         else:
    #              print("Error: Probes file not found. Cannot reconstruct.")
    #              return

    #         # Load measurements
    #         meas_shape = (M, N_out)
    #         meas_mm = np.memmap(meas_file, dtype='uint16', mode='r', shape=meas_shape)

    #         # Prepare to reconstruct H column-by-column (per output pixel)
    #         # memmap for result: complex64 to save space
    #         H_shape = (N_out, N_in)
    #         H_mm = np.memmap(H_file, dtype='complex64', mode='w+', shape=H_shape)

    #         # Reconstruction per output pixel
    #         print("Starting reconstruction of transmission matrix...")
    #         # 如果存在reconstructed_field.npy就直接读取
    #         if os.path.exists("reconstructed_field.npy"):
    #             # print("存在reconstructed_field.npy,直接读取")
    #             H_mm = np.load("reconstructed_field.npy")
    #         else:
    #             for j in tqdm(range(N_out), desc='Reconstructing output pixels', unit='pixel'):
    #                 y = meas_mm[:, j]
    #                 # run streaming GGS2_1
    #                 try:
    #                     _, hvec = self.GGS2_1(P, y, iters=200, step=0.8, init=None)
    #                     # store as complex64
    #                     H_mm[j, :] = hvec.astype(np.complex64)
    #                 except Exception as e:
    #                     print(f"Reconstruction failed for output pixel {j}: {e}")
    #                     H_mm[j, :] = 0
    #                 if j % 100 == 0:
    #                     print(f"Reconstructed {j}/{N_out} output pixels")

    #                 # Update progress via callback (reconstruction phase: 0-100%)
    #                 if self.recon_progress_callback:
    #                     progress = 100 * (j + 1) / N_out
    #                     self.recon_progress_callback(progress, f"Reconstructing: {j + 1}/{N_out} pixels")

    #         # finalize
    #         del H_mm
    #         print("Full-size reconstruction completed. Transmission matrix saved to:", H_file)
    #         # Also save as reconstructed_field.npy for compatibility if needed
    #         # shutil.copy(H_file, 'reconstructed_field.npy') # Optional

    #     except Exception as e:
    #         print(f"Error during reconstruction phase: {e}")
    #     finally:
    #         self.reconstruction_running = False




    def run_reconstruction(self):
        """
        使用 GGS 2-1 恢复传输矩阵。

        数据关系：
            X: (M, N_in)
            y: (M, N_out)
            H: (N_out, N_in)

            y = abs(X @ H.T)
        """
        self.reconstruction_running = True
        self.reconstruction_error = None
        H_mm = None

        try:
            print("Starting GGS 2-1 reconstruction...")

            # ==================== 基本参数 ====================

            N_in = self.dmd_width * self.dmd_height
            pattern_config = get_active_pattern_config()

            roi_w = getattr(self.camera, "roi_width", 128)
            roi_h = getattr(self.camera, "roi_height", 128)
            N_out = roi_w * roi_h

            iters = int(getattr(self, "ggs21_iters", 200))
            ratio = float(getattr(self, "ggs21_ratio", 0.89))

            # 和 MATLAB 版 nWorker 一样，表示分成多少个输出块
            n_worker = max(
                1,
                int(getattr(self, "ggs21_n_worker", 8))
            )

            use_gpu = bool(getattr(self, "ggs21_use_gpu", True))

            # 相机暗场标量，可根据实验设置
            dark_level = float(
                getattr(self, "ggs21_dark_level", 0.0)
            )

            # uint16 相机数据通常是强度 I，需要转换为 sqrt(I)
            measurements_are_intensity = bool(
                getattr(self, "measurements_are_intensity", True)
            )

            if iters <= 0:
                raise ValueError("ggs21_iters must be positive")

            if not 0.0 <= ratio <= 1.0:
                raise ValueError("ggs21_ratio must be in [0, 1]")

            device = torch.device(
                "cuda"
                if use_gpu and torch.cuda.is_available()
                else "cpu"
            )

            if use_gpu and device.type != "cuda":
                print("Warning: CUDA unavailable, using CPU.")

            # ==================== 文件路径 ====================

            base_dir = os.getcwd()

            meas_file = os.path.join(
                base_dir,
                pattern_config["measurement_filename"],
            )

            H_file = os.path.join(
                base_dir,
                self.tm_memmap_filename
            )

            # 防止重建失败时破坏已有的传输矩阵
            partial_file = H_file + ".partial"

            cache_file = os.path.join(
                base_dir,
                self.reconstructed_filename
            )

            err_file = os.path.join(
                base_dir,
                self.error_curve_filename
            )

            if not os.path.exists(meas_file):
                raise FileNotFoundError(
                    f"Measurement file not found: {meas_file}"
                )

            # ==================== 加载探针 ====================

            probe_candidates = [
                os.path.join(
                    pattern_config["directory"],
                    "probe.npy"
                ),
                os.path.join(base_dir, "probes.npy"),
            ]

            probe_file = next(
                (
                    path
                    for path in probe_candidates
                    if os.path.exists(path)
                ),
                None
            )

            if probe_file is None:
                raise FileNotFoundError(
                    "Neither probe.npy nor probes.npy was found"
                )

            probes = np.load(probe_file, mmap_mode="r")

            # probe.npy 可能保存为：
            # (M, height, width) 或 (M, N_in)
            if probes.ndim == 3:
                expected_shape = (
                    self.dmd_height,
                    self.dmd_width,
                )

                if probes.shape[1:] != expected_shape:
                    raise ValueError(
                        f"Probe spatial shape {probes.shape[1:]} "
                        f"does not match DMD shape {expected_shape}"
                    )

                # (M, 24, 32) -> (M, 768)
                X_np = probes.reshape(
                    probes.shape[0],
                    N_in,
                )

            elif probes.ndim == 2:
                if probes.shape[1] == N_in:
                    # 已经是 (M, N_in)
                    X_np = probes
                elif probes.shape[0] == N_in:
                    # (N_in, M) -> (M, N_in)
                    X_np = probes.T
                else:
                    raise ValueError(
                        f"Probe shape {probes.shape} is incompatible "
                        f"with N_in={N_in}"
                    )

            else:
                raise ValueError(
                    f"Probes must be 2-D or 3-D, got {probes.shape}"
                )


            # # GGS 内部使用 X: (M, N_in)
            # if probes.shape[1] == N_in:
            #     X_np = probes
            # elif probes.shape[0] == N_in:
            #     X_np = probes.T
            # else:
            #     raise ValueError(
            #         f"Probe shape {probes.shape} is incompatible "
            #         f"with N_in={N_in}"
            #     )

            M = X_np.shape[0]

            expected_probe_count = pattern_config["probe_multiplier"] * N_in
            if M != expected_probe_count:
                print(
                    f"Warning: pattern set {pattern_config['name']} expects "
                    f"{expected_probe_count} probes, but found {M}."
                )

            # ==================== 加载测量数据 ====================

            expected_bytes = (
                M
                * N_out
                * np.dtype(np.uint16).itemsize
            )

            actual_bytes = os.path.getsize(meas_file)

            if actual_bytes != expected_bytes:
                raise ValueError(
                    f"Measurement file size mismatch: "
                    f"got {actual_bytes} bytes, "
                    f"expected {expected_bytes} bytes for "
                    f"shape ({M}, {N_out}) uint16"
                )

            measurements = np.memmap(
                meas_file,
                dtype=np.uint16,
                mode="r",
                shape=(M, N_out),
            )

            H_shape = (N_out, N_in)

            # MATLAB GPU 版的分块方式
            n_worker = min(n_worker, N_out)
            chunk_size = math.ceil(N_out / n_worker)

            switch_iter = round(ratio * iters)

            print(
                f"Device: {device}\n"
                f"X shape: ({M}, {N_in})\n"
                f"H shape: {H_shape}\n"
                f"Output blocks: {n_worker}\n"
                f"Iterations: {iters}\n"
                f"GS-2 -> GS-1 at iteration: {switch_iter}"
            )

            # ==================== 创建输出 memmap ====================

            H_mm = np.memmap(
                partial_file,
                dtype=np.complex64,
                mode="w+",
                shape=H_shape,
            )

            error_sum = np.zeros(
                iters,
                dtype=np.float64
            )

            start_time = time.perf_counter()

            # ==================== GGS 2-1 ====================

            with torch.inference_mode():

                X = torch.as_tensor(
                    np.ascontiguousarray(X_np),
                    dtype=torch.complex64,
                    device=device,
                )

                # 只计算一次伪逆
                linear_solver = self._prepare_ggs21_linear_solver(
                    X,
                    base_dir,
                    probe_file,
                )

                for start in range(0, N_out, chunk_size):

                    stop = min(
                        start + chunk_size,
                        N_out
                    )

                    block_width = stop - start

                    # 测量块：(M, block_width)
                    measured = np.asarray(
                        measurements[:, start:stop],
                        dtype=np.float32,
                    ).copy()

                    # 暗场扣除
                    if dark_level != 0:
                        measured -= dark_level

                    np.maximum(
                        measured,
                        0.0,
                        out=measured
                    )

                    if measurements_are_intensity:
                        # 相机测得 I = |XH^T|²
                        # GGS21 输入需要 y = |XH^T|
                        np.sqrt(
                            measured,
                            out=measured
                        )

                    y = torch.as_tensor(
                        measured,
                        dtype=torch.float32,
                        device=device,
                    )

                    y_squared = y.square()

                    # 随机相位初始化
                    phase_generator = torch.Generator(device=device)
                    phase_generator.manual_seed(
                        int(self.ggs21_random_seed) + start
                    )
                    random_phase = torch.rand(
                        y.shape,
                        dtype=y.dtype,
                        device=device,
                        generator=phase_generator,
                    ) * (2.0 * torch.pi)

                    Y = torch.polar(
                        y,
                        random_phase
                    )

                    chunk_error = torch.empty(
                        iters,
                        dtype=torch.float32,
                        device=device,
                    )

                    for iteration in range(iters):

                        H_t = self._apply_ggs21_linear_solver(
                            X,
                            linear_solver,
                            Y,
                        )

                        # 正向传播
                        Y_iter = X @ H_t

                        # 记录振幅误差
                        residual = (
                            torch.abs(Y_iter) - y
                        )

                        chunk_error[iteration] = (
                            torch.linalg.vector_norm(
                                residual,
                                dim=0
                            ).mean()
                        )

                        phase = torch.polar(
                            torch.ones_like(y),
                            torch.angle(Y_iter),
                        )

                        if iteration < switch_iter:
                            # GS-2
                            Y = y_squared * phase
                        else:
                            # GS-1
                            Y = y * phase

                    # H_t: (N_in, block_width)
                    # H_block: (block_width, N_in)
                    H_block = H_t.T

                    # 和 MATLAB 实现保持一致
                    invalid = torch.abs(H_block) > 1e10

                    if torch.any(invalid):
                        H_block[invalid] = torch.polar(
                            torch.ones_like(
                                torch.abs(H_block[invalid])
                            ),
                            torch.angle(H_block[invalid]),
                        )

                    H_mm[start:stop] = (
                        H_block
                        .cpu()
                        .numpy()
                        .astype(np.complex64, copy=False)
                    )

                    error_sum += (
                        chunk_error.cpu().numpy()
                        * block_width
                    )

                    H_mm.flush()

                    progress = 100.0 * stop / N_out

                    print(
                        f"Reconstructed "
                        f"{stop}/{N_out} output pixels"
                    )

                    callback = getattr(
                        self,
                        "recon_progress_callback",
                        None
                    )

                    if callback:
                        callback(
                            progress,
                            f"GGS21 reconstruction: "
                            f"{stop}/{N_out} pixels"
                        )

                    del (
                        measured,
                        y,
                        y_squared,
                        Y,
                        Y_iter,
                        H_t,
                        H_block,
                        chunk_error,
                    )

            # ==================== 全局归一化 ====================

            # 对应 MATLAB：
            # A_pr = A_pr / std(A_pr, 1, 'all')

            std_value = float(np.std(H_mm))

            if std_value > 0:
                normalize_rows = 1024

                for start in range(
                    0,
                    N_out,
                    normalize_rows
                ):
                    stop = min(
                        start + normalize_rows,
                        N_out
                    )

                    H_mm[start:stop] /= std_value

            H_mm.flush()
            del H_mm
            H_mm = None

            # 重建全部成功后再替换正式文件
            os.replace(
                partial_file,
                H_file
            )

            # ==================== 保存误差曲线 ====================

            error_curve = (
                error_sum / N_out
            ).astype(np.float32)

            np.save(
                err_file,
                error_curve
            )

            # reconstructed_field.npy 是标准 NPY 文件
            H_source = np.memmap(
                H_file,
                dtype=np.complex64,
                mode="r",
                shape=H_shape,
            )

            np.save(
                cache_file,
                H_source
            )

            elapsed = (
                time.perf_counter()
                - start_time
            )

            print(
                f"GGS21 reconstruction completed "
                f"in {elapsed:.2f} seconds."
            )

            print(
                "Transmission matrix saved to:",
                H_file
            )

            print(
                "Error curve saved to:",
                err_file
            )

        except Exception as exc:

            self.reconstruction_error = str(exc)

            if (
                isinstance(exc, RuntimeError)
                and "out of memory" in str(exc).lower()
            ):
                print(
                    "CUDA out of memory. Increase "
                    "self.ggs21_n_worker to reduce "
                    "the number of output pixels per block."
                )

            print(
                f"Error during GGS21 reconstruction: {exc}"
            )

        finally:

            if H_mm is not None:
                H_mm.flush()
                del H_mm

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.reconstruction_running = False

    def conjugate_focus_at_position(self, target_x, target_y, px=4, ds_method='mean'):
        """
        在选定位置进行传输矩阵的共轭聚焦

        参数:
            target_x: 目标聚焦位置的X坐标 (0-1023)
            target_y: 目标聚焦位置的Y坐标 (0-767)
            px: 超像素尺寸 (默认为4)
            ds_method: 下采样方法 ('mean', 'max', 'min', 'center', 'side')

        返回:
            dict: 包含以下键的字典:
                - success: 是否成功 (bool)
                - focused_image: 捕获的聚焦图像 (numpy.ndarray)
                - target_index: 目标在传输矩阵中的索引 (int)
                - peak_intensity: 峰值强度 (float)
                - mean_intensity: 平均强度 (float)
                - background_intensity: 背景强度 (float)
                - pbr: 峰值背景比 (float)
                - error: 错误信息 (str，仅在失败时)
        """
        result = {
            'success': False,
            'focused_image': None,
            'target_index': None,
            'peak_intensity': 0.0,
            'mean_intensity': 0.0,
            'background_intensity': 0.0,
            'pbr': 0.0,
            'error': None
        }

        try:
            print("\n" + "="*70)
            print("开始传输矩阵共轭聚焦")
            print("="*70)
            print(f"目标位置: ({target_x}, {target_y})")

            # 1. 从reconstructed_field.npy中读取传输矩阵
            tm_file = os.path.join(os.getcwd(), self.reconstructed_filename)
            if not os.path.exists(tm_file):
                raise FileNotFoundError(f"传输矩阵文件不存在: {tm_file}")

            print(f"正在加载传输矩阵: {tm_file}")
            H = np.load(tm_file)
            print(f"传输矩阵形状: {H.shape}")
            print(f"传输矩阵数据类型: {H.dtype}")

            # H的形状应该是 (N_out, N_in)
            # N_out = 16384 (128x128的ROI像素)
            # N_in = 768 (32x24的DMD微像素)

            # 2. 计算目标位置在相机ROI中的索引
            roi_w = getattr(self.camera, 'roi_width', 128)
            roi_h = getattr(self.camera, 'roi_height', 128)

            # 将目标坐标映射到ROI坐标系
            # 假设目标坐标是相对于完整相机图像的
            # 这里我们简化处理，直接使用目标坐标作为ROI内的坐标
            if target_x >= roi_w or target_y >= roi_h:
                print(f"警告: 目标位置 ({target_x}, {target_y}) 超出ROI范围 ({roi_w}x{roi_h})")
                print(f"将使用中心位置 ({roi_w//2}, {roi_h//2})")
                target_x = roi_w // 2
                target_y = roi_h // 2

            # 计算在传输矩阵中的列索引
            col_index = target_y * roi_w + target_x
            print(f"传输矩阵列索引: {col_index}")
            result['target_index'] = col_index

            # 3. 提取对应的传输矩阵列并进行共轭
            h_column = H[col_index, :]  # 形状: (N_in,)
            h_conjugate = build_conjugate_focus_field(h_column)
            print(f"提取的传输矩阵列形状: {h_column.shape}")
            print(f"共轭后的列形状: {h_conjugate.shape}")

            # 4. 将共轭向量重塑为DMD输入场
            # DMD输入场是 (N_y, N_x) = (24, 32)
            N_x = self.dmd_width   # 32
            N_y = self.dmd_height  # 24

            # 重塑为二维场
            input_field = h_conjugate.reshape(N_y, N_x)
            print(f"输入场形状: {input_field.shape}")

            # 5. 上采样到DMD完整分辨率
            full_pattern = np.kron(input_field, np.ones((32, 32)))
            print(f"上采样后全息图形状: {full_pattern.shape}")

            # 6. 使用全息图生成器产生DMD的pattern
            print("\n生成全息图...")
            from holograms.generate_LUT import generate_lut
            from holograms.dmd_holograms import holo_SP

            # 生成查找表
            f_val, px_comb, lut = generate_lut('sp', px)
            print(f"超像素尺寸: {px}x{px}")

            # 生成全息图
            hologram = holo_SP(full_pattern, lut, px_comb, ds_method=ds_method)
            print(f"全息图形状: {hologram.shape}")

            # 归一化到0-255范围
            holo_norm = hologram.astype(np.float32)
            if np.max(holo_norm) > np.min(holo_norm):
                holo_norm = 255 * (holo_norm - np.min(holo_norm)) / (np.max(holo_norm) - np.min(holo_norm))
            hologram_uint8 = holo_norm.astype(np.uint8)
            print(f"全息图归一化完成，范围: [{np.min(hologram_uint8)}, {np.max(hologram_uint8)}]")

            full_hologram = hologram_uint8

            # 确保尺寸匹配DMD分辨率
            if full_hologram.shape[0] != self.original_height or full_hologram.shape[1] != self.original_width:
                print(f"调整全息图尺寸以匹配DMD分辨率 ({self.original_height}x{self.original_width})")
                # 使用resize或padding调整尺寸
                from cv2 import resize
                full_hologram = resize(full_hologram, (self.original_width, self.original_height),
                                       interpolation=cv2.INTER_NEAREST)
                print(f"调整后全息图形状: {full_hologram.shape}")

            # 7. 将全息图加载到DMD并投影
            print("\n加载全息图到DMD...")
            pattern_batch = np.array([full_hologram], dtype=np.uint8)

            if not self.load_pattern(pattern_batch):
                raise RuntimeError("加载全息图到DMD失败")
            print("全息图加载成功")

            # 8. 使用camera进行读取
            print("\n启动相机捕获...")
            self.camera.start()

            # 投影全息图
            print("投影全息图...")
            self.DMD.juoptProjection(self.dev_id, 0, 0)

            # 等待投影稳定
            time.sleep(0.1)

            # 捕获图像
            print("捕获图像...")
            captured_image, _, _ = self.camera.run()
            self.camera.stop()
            self.DMD.juoptStop(self.dev_id)
            self.clear_sequence(0)

            if captured_image is None:
                raise RuntimeError("相机捕获失败")

            print(f"捕获图像形状: {captured_image.shape}")
            print(f"捕获图像数据类型: {captured_image.dtype}")
            print(f"捕获图像范围: [{np.min(captured_image)}, {np.max(captured_image)}]")

            # 9. 分析聚焦效果：采用四相位 PBR 算法，峰值区域扩大为 10×10
            if len(captured_image.shape) == 3:
                captured_image = captured_image[:, :, 0]

            roi_height, roi_width = captured_image.shape
            x = int(np.clip(target_x, 5, roi_width - 5))
            y = int(np.clip(target_y, 5, roi_height - 5))
            peak_region = captured_image[y-5:y+5, x-5:x+5]
            peak_intensity = np.max(peak_region).item()
            mean_intensity = (np.sum(captured_image) - np.max(captured_image)) / (captured_image.size - 1)

            # 背景为排除目标位置 10×10 区域后的其余像素。
            bg_mask = np.ones_like(captured_image, dtype=bool)
            bg_mask[y-5:y+5, x-5:x+5] = False
            background_intensity = max(captured_image[bg_mask].mean().item(), 1e-6)
            pbr = peak_intensity / background_intensity

            print("\n" + "="*70)
            print("聚焦结果分析")
            print("="*70)
            print(f"峰值强度: {peak_intensity:.2f}")
            print(f"平均强度: {mean_intensity:.2f}")
            print(f"背景强度: {background_intensity:.2f}")
            print(f"峰值背景比 (PBR): {pbr:.2f}")
            print("="*70 + "\n")

            # 更新结果字典
            result['success'] = True
            result['focused_image'] = captured_image
            result['peak_intensity'] = peak_intensity
            result['mean_intensity'] = mean_intensity
            result['background_intensity'] = background_intensity
            result['pbr'] = pbr

        except Exception as e:
            print(f"聚焦过程出错: {str(e)}")
            result['error'] = str(e)

        return result

    def _load_transmission_matrix(self, tm_filename=None):
        if tm_filename is None:
            tm_filename = self.reconstructed_filename
        tm_file = os.path.join(os.getcwd(), tm_filename)
        if not os.path.exists(tm_file):
            raise FileNotFoundError(f"传输矩阵文件不存在: {tm_file}")
        H = np.load(tm_file)
        return H

    def _build_focus_hologram(
        self,
        tm_row,
        px=4,
        ds_method='mean',
        lut_cache=None,
    ):
        """Encode one TM row as a full-resolution V4 DMD hologram."""
        tm_row = np.asarray(tm_row, dtype=np.complex64)
        expected_input_count = self.dmd_height * self.dmd_width
        if tm_row.shape != (expected_input_count,):
            raise ValueError(
                'TM row shape {} does not match ({},)'.format(
                    tm_row.shape, expected_input_count
                )
            )
        if not np.all(np.isfinite(tm_row)):
            raise ValueError('TM row contains NaN or infinity')

        input_field = build_conjugate_focus_field(tm_row).reshape(
            self.dmd_height, self.dmd_width
        )
        full_pattern = np.kron(
            input_field,
            np.ones((self.pixel_group_size, self.pixel_group_size)),
        )
        if lut_cache is None:
            lut_cache = generate_lut('sp', px)
        if not isinstance(lut_cache, tuple) or len(lut_cache) != 3:
            raise RuntimeError('Failed to generate the superpixel LUT')
        _, px_comb, lut = lut_cache
        hologram = holo_SP(
            full_pattern,
            lut,
            px_comb,
            ds_method=ds_method,
        ).astype(np.float32)
        hologram_min = float(np.min(hologram))
        hologram_max = float(np.max(hologram))
        if hologram_max > hologram_min:
            hologram = (
                255
                * (hologram - hologram_min)
                / (hologram_max - hologram_min)
            )
        full_hologram = hologram.astype(np.uint8)
        if full_hologram.shape != (
            self.original_height,
            self.original_width,
        ):
            full_hologram = cv2.resize(
                full_hologram,
                (self.original_width, self.original_height),
                interpolation=cv2.INTER_NEAREST,
            )
        return full_hologram

    def _build_focus_hologram_batch(
        self,
        tm_rows,
        px=4,
        ds_method='mean',
        lut_cache=None,
        encode_chunk_size=16,
        progress_callback=None,
    ):
        """Vectorize focus-hologram encoding while bounding temporary memory."""
        tm_rows = np.asarray(tm_rows)
        expected_input_count = self.dmd_height * self.dmd_width
        if tm_rows.ndim != 2 or tm_rows.shape[1] != expected_input_count:
            raise ValueError(
                'TM batch shape {} does not match (batch, {})'.format(
                    tm_rows.shape, expected_input_count
                )
            )
        batch_count = int(tm_rows.shape[0])
        encode_chunk_size = max(1, int(encode_chunk_size))
        patterns = np.zeros(
            (
                batch_count,
                self.original_height,
                self.original_width,
            ),
            dtype=np.uint8,
        )
        errors = [None] * batch_count
        if batch_count == 0:
            return patterns, errors

        if lut_cache is None:
            lut_cache = generate_lut('sp', px)
        if not isinstance(lut_cache, tuple) or len(lut_cache) != 3:
            raise RuntimeError('Failed to generate the superpixel LUT')

        # The vectorized path is the exact equivalent of the report's normal
        # holo_SP(..., ds_method='mean') call. Preserve the scalar fallback for
        # less common down-sampling modes.
        _, pixel_combinations, lut = lut_cache
        combination_length = int(len(pixel_combinations[0]))
        n_sp = int(round(math.sqrt(combination_length)))
        can_vectorize = (
            ds_method == 'mean'
            and n_sp * n_sp == combination_length
            and int(px) == n_sp
            and self.pixel_group_size % n_sp == 0
            and self.dmd_height * self.pixel_group_size
            == self.original_height
            and self.dmd_width * self.pixel_group_size
            == self.original_width
        )
        gpu_enabled = bool(
            getattr(self, 'focus_encoding_use_gpu', True)
        ) and not bool(getattr(self, '_focus_gpu_failed', False))
        if can_vectorize and gpu_enabled and torch.cuda.is_available():
            gpu_device = int(
                getattr(self, 'focus_encoding_gpu_device', 0)
            )
            gpu_chunk_size = max(
                1,
                int(
                    getattr(
                        self,
                        'focus_encoding_gpu_chunk_size',
                        encode_chunk_size,
                    )
                ),
            )
            self.last_focus_encoding_backend = 'GPU cuda:{}'.format(
                gpu_device
            )
            self.last_focus_encoding_fallback_error = None
            try:
                return self._build_focus_hologram_batch_gpu(
                    tm_rows,
                    pixel_combinations=pixel_combinations,
                    lut=lut,
                    n_sp=n_sp,
                    encode_chunk_size=gpu_chunk_size,
                    progress_callback=progress_callback,
                    device_index=gpu_device,
                )
            except Exception as exc:
                self.last_focus_encoding_fallback_error = str(exc)
                # Avoid retrying every hardware batch in this scan. The flag
                # is reset when the next pixel-wise report starts.
                self._focus_gpu_failed = True
                print(
                    'GPU focus encoding failed; falling back to NumPy: '
                    '{}'.format(exc)
                )
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        self.last_focus_encoding_backend = 'CPU NumPy'
        if gpu_enabled and not torch.cuda.is_available():
            self.last_focus_encoding_fallback_error = (
                'CUDA is not available'
            )
        if not can_vectorize:
            for index, tm_row in enumerate(tm_rows):
                try:
                    patterns[index] = self._build_focus_hologram(
                        tm_row,
                        px=px,
                        ds_method=ds_method,
                        lut_cache=lut_cache,
                    )
                except Exception as exc:
                    errors[index] = str(exc)
                if progress_callback:
                    progress_callback(index + 1, batch_count)
            return patterns, errors

        pixel_combinations = np.asarray(
            pixel_combinations,
            dtype=np.uint8,
        )
        lut = np.asarray(lut)
        lut_zero = len(lut) // 2
        repeats_per_input = self.pixel_group_size // n_sp
        downsampled_height = self.dmd_height * repeats_per_input
        downsampled_width = self.dmd_width * repeats_per_input
        row_shifts = (
            n_sp * np.arange(downsampled_height, dtype=np.intp)
        ) % (n_sp**2)
        roll_indices = (
            np.arange(n_sp**2, dtype=np.intp)[None, :]
            + row_shifts[:, None]
        ) % (n_sp**2)
        roll_indices = roll_indices[None, :, None, :]

        for chunk_start in range(0, batch_count, encode_chunk_size):
            chunk_end = min(batch_count, chunk_start + encode_chunk_size)
            chunk_rows = np.asarray(
                tm_rows[chunk_start:chunk_end],
                dtype=np.complex64,
            )
            finite_mask = np.all(np.isfinite(chunk_rows), axis=1)
            amplitudes = np.max(np.abs(chunk_rows), axis=1)
            valid_mask = finite_mask & (amplitudes > 0)
            for local_index in np.flatnonzero(~valid_mask):
                global_index = chunk_start + int(local_index)
                errors[global_index] = (
                    'TM row contains NaN or infinity'
                    if not finite_mask[local_index]
                    else 'Cannot encode an all-zero complex field'
                )

            valid_local_indices = np.flatnonzero(valid_mask)
            if valid_local_indices.size:
                valid_rows = chunk_rows[valid_local_indices]
                fields = build_conjugate_focus_field(valid_rows).reshape(
                    -1,
                    self.dmd_height,
                    self.dmd_width,
                ).astype(np.complex128)
                field_max = np.max(np.abs(fields), axis=(1, 2))
                fields /= field_max[:, None, None]

                # np.kron(..., ones(pixel_group_size)) followed by 4x4 mean
                # down-sampling is equivalent to repeating every logical input
                # ``pixel_group_size / n_sp`` times. Repeat the 16 additions as
                # the reference function does to retain its rounding behavior.
                repeated = np.repeat(
                    np.repeat(fields, repeats_per_input, axis=1),
                    repeats_per_input,
                    axis=2,
                )
                downsampled = np.zeros_like(repeated)
                for _ in range(n_sp**2):
                    downsampled += repeated
                downsampled /= n_sp**2
                downsampled_max = np.max(
                    np.abs(downsampled),
                    axis=(1, 2),
                )
                scaled = downsampled / (
                    downsampled_max[:, None, None] * 0.01
                )
                real_index = (
                    np.rint(np.real(scaled)).astype(np.intp) + lut_zero
                )
                imag_index = (
                    np.rint(np.imag(scaled)).astype(np.intp) + lut_zero
                )
                selected = pixel_combinations[
                    lut[real_index, imag_index]
                ]
                rolled = np.take_along_axis(
                    selected,
                    roll_indices,
                    axis=3,
                )
                holograms = (
                    rolled.reshape(
                        len(valid_local_indices),
                        downsampled_height,
                        downsampled_width,
                        n_sp,
                        n_sp,
                    )
                    .transpose(0, 1, 4, 2, 3)
                    .reshape(
                        len(valid_local_indices),
                        self.original_height,
                        self.original_width,
                    )
                )
                # Match the legacy per-frame 0..255 normalization exactly.
                hologram_min = np.min(holograms, axis=(1, 2))
                hologram_max = np.max(holograms, axis=(1, 2))
                ranged = hologram_max > hologram_min
                holograms[ranged] *= np.uint8(255)
                global_indices = chunk_start + valid_local_indices
                patterns[global_indices] = holograms

            if progress_callback:
                progress_callback(chunk_end, batch_count)
        return patterns, errors

    def _build_focus_hologram_batch_gpu(
        self,
        tm_rows,
        pixel_combinations,
        lut,
        n_sp,
        encode_chunk_size=32,
        progress_callback=None,
        device_index=0,
    ):
        """CUDA implementation matching the complex128 NumPy encoder."""
        tm_rows = np.asarray(tm_rows)
        batch_count = int(tm_rows.shape[0])
        patterns = np.zeros(
            (
                batch_count,
                self.original_height,
                self.original_width,
            ),
            dtype=np.uint8,
        )
        errors = [None] * batch_count
        if batch_count == 0:
            return patterns, errors

        device_index = int(device_index)
        if not 0 <= device_index < torch.cuda.device_count():
            raise ValueError(
                'CUDA device {} is unavailable'.format(device_index)
            )
        device = torch.device('cuda:{}'.format(device_index))
        encode_chunk_size = max(1, int(encode_chunk_size))
        repeats_per_input = self.pixel_group_size // n_sp
        downsampled_height = self.dmd_height * repeats_per_input
        downsampled_width = self.dmd_width * repeats_per_input

        tensor_cache = getattr(self, '_focus_gpu_tensor_cache', None)
        if tensor_cache is None:
            tensor_cache = {}
            self._focus_gpu_tensor_cache = tensor_cache
        cache_key = (
            device_index,
            int(n_sp),
            downsampled_height,
            downsampled_width,
        )
        cached = tensor_cache.get(cache_key)
        if cached is None:
            combinations_tensor = torch.as_tensor(
                np.asarray(pixel_combinations, dtype=np.uint8),
                device=device,
                dtype=torch.uint8,
            )
            lut_tensor = torch.as_tensor(
                np.asarray(lut),
                device=device,
                dtype=torch.long,
            )
            row_shifts = (
                n_sp
                * torch.arange(
                    downsampled_height,
                    device=device,
                    dtype=torch.long,
                )
            ) % (n_sp**2)
            roll_indices = (
                torch.arange(
                    n_sp**2,
                    device=device,
                    dtype=torch.long,
                )[None, :]
                + row_shifts[:, None]
            ) % (n_sp**2)
            roll_indices = roll_indices[None, :, None, :]
            cached = (
                combinations_tensor,
                lut_tensor,
                roll_indices,
            )
            tensor_cache[cache_key] = cached
        combinations_tensor, lut_tensor, roll_indices = cached
        lut_zero = int(len(lut) // 2)

        with torch.no_grad():
            for chunk_start in range(0, batch_count, encode_chunk_size):
                chunk_end = min(
                    batch_count,
                    chunk_start + encode_chunk_size,
                )
                chunk_rows = np.asarray(
                    tm_rows[chunk_start:chunk_end],
                    dtype=np.complex64,
                )
                finite_mask = np.all(np.isfinite(chunk_rows), axis=1)
                amplitudes = np.max(np.abs(chunk_rows), axis=1)
                valid_mask = finite_mask & (amplitudes > 0)
                for local_index in np.flatnonzero(~valid_mask):
                    global_index = chunk_start + int(local_index)
                    errors[global_index] = (
                        'TM row contains NaN or infinity'
                        if not finite_mask[local_index]
                        else 'Cannot encode an all-zero complex field'
                    )

                valid_local_indices = np.flatnonzero(valid_mask)
                if valid_local_indices.size:
                    valid_rows = np.ascontiguousarray(
                        build_conjugate_focus_field(
                            chunk_rows[valid_local_indices]
                        )
                    )
                    rows_tensor = torch.as_tensor(
                        valid_rows,
                        device=device,
                    )
                    fields = rows_tensor.reshape(
                        -1,
                        self.dmd_height,
                        self.dmd_width,
                    ).to(torch.complex128)
                    field_max = torch.amax(
                        torch.abs(fields),
                        dim=(1, 2),
                    )
                    fields /= field_max[:, None, None]
                    repeated = torch.repeat_interleave(
                        fields,
                        repeats_per_input,
                        dim=1,
                    )
                    repeated = torch.repeat_interleave(
                        repeated,
                        repeats_per_input,
                        dim=2,
                    )
                    downsampled = torch.zeros_like(repeated)
                    for _ in range(n_sp**2):
                        downsampled += repeated
                    downsampled /= n_sp**2
                    downsampled_max = torch.amax(
                        torch.abs(downsampled),
                        dim=(1, 2),
                    )
                    scaled = downsampled / (
                        downsampled_max[:, None, None] * 0.01
                    )
                    real_index = (
                        torch.round(scaled.real).to(torch.long)
                        + lut_zero
                    )
                    imag_index = (
                        torch.round(scaled.imag).to(torch.long)
                        + lut_zero
                    )
                    selected = combinations_tensor[
                        lut_tensor[real_index, imag_index]
                    ]
                    expanded_roll_indices = roll_indices.expand(
                        len(valid_local_indices),
                        downsampled_height,
                        downsampled_width,
                        n_sp**2,
                    )
                    rolled = torch.gather(
                        selected,
                        3,
                        expanded_roll_indices,
                    )
                    holograms = (
                        rolled.reshape(
                            len(valid_local_indices),
                            downsampled_height,
                            downsampled_width,
                            n_sp,
                            n_sp,
                        )
                        .permute(0, 1, 4, 2, 3)
                        .reshape(
                            len(valid_local_indices),
                            self.original_height,
                            self.original_width,
                        )
                    )
                    hologram_min = torch.amin(
                        holograms,
                        dim=(1, 2),
                    )
                    hologram_max = torch.amax(
                        holograms,
                        dim=(1, 2),
                    )
                    ranged = hologram_max > hologram_min
                    holograms[ranged] *= 255
                    global_indices = chunk_start + valid_local_indices
                    patterns[global_indices] = holograms.cpu().numpy()

                if progress_callback:
                    progress_callback(chunk_end, batch_count)
        return patterns, errors

    def _analyze_pixelwise_focus_image(
        self,
        captured_image,
        target_x,
        target_y,
    ):
        """Calculate target-specific intensity, PBR, and peak displacement."""
        roi_w = getattr(self.camera, 'roi_width', 128)
        roi_h = getattr(self.camera, 'roi_height', 128)
        target_x = int(target_x)
        target_y = int(target_y)
        if not 0 <= target_x < roi_w or not 0 <= target_y < roi_h:
            raise ValueError(
                'Target ({}, {}) is outside ROI {}x{}'.format(
                    target_x, target_y, roi_w, roi_h
                )
            )
        captured_image = np.asarray(captured_image)
        if captured_image.shape != (roi_h, roi_w):
            raise RuntimeError(
                'Focus image shape {} does not match ROI {}'.format(
                    captured_image.shape, (roi_h, roi_w)
                )
            )

        target_intensity = float(captured_image[target_y, target_x])
        peak_flat_index = int(np.argmax(captured_image))
        peak_y, peak_x = np.unravel_index(
            peak_flat_index, captured_image.shape
        )
        peak_intensity = float(captured_image[peak_y, peak_x])
        peak_distance = float(
            math.hypot(int(peak_x) - target_x, int(peak_y) - target_y)
        )
        background_mask = np.ones(captured_image.shape, dtype=bool)
        background_mask[
            max(0, target_y - 2):min(roi_h, target_y + 3),
            max(0, target_x - 2):min(roi_w, target_x + 3),
        ] = False
        background_pixels = captured_image[background_mask]
        background_intensity = (
            float(np.mean(background_pixels))
            if background_pixels.size
            else float(np.mean(captured_image))
        )
        pbr = (
            target_intensity / background_intensity
            if background_intensity > 0
            else 0.0
        )
        return {
            'success': True,
            'focused_image': captured_image,
            'target_index': target_y * roi_w + target_x,
            'target_intensity': target_intensity,
            'peak_intensity': peak_intensity,
            'peak_position': (int(peak_x), int(peak_y)),
            'peak_distance_px': peak_distance,
            'mean_intensity': float(np.mean(captured_image)),
            'background_intensity': background_intensity,
            'pbr': float(pbr),
            'error': None,
        }

    def _project_and_capture_focus_batch(self, pattern_batch):
        """Load and acquire one focus sequence with a single hardware session."""
        pattern_batch = np.asarray(pattern_batch, dtype=np.uint8)
        if pattern_batch.ndim != 3:
            raise ValueError('Focus pattern batch must be a 3-D array')
        batch_count = int(pattern_batch.shape[0])
        if not 1 <= batch_count <= 1000:
            raise ValueError('Focus batch size must be between 1 and 1000')

        camera_started = False
        projection_started = False
        images = []
        frame_ids = []
        try:
            # Match the proven measurement ordering: arm the camera before
            # loading and starting the DMD sequence so the first trigger is kept.
            self.camera.start()
            camera_started = True
            if not self.load_pattern(pattern_batch):
                raise RuntimeError('Failed to load focus pattern batch')
            projection_result = self.DMD.juoptProjection(
                self.dev_id, 0, 0
            )
            if projection_result not in (None, 0):
                raise RuntimeError(
                    'Focus batch projection failed: {}'.format(
                        projection_result
                    )
                )
            projection_started = True

            for frame_index in range(batch_count):
                image, _, _ = self.camera.run()
                if image is None:
                    raise RuntimeError(
                        'Camera missed focus frame {}/{}'.format(
                            frame_index + 1, batch_count
                        )
                    )
                images.append(np.asarray(image))
                frame_id = getattr(self.camera, 'last_frame_id', None)
                if frame_id is not None:
                    frame_ids.append(int(frame_id))

            if len(frame_ids) >= 2:
                frame_steps = np.diff(
                    np.asarray(frame_ids, dtype=np.int64)
                )
                if np.any(frame_steps != 1):
                    raise RuntimeError(
                        'Camera frame gap detected in focus batch; '
                        'point-to-frame alignment is not trustworthy'
                    )
            return images
        finally:
            if projection_started:
                try:
                    self.DMD.juoptStop(self.dev_id)
                except Exception:
                    pass
            try:
                self.clear_sequence(0)
            except Exception:
                pass
            self.current_pattern = None
            if camera_started:
                try:
                    self.camera.stop()
                except Exception:
                    pass

    def _capture_focus_for_tm_row(
        self,
        tm_row,
        target_x,
        target_y,
        px=4,
        ds_method='mean',
        lut_cache=None,
    ):
        """Project one conjugated TM row and measure target-specific focusing."""
        result = {
            'success': False,
            'focused_image': None,
            'target_index': None,
            'target_intensity': 0.0,
            'peak_intensity': 0.0,
            'peak_position': None,
            'peak_distance_px': float('nan'),
            'mean_intensity': 0.0,
            'background_intensity': 0.0,
            'pbr': 0.0,
            'error': None,
        }
        camera_started = False
        projection_started = False
        try:
            roi_w = getattr(self.camera, 'roi_width', 128)
            roi_h = getattr(self.camera, 'roi_height', 128)
            target_x = int(target_x)
            target_y = int(target_y)
            if not 0 <= target_x < roi_w or not 0 <= target_y < roi_h:
                raise ValueError(
                    'Target ({}, {}) is outside ROI {}x{}'.format(
                        target_x, target_y, roi_w, roi_h
                    )
                )

            tm_row = np.asarray(tm_row, dtype=np.complex64)
            expected_input_count = self.dmd_height * self.dmd_width
            if tm_row.shape != (expected_input_count,):
                raise ValueError(
                    'TM row shape {} does not match ({},)'.format(
                        tm_row.shape, expected_input_count
                    )
                )
            if not np.all(np.isfinite(tm_row)):
                raise ValueError('TM row contains NaN or infinity')

            result['target_index'] = target_y * roi_w + target_x
            input_field = build_conjugate_focus_field(tm_row).reshape(
                self.dmd_height, self.dmd_width
            )
            full_pattern = np.kron(
                input_field,
                np.ones((self.pixel_group_size, self.pixel_group_size)),
            )
            if lut_cache is None:
                lut_cache = generate_lut('sp', px)
            _, px_comb, lut = lut_cache
            hologram = holo_SP(
                full_pattern,
                lut,
                px_comb,
                ds_method=ds_method,
            )
            hologram = hologram.astype(np.float32)
            if np.max(hologram) > np.min(hologram):
                hologram = (
                    255
                    * (hologram - np.min(hologram))
                    / (np.max(hologram) - np.min(hologram))
                )
            full_hologram = hologram.astype(np.uint8)
            if full_hologram.shape != (
                self.original_height,
                self.original_width,
            ):
                full_hologram = cv2.resize(
                    full_hologram,
                    (self.original_width, self.original_height),
                    interpolation=cv2.INTER_NEAREST,
                )
            if not self.load_pattern(
                np.asarray([full_hologram], dtype=np.uint8)
            ):
                raise RuntimeError('Failed to load pixel-wise focus hologram')

            self.camera.start()
            camera_started = True
            self.DMD.juoptProjection(self.dev_id, 0, 0)
            projection_started = True
            time.sleep(0.05)
            captured_image, _, _ = self.camera.run()
            if captured_image is None:
                raise RuntimeError('Camera did not return a focus image')
            captured_image = np.asarray(captured_image)
            if captured_image.shape != (roi_h, roi_w):
                raise RuntimeError(
                    'Focus image shape {} does not match ROI {}'.format(
                        captured_image.shape, (roi_h, roi_w)
                    )
                )

            target_intensity = float(captured_image[target_y, target_x])
            peak_flat_index = int(np.argmax(captured_image))
            peak_y, peak_x = np.unravel_index(
                peak_flat_index, captured_image.shape
            )
            peak_intensity = float(captured_image[peak_y, peak_x])
            peak_distance = float(
                math.hypot(int(peak_x) - target_x, int(peak_y) - target_y)
            )
            background_mask = np.ones(captured_image.shape, dtype=bool)
            background_mask[
                max(0, target_y - 2):min(roi_h, target_y + 3),
                max(0, target_x - 2):min(roi_w, target_x + 3),
            ] = False
            background_pixels = captured_image[background_mask]
            background_intensity = (
                float(np.mean(background_pixels))
                if background_pixels.size
                else float(np.mean(captured_image))
            )
            pbr = (
                target_intensity / background_intensity
                if background_intensity > 0
                else 0.0
            )
            result.update(
                {
                    'success': True,
                    'focused_image': captured_image,
                    'target_intensity': target_intensity,
                    'peak_intensity': peak_intensity,
                    'peak_position': (int(peak_x), int(peak_y)),
                    'peak_distance_px': peak_distance,
                    'mean_intensity': float(np.mean(captured_image)),
                    'background_intensity': background_intensity,
                    'pbr': float(pbr),
                }
            )
        except Exception as exc:
            result['error'] = str(exc)
        finally:
            if camera_started:
                try:
                    self.camera.stop()
                except Exception:
                    pass
            if projection_started:
                try:
                    self.DMD.juoptStop(self.dev_id)
                except Exception:
                    pass
            try:
                self.clear_sequence(0)
            except Exception:
                pass
        return result

    def _conjugate_focus_with_tm(
        self,
        H,
        target_x,
        target_y,
        px=4,
        ds_method='mean',
        lut_cache=None,
    ):
        """Focus one camera ROI point using an already loaded TM."""
        roi_w = getattr(self.camera, 'roi_width', 128)
        roi_h = getattr(self.camera, 'roi_height', 128)
        target_x = int(target_x)
        target_y = int(target_y)
        if not 0 <= target_x < roi_w or not 0 <= target_y < roi_h:
            return {
                'success': False,
                'error': 'Target is outside the camera ROI',
                'focused_image': None,
            }
        target_index = target_y * roi_w + target_x
        result = self._capture_focus_for_tm_row(
            H[target_index, :],
            target_x,
            target_y,
            px=px,
            ds_method=ds_method,
            lut_cache=lut_cache,
        )
        result['target_index'] = target_index
        return result

    def pixelwise_focus_average_pbr(
        self,
        stride=1,
        max_points=None,
        batch_size=1000,
        px=4,
        ds_method='mean',
        progress_callback=None,
        frame_callback=None,
        stop_requested=None,
        output_dir=None,
    ):
        """
        逐像素（按stride抽样）进行共轭聚焦，统计PBR并返回平均值。

        - stride: ROI 上的采样步长（1=全逐像素）
        - max_points: 可选，最多聚焦多少个点（用于快速估计）
        - batch_size: 每次连续加载和采集的全息图数（最大1000）
        - progress_callback: fn(done:int, total:int, message:str)
        - frame_callback: fn(image:ndarray, record:dict)，每次成功采集后调用
        """
        focus_mode = get_active_focus_config()["name"]
        H = self._load_transmission_matrix()
        self._focus_gpu_failed = False
        self.last_focus_encoding_fallback_error = None
        roi_w = getattr(self.camera, 'roi_width', 128)
        roi_h = getattr(self.camera, 'roi_height', 128)

        points = build_pixelwise_points(
            roi_w,
            roi_h,
            stride=max(1, int(stride)),
            max_points=max_points,
        )

        total = len(points)
        if total == 0:
            return {'success': False, 'avg_pbr': 0.0, 'count': 0, 'error': 'No points to focus'}

        batch_size = int(batch_size)
        if not 1 <= batch_size <= 1000:
            raise ValueError('batch_size must be between 1 and 1000')

        # 复用 LUT
        lut_cache = (generate_lut('sp', px))
        # normalize cache format to (f_val, px_comb, lut)
        if isinstance(lut_cache, tuple) and len(lut_cache) == 3:
            pass
        else:
            lut_cache = None

        records = []
        ok = 0
        last_img = None
        encoding_backends = set()
        stopped = False

        batch_total = int(math.ceil(total / float(batch_size)))
        for batch_index, batch_start in enumerate(
            range(0, total, batch_size),
            start=1,
        ):
            if stop_requested is not None and stop_requested():
                stopped = True
                break
            batch_points = points[batch_start:batch_start + batch_size]
            batch_count = len(batch_points)
            target_indices = np.asarray(
                [y * roi_w + x for x, y in batch_points],
                dtype=np.intp,
            )

            def encoding_progress(encoded, encoding_total):
                if progress_callback:
                    generated = batch_start + encoded
                    backend = getattr(
                        self,
                        'last_focus_encoding_backend',
                        'encoding',
                    )
                    progress_callback(
                        generated,
                        total,
                        'Encoding focus batch {}/{} [{}]: {}/{}'.format(
                            batch_index,
                            batch_total,
                            backend,
                            encoded,
                            encoding_total,
                        ),
                    )

            pattern_batch, pattern_errors = (
                self._build_focus_hologram_batch(
                    H[target_indices, :],
                    px=px,
                    ds_method=ds_method,
                    lut_cache=lut_cache,
                    encode_chunk_size=32,
                    progress_callback=encoding_progress,
                )
            )
            encoding_backends.add(
                getattr(
                    self,
                    'last_focus_encoding_backend',
                    'unknown',
                )
            )

            images = None
            batch_error = None
            if any(error is None for error in pattern_errors):
                try:
                    images = self._project_and_capture_focus_batch(
                        pattern_batch
                    )
                except Exception as exc:
                    batch_error = str(exc)
            else:
                batch_error = 'All focus holograms in this batch failed'
            del pattern_batch

            for offset, (x, y) in enumerate(batch_points):
                sample_index = batch_start + offset + 1
                error = pattern_errors[offset] or batch_error
                if error is None:
                    try:
                        res = self._analyze_pixelwise_focus_image(
                            images[offset], x, y
                        )
                    except Exception as exc:
                        res = {
                            'success': False,
                            'focused_image': None,
                            'error': str(exc),
                        }
                else:
                    res = {
                        'success': False,
                        'focused_image': None,
                        'error': error,
                    }

                peak_position = res.get('peak_position')
                peak_x = (
                    peak_position[0]
                    if peak_position is not None else None
                )
                peak_y = (
                    peak_position[1]
                    if peak_position is not None else None
                )
                succeeded = bool(res.get('success'))
                record = {
                    'sample_index': sample_index,
                    'x': x,
                    'y': y,
                    'target_index': y * roi_w + x,
                    'success': succeeded,
                    'target_intensity': (
                        float(res.get('target_intensity', np.nan))
                        if succeeded else np.nan
                    ),
                    'peak_intensity': (
                        float(res.get('peak_intensity', np.nan))
                        if succeeded else np.nan
                    ),
                    'peak_x': peak_x,
                    'peak_y': peak_y,
                    'peak_distance_px': (
                        float(res.get('peak_distance_px', np.nan))
                        if succeeded else np.nan
                    ),
                    'mean_intensity': (
                        float(res.get('mean_intensity', np.nan))
                        if succeeded else np.nan
                    ),
                    'background_intensity': (
                        float(res.get('background_intensity', np.nan))
                        if succeeded else np.nan
                    ),
                    'pbr': (
                        float(res.get('pbr', np.nan))
                        if succeeded else np.nan
                    ),
                    'error': res.get('error'),
                }
                records.append(record)
                if succeeded:
                    ok += 1
                    last_img = res.get('focused_image', last_img)
                    self.current_pbr = float(res.get('pbr', 0.0))
                    self.current_peak_intensity = float(
                        res.get('peak_intensity', 0.0)
                    )
                    if frame_callback and last_img is not None:
                        try:
                            frame_callback(last_img, record)
                        except Exception as exc:
                            print(
                                'Pixel-wise frame callback failed at '
                                '({}, {}): {}'.format(x, y, exc)
                            )

            if progress_callback:
                latest_pbr = records[-1]['pbr'] if records else np.nan
                pbr_text = (
                    '{:.2f}'.format(latest_pbr)
                    if np.isfinite(latest_pbr) else '-'
                )
                progress_callback(
                    batch_start + batch_count,
                    total,
                    'Focus batch {}/{} captured (ok={}, '
                    'target PBR={})'.format(
                        batch_index, batch_total, ok, pbr_text
                    ),
                )

        successful = [item for item in records if item['success']]
        ok = len(successful)
        avg_pbr = (
            float(np.mean([item['pbr'] for item in successful]))
            if successful else 0.0
        )
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'pixelwise_focus_results_v4',
            )
        report = save_pixelwise_focus_report(
            records,
            roi_shape=(roi_h, roi_w),
            output_dir=output_dir,
            run_label='{}_stride{}_batch{}_n{}'.format(
                focus_mode, stride, batch_size, total
            ),
            file_prefix='pixelwise_focus_v4',
        )
        return {
            'success': ok > 0,
            'avg_pbr': avg_pbr,
            'count': ok,
            'total': total,
            'batch_size': batch_size,
            'focus_mode': focus_mode,
            'encoding_backend': ' + '.join(sorted(encoding_backends)),
            'encoding_fallback_error': getattr(
                self,
                'last_focus_encoding_fallback_error',
                None,
            ),
            'last_image': last_img,
            'stopped': stopped,
            'records': records,
            'report': report,
            'error': (
                'Pixel-wise focusing stopped before completion'
                if stopped
                else None if ok > 0 else 'All focus attempts failed'
            )
        }

    def conjugate_focus_at_position_multi_phase(
        self,
        target_x,
        target_y,
        phase_offsets_rad,
        px=4,
        ds_method='mean',
        settle_time_s=0.1,
    ):
        """
        在同一目标位置，叠加多个全局相位偏置 phase_offsets_rad（单位：rad）并逐个投影/采集。

        返回:
            list[dict]: 每个相位对应的结果字典（包含 phase_offset_rad, success, pbr, peak_intensity, focused_image 等）
        """
        results = []

        # 复用单次聚焦中对 TM 的读取与输入场构造逻辑，但把投影/采集放在循环里
        tm_file = os.path.join(os.getcwd(), self.reconstructed_filename)
        if not os.path.exists(tm_file):
            return [{
                'success': False,
                'phase_offset_rad': float(phi),
                'error': f"传输矩阵文件不存在: {tm_file}",
                'focused_image': None,
                'target_index': None,
                'peak_intensity': 0.0,
                'mean_intensity': 0.0,
                'background_intensity': 0.0,
                'pbr': 0.0,
            } for phi in phase_offsets_rad]

        H = np.load(tm_file)

        roi_w = getattr(self.camera, 'roi_width', 128)
        roi_h = getattr(self.camera, 'roi_height', 128)
        if target_x >= roi_w or target_y >= roi_h:
            target_x = roi_w // 2
            target_y = roi_h // 2

        col_index = target_y * roi_w + target_x
        h_column = H[col_index, :]
        h_conjugate = build_conjugate_focus_field(h_column)

        N_x = self.dmd_width
        N_y = self.dmd_height
        base_input_field = h_conjugate.reshape(N_y, N_x)

        # LUT 只生成一次
        f_val, px_comb, lut = generate_lut('sp', px)

        # 相机一次开启，循环采集（更快）
        self.camera.start()
        try:
            for phi in phase_offsets_rad:
                result = {
                    'success': False,
                    'phase_offset_rad': float(phi),
                    'focused_image': None,
                    'target_index': int(col_index),
                    'peak_intensity': 0.0,
                    'mean_intensity': 0.0,
                    'background_intensity': 0.0,
                    'pbr': 0.0,
                    'error': None,
                }
                try:
                    # 叠加全局相位
                    input_field = base_input_field * np.exp(1j * float(phi))

                    # 上采样到 DMD 全分辨率（与原单次逻辑保持一致）
                    full_pattern = np.kron(input_field, np.ones((32, 32)))

                    hologram = holo_SP(full_pattern, lut, px_comb, ds_method=ds_method)

                    holo_norm = hologram.astype(np.float32)
                    if np.max(holo_norm) > np.min(holo_norm):
                        holo_norm = 255 * (holo_norm - np.min(holo_norm)) / (np.max(holo_norm) - np.min(holo_norm))
                    hologram_uint8 = holo_norm.astype(np.uint8)

                    full_hologram = hologram_uint8
                    if full_hologram.shape[0] != self.original_height or full_hologram.shape[1] != self.original_width:
                        full_hologram = cv2.resize(
                            full_hologram,
                            (self.original_width, self.original_height),
                            interpolation=cv2.INTER_NEAREST,
                        )

                    pattern_batch = np.array([full_hologram], dtype=np.uint8)
                    if not self.load_pattern(pattern_batch):
                        raise RuntimeError("加载全息图到DMD失败")

                    self.DMD.juoptProjection(self.dev_id, 0, 0)
                    time.sleep(settle_time_s)

                    captured_image = self.camera.run()
                    self.DMD.juoptStop(self.dev_id)
                    self.clear_sequence(0)
                    if captured_image is None:
                        raise RuntimeError("相机捕获失败")

                    if len(captured_image.shape) == 3:
                        captured_image = np.mean(captured_image, axis=2)

                    peak_intensity = float(np.max(captured_image))
                    mean_intensity = float((np.sum(captured_image) - peak_intensity) / (captured_image.size - 1))

                    edge_size = 10
                    if captured_image.shape[0] > 2 * edge_size and captured_image.shape[1] > 2 * edge_size:
                        background_pixels = np.concatenate([
                            captured_image[:edge_size, :].flatten(),
                            captured_image[-edge_size:, :].flatten(),
                            captured_image[:, :edge_size].flatten(),
                            captured_image[:, -edge_size:].flatten(),
                        ])
                        background_intensity = float(np.mean(background_pixels))
                    else:
                        background_intensity = float(mean_intensity)

                    pbr = float(peak_intensity / background_intensity) if background_intensity > 0 else 0.0

                    result['success'] = True
                    result['focused_image'] = captured_image
                    result['peak_intensity'] = peak_intensity
                    result['mean_intensity'] = mean_intensity
                    result['background_intensity'] = background_intensity
                    result['pbr'] = pbr

                except Exception as e:
                    result['error'] = str(e)

                results.append(result)
        finally:
            self.camera.stop()

        return results

    def cleanup(self):
        if self.is_init and self.DMD:
            self.stop_optimization()
            self.DMD.juoptStop(self.dev_id)
            self.DMD.juoptFree(self.dev_id)
            print("Device released")

class Application(tk.Tk):
    dmd_controller_class = DMDController

    def __init__(self):
        super().__init__()
        self.title("DMD Phase Focusing Optimization System")
        self.geometry("1440x810")

        # Keep native Polarized8 values. Measurement storage remains uint16 so
        # the existing reconstruction file layout does not change.
        self.camera = CameraHandler(cam_index=0, save_path="./camera_1")
        self.camera.convert_to_12bit = False

        # Initialize DMD controller
        self.dmd_controller = self.dmd_controller_class(self.camera)

        # Initialize measurement state
        self.measurement_running = False
        self._stability_last_seq = 0
        self._stability_corr_history = []
        self._stability_time_history = []
        self._closing = False
        self._visualization_after_id = None
        self._measurement_quality_window = None
        self._measurement_quality_figure = None
        self._measurement_quality_canvas = None

        # Setup UI
        self.setup_ui()

        # Try to initialize DMD device
        self.init_dmd_device()

        # Handle window close event
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Visualization update timer
        self._visualization_after_id = self.after(100, self.update_visualization)

    def setup_ui(self):
        # Create main container with left and right panels
        main_container = ttk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel for main controls and displays
        left_panel = ttk.Frame(main_container)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right panel for additional controls
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        # === LEFT PANEL CONTENT ===

        # Measurement progress bar frame
        measure_progress_frame = ttk.LabelFrame(left_panel, text="Measurement Progress")
        measure_progress_frame.pack(fill=tk.X, pady=(0, 10))

        self.measure_progress_bar = ttk.Progressbar(measure_progress_frame, mode='determinate')
        self.measure_progress_bar.pack(pady=5, fill=tk.X, padx=10)

        self.measure_progress_label = ttk.Label(measure_progress_frame, text="Ready")
        self.measure_progress_label.pack()

        # Reconstruction progress bar frame
        recon_progress_frame = ttk.LabelFrame(left_panel, text="Reconstruction Progress")
        recon_progress_frame.pack(fill=tk.X, pady=(0, 10))

        self.recon_progress_bar = ttk.Progressbar(recon_progress_frame, mode='determinate')
        self.recon_progress_bar.pack(pady=5, fill=tk.X, padx=10)

        self.recon_progress_label = ttk.Label(recon_progress_frame, text="Ready")
        self.recon_progress_label.pack()

        # Measurement control buttons frame
        measure_frame = ttk.Frame(left_panel)
        measure_frame.pack(fill=tk.X, pady=(0, 10))

        self.btn_measure_start = ttk.Button(measure_frame, text="Start Measurement",
                                          command=self.start_measurement, state=tk.DISABLED)
        self.btn_measure_start.pack(side=tk.LEFT, padx=(0, 5))

        self.btn_measure_stop = ttk.Button(measure_frame, text="Stop Measurement",
                                         command=self.stop_measurement, state=tk.DISABLED)
        self.btn_measure_stop.pack(side=tk.LEFT, padx=(5, 0))

        # Main image display frame
        image_frame = ttk.LabelFrame(left_panel, text="Camera Image")
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Create matplotlib figure for camera image
        self.fig_camera = plt.figure(figsize=(8, 4.5))
        self.ax_camera = self.fig_camera.add_subplot(1, 1, 1)
        self.ax_camera.set_title("Camera View")
        self.ax_camera.axis('off')
        self.camera_display = self.ax_camera.imshow(np.zeros((100, 100)), cmap='gray')

        # Embed camera figure in Tkinter
        self.camera_canvas = FigureCanvasTkAgg(self.fig_camera, master=image_frame)
        self.camera_canvas.draw()
        self.camera_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # === RIGHT PANEL CONTENT ===

        # Status frame (moved to right panel top)
        status_frame = ttk.LabelFrame(right_panel, text="System Status")
        status_frame.pack(fill=tk.X, pady=(0, 10))

        self.lbl_dmd_status = ttk.Label(status_frame, text="DMD Status: Not Initialized")
        self.lbl_dmd_status.pack(anchor=tk.W)

        self.lbl_camera_status = ttk.Label(status_frame, text="Camera Status: Connected")
        self.lbl_camera_status.pack(anchor=tk.W)

        self.lbl_pbr = ttk.Label(status_frame, text="Current PBR: -")
        self.lbl_pbr.pack(anchor=tk.W)

        self.lbl_peak_intensity = ttk.Label(status_frame, text="Peak Intensity: -")
        self.lbl_peak_intensity.pack(anchor=tk.W)

        self.lbl_background_intensity = ttk.Label(status_frame, text="Background Intensity: -")
        self.lbl_background_intensity.pack(anchor=tk.W)

        self.lbl_correlation = ttk.Label(status_frame, text="Stability Correlation: -")
        self.lbl_correlation.pack(anchor=tk.W)

        # Stability correlation trend (line chart)
        stability_frame = ttk.LabelFrame(right_panel, text="Stability Trend")
        stability_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 10))

        self.fig_stability = plt.figure(figsize=(3.6, 2.2))
        self.ax_stability = self.fig_stability.add_subplot(1, 1, 1)
        self.ax_stability.set_title("Corr vs Batch")
        self.ax_stability.set_ylim(-0.05, 1.05)
        self.ax_stability.grid(True, alpha=0.3)
        (self.stability_line,) = self.ax_stability.plot([], [], lw=1.5)
        self.ax_stability.set_xlabel("Batch #")
        self.ax_stability.set_ylabel("Corr")

        self.stability_canvas = FigureCanvasTkAgg(self.fig_stability, master=stability_frame)
        self.stability_canvas.draw()
        self.stability_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # DMD control frame
        dmd_frame = ttk.LabelFrame(right_panel, text="DMD Control")
        dmd_frame.pack(fill=tk.X, pady=(0, 10))

        self.btn_init = ttk.Button(dmd_frame, text="Initialize DMD", command=self.init_dmd_device)
        self.btn_init.pack(fill=tk.X, pady=2)

        # Original optimization buttons removed - functionality moved to Measurement buttons

        # Transmission matrix frame
        tm_frame = ttk.LabelFrame(right_panel, text="Transmission Matrix")
        tm_frame.pack(fill=tk.X, pady=(0, 10))

        self.btn_tm_recovery = ttk.Button(
            tm_frame,
            text="Recover Transmission Matrix (Local)",
            command=self.recover_transmission_matrix,
        )
        self.btn_tm_recovery.pack(fill=tk.X, pady=2)

        # Remote TM via server (upload measurements_memmap -> run GGS2_1 -> download TM)
        self.btn_tm_remote = ttk.Button(
            tm_frame,
            text="Recover Transmission Matrix (Remote)",
            command=self.remote_tm_from_server,
        )
        self.btn_tm_remote.pack(fill=tk.X, pady=2)

        # Focus control frame
        focus_frame = ttk.LabelFrame(right_panel, text="Focus Control")
        focus_frame.pack(fill=tk.X, pady=(0, 10))

        self.btn_focus = ttk.Button(focus_frame, text="Focus", command=self.focus_position)
        self.btn_focus.pack(fill=tk.X, pady=2)

        # Pixel-wise focus + avg PBR
        self.btn_pixelwise_focus = ttk.Button(
            focus_frame,
            text="Full 128x128 Pixel-wise + PBR Report",
            command=self.pixelwise_focus_avg_pbr
        )
        self.btn_pixelwise_focus.pack(fill=tk.X, pady=2)

        # Focus position selection
        pos_frame = ttk.Frame(focus_frame)
        pos_frame.pack(fill=tk.X, pady=5)

        ttk.Label(pos_frame, text="X:").pack(side=tk.LEFT)
        self.focus_x = tk.Spinbox(pos_frame, from_=0, to=1024, width=5)
        self.focus_x.pack(side=tk.LEFT, padx=2)
        self.focus_x.delete(0, tk.END)
        self.focus_x.insert(0, 64)

        ttk.Label(pos_frame, text="Y:").pack(side=tk.LEFT, padx=(5, 0))
        self.focus_y = tk.Spinbox(pos_frame, from_=0, to=768, width=5)
        self.focus_y.pack(side=tk.LEFT, padx=2)
        self.focus_y.delete(0, tk.END)
        self.focus_y.insert(0, 64)

        # Pixel-wise focus parameters
        pw_frame = ttk.Frame(focus_frame)
        pw_frame.pack(fill=tk.X, pady=5)

        ttk.Label(pw_frame, text="Stride:").pack(side=tk.LEFT)
        self.pw_stride = tk.Spinbox(pw_frame, from_=1, to=128, width=5)
        self.pw_stride.pack(side=tk.LEFT, padx=2)
        self.pw_stride.delete(0, tk.END)
        self.pw_stride.insert(0, 1)

        ttk.Label(pw_frame, text="Max:").pack(side=tk.LEFT, padx=(5, 0))
        self.pw_max_points = tk.Spinbox(pw_frame, from_=0, to=99999, width=7)
        self.pw_max_points.pack(side=tk.LEFT, padx=2)
        self.pw_max_points.delete(0, tk.END)
        self.pw_max_points.insert(0, 0)  # 0 means no limit

        pw_batch_frame = ttk.Frame(focus_frame)
        pw_batch_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(pw_batch_frame, text="Batch size:").pack(side=tk.LEFT)
        self.pw_batch_size = tk.Spinbox(
            pw_batch_frame,
            from_=1,
            to=1000,
            width=7,
        )
        self.pw_batch_size.pack(side=tk.LEFT, padx=2)
        self.pw_batch_size.delete(0, tk.END)
        self.pw_batch_size.insert(0, 1000)
        ttk.Label(pw_batch_frame, text="(max 1000)").pack(side=tk.LEFT)

        # Multi-phase focus controls
        phase_frame = ttk.Frame(focus_frame)
        phase_frame.pack(fill=tk.X, pady=(6, 0))

        ttk.Label(phase_frame, text="Phase list:").pack(anchor=tk.W)
        self.phase_list_entry = ttk.Entry(phase_frame)
        self.phase_list_entry.pack(fill=tk.X, pady=(2, 2))
        self.phase_list_entry.insert(0, "0, 1.57079632679, 3.14159265359")

        phase_unit_frame = ttk.Frame(phase_frame)
        phase_unit_frame.pack(fill=tk.X, pady=(0, 4))
        self.phase_unit = tk.StringVar(value="rad")
        ttk.Radiobutton(phase_unit_frame, text="rad", variable=self.phase_unit, value="rad").pack(side=tk.LEFT)
        ttk.Radiobutton(phase_unit_frame, text="deg", variable=self.phase_unit, value="deg").pack(side=tk.LEFT, padx=(8, 0))

        self.btn_focus_multi_phase = ttk.Button(
            focus_frame,
            text="Multi-Phase Focus",
            command=self.focus_position_multi_phase,
        )
        self.btn_focus_multi_phase.pack(fill=tk.X, pady=(2, 2))

        # Log frame
        log_frame = ttk.LabelFrame(right_panel, text="System Log")
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(log_frame, height=15, width=30, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Store original visualization components for compatibility
        self.fig = self.fig_camera  # Use camera figure as main figure
        self.canvas = self.camera_canvas  # Use camera canvas as main canvas
        self.img_display = self.camera_display  # Use camera display as main display

        # PBR and correlation data for original functionality
        self.pbr_history = []
        self.time_serie = []

    def _parse_phase_list(self, s):
        """
        支持输入:
        - 逗号/空格/换行分隔: "0, 1.57, 3.14"
        - 简单 pi 表达式: "0, pi/2, -pi"
        返回: list[float]（单位: rad）
        """
        if s is None:
            return []
        raw = s.strip()
        if not raw:
            return []

        tokens = re.split(r"[,\s]+", raw)
        phases = []
        for t in tokens:
            if not t:
                continue
            tt = t.strip().lower().replace("π", "pi")
            # 允许 "pi", "pi/2", "3*pi/2", "-pi/4"
            if "pi" in tt:
                # 安全处理：只允许数字、pi、*、/、+、-、.、e
                if not re.fullmatch(r"[0-9e\.\+\-\*/pi]+", tt):
                    raise ValueError(f"Invalid token: {t}")
                val = eval(tt, {"__builtins__": {}}, {"pi": np.pi})  # noqa: S307
            else:
                val = float(tt)
            phases.append(float(val))

        # 单位换算
        if self.phase_unit.get() == "deg":
            phases = [p * np.pi / 180.0 for p in phases]

        return phases

    def update_visualization(self):
        # This callback is no longer pending once Tk starts executing it.
        self._visualization_after_id = None
        if self._closing:
            return

        # Update camera image
        try:
            if not self.camera.image_queue.empty():
                idx, img_data = self.camera.image_queue.get_nowait()
                if len(img_data.shape) == 3:
                    img_data = np.mean(img_data, axis=2)
                self.camera_display.set_data(img_data)
                self.camera_display.set_clim(vmin=img_data.min(), vmax=img_data.max())
                self.camera_canvas.draw()
        except queue.Empty:
            pass

        # Update status labels
        self.lbl_pbr.config(text=f"Current PBR: {self.dmd_controller.current_pbr:.2f}")
        self.lbl_peak_intensity.config(text=f"Peak Intensity: {self.dmd_controller.current_peak_intensity:.1f}")
        corr = self.dmd_controller.current_stability_corr
        corr_txt = "-" if corr is None else f"{corr:.4f}"
        self.lbl_correlation.config(text=f"Stability Correlation: {corr_txt}")

        # Update stability trend once per new batch (seq increments in DMDController)
        try:
            seq = getattr(self.dmd_controller, "_stability_seq", 0)
            if seq != self._stability_last_seq:
                self._stability_last_seq = seq
                if corr is not None:
                    self._stability_corr_history.append(float(corr))
                    self._stability_time_history.append(time.time())
                    # Keep last N points
                    max_points = 200
                    if len(self._stability_corr_history) > max_points:
                        self._stability_corr_history = self._stability_corr_history[-max_points:]
                        self._stability_time_history = self._stability_time_history[-max_points:]

                    x = list(range(1, len(self._stability_corr_history) + 1))
                    self.stability_line.set_data(x, self._stability_corr_history)
                    self.ax_stability.set_xlim(max(1, len(x) - 50), max(50, len(x)))
                    self.stability_canvas.draw()
        except Exception:
            # Avoid UI crash if matplotlib objects not ready
            pass

        if not self._closing:
            self._visualization_after_id = self.after(100, self.update_visualization)

    def init_dmd_device(self):
        def _init():
            self.log("Initializing DMD device...")
            devices = self.dmd_controller.get_devices()
            if devices:
                if self.dmd_controller.initialize_device(devices[0]):
                    self.log("DMD initialized successfully")
                    self.lbl_dmd_status.config(text="DMD Status: Initialized")
                    # Enable measurement button after DMD initialization
                    self.btn_measure_start.config(state=tk.NORMAL)
                else:
                    self.log("DMD initialization failed")
            else:
                self.log("No DMD devices found")

        threading.Thread(target=_init, daemon=True).start()

    def _close_measurement_quality_window(self):
        """Close the previous non-modal quality window and release its figure."""
        window = self._measurement_quality_window
        self._measurement_quality_window = None
        self._measurement_quality_canvas = None
        if window is not None:
            try:
                if window.winfo_exists():
                    window.destroy()
            except tk.TclError:
                pass

        figure = self._measurement_quality_figure
        self._measurement_quality_figure = None
        if figure is not None:
            try:
                figure.clear()
            except Exception:
                pass

    def _show_measurement_quality_result(self, result, outputs):
        """Display average intensity and the raw-count histogram on Tk's thread."""
        if self._closing:
            return

        self._close_measurement_quality_window()
        window = tk.Toplevel(self)
        window.title("Measurement Quality - Average Intensity and Histogram")
        window.geometry("1240x680")
        window.minsize(900, 540)
        window.transient(self)
        window.protocol("WM_DELETE_WINDOW", self._close_measurement_quality_window)

        figure = build_measurement_quality_figure(result)
        canvas = FigureCanvasTkAgg(figure, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))

        footer = ttk.Frame(window)
        footer.pack(fill=tk.X, padx=10, pady=(0, 8))
        ttk.Label(
            footer,
            text="Saved: {}".format(outputs["figure"]),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(
            footer,
            text="Close",
            command=self._close_measurement_quality_window,
        ).pack(side=tk.RIGHT)

        self._measurement_quality_window = window
        self._measurement_quality_figure = figure
        self._measurement_quality_canvas = canvas
        window.lift()

    def start_measurement(self):
        """Start measurement process with optimization"""
        self.btn_measure_start.config(state=tk.DISABLED)
        self.btn_measure_stop.config(state=tk.NORMAL)
        self.log("Starting measurement/optimization process...")

        # Reset progress bars
        self.measure_progress_bar.config(value=0)
        self.recon_progress_bar.config(value=0)
        self.measure_progress_label.config(text="Starting...")
        self.recon_progress_label.config(text="Waiting...")

        def measure_progress_callback(progress, message):
            """Callback function to update measurement progress bar and label"""
            self.after(0, lambda: self.measure_progress_bar.config(value=progress))
            self.after(0, lambda: self.measure_progress_label.config(text=message))

        def recon_progress_callback(progress, message):
            """Callback function to update reconstruction progress bar and label"""
            self.after(0, lambda: self.recon_progress_bar.config(value=progress))
            self.after(0, lambda: self.recon_progress_label.config(text=message))

        def _measure():
            try:
                # Set progress callbacks
                self.dmd_controller.measure_progress_callback = measure_progress_callback
                # self.dmd_controller.recon_progress_callback = recon_progress_callback

                # Start the DMD optimization process
                self.dmd_controller.start_measurement()

                # Wait for optimization to complete or be stopped
                while self.dmd_controller.optimization_running:
                    time.sleep(0.5)

                measurement_error = self.dmd_controller.measurement_error
                if measurement_error:
                    raise RuntimeError(measurement_error)

                # stop_measurement() sets this false; do not analyze a partial run.
                if not self.measurement_running:
                    return

                self.after(
                    0,
                    lambda: self.measure_progress_label.config(
                        text="Analyzing measurement quality..."
                    ),
                )
                measurement_path = os.path.join(
                    os.getcwd(),
                    self.dmd_controller.measurement_filename,
                )
                sensor_max_count = (
                    4095 if self.camera.convert_to_12bit else 255
                )
                quality_result = analyze_measurement_memmap(
                    measurement_path,
                    height=self.camera.roi_height,
                    width=self.camera.roi_width,
                    sensor_max_count=sensor_max_count,
                )
                quality_outputs = save_measurement_quality_outputs(
                    quality_result
                )
                quality_stats = quality_result["statistics"]

                def _show_completed_measurement():
                    if self._closing:
                        return
                    self.measure_progress_bar.config(value=100)
                    self.measure_progress_label.config(
                        text="Measurement completed - quality plot ready"
                    )
                    self.log(
                        "Measurement completed: mean={:.3f} counts, "
                        "zero={:.2%}, max={}".format(
                            quality_stats["global_mean_counts"],
                            quality_stats["zero_fraction"],
                            quality_stats["raw_max_counts"],
                        )
                    )
                    if quality_result["quality_flags"]:
                        self.log(
                            "Measurement quality warning: "
                            + "; ".join(quality_result["quality_flags"])
                        )
                    self._show_measurement_quality_result(
                        quality_result, quality_outputs
                    )

                self.after(0, _show_completed_measurement)

            except Exception as e:
                error_text = str(e)

                def _show_measurement_error():
                    if self._closing:
                        return
                    self.log(f"Measurement error: {error_text}")
                    self.measure_progress_label.config(text="Measurement failed")

                try:
                    self.after(0, _show_measurement_error)
                except tk.TclError:
                    pass
            finally:
                self.measurement_running = False
                self.dmd_controller.measure_progress_callback = None  # Clear callbacks
                self.dmd_controller.recon_progress_callback = None

                def _finish_measurement_ui():
                    if self._closing:
                        return
                    self.btn_measure_start.config(state=tk.NORMAL)
                    self.btn_measure_stop.config(state=tk.DISABLED)

                try:
                    self.after(0, _finish_measurement_ui)
                except tk.TclError:
                    pass

        self.measurement_running = True
        threading.Thread(target=_measure, daemon=True).start()

    def stop_measurement(self):
        """Stop measurement/optimization process"""
        self.measurement_running = False
        self.dmd_controller.stop_optimization()
        self.btn_measure_start.config(state=tk.NORMAL)
        self.btn_measure_stop.config(state=tk.DISABLED)
        self.measure_progress_label.config(text="Measurement stopped")
        self.recon_progress_label.config(text="Reconstruction stopped")
        self.log("Measurement/optimization process stopped")

    def recover_transmission_matrix(self):
        """Recover transmission matrix"""
        self.log("Starting transmission matrix recovery...")
        self.btn_tm_recovery.config(state=tk.DISABLED)

        # Reset reconstruction progress bar
        self.recon_progress_bar.config(value=0)
        self.recon_progress_label.config(text="Starting reconstruction...")

        def recon_progress_callback(progress, message):
            """Callback function to update reconstruction progress bar and label"""
            self.after(0, lambda: self.recon_progress_bar.config(value=progress))
            self.after(0, lambda: self.recon_progress_label.config(text=message))

        def _recover_tm():
            try:
                # Set progress callbacks
                self.dmd_controller.recon_progress_callback = recon_progress_callback

                # Start reconstruction
                self.dmd_controller.start_reconstruction()

                # Wait for reconstruction to complete
                # We need a short sleep loop to check if it started and then if it finished
                time.sleep(1) # Wait for thread to start
                while self.dmd_controller.reconstruction_running:
                    time.sleep(0.5)

                reconstruction_error = getattr(
                    self.dmd_controller,
                    "reconstruction_error",
                    None,
                )
                if reconstruction_error:
                    raise RuntimeError(reconstruction_error)

                self.recon_progress_bar.config(value=100)
                self.recon_progress_label.config(text="Reconstruction completed")
                self.log("Transmission matrix recovery completed")

            except Exception as e:
                self.log(f"TM recovery error: {str(e)}")
                self.recon_progress_label.config(text="TM recovery failed")
            finally:
                self.btn_tm_recovery.config(state=tk.NORMAL)
                self.dmd_controller.recon_progress_callback = None

        threading.Thread(target=_recover_tm, daemon=True).start()

    def remote_tm_from_server(self):
        """
        使用服务器上的 GGS2_1 算法恢复传输矩阵:
        1) 上传本地 measurements_memmap 到服务器
        2) 在服务器运行脚本计算 TM
        3) 下载 TM 到本地 (默认保存为 reconstructed_field.npy 以兼容后续聚焦函数)
        """
        print("DEBUG: remote_tm_from_server called")
        try:
            messagebox.showinfo("DEBUG", "Remote TM function triggered!")
        except Exception as e:
            print(f"DEBUG: Could not show info box: {e}")

        if paramiko is None:
            msg = "paramiko 未安装，请先在本机执行: pip install paramiko"
            print(msg)
            self.log(msg)
            messagebox.showwarning("Missing Dependency", msg)
            return

        # 本地/远端路径和服务器配置 —— 请根据你的实际服务器情况修改
        local_meas_file = os.path.join(
            os.getcwd(),
            self.dmd_controller.measurement_filename,
        )
        local_tm_file = os.path.join(
            os.getcwd(),
            self.dmd_controller.reconstructed_filename,
        )

        ssh_host = "10.102.137.157"
        ssh_port = 22
        ssh_user = "limingfei"
        ssh_password = None # "your_password"  # 如果使用密码登录，请取消注释并填入密码

        # 尝试加载默认 SSH 密钥
        my_pkey = None
        key_path = os.path.expanduser("~/.ssh/id_rsa")
        if os.path.exists(key_path):
             try:
                 my_pkey = paramiko.RSAKey.from_private_key_file(key_path)
             except Exception as e:
                 print(f"DEBUG: Failed to load default SSH key: {e}")

        remote_work_dir = "/home/limingfei/speckle/remoteCompute/"
        remote_meas_file = os.path.join(remote_work_dir, "measurements_memmap.npy")
        remote_tm_file = os.path.join(remote_work_dir, "reconstructed_field.npy")

        # 根据你在服务器上实际运行 GGS2_1 的命令修改这一行
        remote_command = (
            f"cd {remote_work_dir} && "
            f"/home/limingfei/miniconda3/envs/speckle/bin/python /home/limingfei/speckle/remoteCompute/ggs_tm.py"
        )

        if not os.path.exists(local_meas_file):
            self.log(f"本地 measurements_memmap 文件不存在: {local_meas_file}")
            return

        self.log("开始远程传输矩阵恢复 (上传 -> 远端计算 -> 下载)...")
        self.btn_tm_remote.config(state=tk.DISABLED)

        # 进度条简单指示为不定模式
        self.measure_progress_bar.config(mode="indeterminate")
        self.measure_progress_bar.start()
        self.measure_progress_label.config(text="Remote TM: connecting to server...")

        def _run_remote_tm():
            try:
                # 1) 建立 SSH 连接
                self.log(f"连接服务器 {ssh_host}:{ssh_port} ...")
                print(f"DEBUG: Connecting to {ssh_host}...")
                client = paramiko.SSHClient()
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

                # 构建连接参数
                connect_kwargs = {
                    "hostname": ssh_host,
                    "port": ssh_port,
                    "username": ssh_user,
                    "timeout": 10,
                }
                if my_pkey:
                    connect_kwargs["pkey"] = my_pkey
                elif ssh_password:
                    connect_kwargs["password"] = ssh_password

                client.connect(**connect_kwargs)

                # 2) 确保远端工作目录存在 & 上传文件
                sftp = client.open_sftp()
                try:
                    try:
                        sftp.listdir(remote_work_dir)
                    except IOError:
                        # 递归创建目录
                        parts = remote_work_dir.strip("/").split("/")
                        cur = ""
                        for p in parts:
                            cur = cur + "/" + p
                            try:
                                sftp.listdir(cur)
                            except IOError:
                                sftp.mkdir(cur)

                    self.log(f"上传测量数据到服务器: {remote_meas_file}")
                    sftp.put(local_meas_file, remote_meas_file)
                finally:
                    sftp.close()

                # 3) 运行远端 GGS2_1 脚本
                self.measure_progress_label.config(text="Remote TM: running GGS2_1 on server...")
                self.log(f"在服务器上运行命令: {remote_command}")
                print(f"DEBUG: Executing remote command: {remote_command}")
                stdin, stdout, stderr = client.exec_command(remote_command)

                # 实时打印输出到命令行
                self.log("正在执行远程计算，实时进度请查看命令行窗口...")
                while True:
                    line = stdout.readline()
                    if not line:
                        break
                    # 解码并打印
                    line_str = line if isinstance(line, str) else line.decode("utf-8", errors="ignore")
                    print(line_str, end="")

                # 等待命令完全结束
                exit_status = stdout.channel.recv_exit_status()
                err = stderr.read().decode("utf-8", errors="ignore")

                if err.strip():
                    print("Remote STDERR:", err)
                    self.log("远端 STDERR:")
                    for line in err.splitlines():
                        self.log("  " + line)

                if exit_status != 0:
                    raise RuntimeError(f"远端脚本执行失败，退出码: {exit_status}")

                # 4) 下载 TM 文件
                self.measure_progress_label.config(text="Remote TM: downloading result...")
                sftp = client.open_sftp()
                try:
                    self.log(f"从服务器下载 TM: {remote_tm_file}")
                    sftp.get(remote_tm_file, local_tm_file)
                finally:
                    sftp.close()

                self.log(f"远程传输矩阵恢复完成，本地文件: {local_tm_file}")
                self.measure_progress_label.config(text="Remote TM: completed")

            except Exception as e:
                err_msg = f"Remote TM error: {str(e)}"
                print(f"DEBUG: {err_msg}")
                self.log(err_msg)
                self.measure_progress_label.config(text="Remote TM: failed")
                # 如果是认证失败，给出提示
                if "Authentication failed" in str(e) or "NameError" in str(e):
                    self.log("请检查 SSH 配置 (host, user, password/key)")
                    messagebox.showerror("Connection Error", f"SSH Connection Failed:\n{str(e)}")
            finally:
                try:
                    client.close()
                except Exception:
                    pass
                self.btn_tm_remote.config(state=tk.NORMAL)
                self.measure_progress_bar.stop()
                self.measure_progress_bar.config(mode="determinate")

        threading.Thread(target=_run_remote_tm, daemon=True).start()

    def focus_position(self):
        """Focus on selected position using transmission matrix conjugate focusing"""
        try:
            x = int(self.focus_x.get())
            y = int(self.focus_y.get())

            self.log(f"Focusing on position ({x}, {y})")
            self.btn_focus.config(state=tk.DISABLED)

            def _focus():
                try:
                    # Use DMD controller's conjugate focus function
                    result = self.dmd_controller.conjugate_focus_at_position(x, y)

                    if result['success']:
                        # Update progress bar
                        self.measure_progress_bar.config(mode='determinate', value=100)
                        self.measure_progress_label.config(text=f"Focused on ({x}, {y})")

                        # Update status labels with focus results
                        self.lbl_peak_intensity.config(text=f"Peak Intensity: {result['peak_intensity']:.2f}")
                        self.lbl_background_intensity.config(text=f"Background Intensity: {result['background_intensity']:.2f}")
                        self.lbl_pbr.config(text=f"Current PBR: {result['pbr']:.2f}")

                        # Update camera display with focused image
                        focused_image = result['focused_image']
                        if focused_image is not None:
                            self.camera_display.set_data(focused_image)
                            self.camera_display.set_clim(vmin=focused_image.min(), vmax=focused_image.max())
                            self.camera_canvas.draw()

                        # Log detailed results
                        self.log(f"Focus completed on position ({x}, {y})")
                        self.log(f"Target index: {result['target_index']}")
                        self.log(f"Peak intensity: {result['peak_intensity']:.2f}")
                        self.log(f"Mean intensity: {result['mean_intensity']:.2f}")
                        self.log(f"Background intensity: {result['background_intensity']:.2f}")
                        self.log(f"PBR: {result['pbr']:.2f}")
                    else:
                        self.measure_progress_bar.config(mode='determinate', value=0)
                        self.measure_progress_label.config(text="Focus failed")
                        self.log(f"Focus failed: {result['error']}")

                except Exception as e:
                    self.log(f"Focus error: {str(e)}")
                    self.measure_progress_label.config(text="Focus failed")
                finally:
                    self.btn_focus.config(state=tk.NORMAL)

            threading.Thread(target=_focus, daemon=True).start()

        except ValueError:
            self.log("Invalid focus position coordinates")

    def pixelwise_focus_avg_pbr(self):
        """逐像素（抽样）聚焦，计算PBR平均值"""
        self.btn_pixelwise_focus.config(state=tk.DISABLED)
        self.measure_progress_bar.config(value=0)
        self.measure_progress_label.config(text="Pixel-wise focusing...")

        try:
            stride = int(self.pw_stride.get())
        except Exception:
            stride = 1

        try:
            max_points_raw = int(self.pw_max_points.get())
            max_points = None if max_points_raw <= 0 else max_points_raw
        except Exception:
            max_points = None

        try:
            batch_size = max(1, min(1000, int(self.pw_batch_size.get())))
        except Exception:
            batch_size = 1000

        # Acquisition can produce frames faster than Matplotlib can draw them.
        # Retain only the latest frame instead of filling Tk's event queue.
        frame_lock = threading.Lock()
        frame_state = {'latest': None, 'update_pending': False}

        def _render_latest_frame():
            with frame_lock:
                payload = frame_state['latest']
                frame_state['latest'] = None

            if payload is not None:
                image, record = payload
                vmin = float(np.min(image))
                vmax = float(np.max(image))
                if vmax <= vmin:
                    vmax = vmin + 1.0
                self.camera_display.set_data(image)
                self.camera_display.set_clim(vmin=vmin, vmax=vmax)
                self.ax_camera.set_title(
                    'Camera View - target ({}, {})'.format(
                        record['x'], record['y']
                    )
                )
                self.lbl_pbr.config(
                    text='Target PBR: {:.2f}'.format(record['pbr'])
                )
                self.lbl_peak_intensity.config(
                    text='Peak Intensity: {:.2f}'.format(
                        record['peak_intensity']
                    )
                )
                self.lbl_background_intensity.config(
                    text='Background Intensity: {:.2f}'.format(
                        record['background_intensity']
                    )
                )
                self.camera_canvas.draw_idle()

            schedule_again = False
            with frame_lock:
                if frame_state['latest'] is not None:
                    schedule_again = True
                else:
                    frame_state['update_pending'] = False
            if schedule_again:
                self.after(0, _render_latest_frame)

        def _frame(image, record):
            # Keep an owned copy in case the camera SDK reuses its buffer.
            payload = (np.array(image, copy=True), dict(record))
            schedule_update = False
            with frame_lock:
                frame_state['latest'] = payload
                if not frame_state['update_pending']:
                    frame_state['update_pending'] = True
                    schedule_update = True
            if schedule_update:
                self.after(0, _render_latest_frame)

        def _progress(done, total, message):
            if total > 0:
                pct = 100.0 * done / total
            else:
                pct = 0.0
            self.after(0, lambda: self.measure_progress_bar.config(value=pct))
            self.after(0, lambda: self.measure_progress_label.config(text=message))

        def _run():
            try:
                self.log(
                    'Pixel-wise focus started '
                    '(stride={}, max_points={}, batch_size={})'.format(
                        stride,
                        max_points or 'None',
                        batch_size,
                    )
                )
                res = self.dmd_controller.pixelwise_focus_average_pbr(
                    stride=stride,
                    max_points=max_points,
                    batch_size=batch_size,
                    progress_callback=_progress,
                    frame_callback=_frame,
                )
                if res.get('success'):
                    avg_pbr = float(res.get('avg_pbr', 0.0))
                    count = res.get('count', 0)
                    total = res.get('total', 0)
                    self.log(f"Pixel-wise focus done: avg PBR={avg_pbr:.3f} (ok={count}/{total})")
                    self.log(
                        'Focus encoding backend: {}'.format(
                            res.get('encoding_backend', 'unknown')
                        )
                    )
                    fallback_error = res.get('encoding_fallback_error')
                    if fallback_error:
                        self.log(
                            'GPU encoding fallback reason: {}'.format(
                                fallback_error
                            )
                        )
                    report = res.get('report', {})
                    files = report.get('files', {})
                    if files:
                        self.log(
                            'Point records: {}'.format(
                                files.get('points_csv', '-')
                            )
                        )
                        self.log(
                            'Focus distributions: {}'.format(
                                files.get('distribution_png', '-')
                            )
                        )
                        self.log(
                            'PBR heatmap: {}'.format(
                                files.get('pbr_heatmap_png', '-')
                            )
                        )
                    self.after(0, lambda: self.lbl_pbr.config(text=f"Avg PBR: {avg_pbr:.2f} (ok={count}/{total})"))
                    self.after(
                        0,
                        lambda: self.measure_progress_label.config(
                            text='Pixel-wise report saved'
                        ),
                    )
                else:
                    self.log(f"Pixel-wise focus failed: {res.get('error')}")
                    self.after(0, lambda: self.measure_progress_label.config(text="Pixel-wise focus failed"))
            except Exception as e:
                self.log(f"Pixel-wise focus error: {str(e)}")
                self.after(0, lambda: self.measure_progress_label.config(text="Pixel-wise focus error"))
            finally:
                self.after(0, lambda: self.btn_pixelwise_focus.config(state=tk.NORMAL))

        threading.Thread(target=_run, daemon=True).start()

    def focus_position_multi_phase(self):
        """Focus on selected position with multiple global phase offsets"""
        try:
            x = int(self.focus_x.get())
            y = int(self.focus_y.get())
            phase_list = self._parse_phase_list(self.phase_list_entry.get())
            if not phase_list:
                self.log("Phase list is empty")
                return

            self.log(f"Multi-phase focusing on ({x}, {y}) with {len(phase_list)} phases ({self.phase_unit.get()})")
            self.btn_focus.config(state=tk.DISABLED)
            self.btn_focus_multi_phase.config(state=tk.DISABLED)

            def _focus_multi():
                try:
                    results = self.dmd_controller.conjugate_focus_at_position_multi_phase(x, y, phase_list)
                    ok_results = [r for r in results if r.get('success')]
                    if not ok_results:
                        self.measure_progress_label.config(text="Multi-phase focus failed")
                        self.log("Multi-phase focus failed for all phases")
                        for r in results:
                            self.log(f"phi={r.get('phase_offset_rad', 0):.6f} rad -> error={r.get('error')}")
                        return

                    best = max(ok_results, key=lambda r: r.get('pbr', 0.0))

                    # 更新显示为最佳相位结果
                    focused_image = best.get('focused_image')
                    if focused_image is not None:
                        self.camera_display.set_data(focused_image)
                        self.camera_display.set_clim(vmin=float(np.min(focused_image)), vmax=float(np.max(focused_image)))
                        self.camera_canvas.draw()

                    self.lbl_peak_intensity.config(text=f"Peak Intensity: {best['peak_intensity']:.2f}")
                    self.lbl_background_intensity.config(text=f"Background Intensity: {best['background_intensity']:.2f}")
                    self.lbl_pbr.config(text=f"Current PBR: {best['pbr']:.2f}")
                    self.measure_progress_label.config(text="Multi-phase focus done")

                    self.log(f"Best phase: {best['phase_offset_rad']:.6f} rad | PBR={best['pbr']:.2f} | Peak={best['peak_intensity']:.2f}")

                    # 逐项记录
                    for r in results:
                        if r.get('success'):
                            self.log(f"phi={r['phase_offset_rad']:.6f} rad -> PBR={r['pbr']:.2f}, Peak={r['peak_intensity']:.2f}")
                        else:
                            self.log(f"phi={r.get('phase_offset_rad', 0):.6f} rad -> error={r.get('error')}")

                except Exception as e:
                    self.log(f"Multi-phase focus error: {str(e)}")
                    self.measure_progress_label.config(text="Multi-phase focus failed")
                finally:
                    self.btn_focus.config(state=tk.NORMAL)
                    self.btn_focus_multi_phase.config(state=tk.NORMAL)

            threading.Thread(target=_focus_multi, daemon=True).start()

        except ValueError:
            self.log("Invalid focus position coordinates / phase list")

    def log(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def on_closing(self):
        if self._closing:
            return
        self._closing = True

        # Stop background work and prevent worker callbacks from touching Tk
        # while the widget tree is being destroyed.
        self.measurement_running = False
        self.dmd_controller.stop_optimization()
        self.dmd_controller.reconstruction_running = False
        self.dmd_controller.measure_progress_callback = None
        self.dmd_controller.recon_progress_callback = None

        if self._visualization_after_id is not None:
            try:
                self.after_cancel(self._visualization_after_id)
            except tk.TclError:
                pass
            self._visualization_after_id = None

        self._close_measurement_quality_window()

        try:
            self.dmd_controller.cleanup()
        except Exception as exc:
            print(f"Error releasing DMD resources: {exc}")
        if hasattr(self, 'camera'):
            try:
                self.camera.cleanup()
            except Exception as exc:
                print(f"Error releasing camera resources: {exc}")

        # The camera/DMD SDK can occasionally leave a native worker alive even
        # after releasing its resources.  This daemon watchdog only fires when
        # the normal GUI shutdown below has failed to terminate the process.
        def _force_exit_if_stuck():
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                os._exit(0)

        shutdown_watchdog = threading.Timer(3.0, _force_exit_if_stuck)
        shutdown_watchdog.daemon = True
        shutdown_watchdog.start()

        # Close pyplot-managed figures before tearing down their Tk widgets.
        for figure_name in ('fig_camera', 'fig_stability'):
            figure = getattr(self, figure_name, None)
            if figure is not None:
                try:
                    plt.close(figure)
                except Exception:
                    pass

        try:
            self.quit()
        except tk.TclError:
            pass
        try:
            self.destroy()
        except tk.TclError:
            pass

if __name__ == "__main__":
    app = Application()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
