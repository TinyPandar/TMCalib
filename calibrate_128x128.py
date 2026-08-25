# Transmission-matrix calibration: independent 128 x 128 / px=4 test variant.

import json
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
from dmd_pattern_128 import (
    ACTIVE_HEIGHT,
    ACTIVE_WIDTH,
    ACTIVE_X,
    ACTIVE_Y,
    DMD_HEIGHT,
    DMD_WIDTH,
    HOLOGRAM_SUPERPIXEL_SIZE,
    INPUT_HEIGHT,
    INPUT_MACRO_PIXEL_SIZE,
    INPUT_WIDTH,
    active_region_mask,
    get_superpixel_lut,
    input_field_to_dmd_pattern as encode_input_field_128,
)
from tm_reconstruction_128 import ReconstructionConfig, reconstruct_tm
from partial_tm_focus_128 import load_partial_tm_row
from pixelwise_focus_report_128 import (
    build_pixelwise_points,
    save_pixelwise_focus_report,
)
from measurement_quality_report import (
    analyze_measurement_memmap,
    build_measurement_quality_figure,
    save_measurement_quality_outputs,
)
import atexit
import signal

import re
import math

try:
    import paramiko
except ImportError:
    paramiko = None
warnings.filterwarnings('ignore')

# Acquisition timing. Keep the camera exposure shorter than the DMD picture
# period so every hardware trigger can start a new exposure.
TARGET_ACQUISITION_FPS = 500.0
DMD_PICTURE_TIME_US = 1_000_000.0 / TARGET_ACQUISITION_FPS
# CAMERA_EXPOSURE_US = 1500.0
CAMERA_EXPOSURE_US = 60.0

# Full-calibration dataset selection for the 128 x 128 input grid. Change only
# ``active`` to switch between the retained 4N data and the new 8N data.
PATTERN_128_CONFIG = {
    "active": "8N",
    "sets": {
        "4N": {
            "directory": "pregenerated_patterns_128_px4_active512_full",
            "probe_multiplier": 4,
            "reconstruction_output_chunk_size": 512,
            "reconstruction_solver": "cholesky",
            "measurement_filename": "measurements_128_px4_active512_full_memmap.npy",
            "tm_memmap_filename": "transmission_matrix_128_px4_active512_memmap.npy",
            "reconstructed_filename": "reconstructed_field_128_px4_active512.npy",
            "error_curve_filename": "ggs21_error_curve_128_px4_active512.npy",
            "cholesky_cache_filename": "probe_cholesky_128_px4_active512.npy",
            "pinv_real_filename": "probe_pinv_128_px4_active512_fp16_real.npy",
            "pinv_imag_filename": "probe_pinv_128_px4_active512_fp16_imag.npy",
            "pinv_metadata_filename": "probe_pinv_128_px4_active512_fp16.json",
            "reconstruction_metadata_filename": "tm_reconstruction_128_px4_active512.json",
        },
        "8N": {
            "directory": "pregenerated_patterns_128_px4_active512_8N_full",
            "probe_multiplier": 8,
            "reconstruction_output_chunk_size": 512,
            "reconstruction_solver": "complex32_pinv",
            "measurement_filename": "measurements_128_px4_active512_8N_full_memmap.npy",
            "tm_memmap_filename": "transmission_matrix_128_px4_active512_8N_memmap.npy",
            "reconstructed_filename": "reconstructed_field_128_px4_active512_8N.npy",
            "error_curve_filename": "ggs21_error_curve_128_px4_active512_8N.npy",
            "cholesky_cache_filename": "probe_cholesky_128_px4_active512_8N.npy",
            "pinv_real_filename": "probe_pinv_128_px4_active512_8N_fp16_real.npy",
            "pinv_imag_filename": "probe_pinv_128_px4_active512_8N_fp16_imag.npy",
            "pinv_metadata_filename": "probe_pinv_128_px4_active512_8N_fp16.json",
            "reconstruction_metadata_filename": "tm_reconstruction_128_px4_active512_8N.json",
        },
    },
}


def get_active_128_pattern_config():
    """Return and validate the configured full-calibration pattern set."""
    active = PATTERN_128_CONFIG.get("active")
    datasets = PATTERN_128_CONFIG.get("sets", {})
    if active not in datasets:
        available = ", ".join(sorted(datasets)) or "<none>"
        raise ValueError(
            f"Unknown 128 x 128 pattern dataset {active!r}; "
            f"available datasets: {available}"
        )

    config = dict(datasets[active])
    config["name"] = active
    config["directory"] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        config["directory"],
    )
    config["probe_multiplier"] = int(config["probe_multiplier"])
    config["reconstruction_output_chunk_size"] = int(
        config["reconstruction_output_chunk_size"]
    )
    return config

class CameraHandler:
    def __init__(self, cam_index, save_path):
        self.cam_index = cam_index
        self.save_path = save_path
        self.is_running = True
        self.image_queue = queue.Queue(maxsize=1)
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

        # 1.5 ms exposure leaves 0.5 ms of margin in the 2 ms (500 Hz)
        # DMD picture period configured below.
        self.configure_exposure(exposure_time=CAMERA_EXPOSURE_US)
        self.configure_gamma(gamma=1)
        self.configure_gain(gain=0)
        self.configure_buffer_handling()
        self.configure_trigger()

    def publish_latest_image(self, frame_index, image):
        """Queue an owned image copy for the Tk camera view without blocking."""
        payload = (int(frame_index), np.array(image, copy=True))

        # The display only needs the newest completed batch.  Drop a stale
        # pending frame rather than ever slowing down the acquisition thread.
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
            # The GUI (or another producer) won the race; acquisition must not
            # wait for visualization.
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

            # Accept the next trigger after exposure has finished, even while
            # the previous frame is still being read out. This prevents short
            # DMD trigger periods from silently turning into every-other-frame
            # acquisition.
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

            # Spinnaker timestamps are generated by the camera in nanoseconds.
            # Retain the metadata for batch-level rate and gap diagnostics.
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
        # 128 x 128 logical-input variant.  Keep the physical DMD canvas and
        # optical superpixel at its existing value. Each logical input maps to
        # one 4 x 4 optical superpixel, so only the central 512 x 512 square is
        # active and all other DMD pixels remain off.
        self.original_width = DMD_WIDTH
        self.original_height = DMD_HEIGHT
        self.dmd_width = INPUT_WIDTH
        self.dmd_height = INPUT_HEIGHT
        self.pixel_group_size = INPUT_MACRO_PIXEL_SIZE
        self.hologram_superpixel_size = HOLOGRAM_SUPERPIXEL_SIZE
        self.active_width = ACTIVE_WIDTH
        self.active_height = ACTIVE_HEIGHT
        self.active_x = ACTIVE_X
        self.active_y = ACTIVE_Y

        # This entry point is the full 128 x 128 calibration version.
        # No environment-variable setup is required.
        self.test_mode = False
        self.test_probe_count = 64
        self.full_pattern_config = get_active_128_pattern_config()
        self.full_probe_multiplier = self.full_pattern_config["probe_multiplier"]
        self.full_probe_count = (
            self.full_probe_multiplier * self.dmd_width * self.dmd_height
        )
        self.pregenerated_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            (
                "pregenerated_patterns_128_px4_active512_test"
                if self.test_mode
                else self.full_pattern_config["directory"]
            ),
        )
        self.measurement_filename = (
            "measurements_128_px4_active512_test_memmap.npy"
            if self.test_mode
            else self.full_pattern_config["measurement_filename"]
        )
        self.tm_memmap_filename = self.full_pattern_config["tm_memmap_filename"]
        self.reconstructed_filename = self.full_pattern_config["reconstructed_filename"]
        self.error_curve_filename = self.full_pattern_config["error_curve_filename"]
        # The 128-grid reconstruction uses a blockwise GPU solver.  The 8N
        # dataset prefers a precomputed planar-complex32 inverse; Cholesky is
        # retained as the safe fallback.
        self.local_reconstruction_enabled = True
        self.remote_reconstruction_enabled = False
        self.camera = camera_handler
        self.current_pattern = None
        self.pbr_history = []
        self.optimization_running = False
        self.measurement_completed = False
        self.measurement_error = None
        self.reconstruction_running = False
        self.current_pbr = 0
        self.current_peak_intensity = 0
        self.current_stability_corr = None  # Pearson correlation vs baseline white-speckle
        self.time_serie = []
        self.measure_progress_callback = None  # Callback for measurement progress
        self.recon_progress_callback = None  # Callback for reconstruction progress

        # Synchronization parameters
        self.trigger_delay = 50 # microseconds, delay between DMD display and camera trigger
        self.exposure_duration = 6000  # microseconds, camera exposure time
        self.continue_recording = True
        self.current_pattern_index = 0
        self.pattern_lock = threading.Lock()
        # Stability monitoring (baseline white-speckle captured on first batch)
        self._stability_baseline_white_speckle = None
        self._stability_seq = 0
        self.ggs21_iters = 200
        self.ggs21_ratio = 0.89
        self.ggs21_output_chunk_size = self.full_pattern_config[
            "reconstruction_output_chunk_size"
        ]
        self.ggs21_probe_storage = self.full_pattern_config.get(
            "reconstruction_probe_storage", "complex64"
        )
        self.ggs21_use_gpu = True
        # Pixel-wise hologram encoding prefers the first CUDA device and
        # automatically falls back to the bit-exact NumPy implementation.
        self.focus_encoding_use_gpu = True
        self.focus_encoding_gpu_device = 0
        self.focus_encoding_gpu_chunk_size = 32
        self.last_focus_encoding_backend = "not started"
        self.last_focus_encoding_fallback_error = None
        self._focus_gpu_tensor_cache = {}
        self._focus_gpu_failed = False
        self.ggs21_ridge = 1e-4
        self.ggs21_device = "auto"
        self.ggs21_cholesky_cache_filename = self.full_pattern_config[
            "cholesky_cache_filename"
        ]
        self.ggs21_pinv_real_filename = self.full_pattern_config[
            "pinv_real_filename"
        ]
        self.ggs21_pinv_imag_filename = self.full_pattern_config[
            "pinv_imag_filename"
        ]
        self.ggs21_pinv_metadata_filename = self.full_pattern_config[
            "pinv_metadata_filename"
        ]
        base_dir = os.path.dirname(os.path.abspath(__file__))
        low_precision_files_exist = all(
            os.path.isfile(os.path.join(base_dir, filename))
            for filename in (
                self.ggs21_pinv_real_filename,
                self.ggs21_pinv_imag_filename,
                self.ggs21_pinv_metadata_filename,
            )
        )
        preferred_solver = self.full_pattern_config.get(
            "reconstruction_solver", "cholesky"
        )
        self.ggs21_solver = (
            preferred_solver
            if preferred_solver != "complex32_pinv" or low_precision_files_exist
            else "cholesky"
        )
        if self.ggs21_solver == "cholesky" and self.full_probe_multiplier >= 8:
            self.ggs21_output_chunk_size = min(
                self.ggs21_output_chunk_size, 256
            )
        self.ggs21_metadata_filename = self.full_pattern_config[
            "reconstruction_metadata_filename"
        ]
        self.reconstruction_error = None

        # 原始相机数据是强度时保持 True
        self.measurements_are_intensity = True

        # 如果有暗场平均值，可以填入
        self.ggs21_dark_level = 0.0

    def set_measurement_mode(self, test_mode):
        """Switch safely between the 64-pattern test and full calibration."""
        if self.optimization_running or self.reconstruction_running:
            raise RuntimeError(
                "Cannot change measurement mode while acquisition or reconstruction is running."
            )

        self.test_mode = bool(test_mode)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.pregenerated_dir = os.path.join(
            base_dir,
            (
                "pregenerated_patterns_128_px4_active512_test"
                if self.test_mode
                else self.full_pattern_config["directory"]
            ),
        )
        self.measurement_filename = (
            "measurements_128_px4_active512_test_memmap.npy"
            if self.test_mode
            else self.full_pattern_config["measurement_filename"]
        )
        # A fresh stability baseline is required after changing pattern sets.
        self._stability_baseline_white_speckle = None
        self.current_stability_corr = None
        return self.measurement_filename

    def _input_field_to_dmd_pattern(
        self,
        input_field,
        px=None,
        ds_method="mean",
        lut_cache=None,
    ):
        """Use the one authoritative 128-grid-to-DMD mapping."""
        if px is None:
            px = self.hologram_superpixel_size
        return encode_input_field_128(
            input_field,
            px=int(px),
            ds_method=ds_method,
            lut_cache=lut_cache,
        )

    @staticmethod
    def _phase_only_conjugate(field):
        """Return unit-amplitude phase conjugation of a complex TM row."""
        field = np.asarray(field, dtype=np.complex64)
        if not np.all(np.isfinite(field)):
            raise ValueError("TM row contains NaN or infinity")
        return np.exp(-1j * np.angle(field)).astype(np.complex64, copy=False)

    def _build_focus_hologram_batch(
        self,
        tm_rows,
        px=4,
        ds_method="mean",
        lut_cache=None,
        encode_chunk_size=32,
        progress_callback=None,
    ):
        """Encode aligned integer-macro-pixel TM rows in CPU/GPU chunks."""
        tm_rows = np.asarray(tm_rows)
        expected_input_count = self.dmd_height * self.dmd_width
        if tm_rows.ndim != 2 or tm_rows.shape[1] != expected_input_count:
            raise ValueError(
                "TM batch shape {} does not match (batch, {})".format(
                    tm_rows.shape, expected_input_count
                )
            )
        batch_count = int(tm_rows.shape[0])
        encode_chunk_size = max(1, int(encode_chunk_size))
        patterns = np.zeros(
            (batch_count, self.original_height, self.original_width),
            dtype=np.uint8,
        )
        errors = [None] * batch_count
        if batch_count == 0:
            return patterns, errors

        if lut_cache is None:
            lut_cache = get_superpixel_lut(px)
        if not isinstance(lut_cache, tuple) or len(lut_cache) != 3:
            raise RuntimeError("Failed to generate the superpixel LUT")
        _, pixel_combinations, lut = lut_cache
        combination_length = int(len(pixel_combinations[0]))
        n_sp = int(round(math.sqrt(combination_length)))
        can_vectorize = (
            ds_method == "mean"
            and n_sp * n_sp == combination_length
            and int(px) == n_sp
            and self.pixel_group_size % n_sp == 0
            and self.dmd_height * self.pixel_group_size
            == self.active_height
            and self.dmd_width * self.pixel_group_size
            == self.active_width
        )

        gpu_enabled = bool(
            getattr(self, "focus_encoding_use_gpu", True)
        ) and not bool(getattr(self, "_focus_gpu_failed", False))
        if can_vectorize and gpu_enabled and torch.cuda.is_available():
            device_index = int(
                getattr(self, "focus_encoding_gpu_device", 0)
            )
            gpu_chunk_size = max(
                1,
                int(
                    getattr(
                        self,
                        "focus_encoding_gpu_chunk_size",
                        encode_chunk_size,
                    )
                ),
            )
            self.last_focus_encoding_backend = "GPU cuda:{}".format(
                device_index
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
                    device_index=device_index,
                )
            except Exception as exc:
                self.last_focus_encoding_fallback_error = str(exc)
                self._focus_gpu_failed = True
                print(
                    "GPU focus encoding failed; falling back to NumPy: "
                    "{}".format(exc)
                )
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        self.last_focus_encoding_backend = "CPU NumPy"
        if gpu_enabled and not torch.cuda.is_available():
            self.last_focus_encoding_fallback_error = (
                "CUDA is not available"
            )
        if not can_vectorize:
            for index, tm_row in enumerate(tm_rows):
                try:
                    input_field = self._phase_only_conjugate(tm_row).reshape(
                        self.dmd_height, self.dmd_width
                    )
                    patterns[index] = self._input_field_to_dmd_pattern(
                        input_field,
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
                    "TM row contains NaN or infinity"
                    if not finite_mask[local_index]
                    else "Cannot encode an all-zero complex field"
                )

            valid_local_indices = np.flatnonzero(valid_mask)
            if valid_local_indices.size:
                fields = self._phase_only_conjugate(
                    chunk_rows[valid_local_indices]
                ).reshape(
                    -1, self.dmd_height, self.dmd_width
                ).astype(np.complex64, copy=False)
                field_max = np.max(np.abs(fields), axis=(1, 2))
                fields /= field_max[:, None, None]
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
                    np.abs(downsampled), axis=(1, 2)
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
                    selected, roll_indices, axis=3
                )
                active_holograms = (
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
                        self.active_height,
                        self.active_width,
                    )
                    * np.uint8(255)
                )
                global_indices = chunk_start + valid_local_indices
                patterns[
                    global_indices,
                    self.active_y:self.active_y + self.active_height,
                    self.active_x:self.active_x + self.active_width,
                ] = active_holograms

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
        """CUDA encoder for aligned integer-macro-pixel holograms."""
        tm_rows = np.asarray(tm_rows)
        batch_count = int(tm_rows.shape[0])
        patterns = np.zeros(
            (batch_count, self.original_height, self.original_width),
            dtype=np.uint8,
        )
        errors = [None] * batch_count
        if batch_count == 0:
            return patterns, errors

        device_index = int(device_index)
        if not 0 <= device_index < torch.cuda.device_count():
            raise ValueError(
                "CUDA device {} is unavailable".format(device_index)
            )
        device = torch.device("cuda:{}".format(device_index))
        encode_chunk_size = max(1, int(encode_chunk_size))
        repeats_per_input = self.pixel_group_size // n_sp
        downsampled_height = self.dmd_height * repeats_per_input
        downsampled_width = self.dmd_width * repeats_per_input

        tensor_cache = getattr(self, "_focus_gpu_tensor_cache", None)
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
            cached = combinations_tensor, lut_tensor, roll_indices
            tensor_cache[cache_key] = cached
        combinations_tensor, lut_tensor, roll_indices = cached
        lut_zero = int(len(lut) // 2)

        with torch.no_grad():
            for chunk_start in range(0, batch_count, encode_chunk_size):
                chunk_end = min(
                    batch_count, chunk_start + encode_chunk_size
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
                        "TM row contains NaN or infinity"
                        if not finite_mask[local_index]
                        else "Cannot encode an all-zero complex field"
                    )

                valid_local_indices = np.flatnonzero(valid_mask)
                if valid_local_indices.size:
                    rows_tensor = torch.as_tensor(
                        np.ascontiguousarray(
                            chunk_rows[valid_local_indices]
                        ),
                        device=device,
                    )
                    phases = torch.angle(rows_tensor)
                    fields = torch.polar(
                        torch.ones_like(phases), -phases
                    ).reshape(
                        -1, self.dmd_height, self.dmd_width
                    )
                    field_max = torch.amax(
                        torch.abs(fields), dim=(1, 2)
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
                        torch.abs(downsampled), dim=(1, 2)
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
                        selected, 3, expanded_roll_indices
                    )
                    active_holograms = (
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
                            self.active_height,
                            self.active_width,
                        )
                        * 255
                    )
                    global_indices = chunk_start + valid_local_indices
                    patterns[
                        global_indices,
                        self.active_y:self.active_y + self.active_height,
                        self.active_x:self.active_x + self.active_width,
                    ] = active_holograms.cpu().numpy()

                if progress_callback:
                    progress_callback(chunk_end, batch_count)
        return patterns, errors

    def _validate_tm_shape(self, H):
        roi_w = getattr(self.camera, "roi_width", 128)
        roi_h = getattr(self.camera, "roi_height", 128)
        expected = (roi_h * roi_w, self.dmd_height * self.dmd_width)
        if H.shape != expected:
            raise ValueError(
                f"Transmission matrix shape {H.shape} does not match the "
                f"128 x 128 configuration {expected}"
            )

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
        # Keep the unused left/right DMD margins off in this 128-grid variant.
        white_pattern = active_region_mask()[None, ...]
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

        matrix = np.asarray(matrix)
        expected_image_shape = (self.original_height, self.original_width)
        if matrix.ndim != 3 or matrix.shape[1:] != expected_image_shape:
            print(
                f"Error: DMD pattern shape {matrix.shape} must be "
                f"(batch, {self.original_height}, {self.original_width})"
            )
            return False
        if matrix.dtype != np.uint8:
            print(f"Error: DMD pattern dtype {matrix.dtype} must be uint8")
            return False

        batch = matrix.shape[0]
        if batch > 1000:
            print(f"Error: Batch size {batch} exceeds allocated size 1000")
            return False
        # Keep one contiguous byte buffer for the DLL. ``flatten().tobytes()``
        # creates two full-size copies, which is costly for 1000-frame batches.
        matrix = np.ascontiguousarray(matrix, dtype=np.uint8)
        arr = matrix.tobytes(order="C")

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
            self.measurement_completed = False
            self.measurement_error = None
            self.pbr_history = []

            def _run_measurement_guarded():
                try:
                    self.measurement_completed = bool(self.run_measurement())
                except Exception as exc:
                    self.measurement_error = str(exc)
                    print("Error during measurement: {}".format(exc))
                finally:
                    self.optimization_running = False

            thread = threading.Thread(target=_run_measurement_guarded)
            thread.daemon = True
            thread.start()

    def start_reconstruction(self):
        if self.test_mode:
            raise RuntimeError(
                "The 64-pattern optical test is not sufficient for TM recovery."
            )
        thread = threading.Thread(target=self.run_reconstruction)
        thread.daemon = True
        thread.start()

    def stop_optimization(self):
        self.optimization_running = False

    def run_measurement(self):
        # 128 x 128 logical-input measurement on the central 512 x 512 DMD area.
        # ---------------------------------------------------------------------

        # Parameters (you can tune these)
        N_x = self.dmd_width
        N_y = self.dmd_height
        N_in = N_x * N_y            # number of micro-pixels

        # Which camera ROI pixels to reconstruct (use full ROI by default)
        roi_w = getattr(self.camera, 'roi_width', 128)
        roi_h = getattr(self.camera, 'roi_height', 128)
        N_out = roi_w * roi_h

        # The test set remains small; full-calibration probe count is selected
        # by PATTERN_128_CONFIG at the top of this file.
        M = (
            self.test_probe_count
            if self.test_mode
            else N_in * self.full_probe_multiplier
        )

        # Files for measurements
        meas_file = os.path.join(os.getcwd(), self.measurement_filename)

        mode_name = "OPTICAL TEST" if self.test_mode else "FULL CALIBRATION"
        pattern_set_name = (
            "test-64" if self.test_mode else self.full_pattern_config["name"]
        )
        print(
            f"Starting {mode_name}: N_in={N_in}, N_out={N_out}, M={M}, "
            f"pattern_set={pattern_set_name}, "
            f"input={N_y}x{N_x}, macro={self.pixel_group_size}, "
            f"hologram px={self.hologram_superpixel_size}"
        )

        # Create memmap for measurements: shape (M, N_out), dtype=float32
        # Keep the legacy uint16 container so the reconstruction reader and
        # raw-file size checks remain compatible. Values are native
        # Polarized8 codes in the range 0..255.
        meas_shape = (M, N_out)
        meas_mm = np.memmap(meas_file, dtype='uint16', mode='w+', shape=meas_shape)

        pregenerated_dir = self.pregenerated_dir
        probe_file = os.path.join(pregenerated_dir, "probe.npy")
        pattern_file = os.path.join(pregenerated_dir, "patterns_pregenerated.npy")
        metadata_file = os.path.join(pregenerated_dir, "metadata.json")

        if not (
            os.path.exists(probe_file)
            and os.path.exists(pattern_file)
            and os.path.exists(metadata_file)
        ):
            if not self.test_mode:
                raise FileNotFoundError(
                    f"Full {N_x} x {N_y} probe data are not present. Generate them "
                    "with a streaming/full-calibration workflow first."
                )
            print("Pre-generated optical-test data missing; generating them now...")
            from generate_probe_samples_128 import generate_dataset
            generate_dataset(
                pregenerated_dir,
                count=M,
                seed=12804,
            )

        print("\n" + "=" * 70)
        print(f"Loading pre-generated {N_x} x {N_y} patterns...")
        print("=" * 70)
        P = np.load(probe_file, mmap_mode="r")
        full_patterns = np.load(pattern_file, mmap_mode="r")
        with open(metadata_file, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        expected_metadata = {
            "probe_count": M,
            "input_shape": [N_y, N_x],
            "input_macro_pixel_size": self.pixel_group_size,
            "hologram_superpixel_size": self.hologram_superpixel_size,
            "active_shape": [self.active_height, self.active_width],
            "active_offset_xy": [self.active_x, self.active_y],
            "dmd_shape": [self.original_height, self.original_width],
            "mapping_version": self.full_pattern_config.get(
                "mapping_version", "aligned_active512_v1"
            ),
            "reconstruction_ready": False if self.test_mode else True,
        }
        for key, expected_value in expected_metadata.items():
            actual_value = metadata.get(key)
            if actual_value != expected_value:
                raise ValueError(
                    f"Metadata mismatch for {key}: got {actual_value!r}, "
                    f"expected {expected_value!r}"
                )

        expected_probe_shape = (M, N_y, N_x)
        expected_pattern_shape = (
            M,
            self.original_height,
            self.original_width,
        )
        if P.shape != expected_probe_shape:
            raise ValueError(
                f"Probe shape {P.shape} does not match {expected_probe_shape}"
            )
        if full_patterns.shape != expected_pattern_shape:
            raise ValueError(
                f"Pattern shape {full_patterns.shape} does not match "
                f"{expected_pattern_shape}"
            )
        if P.dtype != np.complex64 or full_patterns.dtype != np.uint8:
            raise ValueError(
                f"Unexpected data types: probes={P.dtype}, patterns={full_patterns.dtype}"
            )

        print(f"  Loaded probes: {P.shape}, {P.dtype}")
        print(f"  Loaded patterns: {full_patterns.shape}, {full_patterns.dtype}")
        if self.test_mode:
            print("  TEST DATA ONLY: do not run transmission-matrix recovery.")
        print("  ✓ Pre-generated patterns validated successfully!")
        print("=" * 70 + "\n")

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

            # Show only the final speckle frame from this completed batch.  The
            # one-slot queue keeps visualization asynchronous and prevents a
            # slow Matplotlib redraw from delaying camera/DMD acquisition.
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
        # true. A user stop clears it, so the caller can distinguish a complete
        # calibration file from a deliberately interrupted partial file.
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
        """Recover the full 128 x 128 TM with the scalable blockwise solver."""
        self.reconstruction_running = True
        self.reconstruction_error = None
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config = ReconstructionConfig(
                measurement_path=os.path.join(
                    base_dir, self.measurement_filename
                ),
                probe_path=os.path.join(
                    self.pregenerated_dir, "probe.npy"
                ),
                output_path=os.path.join(
                    base_dir, self.reconstructed_filename
                ),
                error_curve_path=os.path.join(
                    base_dir, self.error_curve_filename
                ),
                metadata_path=os.path.join(
                    base_dir, self.ggs21_metadata_filename
                ),
                cholesky_cache_path=os.path.join(
                    base_dir, self.ggs21_cholesky_cache_filename
                ),
                pinv_real_path=os.path.join(
                    base_dir, self.ggs21_pinv_real_filename
                ),
                pinv_imag_path=os.path.join(
                    base_dir, self.ggs21_pinv_imag_filename
                ),
                pinv_metadata_path=os.path.join(
                    base_dir, self.ggs21_pinv_metadata_filename
                ),
                input_shape=(self.dmd_height, self.dmd_width),
                output_shape=(
                    getattr(self.camera, "roi_height", 128),
                    getattr(self.camera, "roi_width", 128),
                ),
                iterations=int(self.ggs21_iters),
                gs2_ratio=float(self.ggs21_ratio),
                output_chunk_size=int(self.ggs21_output_chunk_size),
                probe_storage=str(self.ggs21_probe_storage),
                ridge=float(self.ggs21_ridge),
                solver=str(self.ggs21_solver),
                device=(
                    str(self.ggs21_device)
                    if self.ggs21_use_gpu
                    else "cpu"
                ),
                dark_level=float(self.ggs21_dark_level),
                measurements_are_intensity=bool(
                    self.measurements_are_intensity
                ),
                resume=True,
                normalize_tm=True,
            )
            result = reconstruct_tm(
                config,
                progress=self.recon_progress_callback,
                stop_requested=lambda: not self.reconstruction_running,
            )
            print(
                "GGS21 reconstruction completed in {:.1f} s: {}".format(
                    result["elapsed_seconds"], result["output_path"]
                )
            )
        except Exception as exc:
            self.reconstruction_error = str(exc)
            print("Error during GGS21 reconstruction: {}".format(exc))
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.reconstruction_running = False

    def _run_reconstruction_legacy_pinv(self):
        """
        使用 GGS 2-1 恢复传输矩阵。

        数据关系：
            X: (M, N_in)
            y: (M, N_out)
            H: (N_out, N_in)

            y = abs(X @ H.T)
        """
        self.reconstruction_running = True
        if not self.local_reconstruction_enabled:
            print(
                "Local TM reconstruction is disabled for the 128 x 128 version: "
                "the current direct pseudoinverse implementation is not safe at "
                "this matrix size."
            )
            self.reconstruction_running = False
            return
        H_mm = None

        try:
            print("Starting GGS 2-1 reconstruction...")

            # ==================== 基本参数 ====================

            N_in = self.dmd_width * self.dmd_height

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
                self.measurement_filename
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
                    base_dir,
                    self.pregenerated_dir,
                    "probe.npy"
                ),
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

                # (M, 128, 128) -> (M, 16384)
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

            expected_probe_count = self.full_probe_multiplier * N_in
            if M != expected_probe_count:
                print(
                    f"Warning: pattern set {self.full_pattern_config['name']} "
                    f"expects {expected_probe_count} probes, but found {M}."
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
                X_inv = torch.linalg.pinv(X)

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
                    random_phase = (
                        torch.rand_like(y)
                        * 2.0
                        * torch.pi
                    )

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

                        # H.T = pinv(X) @ Y
                        H_t = X_inv @ Y

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
            'target_intensity': 0.0,
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
            self._validate_tm_shape(H)

            # H的形状应该是 (N_out, N_in)
            # N_out = 16384 (128x128的ROI像素)
            # N_in = 16384 (128x128 logical DMD input channels)

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
            h_conjugate = self._phase_only_conjugate(h_column)
            print(f"提取的传输矩阵列形状: {h_column.shape}")
            print(f"共轭后的列形状: {h_conjugate.shape}")

            # 4. 将共轭向量重塑为DMD输入场
            # DMD logical input field is (N_y, N_x) = (128, 128)
            N_x = self.dmd_width
            N_y = self.dmd_height

            # 重塑为二维场
            input_field = h_conjugate.reshape(N_y, N_x)
            print(f"输入场形状: {input_field.shape}")

            # 5-6. Encode each logical input as one 4 x 4 superpixel and centre
            # the 512 x 512 hologram on the full 1024 x 768 canvas.
            print("\n生成128 x 128输入版本全息图...")
            full_hologram = self._input_field_to_dmd_pattern(
                input_field,
                px=px,
                ds_method=ds_method,
            )
            print(f"超像素尺寸: {px}x{px}")
            print(f"完整DMD图案形状: {full_hologram.shape}")

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

            # 9. 分析聚焦效果
            if len(captured_image.shape) == 3:
                captured_image = np.mean(captured_image, axis=2)

            # Use exactly the same target/background definition as the
            # pixel-wise focus path.
            focus_analysis = self._analyze_pixelwise_focus_image(
                captured_image,
                target_x,
                target_y,
            )
            target_intensity = focus_analysis['target_intensity']
            peak_intensity = focus_analysis['peak_intensity']
            mean_intensity = focus_analysis['mean_intensity']
            background_intensity = focus_analysis['background_intensity']
            pbr = focus_analysis['pbr']

            print("\n" + "="*70)
            print("聚焦结果分析")
            print("="*70)
            print(f"目标点强度: {target_intensity:.2f}")
            print(f"峰值强度: {peak_intensity:.2f}")
            print(f"平均强度: {mean_intensity:.2f}")
            print(f"背景强度: {background_intensity:.2f}")
            print(f"峰值背景比 (PBR): {pbr:.2f}")
            print("="*70 + "\n")

            # 更新结果字典
            result.update(focus_analysis)

        except Exception as e:
            print(f"聚焦过程出错: {str(e)}")
            result['error'] = str(e)

        return result

    def conjugate_focus_with_partial_tm(
        self,
        target_x,
        target_y,
        tm_filename="reconstructed_field_128_px4_active512_test_subset.npy",
        metadata_filename="tm_reconstruction_128_px4_active512_test_subset.json",
        px=4,
        ds_method="mean",
    ):
        """Focus one camera point using a row from the partial test TM."""
        result = {
            "success": False,
            "focused_image": None,
            "target_index": None,
            "partial_row_index": None,
            "target_intensity": 0.0,
            "peak_intensity": 0.0,
            "peak_position": None,
            "mean_intensity": 0.0,
            "background_intensity": 0.0,
            "pbr": 0.0,
            "error": None,
        }
        camera_started = False
        projection_started = False
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            tm_path = os.path.join(base_dir, tm_filename)
            metadata_path = os.path.join(base_dir, metadata_filename)
            roi_w = getattr(self.camera, "roi_width", 128)
            roi_h = getattr(self.camera, "roi_height", 128)
            input_count = self.dmd_height * self.dmd_width
            tm_row, mapping = load_partial_tm_row(
                tm_path,
                metadata_path,
                target_x,
                target_y,
                roi_shape=(roi_h, roi_w),
                input_count=input_count,
            )
            result.update(mapping)

            input_field = self._phase_only_conjugate(tm_row).reshape(
                self.dmd_height, self.dmd_width
            )
            full_hologram = self._input_field_to_dmd_pattern(
                input_field,
                px=px,
                ds_method=ds_method,
            )
            pattern_batch = np.asarray([full_hologram], dtype=np.uint8)
            if not self.load_pattern(pattern_batch):
                raise RuntimeError("Failed to load the partial-TM focus hologram")

            self.camera.start()
            camera_started = True
            self.DMD.juoptProjection(self.dev_id, 0, 0)
            projection_started = True
            time.sleep(0.1)
            captured_image, _, _ = self.camera.run()
            if captured_image is None:
                raise RuntimeError("Camera did not return a partial-TM focus image")
            captured_image = np.asarray(captured_image)
            if captured_image.shape != (roi_h, roi_w):
                raise RuntimeError(
                    "Focused image shape {} does not match ROI {}".format(
                        captured_image.shape, (roi_h, roi_w)
                    )
                )

            target_x = int(target_x)
            target_y = int(target_y)
            target_intensity = float(captured_image[target_y, target_x])
            peak_flat_index = int(np.argmax(captured_image))
            peak_y, peak_x = np.unravel_index(
                peak_flat_index, captured_image.shape
            )
            peak_intensity = float(captured_image[peak_y, peak_x])

            # Estimate the speckle background outside a 5x5 box around the
            # requested focus. The reported PBR is target/background, so a
            # bright peak elsewhere cannot masquerade as successful focusing.
            background_mask = np.ones(captured_image.shape, dtype=bool)
            y0 = max(0, target_y - 2)
            y1 = min(roi_h, target_y + 3)
            x0 = max(0, target_x - 2)
            x1 = min(roi_w, target_x + 3)
            background_mask[y0:y1, x0:x1] = False
            background_intensity = float(
                np.mean(captured_image[background_mask])
            )
            mean_intensity = float(np.mean(captured_image))
            pbr = (
                target_intensity / background_intensity
                if background_intensity > 0
                else 0.0
            )

            result.update(
                {
                    "success": True,
                    "focused_image": captured_image,
                    "target_intensity": target_intensity,
                    "peak_intensity": peak_intensity,
                    "peak_position": (int(peak_x), int(peak_y)),
                    "mean_intensity": mean_intensity,
                    "background_intensity": background_intensity,
                    "pbr": float(pbr),
                }
            )
            self.current_pbr = float(pbr)
            self.current_peak_intensity = peak_intensity
        except Exception as exc:
            result["error"] = str(exc)
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

    def _load_transmission_matrix(self, tm_filename=None):
        if tm_filename is None:
            tm_filename = self.reconstructed_filename
        tm_file = (
            tm_filename
            if os.path.isabs(tm_filename)
            else os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                tm_filename,
            )
        )
        if not os.path.exists(tm_file):
            raise FileNotFoundError(f"传输矩阵文件不存在: {tm_file}")
        H = np.load(tm_file, mmap_mode="r")
        self._validate_tm_shape(H)
        return H

    def _conjugate_focus_with_tm_legacy(self, H, target_x, target_y, px=4, ds_method='mean', lut_cache=None):
        """
        基于已加载的传输矩阵H做一次共轭聚焦（避免重复读盘）。
        返回值结构与 conjugate_focus_at_position() 一致。
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
            roi_w = getattr(self.camera, 'roi_width', 128)
            roi_h = getattr(self.camera, 'roi_height', 128)

            if target_x >= roi_w or target_y >= roi_h or target_x < 0 or target_y < 0:
                raise ValueError(f"目标位置 ({target_x}, {target_y}) 超出ROI范围 ({roi_w}x{roi_h})")

            col_index = target_y * roi_w + target_x
            result['target_index'] = col_index

            h_column = H[col_index, :]
            h_conjugate = self._phase_only_conjugate(h_column)

            N_x = self.dmd_width
            N_y = self.dmd_height
            input_field = h_conjugate.reshape(N_y, N_x)

            full_hologram = self._input_field_to_dmd_pattern(
                input_field,
                px=px,
                ds_method=ds_method,
                lut_cache=lut_cache,
            )

            pattern_batch = np.array([full_hologram], dtype=np.uint8)
            if not self.load_pattern(pattern_batch):
                raise RuntimeError("加载全息图到DMD失败")

            self.camera.start()
            self.DMD.juoptProjection(self.dev_id, 0, 0)
            time.sleep(0.05)
            captured_image = self.camera.run()
            self.camera.stop()
            self.DMD.juoptStop(self.dev_id)
            self.clear_sequence(0)

            if captured_image is None:
                raise RuntimeError("相机捕获失败")

            if len(captured_image.shape) == 3:
                captured_image = np.mean(captured_image, axis=2)

            peak_intensity = float(np.max(captured_image))
            if captured_image.size > 1:
                mean_intensity = float((np.sum(captured_image) - peak_intensity) / (captured_image.size - 1))
            else:
                mean_intensity = float(peak_intensity)

            edge_size = 10
            if captured_image.shape[0] > 2 * edge_size and captured_image.shape[1] > 2 * edge_size:
                background_pixels = np.concatenate([
                    captured_image[:edge_size, :].flatten(),
                    captured_image[-edge_size:, :].flatten(),
                    captured_image[:, :edge_size].flatten(),
                    captured_image[:, -edge_size:].flatten()
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

        return result

    def _analyze_pixelwise_focus_image(
        self,
        captured_image,
        target_x,
        target_y,
    ):
        """Calculate target-specific intensity, PBR, and peak displacement."""
        roi_w = getattr(self.camera, "roi_width", 128)
        roi_h = getattr(self.camera, "roi_height", 128)
        target_x = int(target_x)
        target_y = int(target_y)
        if not 0 <= target_x < roi_w or not 0 <= target_y < roi_h:
            raise ValueError(
                "Target ({}, {}) is outside ROI {}x{}".format(
                    target_x, target_y, roi_w, roi_h
                )
            )
        captured_image = np.asarray(captured_image)
        if captured_image.shape != (roi_h, roi_w):
            raise RuntimeError(
                "Focus image shape {} does not match ROI {}".format(
                    captured_image.shape, (roi_h, roi_w)
                )
            )

        target_intensity = float(captured_image[target_y, target_x])
        peak_flat_index = int(np.argmax(captured_image))
        peak_y, peak_x = np.unravel_index(
            peak_flat_index, captured_image.shape
        )
        peak_intensity = float(captured_image[peak_y, peak_x])
        image_min_intensity = float(np.min(captured_image))
        image_max_intensity = peak_intensity
        image_intensity_range = image_max_intensity - image_min_intensity
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
            "success": True,
            "focused_image": captured_image,
            "target_index": target_y * roi_w + target_x,
            "target_intensity": target_intensity,
            "peak_intensity": peak_intensity,
            "peak_position": (int(peak_x), int(peak_y)),
            "peak_distance_px": peak_distance,
            "image_min_intensity": image_min_intensity,
            "image_max_intensity": image_max_intensity,
            "image_intensity_range": image_intensity_range,
            "mean_intensity": float(np.mean(captured_image)),
            "background_intensity": background_intensity,
            "pbr": float(pbr),
            "error": None,
        }

    def _project_and_capture_focus_batch(self, pattern_batch):
        """Load and acquire one focus sequence with one hardware session."""
        pattern_batch = np.asarray(pattern_batch, dtype=np.uint8)
        if pattern_batch.ndim != 3:
            raise ValueError("Focus pattern batch must be a 3-D array")
        batch_count = int(pattern_batch.shape[0])
        if not 1 <= batch_count <= 1000:
            raise ValueError("Focus batch size must be between 1 and 1000")

        camera_started = False
        projection_started = False
        images = []
        frame_ids = []
        try:
            self.camera.start()
            camera_started = True
            if not self.load_pattern(pattern_batch):
                raise RuntimeError("Failed to load focus pattern batch")
            projection_result = self.DMD.juoptProjection(
                self.dev_id, 0, 0
            )
            if projection_result not in (None, 0):
                raise RuntimeError(
                    "Focus batch projection failed: {}".format(
                        projection_result
                    )
                )
            projection_started = True

            for frame_index in range(batch_count):
                image, _, _ = self.camera.run()
                if image is None:
                    raise RuntimeError(
                        "Camera missed focus frame {}/{}".format(
                            frame_index + 1, batch_count
                        )
                    )
                images.append(np.asarray(image))
                frame_id = getattr(self.camera, "last_frame_id", None)
                if frame_id is not None:
                    frame_ids.append(int(frame_id))

            if len(frame_ids) >= 2:
                frame_steps = np.diff(
                    np.asarray(frame_ids, dtype=np.int64)
                )
                if np.any(frame_steps != 1):
                    raise RuntimeError(
                        "Camera frame gap detected in focus batch; "
                        "point-to-frame alignment is not trustworthy"
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
        ds_method="mean",
        lut_cache=None,
    ):
        """Project one conjugated TM row and measure target-specific focusing."""
        result = {
            "success": False,
            "focused_image": None,
            "target_index": None,
            "target_intensity": 0.0,
            "peak_intensity": 0.0,
            "peak_position": None,
            "peak_distance_px": float("nan"),
            "mean_intensity": 0.0,
            "background_intensity": 0.0,
            "pbr": 0.0,
            "error": None,
        }
        camera_started = False
        projection_started = False
        try:
            roi_w = getattr(self.camera, "roi_width", 128)
            roi_h = getattr(self.camera, "roi_height", 128)
            target_x = int(target_x)
            target_y = int(target_y)
            if not 0 <= target_x < roi_w or not 0 <= target_y < roi_h:
                raise ValueError(
                    "Target ({}, {}) is outside ROI {}x{}".format(
                        target_x, target_y, roi_w, roi_h
                    )
                )
            tm_row = np.asarray(tm_row, dtype=np.complex64)
            expected_input_count = self.dmd_height * self.dmd_width
            if tm_row.shape != (expected_input_count,):
                raise ValueError(
                    "TM row shape {} does not match ({},)".format(
                        tm_row.shape, expected_input_count
                    )
                )
            if not np.all(np.isfinite(tm_row)):
                raise ValueError("TM row contains NaN or infinity")

            result["target_index"] = target_y * roi_w + target_x
            input_field = self._phase_only_conjugate(tm_row).reshape(
                self.dmd_height, self.dmd_width
            )
            full_hologram = self._input_field_to_dmd_pattern(
                input_field,
                px=px,
                ds_method=ds_method,
                lut_cache=lut_cache,
            )
            if not self.load_pattern(
                np.asarray([full_hologram], dtype=np.uint8)
            ):
                raise RuntimeError("Failed to load pixel-wise focus hologram")

            self.camera.start()
            camera_started = True
            self.DMD.juoptProjection(self.dev_id, 0, 0)
            projection_started = True
            time.sleep(0.05)
            captured_image, _, _ = self.camera.run()
            if captured_image is None:
                raise RuntimeError("Camera did not return a focus image")
            captured_image = np.asarray(captured_image)
            if captured_image.shape != (roi_h, roi_w):
                raise RuntimeError(
                    "Focus image shape {} does not match ROI {}".format(
                        captured_image.shape, (roi_h, roi_w)
                    )
                )

            target_intensity = float(captured_image[target_y, target_x])
            peak_flat_index = int(np.argmax(captured_image))
            peak_y, peak_x = np.unravel_index(
                peak_flat_index, captured_image.shape
            )
            peak_intensity = float(captured_image[peak_y, peak_x])
            image_min_intensity = float(np.min(captured_image))
            image_max_intensity = peak_intensity
            image_intensity_range = image_max_intensity - image_min_intensity
            peak_distance = float(
                math.hypot(int(peak_x) - target_x, int(peak_y) - target_y)
            )
            background_mask = np.ones(captured_image.shape, dtype=bool)
            background_mask[
                max(0, target_y - 2) : min(roi_h, target_y + 3),
                max(0, target_x - 2) : min(roi_w, target_x + 3),
            ] = False
            background_intensity = float(
                np.mean(captured_image[background_mask])
            )
            pbr = (
                target_intensity / background_intensity
                if background_intensity > 0
                else 0.0
            )
            result.update(
                {
                    "success": True,
                    "focused_image": captured_image,
                    "target_intensity": target_intensity,
                    "peak_intensity": peak_intensity,
                    "peak_position": (int(peak_x), int(peak_y)),
                    "peak_distance_px": peak_distance,
                    "image_min_intensity": image_min_intensity,
                    "image_max_intensity": image_max_intensity,
                    "image_intensity_range": image_intensity_range,
                    "mean_intensity": float(np.mean(captured_image)),
                    "background_intensity": background_intensity,
                    "pbr": float(pbr),
                }
            )
        except Exception as exc:
            result["error"] = str(exc)
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
        ds_method="mean",
        lut_cache=None,
    ):
        roi_w = getattr(self.camera, "roi_width", 128)
        roi_h = getattr(self.camera, "roi_height", 128)
        target_x = int(target_x)
        target_y = int(target_y)
        if not 0 <= target_x < roi_w or not 0 <= target_y < roi_h:
            return {
                "success": False,
                "error": "Target is outside the camera ROI",
                "focused_image": None,
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
        result["target_index"] = target_index
        return result

    def pixelwise_focus_average_pbr(
        self,
        stride=1,
        max_points=None,
        batch_size=1000,
        px=4,
        ds_method="mean",
        progress_callback=None,
        frame_callback=None,
        output_dir=None,
        save_raw_images=True,
    ):
        """
        Save captured ROI frames to one memory-mapped .npy stack when enabled.
        逐像素（按stride抽样）进行共轭聚焦，统计PBR并返回平均值。

        - stride: ROI 上的采样步长（1=全逐像素；建议 >=4）
        - max_points: 可选，最多聚焦多少个点（用于快速估计）
        - batch_size: 每次连续加载和采集的全息图数（最大1000）
        - progress_callback: fn(done:int, total:int, message:str)
        - frame_callback: fn(image:ndarray, record:dict)，每次成功采集后调用
        """
        focus_started = time.perf_counter()
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
            raise ValueError("batch_size must be between 1 and 1000")

        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "pixelwise_focus_results_128_px4_active512",
            )
        os.makedirs(output_dir, exist_ok=True)

        raw_images_path = None
        raw_images_mm = None
        if save_raw_images:
            raw_dtype = (
                np.uint16
                if bool(getattr(self.camera, "convert_to_12bit", False))
                else np.uint8
            )
            raw_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            raw_images_path = os.path.join(
                output_dir,
                "pixelwise_focus_raw_images_{}_stride{}_n{}.npy".format(
                    raw_timestamp,
                    max(1, int(stride)),
                    total,
                ),
            )
            raw_images_mm = np.lib.format.open_memmap(
                raw_images_path,
                mode="w+",
                dtype=raw_dtype,
                shape=(total, roi_h, roi_w),
            )

        # 复用 LUT
        lut_cache = get_superpixel_lut(px)
        # normalize cache format to (f_val, px_comb, lut)
        if isinstance(lut_cache, tuple) and len(lut_cache) == 3:
            pass
        else:
            lut_cache = None

        records = []
        ok = 0
        last_img = None
        encoding_backends = set()

        batch_total = int(math.ceil(total / float(batch_size)))
        for batch_index, batch_start in enumerate(
            range(0, total, batch_size), start=1
        ):
            batch_points = points[batch_start:batch_start + batch_size]
            batch_count = len(batch_points)
            target_indices = np.asarray(
                [y * roi_w + x for x, y in batch_points],
                dtype=np.intp,
            )

            def encoding_progress(encoded, encoding_total):
                if progress_callback:
                    backend = getattr(
                        self,
                        "last_focus_encoding_backend",
                        "encoding",
                    )
                    progress_callback(
                        batch_start + encoded,
                        total,
                        "Encoding focus batch {}/{} [{}]: {}/{}".format(
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
                    "last_focus_encoding_backend",
                    "unknown",
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
                batch_error = "All focus holograms in this batch failed"
            del pattern_batch

            for offset, (x, y) in enumerate(batch_points):
                sample_index = batch_start + offset + 1
                raw_image_index = None
                if images is not None and offset < len(images):
                    raw_image = np.asarray(images[offset])
                    if raw_image.shape != (roi_h, roi_w):
                        raise RuntimeError(
                            "Raw focus image shape {} does not match ROI {}".format(
                                raw_image.shape,
                                (roi_h, roi_w),
                            )
                        )
                    if raw_images_mm is not None:
                        raw_image_index = batch_start + offset
                        raw_images_mm[raw_image_index, :, :] = raw_image
                error = pattern_errors[offset] or batch_error
                if error is None:
                    try:
                        res = self._analyze_pixelwise_focus_image(
                            images[offset], x, y
                        )
                    except Exception as exc:
                        res = {
                            "success": False,
                            "focused_image": None,
                            "error": str(exc),
                        }
                else:
                    res = {
                        "success": False,
                        "focused_image": None,
                        "error": error,
                    }

                peak_position = res.get("peak_position")
                peak_x = (
                    peak_position[0]
                    if peak_position is not None else None
                )
                peak_y = (
                    peak_position[1]
                    if peak_position is not None else None
                )
                succeeded = bool(res.get("success"))
                record = {
                    "sample_index": sample_index,
                    "raw_image_index": raw_image_index,
                    "x": x,
                    "y": y,
                    "target_index": y * roi_w + x,
                    "success": succeeded,
                    "target_intensity": (
                        float(res.get("target_intensity", np.nan))
                        if succeeded else np.nan
                    ),
                    "peak_intensity": (
                        float(res.get("peak_intensity", np.nan))
                        if succeeded else np.nan
                    ),
                    "peak_x": peak_x,
                    "peak_y": peak_y,
                    "peak_distance_px": (
                        float(res.get("peak_distance_px", np.nan))
                        if succeeded else np.nan
                    ),
                    "image_min_intensity": (
                        float(res.get("image_min_intensity", np.nan))
                        if succeeded else np.nan
                    ),
                    "image_max_intensity": (
                        float(res.get("image_max_intensity", np.nan))
                        if succeeded else np.nan
                    ),
                    "image_intensity_range": (
                        float(res.get("image_intensity_range", np.nan))
                        if succeeded else np.nan
                    ),
                    "mean_intensity": (
                        float(res.get("mean_intensity", np.nan))
                        if succeeded else np.nan
                    ),
                    "background_intensity": (
                        float(res.get("background_intensity", np.nan))
                        if succeeded else np.nan
                    ),
                    "pbr": (
                        float(res.get("pbr", np.nan))
                        if succeeded else np.nan
                    ),
                    "error": res.get("error"),
                }
                records.append(record)
                if succeeded:
                    ok += 1
                    last_img = res.get("focused_image", last_img)
                    self.current_pbr = float(res.get("pbr", 0.0))
                    self.current_peak_intensity = float(
                        res.get("peak_intensity", 0.0)
                    )
                    if frame_callback and last_img is not None:
                        try:
                            frame_callback(last_img, record)
                        except Exception as exc:
                            print(
                                "Pixel-wise frame callback failed at "
                                "({}, {}): {}".format(x, y, exc)
                            )

            if raw_images_mm is not None:
                raw_images_mm.flush()

            if progress_callback:
                latest_pbr = records[-1]["pbr"] if records else np.nan
                pbr_text = (
                    "{:.2f}".format(latest_pbr)
                    if np.isfinite(latest_pbr) else "-"
                )
                progress_callback(
                    batch_start + batch_count,
                    total,
                    "Focus batch {}/{} captured (ok={}, "
                    "target PBR={})".format(
                        batch_index, batch_total, ok, pbr_text
                    ),
                )

        successful = [item for item in records if item["success"]]
        ok = len(successful)
        avg_pbr = (
            float(np.mean([item["pbr"] for item in successful]))
            if successful
            else 0.0
        )
        if raw_images_mm is not None:
            raw_images_mm.flush()
            del raw_images_mm
        focus_elapsed_seconds = time.perf_counter() - focus_started
        report_started = time.perf_counter()
        report = save_pixelwise_focus_report(
            records,
            roi_shape=(roi_h, roi_w),
            output_dir=output_dir,
            run_label="stride{}_batch{}_n{}".format(
                stride, batch_size, total
            ),
            raw_images_path=raw_images_path,
        )
        report_elapsed_seconds = time.perf_counter() - report_started
        return {
            'success': ok > 0,
            'avg_pbr': avg_pbr,
            'count': ok,
            'total': total,
            'batch_size': batch_size,
            'encoding_backend': " + ".join(sorted(encoding_backends)),
            'encoding_fallback_error': getattr(
                self,
                "last_focus_encoding_fallback_error",
                None,
            ),
            'last_image': last_img,
            'records': records,
            'report': report,
            'raw_images_path': raw_images_path,
            'focus_elapsed_seconds': focus_elapsed_seconds,
            'report_elapsed_seconds': report_elapsed_seconds,
            'error': None if ok > 0 else 'All focus attempts failed'
        }

    def pixelwise_focus_partial_tm_report(
        self,
        max_points=None,
        px=4,
        ds_method="mean",
        progress_callback=None,
        tm_filename="reconstructed_field_128_px4_active512_test_subset.npy",
        metadata_filename="tm_reconstruction_128_px4_active512_test_subset.json",
        output_dir=None,
    ):
        """Test every recovered partial-TM row and export the same QA report."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        tm_path = os.path.join(base_dir, tm_filename)
        metadata_path = os.path.join(base_dir, metadata_filename)
        if not os.path.isfile(tm_path):
            return {
                "success": False,
                "count": 0,
                "total": 0,
                "error": "Partial TM file not found: {}".format(tm_path),
            }
        if not os.path.isfile(metadata_path):
            return {
                "success": False,
                "count": 0,
                "total": 0,
                "error": "Partial TM metadata not found: {}".format(
                    metadata_path
                ),
            }
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        selected_range = metadata.get("selected_output_range")
        if not isinstance(selected_range, list) or len(selected_range) != 2:
            return {
                "success": False,
                "count": 0,
                "total": 0,
                "error": "Partial TM metadata has no selected_output_range",
            }
        selection_start, selection_stop = map(int, selected_range)
        roi_w = getattr(self.camera, "roi_width", 128)
        roi_h = getattr(self.camera, "roi_height", 128)
        input_count = self.dmd_height * self.dmd_width
        H = np.load(tm_path, mmap_mode="r")
        expected_shape = (selection_stop - selection_start, input_count)
        if H.shape != expected_shape or H.dtype != np.complex64:
            return {
                "success": False,
                "count": 0,
                "total": 0,
                "error": "Partial TM {} does not match expected {}".format(
                    (H.shape, H.dtype), expected_shape
                ),
            }

        global_indices = list(range(selection_start, selection_stop))
        if max_points is not None:
            global_indices = global_indices[: max(0, int(max_points))]
        total = len(global_indices)
        if total == 0:
            return {
                "success": False,
                "count": 0,
                "total": 0,
                "error": "No partial-TM points selected",
            }
        lut_cache = get_superpixel_lut(px)
        if not isinstance(lut_cache, tuple) or len(lut_cache) != 3:
            lut_cache = None

        records = []
        last_img = None
        ok = 0
        for sample_index, global_index in enumerate(global_indices, start=1):
            y, x = divmod(global_index, roi_w)
            local_index = global_index - selection_start
            res = self._capture_focus_for_tm_row(
                H[local_index, :],
                x,
                y,
                px=px,
                ds_method=ds_method,
                lut_cache=lut_cache,
            )
            peak_position = res.get("peak_position")
            succeeded = bool(res.get("success"))
            record = {
                "sample_index": sample_index,
                "x": x,
                "y": y,
                "target_index": global_index,
                "success": succeeded,
                "target_intensity": (
                    float(res.get("target_intensity", np.nan))
                    if succeeded
                    else np.nan
                ),
                "peak_intensity": (
                    float(res.get("peak_intensity", np.nan))
                    if succeeded
                    else np.nan
                ),
                "peak_x": (
                    peak_position[0] if peak_position is not None else None
                ),
                "peak_y": (
                    peak_position[1] if peak_position is not None else None
                ),
                "peak_distance_px": (
                    float(res.get("peak_distance_px", np.nan))
                    if succeeded
                    else np.nan
                ),
                "image_min_intensity": (
                    float(res.get("image_min_intensity", np.nan))
                    if succeeded
                    else np.nan
                ),
                "image_max_intensity": (
                    float(res.get("image_max_intensity", np.nan))
                    if succeeded
                    else np.nan
                ),
                "image_intensity_range": (
                    float(res.get("image_intensity_range", np.nan))
                    if succeeded
                    else np.nan
                ),
                "mean_intensity": (
                    float(res.get("mean_intensity", np.nan))
                    if succeeded
                    else np.nan
                ),
                "background_intensity": (
                    float(res.get("background_intensity", np.nan))
                    if succeeded
                    else np.nan
                ),
                "pbr": (
                    float(res.get("pbr", np.nan))
                    if succeeded
                    else np.nan
                ),
                "error": res.get("error"),
            }
            records.append(record)
            if record["success"]:
                ok += 1
                last_img = res.get("focused_image", last_img)
                self.current_pbr = record["pbr"]
                self.current_peak_intensity = record["peak_intensity"]
            if progress_callback:
                pbr_text = (
                    "{:.2f}".format(record["pbr"])
                    if record["success"]
                    else "-"
                )
                progress_callback(
                    sample_index,
                    total,
                    "Partial-TM pixel-wise {}/{} (ok={}, target PBR={})".format(
                        sample_index, total, ok, pbr_text
                    ),
                )

        successful = [record for record in records if record["success"]]
        avg_pbr = (
            float(np.mean([record["pbr"] for record in successful]))
            if successful
            else 0.0
        )
        if output_dir is None:
            output_dir = os.path.join(
                base_dir, "pixelwise_focus_results_128_px4_active512"
            )
        report = save_pixelwise_focus_report(
            records,
            roi_shape=(roi_h, roi_w),
            output_dir=output_dir,
            run_label="partial_n{}".format(total),
        )
        return {
            "success": ok > 0,
            "avg_pbr": avg_pbr,
            "count": ok,
            "total": total,
            "last_image": last_img,
            "records": records,
            "report": report,
            "error": None if ok > 0 else "All partial-TM focus attempts failed",
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
        self._validate_tm_shape(H)

        roi_w = getattr(self.camera, 'roi_width', 128)
        roi_h = getattr(self.camera, 'roi_height', 128)
        if target_x >= roi_w or target_y >= roi_h:
            target_x = roi_w // 2
            target_y = roi_h // 2

        col_index = target_y * roi_w + target_x
        h_column = H[col_index, :]
        h_conjugate = self._phase_only_conjugate(h_column)

        N_x = self.dmd_width
        N_y = self.dmd_height
        base_input_field = h_conjugate.reshape(N_y, N_x)

        # LUT 只生成一次
        lut_cache = get_superpixel_lut(px)

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

                    full_hologram = self._input_field_to_dmd_pattern(
                        input_field,
                        px=px,
                        ds_method=ds_method,
                        lut_cache=lut_cache,
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
    def __init__(self):
        super().__init__()
        self.title("DMD TM Calibration - 128x128 Input / px=4")
        self.geometry("1440x810")

        # Keep native Polarized8 values. Measurement storage remains uint16 so
        # the existing reconstruction file layout does not change.
        self.camera = CameraHandler(cam_index=0, save_path="./camera_1")
        self.camera.convert_to_12bit = False

        # Initialize DMD controller
        self.dmd_controller = DMDController(self.camera)
        active_mode = (
            "64-pattern optical test"
            if self.dmd_controller.test_mode
            else f"{self.dmd_controller.full_probe_count}-pattern full measurement"
        )
        self.title(f"DMD TM Calibration - 128x128 / px=4 - {active_mode}")

        # Initialize measurement state
        self.measurement_running = False
        self.one_click_running = False
        self.one_click_stop_requested = False
        self._stability_last_seq = 0
        self._stability_corr_history = []
        self._stability_time_history = []
        self._closing = False
        self._visualization_after_id = None
        self._measurement_quality_window = None
        self._measurement_quality_figure = None
        self._measurement_quality_canvas = None
        self._measurement_quality_thread = None

        # Setup UI
        self.setup_ui()
        self.refresh_measurement_mode_ui()

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

        # Measurement mode selector. Test and full measurements use separate
        # pattern directories and output files.
        measure_mode_frame = ttk.LabelFrame(left_panel, text="Measurement Mode")
        measure_mode_frame.pack(fill=tk.X, pady=(0, 6))

        self.test_mode_var = tk.BooleanVar(
            value=self.dmd_controller.test_mode
        )
        self.chk_test_mode = ttk.Checkbutton(
            measure_mode_frame,
            text="64-Pattern Optical Test (no TM recovery)",
            variable=self.test_mode_var,
            command=self.on_measurement_mode_changed,
        )
        self.chk_test_mode.pack(side=tk.LEFT, padx=8, pady=4)

        self.lbl_measurement_mode = ttk.Label(measure_mode_frame, text="")
        self.lbl_measurement_mode.pack(side=tk.LEFT, padx=(12, 8), pady=4)

        # Measurement control buttons frame
        measure_frame = ttk.Frame(left_panel)
        measure_frame.pack(fill=tk.X, pady=(0, 10))

        measurement_button_text = (
            "Run 64-Pattern Optical Test"
            if self.dmd_controller.test_mode
            else f"Start Full {self.dmd_controller.full_probe_count}-Pattern Measurement"
        )
        self.btn_measure_start = ttk.Button(measure_frame, text=measurement_button_text,
                                          command=self.start_measurement, state=tk.DISABLED)
        self.btn_measure_start.pack(side=tk.LEFT, padx=(0, 5))

        self.btn_measure_stop = ttk.Button(measure_frame, text="Stop Measurement",
                                         command=self.stop_measurement, state=tk.DISABLED)
        self.btn_measure_stop.pack(side=tk.LEFT, padx=(5, 0))

        self.btn_one_click = ttk.Button(
            measure_frame,
            text="One-Click: Measure + Recover + Pixel-wise + PBR Report",
            command=self.run_one_click_calibration,
            state=tk.DISABLED,
        )
        self.btn_one_click.pack(side=tk.LEFT, padx=(10, 0))

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

        ttk.Label(
            dmd_frame,
            text="128×128 input | 4×4 aligned | active 512×512",
        ).pack(fill=tk.X, pady=2)

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
            state=(
                tk.NORMAL
                if self.dmd_controller.local_reconstruction_enabled
                else tk.DISABLED
            ),
        )
        self.btn_tm_recovery.pack(fill=tk.X, pady=2)

        # Remote TM via server (upload measurements_memmap -> run GGS2_1 -> download TM)
        self.btn_tm_remote = ttk.Button(
            tm_frame,
            text="Recover Transmission Matrix (Remote)",
            command=self.remote_tm_from_server,
            state=(
                tk.NORMAL
                if self.dmd_controller.remote_reconstruction_enabled
                else tk.DISABLED
            ),
        )
        self.btn_tm_remote.pack(fill=tk.X, pady=2)

        # Focus control frame
        focus_frame = ttk.LabelFrame(right_panel, text="Focus Control")
        focus_frame.pack(fill=tk.X, pady=(0, 10))

        self.btn_focus = ttk.Button(focus_frame, text="Focus", command=self.focus_position)
        self.btn_focus.pack(fill=tk.X, pady=2)

        self.btn_focus_partial_tm = ttk.Button(
            focus_frame,
            text="Test Focus (Partial TM)",
            command=self.focus_position_partial_tm,
        )
        self.btn_focus_partial_tm.pack(fill=tk.X, pady=2)

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
        self.focus_x = tk.Spinbox(pos_frame, from_=0, to=127, width=5)
        self.focus_x.pack(side=tk.LEFT, padx=2)
        self.focus_x.delete(0, tk.END)
        self.focus_x.insert(0, 64)

        ttk.Label(pos_frame, text="Y:").pack(side=tk.LEFT, padx=(5, 0))
        self.focus_y = tk.Spinbox(pos_frame, from_=0, to=127, width=5)
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
                img_min = float(np.min(img_data))
                img_mean = float(np.mean(img_data))
                img_max = float(np.max(img_data))
                self.camera_display.set_data(img_data)
                # Use the camera's absolute code range so batch-to-batch
                # brightness changes remain visible instead of being hidden by
                # per-frame auto contrast.
                display_max = 4095.0 if self.camera.convert_to_12bit else 255.0
                self.camera_display.set_clim(vmin=0.0, vmax=display_max)
                total_frames = (
                    self.dmd_controller.test_probe_count
                    if self.dmd_controller.test_mode
                    else self.dmd_controller.full_probe_count
                )
                self.ax_camera.set_title(
                    "Camera View - batch last frame {}/{} | "
                    "min/mean/max: {:.0f}/{:.1f}/{:.0f}".format(
                        idx,
                        total_frames,
                        img_min,
                        img_mean,
                        img_max,
                    )
                )
                self.camera_canvas.draw_idle()
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
                    self.btn_one_click.config(state=tk.NORMAL)
                else:
                    self.log("DMD initialization failed")
            else:
                self.log("No DMD devices found")

        threading.Thread(target=_init, daemon=True).start()

    def refresh_measurement_mode_ui(self):
        """Keep mode-dependent labels and unsafe actions in sync."""
        test_mode = self.dmd_controller.test_mode
        mode_title = (
            "64-pattern optical test"
            if test_mode
            else f"{self.dmd_controller.full_probe_count}-pattern full measurement"
        )
        self.title(f"DMD TM Calibration - 128x128 / px=4 - {mode_title}")
        self.btn_measure_start.config(
            text=(
                "Run 64-Pattern Optical Test"
                if test_mode
                else f"Start Full {self.dmd_controller.full_probe_count}-Pattern Measurement"
            )
        )
        self.lbl_measurement_mode.config(
            text=f"Output: {self.dmd_controller.measurement_filename}"
        )
        self.btn_tm_recovery.config(
            state=(
                tk.DISABLED
                if test_mode
                else (
                    tk.NORMAL
                    if self.dmd_controller.local_reconstruction_enabled
                    else tk.DISABLED
                )
            )
        )
        self.btn_tm_remote.config(
            state=(
                tk.DISABLED
                if test_mode
                else (
                    tk.NORMAL
                    if self.dmd_controller.remote_reconstruction_enabled
                    else tk.DISABLED
                )
            )
        )
        if hasattr(self, "btn_one_click") and not self.one_click_running:
            self.btn_one_click.config(
                state=(
                    tk.NORMAL
                    if self.dmd_controller.is_init
                    else tk.DISABLED
                )
            )

    def on_measurement_mode_changed(self):
        requested_test_mode = bool(self.test_mode_var.get())
        try:
            output_name = self.dmd_controller.set_measurement_mode(
                requested_test_mode
            )
        except RuntimeError as exc:
            self.test_mode_var.set(self.dmd_controller.test_mode)
            messagebox.showwarning("Measurement Mode", str(exc))
            return

        self.refresh_measurement_mode_ui()
        mode_name = (
            "64-pattern optical test"
            if requested_test_mode
            else "full measurement"
        )
        self.log(f"Selected {mode_name}; output file: {output_name}")

    def _set_one_click_controls(self, running):
        """Lock conflicting controls while the full calibration is running."""
        self.one_click_running = bool(running)
        if running:
            for button in (
                self.btn_one_click,
                self.btn_measure_start,
                self.btn_init,
                self.btn_tm_recovery,
                self.btn_tm_remote,
                self.btn_focus,
                self.btn_focus_partial_tm,
                self.btn_pixelwise_focus,
                self.btn_focus_multi_phase,
            ):
                button.config(state=tk.DISABLED)
            self.btn_measure_stop.config(state=tk.NORMAL)
            self.chk_test_mode.config(state=tk.DISABLED)
            return

        self.btn_init.config(state=tk.NORMAL)
        self.btn_measure_stop.config(state=tk.DISABLED)
        self.chk_test_mode.config(state=tk.NORMAL)
        for button in (
            self.btn_focus,
            self.btn_focus_partial_tm,
            self.btn_pixelwise_focus,
            self.btn_focus_multi_phase,
        ):
            button.config(state=tk.NORMAL)
        self.btn_measure_start.config(
            state=(
                tk.NORMAL
                if self.dmd_controller.is_init
                else tk.DISABLED
            )
        )
        self.refresh_measurement_mode_ui()

    @staticmethod
    def _format_elapsed(seconds):
        """Format a duration as HH:MM:SS.ss for the UI log."""
        seconds = max(0.0, float(seconds))
        hours, remainder = divmod(seconds, 3600.0)
        minutes, seconds = divmod(remainder, 60.0)
        return "{:02d}:{:02d}:{:05.2f}".format(
            int(hours), int(minutes), seconds
        )

    def _save_one_click_timings(self, result, timings):
        """Add one-click stage timings to the in-memory and JSON reports."""
        report = result.get("report", {})
        files = report.get("files", {})
        summary_path = files.get("summary_json")
        if not summary_path:
            raise RuntimeError("PBR report has no summary JSON path")

        normalized_timings = {
            key: (float(value) if key.endswith("_seconds") else value)
            for key, value in timings.items()
        }
        report["one_click_timing"] = normalized_timings
        result["one_click_timing"] = normalized_timings

        temporary_path = summary_path + ".timing.tmp"
        try:
            with open(temporary_path, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            os.replace(temporary_path, summary_path)
        finally:
            if os.path.exists(temporary_path):
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass

    def _finish_one_click_success(self, result):
        """Render the final focus frame and summarize the generated report."""
        avg_pbr = float(result.get("avg_pbr", 0.0))
        count = int(result.get("count", 0))
        total = int(result.get("total", 0))
        self.measure_progress_bar.config(value=100)
        self.recon_progress_bar.config(value=100)
        timings = result.get("one_click_timing", {})
        total_elapsed = timings.get("total_seconds")
        completion_text = "One-click complete - PBR report saved"
        if total_elapsed is not None:
            completion_text += " ({})".format(
                self._format_elapsed(total_elapsed)
            )
        self.measure_progress_label.config(text=completion_text)
        self.recon_progress_label.config(text="Reconstruction completed")
        self.lbl_pbr.config(
            text="Avg PBR: {:.2f} (ok={}/{})".format(
                avg_pbr, count, total
            )
        )

        last_image = result.get("last_image")
        if last_image is not None:
            vmin = float(np.min(last_image))
            vmax = float(np.max(last_image))
            if vmax <= vmin:
                vmax = vmin + 1.0
            self.camera_display.set_data(last_image)
            self.camera_display.set_clim(vmin=vmin, vmax=vmax)
            self.ax_camera.set_title("Camera View - final pixel-wise focus")
            self.camera_canvas.draw_idle()

    def run_one_click_calibration(self):
        """Measure, recover the full TM, focus every point, and save PBR QA."""
        if self.one_click_running:
            return
        if (
            self.dmd_controller.optimization_running
            or self.dmd_controller.reconstruction_running
        ):
            messagebox.showwarning(
                "One-Click Calibration",
                "Measurement or reconstruction is already running.",
            )
            return
        if not self.dmd_controller.is_init:
            messagebox.showwarning(
                "One-Click Calibration",
                "Initialize the DMD before starting the full calibration.",
            )
            return
        if not self.dmd_controller.local_reconstruction_enabled:
            messagebox.showwarning(
                "One-Click Calibration",
                "Local transmission-matrix recovery is disabled.",
            )
            return

        # This workflow always needs the complete recoverable data set, even
        # when the UI was previously left in the 64-pattern optical-test mode.
        try:
            self.dmd_controller.set_measurement_mode(False)
        except RuntimeError as exc:
            messagebox.showwarning("One-Click Calibration", str(exc))
            return
        self.test_mode_var.set(False)

        try:
            stride = max(1, min(128, int(self.pw_stride.get())))
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

        self.one_click_stop_requested = False
        self._set_one_click_controls(True)
        self.measurement_running = True
        self.measure_progress_bar.config(value=0)
        self.recon_progress_bar.config(value=0)
        self.measure_progress_label.config(
            text="[1/4] Starting full measurement..."
        )
        self.recon_progress_label.config(text="[2/4] Waiting...")
        self.log(
            "One-click calibration started: full measurement -> local TM "
            "recovery -> pixel-wise focus -> PBR report "
            "(stride={}, max_points={}, batch_size={})".format(
                stride, max_points or "None", batch_size
            )
        )

        def measurement_progress(progress, message):
            self.after(
                0,
                lambda value=progress: self.measure_progress_bar.config(
                    value=value
                ),
            )
            self.after(
                0,
                lambda text=message: self.measure_progress_label.config(
                    text="[1/4] {}".format(text)
                ),
            )

        def reconstruction_progress(progress, message):
            self.after(
                0,
                lambda value=progress: self.recon_progress_bar.config(
                    value=value
                ),
            )
            self.after(
                0,
                lambda text=message: self.recon_progress_label.config(
                    text="[2/4] {}".format(text)
                ),
            )

        def focus_progress(done, total, message):
            progress = 100.0 * done / total if total else 0.0
            self.after(
                0,
                lambda value=progress: self.measure_progress_bar.config(
                    value=value
                ),
            )
            self.after(
                0,
                lambda text=message: self.measure_progress_label.config(
                    text="[3/4] {}".format(text)
                ),
            )

        def worker():
            pipeline_started = time.perf_counter()
            timings = {
                "started_at": datetime.now().astimezone().isoformat(),
            }
            try:
                controller = self.dmd_controller

                self.after(
                    0,
                    lambda: self.log(
                        f"[1/4] Measuring all {controller.full_probe_count:,} probe patterns..."
                    ),
                )
                controller.measure_progress_callback = measurement_progress
                measurement_started = time.perf_counter()
                controller.start_measurement()
                while controller.optimization_running:
                    if self.one_click_stop_requested:
                        controller.stop_optimization()
                    time.sleep(0.5)
                timings["measurement_seconds"] = (
                    time.perf_counter() - measurement_started
                )

                if controller.measurement_error:
                    raise RuntimeError(
                        "Measurement failed: {}".format(
                            controller.measurement_error
                        )
                    )
                if not controller.measurement_completed:
                    raise RuntimeError("Measurement was stopped before completion")

                self.after(
                    0,
                    lambda: self.measure_progress_bar.config(value=100),
                )
                self.after(
                    0,
                    lambda: self.btn_measure_stop.config(state=tk.DISABLED),
                )
                self.after(
                    0,
                    lambda elapsed=timings["measurement_seconds"]: self.log(
                        "[1/4] Full measurement completed in {}.".format(
                            self._format_elapsed(elapsed)
                        )
                    ),
                )

                if self.one_click_stop_requested:
                    raise RuntimeError("One-click calibration was stopped")

                # The quality report scans the complete 4 GiB measurement file.
                # It is useful diagnostic output, but it is not an input to TM
                # recovery.  Run it independently so One Click cannot be held
                # at the average-intensity plot before starting stage 2.
                self.after(
                    0,
                    lambda: self.recon_progress_label.config(
                        text=(
                            "[2/4] Starting local TM recovery "
                            "(quality report runs in background)..."
                        )
                    ),
                )
                self.after(
                    0,
                    lambda: self.log(
                        "[1/4] Measurement quality report started in the "
                        "background; One Click is continuing."
                    ),
                )
                self._start_measurement_quality_analysis()
                self.after(
                    0,
                    lambda: self.log(
                        "[2/4] Recovering the transmission matrix locally..."
                    ),
                )
                controller.measure_progress_callback = None
                controller.recon_progress_callback = reconstruction_progress
                # Run in this pipeline worker so the next step cannot race the
                # reconstruction thread's startup flag.
                recovery_started = time.perf_counter()
                controller.run_reconstruction()
                timings["tm_recovery_seconds"] = (
                    time.perf_counter() - recovery_started
                )
                if controller.reconstruction_error:
                    raise RuntimeError(
                        "TM recovery failed: {}".format(
                            controller.reconstruction_error
                        )
                    )
                self.after(
                    0,
                    lambda: self.recon_progress_bar.config(value=100),
                )
                self.after(
                    0,
                    lambda elapsed=timings["tm_recovery_seconds"]: self.log(
                        "[2/4] TM recovery completed in {}.".format(
                            self._format_elapsed(elapsed)
                        )
                    ),
                )

                if self.one_click_stop_requested:
                    raise RuntimeError("One-click calibration was stopped")

                self.after(
                    0,
                    lambda: self.measure_progress_bar.config(value=0),
                )
                self.after(
                    0,
                    lambda: self.measure_progress_label.config(
                        text="[3/4] Starting pixel-wise focus..."
                    ),
                )
                self.after(
                    0,
                    lambda: self.log(
                        "[3/4] Running pixel-wise focus and collecting PBR..."
                    ),
                )
                pixelwise_started = time.perf_counter()
                result = controller.pixelwise_focus_average_pbr(
                    stride=stride,
                    max_points=max_points,
                    batch_size=batch_size,
                    progress_callback=focus_progress,
                )
                pixelwise_total_seconds = (
                    time.perf_counter() - pixelwise_started
                )
                timings["pixelwise_focus_seconds"] = float(
                    result.get(
                        "focus_elapsed_seconds", pixelwise_total_seconds
                    )
                )
                timings["pbr_report_seconds"] = float(
                    result.get(
                        "report_elapsed_seconds",
                        max(
                            0.0,
                            pixelwise_total_seconds
                            - timings["pixelwise_focus_seconds"],
                        ),
                    )
                )
                if not result.get("success"):
                    raise RuntimeError(
                        "Pixel-wise focus failed: {}".format(
                            result.get("error", "unknown error")
                        )
                    )

                report = result.get("report", {})
                files = report.get("files", {})
                if not files:
                    raise RuntimeError("PBR report did not produce output files")
                summary_path = files.get("summary_json")
                if not summary_path or not os.path.isfile(summary_path):
                    raise RuntimeError("PBR report summary file was not saved")

                avg_pbr = float(result.get("avg_pbr", 0.0))
                count = int(result.get("count", 0))
                total = int(result.get("total", 0))
                timings["completed_at"] = (
                    datetime.now().astimezone().isoformat()
                )
                timings["total_seconds"] = sum(
                    timings[key]
                    for key in (
                        "measurement_seconds",
                        "tm_recovery_seconds",
                        "pixelwise_focus_seconds",
                        "pbr_report_seconds",
                    )
                )
                timings["wall_clock_seconds"] = (
                    time.perf_counter() - pipeline_started
                )
                self._save_one_click_timings(result, timings)
                self.after(
                    0,
                    lambda elapsed=timings["pixelwise_focus_seconds"]: self.log(
                        "[3/4] Pixel-wise focus completed: avg PBR={:.3f} "
                        "(ok={}/{}) in {}.".format(
                            avg_pbr,
                            count,
                            total,
                            self._format_elapsed(elapsed),
                        )
                    ),
                )
                self.after(
                    0,
                    lambda elapsed=timings["pbr_report_seconds"]: self.log(
                        "[4/4] PBR report saved in {}: {}".format(
                            self._format_elapsed(elapsed),
                            files.get("summary_json", files)
                        )
                    ),
                )
                self.after(
                    0,
                    lambda elapsed=timings["total_seconds"]: self.log(
                        "One-click total stage time: {}.".format(
                            self._format_elapsed(elapsed)
                        )
                    ),
                )
                self.after(
                    0,
                    lambda value=result: self._finish_one_click_success(value),
                )
            except Exception as exc:
                message = str(exc)
                failed_after_seconds = time.perf_counter() - pipeline_started
                self.after(
                    0,
                    lambda text=message: self.log(
                        "One-click calibration failed: {}".format(text)
                    ),
                )
                self.after(
                    0,
                    lambda elapsed=failed_after_seconds: self.log(
                        "One-click stopped after {}.".format(
                            self._format_elapsed(elapsed)
                        )
                    ),
                )
                self.after(
                    0,
                    lambda text=message: self.measure_progress_label.config(
                        text="One-click failed: {}".format(text)
                    ),
                )
            finally:
                self.measurement_running = False
                self.dmd_controller.measure_progress_callback = None
                self.dmd_controller.recon_progress_callback = None
                self.after(0, lambda: self._set_one_click_controls(False))

        threading.Thread(target=worker, daemon=True).start()

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
        footer_text = "Saved: {}".format(outputs["figure"])
        if self.one_click_running:
            footer_text = (
                "One Click is continuing with TM recovery in the main window; "
                "this plot can be closed safely.\n" + footer_text
            )
        ttk.Label(
            footer,
            text=footer_text,
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

    def _analyze_and_queue_measurement_quality(self, completed_status=None):
        """Analyze the current 128-grid output and queue its non-modal plot."""
        measurement_path = os.path.join(
            os.getcwd(), self.dmd_controller.measurement_filename
        )
        sensor_max_count = 4095 if self.camera.convert_to_12bit else 255
        try:
            result = analyze_measurement_memmap(
                measurement_path,
                height=self.camera.roi_height,
                width=self.camera.roi_width,
                sensor_max_count=sensor_max_count,
            )
            outputs = save_measurement_quality_outputs(result)
        except Exception as exc:
            error_text = str(exc)

            def _show_quality_error():
                if self._closing:
                    return
                self.log("Measurement quality analysis failed: {}".format(error_text))
                if completed_status is not None:
                    self.measure_progress_label.config(
                        text="Measurement completed - quality analysis failed"
                    )

            try:
                self.after(0, _show_quality_error)
            except tk.TclError:
                pass
            return None

        stats = result["statistics"]

        def _show_quality_result():
            if self._closing:
                return
            if completed_status is not None:
                self.measure_progress_label.config(text=completed_status)
            self.log(
                "Measurement quality: mean={:.3f} counts, zero={:.2%}, max={}".format(
                    stats["global_mean_counts"],
                    stats["zero_fraction"],
                    stats["raw_max_counts"],
                )
            )
            if result["quality_flags"]:
                self.log(
                    "Measurement quality warning: "
                    + "; ".join(result["quality_flags"])
                )
            self._show_measurement_quality_result(result, outputs)

        try:
            self.after(0, _show_quality_result)
        except tk.TclError:
            pass
        return result, outputs

    def _start_measurement_quality_analysis(self, completed_status=None):
        """Start the optional quality report without blocking One Click."""
        active_thread = self._measurement_quality_thread
        if active_thread is not None and active_thread.is_alive():
            return active_thread

        def _quality_worker():
            try:
                self._analyze_and_queue_measurement_quality(
                    completed_status=completed_status
                )
            finally:
                # Keep the Thread object alive until its target has completely
                # returned, then allow a later measurement to start a new one.
                self._measurement_quality_thread = None

        thread = threading.Thread(
            target=_quality_worker,
            name="measurement-quality-report",
            daemon=True,
        )
        self._measurement_quality_thread = thread
        thread.start()
        return thread

    def start_measurement(self):
        """Start measurement process with optimization"""
        try:
            self.dmd_controller.set_measurement_mode(
                bool(self.test_mode_var.get())
            )
        except RuntimeError as exc:
            messagebox.showwarning("Measurement Mode", str(exc))
            return

        self.btn_measure_start.config(state=tk.DISABLED)
        self.btn_measure_stop.config(state=tk.NORMAL)
        self.btn_one_click.config(state=tk.DISABLED)
        self.chk_test_mode.config(state=tk.DISABLED)
        mode_name = (
            "64-pattern optical test"
            if self.dmd_controller.test_mode
            else f"full {self.dmd_controller.full_probe_count}-pattern measurement"
        )
        self.log(f"Starting {mode_name}...")

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

                error = self.dmd_controller.measurement_error
                if error:
                    raise RuntimeError(error)
                if not self.dmd_controller.measurement_completed:
                    self.measure_progress_label.config(
                        text="Measurement stopped"
                    )
                    self.log("Measurement process stopped before completion")
                    return

                self.after(
                    0,
                    lambda: self.measure_progress_label.config(
                        text="Analyzing measurement quality..."
                    ),
                )
                self._analyze_and_queue_measurement_quality(
                    completed_status="Measurement completed - quality plot ready"
                )
                self.measure_progress_bar.config(value=100)
                # self.recon_progress_bar.config(value=100)
                # self.recon_progress_label.config(text="Reconstruction completed")
                self.log("Measurement process completed")

            except Exception as e:
                self.log(f"Measurement error: {str(e)}")
                self.measure_progress_label.config(text="Measurement failed")
                # self.recon_progress_label.config(text="Reconstruction failed")
            finally:
                self.btn_measure_start.config(state=tk.NORMAL)
                self.btn_measure_stop.config(state=tk.DISABLED)
                self.btn_one_click.config(
                    state=(
                        tk.NORMAL
                        if self.dmd_controller.is_init
                        else tk.DISABLED
                    )
                )
                self.chk_test_mode.config(state=tk.NORMAL)
                self.measurement_running = False
                self.dmd_controller.measure_progress_callback = None  # Clear callbacks
                self.dmd_controller.recon_progress_callback = None

        self.measurement_running = True
        threading.Thread(target=_measure, daemon=True).start()

    def stop_measurement(self):
        """Stop measurement/optimization process"""
        if self.one_click_running:
            self.one_click_stop_requested = True
            self.measurement_running = False
            self.dmd_controller.stop_optimization()
            self.btn_measure_stop.config(state=tk.DISABLED)
            self.measure_progress_label.config(
                text="Stopping one-click calibration..."
            )
            self.log("Stopping one-click calibration...")
            return
        self.measurement_running = False
        self.dmd_controller.stop_optimization()
        self.btn_measure_start.config(state=tk.NORMAL)
        self.btn_measure_stop.config(state=tk.DISABLED)
        self.measure_progress_label.config(text="Measurement stopped")
        self.recon_progress_label.config(text="Reconstruction stopped")
        self.log("Measurement/optimization process stopped")

    def recover_transmission_matrix(self):
        """Recover transmission matrix"""
        if self.dmd_controller.test_mode:
            self.log("64-pattern optical-test data cannot recover a full TM.")
            messagebox.showinfo(
                "Transmission Matrix",
                "Switch to full measurement mode before TM recovery.",
            )
            return
        if not self.dmd_controller.local_reconstruction_enabled:
            self.log("Local TM recovery is disabled.")
            return
        self.log("Starting transmission matrix recovery...")
        self.btn_tm_recovery.config(state=tk.DISABLED)
        self.btn_one_click.config(state=tk.DISABLED)

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

                error = self.dmd_controller.reconstruction_error
                if error:
                    raise RuntimeError(error)
                self.after(
                    0,
                    lambda: self.recon_progress_bar.config(value=100),
                )
                self.after(
                    0,
                    lambda: self.recon_progress_label.config(
                        text="Reconstruction completed"
                    ),
                )
                self.log("Transmission matrix recovery completed")

            except Exception as e:
                self.log(f"TM recovery error: {str(e)}")
                self.after(
                    0,
                    lambda: self.recon_progress_label.config(
                        text="TM recovery failed"
                    ),
                )
            finally:
                self.after(
                    0,
                    lambda: self.btn_tm_recovery.config(state=tk.NORMAL),
                )
                self.after(
                    0,
                    lambda: self.btn_one_click.config(
                        state=(
                            tk.NORMAL
                            if self.dmd_controller.is_init
                            else tk.DISABLED
                        )
                    ),
                )
                self.dmd_controller.recon_progress_callback = None

        threading.Thread(target=_recover_tm, daemon=True).start()

    def remote_tm_from_server(self):
        """
        使用服务器上的 GGS2_1 算法恢复传输矩阵:
        1) 上传本地 measurements_memmap 到服务器
        2) 在服务器运行脚本计算 TM
        3) 下载 TM 到本地 (默认保存为 reconstructed_field.npy 以兼容后续聚焦函数)
        """
        if not self.dmd_controller.remote_reconstruction_enabled:
            self.log(
                "Remote TM recovery is disabled until the server-side 128-grid "
                "script has been validated."
            )
            return

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

    def focus_position_partial_tm(self):
        """Focus using one of the camera pixels recovered by the partial TM test."""
        try:
            x = int(self.focus_x.get())
            y = int(self.focus_y.get())
        except ValueError:
            self.log("Invalid partial-TM focus coordinates")
            return

        self.log("Partial-TM focusing on position ({}, {})".format(x, y))
        self.btn_focus_partial_tm.config(state=tk.DISABLED)
        self.measure_progress_label.config(text="Partial-TM focusing...")

        def worker():
            result = self.dmd_controller.conjugate_focus_with_partial_tm(x, y)

            def finish():
                try:
                    if result["success"]:
                        focused_image = result["focused_image"]
                        self.camera_display.set_data(focused_image)
                        self.camera_display.set_clim(
                            vmin=float(np.min(focused_image)),
                            vmax=float(np.max(focused_image)),
                        )
                        self.camera_canvas.draw()
                        self.lbl_peak_intensity.config(
                            text="Peak Intensity: {:.2f}".format(
                                result["peak_intensity"]
                            )
                        )
                        self.lbl_background_intensity.config(
                            text="Background Intensity: {:.2f}".format(
                                result["background_intensity"]
                            )
                        )
                        self.lbl_pbr.config(
                            text="Target PBR: {:.2f}".format(result["pbr"])
                        )
                        self.measure_progress_bar.config(
                            mode="determinate", value=100
                        )
                        self.measure_progress_label.config(
                            text="Partial-TM focus at ({}, {})".format(x, y)
                        )
                        self.log(
                            "Partial-TM focus completed: target=({},{}) "
                            "target intensity={:.2f}, peak={} {:.2f}, "
                            "target PBR={:.2f}".format(
                                x,
                                y,
                                result["target_intensity"],
                                result["peak_position"],
                                result["peak_intensity"],
                                result["pbr"],
                            )
                        )
                    else:
                        self.measure_progress_bar.config(
                            mode="determinate", value=0
                        )
                        self.measure_progress_label.config(
                            text="Partial-TM focus failed"
                        )
                        self.log(
                            "Partial-TM focus failed: {}".format(
                                result["error"]
                            )
                        )
                finally:
                    self.btn_focus_partial_tm.config(state=tk.NORMAL)

            self.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    def pixelwise_focus_avg_pbr(self):
        """逐像素（抽样）聚焦，计算PBR平均值"""
        self.btn_pixelwise_focus.config(state=tk.DISABLED)
        self.btn_one_click.config(state=tk.DISABLED)
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
        # Retain only the latest image instead of filling Tk's event queue with
        # stale frames during a complete 16,384-point scan.
        frame_lock = threading.Lock()
        frame_state = {"latest": None, "update_pending": False}

        def _render_latest_frame():
            with frame_lock:
                payload = frame_state["latest"]
                frame_state["latest"] = None

            if payload is not None:
                image, record = payload
                vmin = float(np.min(image))
                vmax = float(np.max(image))
                if vmax <= vmin:
                    vmax = vmin + 1.0
                self.camera_display.set_data(image)
                self.camera_display.set_clim(vmin=vmin, vmax=vmax)
                self.ax_camera.set_title(
                    "Camera View - target ({}, {})".format(
                        record["x"], record["y"]
                    )
                )
                self.lbl_pbr.config(
                    text="Target PBR: {:.2f}".format(record["pbr"])
                )
                self.lbl_peak_intensity.config(
                    text="Peak Intensity: {:.2f}".format(
                        record["peak_intensity"]
                    )
                )
                self.lbl_background_intensity.config(
                    text="Background Intensity: {:.2f}".format(
                        record["background_intensity"]
                    )
                )
                self.camera_canvas.draw_idle()

            schedule_again = False
            with frame_lock:
                if frame_state["latest"] is not None:
                    schedule_again = True
                else:
                    frame_state["update_pending"] = False
            if schedule_again:
                self.after(0, _render_latest_frame)

        def _frame(image, record):
            # Keep an owned copy in case a camera SDK reuses its image buffer.
            payload = (np.array(image, copy=True), dict(record))
            schedule_update = False
            with frame_lock:
                frame_state["latest"] = payload
                if not frame_state["update_pending"]:
                    frame_state["update_pending"] = True
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
                    "Pixel-wise focus started "
                    "(stride={}, max_points={}, batch_size={})".format(
                        stride,
                        max_points or "None",
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
                        "Pixel-wise focus time: {} | PBR report time: {}".format(
                            self._format_elapsed(
                                res.get("focus_elapsed_seconds", 0.0)
                            ),
                            self._format_elapsed(
                                res.get("report_elapsed_seconds", 0.0)
                            ),
                        )
                    )
                    self.log(
                        "Focus encoding backend: {}".format(
                            res.get("encoding_backend", "unknown")
                        )
                    )
                    fallback_error = res.get("encoding_fallback_error")
                    if fallback_error:
                        self.log(
                            "GPU encoding fallback reason: {}".format(
                                fallback_error
                            )
                        )
                    report = res.get("report", {})
                    files = report.get("files", {})
                    if files:
                        self.log(
                            "Point records: {}".format(
                                files.get("points_csv", "-")
                            )
                        )
                        self.log(
                            "Focus distributions: {}".format(
                                files.get("distribution_png", "-")
                            )
                        )
                        self.log(
                            "PBR heatmap: {}".format(
                                files.get("pbr_heatmap_png", "-")
                            )
                        )
                        self.log(
                            "Raw focus images: {}".format(
                                files.get("raw_images_npy", "-")
                            )
                        )
                    self.after(0, lambda: self.lbl_pbr.config(text=f"Avg PBR: {avg_pbr:.2f} (ok={count}/{total})"))
                    self.after(
                        0,
                        lambda: self.measure_progress_label.config(
                            text="Pixel-wise report saved"
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
                self.after(
                    0,
                    lambda: self.btn_one_click.config(
                        state=(
                            tk.NORMAL
                            if self.dmd_controller.is_init
                            else tk.DISABLED
                        )
                    ),
                )

        threading.Thread(target=_run, daemon=True).start()

    def pixelwise_focus_partial_tm_report(self):
        """Run every recovered partial-TM point and save spatial QA outputs."""
        self.btn_pixelwise_partial_tm.config(state=tk.DISABLED)
        self.measure_progress_bar.config(value=0)
        self.measure_progress_label.config(
            text="Partial-TM pixel-wise focusing..."
        )
        try:
            max_points_raw = int(self.pw_max_points.get())
            max_points = None if max_points_raw <= 0 else max_points_raw
        except Exception:
            max_points = None

        def progress(done, total, message):
            percent = 100.0 * done / total if total else 0.0
            self.after(
                0,
                lambda value=percent: self.measure_progress_bar.config(
                    value=value
                ),
            )
            self.after(
                0,
                lambda text=message: self.measure_progress_label.config(
                    text=text
                ),
            )

        def run():
            try:
                self.log(
                    "Partial-TM pixel-wise report started "
                    "(max_points={})".format(max_points or "all recovered")
                )
                result = (
                    self.dmd_controller.pixelwise_focus_partial_tm_report(
                        max_points=max_points,
                        progress_callback=progress,
                    )
                )
                if not result.get("success"):
                    raise RuntimeError(result.get("error"))
                average_pbr = float(result.get("avg_pbr", 0.0))
                count = int(result.get("count", 0))
                total = int(result.get("total", 0))
                report = result.get("report", {})
                files = report.get("files", {})
                self.log(
                    "Partial-TM pixel-wise done: avg target PBR={:.3f} "
                    "(ok={}/{})".format(average_pbr, count, total)
                )
                self.log(
                    "Point records: {}".format(files.get("points_csv", "-"))
                )
                self.log(
                    "Focus distributions: {}".format(
                        files.get("distribution_png", "-")
                    )
                )
                self.log(
                    "PBR heatmap: {}".format(
                        files.get("pbr_heatmap_png", "-")
                    )
                )

                last_image = result.get("last_image")

                def finish_success():
                    self.lbl_pbr.config(
                        text="Avg target PBR: {:.2f} (ok={}/{})".format(
                            average_pbr, count, total
                        )
                    )
                    self.measure_progress_label.config(
                        text="Partial-TM pixel-wise report saved"
                    )
                    if last_image is not None:
                        self.camera_display.set_data(last_image)
                        self.camera_display.set_clim(
                            vmin=float(np.min(last_image)),
                            vmax=float(np.max(last_image)),
                        )
                        self.camera_canvas.draw()

                self.after(0, finish_success)
            except Exception as exc:
                message = str(exc)
                self.log("Partial-TM pixel-wise error: {}".format(message))
                self.after(
                    0,
                    lambda: self.measure_progress_label.config(
                        text="Partial-TM pixel-wise report failed"
                    ),
                )
            finally:
                self.after(
                    0,
                    lambda: self.btn_pixelwise_partial_tm.config(
                        state=tk.NORMAL
                    ),
                )

        threading.Thread(target=run, daemon=True).start()

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
        self.one_click_stop_requested = True
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
