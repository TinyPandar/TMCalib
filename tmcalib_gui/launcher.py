"""Hardware-independent launch configuration used by the Qt frontend."""

import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from calibration_profiles import PROFILES


PROFILE_LABELS: Dict[str, str] = {
    "v4_32x24": "32 x 24 标准测量",
    "v4_32x24_cholesky": "32 x 24 Cholesky 重建",
    "fivefold_160x120": "160 x 120 全幅测量",
    "dense_128x128": "128 x 128 稠密测量",
    "dense_128x128_roi26": "128 x 128 偏振 ROI 测量",
}

PROFILE_ALIASES: Dict[str, str] = {
    "v4_32x24_cholesky": "v4_32x24",
}

CAPABILITY_LABELS: Dict[str, str] = {
    "full_tm": "完整 TM",
    "one_click": "一键流程",
    "partial_tm": "局部 TM",
    "polarization": "偏振通道",
    "remote_reconstruction": "远程重建",
    "test64": "64 点测试",
}


@dataclass(frozen=True)
class LaunchSpec:
    """A validated calibration child-process invocation."""

    profile: str
    channel: Optional[str]
    program: str
    arguments: Tuple[str, ...]
    working_directory: str

    @property
    def command(self) -> Tuple[str, ...]:
        return (self.program,) + self.arguments


def profile_supports_channel(profile_name: str) -> bool:
    """Return whether the selected profile exposes a polarization channel."""
    resolved_name = PROFILE_ALIASES.get(profile_name, profile_name)
    return "polarization" in PROFILES[resolved_name].capabilities


def normalize_channel(profile_name: str, channel: Optional[str]) -> Optional[str]:
    """Normalize a channel for a profile and reject invalid combinations."""
    if profile_name not in PROFILE_LABELS:
        raise ValueError("Unknown calibration profile: {}".format(profile_name))
    if not profile_supports_channel(profile_name):
        if channel is not None:
            raise ValueError(
                "Profile {} does not support a polarization channel".format(
                    profile_name
                )
            )
        return None

    normalized = (channel or "I0").strip().upper()
    if normalized not in ("I0", "I90"):
        raise ValueError("Unsupported polarization channel: {}".format(channel))
    return normalized


def build_launch_spec(
    profile_name: str,
    channel: Optional[str] = None,
    python_executable: Optional[str] = None,
    repository_root: Optional[str] = None,
) -> LaunchSpec:
    """Build the command that starts one existing calibration application."""
    normalized_channel = normalize_channel(profile_name, channel)
    root = os.path.abspath(
        repository_root
        or os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
    )
    args = ["-u", os.path.join(root, "run_calibration.py"), "--profile", profile_name]
    if normalized_channel is not None:
        args.extend(["--channel", normalized_channel])

    return LaunchSpec(
        profile=profile_name,
        channel=normalized_channel,
        program=python_executable or sys.executable,
        arguments=tuple(args),
        working_directory=root,
    )


def format_profile_summary(profile_name: str) -> str:
    """Return concise dimensions and capabilities for the selection panel."""
    resolved_name = PROFILE_ALIASES.get(profile_name, profile_name)
    profile = PROFILES[resolved_name]
    capabilities = "、".join(
        CAPABILITY_LABELS.get(item, item)
        for item in sorted(profile.capabilities)
    )
    return (
        "输入：{input_w} x {input_h}    相机：{camera_w} x {camera_h}\n"
        "DMD 有效区：{active_w} x {active_h}    宏像素：{macro}px\n"
        "功能：{capabilities}"
    ).format(
        input_w=profile.input_shape[1],
        input_h=profile.input_shape[0],
        camera_w=profile.camera_roi[1],
        camera_h=profile.camera_roi[0],
        active_w=profile.active_shape[1],
        active_h=profile.active_shape[0],
        macro=profile.input_macro_pixel_size,
        capabilities=capabilities,
    )


def display_command(command: Sequence[str]) -> str:
    """Render a readable Windows-friendly command preview."""
    return " ".join('"{}"'.format(part) if " " in part else part for part in command)
