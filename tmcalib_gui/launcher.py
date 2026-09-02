"""CLI compatibility helpers backed by the canonical profile registry."""

import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from tmcalib.profiles import PROFILES, get_profile


PROFILE_LABELS: Dict[str, str] = {
    key: profile.display_name for key, profile in PROFILES.items()
}


@dataclass(frozen=True)
class LaunchSpec:
    profile: str
    channel: Optional[str]
    program: str
    arguments: Tuple[str, ...]
    working_directory: str

    @property
    def command(self) -> Tuple[str, ...]:
        return (self.program,) + self.arguments


def profile_supports_channel(profile_name: str) -> bool:
    return bool(get_profile(profile_name).available_channels)


def normalize_channel(profile_name: str, channel: Optional[str]) -> Optional[str]:
    selected = get_profile(profile_name, channel)
    return selected.default_channel if selected.available_channels else None


def build_launch_spec(
    profile_name: str,
    channel: Optional[str] = None,
    python_executable: Optional[str] = None,
    repository_root: Optional[str] = None,
) -> LaunchSpec:
    selected = get_profile(profile_name, channel)
    root = os.path.abspath(
        repository_root
        or os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
    )
    args = ["-u", os.path.join(root, "run_calibration.py"), "--profile", profile_name]
    selected_channel = (
        selected.default_channel if selected.available_channels else None
    )
    if selected_channel is not None:
        args.extend(["--channel", selected_channel])
    return LaunchSpec(
        profile=profile_name,
        channel=selected_channel,
        program=python_executable or sys.executable,
        arguments=tuple(args),
        working_directory=root,
    )


def format_profile_summary(profile_name: str) -> str:
    profile = get_profile(profile_name)
    capabilities = "、".join(sorted(profile.capabilities))
    return (
        "输入：{input_w} × {input_h}    相机：{camera_w} × {camera_h}\n"
        "DMD 有效区：{active_w} × {active_h}    宏像素：{macro}px\n"
        "编码器：{encoder}    重建器：{reconstructor}\n"
        "功能：{capabilities}"
    ).format(
        input_w=profile.input_shape[1],
        input_h=profile.input_shape[0],
        camera_w=profile.camera_roi[1],
        camera_h=profile.camera_roi[0],
        active_w=profile.active_shape[1],
        active_h=profile.active_shape[0],
        macro=profile.input_macro_pixel_size,
        encoder=profile.encoder_key,
        reconstructor=profile.reconstructor_key,
        capabilities=capabilities,
    )


def display_command(command: Sequence[str]) -> str:
    return " ".join('"{}"'.format(part) if " " in part else part for part in command)
