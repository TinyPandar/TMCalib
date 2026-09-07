"""Immutable profile specifications; profiles contain data, never GUI logic."""

from dataclasses import dataclass, replace
from typing import Dict, FrozenSet, Optional, Tuple


POLARIZATION_CHANNELS = ("I0", "I90")


@dataclass(frozen=True)
class ProfileSpec:
    """Everything that may vary between calibration configurations."""

    key: str
    display_name: str
    input_shape: Tuple[int, int]
    camera_roi: Tuple[int, int]
    active_shape: Tuple[int, int]
    input_macro_pixel_size: int
    default_exposure_us: float
    capabilities: FrozenSet[str]
    controller_module: str
    camera_module: str
    encoder_key: str
    reconstructor_key: str
    default_channel: Optional[str] = None
    available_channels: Tuple[str, ...] = ()

    def select_channel(self, channel: Optional[str]) -> "ProfileSpec":
        """Return a new profile with a validated polarization selection."""
        if not self.available_channels:
            if channel is not None:
                raise ValueError(
                    "Profile {} does not support channel selection".format(self.key)
                )
            return self

        normalized = str(channel or self.default_channel or "I0").strip().upper()
        if normalized not in self.available_channels:
            raise ValueError(
                "Unsupported channel {!r}; choose {}".format(
                    channel, ", ".join(self.available_channels)
                )
            )
        return replace(self, default_channel=normalized)

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities


PROFILES: Dict[str, ProfileSpec] = {
    "v4_32x24": ProfileSpec(
        key="v4_32x24",
        display_name="32 × 24 标准测量",
        input_shape=(24, 32),
        camera_roi=(128, 128),
        active_shape=(768, 1024),
        input_macro_pixel_size=32,
        default_exposure_us=1500.0,
        capabilities=frozenset(
            {"exposure", "preview", "measurement", "reconstruction", "focus", "pixelwise_report", "remote_reconstruction"}
        ),
        controller_module="calibrate_v4_32x24",
        camera_module="calibrate_v4_32x24",
        encoder_key="coarse_32x24",
        reconstructor_key="ggs21_pinv",
        default_channel="I90",
    ),
    "v4_32x24_cholesky": ProfileSpec(
        key="v4_32x24_cholesky",
        display_name="32 × 24 Cholesky 重建",
        input_shape=(24, 32),
        camera_roi=(128, 128),
        active_shape=(768, 1024),
        input_macro_pixel_size=32,
        default_exposure_us=1500.0,
        capabilities=frozenset(
            {"exposure", "preview", "measurement", "reconstruction", "focus", "pixelwise_report"}
        ),
        controller_module="calibrate_v4_32x24_cholesky",
        camera_module="calibrate_v4_32x24",
        encoder_key="coarse_32x24",
        reconstructor_key="ggs21_cholesky",
        default_channel="I90",
    ),
    "fourfold_128x96": ProfileSpec(
        key="fourfold_128x96",
        display_name="128 x 96 full-field measurement",
        input_shape=(96, 128),
        camera_roi=(128, 128),
        active_shape=(768, 1024),
        input_macro_pixel_size=8,
        default_exposure_us=60.0,
        capabilities=frozenset(
            {"exposure", "preview", "measurement", "reconstruction", "focus", "pixelwise_report", "one_click"}
        ),
        controller_module="calibrate_128x96",
        camera_module="calibrate_128x128",
        encoder_key="aligned_repeat2_128x96",
        reconstructor_key="ggs21_cholesky",
        default_channel="I90",
    ),
    "fivefold_160x120": ProfileSpec(
        key="fivefold_160x120",
        display_name="160 × 120 全幅测量",
        input_shape=(120, 160),
        camera_roi=(128, 128),
        active_shape=(768, 1024),
        input_macro_pixel_size=4,
        default_exposure_us=60.0,
        capabilities=frozenset(
            {"exposure", "preview", "measurement", "reconstruction", "focus", "pixelwise_report", "one_click"}
        ),
        controller_module="calibrate_160x120",
        camera_module="calibrate_128x128",
        encoder_key="nearest_fill_160x120",
        reconstructor_key="ggs21_cholesky",
        default_channel="I90",
    ),
    "dense_128x128": ProfileSpec(
        key="dense_128x128",
        display_name="128 × 128 稠密测量",
        input_shape=(128, 128),
        camera_roi=(128, 128),
        active_shape=(512, 512),
        input_macro_pixel_size=4,
        default_exposure_us=60.0,
        capabilities=frozenset(
            {"exposure", "preview", "measurement", "reconstruction", "focus", "pixelwise_report", "one_click", "partial_tm", "test64"}
        ),
        controller_module="calibrate_128x128",
        camera_module="calibrate_128x128",
        encoder_key="dense_128",
        reconstructor_key="complex32_pinv",
        default_channel="I90",
    ),
    "dense_128x128_roi26": ProfileSpec(
        key="dense_128x128_roi26",
        display_name="128 × 128 偏振 ROI 测量",
        input_shape=(128, 128),
        camera_roi=(26, 26),
        active_shape=(512, 512),
        input_macro_pixel_size=4,
        default_exposure_us=1500.0,
        capabilities=frozenset(
            {"exposure", "preview", "measurement", "reconstruction", "focus", "pixelwise_report", "one_click", "partial_tm", "test64", "polarization"}
        ),
        controller_module="calibrate_128x128",
        camera_module="calibrate_128x128",
        encoder_key="dense_128",
        reconstructor_key="complex32_pinv",
        default_channel="I0",
        available_channels=POLARIZATION_CHANNELS,
    ),
}


def get_profile(key: str, channel: Optional[str] = None) -> ProfileSpec:
    """Resolve one profile without importing a vendor SDK or GUI module."""
    try:
        profile = PROFILES[key]
    except KeyError as exc:
        raise ValueError("Unknown calibration profile: {}".format(key)) from exc
    return profile.select_channel(channel)
