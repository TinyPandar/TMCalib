"""Stable identifiers for the supported TM calibration configurations."""

from dataclasses import dataclass
from typing import FrozenSet, Tuple


POLARIZATION_CHANNELS = ("I0", "I90")


@dataclass(frozen=True)
class CalibrationProfile:
    """Dimensions and optional capabilities for one calibration workflow."""

    name: str
    input_shape: Tuple[int, int]
    camera_roi: Tuple[int, int]
    input_macro_pixel_size: int
    active_shape: Tuple[int, int]
    capabilities: FrozenSet[str]


PROFILES = {
    "v4_32x24": CalibrationProfile(
        name="v4_32x24",
        input_shape=(24, 32),
        camera_roi=(128, 128),
        input_macro_pixel_size=32,
        active_shape=(768, 1024),
        capabilities=frozenset({"full_tm", "remote_reconstruction"}),
    ),
    "fivefold_160x120": CalibrationProfile(
        name="fivefold_160x120",
        input_shape=(120, 160),
        camera_roi=(128, 128),
        input_macro_pixel_size=4,
        active_shape=(768, 1024),
        capabilities=frozenset({"full_tm", "one_click"}),
    ),
    "dense_128x128": CalibrationProfile(
        name="dense_128x128",
        input_shape=(128, 128),
        camera_roi=(128, 128),
        input_macro_pixel_size=4,
        active_shape=(512, 512),
        capabilities=frozenset({"full_tm", "test64", "partial_tm", "one_click"}),
    ),
    "dense_128x128_roi26": CalibrationProfile(
        name="dense_128x128_roi26",
        input_shape=(128, 128),
        camera_roi=(26, 26),
        input_macro_pixel_size=4,
        active_shape=(512, 512),
        capabilities=frozenset(
            {"full_tm", "test64", "partial_tm", "one_click", "polarization"}
        ),
    ),
}


def normalize_polarization_channel(channel: str) -> str:
    """Return a canonical channel name and reject ambiguous values."""
    normalized = str(channel).strip().upper()
    if normalized not in POLARIZATION_CHANNELS:
        raise ValueError(
            "Unsupported polarization channel {!r}; choose {}".format(
                channel, ", ".join(POLARIZATION_CHANNELS)
            )
        )
    return normalized


def channelized_filename(filename: str, channel: str) -> str:
    """Replace an existing I0/I90 filename tag with the selected channel."""
    normalized = normalize_polarization_channel(channel)
    result = str(filename)
    for existing in POLARIZATION_CHANNELS:
        result = result.replace("_{}".format(existing), "_{}".format(normalized))
    return result


def pyspin_polarization_quadrant(pyspin, channel: str):
    """Resolve the selected channel without importing the vendor SDK here."""
    normalized = normalize_polarization_channel(channel)
    return getattr(
        pyspin,
        "SPINNAKER_POLARIZATION_QUADRANT_{}".format(normalized),
    )
