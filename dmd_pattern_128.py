"""Pure pattern-generation helpers for the aligned 128 x 128 input grid.

Every logical transmission-matrix input maps to exactly one 4 x 4 optical
superpixel. This produces a 512 x 512 binary hologram centred on the physical
1024 x 768 DMD, with all pixels outside the active square held at zero.
"""

from functools import lru_cache

import numpy as np

# The hologram encoder is vendored as source in this repository.
from holograms.dmd_holograms import holo_SP
from holograms.generate_LUT import generate_lut


DMD_WIDTH = 1024
DMD_HEIGHT = 768
INPUT_WIDTH = 128
INPUT_HEIGHT = 128
INPUT_MACRO_PIXEL_SIZE = 4
HOLOGRAM_SUPERPIXEL_SIZE = 4

ACTIVE_WIDTH = INPUT_WIDTH * INPUT_MACRO_PIXEL_SIZE
ACTIVE_HEIGHT = INPUT_HEIGHT * INPUT_MACRO_PIXEL_SIZE
ACTIVE_X = (DMD_WIDTH - ACTIVE_WIDTH) // 2
ACTIVE_Y = (DMD_HEIGHT - ACTIVE_HEIGHT) // 2


if ACTIVE_WIDTH > DMD_WIDTH or ACTIVE_HEIGHT > DMD_HEIGHT:
    raise RuntimeError("The configured 128 x 128 active field does not fit the DMD")


@lru_cache(maxsize=4)
def get_superpixel_lut(px=HOLOGRAM_SUPERPIXEL_SIZE):
    """Return and cache the LUT tuple used by ``holo_SP``."""
    px = int(px)
    if px <= 0:
        raise ValueError("px must be positive")
    return generate_lut("sp", px)


def validate_input_field(field):
    """Validate and return a 128 x 128 complex64 input field."""
    field = np.asarray(field)
    expected_shape = (INPUT_HEIGHT, INPUT_WIDTH)
    if field.shape != expected_shape:
        raise ValueError(
            f"Input field shape {field.shape} does not match {expected_shape}"
        )
    if not np.all(np.isfinite(field)):
        raise ValueError("Input field contains NaN or infinite values")
    return field.astype(np.complex64, copy=False)


def _holo_sp_mean_vectorized(
    field,
    lut,
    pixel_combinations,
    step=0.01,
    renorm=True,
):
    """Bit-exact vectorized equivalent of ``holo_SP(..., ds_method='mean')``.

    The reference implementation loops over all 192 x 192 optical
    superpixels in Python.  This version keeps the same summation order,
    rounding, LUT indexing, row-dependent roll, and per-block transpose while
    performing the LUT lookup and block placement with NumPy arrays.
    """
    field = np.asarray(field).copy()
    max_amplitude = np.max(np.abs(field))
    if renorm:
        if max_amplitude <= 0:
            raise ValueError("Cannot renormalize an all-zero complex field")
        field /= max_amplitude
    elif max_amplitude > 1.0 + 1e-6:
        raise ValueError("Absolute complex-field amplitudes must lie in [0, 1]")

    combination_length = len(pixel_combinations[0])
    n_sp = int(np.sqrt(combination_length))
    if n_sp * n_sp != combination_length:
        raise ValueError("LUT pixel combinations do not form square superpixels")

    # Preserve the exact accumulation order used by holograms._down_sample.
    downsampled = np.zeros_like(field[n_sp // 2 :: n_sp, n_sp // 2 :: n_sp])
    for row_offset in range(n_sp):
        for column_offset in range(n_sp):
            downsampled += field[row_offset::n_sp, column_offset::n_sp]
    downsampled /= n_sp**2

    downsampled_max = np.max(np.abs(downsampled))
    if renorm:
        if downsampled_max <= 0:
            raise ValueError("Downsampled complex field is all zero")
        scaled = downsampled / (downsampled_max * step)
    else:
        # The LUT spans real/imaginary values in [-1, 1].  Do not divide by
        # the field maximum here: calibration needs 0.5 and 1.0 to select
        # different superpixel codes instead of both becoming full scale.
        scaled = downsampled / step

    lut_zero = len(lut) // 2
    real_index = np.rint(np.real(scaled)).astype(np.intp) + lut_zero
    imag_index = np.rint(np.imag(scaled)).astype(np.intp) + lut_zero
    selected = pixel_combinations[lut[real_index, imag_index]]

    row_shifts = (n_sp * np.arange(downsampled.shape[0])) % (n_sp**2)
    roll_indices = (
        np.arange(n_sp**2)[None, :] + row_shifts[:, None]
    ) % (n_sp**2)
    rolled = np.take_along_axis(selected, roll_indices[:, None, :], axis=2)

    # Reference assignment is sp_pixel.reshape(n_sp, n_sp).T for every block.
    return (
        rolled.reshape(
            downsampled.shape[0],
            downsampled.shape[1],
            n_sp,
            n_sp,
        )
        .transpose(0, 3, 1, 2)
        .reshape(field.shape)
    )


def input_field_to_active_hologram(
    field,
    px=HOLOGRAM_SUPERPIXEL_SIZE,
    ds_method="mean",
    lut_cache=None,
    renorm=True,
):
    """Encode a 128 x 128 field into a 512 x 512 binary active hologram."""
    field = validate_input_field(field)
    px = int(px)

    if ACTIVE_HEIGHT % px or ACTIVE_WIDTH % px:
        raise ValueError(
            f"Active shape {(ACTIVE_HEIGHT, ACTIVE_WIDTH)} must be divisible by px={px}"
        )

    expanded_field = np.repeat(
        np.repeat(field, INPUT_MACRO_PIXEL_SIZE, axis=0),
        INPUT_MACRO_PIXEL_SIZE,
        axis=1,
    )

    if lut_cache is None:
        lut_cache = get_superpixel_lut(px)
    if not isinstance(lut_cache, tuple) or len(lut_cache) != 3:
        raise ValueError("lut_cache must be the (field_values, combinations, lut) tuple")

    _, pixel_combinations, lut = lut_cache
    if ds_method == "mean":
        hologram = _holo_sp_mean_vectorized(
            expanded_field,
            lut,
            pixel_combinations,
            renorm=renorm,
        )
    else:
        if not renorm:
            raise ValueError("renorm=False currently requires ds_method='mean'")
        hologram = holo_SP(
            expanded_field,
            lut,
            pixel_combinations,
            ds_method=ds_method,
        )

    expected_shape = (ACTIVE_HEIGHT, ACTIVE_WIDTH)
    if hologram.shape != expected_shape:
        raise RuntimeError(
            f"holo_SP returned {hologram.shape}; expected {expected_shape}"
        )

    # The DMD SDK consumes 8-bit images but is configured for a 1-bit plane.
    return (np.asarray(hologram) > 0).astype(np.uint8) * np.uint8(255)


def active_hologram_to_dmd_canvas(active_hologram):
    """Centre a 512 x 512 hologram on a zero-valued 1024 x 768 canvas."""
    active_hologram = np.asarray(active_hologram, dtype=np.uint8)
    expected_shape = (ACTIVE_HEIGHT, ACTIVE_WIDTH)
    if active_hologram.shape != expected_shape:
        raise ValueError(
            f"Active hologram shape {active_hologram.shape} does not match {expected_shape}"
        )

    canvas = np.zeros((DMD_HEIGHT, DMD_WIDTH), dtype=np.uint8)
    canvas[
        ACTIVE_Y : ACTIVE_Y + ACTIVE_HEIGHT,
        ACTIVE_X : ACTIVE_X + ACTIVE_WIDTH,
    ] = active_hologram
    return canvas


def input_field_to_dmd_pattern(
    field,
    px=HOLOGRAM_SUPERPIXEL_SIZE,
    ds_method="mean",
    lut_cache=None,
    renorm=True,
):
    """Convert one logical complex field into one full-size DMD pattern."""
    active_hologram = input_field_to_active_hologram(
        field,
        px=px,
        ds_method=ds_method,
        lut_cache=lut_cache,
        renorm=renorm,
    )
    return active_hologram_to_dmd_canvas(active_hologram)


def active_region_mask(value=255):
    """Return a full DMD canvas with only the central active square enabled."""
    active = np.full((ACTIVE_HEIGHT, ACTIVE_WIDTH), value, dtype=np.uint8)
    return active_hologram_to_dmd_canvas(active)
