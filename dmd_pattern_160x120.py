"""DMD mapping for the 5x 160 x 120 logical input profile.

The source field is expanded with centre-aligned nearest-neighbour sampling
onto the DMD's complete 256 x 192 optical-superpixel grid.  Every expanded
sample is then encoded by one 4 x 4 ``holo_SP`` superpixel, so the resulting
binary pattern fills the complete 1024 x 768 DMD without an outer padding
region.
"""

import numpy as np


DMD_WIDTH = 1024
DMD_HEIGHT = 768
INPUT_WIDTH = 160
INPUT_HEIGHT = 120
INPUT_MACRO_PIXEL_SIZE = 4
HOLOGRAM_SUPERPIXEL_SIZE = 4

EXPANDED_WIDTH = DMD_WIDTH // HOLOGRAM_SUPERPIXEL_SIZE
EXPANDED_HEIGHT = DMD_HEIGHT // HOLOGRAM_SUPERPIXEL_SIZE
ACTIVE_WIDTH = DMD_WIDTH
ACTIVE_HEIGHT = DMD_HEIGHT
ACTIVE_X = 0
ACTIVE_Y = 0
MAPPING_VERSION = "nearest_fill_160x120_to_256x192_v1"


def get_superpixel_lut(px=HOLOGRAM_SUPERPIXEL_SIZE):
    """Reuse the repository's authoritative cached holo_SP LUT."""
    from dmd_pattern_128 import get_superpixel_lut as get_cached_lut

    return get_cached_lut(px)


def nearest_source_indices(source_size, target_size):
    """Map target samples to source samples with centre-aligned nearest-neighbour."""
    source_size = int(source_size)
    target_size = int(target_size)
    if source_size <= 0 or target_size <= 0:
        raise ValueError("source_size and target_size must be positive")

    indices = np.floor(
        (np.arange(target_size, dtype=np.float64) + 0.5)
        * source_size
        / target_size
    ).astype(np.intp)
    return np.clip(indices, 0, source_size - 1)


SOURCE_X_INDICES = nearest_source_indices(INPUT_WIDTH, EXPANDED_WIDTH)
SOURCE_Y_INDICES = nearest_source_indices(INPUT_HEIGHT, EXPANDED_HEIGHT)


def source_repeat_counts(axis="x"):
    """Return how many optical superpixels represent each source-axis sample."""
    if axis == "x":
        return np.bincount(SOURCE_X_INDICES, minlength=INPUT_WIDTH)
    if axis == "y":
        return np.bincount(SOURCE_Y_INDICES, minlength=INPUT_HEIGHT)
    raise ValueError("axis must be 'x' or 'y'")


def validate_input_field(field):
    """Validate and return a 120 x 160 complex64 source field."""
    field = np.asarray(field)
    expected_shape = (INPUT_HEIGHT, INPUT_WIDTH)
    if field.shape != expected_shape:
        raise ValueError(
            f"Input field shape {field.shape} does not match {expected_shape}"
        )
    if not np.all(np.isfinite(field)):
        raise ValueError("Input field contains NaN or infinite values")
    return field.astype(np.complex64, copy=False)


def expand_input_field(field):
    """Expand a 120 x 160 source field to the full 192 x 256 SP grid."""
    field = validate_input_field(field)
    return field[SOURCE_Y_INDICES[:, None], SOURCE_X_INDICES[None, :]]


def input_field_to_active_hologram(
    field,
    px=HOLOGRAM_SUPERPIXEL_SIZE,
    ds_method="mean",
    lut_cache=None,
):
    """Encode one source field into a full-size 768 x 1024 binary hologram."""
    px = int(px)
    if px != HOLOGRAM_SUPERPIXEL_SIZE:
        raise ValueError(
            f"The 160 x 120 full-field mapping requires px={HOLOGRAM_SUPERPIXEL_SIZE}"
        )

    expanded_logical = expand_input_field(field)
    expanded_physical = np.repeat(
        np.repeat(expanded_logical, px, axis=0),
        px,
        axis=1,
    )

    if lut_cache is None:
        lut_cache = get_superpixel_lut(px)
    if not isinstance(lut_cache, tuple) or len(lut_cache) != 3:
        raise ValueError("lut_cache must be the (field_values, combinations, lut) tuple")

    _, pixel_combinations, lut = lut_cache
    if ds_method == "mean":
        from dmd_pattern_128 import _holo_sp_mean_vectorized

        hologram = _holo_sp_mean_vectorized(
            expanded_physical,
            lut,
            pixel_combinations,
        )
    else:
        from holograms.dmd_holograms import holo_SP

        hologram = holo_SP(
            expanded_physical,
            lut,
            pixel_combinations,
            ds_method=ds_method,
        )

    expected_shape = (DMD_HEIGHT, DMD_WIDTH)
    if hologram.shape != expected_shape:
        raise RuntimeError(
            f"holo_SP returned {hologram.shape}; expected {expected_shape}"
        )
    return (np.asarray(hologram) > 0).astype(np.uint8) * np.uint8(255)


def active_hologram_to_dmd_canvas(active_hologram):
    """Validate a full-field hologram and return an owned DMD canvas copy."""
    active_hologram = np.asarray(active_hologram, dtype=np.uint8)
    expected_shape = (DMD_HEIGHT, DMD_WIDTH)
    if active_hologram.shape != expected_shape:
        raise ValueError(
            f"Active hologram shape {active_hologram.shape} does not match {expected_shape}"
        )
    return np.array(active_hologram, copy=True)


def input_field_to_dmd_pattern(
    field,
    px=HOLOGRAM_SUPERPIXEL_SIZE,
    ds_method="mean",
    lut_cache=None,
):
    """Convert one 120 x 160 complex source field to a 1024 x 768 DMD pattern."""
    return input_field_to_active_hologram(
        field,
        px=px,
        ds_method=ds_method,
        lut_cache=lut_cache,
    )


def active_region_mask(value=255):
    """Return a full-DMD mask; this profile has no outer zero-padding area."""
    return np.full((DMD_HEIGHT, DMD_WIDTH), value, dtype=np.uint8)
