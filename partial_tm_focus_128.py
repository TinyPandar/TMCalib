"""Coordinate-safe loading helpers for a partially reconstructed 128-grid TM."""

import json
import os
from typing import Dict, Tuple

import numpy as np


def load_partial_tm_row(
    tm_path: str,
    metadata_path: str,
    target_x: int,
    target_y: int,
    roi_shape: Tuple[int, int] = (128, 128),
    input_count: int = 128 * 128,
) -> Tuple[np.ndarray, Dict]:
    """Return the TM row mapped to one camera coordinate and its mapping info."""
    roi_height, roi_width = map(int, roi_shape)
    target_x = int(target_x)
    target_y = int(target_y)
    if not 0 <= target_x < roi_width or not 0 <= target_y < roi_height:
        raise ValueError(
            "Target ({}, {}) is outside the {}x{} camera ROI".format(
                target_x, target_y, roi_width, roi_height
            )
        )
    if not os.path.isfile(tm_path):
        raise FileNotFoundError("Partial TM file not found: {}".format(tm_path))
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            "Partial TM metadata not found: {}".format(metadata_path)
        )

    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    selected_range = metadata.get("selected_output_range")
    if (
        not isinstance(selected_range, list)
        or len(selected_range) != 2
        or not all(isinstance(value, int) for value in selected_range)
    ):
        raise ValueError("Metadata does not contain a valid selected_output_range")
    selection_start, selection_stop = selected_range
    if not 0 <= selection_start < selection_stop <= roi_width * roi_height:
        raise ValueError(
            "Invalid selected output range [{}, {})".format(
                selection_start, selection_stop
            )
        )

    target_index = target_y * roi_width + target_x
    if not selection_start <= target_index < selection_stop:
        first_y, first_x = divmod(selection_start, roi_width)
        last_y, last_x = divmod(selection_stop - 1, roi_width)
        raise ValueError(
            "Target ({}, {}) was not reconstructed. Available flattened range "
            "[{}:{}) runs from (x={}, y={}) to (x={}, y={}).".format(
                target_x,
                target_y,
                selection_start,
                selection_stop,
                first_x,
                first_y,
                last_x,
                last_y,
            )
        )

    tm = np.load(tm_path, mmap_mode="r")
    expected_shape = (selection_stop - selection_start, int(input_count))
    if tm.shape != expected_shape:
        raise ValueError(
            "Partial TM shape {} does not match metadata {}".format(
                tm.shape, expected_shape
            )
        )
    if tm.dtype != np.complex64:
        raise ValueError("Partial TM dtype must be complex64, got {}".format(tm.dtype))

    local_index = target_index - selection_start
    row = np.array(tm[local_index], dtype=np.complex64, copy=True)
    del tm
    if not np.all(np.isfinite(row)):
        raise ValueError("Selected partial TM row contains NaN or infinity")
    info = {
        "target_x": target_x,
        "target_y": target_y,
        "target_index": target_index,
        "partial_row_index": local_index,
        "selected_output_range": [selection_start, selection_stop],
    }
    return row, info
