"""Input-field policies used for transmission-matrix phase-conjugate focus."""

from typing import Optional

import numpy as np


FOCUS_MODES = (
    "complex",
    "phase_only",
    "phase_only_2",
    "phase_only_4",
    "amplitude_only_binary",
)


def prepare_conjugate_focus_field(
    tm_values,
    mode: str = "complex",
    phase_levels: Optional[int] = None,
):
    """Return the conjugated input field requested by a focus policy.

    ``complex`` retains the reconstructed amplitudes and phases.
    ``phase_only`` uses unit amplitude without explicit phase quantization.
    ``phase_only_2`` quantizes the unit-amplitude field to 0 or pi.
    ``phase_only_4`` quantizes the unit-amplitude field to four phase levels.
    ``amplitude_only_binary`` uses TM phase only to select an on/off mask; all
    enabled channels are sent to the DMD with the same zero phase.
    The returned field still passes through the DMD superpixel LUT afterward.
    """
    values = np.asarray(tm_values, dtype=np.complex64)
    if mode not in FOCUS_MODES:
        raise ValueError(
            "Unknown focus mode {!r}; choose {}".format(
                mode,
                ", ".join(FOCUS_MODES),
            )
        )

    conjugated = np.conj(values)
    if mode == "complex":
        return conjugated

    if mode == "amplitude_only_binary":
        positive_mask = np.real(values) >= 0.0
        negative_mask = ~positive_mask
        positive_field = np.sum(values * positive_mask, axis=-1)
        negative_field = np.sum(values * negative_mask, axis=-1)
        use_negative = np.abs(negative_field) > np.abs(positive_field)
        selected_mask = np.where(
            np.expand_dims(use_negative, axis=-1),
            negative_mask,
            positive_mask,
        )
        return selected_mask.astype(np.complex64)

    phases = np.angle(conjugated)
    if mode == "phase_only":
        return np.exp(1j * phases).astype(np.complex64)

    default_levels = 2 if mode == "phase_only_2" else 4
    levels = default_levels if phase_levels is None else int(phase_levels)
    if levels < 2:
        raise ValueError("phase_levels must be at least 2")
    phase_step = 2.0 * np.pi / levels
    indices = np.rint(np.mod(phases, 2.0 * np.pi) / phase_step)
    indices = np.mod(indices, levels)
    return np.exp(1j * indices * phase_step).astype(np.complex64)
