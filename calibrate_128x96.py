"""4x TM calibration: 128 x 96 logical input, 128 x 128 camera output.

This entry point specializes the established 128-output acquisition chain.
Each logical input occupies an aligned 8 x 8 DMD macro-pixel and is encoded
as a 2 x 2 group of 4 x 4 ``holo_SP`` tiles across the full 1024 x 768 DMD.
"""

import os

import calibrate_128x128 as core
from dmd_pattern_128x96 import (
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
    MAPPING_VERSION,
    active_region_mask,
    get_superpixel_lut,
    input_field_to_dmd_pattern,
)


PATTERN_128X96_CONFIG = {
    "active": "8N",
    "sets": {
        "8N": {
            "directory": "pregenerated_patterns_128x96_fill_8N_full",
            "probe_multiplier": 8,
            "reconstruction_output_chunk_size": 256,
            "reconstruction_solver": "cholesky",
            "reconstruction_probe_storage": "planar_complex32",
            "measurement_filename": (
                "measurements_128x96_to_128x128_fill_8N_memmap.npy"
            ),
            "tm_memmap_filename": (
                "transmission_matrix_128x96_to_128x128_fill_8N_memmap.npy"
            ),
            "reconstructed_filename": (
                "reconstructed_field_128x96_to_128x128_fill_8N.npy"
            ),
            "error_curve_filename": (
                "ggs21_error_curve_128x96_to_128x128_fill_8N.npy"
            ),
            "cholesky_cache_filename": "probe_cholesky_128x96_fill_8N.npy",
            "pinv_real_filename": "probe_pinv_128x96_fill_8N_fp16_real.npy",
            "pinv_imag_filename": "probe_pinv_128x96_fill_8N_fp16_imag.npy",
            "pinv_metadata_filename": "probe_pinv_128x96_fill_8N_fp16.json",
            "reconstruction_metadata_filename": (
                "tm_reconstruction_128x96_to_128x128_fill_8N.json"
            ),
            "mapping_version": MAPPING_VERSION,
        },
    },
}


def get_active_128x96_pattern_config(output_tag=None):
    """Return and validate the selected 128 x 96 pattern dataset."""
    # The shared controller always supplies its optional output tag. This
    # profile is not channelized, so the tag does not alter dataset filenames.
    del output_tag
    active = PATTERN_128X96_CONFIG.get("active")
    datasets = PATTERN_128X96_CONFIG.get("sets", {})
    if active not in datasets:
        available = ", ".join(sorted(datasets)) or "<none>"
        raise ValueError(
            f"Unknown 128 x 96 pattern dataset {active!r}; available: {available}"
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


# The shared acquisition implementation resolves these symbols from the core
# module at runtime.  Only the optical mapping, dimensions, and dataset contract
# are specialized here; camera and DMD trigger ordering remain unchanged.
core.DMD_WIDTH = DMD_WIDTH
core.DMD_HEIGHT = DMD_HEIGHT
core.INPUT_WIDTH = INPUT_WIDTH
core.INPUT_HEIGHT = INPUT_HEIGHT
core.INPUT_MACRO_PIXEL_SIZE = INPUT_MACRO_PIXEL_SIZE
core.HOLOGRAM_SUPERPIXEL_SIZE = HOLOGRAM_SUPERPIXEL_SIZE
core.ACTIVE_WIDTH = ACTIVE_WIDTH
core.ACTIVE_HEIGHT = ACTIVE_HEIGHT
core.ACTIVE_X = ACTIVE_X
core.ACTIVE_Y = ACTIVE_Y
core.active_region_mask = active_region_mask
core.get_superpixel_lut = get_superpixel_lut
core.encode_input_field_128 = input_field_to_dmd_pattern
core.PATTERN_128_CONFIG = PATTERN_128X96_CONFIG
core.get_active_128_pattern_config = get_active_128x96_pattern_config


_BaseDMDController = core.DMDController


class DMDController(_BaseDMDController):
    """Full-field controller using the aligned 128 x 96 source mapping."""

    def set_measurement_mode(self, test_mode):
        if test_mode:
            raise RuntimeError(
                "The 64-pattern test set is not defined for the 128 x 96 profile."
            )
        return super().set_measurement_mode(False)


core.DMDController = DMDController


class Application(core.Application):
    """128 x 128 camera GUI for the 128 x 96 full-field TM profile."""

    def __init__(self):
        super().__init__()
        self.chk_test_mode.state(["disabled"])
        self.chk_test_mode.configure(
            text="128x96 full-field 8N mode (64-pattern test unavailable)"
        )
        self._set_profile_title()

    def _set_profile_title(self):
        mode = f"{self.dmd_controller.full_probe_count:,}-pattern full measurement"
        self.title(
            "DMD TM Calibration - 128x96 -> 128x128 / full field / px=4 - "
            + mode
        )

    def refresh_measurement_mode_ui(self):
        super().refresh_measurement_mode_ui()
        self._set_profile_title()


if __name__ == "__main__":
    app = Application()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
