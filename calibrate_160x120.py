"""5x TM calibration: 160 x 120 logical input, 128 x 128 camera output.

This entry point specializes the established 128-output acquisition chain.
The logical source is expanded to 256 x 192 optical superpixels and encoded
with 4 x 4 holo_SP tiles across the complete 1024 x 768 DMD.
"""

import os

import calibrate_128x128 as core
from dmd_pattern_160x120 import (
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


PATTERN_160X120_CONFIG = {
    "active": "8N",
    "sets": {
        "8N": {
            "directory": "pregenerated_patterns_160x120_fill_8N_full",
            "probe_multiplier": 8,
            "reconstruction_output_chunk_size": 256,
            "reconstruction_solver": "cholesky",
            "measurement_filename": "measurements_160x120_to_128x128_fill_8N_memmap.npy",
            "tm_memmap_filename": "transmission_matrix_160x120_to_128x128_fill_8N_memmap.npy",
            "reconstructed_filename": "reconstructed_field_160x120_to_128x128_fill_8N.npy",
            "error_curve_filename": "ggs21_error_curve_160x120_to_128x128_fill_8N.npy",
            "cholesky_cache_filename": "probe_cholesky_160x120_fill_8N.npy",
            "pinv_real_filename": "probe_pinv_160x120_fill_8N_fp16_real.npy",
            "pinv_imag_filename": "probe_pinv_160x120_fill_8N_fp16_imag.npy",
            "pinv_metadata_filename": "probe_pinv_160x120_fill_8N_fp16.json",
            "reconstruction_metadata_filename": "tm_reconstruction_160x120_to_128x128_fill_8N.json",
            "mapping_version": MAPPING_VERSION,
        },
    },
}


def get_active_160x120_pattern_config():
    """Return and validate the selected 160 x 120 pattern dataset."""
    active = PATTERN_160X120_CONFIG.get("active")
    datasets = PATTERN_160X120_CONFIG.get("sets", {})
    if active not in datasets:
        available = ", ".join(sorted(datasets)) or "<none>"
        raise ValueError(
            f"Unknown 160 x 120 pattern dataset {active!r}; available: {available}"
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


# The hardware acquisition implementation resolves these symbols from the
# core module at runtime.  Specializing them here preserves its tested trigger
# and camera sequence while changing only the optical mapping and dimensions.
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
core.PATTERN_128_CONFIG = PATTERN_160X120_CONFIG
core.get_active_128_pattern_config = get_active_160x120_pattern_config


_BaseDMDController = core.DMDController


class DMDController(_BaseDMDController):
    """Full-field controller using the 160 x 120 source mapping."""

    def __init__(self, camera_handler=None):
        super().__init__(camera_handler=camera_handler)
        # The inherited CUDA encoder assumes a one-to-one rectangular logical
        # grid.  Use the exact CPU encoder for the non-integer 1.6x expansion.
        self.focus_encoding_use_gpu = False

    def set_measurement_mode(self, test_mode):
        if test_mode:
            raise RuntimeError(
                "The 64-pattern test set is not defined for the 160 x 120 profile."
            )
        return super().set_measurement_mode(False)


core.DMDController = DMDController


class Application(core.Application):
    """128 x 128 camera GUI for the 160 x 120 full-field TM profile."""

    def __init__(self):
        super().__init__()
        self.chk_test_mode.state(["disabled"])
        self.chk_test_mode.configure(
            text="160x120 full-field 8N mode (64-pattern test unavailable)"
        )
        self._set_profile_title()

    def _set_profile_title(self):
        mode = f"{self.dmd_controller.full_probe_count:,}-pattern full measurement"
        self.title(f"DMD TM Calibration - 160x120 -> 128x128 / full field / px=4 - {mode}")

    def refresh_measurement_mode_ui(self):
        super().refresh_measurement_mode_ui()
        self._set_profile_title()


if __name__ == "__main__":
    app = Application()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
