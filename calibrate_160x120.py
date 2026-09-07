"""5x TM calibration: 160 x 120 logical input, 128 x 128 camera output.

This entry point specializes the established 128-output acquisition chain.
The logical source is expanded to 256 x 192 optical superpixels and encoded
with 4 x 4 holo_SP tiles across the complete 1024 x 768 DMD.
"""

import os

import numpy as np

import calibrate_128x128 as core
from dmd_pattern_160x120 import (
    ACTIVE_HEIGHT,
    ACTIVE_WIDTH,
    ACTIVE_X,
    ACTIVE_Y,
    DMD_HEIGHT,
    DMD_WIDTH,
    EXPANDED_HEIGHT,
    EXPANDED_WIDTH,
    HOLOGRAM_SUPERPIXEL_SIZE,
    INPUT_HEIGHT,
    INPUT_MACRO_PIXEL_SIZE,
    INPUT_WIDTH,
    MAPPING_VERSION,
    SOURCE_X_INDICES,
    SOURCE_Y_INDICES,
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
            "reconstruction_probe_storage": "planar_complex32",
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


def get_active_160x120_pattern_config(output_tag=None):
    """Return and validate the selected 160 x 120 pattern dataset."""
    # Match the shared get_active_128_pattern_config(output_tag=None)
    # provider signature. Full-field profiles are not channelized.
    del output_tag
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
        # This profile supplies a CUDA encoder for its non-integer nearest-fill
        # expansion instead of using the base class's direct-grid encoder.
        self.focus_encoding_use_gpu = True

    def _build_focus_hologram_batch(
        self,
        tm_rows,
        px=HOLOGRAM_SUPERPIXEL_SIZE,
        ds_method="mean",
        lut_cache=None,
        encode_chunk_size=32,
        progress_callback=None,
    ):
        """Encode 160 x 120 TM rows with the nearest-fill CUDA mapping."""
        gpu_enabled = bool(
            getattr(self, "focus_encoding_use_gpu", True)
        ) and not bool(getattr(self, "_focus_gpu_failed", False))
        supported_mapping = (
            ds_method == "mean"
            and int(px) == HOLOGRAM_SUPERPIXEL_SIZE
            and self.dmd_height == INPUT_HEIGHT
            and self.dmd_width == INPUT_WIDTH
            and self.active_height == EXPANDED_HEIGHT * int(px)
            and self.active_width == EXPANDED_WIDTH * int(px)
        )

        if (
            not supported_mapping
            or not gpu_enabled
            or not core.torch.cuda.is_available()
        ):
            return super()._build_focus_hologram_batch(
                tm_rows,
                px=px,
                ds_method=ds_method,
                lut_cache=lut_cache,
                encode_chunk_size=encode_chunk_size,
                progress_callback=progress_callback,
            )

        tm_rows = np.asarray(tm_rows)
        expected_input_count = self.dmd_height * self.dmd_width
        if tm_rows.ndim != 2 or tm_rows.shape[1] != expected_input_count:
            raise ValueError(
                "TM batch shape {} does not match (batch, {})".format(
                    tm_rows.shape,
                    expected_input_count,
                )
            )

        if lut_cache is None:
            lut_cache = get_superpixel_lut(px)
        if not isinstance(lut_cache, tuple) or len(lut_cache) != 3:
            raise RuntimeError("Failed to generate the superpixel LUT")
        _, pixel_combinations, lut = lut_cache
        combination_length = int(len(pixel_combinations[0]))
        n_sp = int(round(combination_length**0.5))
        if n_sp * n_sp != combination_length or n_sp != int(px):
            return super()._build_focus_hologram_batch(
                tm_rows,
                px=px,
                ds_method=ds_method,
                lut_cache=lut_cache,
                encode_chunk_size=encode_chunk_size,
                progress_callback=progress_callback,
            )

        device_index = int(getattr(self, "focus_encoding_gpu_device", 0))
        gpu_chunk_size = max(
            1,
            int(
                getattr(
                    self,
                    "focus_encoding_gpu_chunk_size",
                    encode_chunk_size,
                )
            ),
        )
        self.last_focus_encoding_backend = (
            "GPU cuda:{} (160x120 nearest-fill)".format(device_index)
        )
        self.last_focus_encoding_fallback_error = None
        try:
            return self._build_focus_hologram_batch_gpu_160x120(
                tm_rows,
                pixel_combinations=pixel_combinations,
                lut=lut,
                n_sp=n_sp,
                encode_chunk_size=gpu_chunk_size,
                progress_callback=progress_callback,
                device_index=device_index,
            )
        except Exception as exc:
            self.last_focus_encoding_fallback_error = str(exc)
            self._focus_gpu_failed = True
            print(
                "160x120 GPU focus encoding failed; falling back to NumPy: "
                "{}".format(exc)
            )
            try:
                core.torch.cuda.empty_cache()
            except Exception:
                pass
            return super()._build_focus_hologram_batch(
                tm_rows,
                px=px,
                ds_method=ds_method,
                lut_cache=lut_cache,
                encode_chunk_size=encode_chunk_size,
                progress_callback=progress_callback,
            )

    def _build_focus_hologram_batch_gpu_160x120(
        self,
        tm_rows,
        pixel_combinations,
        lut,
        n_sp,
        encode_chunk_size=32,
        progress_callback=None,
        device_index=0,
    ):
        """CUDA encoder for the 160 x 120 to 256 x 192 nearest-fill map."""
        tm_rows = np.asarray(tm_rows)
        batch_count = int(tm_rows.shape[0])
        patterns = np.zeros(
            (batch_count, self.original_height, self.original_width),
            dtype=np.uint8,
        )
        errors = [None] * batch_count
        if batch_count == 0:
            return patterns, errors

        device_index = int(device_index)
        if not 0 <= device_index < core.torch.cuda.device_count():
            raise ValueError(
                "CUDA device {} is unavailable".format(device_index)
            )
        device = core.torch.device("cuda:{}".format(device_index))
        encode_chunk_size = max(1, int(encode_chunk_size))

        tensor_cache = getattr(self, "_focus_gpu_tensor_cache", None)
        if tensor_cache is None:
            tensor_cache = {}
            self._focus_gpu_tensor_cache = tensor_cache
        cache_key = (
            "160x120_nearest_fill",
            device_index,
            int(n_sp),
            INPUT_HEIGHT,
            INPUT_WIDTH,
            EXPANDED_HEIGHT,
            EXPANDED_WIDTH,
        )
        cached = tensor_cache.get(cache_key)
        if cached is None:
            combinations_tensor = core.torch.as_tensor(
                np.asarray(pixel_combinations, dtype=np.uint8),
                device=device,
                dtype=core.torch.uint8,
            )
            lut_tensor = core.torch.as_tensor(
                np.asarray(lut),
                device=device,
                dtype=core.torch.long,
            )
            source_y_indices = core.torch.as_tensor(
                SOURCE_Y_INDICES,
                device=device,
                dtype=core.torch.long,
            )
            source_x_indices = core.torch.as_tensor(
                SOURCE_X_INDICES,
                device=device,
                dtype=core.torch.long,
            )
            row_shifts = (
                n_sp
                * core.torch.arange(
                    EXPANDED_HEIGHT,
                    device=device,
                    dtype=core.torch.long,
                )
            ) % (n_sp**2)
            roll_indices = (
                core.torch.arange(
                    n_sp**2,
                    device=device,
                    dtype=core.torch.long,
                )[None, :]
                + row_shifts[:, None]
            ) % (n_sp**2)
            roll_indices = roll_indices[None, :, None, :]
            cached = (
                combinations_tensor,
                lut_tensor,
                source_y_indices,
                source_x_indices,
                roll_indices,
            )
            tensor_cache[cache_key] = cached
        (
            combinations_tensor,
            lut_tensor,
            source_y_indices,
            source_x_indices,
            roll_indices,
        ) = cached
        lut_zero = int(len(lut) // 2)

        with core.torch.no_grad():
            for chunk_start in range(0, batch_count, encode_chunk_size):
                chunk_end = min(
                    batch_count,
                    chunk_start + encode_chunk_size,
                )
                chunk_rows = np.asarray(
                    tm_rows[chunk_start:chunk_end],
                    dtype=np.complex64,
                )
                finite_mask = np.all(np.isfinite(chunk_rows), axis=1)
                amplitudes = np.max(np.abs(chunk_rows), axis=1)
                valid_mask = finite_mask & (amplitudes > 0)
                for local_index in np.flatnonzero(~valid_mask):
                    global_index = chunk_start + int(local_index)
                    errors[global_index] = (
                        "TM row contains NaN or infinity"
                        if not finite_mask[local_index]
                        else "Cannot encode an all-zero complex field"
                    )

                valid_local_indices = np.flatnonzero(valid_mask)
                if valid_local_indices.size:
                    rows_tensor = core.torch.as_tensor(
                        np.ascontiguousarray(
                            chunk_rows[valid_local_indices]
                        ),
                        device=device,
                    )
                    phases = core.torch.angle(rows_tensor)
                    fields = core.torch.polar(
                        core.torch.ones_like(phases),
                        -phases,
                    ).reshape(
                        -1,
                        INPUT_HEIGHT,
                        INPUT_WIDTH,
                    )
                    expanded_fields = fields.index_select(
                        1,
                        source_y_indices,
                    ).index_select(
                        2,
                        source_x_indices,
                    )
                    field_max = core.torch.amax(
                        core.torch.abs(expanded_fields),
                        dim=(1, 2),
                    )
                    expanded_fields /= field_max[:, None, None]

                    # Match the reference holo_SP accumulation order after its
                    # 4 x 4 nearest-neighbour physical expansion.
                    downsampled = core.torch.zeros_like(expanded_fields)
                    for _ in range(n_sp**2):
                        downsampled += expanded_fields
                    downsampled /= n_sp**2
                    downsampled_max = core.torch.amax(
                        core.torch.abs(downsampled),
                        dim=(1, 2),
                    )
                    scaled = downsampled / (
                        downsampled_max[:, None, None] * 0.01
                    )
                    real_index = (
                        core.torch.round(scaled.real).to(core.torch.long)
                        + lut_zero
                    )
                    imag_index = (
                        core.torch.round(scaled.imag).to(core.torch.long)
                        + lut_zero
                    )
                    selected = combinations_tensor[
                        lut_tensor[real_index, imag_index]
                    ]
                    expanded_roll_indices = roll_indices.expand(
                        len(valid_local_indices),
                        EXPANDED_HEIGHT,
                        EXPANDED_WIDTH,
                        n_sp**2,
                    )
                    rolled = core.torch.gather(
                        selected,
                        3,
                        expanded_roll_indices,
                    )
                    holograms = (
                        rolled.reshape(
                            len(valid_local_indices),
                            EXPANDED_HEIGHT,
                            EXPANDED_WIDTH,
                            n_sp,
                            n_sp,
                        )
                        .permute(0, 1, 4, 2, 3)
                        .reshape(
                            len(valid_local_indices),
                            self.original_height,
                            self.original_width,
                        )
                        * 255
                    )
                    global_indices = chunk_start + valid_local_indices
                    patterns[global_indices] = holograms.cpu().numpy()

                if progress_callback:
                    progress_callback(chunk_end, batch_count)

        return patterns, errors

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
