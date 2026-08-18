"""V4 ordinary-grid application with Cholesky-based GGS21 recovery.

This variant intentionally reuses the camera, DMD, GUI, acquisition, and
focusing code from ``calibrate_v4_32x24.py``. Only the local transmission-matrix
recovery is replaced.  Results use variant-specific file names so a recovery
from the ordinary pseudoinverse version is never overwritten.
"""

import os

import numpy as np
import torch

from calibrate_v4_32x24 import Application as _V4Application
from calibrate_v4_32x24 import DMDController as _V4DMDController


class DMDController(_V4DMDController):
    """V4 controller whose local GGS21 projection uses Cholesky solves."""

    def __init__(self, camera_handler=None):
        super().__init__(camera_handler)

        # Do not overwrite the ordinary pinv-based recovery.  The inherited
        # focusing methods read ``self.reconstructed_filename``.
        self.tm_memmap_filename = "transmission_matrix_memmap_cholesky.npy"
        self.reconstructed_filename = "reconstructed_field_cholesky.npy"
        self.error_curve_filename = "ggs21_error_curve_cholesky.npy"
        self.ggs21_cholesky_factor_filename = "probe_cholesky_v4.npy"

        # Zero ridge makes this a controlled factorization-only comparison
        # against pinv(X). Increase this value only if the Gram matrix is not
        # numerically positive definite for a different probe set.
        self.ggs21_ridge = 0.0

    def _prepare_ggs21_linear_solver(self, X, base_dir, probe_file):
        """Factor X.H @ X once and retain L for every GGS21 iteration."""
        del probe_file
        print("Building X.H @ X for Cholesky GGS21...")
        gram = X.mH @ X
        ridge = float(self.ggs21_ridge)
        if ridge < 0:
            raise ValueError("ggs21_ridge must be non-negative")
        if ridge:
            gram.diagonal().add_(ridge)
        try:
            factor = torch.linalg.cholesky(gram)
        except RuntimeError as exc:
            raise RuntimeError(
                "Cholesky factorization failed. The probe Gram matrix may be "
                "rank deficient; set ggs21_ridge to a small positive value."
            ) from exc
        del gram

        factor_path = os.path.join(
            base_dir,
            self.ggs21_cholesky_factor_filename,
        )
        np.save(
            factor_path,
            factor.detach().cpu().numpy().astype(np.complex64, copy=False),
        )
        print("Cholesky factor saved to:", factor_path)
        return factor

    def _apply_ggs21_linear_solver(self, X, solver_state, detector_field):
        """Apply (X.H X)^-1 X.H through two triangular Cholesky solves."""
        rhs = X.mH @ detector_field
        return torch.cholesky_solve(rhs, solver_state)


class Application(_V4Application):
    dmd_controller_class = DMDController

    def __init__(self):
        super().__init__()
        self.title("DMD Phase Focusing Optimization System - Cholesky GGS21")
        self.btn_tm_recovery.config(text="Recover TM (Cholesky GGS21)")
        # The inherited remote action runs a different server-side algorithm,
        # so keep it unavailable in this explicitly local Cholesky variant.
        self.btn_tm_remote.config(
            text="Remote TM (not Cholesky)",
            state="disabled",
        )


if __name__ == "__main__":
    app = Application()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
