"""Batched phase-retrieval algorithms for transmission-matrix recovery.

The solvers in this module operate on the amplitude model

    ``y = abs(X @ H.T)``

where ``X`` is the complex probe matrix with shape ``(M, N)`` and ``H`` is
returned with shape ``(K, N)`` for ``K`` camera pixels.  Camera intensities
must be converted to amplitudes by the caller.

The implementation is intentionally hardware independent.  The 32 x 24
profile has only 768 input modes, so the Gram eigendecomposition is a useful
shared precomputation and keeps all algorithm variants comparable.
"""

import math
from typing import Callable, Optional, Tuple

import torch


# Keep the short names stable: these are also used by the legacy Tk UI.
ALGORITHMS = (
    "GGS21",
    "GS",
    "RAF21",
    "RAF",
    "AF",
    "TAF",
    "WF",
    "prVBEM",
    "prVAMP",
)


def canonical_algorithm(name: str) -> str:
    """Resolve case-, dash-, and underscore-insensitive algorithm names."""

    key = str(name).lower().replace("-", "").replace("_", "")
    # The paper/project shorthand is commonly written prVAM; the actual
    # algorithm is the complex VAMP variant and is exposed as prVAMP here.
    if key == "prvam":
        key = "prvamp"
    for algorithm in ALGORITHMS:
        if algorithm.lower() == key:
            return algorithm
    raise ValueError("Unknown recovery algorithm: {}".format(name))


def phase(z: torch.Tensor) -> torch.Tensor:
    """Return unit-magnitude complex phases, including a safe zero phase."""

    return torch.polar(torch.ones_like(z.real), torch.angle(z))


def bessel_ratio(kappa: torch.Tensor) -> torch.Tensor:
    """Stable ``I1(kappa) / I0(kappa)`` for the von-Mises amplitude channel."""

    # The exponentially scaled Bessel functions avoid overflow for the large
    # concentration values that occur late in prVAMP/prVBEM iterations.
    return torch.special.i1e(kappa) / torch.special.i0e(kappa).clamp_min(1e-30)


ProgressCallback = Callable[[int, int], None]


class RecoverySolver:
    """Prepare shared linear operators and solve output pixels in a batch.

    ``prVAMP`` is a damped complex GLM-VAMP update with a Gaussian prior and
    exact-magnitude likelihood.  ``prVBEM`` uses sequential mean-field
    coordinate updates and estimates an effective detector noise variance.
    The remaining methods are the standard GS/GGS or amplitude-flow family.

    Parameters are deliberately modest for the 32 x 24 optical grid.  A
    caller can use ``power_iterations=0`` for a quick smoke test, although a
    positive value is recommended for real recovery.
    """

    def __init__(
        self,
        X: torch.Tensor,
        algorithm: str,
        iterations: int = 200,
        ratio: Optional[float] = None,
        step: Optional[float] = None,
        damping: float = 0.5,
        power_iterations: int = 30,
    ) -> None:
        self.algorithm = canonical_algorithm(algorithm)
        if (
            X.ndim != 2
            or min(X.shape) == 0
            or not torch.isfinite(X).all()
        ):
            raise ValueError("X must be a finite nonempty matrix")
        if int(iterations) < 1 or int(power_iterations) < 1:
            raise ValueError("Iteration counts must be positive")

        if ratio is None:
            ratio = 2.0 / 3.0 if self.algorithm == "RAF21" else 0.89
        if not (0.0 <= float(ratio) <= 1.0) or not (
            0.0 < float(damping) <= 1.0
        ):
            raise ValueError("ratio must be in [0,1], damping in (0,1]")
        if step is not None and (
            not math.isfinite(float(step)) or float(step) <= 0.0
        ):
            raise ValueError("step must be finite and positive")

        self.iterations = int(iterations)
        self.ratio = float(ratio)
        self.damping = float(damping)
        self.power_iterations = int(power_iterations)
        self.step = None if step is None else float(step)

        # Normalize the probe matrix once.  The positive scale is restored on
        # return, so this does not alter recovered phase or relative amplitude.
        self.x_scale = X.abs().square().mean().sqrt()
        if float(self.x_scale) <= 0.0:
            raise ValueError("Probe matrix is all zero")
        self.X = X.to(torch.complex64) / self.x_scale

        # The Gram matrix is small enough for the 768-mode profile and its
        # eigensystem lets GS/prVBEM use a stable regularized inverse while
        # prVAMP reuses the same right singular basis.
        self.gram = self.X.mH @ self.X
        self.evals, self.evecs = torch.linalg.eigh(self.gram)
        self.evals = self.evals.clamp_min(0.0)
        self.lipschitz = float(self.evals[-1]) / float(self.X.shape[0])
        self.lipschitz = max(self.lipschitz, torch.finfo(torch.float32).eps)

        if self.algorithm in ("GS", "GGS21", "prVBEM"):
            cutoff = self.evals[-1] * max(self.X.shape) * torch.finfo(
                torch.float32
            ).eps
            inv = torch.where(
                self.evals > cutoff,
                1.0 / self.evals.clamp_min(1e-30),
                torch.zeros_like(self.evals),
            )
            self.pinv = (self.evecs * inv) @ self.evecs.mH @ self.X.mH

    @torch.inference_mode()
    def solve(
        self,
        amplitude: torch.Tensor,
        seed: int = 24032,
        progress: Optional[ProgressCallback] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recover a batch of output rows and return ``(H, errors)``.

        ``amplitude`` has shape ``(M, K)``.  ``errors`` is a float tensor of
        length ``iterations`` containing the mean per-output amplitude error
        in the original, unnormalized measurement scale.
        """

        X = self.X
        y = torch.as_tensor(amplitude, dtype=torch.float32, device=X.device)
        if y.ndim != 2 or y.shape[0] != X.shape[0] or y.shape[1] == 0:
            raise ValueError("Amplitude must have shape (probe_count, output_count)")
        if not torch.isfinite(y).all() or (y < 0).any():
            raise ValueError("Amplitudes must be finite and nonnegative")

        scale = y.square().mean(0).sqrt().clamp_min(1e-12)
        original_y = y
        y = y / scale
        m, n = X.shape
        block_width = y.shape[1]
        generator = torch.Generator(device=X.device).manual_seed(int(seed))

        h = torch.complex(
            torch.randn(n, block_width, device=X.device, generator=generator),
            torch.randn(n, block_width, device=X.device, generator=generator),
        )

        # Spectral-style initialization.  The 77th percentile is a robust
        # high-intensity mask and avoids starting every pixel from pure noise.
        weights = (y * (y >= torch.quantile(y, 0.7692307692307692, dim=0))).sqrt()
        for _ in range(self.power_iterations):
            h = X.mH @ (weights * (X @ h))
            h = h / torch.linalg.vector_norm(h, dim=0).clamp_min(1e-12)
        h *= y.square().mean(0).sqrt() / (
            (X @ h).abs().square().mean(0).sqrt().clamp_min(1e-12)
        )

        algorithm = self.algorithm
        if algorithm == "prVBEM":
            prior = (self.pinv @ y.to(X.dtype)).abs().square().amax(0).clamp_min(
                1e-4
            )
            noise = torch.full((block_width,), 1e-3, device=X.device)
            diag = self.gram.diagonal().real
            variance = prior[:, None] * noise[:, None] / (
                noise[:, None] + diag[None, :] * prior[:, None]
            )
            ybar = y * phase(X @ h)
        elif algorithm == "prVAMP":
            p = X @ h
            tau = torch.full((block_width,), 10.0, device=X.device)
            prior_precision = float(n)

        errors = []
        switch = math.ceil(self.ratio * self.iterations)
        for iteration in range(self.iterations):
            z = X @ h
            magnitude = z.abs()

            if algorithm in ("GS", "GGS21"):
                target = (
                    y.square()
                    if algorithm == "GGS21" and iteration < switch
                    else y
                )
                h = self.pinv @ (target * phase(z))
            elif algorithm == "prVBEM":
                # EM update for effective additive complex detector noise.
                if n > 128:
                    # A literal coordinate sweep is useful as a reference
                    # implementation, but it launches O(N) tiny operations
                    # per iteration.  The 768-mode profile uses the
                    # equivalent parallel variational relaxation, which
                    # keeps the same EM magnitude channel and scales like
                    # the other batched methods.
                    noise = (
                        magnitude.square()
                        + y.square()
                        - 2.0 * (ybar.conj() * z).real
                    ).mean(0).clamp_min(1e-4)
                    ybar = y * phase(z) * bessel_ratio(
                        2.0 * y * magnitude / noise
                    )
                    gradient = X.mH @ (z - ybar) / m
                    step = (
                        self.step
                        if self.step is not None
                        else 0.8 / self.lipschitz
                    )
                    h = (1.0 - self.damping) * h + self.damping * (
                        h - step * gradient
                    )
                else:
                    noise = (
                        (
                            magnitude.square()
                            + y.square()
                            - 2.0 * (ybar.conj() * z).real
                        ).sum(0)
                        + (diag[None, :] * variance).sum(1)
                    ).div(m).clamp_min(1e-8)
                    rhs = X.mH @ ybar
                    # Sequential mean-field coordinate update.  Small
                    # matrices retain this direct VBEM reference path.
                    for k in range(n):
                        residual = rhs[k] - self.gram[k] @ h + diag[k] * h[k]
                        h[k] = prior * residual / (noise + prior * diag[k])
                    variance = prior[:, None] * noise[:, None] / (
                        noise[:, None] + diag[None, :] * prior[:, None]
                    )
                    z = X @ h
                    ybar = y * phase(z) * bessel_ratio(
                        2.0 * y * z.abs() / noise
                    )
            elif algorithm == "prVAMP":
                # Nonlinear magnitude-channel denoiser followed by the exact
                # LMMSE/VAMP linear module in the Gram eigensystem.
                zhat = y * phase(p) * bessel_ratio(
                    2.0 * tau * y * p.abs()
                )
                vz = (y.square() - zhat.abs().square()).mean(0).clamp_min(1e-8)
                alpha = (tau * vz).clamp(1e-4, 0.99)
                gamma = (tau * (1.0 - alpha) / alpha).clamp(1e-5, 1e5)
                r = (zhat - alpha * p) / (1.0 - alpha)
                denom = prior_precision + self.evals[None, :] * gamma[:, None]
                h = self.evecs @ (
                    self.evecs.mH @ (X.mH @ r) * gamma[None, :] / denom.T
                )
                zlin = X @ h
                vlin = (
                    self.evals[:, None] / denom.T
                ).sum(0).div(m).clamp_min(1e-10)
                alpha_lin = (gamma * vlin).clamp(1e-6, 0.99)
                p_new = (zlin - alpha_lin * r) / (1.0 - alpha_lin)
                tau_new = (
                    gamma * (1.0 - alpha_lin) / alpha_lin
                ).clamp(1e-5, 1e5)
                blended_tau = (
                    (1.0 - self.damping) * tau + self.damping * tau_new
                )
                p = (
                    (1.0 - self.damping) * tau * p
                    + self.damping * tau_new * p_new
                ) / blended_tau
                tau = blended_tau
            else:
                # Amplitude-flow family.  RAF21 uses the same 2-to-1 target
                # continuation as GGS21, while RAF adds its robust residual
                # weighting and AF/TAF/WF provide the common baselines.
                target = (
                    y.square()
                    if algorithm == "RAF21" and iteration < switch
                    else y
                )
                residual = z - target * phase(z)
                if algorithm in ("RAF", "RAF21"):
                    residual *= magnitude / (
                        magnitude + 5.0 * y
                    ).clamp_min(1e-12)
                elif algorithm == "TAF":
                    residual *= magnitude >= y / 1.7
                elif algorithm == "WF":
                    residual = (magnitude.square() - y.square()) * z

                gradient = X.mH @ residual / m
                default_step = (
                    2.0 if algorithm in ("RAF", "RAF21") else 0.8
                ) / self.lipschitz
                if algorithm == "WF":
                    default_step = 0.15 / self.lipschitz
                step = self.step if self.step is not None else default_step
                h = h - step * gradient

            predicted_error = torch.linalg.vector_norm(
                (X @ h).abs() * scale - original_y, dim=0
            )
            if not torch.isfinite(predicted_error).all():
                raise RuntimeError(
                    "{} diverged at iteration {}; reduce step/damping".format(
                        algorithm, iteration + 1
                    )
                )
            errors.append(predicted_error.mean())
            if progress and (
                (iteration + 1) % 10 == 0
                or iteration + 1 == self.iterations
            ):
                progress(iteration + 1, self.iterations)

        return (
            (h * (scale / self.x_scale)).T.contiguous(),
            torch.stack(errors),
        )
