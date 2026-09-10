"""Gradient-based low-dimensional correction for a measured transmission matrix.

The first MVP only learns an input-side complex gain field represented by a small
2-D DCT basis:

    T_corrected = T0 @ diag(g(z))
    g(z) = exp(log_amp(z) + 1j * phase(z))

The measured camera intensity is used as the supervision target.  The real
optical system is therefore not differentiated through; gradients only pass
through the digital TM model.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn


def build_dct2_basis(
    input_shape: Tuple[int, int],
    mode_shape: Tuple[int, int],
    exclude_dc: bool = True,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return low-frequency separable DCT-II modes flattened as ``[Q, N]``.

    Each mode is normalized to unit RMS, so coefficients have a convenient
    interpretation: phase coefficients are roughly in radians and log-amplitude
    coefficients are roughly fractional gain values for small magnitudes.
    """
    height, width = (int(input_shape[0]), int(input_shape[1]))
    modes_y, modes_x = (int(mode_shape[0]), int(mode_shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError("input_shape must contain positive dimensions")
    if modes_y < 0 or modes_x < 0:
        raise ValueError("mode_shape cannot contain negative values")
    if modes_y == 0 or modes_x == 0:
        return torch.empty((0, height * width), dtype=dtype, device=device)

    y = torch.arange(height, dtype=dtype, device=device)
    x = torch.arange(width, dtype=dtype, device=device)
    basis = []
    for ky in range(modes_y):
        by = torch.cos(math.pi * (y + 0.5) * ky / height)
        for kx in range(modes_x):
            if exclude_dc and ky == 0 and kx == 0:
                continue
            bx = torch.cos(math.pi * (x + 0.5) * kx / width)
            mode = (by[:, None] * bx[None, :]).reshape(-1)
            rms = torch.sqrt(torch.mean(mode.square())).clamp_min(1e-12)
            basis.append(mode / rms)

    if not basis:
        return torch.empty((0, height * width), dtype=dtype, device=device)
    return torch.stack(basis, dim=0)


def pearson_corr_batch(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean per-sample Pearson correlation over flattened camera pixels."""
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have the same shape, got {} vs {}".format(
                tuple(prediction.shape), tuple(target.shape)
            )
        )
    prediction = prediction.reshape(prediction.shape[0], -1)
    target = target.reshape(target.shape[0], -1)

    prediction = prediction - prediction.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)

    numerator = (prediction * target).sum(dim=1)
    denominator = torch.sqrt(
        prediction.square().sum(dim=1) * target.square().sum(dim=1)
    ).clamp_min(eps)
    return (numerator / denominator).mean()


class InputDCTCorrection(nn.Module):
    """Input-side low-dimensional correction attached to a fixed complex TM."""

    def __init__(
        self,
        tm: torch.Tensor,
        input_shape: Tuple[int, int],
        phase_modes: Tuple[int, int] = (4, 4),
        amplitude_modes: Tuple[int, int] = (2, 2),
        max_log_amplitude: float = 0.7,
    ):
        super().__init__()
        if not torch.is_complex(tm):
            raise TypeError("tm must be a complex tensor")
        if tm.ndim != 2:
            raise ValueError("tm must be 2-D [N_output, N_input]")

        n_input = int(input_shape[0]) * int(input_shape[1])
        if tm.shape[1] != n_input:
            raise ValueError(
                "TM input dimension {} does not match input_shape {} (N={})".format(
                    tm.shape[1], input_shape, n_input
                )
            )

        self.input_shape = (int(input_shape[0]), int(input_shape[1]))
        self.max_log_amplitude = float(max_log_amplitude)
        self.register_buffer("tm", tm.to(torch.complex64))
        self.register_buffer(
            "phase_basis",
            build_dct2_basis(self.input_shape, phase_modes, exclude_dc=True),
        )
        self.register_buffer(
            "amplitude_basis",
            build_dct2_basis(self.input_shape, amplitude_modes, exclude_dc=True),
        )

        self.phase_coeff = nn.Parameter(torch.zeros(self.phase_basis.shape[0]))
        self.amplitude_coeff = nn.Parameter(
            torch.zeros(self.amplitude_basis.shape[0])
        )

    @property
    def parameter_count(self) -> int:
        return int(self.phase_coeff.numel() + self.amplitude_coeff.numel())

    def correction_maps(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``phase`` and ``log_amplitude`` maps flattened to ``[N]``."""
        n_input = self.tm.shape[1]
        if self.phase_basis.shape[0] == 0:
            phase = torch.zeros(
                n_input, dtype=torch.float32, device=self.tm.device
            )
        else:
            phase = self.phase_coeff @ self.phase_basis

        if self.amplitude_basis.shape[0] == 0:
            log_amplitude = torch.zeros(
                n_input, dtype=torch.float32, device=self.tm.device
            )
        else:
            log_amplitude = self.amplitude_coeff @ self.amplitude_basis

        if self.max_log_amplitude > 0:
            log_amplitude = log_amplitude.clamp(
                -self.max_log_amplitude, self.max_log_amplitude
            )
        return phase, log_amplitude

    def input_gain(self) -> torch.Tensor:
        """Return the learned complex multiplicative correction ``g``."""
        phase, log_amplitude = self.correction_maps()
        exponent = log_amplitude.to(torch.complex64) + 1j * phase.to(
            torch.complex64
        )
        return torch.exp(exponent)

    def forward_field(self, probes: torch.Tensor) -> torch.Tensor:
        """Predict output complex field for probes ``[B, N_input]``."""
        if probes.ndim != 2 or probes.shape[1] != self.tm.shape[1]:
            raise ValueError(
                "probes must have shape [B, {}], got {}".format(
                    self.tm.shape[1], tuple(probes.shape)
                )
            )
        probes = probes.to(torch.complex64)
        corrected = probes * self.input_gain()
        return corrected @ self.tm.transpose(0, 1)

    def forward(self, probes: torch.Tensor) -> torch.Tensor:
        """Predict camera intensity ``|T (g ⊙ x)|^2``."""
        field = self.forward_field(probes)
        return field.abs().square()


@dataclass
class FitResult:
    baseline_train_pcc: float
    baseline_val_pcc: float
    final_train_pcc: float
    final_val_pcc: float
    best_val_pcc: float
    history: np.ndarray


def _split_indices(
    sample_count: int,
    val_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if sample_count < 2:
        raise ValueError("at least two calibration samples are required")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    order = rng.permutation(sample_count)
    val_count = max(1, int(round(sample_count * val_fraction)))
    val_count = min(val_count, sample_count - 1)
    return order[val_count:], order[:val_count]


@torch.no_grad()
def _evaluate_pcc(
    model: InputDCTCorrection,
    probes: torch.Tensor,
    target: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
) -> float:
    values = []
    for start in range(0, len(indices), batch_size):
        batch_idx = torch.as_tensor(
            indices[start : start + batch_size],
            dtype=torch.long,
            device=probes.device,
        )
        values.append(
            pearson_corr_batch(
                model(probes.index_select(0, batch_idx)),
                target.index_select(0, batch_idx),
            ).item()
        )
    return float(np.mean(values))


def fit_input_dct_correction(
    tm: np.ndarray,
    probes: np.ndarray,
    measured_intensity: np.ndarray,
    input_shape: Tuple[int, int],
    phase_modes: Tuple[int, int] = (4, 4),
    amplitude_modes: Tuple[int, int] = (2, 2),
    epochs: int = 200,
    batch_size: int = 32,
    learning_rate: float = 3e-2,
    l2_weight: float = 1e-4,
    val_fraction: float = 0.2,
    seed: int = 0,
    device: str = "auto",
    log_every: int = 10,
) -> Tuple[InputDCTCorrection, FitResult]:
    """Fit the low-dimensional correction from measured camera intensities.

    Arrays are expected to be already restricted to the calibration probe subset
    and the desired output-pixel subset.
    """
    n_input = int(input_shape[0]) * int(input_shape[1])
    tm = np.asarray(tm)
    probes = np.asarray(probes)
    measured_intensity = np.asarray(measured_intensity)

    if tm.ndim != 2 or tm.shape[1] != n_input:
        raise ValueError(
            "tm must have shape [N_output, {}], got {}".format(
                n_input, tuple(tm.shape)
            )
        )
    probes = probes.reshape(probes.shape[0], -1)
    measured_intensity = measured_intensity.reshape(
        measured_intensity.shape[0], -1
    )
    if probes.shape[1] != n_input:
        raise ValueError(
            "probe input dimension {} does not match input_shape {} (N={})".format(
                probes.shape[1], input_shape, n_input
            )
        )
    if measured_intensity.shape[1] != tm.shape[0]:
        raise ValueError(
            "measurement output dimension {} does not match TM output {}".format(
                measured_intensity.shape[1], tm.shape[0]
            )
        )
    if probes.shape[0] != measured_intensity.shape[0]:
        raise ValueError("probes and measurements must have the same sample count")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    if device == "auto":
        resolved_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        resolved_device = torch.device(device)

    torch.manual_seed(seed)
    x = torch.from_numpy(
        np.asarray(probes, dtype=np.complex64)
    ).to(resolved_device)
    y = torch.from_numpy(
        np.asarray(measured_intensity, dtype=np.float32)
    ).to(resolved_device)
    tm_tensor = torch.from_numpy(
        np.asarray(tm, dtype=np.complex64)
    ).to(resolved_device)

    model = InputDCTCorrection(
        tm=tm_tensor,
        input_shape=input_shape,
        phase_modes=phase_modes,
        amplitude_modes=amplitude_modes,
    ).to(resolved_device)

    train_idx, val_idx = _split_indices(x.shape[0], val_fraction, seed)
    eval_batch_size = max(1, min(batch_size, len(train_idx), len(val_idx)))
    baseline_train = _evaluate_pcc(
        model, x, y, train_idx, eval_batch_size
    )
    baseline_val = _evaluate_pcc(model, x, y, val_idx, eval_batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed + 1)
    history = []
    best_val = baseline_val
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(1, int(epochs) + 1):
        model.train()
        shuffled = rng.permutation(train_idx)
        epoch_losses = []
        epoch_pcc = []

        for start in range(0, len(shuffled), batch_size):
            batch_np = shuffled[start : start + batch_size]
            batch_idx = torch.as_tensor(
                batch_np, dtype=torch.long, device=resolved_device
            )
            batch_x = x.index_select(0, batch_idx)
            batch_y = y.index_select(0, batch_idx)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            pcc = pearson_corr_batch(prediction, batch_y)
            regularizer = torch.zeros((), device=resolved_device)
            if model.phase_coeff.numel():
                regularizer = regularizer + model.phase_coeff.square().mean()
            if model.amplitude_coeff.numel():
                regularizer = (
                    regularizer + model.amplitude_coeff.square().mean()
                )
            loss = 1.0 - pcc + float(l2_weight) * regularizer
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu()))
            epoch_pcc.append(float(pcc.detach().cpu()))

        model.eval()
        val_pcc = _evaluate_pcc(
            model, x, y, val_idx, eval_batch_size
        )
        mean_loss = float(np.mean(epoch_losses))
        mean_train_pcc = float(np.mean(epoch_pcc))
        history.append((epoch, mean_loss, mean_train_pcc, val_pcc))

        if val_pcc > best_val:
            best_val = val_pcc
            best_state = {
                "phase_coeff": model.phase_coeff.detach().cpu().clone(),
                "amplitude_coeff": model.amplitude_coeff.detach().cpu().clone(),
            }

        if log_every and (epoch == 1 or epoch % log_every == 0 or epoch == epochs):
            print(
                "epoch {:4d} | loss {:.6f} | train PCC {:.5f} | val PCC {:.5f}".format(
                    epoch, mean_loss, mean_train_pcc, val_pcc
                )
            )

    if best_state is not None:
        with torch.no_grad():
            model.phase_coeff.copy_(
                best_state["phase_coeff"].to(resolved_device)
            )
            model.amplitude_coeff.copy_(
                best_state["amplitude_coeff"].to(resolved_device)
            )

    model.eval()
    final_train = _evaluate_pcc(
        model, x, y, train_idx, eval_batch_size
    )
    final_val = _evaluate_pcc(model, x, y, val_idx, eval_batch_size)

    return model, FitResult(
        baseline_train_pcc=baseline_train,
        baseline_val_pcc=baseline_val,
        final_train_pcc=final_train,
        final_val_pcc=final_val,
        best_val_pcc=best_val,
        history=np.asarray(history, dtype=np.float64),
    )


def export_corrected_tm(
    tm_path: str,
    gain: np.ndarray,
    output_path: str,
    layout: str = "output-input",
    chunk_size: int = 256,
) -> None:
    """Write a corrected full TM without materializing the full matrix in RAM."""
    source = np.load(tm_path, mmap_mode="r")
    gain = np.asarray(gain, dtype=np.complex64).reshape(-1)

    if layout == "output-input":
        if source.ndim != 2 or source.shape[1] != gain.size:
            raise ValueError("TM shape is incompatible with input gain")
        output = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=np.complex64, shape=source.shape
        )
        for start in range(0, source.shape[0], chunk_size):
            stop = min(start + chunk_size, source.shape[0])
            output[start:stop] = (
                np.asarray(source[start:stop], dtype=np.complex64)
                * gain[None, :]
            )
    elif layout == "input-output":
        if source.ndim != 2 or source.shape[0] != gain.size:
            raise ValueError("TM shape is incompatible with input gain")
        output = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=np.complex64, shape=source.shape
        )
        for start in range(0, source.shape[1], chunk_size):
            stop = min(start + chunk_size, source.shape[1])
            output[:, start:stop] = (
                np.asarray(source[:, start:stop], dtype=np.complex64)
                * gain[:, None]
            )
    else:
        raise ValueError("layout must be 'output-input' or 'input-output'")

    output.flush()
