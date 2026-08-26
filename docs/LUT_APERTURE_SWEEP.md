# LUT / Fourier aperture sweep

This diagnostic isolates the DMD encoding and Fourier-plane aperture from the
scattering medium. It is intended to answer whether the same physical aperture
can recover the commanded complex field equally well for the 32x24 repeated
4x4 encoding and the 160x120 nearest-fill encoding.

The simulation uses the repository's current `holo_SP` carrier:

- `fx = 1/4` cycles per DMD mirror
- `fy = 1/16` cycles per DMD mirror
- direct first order for NumPy FFT sign convention: `order_sign=-1`

On the full 768x1024 DMD FFT this direct order is centred at `(y, x) = (336,
256)`. The opposite conjugate order is `(432, 768)`.

## Run

Start from an existing amplitude+phase correction dataset:

```powershell
python -m tools.simulate_lut_aperture_sweep `
  --pattern-dir correction_patterns_32x24_complex `
  --sample-count 8
```

By default the script also synthesizes 8 matched 160x120 random
amplitude+phase fields using the current `dmd_pattern_160x120.py` encoder. Use
`--no-compare-160` for a faster 32x24-only run.

The default radii are:

```text
16 24 32 48 64 80 96 112 128 144 160 176 192 208 224
```

The radius is expressed in x-axis FFT bins of the full 1024-pixel DMD width.
Because the physical DMD is 1024x768 with equal mirror pitch, a circular
physical-frequency aperture with x-radius `rx` has y-radius `0.75 * rx` FFT
bins. The code applies this rectangular-grid correction; it does not use a
naive pixel-space circle.

Custom example:

```powershell
python -m tools.simulate_lut_aperture_sweep `
  --pattern-dir correction_patterns_32x24_complex `
  --sample-count 16 `
  --radii 32 48 64 80 96 112 128 144 160 176 192 208
```

## Outputs

The default `lut_aperture_sweep/` directory contains:

- `aperture_sweep.csv`: recovery metrics for every profile/radius
- `lut_point_errors.csv`: ideal 4x4 LUT nearest-point quantization error
- `fidelity_vs_radius.png`: complex-field fidelity after one global complex-gain alignment
- `phase_rmse_vs_radius.png`: logical-field phase RMSE
- `amplitude_nrmse_vs_radius.png`: logical-field amplitude NRMSE
- `spectrum_32x24.png`: average 32x24 DMD Fourier power with representative apertures
- `spectrum_160x120.png`: matched 160x120 Fourier power
- `summary.json`: carrier geometry, LUT error, and coarse spectral peaks

## Interpretation

Three regimes are expected in a useful sweep:

1. **Aperture too small**: target first-order bandwidth is clipped. Fidelity is
   low and phase/amplitude errors are high.
2. **Useful aperture window**: the target order is passed while nearby unwanted
   spectrum remains rejected. Complex fidelity reaches a plateau/maximum.
3. **Aperture too large**: unwanted diffraction content enters the passband.
   Fidelity falls again even though passed optical energy increases.

The 32x24 and 160x120 curves should be compared at the **same radius**, because
both patterns use the same physical DMD and carrier. If the radius range that is
good for 160x120 is already on the falling/leakage side for 32x24, that supports
the hypothesis that the current physical aperture is oversized for the 32x24
encoding.

This simulation cannot determine the actual physical aperture diameter without
experimental Fourier-plane scale information (lens focal length, wavelength,
DMD mirror pitch/magnification, and the real iris diameter). It is therefore a
relative diagnostic first; physical units can be added once those values are
known.
