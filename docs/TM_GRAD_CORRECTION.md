# Gradient TM correction (MVP)

This tool fits a small input-side correction on top of an existing reconstructed
transmission matrix. It is intended as the first differentiable baseline before
trying hardware-in-the-loop RL.

## Model

The reconstructed TM is fixed:

\[
T_0 \in \mathbb{C}^{M\times N}.
\]

The learnable correction is a complex gain on the input modes:

\[
T_{\mathrm{corr}} = T_0 D_{\mathrm{in}}(z),
\qquad
D_{\mathrm{in}}=\operatorname{diag}(g).
\]

The gain is represented with a low-frequency 2-D DCT basis:

\[
g=\exp(\ell(z_a)+i\phi(z_\phi)).
\]

For the default `4x4` phase basis and `2x2` amplitude basis the DC term is
removed, so only 18 real parameters are optimized:

- phase: 15 coefficients
- log-amplitude: 3 coefficients

Camera prediction is

\[
I_{\mathrm{pred}} = |T_0(g\odot x)|^2.
\]

The real camera image is only a target. No gradient is propagated through the
physical optical system. Adam minimizes `1 - PCC` between predicted and
measured intensity plus a small coefficient L2 penalty.

This MVP deliberately does **not** learn a full residual matrix or low-rank
`U diag(c) V^H` term yet. First use it to test whether the remaining TM error is
well described by a low-dimensional input calibration field.

## Required data

You need three array files with matching sample order:

1. reconstructed complex TM
2. complex input probes
3. real measured camera intensities for those probes; this may be either a
   standard `.npy` array or TMCalib's headerless `uint16` measurement memmap
   (commonly named `measurements_memmap.npy` despite having no NPY header)

For `v4_32x24`:

- probe sample shape: `32x24` (stored as `(24, 32)`)
- camera sample shape: `128x128`
- TM convention: `[N_output, N_input]`

The calibration probes used to reconstruct `T0` are valid as a smoke test, but
for a meaningful correction/validation experiment use a separate random
amplitude+phase dataset. Reusing the GGS21 phase-only probes mostly tests the
same regime that created the TM and can hide amplitude-dependent mismatch.

## Dedicated random complex correction dataset

Generate 512 post-reconstruction probes. The default amplitudes use 8 levels
from 0.2 to 1.0 and the phase uses 16 levels over `[0, 2pi)`:

```powershell
python -m tools.generate_complex_correction_probes_32x24 `
  --count 512 `
  --amplitude-levels 8 `
  --amplitude-min 0.2 `
  --phase-levels 16
```

This writes `correction_patterns_32x24_complex/probe.npy` plus the corresponding
`768x1024` DMD patterns. The default 512-pattern bitmap file is about 0.38 GiB.
The logical complex values are encoded with the same 4x4 super-pixel LUT and
32x32 logical-pixel geometry as the `v4_32x24` measurement path.

On the Windows measurement workstation, initialize the optical path as usual
and acquire this dataset with the dedicated CLI:

```powershell
python -m tools.acquire_complex_correction_32x24 `
  --pattern-dir correction_patterns_32x24_complex `
  --output correction_measurements_memmap.npy
```

The acquisition script reuses `CameraHandler` and `DMDController` from
`calibrate_v4_32x24.py`, but writes a separate headerless `uint16` memmap. It
does not overwrite `measurements_memmap.npy` used by GGS21.

## Run gradient correction

Example for the dedicated complex dataset:

```powershell
python -m tools.run_tm_grad_correction `
  --profile v4_32x24 `
  --tm reconstructed_field.npy `
  --probes correction_patterns_32x24_complex\probe.npy `
  --measurements correction_measurements_memmap.npy `
  --phase-modes 4 4 `
  --amplitude-modes 2 2 `
  --max-probes 512 `
  --output-pixels 1024 `
  --epochs 200 `
  --device cuda
```

`--output-pixels 1024` trains against a fixed random subset of camera pixels so
the matrix multiplication stays small. Set it to `0` to use all camera pixels.
For the dedicated dataset, `--max-probes 512` uses all generated samples before
the internal train/validation split.

The output directory contains:

- `input_gain.npy`: complex input correction, flattened in row-major input order
- `phase_correction.npy`: phase map in radians
- `amplitude_correction.npy`: multiplicative amplitude map
- `history.npy`: `[epoch, loss, train_pcc, val_pcc]`
- `summary.json`: baseline/final train and validation PCC
- `sample_indices.npy` and `output_indices.npy`: exact subset used

By default the full corrected TM is **not** written. Use `--export-tm` only when
needed; for dense 128x128 configurations it can be multiple GiB.

If no epoch beats the zero-correction held-out validation PCC, the CLI now
restores the identity correction instead of exporting a worse last-epoch model.
`summary.json` records this as `reverted_to_baseline: true`.

## First experiment to run

Start with only phase correction:

```powershell
python -m tools.run_tm_grad_correction `
  --profile v4_32x24 `
  --tm reconstructed_field.npy `
  --probes correction_patterns_32x24_complex\probe.npy `
  --measurements correction_measurements_memmap.npy `
  --phase-modes 4 4 `
  --amplitude-modes 0 0 `
  --max-probes 512 `
  --output-pixels 1024 `
  --epochs 200 `
  --device cuda
```

Then compare against phase + amplitude by changing only:

```powershell
--phase-modes 4 4 --amplitude-modes 2 2
```

The important number is held-out validation PCC, not training PCC. If
validation PCC improves clearly with only ~15-20 parameters, that is evidence
that a useful part of the TM mismatch lies on a low-dimensional correction
manifold. If training improves but validation does not, increasing correction
dimension is unlikely to help without changing the error model.

The most useful initial comparison is therefore:

1. original TM on the new amplitude+phase probes
2. phase-only DCT correction
3. phase + amplitude DCT correction

If the original TM reproduces the previously observed amplitude-domain drop but
DCT correction cannot improve held-out PCC, the next model should add structured
mode mixing (for example a low-rank residual) rather than simply increasing the
DCT grid.

## Identifiability note

Intensity-only measurements do not identify arbitrary output phase. For
example, multiplying every output row of the TM by an arbitrary unit-magnitude
phase leaves `|Tx|^2` unchanged. This MVP therefore only learns input-side
corrections and excludes the DCT DC terms, avoiding obvious flat directions in
the PCC objective.
