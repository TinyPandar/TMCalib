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

You need three `.npy` files with matching sample order:

1. reconstructed complex TM
2. complex input probes
3. real measured camera intensities for those probes

For `v4_32x24`:

- probe sample shape: `32x24` (stored as `(24, 32)`)
- camera sample shape: `128x128`
- TM convention: `[N_output, N_input]`

The calibration probes used to reconstruct `T0` are valid as a smoke test, but
for a meaningful correction/validation experiment it is better to collect a
separate set of random phase/amplitude inputs after TM reconstruction.

## Run

Example for the 32x24 profile:

```powershell
python -m tools.run_tm_grad_correction ^
  --profile v4_32x24 ^
  --tm reconstructed_field.npy ^
  --probes correction_probes.npy ^
  --measurements correction_measurements.npy ^
  --phase-modes 4 4 ^
  --amplitude-modes 2 2 ^
  --max-probes 256 ^
  --output-pixels 1024 ^
  --epochs 200 ^
  --device cuda
```

`--output-pixels 1024` trains against a fixed random subset of camera pixels so
the matrix multiplication stays small. Set it to `0` to use all camera pixels.

The output directory contains:

- `input_gain.npy`: complex input correction, flattened in row-major input order
- `phase_correction.npy`: phase map in radians
- `amplitude_correction.npy`: multiplicative amplitude map
- `history.npy`: `[epoch, loss, train_pcc, val_pcc]`
- `summary.json`: baseline/final train and validation PCC
- `sample_indices.npy` and `output_indices.npy`: exact subset used

By default the full corrected TM is **not** written. Use `--export-tm` only when
needed; for dense 128x128 configurations it can be multiple GiB.

## First experiment to run

Start with only phase correction:

```powershell
python -m tools.run_tm_grad_correction ... ^
  --phase-modes 4 4 ^
  --amplitude-modes 0 0
```

Then compare against phase + amplitude:

```powershell
python -m tools.run_tm_grad_correction ... ^
  --phase-modes 4 4 ^
  --amplitude-modes 2 2
```

The important number is held-out validation PCC, not training PCC. If
validation PCC improves clearly with only ~15-20 parameters, that is evidence
that a useful part of the TM mismatch lies on a low-dimensional correction
manifold. If training improves but validation does not, increasing correction
dimension is unlikely to help without changing the error model.

## Identifiability note

Intensity-only measurements do not identify arbitrary output phase. For
example, multiplying every output row of the TM by an arbitrary unit-magnitude
phase leaves `|Tx|^2` unchanged. This MVP therefore only learns input-side
corrections and excludes the DCT DC terms, avoiding obvious flat directions in
the PCC objective.
