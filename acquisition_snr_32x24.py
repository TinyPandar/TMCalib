"""Measure temporal single-frame SNR using one repeated calibration bitmap.

Run from the repository root in the py38 hardware environment. Black DMD
frames measure optical background plus camera offset, NOT shuttered dark.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def analyze(frames, background, saturation=255):
    frames = np.asarray(frames, dtype=np.float64)
    background = np.asarray(background, dtype=np.float64)
    if (frames.ndim != 3 or background.ndim != 3 or
            min(len(frames), len(background)) < 2 or
            frames.shape[1:] != background.shape[1:] or
            not np.isfinite(frames).all() or not np.isfinite(background).all()):
        raise ValueError('Need finite repeated signal/background stacks of matching shape')
    mean = frames.mean(axis=0)
    signal = mean - background.mean(axis=0)
    noise = frames.std(axis=0, ddof=1)
    saturated = (frames >= saturation).any(axis=0) | (background >= saturation).any(axis=0)
    valid = ~saturated & (signal > 0) & (noise > 0)
    snr = np.full(signal.shape, np.nan)
    snr[valid] = 20 * np.log10(signal[valid] / noise[valid])
    summary = {
        'frames': len(frames), 'background_frames': len(background),
        'background_kind': 'black DMD bitmap, not shuttered sensor dark',
        'metric': 'single-frame temporal SNR; no drift removal',
        'valid_pixels': int(valid.sum()), 'total_pixels': int(valid.size),
        'saturated_pixel_fraction': float(saturated.mean()),
        'zero_temporal_std_pixel_fraction': float((noise == 0).mean()),
        'mean_dn': float(mean.mean()),
        'black_background_mean_dn': float(background.mean()),
    }
    if valid.any():
        denominator = np.sum(noise[valid] ** 2)
        summary.update(
            background_subtracted_roi_snr_db=float(10 * np.log10(np.sum(signal[valid] ** 2) / denominator)),
            raw_roi_snr_db_same_mask=float(10 * np.log10(np.sum(mean[valid] ** 2) / denominator)),
            pixel_snr_db_percentiles_10_50_90=np.percentile(snr[valid], [10, 50, 90]).tolist(),
            temporal_noise_rms_dn=float(np.sqrt(np.mean(noise[valid] ** 2))),
        )
    return summary, dict(mean_dn=mean, signal_dn=signal, noise_std_dn=noise,
                         snr_db=snr, valid_mask=valid,
                         frame_mean_dn=frames.mean(axis=(1, 2)))


def capture(controller, camera, pattern, count, output, label, period_ns):
    # Discard warmup frames while keeping the exact calibration trigger cadence.
    warmup = 10
    batch = np.repeat(pattern[None], count + warmup, axis=0)
    images, ids, timestamps = [], [], []
    started = False
    try:
        if not controller.load_pattern(batch):
            raise RuntimeError('DMD pattern load failed')
        camera.start()
        started = True
        result = controller.DMD.juoptProjection(controller.dev_id, 0, 0)
        if result not in (None, 0):
            raise RuntimeError('DMD projection failed: {}'.format(result))
        for index in range(len(batch)):
            image, _, _ = camera.run()
            if image is None:
                raise RuntimeError('{}: missing frame {}'.format(label, index))
            images.append(np.asarray(image).copy())
            ids.append(camera.last_frame_id)
            timestamps.append(camera.last_frame_timestamp_ns)
        if any(value is None for value in ids + timestamps):
            raise RuntimeError('Camera frame IDs/timestamps unavailable')
        if not np.all(np.diff(ids) == 1):
            raise RuntimeError('Camera frame ID discontinuity')
        intervals = np.diff(timestamps)
        if np.any(np.abs(intervals - period_ns) > period_ns * 0.25):
            raise RuntimeError('Camera trigger cadence outside 25% tolerance')
        stack = np.stack(images)[warmup:]
        np.save(output / (label + '_frames.npy'), stack)
        return stack
    finally:
        # Preserve partial evidence even when capture is rejected.
        (output / (label + '_timing.json')).write_text(json.dumps(
            dict(frame_ids=ids, timestamps_ns=timestamps, warmup_frames=warmup), indent=2))
        try:
            controller.DMD.juoptStop(controller.dev_id)
        finally:
            try:
                if started:
                    camera.stop()
            finally:
                controller.clear_sequence(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=100)
    parser.add_argument('--probe-index', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, default=Path('acquisition_snr_results') / datetime.now().strftime('%Y%m%d_%H%M%S'))
    args = parser.parse_args()
    if not 2 <= args.frames <= 990:
        parser.error('--frames must be between 2 and 990')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import calibrate_v4_32x24 as core
    from hdr_pbr_32x24 import initialize_hardware, set_exposure_us

    config = core.get_active_pattern_config()
    patterns = np.load(Path(config['directory']) / 'patterns_pregenerated.npy', mmap_mode='r')
    if not 0 <= args.probe_index < len(patterns):
        parser.error('probe index outside dataset')
    pattern = np.array(patterns[args.probe_index], dtype=np.uint8)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / 'pattern.npy', pattern)
    camera = controller = None
    try:
        camera, controller = initialize_hardware(output / 'camera')
        exposure = set_exposure_us(camera, core.CAMERA_EXPOSURE_US)
        if not camera.configure_gain(auto_gain='Off', gain=0):
            raise RuntimeError('Cannot lock camera gain')
        metadata = dict(pattern_set=config['name'], probe_index=args.probe_index,
                        exposure_us=exposure, gain_db=float(camera.cam.Gain.GetValue()),
                        target_fps=core.TARGET_ACQUISITION_FPS,
                        roi=[camera.roi_x, camera.roi_y, camera.roi_width, camera.roi_height])
        (output / 'settings.json').write_text(json.dumps(metadata, indent=2))
        period_ns = core.DMD_PICTURE_TIME_US * 1000
        frames = capture(controller, camera, pattern, args.frames, output, 'signal', period_ns)
        background = capture(controller, camera, np.zeros_like(pattern), args.frames, output, 'black', period_ns)
        summary, maps = analyze(frames, background)
        summary.update(metadata)
        np.savez(output / 'snr_maps.npz', **maps)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for ax, key, title in zip(axes.flat, ['signal_dn', 'noise_std_dn', 'snr_db'],
                                 ['Signal minus black (DN)', 'Temporal noise std (DN)', 'Single-frame SNR (dB)']):
            fig.colorbar(ax.imshow(maps[key]), ax=ax)
            ax.set_title(title)
        axes[1, 1].plot(maps['frame_mean_dn'])
        axes[1, 1].set(xlabel='Frame', ylabel='ROI mean (DN)', title='Intensity stability')
        fig.tight_layout()
        fig.savefig(output / 'snr_report.png', dpi=160)
        plt.close(fig)
        print(json.dumps(summary, indent=2))
        print('Results: {}'.format(output))
    except Exception as exc:
        (output / 'failure.txt').write_text(str(exc), encoding='utf-8')
        raise
    finally:
        try:
            if controller is not None:
                controller.cleanup()
        finally:
            if camera is not None:
                camera.cleanup()


if __name__ == '__main__':
    main()
