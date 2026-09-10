"""Acquire camera responses for the 32x24 complex correction probe set.

This is a small hardware-only acquisition entry point. It reuses the tested
CameraHandler and DMDController from calibrate_v4_32x24.py, but writes a
separate measurement file so the GGS21 reconstruction data are not touched.

Run from the TMCalib repository root on the Windows measurement workstation.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from calibrate_v4_32x24 import CameraHandler, DMDController


EXPECTED_INPUT_SHAPE = (24, 32)
EXPECTED_DMD_SHAPE = (768, 1024)


def _load_dataset(pattern_dir):
    pattern_dir = Path(pattern_dir).resolve()
    probe_path = pattern_dir / "probe.npy"
    pattern_path = pattern_dir / "patterns_pregenerated.npy"
    metadata_path = pattern_dir / "metadata.json"

    if not probe_path.exists():
        raise FileNotFoundError("probe file not found: {}".format(probe_path))
    if not pattern_path.exists():
        raise FileNotFoundError("pattern file not found: {}".format(pattern_path))

    probes = np.load(str(probe_path), mmap_mode="r", allow_pickle=False)
    patterns = np.load(str(pattern_path), mmap_mode="r", allow_pickle=False)

    if probes.ndim != 3 or tuple(probes.shape[1:]) != EXPECTED_INPUT_SHAPE:
        raise ValueError(
            "probe shape must be [K, 24, 32], got {}".format(probes.shape)
        )
    if patterns.ndim != 3 or tuple(patterns.shape[1:]) != EXPECTED_DMD_SHAPE:
        raise ValueError(
            "pattern shape must be [K, 768, 1024], got {}".format(patterns.shape)
        )
    if probes.shape[0] != patterns.shape[0]:
        raise ValueError(
            "probe/pattern counts differ: {} vs {}".format(
                probes.shape[0], patterns.shape[0]
            )
        )

    metadata = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)

    return pattern_dir, probes, patterns, metadata


def _release_dmd(dmd):
    if dmd is None or dmd.DMD is None or dmd.dev_id is None:
        return
    try:
        dmd.DMD.juoptStop(dmd.dev_id)
    except Exception:
        pass
    try:
        dmd.clear_sequence(0)
    except Exception:
        pass
    try:
        dmd.DMD.juoptFree(dmd.dev_id)
    except Exception:
        pass
    dmd.dev_id = None
    dmd.is_init = False


def _flush_camera_stream(camera, dmd, roi_h, roi_w):
    """Project one white frame in a separate acquisition session.

    The main v4 GGS21 measurement path deliberately does this before every
    pattern batch. juoptStop/clear_sequence can leave a stale/blank frame in the
    camera stream; stopping the camera after this white frame flushes that state
    before the real batch starts. Without this step probe/measurement pairing can
    become shifted by one frame and the TM PCC collapses to ~0 even when the TM
    itself is fine.
    """
    camera.start()
    try:
        white_roi = dmd._capture_white_speckle_roi(roi_h, roi_w)
        if white_roi is None:
            print("warning: camera preflush white frame was not captured")
        else:
            print(
                "camera preflush: mean={:.3f} std={:.3f}".format(
                    float(np.mean(white_roi)), float(np.std(white_roi))
                )
            )
    finally:
        camera.stop()


def acquire(
    pattern_dir,
    output_path,
    camera_index=0,
    dmd_index=0,
    batch_size=256,
    overwrite=False,
):
    batch_size = int(batch_size)
    if batch_size <= 0 or batch_size > 1000:
        raise ValueError("batch_size must be in [1, 1000]")

    pattern_dir, probes, patterns, source_metadata = _load_dataset(pattern_dir)
    sample_count = int(probes.shape[0])

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            "measurement output already exists; use --overwrite: {}".format(
                output_path
            )
        )
    if overwrite:
        if output_path.exists():
            output_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()

    camera = None
    dmd = None
    measurements = None
    try:
        camera = CameraHandler(
            cam_index=int(camera_index),
            save_path=str(output_path.parent),
        )
        roi_h = int(getattr(camera, "roi_height", 128))
        roi_w = int(getattr(camera, "roi_width", 128))
        output_count = roi_h * roi_w

        dmd = DMDController(camera_handler=camera)
        devices = dmd.get_devices()
        if not devices:
            raise RuntimeError("no JUOPT DMD device found")
        if not 0 <= int(dmd_index) < len(devices):
            raise ValueError(
                "dmd-index {} is invalid for {} detected devices".format(
                    dmd_index, len(devices)
                )
            )
        device_name = devices[int(dmd_index)]
        if not dmd.initialize_device(device_name):
            raise RuntimeError("failed to initialize DMD device {}".format(device_name))

        # Keep the same raw storage convention as the GGS21 acquisition:
        # native Polarized8 codes are stored in a headerless uint16 memmap.
        measurements = np.memmap(
            str(output_path),
            dtype=np.uint16,
            mode="w+",
            shape=(sample_count, output_count),
        )

        print("complex correction acquisition")
        print("  patterns: {}".format(pattern_dir))
        print("  samples: {}".format(sample_count))
        print("  camera ROI: {}x{}".format(roi_w, roi_h))
        print("  DMD: {}".format(device_name))
        print("  batch size: {}".format(batch_size))
        print("  preflush: white frame + fresh camera session before every batch")
        print("  output: {}".format(output_path))

        for start in range(0, sample_count, batch_size):
            stop = min(start + batch_size, sample_count)
            batch_patterns = np.asarray(patterns[start:stop], dtype=np.uint8)

            # Match calibrate_v4_32x24.run_measurement(): use a separate white
            # acquisition to consume any stale frame, stop the stream, then
            # start a clean acquisition for the actual pattern batch.
            _flush_camera_stream(camera, dmd, roi_h, roi_w)

            camera.start()
            try:
                images = dmd.project_and_caption(batch_patterns)
            finally:
                camera.stop()

            if images is None:
                raise RuntimeError(
                    "DMD/camera acquisition returned no images for samples {}:{}".format(
                        start, stop
                    )
                )
            if images.shape[0] != stop - start:
                raise RuntimeError(
                    "captured {} frames for a {}-pattern batch".format(
                        images.shape[0], stop - start
                    )
                )
            if images.shape[1] < roi_h or images.shape[2] < roi_w:
                raise RuntimeError(
                    "camera image shape {} is smaller than ROI {}x{}".format(
                        images.shape, roi_w, roi_h
                    )
                )

            roi = images[:, :roi_h, :roi_w]
            flattened = roi.reshape(stop - start, -1)
            frame_std = np.std(flattened.astype(np.float32), axis=1)
            frame_mean = np.mean(flattened.astype(np.float32), axis=1)
            print(
                "  batch {}:{} frame mean median={:.3f}, std min/median={:.3f}/{:.3f}".format(
                    start,
                    stop,
                    float(np.median(frame_mean)),
                    float(np.min(frame_std)),
                    float(np.median(frame_std)),
                )
            )
            near_flat = np.flatnonzero(frame_std < 1e-3)
            if near_flat.size:
                print(
                    "  WARNING: near-flat captured frames at dataset indices {}".format(
                        (near_flat[:16] + start).tolist()
                    )
                )

            measurements[start:stop] = flattened.astype(np.uint16)
            measurements.flush()
            print("measured correction probes: {}/{}".format(stop, sample_count))

        metadata = {
            "purpose": "post-reconstruction gradient TM correction",
            "pattern_directory": str(pattern_dir),
            "sample_count": sample_count,
            "input_shape": list(EXPECTED_INPUT_SHAPE),
            "camera_roi": [roi_h, roi_w],
            "storage": "headerless uint16 memmap",
            "measurement_shape": [sample_count, output_count],
            "camera_index": int(camera_index),
            "dmd_index": int(dmd_index),
            "dmd_device": device_name,
            "batch_size": batch_size,
            "camera_preflush_white_frame": True,
            "source_metadata": source_metadata,
        }
        with metadata_path.open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, ensure_ascii=False)
            stream.write("\n")

        print("acquisition complete")
        print("  measurements: {}".format(output_path))
        print("  metadata: {}".format(metadata_path))

    finally:
        if measurements is not None:
            try:
                measurements.flush()
            except Exception:
                pass
            del measurements
        _release_dmd(dmd)
        if camera is not None:
            try:
                camera.cleanup()
            except Exception as exc:
                print("camera cleanup warning: {}".format(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pattern-dir",
        default="correction_patterns_32x24_complex",
    )
    parser.add_argument(
        "--output",
        default="correction_measurements_memmap.npy",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--dmd-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    acquire(
        pattern_dir=args.pattern_dir,
        output_path=args.output,
        camera_index=args.camera_index,
        dmd_index=args.dmd_index,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
