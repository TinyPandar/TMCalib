"""Compatibility entry point for the 26 x 26 polarization profile.

The implementation lives in :mod:`calibrate_128x128`.  Keeping this module as
a thin profile wrapper preserves the old command line while preventing feature
drift between two copied applications.
"""

import argparse

from calibration_profiles import normalize_polarization_channel
from calibrate_128x128 import Application as _CoreApplication


class Application(_CoreApplication):
    """Configure the shared 128 x 128 application for a polarization ROI."""

    def __init__(self, polarization_channel="I0"):
        channel = normalize_polarization_channel(polarization_channel)
        super().__init__(
            camera_roi=(26, 26),
            polarization_channel=channel,
            exposure_us=1500.0,
            output_tag=channel,
            profile_title=(
                "DMD TM Calibration - 128x128 / px=4 / "
                "camera 26x26 / {}"
            ).format(channel),
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Launch the selectable 26 x 26 polarization profile."
    )
    parser.add_argument(
        "--polarization-channel",
        choices=("I0", "I90"),
        default="I0",
        help="Polarization quadrant to acquire.",
    )
    args = parser.parse_args(argv)
    app = Application(polarization_channel=args.polarization_channel)
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


if __name__ == "__main__":
    main()
