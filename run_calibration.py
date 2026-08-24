"""Launch one of the supported TM calibration profiles."""

import argparse
import importlib


PROFILE_MODULES = {
    "v4_32x24": "calibrate_v4_32x24",
    "v4_32x24_cholesky": "calibrate_v4_32x24_cholesky",
    "fivefold_160x120": "calibrate_160x120",
    "dense_128x128": "calibrate_128x128",
    "dense_128x128_roi26": "calibrate_128x128_26x26",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILE_MODULES), required=True)
    parser.add_argument(
        "--channel",
        choices=("I0", "I90"),
        help="Required only by the dense_128x128_roi26 polarization profile.",
    )
    args = parser.parse_args()
    if args.profile == "dense_128x128_roi26":
        args.channel = args.channel or "I0"
    elif args.channel is not None:
        parser.error("--channel is supported only by dense_128x128_roi26")
    return args


def main():
    args = parse_args()
    module = importlib.import_module(PROFILE_MODULES[args.profile])
    if args.profile == "dense_128x128_roi26":
        app = module.Application(polarization_channel=args.channel)
    else:
        app = module.Application()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


if __name__ == "__main__":
    main()
