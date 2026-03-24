from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve


DEFAULT_URL = "https://raw.githubusercontent.com/piergiaj/pytorch-i3d/master/models/rgb_imagenet.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the upstream I3D RGB ImageNet checkpoint on demand.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Source checkpoint URL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/rgb_imagenet.pt"),
        help="Where to store the checkpoint.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists() and not args.force:
        print(f"Checkpoint already exists: {output}")
        print("Use --force to download it again.")
        return

    print(f"Downloading {args.url}")
    print(f"Saving to {output}")
    urlretrieve(args.url, output)
    print("Done.")


if __name__ == "__main__":
    main()
