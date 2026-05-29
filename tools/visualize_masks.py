"""Create colorized mask and overlay previews for local inspection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.visualization import blend_image_mask, colorize_mask, make_prediction_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize class-id segmentation masks.")
    parser.add_argument("--mask", type=Path, required=True, help="Class-id PNG mask.")
    parser.add_argument("--image", type=Path, default=None, help="Optional RGB image for overlay.")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG/JPG path.")
    parser.add_argument(
        "--mode",
        choices=["color", "overlay", "grid"],
        default="grid",
        help="Visualization mode. Overlay/grid require --image.",
    )
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "color":
        result = colorize_mask(args.mask)
    else:
        if args.image is None:
            raise SystemExit(f"--image is required when --mode={args.mode}")
        if args.mode == "overlay":
            result = blend_image_mask(args.image, args.mask, alpha=args.alpha)
        else:
            result = make_prediction_grid(args.image, args.mask, alpha=args.alpha)

    result.save(args.output)
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
