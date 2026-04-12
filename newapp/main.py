#!/usr/bin/env python3
"""Command-line entry point for the local egg-counting pipeline.

Usage
-----
    python main.py path/to/image.jpg [--weights path/to/eggs.pth] [--outdir results/]

What it does
------------
1. Loads the trained SplineDist egg counter (from ``newapp/models/`` by
   default, or a path you pass with ``--weights``).
2. Calls :func:`eggcount.chamber_detection.detect_regions` on the image to
   get the egg-laying regions. The default implementation returns a single
   region covering the whole image — **replace that function with your own
   chamber detection logic** (hardcoded or CV-based). See
   ``eggcount/chamber_detection.py``.
3. Runs the egg counter on each region.
4. Writes three outputs next to the input image (or under ``--outdir``):
   - ``<stem>_annotated.png`` — the input image with region boxes and egg
     outlines drawn
   - ``<stem>_counts.json``   — structured counts + per-egg coordinates
   - ``<stem>_counts.csv``    — one row per region with its count
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``newapp/`` the package root for ``import eggcount`` regardless of
# where the user runs this script from.
_NEWAPP_ROOT = Path(__file__).resolve().parent
if str(_NEWAPP_ROOT) not in sys.path:
    sys.path.insert(0, str(_NEWAPP_ROOT))

from eggcount.egg_model import EggModel
from eggcount.pipeline import (
    analyze_image,
    load_image,
    save_annotated_image,
    save_csv,
    save_json,
)


DEFAULT_WEIGHTS_DIR = _NEWAPP_ROOT / "models"


def _default_weights_path() -> Path | None:
    """Find a .pth file under newapp/models/, if any."""
    if not DEFAULT_WEIGHTS_DIR.is_dir():
        return None
    candidates = sorted(DEFAULT_WEIGHTS_DIR.glob("*.pth"))
    return candidates[0] if candidates else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to the image to analyze")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help=(
            "Path to the egg model .pth weights. Defaults to the first .pth "
            f"found under {DEFAULT_WEIGHTS_DIR}/"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Config JSON for the egg model. Defaults to "
            "unet_backbone_rand_zoom.json under newapp/configs/"
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Directory to write outputs (default: alongside the input image)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=("cuda", "cpu"),
        help="Force device (default: cuda if available, else cpu)",
    )
    args = parser.parse_args(argv)

    if not args.image.is_file():
        print(f"error: image not found: {args.image}", file=sys.stderr)
        return 1

    weights = args.weights or _default_weights_path()
    if weights is None:
        print(
            f"error: no egg model weights found. Drop a .pth under "
            f"{DEFAULT_WEIGHTS_DIR}/ or pass --weights.",
            file=sys.stderr,
        )
        return 1

    outdir = args.outdir or args.image.parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem

    print(f"loading egg model from {weights} ...")
    model = EggModel(
        weights_path=weights, config_path=args.config, device=args.device
    )
    print(f"analyzing {args.image} ...")
    result = analyze_image(args.image, model)

    print(f"detected {len(result.regions)} region(s), total count = {result.total_count}")
    for r in result.regions:
        print(f"  {r.region.label}: {r.count}")

    annotated_path = outdir / f"{stem}_annotated.png"
    json_path = outdir / f"{stem}_counts.json"
    csv_path = outdir / f"{stem}_counts.csv"

    img = load_image(args.image)
    save_annotated_image(img, result, annotated_path)
    save_json(result, json_path)
    save_csv(result, csv_path)

    print(f"wrote {annotated_path}")
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
