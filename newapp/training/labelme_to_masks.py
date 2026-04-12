#!/usr/bin/env python3
"""Convert Labelme polygon annotations to instance-label mask images.

Usage
-----
    python labelme_to_masks.py path/to/labelme_jsons/ path/to/output_masks/

Each ``.json`` file produced by Labelme contains a list of polygon shapes.
This script rasterizes each polygon as a distinct integer label (1, 2, 3, ...)
on a zero background and saves the result as a 16-bit PNG alongside the
original image.

It also handles **napari-exported label images** (single-channel PNGs where
each object already has a unique integer ID). If a ``.png`` with the same
stem as the input image already exists in the output directory, it is
skipped.

Labelme JSON format (the relevant parts)::

    {
      "imagePath": "sample_001.jpg",
      "imageWidth": 1024,
      "imageHeight": 768,
      "shapes": [
        {
          "label": "egg",
          "shape_type": "polygon",
          "points": [[x1, y1], [x2, y2], ...]
        },
        ...
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def labelme_json_to_mask(json_path: Path) -> tuple[np.ndarray, str]:
    """Read a Labelme JSON and return ``(mask, image_filename)``.

    ``mask`` is an ``int32`` array of shape ``(H, W)`` where each polygon
    shape gets a unique integer label starting from 1.
    """
    with open(json_path) as f:
        data = json.load(f)

    h = data["imageHeight"]
    w = data["imageWidth"]
    mask = np.zeros((h, w), dtype=np.int32)

    obj_id = 1
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "polygon":
            continue
        pts = np.array(shape["points"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], color=obj_id)
        obj_id += 1

    return mask, data.get("imagePath", json_path.stem)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert Labelme JSON annotations to instance-label PNGs."
    )
    parser.add_argument("input_dir", type=Path, help="Directory of .json files")
    parser.add_argument("output_dir", type=Path, help="Directory for output masks")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_files = sorted(args.input_dir.glob("*.json"))
    if not json_files:
        print(f"No .json files found in {args.input_dir}", file=sys.stderr)
        return 1

    for jf in json_files:
        out_path = args.output_dir / f"{jf.stem}.png"
        if out_path.exists():
            print(f"  skip {jf.name} (output exists)")
            continue
        mask, img_name = labelme_json_to_mask(jf)
        n_objects = len(np.unique(mask)) - 1
        # Save as 16-bit PNG to support > 255 objects per image
        cv2.imwrite(str(out_path), mask.astype(np.uint16))
        print(f"  {jf.name} -> {out_path.name}  ({n_objects} objects)")

    print(f"done. masks written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
