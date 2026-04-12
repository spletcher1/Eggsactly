#!/usr/bin/env python3
"""Organize images and labels into train/val splits.

Usage
-----
    python prepare_data.py path/to/images/ path/to/labels/ path/to/output/ \\
        --val-fraction 0.15

Creates::

    output/
    ├── train/
    │   ├── images/
    │   └── labels/
    └── val/
        ├── images/
        └── labels/

Images and labels are **symlinked** (not copied) to save disk space.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path


def find_pairs(
    images_dir: Path,
    labels_dir: Path,
    image_exts: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
) -> list[tuple[Path, Path]]:
    """Return a list of (image_path, label_path) pairs matched by stem."""
    img_stems: dict[str, Path] = {}
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() in image_exts:
            img_stems[p.stem] = p

    pairs = []
    for stem, img_path in sorted(img_stems.items()):
        for ext in (".png", ".tif", ".tiff"):
            candidate = labels_dir / f"{stem}{ext}"
            if candidate.is_file():
                pairs.append((img_path, candidate))
                break
    return pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images_dir", type=Path)
    parser.add_argument("labels_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--val-fraction", type=float, default=0.15,
        help="Fraction of data reserved for validation (default: 0.15)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy", action="store_true",
        help="Copy files instead of symlinking",
    )
    args = parser.parse_args(argv)

    pairs = find_pairs(args.images_dir, args.labels_dir)
    if not pairs:
        print("error: no matched image/label pairs found", file=sys.stderr)
        return 1

    random.seed(args.seed)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_fraction))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        img_out = args.output_dir / split_name / "images"
        lbl_out = args.output_dir / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path in split_pairs:
            dst_img = img_out / img_path.name
            dst_lbl = lbl_out / lbl_path.name
            if not dst_img.exists():
                if args.copy:
                    shutil.copy2(img_path, dst_img)
                else:
                    os.symlink(img_path.resolve(), dst_img)
            if not dst_lbl.exists():
                if args.copy:
                    shutil.copy2(lbl_path, dst_lbl)
                else:
                    os.symlink(lbl_path.resolve(), dst_lbl)

    print(f"train: {len(train_pairs)} pairs -> {args.output_dir / 'train'}")
    print(f"val:   {len(val_pairs)} pairs -> {args.output_dir / 'val'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
