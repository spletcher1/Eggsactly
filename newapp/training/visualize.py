#!/usr/bin/env python3
"""Visualize image/label pairs and (optionally) model predictions.

Usage
-----
    # Just check labels look right
    python visualize.py path/to/images/ path/to/labels/ --n 5

    # Also overlay model predictions on top
    python visualize.py path/to/images/ path/to/labels/ --n 5 \\
        --weights checkpoints/best.pth

Saves annotated PNGs to ``--outdir`` (default: ``viz_output/``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_NEWAPP_ROOT = Path(__file__).resolve().parent.parent
if str(_NEWAPP_ROOT) not in sys.path:
    sys.path.insert(0, str(_NEWAPP_ROOT))

from dataset import load_image_label_pairs


def colorize_labels(label: np.ndarray) -> np.ndarray:
    """Map each integer label to a random but distinct RGB color."""
    ids = np.unique(label)
    ids = ids[ids > 0]
    color_img = np.zeros((*label.shape, 3), dtype=np.uint8)
    rng = np.random.RandomState(42)
    for obj_id in ids:
        color = rng.randint(80, 255, size=3).tolist()
        color_img[label == obj_id] = color
    return color_img


def overlay(img: np.ndarray, label: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a colorized label mask on top of the image."""
    label_rgb = colorize_labels(label)
    mask = (label > 0).astype(np.float32)[..., np.newaxis]
    blended = img.astype(np.float32) * (1 - alpha * mask) + label_rgb.astype(
        np.float32
    ) * (alpha * mask)
    return np.clip(blended, 0, 255).astype(np.uint8)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images_dir", type=Path)
    parser.add_argument("labels_dir", type=Path)
    parser.add_argument("--n", type=int, default=5, help="Number of images to visualize")
    parser.add_argument("--outdir", type=Path, default=Path("viz_output"))
    parser.add_argument("--weights", type=Path, default=None,
                        help="Optional: overlay model predictions too")
    args = parser.parse_args(argv)

    images, labels = load_image_label_pairs(args.images_dir, args.labels_dir)
    n = min(args.n, len(images))
    args.outdir.mkdir(parents=True, exist_ok=True)

    model = None
    if args.weights:
        from eggcount.egg_model import EggModel
        model = EggModel(weights_path=args.weights)
        print(f"loaded model from {args.weights}")

    for i in range(n):
        img, lbl = images[i], labels[i]
        n_objects = len(np.unique(lbl)) - 1

        # Ground truth overlay
        vis = overlay(img, lbl)
        cv2.putText(
            vis, f"labels: {n_objects} objects",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )

        if model is not None:
            preds = model.count(img)
            # Draw predicted outlines on the image
            for outline in preds.outlines:
                pts = np.array(
                    [[int(round(pt[1])), int(round(pt[0]))] for pt in outline],
                    dtype=np.int32,
                )
                cv2.polylines(vis, [pts], True, (0, 165, 255), 1, cv2.LINE_AA)
            cv2.putText(
                vis, f"predicted: {preds.count}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2,
            )

        out_path = args.outdir / f"viz_{i:03d}.png"
        cv2.imwrite(str(out_path), vis)
        print(f"  {out_path} — {n_objects} labeled objects"
              + (f", {preds.count} predicted" if model else ""))

    print(f"done. {n} visualizations saved to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
