"""End-to-end analysis pipeline.

Given an image path and a loaded :class:`eggcount.egg_model.EggModel`, run
region detection, egg counting per region, and return structured results
plus (optionally) an annotated image.

This is the local, single-image equivalent of the upstream
``SessionManager`` orchestration — with no Flask, no SocketIO, no task
queue, and no worker process.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .chamber_detection import Region, RegionDetector, detect_regions
from .egg_model import EggModel, EggPredictions


@dataclass
class RegionResult:
    """Per-region analysis output.

    All coordinates in this structure are in **original image pixel space**,
    even the egg outlines — :meth:`analyze_image` translates the egg
    counter's per-crop coordinates back to the full-image frame so you can
    draw everything on one canvas.
    """

    region: Region
    count: int
    #: Egg centroids in original image coordinates, shape ``(N, 2)``.
    points: np.ndarray
    #: Per-egg confidence scores, shape ``(N,)``.
    prob: np.ndarray
    #: Per-egg polygonal outlines in original image coordinates (list of
    #: lists of ``(y, x)`` floats).
    outlines: list

    def to_jsonable(self) -> dict:
        """Return a plain-Python dict suitable for ``json.dumps``."""
        return {
            "label": self.region.label,
            "bbox": list(self.region.bbox),
            "circle": list(self.region.circle) if self.region.circle else None,
            "meta": self.region.meta,
            "count": int(self.count),
            "points": self.points.tolist() if len(self.points) else [],
            "prob": self.prob.tolist() if len(self.prob) else [],
            "outlines": self.outlines,
        }


@dataclass
class AnalysisResult:
    """Full result of analyzing one image."""

    image_path: Path
    image_shape: tuple[int, int]  # (height, width)
    regions: list[RegionResult] = field(default_factory=list)

    @property
    def total_count(self) -> int:
        return sum(r.count for r in self.regions)

    def to_jsonable(self) -> dict:
        return {
            "image_path": str(self.image_path),
            "image_shape": list(self.image_shape),
            "total_count": self.total_count,
            "regions": [r.to_jsonable() for r in self.regions],
        }


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk as a uint8 BGR array (OpenCV convention)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"OpenCV could not decode image: {path}")
    return img


def _apply_circle_mask(crop: np.ndarray, region: Region) -> np.ndarray:
    """Zero out pixels outside ``region.circle`` in a crop.

    The region's circle is given in original image coordinates; we translate
    to crop-local coordinates before masking.
    """
    if region.circle is None:
        return crop
    cx, cy, r = region.circle
    bx, by, _, _ = region.bbox
    local_cx, local_cy = cx - bx, cy - by
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (int(local_cx), int(local_cy)), int(r), 255, thickness=-1)
    if crop.ndim == 3:
        mask3 = cv2.merge([mask] * crop.shape[2])
        return cv2.bitwise_and(crop, mask3)
    return cv2.bitwise_and(crop, mask)


def _translate_predictions_to_image_frame(
    preds: EggPredictions, region: Region
) -> tuple[np.ndarray, list]:
    """Shift per-crop coordinates back into original image coordinates."""
    bx, by, _, _ = region.bbox
    if len(preds.points) == 0:
        return np.empty((0, 2)), []
    # SplineDist returns points in (row, col) = (y, x) order.
    points = preds.points.copy().astype(float)
    points[:, 0] += by
    points[:, 1] += bx
    outlines = []
    for outline in preds.outlines:
        shifted = [[pt[0] + by, pt[1] + bx] for pt in outline]
        outlines.append(shifted)
    return points, outlines


def analyze_image(
    image_path: str | Path,
    egg_model: EggModel,
    region_detector: RegionDetector = detect_regions,
) -> AnalysisResult:
    """Run the full pipeline on a single image.

    Parameters
    ----------
    image_path
        Path to the image on disk.
    egg_model
        A loaded :class:`EggModel` instance.
    region_detector
        Callable returning the egg-laying regions. Defaults to
        :func:`eggcount.chamber_detection.detect_regions`. Replace with your
        own implementation as described in the chamber_detection module.
    """
    img = load_image(image_path)
    h, w = img.shape[:2]

    regions = list(region_detector(img))
    for i, region in enumerate(regions):
        if not region.label:
            region.label = f"region_{i}"

    results: list[RegionResult] = []
    for region in regions:
        x, y, rw, rh = region.bbox
        # Clamp the bbox to the image so a slightly-oversized region
        # from a hand-coded detector doesn't explode.
        x2 = min(x + rw, w)
        y2 = min(y + rh, h)
        crop = img[y:y2, x:x2]
        crop = _apply_circle_mask(crop, region)
        preds = egg_model.count(crop)
        points, outlines = _translate_predictions_to_image_frame(preds, region)
        results.append(
            RegionResult(
                region=region,
                count=preds.count,
                points=points,
                prob=preds.prob,
                outlines=outlines,
            )
        )

    return AnalysisResult(
        image_path=Path(image_path),
        image_shape=(h, w),
        regions=results,
    )


# --------- Rendering / export helpers ---------


def render_annotated(img: np.ndarray, result: AnalysisResult) -> np.ndarray:
    """Draw region bboxes, circles, egg outlines, and counts on a copy of ``img``.

    Returns a new image; does not mutate the input.
    """
    out = img.copy()
    green = (0, 255, 0)
    yellow = (0, 255, 255)
    white = (255, 255, 255)

    for r in result.regions:
        x, y, w, h = r.region.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), green, thickness=2)
        if r.region.circle is not None:
            cx, cy, rad = r.region.circle
            cv2.circle(out, (int(cx), int(cy)), int(rad), yellow, thickness=2)
        for outline in r.outlines:
            pts = np.array(
                [[int(round(pt[1])), int(round(pt[0]))] for pt in outline],
                dtype=np.int32,
            )
            cv2.polylines(out, [pts], True, (0, 165, 255), thickness=1, lineType=cv2.LINE_AA)
        label = f"{r.region.label}: {r.count}"
        cv2.putText(
            out,
            label,
            (x + 5, y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            white,
            2,
            cv2.LINE_AA,
        )
    return out


def save_annotated_image(
    img: np.ndarray, result: AnalysisResult, out_path: str | Path
) -> Path:
    """Render annotations on top of ``img`` and save to ``out_path``."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    annotated = render_annotated(img, result)
    cv2.imwrite(str(out_path), annotated)
    return out_path


def save_json(result: AnalysisResult, out_path: str | Path) -> Path:
    """Write the analysis result as JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result.to_jsonable(), f, indent=2)
    return out_path


def save_csv(result: AnalysisResult, out_path: str | Path) -> Path:
    """Write one row per region to a CSV (label, bbox, count)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "region", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "count"])
        for r in result.regions:
            x, y, w, h = r.region.bbox
            writer.writerow([result.image_path.name, r.region.label, x, y, w, h, r.count])
        writer.writerow([])
        writer.writerow(["total", "", "", "", "", "", result.total_count])
    return out_path
