"""Chamber / egg-laying region detection.

This is the plug-in point for your own chamber geometry. Instead of running
the original Eggsactly arena-detection model (which assumes *Drosophila*
chambers and fails on other shapes), the pipeline asks ``detect_regions``
for a list of ``Region`` objects — one per area where eggs may be laid —
and the egg counter is run on each region's crop.

To adapt for a new chamber design you have two options:

1. **Hardcode it.** Fill in :func:`detect_regions` with bounding boxes you
   compute yourself — e.g. a fixed grid, a single central region, or
   coordinates loaded from a JSON sidecar. This is the quickest way to get
   end-to-end results while you're still working on the real detector.

2. **Replace it with a CV / ML detector.** ``detect_regions`` is just a
   plain function from a numpy image to a list of ``Region``. You can swap
   in Hough circles, template matching, a small YOLO model, or whatever
   fits your chamber shape. The pipeline doesn't care how the regions are
   produced, only that the returned ``(x, y, w, h)`` bboxes are in
   **original image pixel coordinates** and lie inside the image bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


@dataclass
class Region:
    """A single egg-laying region in an image.

    The ``bbox`` is always the rectangular crop that the egg counter will
    receive. If ``circle`` is provided, pixels outside the circle are masked
    out before counting, which is useful when your chambers are round pits
    in a flat plate — it prevents the egg counter from seeing eggs that
    belong to neighboring pits.

    Attributes
    ----------
    bbox : tuple of int
        ``(x, y, width, height)`` in **original image pixel coordinates**.
    circle : tuple of int, optional
        ``(cx, cy, radius)`` in **original image pixel coordinates**. If set,
        a circular mask is applied to the crop before egg counting.
    label : str
        Optional human-readable label for this region. Shown in UI and CSV
        output. Defaults to an empty string; the pipeline will fill in an
        auto-generated label (``"region_0"``, ``"region_1"``, ...) if empty.
    meta : dict
        Free-form metadata. The pipeline copies this through to the result
        so you can round-trip arbitrary info (row/col index, well ID, etc.).
    """

    bbox: tuple[int, int, int, int]
    circle: tuple[int, int, int] | None = None
    label: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        x, y, w, h = self.bbox
        if w <= 0 or h <= 0:
            raise ValueError(
                f"Region bbox must have positive width and height, got {self.bbox!r}"
            )
        if x < 0 or y < 0:
            raise ValueError(
                f"Region bbox must have non-negative origin, got {self.bbox!r}"
            )


# A ``RegionDetector`` is any callable that maps an image to a list of regions.
# The pipeline accepts either the default :func:`detect_regions` below or any
# user-supplied callable with the same signature.
RegionDetector = Callable[[np.ndarray], Sequence[Region]]


def detect_regions(img: np.ndarray) -> list[Region]:
    """Return the egg-laying regions in ``img``.

    **This is the default implementation, and it is deliberately naive:**
    it returns a single region covering the entire image. Replace it with
    real logic for your chamber design.

    Parameters
    ----------
    img : numpy.ndarray
        The full-resolution image as a uint8 array in BGR or RGB order (the
        pipeline doesn't depend on the channel order — SplineDist normalizes
        its input — but consistency matters if you want to draw annotations
        back on the same image).

    Returns
    -------
    list[Region]
        One ``Region`` per egg-laying area.

    Notes
    -----
    Common patterns for hardcoded implementations:

    >>> # A single centered circular region covering 80% of the image
    >>> h, w = img.shape[:2]
    >>> cx, cy = w // 2, h // 2
    >>> r = int(0.4 * min(w, h))
    >>> return [Region(bbox=(cx - r, cy - r, 2 * r, 2 * r),
    ...                circle=(cx, cy, r),
    ...                label="chamber_1")]

    >>> # A fixed 3x3 grid of square regions
    >>> rows, cols = 3, 3
    >>> cell_w, cell_h = w // cols, h // rows
    >>> return [
    ...     Region(bbox=(c * cell_w, r * cell_h, cell_w, cell_h),
    ...            label=f"r{r}c{c}")
    ...     for r in range(rows)
    ...     for c in range(cols)
    ... ]
    """
    h, w = img.shape[:2]
    return [Region(bbox=(0, 0, w, h), label="full_image")]
