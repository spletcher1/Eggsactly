"""SplineDist patch dataset for training.

Loads full images + instance-label masks, samples random patches on the fly,
and computes the SplineDist probability and distance targets that the loss
function expects.

This replaces the upstream ``SplineDistData2D`` class (which was deleted
as it had broken top-level imports) with a simpler, self-contained
implementation that leans on the existing utility functions in
``eggcount.splinedist``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# Make sure eggcount is importable from training scripts
_NEWAPP_ROOT = Path(__file__).resolve().parent.parent
if str(_NEWAPP_ROOT) not in sys.path:
    sys.path.insert(0, str(_NEWAPP_ROOT))

from eggcount.splinedist.geometry.geom2d import spline_dist
from eggcount.splinedist.utils import edt_prob, fill_label_holes


def load_image_label_pairs(
    images_dir: str | Path,
    labels_dir: str | Path,
    image_exts: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load matched image + label pairs from two directories.

    Pairing is by stem name: ``images/foo.jpg`` pairs with
    ``labels/foo.png``.  Label images must be single-channel integer
    arrays where each object has a unique non-zero ID and background
    is 0.

    Returns ``(images, labels)`` — two parallel lists of numpy arrays.
    """
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    img_stems: dict[str, Path] = {}
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() in image_exts:
            img_stems[p.stem] = p

    images, labels = [], []
    for stem, img_path in sorted(img_stems.items()):
        # Try common label extensions
        lbl_path = None
        for ext in (".png", ".tif", ".tiff"):
            candidate = labels_dir / f"{stem}{ext}"
            if candidate.is_file():
                lbl_path = candidate
                break
        if lbl_path is None:
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"warning: could not read image {img_path}, skipping")
            continue
        lbl = cv2.imread(str(lbl_path), cv2.IMREAD_UNCHANGED)
        if lbl is None:
            print(f"warning: could not read label {lbl_path}, skipping")
            continue
        # Ensure label is single-channel integer
        if lbl.ndim == 3:
            lbl = lbl[:, :, 0]
        lbl = lbl.astype(np.int32)
        lbl = fill_label_holes(lbl)

        images.append(img)
        labels.append(lbl)

    if not images:
        raise FileNotFoundError(
            f"No matched image/label pairs found in {images_dir} + {labels_dir}"
        )
    print(f"loaded {len(images)} image/label pairs")
    return images, labels


class SplineDistPatchDataset:
    """Yields random patches with precomputed SplineDist targets.

    Each call to :meth:`sample_batch` returns a batch of:

    * ``patches``       — ``(B, H, W, C)`` float32 image patches
    * ``prob_targets``  — ``(B, H', W')`` float32 EDT-based prob maps
    * ``dist_targets``  — ``(B, H', W', 2*contoursize_max + 1)`` float32,
      where the last channel is a binary mask
    * ``counts``        — ``(B,)`` int, number of objects per patch

    ``H', W'`` are ``H // grid, W // grid`` (subsampled).

    Parameters
    ----------
    images : list of ndarray
        Full input images (BGR uint8).
    labels : list of ndarray
        Matching instance-label images (int32, background=0).
    patch_size : tuple of int
        ``(height, width)`` of each training patch.
    contoursize_max : int
        Maximum contour length across the dataset. Compute with
        :func:`eggcount.splinedist.utils.get_contoursize_max`.
    grid : tuple of int
        Subsampling factor for targets, e.g. ``(2, 2)``.
    foreground_prob : float
        Probability that a sampled patch will be centered on a foreground
        object. ``0.9`` is the upstream default.
    augmenter : callable, optional
        ``augmenter(image, label) -> (image, label)`` applied after
        cropping.
    """

    def __init__(
        self,
        images: list[np.ndarray],
        labels: list[np.ndarray],
        patch_size: tuple[int, int],
        contoursize_max: int,
        grid: tuple[int, int] = (2, 2),
        foreground_prob: float = 0.9,
        augmenter=None,
    ):
        self.images = images
        self.labels = labels
        self.patch_size = patch_size
        self.contoursize_max = int(contoursize_max)
        self.grid = grid
        self.foreground_prob = foreground_prob
        self.augmenter = augmenter

        # Precompute per-image object centroids for foreground-biased sampling
        from skimage.measure import regionprops

        self.centroids: list[np.ndarray] = []
        for lbl in labels:
            regs = regionprops(lbl)
            if regs:
                self.centroids.append(
                    np.array([r.centroid for r in regs]).astype(int)
                )
            else:
                self.centroids.append(np.empty((0, 2), dtype=int))

    def _sample_patch_location(
        self, img_shape: tuple[int, ...], centroids: np.ndarray
    ) -> tuple[int, int]:
        """Pick a top-left corner for a random patch."""
        ph, pw = self.patch_size
        h, w = img_shape[:2]
        if (
            len(centroids) > 0
            and np.random.rand() < self.foreground_prob
        ):
            # Center patch on a random object
            idx = np.random.randint(len(centroids))
            cy, cx = centroids[idx]
            y = int(np.clip(cy - ph // 2, 0, max(h - ph, 0)))
            x = int(np.clip(cx - pw // 2, 0, max(w - pw, 0)))
        else:
            y = np.random.randint(0, max(h - ph, 1))
            x = np.random.randint(0, max(w - pw, 1))
        return y, x

    def _compute_targets(
        self, label_patch: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Compute prob and dist targets for a label patch."""
        prob = edt_prob(label_patch)
        dist = spline_dist(label_patch, self.contoursize_max)
        # dist shape: (H, W, contoursize_max * 2)

        # Subsample by grid factor
        gy, gx = self.grid
        prob = prob[::gy, ::gx]
        dist = dist[::gy, ::gx]

        # Append a binary mask channel (1 where foreground)
        mask = (label_patch[::gy, ::gx] > 0).astype(np.float32)[..., np.newaxis]
        dist_with_mask = np.concatenate([dist, mask], axis=-1)

        n_objects = len(np.unique(label_patch)) - 1  # exclude background
        return prob, dist_with_mask, max(n_objects, 0)

    def sample_one(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Sample a single random patch with its targets."""
        idx = np.random.randint(len(self.images))
        img = self.images[idx]
        lbl = self.labels[idx]

        ph, pw = self.patch_size
        # If image is smaller than patch, pad
        h, w = img.shape[:2]
        if h < ph or w < pw:
            pad_h = max(ph - h, 0)
            pad_w = max(pw - w, 0)
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            lbl = np.pad(lbl, ((0, pad_h), (0, pad_w)), mode="constant")

        y, x = self._sample_patch_location(img.shape, self.centroids[idx])
        img_patch = img[y : y + ph, x : x + pw].astype(np.float32)
        lbl_patch = lbl[y : y + ph, x : x + pw]

        if self.augmenter is not None:
            img_patch, lbl_patch = self.augmenter(img_patch, lbl_patch)

        prob, dist_with_mask, n_obj = self._compute_targets(lbl_patch)
        return img_patch, prob, dist_with_mask, n_obj

    def sample_batch(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of patches with targets.

        Returns
        -------
        patches : ndarray, shape (B, H, W, C)
        prob : ndarray, shape (B, H', W')
        dist : ndarray, shape (B, H', W', 2*contoursize_max + 1)
        counts : ndarray, shape (B,)
        """
        items = [self.sample_one() for _ in range(batch_size)]
        patches = np.stack([it[0] for it in items])
        prob = np.stack([it[1] for it in items])
        dist = np.stack([it[2] for it in items])
        counts = np.array([it[3] for it in items])
        return patches, prob, dist, counts
