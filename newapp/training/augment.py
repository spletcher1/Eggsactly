"""Simple image + label augmentation for SplineDist training.

Augmentations must be applied identically to both the image and the label
mask so that the per-pixel distance targets remain correct. All functions
here operate on ``(image, label)`` tuples in-place (or return new arrays)
and preserve label IDs.
"""

from __future__ import annotations

import cv2
import numpy as np


def random_flip_lr(image: np.ndarray, label: np.ndarray):
    """Randomly flip both arrays left-right with 50% probability."""
    if np.random.rand() > 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        label = np.ascontiguousarray(label[:, ::-1])
    return image, label


def random_flip_ud(image: np.ndarray, label: np.ndarray):
    """Randomly flip both arrays top-bottom with 50% probability."""
    if np.random.rand() > 0.5:
        image = np.ascontiguousarray(image[::-1, :])
        label = np.ascontiguousarray(label[::-1, :])
    return image, label


def random_rot90(image: np.ndarray, label: np.ndarray):
    """Randomly rotate both arrays by 0, 90, 180, or 270 degrees."""
    k = np.random.randint(4)
    if k > 0:
        image = np.rot90(image, k, axes=(0, 1)).copy()
        label = np.rot90(label, k, axes=(0, 1)).copy()
    return image, label


def random_intensity(image: np.ndarray, label: np.ndarray):
    """Randomly adjust brightness and contrast of the image.

    Label is passed through unchanged so the signature stays uniform.
    """
    # Brightness shift
    image = image + np.random.uniform(-25, 25)
    # Contrast scale
    image = image * np.random.uniform(0.8, 1.2)
    image = np.clip(image, 0, 255)
    return image, label


def default_augmenter(image: np.ndarray, label: np.ndarray):
    """Chain of augmentations suitable for egg-counting images.

    Pass this as the ``augmenter`` argument to
    :class:`training.dataset.SplineDistPatchDataset`.
    """
    image, label = random_flip_lr(image, label)
    image, label = random_flip_ud(image, label)
    image, label = random_rot90(image, label)
    image, label = random_intensity(image, label)
    return image, label
