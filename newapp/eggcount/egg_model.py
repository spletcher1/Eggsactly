"""Thin wrapper around the SplineDist egg-counting model.

The upstream project loaded two separate SplineDist models (arena detector
and egg counter). Here we only use the egg counter — the arena detector's
role is filled by :mod:`eggcount.chamber_detection` instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from csbdeep.utils import normalize

from .splinedist.config import Config
from .splinedist.models.model2d import SplineDist2D
from .torch_utils import load_state_dict_compat

# Default config (one of the JSONs under newapp/configs/). Users with a
# custom-trained egg model should pass their own config path to EggModel.
DEFAULT_CONFIG_NAME = "unet_backbone_rand_zoom.json"

# Percentile normalization range used by the upstream worker before inference.
# See project/gpu_backend/worker.py:221 in the upstream project.
_NORM_LO = 1.0
_NORM_HI = 99.8


@dataclass
class EggPredictions:
    """Raw result of running the egg counter on a single image.

    Attributes
    ----------
    count : int
        Number of detected eggs. This is just ``len(points)``.
    points : numpy.ndarray, shape (N, 2)
        Centroid coordinates of each detected egg, in pixels of the crop
        that was passed to :meth:`EggModel.count`.
    prob : numpy.ndarray, shape (N,)
        Per-detection confidence score from the model.
    outlines : list
        Per-detection polygonal outlines (as ``(y, x)`` floats) in the
        coordinate frame of the crop. Suitable for drawing with
        ``cv2.polylines``.
    """

    count: int
    points: np.ndarray
    prob: np.ndarray
    outlines: list


def _compute_n_tiles(img: np.ndarray) -> list[int]:
    """Adaptive tiling that matches the upstream worker's heuristic.

    SplineDist tiles its input to stay under GPU memory on large images.
    The rule used in the upstream worker: any spatial dim ≥ 1600 px gets
    ``2 + (dim - 1600) // 300`` tiles; channel dim is always 1.
    """
    n_tiles = [1, 1, 1]
    for dim in range(2):
        if img.shape[dim] >= 1600:
            n_tiles[dim] = 2 + (img.shape[dim] - 1600) // 300
    return n_tiles


class EggModel:
    """Loads a trained SplineDist egg counter and runs inference on crops.

    Parameters
    ----------
    weights_path : str or Path
        Path to the ``.pth`` weights file. In newapp, drop trained weights
        under ``newapp/models/`` and pass the full path here.
    config_path : str, optional
        Path (or bare filename under ``newapp/configs/``) to the config
        JSON. Defaults to ``unet_backbone_rand_zoom.json`` — the config
        used by the upstream egg model.
    n_channel_in : int
        Number of input channels the model expects. The upstream model is
        3-channel (BGR). If you retrain on grayscale, pass ``1``.
    device : str, optional
        ``"cuda"``, ``"cpu"``, or ``None`` to auto-select CUDA when
        available. Inference is much faster on a GPU; the model *will*
        run on CPU for testing but will be slow.
    """

    def __init__(
        self,
        weights_path: str | Path,
        config_path: str | None = None,
        n_channel_in: int = 3,
        device: str | None = None,
    ) -> None:
        import torch

        weights_path = Path(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Egg model weights not found: {weights_path}\n"
                "Drop the trained .pth file into newapp/models/ (or pass an "
                "explicit path) before constructing EggModel."
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        cfg = Config(config_path or DEFAULT_CONFIG_NAME, n_channel_in=n_channel_in)
        model = SplineDist2D(cfg, train=False)
        if self.device.type == "cuda":
            model.cuda()
        else:
            model.to(self.device)
        model.train(False)

        state = load_state_dict_compat(str(weights_path), map_location=self.device)
        model.load_state_dict(state)
        self.model = model
        self.config = cfg

    def count(self, img: np.ndarray) -> EggPredictions:
        """Run the egg counter on a single image crop.

        ``img`` should be a uint8 (or float) BGR image — the crop of a
        single egg-laying region. The image is percentile-normalized with
        the same settings the upstream worker uses before inference.

        Returns an :class:`EggPredictions` with the count, per-egg
        centroids, probabilities, and pixel-space outlines.
        """
        # Avoid the upstream hazard of a zero-size crop: guard explicitly.
        if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
            return EggPredictions(
                count=0, points=np.empty((0, 2)), prob=np.empty((0,)), outlines=[]
            )

        # csbdeep percentile normalization — same as upstream worker.
        norm_img = normalize(img.astype(np.float32), _NORM_LO, _NORM_HI, axis=(0, 1))
        n_tiles = _compute_n_tiles(norm_img)

        # SplineDist returns (label_image, detail_dict). We only need the
        # detail dict; the label image is an instance segmentation mask we
        # could draw if we wanted.
        _, details = self.model.predict_instances(norm_img, n_tiles=n_tiles)

        from .drawing import get_interpolated_points

        coord = details["coord"]
        points = details["points"]
        prob = details["prob"]
        outlines = get_interpolated_points(coord) if len(points) else []

        return EggPredictions(
            count=int(len(points)),
            points=np.asarray(points),
            prob=np.asarray(prob),
            outlines=outlines,
        )
