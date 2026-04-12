"""Drawing / outline utilities.

Extracted from the upstream project/lib/image/drawing.py but trimmed to only
the outline interpolation needed for the inference pipeline. Rendering code
(annotated images) is in pipeline.py.
"""

from __future__ import annotations

import os

import numpy as np

from . import splinedist
from .splinedist import spline_generator as sg
from .splinedist.path_helpers import data_dir

_PHI_CACHE: np.ndarray | None = None


def _phi() -> np.ndarray:
    """Lazily load the ``phi_8.npy`` spline basis used for outline interpolation."""
    global _PHI_CACHE
    if _PHI_CACHE is None:
        _PHI_CACHE = np.load(os.path.join(data_dir(), "phi_" + str(8) + ".npy"))
    return _PHI_CACHE


def get_interpolated_points(data: np.ndarray, n_points: int = 30) -> list[list[list[float]]]:
    """Convert SplineDist coord arrays into pixel-space outlines.

    ``data`` has shape ``(N, 2, n_control_points)`` — the output of
    ``predict_instances()["coord"]``. Returns a list of N outlines, each of
    which is a list of ``(y, x)`` floating-point points along the object
    boundary (already shifted by +1 to match the upstream convention).
    """
    M = np.shape(data)[2]
    spline_contour = sg.SplineCurveVectorized(
        M, sg.B3(), True, np.transpose(data, [0, 2, 1])
    )
    more_coords = spline_contour.sampleSequential(_phi())
    sampling_interval = max(more_coords.shape[1] // n_points, 1)
    sampled_points = more_coords[:, slice(0, more_coords.shape[1], sampling_interval)]
    sampled_points = np.add(sampled_points, 1)
    return sampled_points.astype(float).tolist()
