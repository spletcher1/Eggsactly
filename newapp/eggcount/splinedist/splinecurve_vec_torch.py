import torch

from .spline_generator import SplineCurve


class SplineCurveVectorizedTorch(SplineCurve):
    def sampleSequential(self, phi):
        contour_points = torch.matmul(phi, self.coefs)
        return contour_points
