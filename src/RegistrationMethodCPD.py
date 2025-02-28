import logging
import numpy as np
from probreg import cpd

from src.RegistrationMethod import RegistrationMethod
from src.image.util import detect_area_points
from src.util import points_to_3d


class RegistrationMethodCPD(RegistrationMethod):
    def registration(self, fixed_data, moving_data, **kwargs) -> dict:
        max_iter = kwargs.get('max_iter', 1000)

        fixed_points = points_to_3d([point for point, area in detect_area_points(fixed_data.data.astype(self.source_type))])
        moving_points = points_to_3d(
            [point for point, area in detect_area_points(moving_data.data.astype(self.source_type))])

        if len(moving_points) > 1 and len(fixed_points) > 1:
            result_cpd = cpd.registration_cpd(moving_points, fixed_points, maxiter=max_iter)
            transformation = result_cpd.transformation
            S = transformation.scale * np.eye(3)
            R = transformation.rot
            T = np.eye(3) + np.hstack([np.zeros((3, 2)), transformation.t.reshape(-1, 1)])
            transform = T @ R @ S
        else:
            logging.error('Not enough points for CPD registration')
            transform = np.eye(3)

        return {
            "affine_matrix": transform,  # homogenous matrix of shape (ndim + 1, ndim + 1), axis order (z, y, x)
            "quality": 1  # float between 0 and 1 (if not available, set to 1.0)
        }
