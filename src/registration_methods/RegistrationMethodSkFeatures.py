# https://scikit-image.org/docs/stable/api/skimage.feature.html
# https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_brief.html

import logging
import numpy as np
from multiview_stitcher import param_utils
from skimage.feature import match_descriptors, SIFT, ORB
from skimage.measure import ransac
from skimage.transform import rescale, AffineTransform
from spatial_image import SpatialImage

from src.image.util import *
from src.metrics import calc_match_metrics
from src.registration_methods.RegistrationMethod import RegistrationMethod


class RegistrationMethodSkFeatures(RegistrationMethod):
    def __init__(self, source_type):
        super().__init__(source_type)
        self.feature_model = ORB()
        #self.feature_model = SIFT(c_dog=0.1 / 3)
        self.feature_store = {}

    def detect_features(self, data0):
        data = self.convert_data_to_float(data0)
        scale = min(1000 / np.linalg.norm(data.shape), 1)
        data = rescale(data, scale)

        self.feature_model.detect_and_extract(data)
        points = np.flip(self.feature_model.keypoints, axis=-1) / scale     # rescale and convert to (z)yx
        desc = self.feature_model.descriptors

        show_image(draw_keypoints(data, np.flip(self.feature_model.keypoints, axis=-1)))

        return points, desc

    def registration(self, fixed_data: SpatialImage, moving_data: SpatialImage, **kwargs) -> dict:
        min_samples = 5
        fixed_points, fixed_desc = self.detect_features(fixed_data)
        moving_points, moving_desc = self.detect_features(moving_data)
        threshold = get_mean_nn_distance(fixed_points, moving_points)

        matches = match_descriptors(fixed_desc, moving_desc, cross_check=True, max_ratio=0.92)

        transform = None
        quality = 0
        if len(matches) >= min_samples:
            fixed_points2 = np.array([fixed_points[match[0]] for match in matches])
            moving_points2 = np.array([moving_points[match[1]] for match in matches])
            transform, inliers = ransac((fixed_points2, moving_points2), AffineTransform, min_samples=min_samples,
                                               residual_threshold=threshold, max_trials=1000)

            show_image(draw_keypoint_matches(fixed_data.astype(self.source_type), fixed_points,
                                             moving_data.astype(self.source_type), moving_points,
                                             matches, inliers))

            if transform is not None and not np.any(np.isnan(transform)):
                transform = np.array(transform)
                fixed_points3 = [point for point, is_inlier in zip(fixed_points2, inliers) if is_inlier]
                moving_points3 = [point for point, is_inlier in zip(moving_points2, inliers) if is_inlier]
                metrics = calc_match_metrics(fixed_points3, moving_points3, transform, threshold)
                #quality = np.mean(inliers)
                quality = metrics['nmatches'] / min(len(fixed_points2), len(moving_points2))

        if not validate_transform(transform, get_sim_physical_size(fixed_data, invert=True)):
            logging.error('Unable to find feature-based registration')
            transform = np.eye(3)

        return {
            "affine_matrix": param_utils.invert_coordinate_order(transform),  # homogenous matrix of shape (ndim + 1, ndim + 1), axis order (z, y, x)
            "quality": quality  # float between 0 and 1 (if not available, set to 1.0)
        }
