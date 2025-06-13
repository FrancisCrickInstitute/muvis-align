# https://scikit-image.org/docs/stable/api/skimage.feature.html
# https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_orb.html
# https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_brief.html

import logging
import numpy as np
from multiview_stitcher import param_utils
from skimage.feature import match_descriptors, ORB
from skimage.filters import gaussian
from skimage.measure import ransac
from skimage.transform import EuclideanTransform
from spatial_image import SpatialImage

from src.image.util import *
from src.registration_methods.RegistrationMethod import RegistrationMethod


class RegistrationMethodSkFeatures(RegistrationMethod):
    def __init__(self, source_type, params):
        super().__init__(source_type, params)

        downscale = params.get('downscale_factor', params.get('downscale', np.sqrt(2)))
        self.feature_model = ORB(n_keypoints=5000, downscale=downscale)
        self.gaussian_sigma = params.get('gaussian_sigma', params.get('sigma', 1))

        self.label = 'matches_slice_'
        self.counter = 0

    def detect_features(self, data0):
        points = []
        desc = []

        data = self.convert_data_to_float(data0)
        data = norm_image_variance(data)
        data = gaussian(data, sigma=self.gaussian_sigma)

        try:
            self.feature_model.detect_and_extract(data)
            points = self.feature_model.keypoints
            desc = self.feature_model.descriptors
            if len(points) == 0:
                logging.error('No features detected!')
        except RuntimeError as e:
            logging.error(e)

        if len(points) < 20:
            # TODO: if #points is too low: alternative feature detection?
            logging.warning(f'Low number of features: {len(points)}')

        #inliers = filter_edge_points(points, np.flip(data0.shape[:2]))
        #points = points[inliers]
        #desc = desc[inliers]

        #show_image(draw_keypoints(data, np.flip(self.feature_model.keypoints, axis=-1)))

        return points, desc, data

    def match(self, fixed_points, fixed_desc, moving_points, moving_desc,
              min_matches, cross_check, lowe_ratio, inlier_threshold, max_offset):
        transform = None
        quality = 0
        inliers = None

        matches = match_descriptors(fixed_desc, moving_desc, cross_check=cross_check, max_ratio=lowe_ratio)
        if len(matches) >= min_matches:
            fixed_points2 = np.array([fixed_points[match[0]] for match in matches])
            moving_points2 = np.array([moving_points[match[1]] for match in matches])
            transform, inliers = ransac((fixed_points2, moving_points2), EuclideanTransform,
                                        min_samples=min_matches,
                                        residual_threshold=inlier_threshold,
                                        max_trials=1000)
            if validate_transform(transform, max_offset):
                quality = np.mean(inliers)
        return transform, quality, matches, inliers

    def registration(self, fixed_data: SpatialImage, moving_data: SpatialImage, **kwargs) -> dict:
        counter = self.counter
        self.counter += 1

        transform = np.eye(3)
        quality = 0

        lowe_ratio = 0.92
        mean_size = np.mean([np.linalg.norm(data.shape) / np.sqrt(2) for data in [fixed_data, moving_data]])
        inlier_threshold = mean_size * 0.05
        min_matches = 5
        max_offset = dict_to_xyz(fixed_data.sizes, 'zyx')

        fixed_points, fixed_desc, fixed_data2 = self.detect_features(fixed_data)
        moving_points, moving_desc, moving_data2 = self.detect_features(moving_data)

        if len(fixed_desc) > 0 and len(moving_desc) > 0:
            transform, quality, matches, inliers = self.match(fixed_points, fixed_desc, moving_points, moving_desc,
                                            min_matches=min_matches, cross_check=True,
                                            lowe_ratio=lowe_ratio, inlier_threshold=inlier_threshold,
                                            max_offset=max_offset)

            if quality == 0:
                print('Matching failed')

                draw_keypoints_matches(fixed_data2, fixed_points,
                                       moving_data2, moving_points,
                                       matches, inliers,
                                       show_plot=False, output_filename=self.label + str(counter) + '.tiff')

            #if transform is not None and not np.any(np.isnan(transform)):
            #    print('translation', transform.translation, 'rotation', np.rad2deg(transform.rotation),
            #          'quality', quality)
        if quality == 0:
            logging.error('Unable to find feature-based registration')
            transform = np.eye(3)

        return {
            "affine_matrix": np.array(transform),  # homogenous matrix of shape (ndim + 1, ndim + 1), axis order (z, y, x)
            "quality": quality  # float between 0 and 1 (if not available, set to 1.0)
        }
