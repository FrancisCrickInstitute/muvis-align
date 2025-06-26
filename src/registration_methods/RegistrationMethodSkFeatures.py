# https://scikit-image.org/docs/stable/api/skimage.feature.html
# https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_orb.html
# https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_brief.html

from datetime import datetime
import logging
import numpy as np
from multiview_stitcher import param_utils
from skimage.feature import match_descriptors, ORB
from skimage.filters import gaussian
from skimage.measure import ransac
from skimage.transform import EuclideanTransform
from spatial_image import SpatialImage

from src.image.ome_tiff_helper import save_tiff
from src.image.util import *
from src.registration_methods.RegistrationMethod import RegistrationMethod


class RegistrationMethodSkFeatures(RegistrationMethod):
    def __init__(self, source, params):
        super().__init__(source, params)

        self.nkeypoints = params.get('nkeypoints', 5000)
        self.downscale_factor = params.get('downscale_factor', params.get('downscale', np.sqrt(2)))
        self.full_size_gaussian_sigma = params.get('gaussian_sigma', params.get('sigma', 1))
        self.max_trails = params.get('max_trials', 10000)
        self.ransac_iterations = params.get('ransac_iterations', 10)
        self.inlier_threshold_factor = params.get('inlier_threshold_factor', 0.05)

    def detect_features(self, data0, gaussian_sigma=None):
        points = []
        desc = []

        data = self.convert_data_to_float(data0)
        data = norm_image_variance(data)
        if gaussian_sigma:
            data = gaussian(data, sigma=gaussian_sigma)

        try:
            # not thread-safe - create instance that is not re-used in other thread
            feature_model = ORB(n_keypoints=self.nkeypoints, downscale=self.downscale_factor)
            feature_model.detect_and_extract(data)
            points = feature_model.keypoints
            desc = feature_model.descriptors
            if len(points) == 0:
                logging.error('No features detected!')
        except RuntimeError as e:
            logging.error(e)

        if len(points) < self.nkeypoints / 100:
            # TODO: if #points is too low: alternative feature detection?
            logging.warning(f'Low number of features: {len(points)}')

        #inliers = filter_edge_points(points, np.flip(data0.shape[:2]))
        #points = points[inliers]
        #desc = desc[inliers]

        #show_image(draw_keypoints(data, np.flip(self.feature_model.keypoints, axis=-1)))

        return points, desc, data

    def match(self, fixed_points, fixed_desc, moving_points, moving_desc,
              min_matches, cross_check, lowe_ratio, inlier_threshold, mean_size_dist):
        transform = None
        quality = 0
        inliers = []

        matches = match_descriptors(fixed_desc, moving_desc, cross_check=cross_check, max_ratio=lowe_ratio)
        if len(matches) >= min_matches:
            fixed_points2 = np.array([fixed_points[match[0]] for match in matches])
            moving_points2 = np.array([moving_points[match[1]] for match in matches])

            transforms = []
            inliers_list = []
            translations = []
            tot_weight = 0
            tot_translation = None
            for i in range(self.ransac_iterations):
                transform, inliers = ransac((fixed_points2, moving_points2), EuclideanTransform,
                                            min_samples=min_matches,
                                            residual_threshold=inlier_threshold,
                                            max_trials=self.max_trails)
                if inliers is None:
                    inliers = []
                if len(inliers) > 0 and validate_transform(transform):
                    weight = np.mean(inliers)
                    weighted_translation = transform.translation * weight
                    tot_weight += weight
                    if tot_translation is None:
                        tot_translation = weighted_translation
                    else:
                        tot_translation += weighted_translation
                    translations.append(transform.translation)
                    transforms.append(transform)
                    inliers_list.append(inliers)
                    quality += (np.sum(inliers) / self.nkeypoints) ** (1/3) # ^1/3 to decrease sensitivity

            quality /= self.ransac_iterations

            if tot_weight > 0:
                mean_translation = tot_translation / tot_weight
                best_index = np.argmin(np.linalg.norm(translations - mean_translation, axis=1))
                transform = transforms[best_index]
                inliers = inliers_list[best_index]
                quality *= 1 - (np.linalg.norm(np.std(translations, axis=0)) / mean_size_dist) ** 3  # ^3 to increase sensitivity
                #print('norm translation', mean_translation / mean_size_dist, 'norm SD', np.linalg.norm(np.std(translations, axis=0)) / mean_size_dist)
            #print('%inliers', np.mean(inliers), '#good ransac iterations', len(inliers_list))

        return transform, quality, matches, inliers

    def registration(self, fixed_data: SpatialImage, moving_data: SpatialImage, **kwargs) -> dict:
        eye_transform = np.eye(self.ndims + 1)
        transform = eye_transform
        quality = 0
        matches = []
        inliers = []

        lowe_ratio = 0.92
        full_size_dist = np.linalg.norm(self.full_size)
        mean_size_dist = np.mean([np.linalg.norm(data.shape) for data in [fixed_data, moving_data]])
        scale = mean_size_dist / full_size_dist
        gaussian_sigma = self.full_size_gaussian_sigma * (scale ** (1/3))
        mean_size = np.mean([np.linalg.norm(data.shape) / np.sqrt(self.ndims) for data in [fixed_data, moving_data]])
        inlier_threshold = mean_size * self.inlier_threshold_factor
        min_matches = 10

        fixed_points, fixed_desc, fixed_data2 = self.detect_features(fixed_data, gaussian_sigma)
        moving_points, moving_desc, moving_data2 = self.detect_features(moving_data, gaussian_sigma)

        if len(fixed_desc) > 0 and len(moving_desc) > 0:
            transform, quality, matches, inliers = self.match(fixed_points, fixed_desc, moving_points, moving_desc,
                                                              min_matches=min_matches, cross_check=True,
                                                              lowe_ratio=lowe_ratio, inlier_threshold=inlier_threshold,
                                                              mean_size_dist=mean_size_dist)
        #print(f'#keypoints: {len(fixed_desc)},{len(moving_desc)}'
        #      f' #matches: {len(matches)} #inliers: {np.sum(inliers):.0f} quality: {quality:.3f}')

        # for debugging:
        #output_filename = 'matches_slice_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        #save_tiff(output_filename + '_f.tiff', fixed_data.astype(self.source_type))
        #save_tiff(output_filename + '_m.tiff', moving_data.astype(self.source_type))
        #if np.sum(inliers) > 0:
        #    draw_keypoints_matches_sk(fixed_data2, fixed_points,
        #                              moving_data2, moving_points,
        #                              matches[inliers],
        #                              show_plot=False, output_filename=output_filename + '_i.tiff')
        #draw_keypoints_matches(fixed_data2, fixed_points,
        #                       moving_data2, moving_points,
        #                       matches, inliers,
        #                       show_plot=False, output_filename=output_filename + '.tiff')

        if quality == 0 or np.sum(inliers) == 0:
            logging.error('Unable to find feature-based registration')
            transform = eye_transform
            quality = 0.1001   # don't drop completely

        return {
            "affine_matrix": np.array(transform),  # homogenous matrix of shape (ndim + 1, ndim + 1), axis order (z, y, x)
            "quality": quality  # float between 0 and 1 (if not available, set to 1.0)
        }
