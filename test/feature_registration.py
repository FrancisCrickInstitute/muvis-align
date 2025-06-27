import glob
import logging
from multiview_stitcher import spatial_image_utils as si_utils
import numpy as np
import yaml

from src.MVSRegistration import MVSRegistration
from src.image.source_helper import create_dask_source
from src.image.util import *
from src.registration_methods.RegistrationMethodSkFeatures import RegistrationMethodSkFeatures as RegMethod


def test_feature_registration():
    params = 'resources/params_EMPIAR12193.yml'
    with open(params, 'r', encoding='utf8') as file:
        params = yaml.safe_load(file)

    logging.basicConfig(level=logging.INFO)

    operation = params['operations'][1]

    target_scale = 4
    input_path = operation['input']
    filenames = glob.glob(input_path)[:2]
    if len(filenames) == 0:
        raise FileNotFoundError(f"No files found for pattern: {input_path}")
    reg = MVSRegistration(params['general'])
    sims, _, _, _ = reg.init_sims(filenames, operation, target_scale=target_scale)
    norm_sims, _ = reg.preprocess(sims, operation)
    sim0 = norm_sims[0]
    reg_method = RegMethod(sim0.dtype, operation['method'])

    origins = np.array([si_utils.get_origin_from_sim(sim, asarray=True) for sim in sims])
    size = get_sim_physical_size(sim0)
    pairs, _ = get_orthogonal_pairs(origins, size)
    for pair in pairs:
        overlap_sims = reg.get_overlap_images((norm_sims[pair[0]], norm_sims[pair[1]]), reg.source_transform_key)
        result = reg_method.registration(*overlap_sims)
        print(result)


def test_feature_registration_simple():
    folder = 'D:/slides/12193/data_overlaps/'
    filenames = [
        folder + 'slice_37.tiff',
        folder + 'slice_46.tiff'
    ]
    reg_params = {
        'gaussian_sigma': 6,
        'downscale_factor': 1.414,
        'inlier_threshold_factor': 0.05,
        'max_trials': 10000,
        'ransac_iterations': 10,
    }

    images = [create_dask_source(filename).get_data() for filename in filenames]
    image0 = images[0]

    reg_method = RegMethod(image0, reg_params)
    result = reg_method.registration(*images)
    print(result)


if __name__ == "__main__":

    #test_feature_registration()
    test_feature_registration_simple()
