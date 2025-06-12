import glob
import logging
from multiview_stitcher import spatial_image_utils as si_utils
import numpy as np
import yaml

from src.MVSRegistration import MVSRegistration
from src.image.util import *
from src.registration_methods.RegistrationMethodSkFeatures import RegistrationMethodSkFeatures as RegMethod


def test_feature_registration(params, operation):
    target_scale = 0
    input_path = operation['input']
    if operation['operation'].endswith('S'):
        # debugging: simulate matching
        input_path = input_path.replace('/S???/', '/S000/')
    filenames = glob.glob(input_path)
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
        #print(result)


if __name__ == "__main__":
    params = 'resources/params_EMPIAR12193.yml'
    with open(params, 'r', encoding='utf8') as file:
        params = yaml.safe_load(file)

    logging.basicConfig(level=logging.INFO)

    operation = params['operations'][0]

    test_feature_registration(params, operation)
