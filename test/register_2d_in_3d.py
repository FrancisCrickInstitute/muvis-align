# example of embedding 2d sim into 3d with correct transforms
# for registering slices in a 3d stack

import numpy as np
from copy import deepcopy
import dask
from matplotlib import pyplot as plt
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher import registration, param_utils, msi_utils, fusion
from multiview_stitcher.spatial_image_utils import DEFAULT_TRANSFORM_KEY

import matplotlib

#matplotlib.use('TkAgg')

from src.registration_methods.RegistrationMethodSkFeatures import RegistrationMethodSkFeatures


dask.config.set(scheduler='threads')


def register_3d_sims_in_2d(
    fixed_data,
    moving_data,
    *,
    fixed_origin,
    moving_origin,
    fixed_spacing,
    moving_spacing,
    initial_affine,
    transform_types=None,
    **ants_registration_kwargs,
    ):
    """
    Register two 3d sims by projecting them to 2d and using 2d registration.
    The z component of the resulting affine matrix is set to identity.
    """
    fixed_data = fixed_data.max('z')
    moving_data = moving_data.max('z')

    dims2d = ['y', 'x']
    fixed_origin = {dim: fixed_origin[dim] for dim in dims2d}
    moving_origin = {dim: moving_origin[dim] for dim in dims2d}
    fixed_spacing = {dim: fixed_spacing[dim] for dim in dims2d}
    moving_spacing = {dim: moving_spacing[dim] for dim in dims2d}

    initial_affine = initial_affine[1: , 1:]

    # call 2d registration on the projected sims
    # reg_res_2d = registration.phase_correlation_registration(
    #     sim1, sim2, **kwargs)
    reg_res_2d = registration.registration_ANTsPy(
        fixed_data,
        moving_data,
        fixed_origin=fixed_origin,
        moving_origin=moving_origin,
        fixed_spacing=fixed_spacing,
        moving_spacing=moving_spacing,
        initial_affine=initial_affine,
        transform_types=transform_types,
        **ants_registration_kwargs,
        )

    # embed resulting 2d affine matrix into 3d affine matrix
    reg_res_3d = deepcopy(reg_res_2d)
    reg_res_3d['affine_matrix'] = param_utils.identity_transform(3)
    reg_res_3d['affine_matrix'][1:, 1:] = reg_res_2d['affine_matrix']

    return reg_res_3d


def calc_shape_pattern(y, x, offset):
    h, w = y.shape
    xoffset, yoffset = offset
    image = (1 + np.cos((xoffset + x / w) * np.pi)
             * np.cos((yoffset + y / h) * np.pi)) / 2
    image[image < 0.01] = 1
    return image

# create simulated 2d sims
size = (100, 100)
#data = np.random.random((1, 100, 100))

dz = 1 # currently only works for dz = 1
dxy = 0.1
sims = []
for i in range(5):
    offset = ((i + 1) * 0.1, (i + 1) * 0.1)
    data = np.fromfunction(calc_shape_pattern, size, offset=offset, dtype=np.float32)
    data = np.expand_dims(data, 0)
    sims.append(si_utils.get_sim_from_array(
        data,
        translation={'z': i * dz, 'y': 0, 'x': 0},
        scale={'z': dz, 'y': dxy, 'x': dxy},
        ))

msims = [msi_utils.get_msim_from_sim(im) for im in sims]

# register indicating a z overlap tolerance for pairing slices
reg_method = RegistrationMethodSkFeatures(sims[0], {})
registration_method = reg_method.registration
params = registration.register(
    msims,
    transform_key=DEFAULT_TRANSFORM_KEY,
    new_transform_key='affine_registered_in_2d',
    reg_channel_index=0,
    overlap_tolerance={'x': 0.1, 'y': 0.1, 'z': 1}, # allow pairing of slices that don't initially overlap
    #pairwise_reg_func=register_3d_sims_in_2d,
    pairwise_reg_func=registration_method,
)

# calculate output stack properties for fusion
output_stack_properties = fusion.calc_fusion_stack_properties(
    sims, params, spacing={'z': dz, 'y': dxy, 'x': dxy}, mode='union')

# fuse registered sims
fused = fusion.fuse(
    [msi_utils.get_sim_from_msim(msim) for msim in msims],
    transform_key='affine_registered_in_2d',
    output_stack_properties=output_stack_properties
).compute()

# visualize fused result
plt.figure()
fig, axs = plt.subplots(1, len(sims))
for i in range(len(sims)):
    axs[i].imshow(fused.data[0, 0, i])
plt.show()
