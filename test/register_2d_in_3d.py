# example of embedding 2d sim into 3d with correct transforms
# for registering slices in a 3d stack

import numpy as np
from copy import deepcopy
import dask
from matplotlib import pyplot as plt
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher import registration, param_utils, msi_utils, fusion

import matplotlib
matplotlib.use('TkAgg')


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

# create simulated 2d sims
dz = 1 # currently only works for dz = 1
dxy = 0.1
sims = []
for i in range(5):
    sims.append(si_utils.get_sim_from_array(
        np.random.randint(0, 100, (1, 10, 10)),
        translation={'z': i * dz, 'y': 0, 'x': 0},
        scale={'y': dxy, 'x': dxy, 'z': dz},
        ))
sims.append(si_utils.get_sim_from_array(
    np.random.randint(0, 100, (1, 10, 10)),
    translation={'z': 0, 'y': 0, 'x': 0},
    scale={'y': dxy, 'x': dxy, 'z': dz},
    ))

pairs = [(0, 5), (0, 1), (5, 1), (1, 2), (2, 3), (3, 4)]

msims = [msi_utils.get_msim_from_sim(im) for im in sims]

# register indicating a z overlap tolerance for pairing slices
params = registration.register(
    msims,
    transform_key='affine_metadata',
    new_transform_key='affine_registered_in_2d',
    reg_channel_index=0,
    overlap_tolerance={'z': 1}, # allow pairing of slices that don't initially overlap
    pairwise_reg_func=register_3d_sims_in_2d,
    pairs=pairs,
    n_parallel_pairwise_regs=1
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
    axs[i].imshow(fused.data[0, 0, i-1])
plt.show()
