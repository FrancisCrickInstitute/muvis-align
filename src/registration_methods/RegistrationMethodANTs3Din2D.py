from copy import deepcopy
from multiview_stitcher import registration, param_utils

from src.registration_methods.RegistrationMethod import RegistrationMethod


class RegistrationMethodANTs3Din2D(RegistrationMethod):
    def __init__(self, source, params, debug):
        super().__init__(source, params, debug)
        self.count=0

    def registration(
            self,
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
        print(self.count)
        self.count+=1

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
