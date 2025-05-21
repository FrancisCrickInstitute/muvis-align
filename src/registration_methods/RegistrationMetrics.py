from spatial_image import SpatialImage

from src.metrics import calc_ncc, calc_ssim
from src.registration_methods.RegistrationMethod import RegistrationMethod


class RegistrationMetrics(RegistrationMethod):
    def __init__(self, source_type, reg_function):
        super().__init__(source_type)
        self.reg_function = reg_function
        self.nccs = []
        self.ssims = []

    def registration(self, fixed_data: SpatialImage, moving_data: SpatialImage, **kwargs) -> dict:
        results = self.reg_function(fixed_data, moving_data, **kwargs)

        # TODO: move moving_data using returned transform - see phase_correlation_registration()
        fixed_data = fixed_data.squeeze()
        moving_data = moving_data.squeeze()
        self.nccs.append(calc_ncc(fixed_data, moving_data))
        self.ssims.append(calc_ssim(fixed_data, moving_data))

        return results
