import numpy as np

from src.util import get_value_units_micrometer


class DaskSource:
    def __init__(self, filename):
        self.filename = filename
        self.dimension_order = ''
        self.is_rgb = False
        self.shapes = []
        self.shape = []
        self.dtype = None
        self.pixel_sizes = []
        self.pixel_size = {}
        self.scales = []
        self.positions = []
        self.position = {}
        self.rotation = 0
        self.channels = []

    def get_shape(self, level=0):
        # shape in pixels
        return self.shapes[level]

    def get_size(self, level=0):
        # size in pixels
        return {dim: size for dim, size in zip(self.dimension_order, self.get_shape(level))}

    def get_pixel_size(self, level=0):
        # pixel size in micrometers
        if self.pixel_sizes:
            pixel_size = get_value_units_micrometer(self.pixel_sizes[level])
        else:
            scale = self.scales[level]
            pixel_size0 = get_value_units_micrometer(self.pixel_size)
            pixel_size = {dim: size * scale for dim, size in pixel_size0.items()}
        return pixel_size

    def get_position(self, level=0):
        # position in micrometers
        if self.positions:
            return get_value_units_micrometer(self.positions[level])
        else:
            return get_value_units_micrometer(self.position)

    def get_rotation(self):
        # rotation in degrees
        return self.rotation

    def get_nchannels(self):
        return self.get_size().get('c', 1)

    def get_channels(self):
        if len(self.channels) == 0:
            if self.is_rgb:
                return [{'label': ''}]
            else:
                return [{'label': ''}] * self.get_nchannels()
        return self.channels

    def get_data(self, level=0):
        raise NotImplementedError()

    def fix_metadata(self):
        for shape in self.shapes:
            scale1 = []
            for dim in 'xy':
                index = self.dimension_order.index(dim)
                scale1.append(self.shape[index] / shape[index])
            self.scales.append(np.mean(scale1))
