import numpy as np

from muvis_align.util import get_value_units_micrometer, find_all_numbers, split_numeric_dict, eval_context


class DaskSource:
    default_physical_unit = 'µm'

    def __init__(self, filename, source_metadata=None, index=None):
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
        self.init_metadata()
        self.fix_metadata(source_metadata, index=index)

    def init_metadata(self):
        raise NotImplementedError("Dask source should implement init_metadata() to initialize metadata")

    def fix_metadata(self, source_metadata=None, index=None):
        if isinstance(source_metadata, dict):
            filename_numeric = find_all_numbers(self.filename)
            filename_dict = {key: int(value) for key, value in split_numeric_dict(self.filename).items()}
            context = {'filename_numeric': filename_numeric, 'fn': filename_numeric} | filename_dict
            if 'position' in source_metadata:
                translation0 = source_metadata['position']
                if index is not None and isinstance(translation0, list):
                    translation0 = translation0[index]
                if 'x' in translation0:
                    self.position['x'] = eval_context(translation0, 'x', 0, context)
                if 'y' in translation0:
                    self.position['y'] = eval_context(translation0, 'y', 0, context)
                if 'z' in translation0:
                    self.position['z'] = eval_context(translation0, 'z', 0, context)
            if 'scale' in source_metadata:
                scale0 = source_metadata['scale']
                if index is not None and isinstance(scale0, list):
                    scale0 = scale0[index]
                if 'x' in scale0:
                    self.pixel_size['x'] = eval_context(scale0, 'x', 1, context)
                if 'y' in scale0:
                    self.pixel_size['y'] = eval_context(scale0, 'y', 1, context)
                if 'z' in scale0:
                    self.pixel_size['z'] = eval_context(scale0, 'z', 1, context)
            if 'rotation' in source_metadata:
                self.rotation = source_metadata['rotation']

        if len(self.scales) == 0:
            for shape in self.shapes:
                scale1 = []
                for dim in 'xy':
                    index = self.dimension_order.index(dim)
                    scale1.append(self.shape[index] / shape[index])
                self.scales.append(float(np.mean(scale1)))

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

    def get_physical_size(self):
        pixel_size = self.get_pixel_size()
        size = self.get_size()
        return {dim: size[dim] * pixel_size[dim] for dim in size}

    def get_position(self, level=0):
        # position in micrometers
        if self.positions and level > 0:
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
