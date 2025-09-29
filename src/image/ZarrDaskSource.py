import dask.array as da
import os.path
import zarr

from src.image.DaskSource import DaskSource
from src.image.color_conversion import int_to_rgba, hexrgb_to_rgba


class ZarrDaskSource(DaskSource):
    def init_metadata(self):
        group = zarr.open_group(self.filename, mode='r')
        if 'ome' in group.attrs:
            self.metadata = group.attrs['ome']['multiscales'][0]
        else:
            self.metadata = group.attrs['multiscales'][0]
        self.omero_metdata = group.attrs.get('omero', {})
        self.paths = [dataset['path'] for dataset in self.metadata['datasets']]

        self.shapes = [group[path].shape for path in self.paths]
        self.shape = self.shapes[0]
        self.dtype = group['0'].dtype

        self.dimension_order = ''.join([axis['name'] for axis in self.metadata['axes']])
        units = {axis['name']: axis['unit'] for axis in self.metadata['axes'] if axis['type'] == 'space' and 'unit' in axis}

        pixel_sizes = []
        positions = []
        channels = []
        for dataset in self.metadata.get('datasets', []):
            transforms = dataset.get('coordinateTransformations', [])
            pixel_size = {}
            position = {}
            scale1 = []
            position1 = []
            for transform in transforms:
                if transform['type'] == 'scale':
                    scale1 = transform['scale']
                if transform['type'] == 'translation':
                    position1 = transform['translation']
            for index, dim in enumerate(self.dimension_order):
                if dim in 'xyz':
                    pixel_size[dim] = (scale1[index], units[dim])
                    position[dim] = (position1[index], units[dim])
            pixel_sizes.append(pixel_size)
            positions.append(position)
        # look for channel metadata
        for channel0 in self.omero_metdata.get('channels', []):
            channel = channel0.copy()
            color = channel.pop('color', '')
            if color != '':
                if isinstance(color, int):
                    color = int_to_rgba(color)
                else:
                    color = hexrgb_to_rgba(color)
                channel['color'] = color
            channels.append(channel)
        self.pixel_sizes = pixel_sizes
        self.positions = positions
        self.rotation = 0
        self.channels = channels

    def get_data(self, level=0):
        return da.from_zarr(os.path.join(self.filename, self.paths[level]))
