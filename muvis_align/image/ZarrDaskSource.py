from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import os.path

from muvis_align.image.DaskSource import DaskSource
from muvis_align.util import convert_to_um


class ZarrDaskSource(DaskSource):
    def init_metadata(self):
        location = parse_url(self.filename)
        if location is None:
            raise FileNotFoundError(f'Error parsing ome-zarr file {self.filename}')
        if 'bioformats2raw.layout' in location.root_attrs:
            location = parse_url(os.path.join(self.filename, '0'))
            if location is None:
                raise FileNotFoundError(f'Error parsing ome-zarr file {self.filename}')
        reader = Reader(location)
        nodes = list(reader())
        image_node = nodes[0]
        self.data = image_node.data
        self.metadata = image_node.metadata

        self.shapes = [level.shape for level in self.data]
        self.shape = self.shapes[0]
        self.dtype = self.data[0].dtype
        axes = self.metadata['axes']
        self.dimension_order = ''.join([axis['name'] for axis in axes])
        units = {axis['name']: axis['unit'] for axis in axes if 'unit' in axis}

        pixel_sizes = []
        positions = []
        channels = []
        for transforms in self.metadata.get('coordinateTransformations', []):
            pixel_size = {}
            position = {}
            scale1 = []
            position1 = None
            for transform in transforms:
                if transform['type'] == 'scale':
                    scale1 = transform['scale']
                if transform['type'] == 'translation':
                    position1 = transform.get('translation')
            for index, dim in enumerate(self.dimension_order):
                if dim in 'xyz':
                    pixel_size[dim] = convert_to_um(scale1[index], units.get(dim, ''))
                    if position1 is not None:
                        position[dim] = (position1[index], units.get(dim, ''))
                    else:
                        position[dim] = 0
            pixel_sizes.append(pixel_size)
            positions.append(position)

        colormaps = self.metadata['colormap']
        for channeli, channel0 in enumerate(self.metadata['channel_names']):
            channel = {'label': channel0}
            if channeli < len(colormaps):
                channel['color'] = colormaps[channeli][-1]
            channels.append(channel)
        self.pixel_sizes = pixel_sizes
        self.positions = positions
        self.rotation = 0
        self.channels = channels

    def get_data(self, level=0):
        return self.data[level]
