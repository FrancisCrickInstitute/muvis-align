import dask.array as da
import os.path
import zarr
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader

from src.image.DaskSource import DaskSource
from src.image.color_conversion import int_to_rgba, hexrgb_to_rgba


class ZarrDaskSource(DaskSource):
    def init_metadata(self):
        location = parse_url(self.filename)
        if location is None:
            raise FileNotFoundError(f'Error parsing ome-zarr file {self.filename}')
        reader = Reader(location)
        nodes = list(reader())
        image_node = nodes[0]
        self.data = image_node.data
        self.metadata = image_node.metadata
        self.omero_metdata = reader.zarr.root_attrs.get('omero', {})

        self.shapes = [level.shape for level in self.data]
        self.shape = self.shapes[0]
        self.dtype = self.data[0].dtype
        self.dimension_order = ''.join([axis['name'] for axis in self.metadata['axes']])
        units = {axis['name']: axis['unit'] for axis in self.metadata['axes'] if axis['type'] == 'space' and 'unit' in axis}

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
                    pixel_size[dim] = (scale1[index], units[dim])
                    if position1 is not None:
                        position[dim] = (position1[index], units[dim])
                    else:
                        position[dim] = 0
            pixel_sizes.append(pixel_size)
            positions.append(position)
        # look for channel metadata
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
        #return da.from_zarr(os.path.join(self.filename, self.paths[level]))
        return self.data[level]
