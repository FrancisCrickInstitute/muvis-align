import dask.array as da
import dask.array
import numpy as np
import tifffile
from tifffile import PHOTOMETRIC

from src.muvis_align.image.DaskSource import DaskSource
from src.muvis_align.image.color_conversion import int_to_rgba
from src.muvis_align.util import ensure_list


class TiffDaskSource(DaskSource):
    def init_metadata(self):
        tiff = tifffile.TiffFile(self.filename)
        pages = []
        if tiff.series and not tiff.is_mmstack:
            for level in tiff.series[0].levels:
                pages.append(level.pages[0])
        if len(pages) == 0:
            pages = tiff.pages
        page0 = pages[0]
        self.shapes = [page.shape for page in pages]
        self.shape = self.shapes[0]
        self.dtype = page0.dtype.type
        self.dimension_order = page0.axes.lower()
        photometric = page0.keyframe.photometric
        nchannels = self.get_nchannels()
        self.is_rgb = (photometric in (PHOTOMETRIC.RGB, PHOTOMETRIC.PALETTE) and nchannels in (3, 4))

        pixel_size = {}
        position = {}
        rotation = None
        channels = []
        if tiff.is_ome and tiff.ome_metadata is not None:
            xml_metadata = tiff.ome_metadata
            self.metadata = tifffile.xml2dict(xml_metadata)
            if 'OME' in self.metadata:
                self.metadata = self.metadata['OME']

                images = ensure_list(self.metadata.get('Image', {}))[0]
                pixels = images.get('Pixels', {})
                size = float(pixels.get('PhysicalSizeX', 0))
                if size:
                    pixel_size['x'] = (size, pixels.get('PhysicalSizeXUnit', self.default_physical_unit))
                size = float(pixels.get('PhysicalSizeY', 0))
                if size:
                    pixel_size['y'] = (size, pixels.get('PhysicalSizeYUnit', self.default_physical_unit))
                size = float(pixels.get('PhysicalSizeZ', 0))
                if size:
                    pixel_size['z'] =  (size, pixels.get('PhysicalSizeZUnit', self.default_physical_unit))

                for plane in ensure_list(pixels.get('Plane', [])):
                    if 'PositionX' in plane:
                        position['x'] = (float(plane.get('PositionX')), plane.get('PositionXUnit', self.default_physical_unit))
                    if 'PositionY' in plane:
                        position['y'] = (float(plane.get('PositionY')), plane.get('PositionYUnit', self.default_physical_unit))
                    if 'PositionZ' in plane:
                        position['z'] = (float(plane.get('PositionZ')), plane.get('PositionZUnit', self.default_physical_unit))
                    # c, z, t = plane.get('TheC'), plane.get('TheZ'), plane.get('TheT')

                annotations = self.metadata.get('StructuredAnnotations')
                if annotations is not None:
                    if not isinstance(annotations, (list, tuple)):
                        annotations = [annotations]
                    for annotation_item in annotations:
                        for annotations2 in annotation_item.values():
                            if not isinstance(annotations2, (list, tuple)):
                                annotations2 = [annotations2]
                            for annotation in annotations2:
                                value = annotation.get('Value')
                                unit = None
                                if isinstance(value, dict) and 'Modulo' in value:
                                    modulo = value.get('Modulo', {}).get('ModuloAlongZ', {})
                                    unit = modulo.get('Unit')
                                    value = modulo.get('Label')
                                elif isinstance(value, str) and value.lower().startswith('angle'):
                                    if ':' in value:
                                        value = value.split(':')[1].split()
                                    elif '=' in value:
                                        value = value.split('=')[1].split()
                                    else:
                                        value = value.split()[1:]
                                    if len(value) >= 2:
                                        unit = value[1]
                                    value = value[0]
                                else:
                                    value = None
                                if value is not None:
                                    rotation = float(value)
                                    if 'rad' in unit.lower():
                                        rotation = np.rad2deg(rotation)

                for channel0 in ensure_list(pixels.get('Channel', [])):
                    channel = {'label': channel0.get('Name', '')}
                    color = channel0.get('Color')
                    if color:
                        channel['color'] = int_to_rgba(int(color))
                    channels.append(channel)
        self.pixel_size = pixel_size
        self.position = position
        self.rotation = rotation
        self.channels = channels

    def get_data(self, level=0):
        if level < 0:
            dask_data = []
            for level in range(len(self.shapes)):
                lazy_array = dask.delayed(tifffile.imread)(self.filename, level=level)
                data = dask.array.from_delayed(lazy_array, shape=self.shapes[level], dtype=self.dtype)
                dask_data.append(data)
        else:
            lazy_array = dask.delayed(tifffile.imread)(self.filename, level=level)
            dask_data = dask.array.from_delayed(lazy_array, shape=self.shapes[level], dtype=self.dtype)
        return dask_data
