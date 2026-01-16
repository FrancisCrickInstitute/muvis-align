import os.path
from ome_zarr.scale import Scaler

from muvis_align.constants import zarr_extension, tiff_extension
from muvis_align.image.ome_ngff_helper import save_ome_ngff
from muvis_align.image.ome_tiff_helper import save_ome_tiff
from muvis_align.image.ome_zarr_helper import save_ome_zarr
from muvis_align.image.util import *


def save_image(filename, sim, output_format=zarr_extension, params={},
               transform_key=None, channels=None, translation0=None, verbose=False):
    dimension_order = ''.join(sim.dims)
    pixel_size = si_utils.get_spacing_from_sim(sim)
    # metadata: only use coords of fused image
    position, rotation = get_data_mapping(sim, transform_key=transform_key, translation0=translation0)
    nplanes = sim.sizes.get('z', 1) * sim.sizes.get('c', 1) * sim.sizes.get('t', 1)
    positions = [position] * nplanes

    if channels is None:
        channels = sim.attrs.get('channels', [])

    tile_size = params.get('tile_size')
    if tile_size:
        if 'z' in sim.dims and len(tile_size) < 3:
            tile_size = list(tile_size) + [1]
        tile_size = tuple(reversed(tile_size))
        chunking = retuple(tile_size, sim.shape)
        sim = sim.chunk(chunks=chunking)

    compression = params.get('compression')
    pyramid_downsample = params.get('pyramid_downsample', 2)
    npyramid_add = get_max_downsamples(sim.shape, params.get('npyramid_add', 0), pyramid_downsample)
    scaler = Scaler(downscale=pyramid_downsample, max_layer=npyramid_add)
    ome_version = params.get('ome_version', '0.4')

    if 'zar' in output_format:
        #save_ome_zarr(str(filename) + zarr_extension, sim.data, dimension_order, pixel_size,
        #              channels, position, rotation, compression=compression, scaler=scaler,
        #              zarr_version=3, ome_version='0.5')
        save_ome_ngff(str(filename) + zarr_extension, sim, pyramid_downsample=pyramid_downsample,
                      ome_version=ome_version, verbose=verbose)
    if 'tif' in output_format:
        save_ome_tiff(str(filename) + tiff_extension, sim.data, dimension_order, pixel_size,
                      channels, positions, rotation, tile_size=tile_size, compression=compression, scaler=scaler)

    open(str(filename), 'w')


def exists_output_image(filename):
    return os.path.exists(filename)
