from ome_zarr.scale import Scaler

from muvis_align.constants import zarr_extension, tiff_extension
from muvis_align.image.ome_ngff_helper import save_ome_ngff
from muvis_align.image.ome_tiff_helper import save_ome_tiff
from muvis_align.image.ome_zarr_helper import save_ome_zarr
from muvis_align.image.util import *


def save_image(filename, sim, output_format=zarr_extension, params={},
               transform_key=None, channels=None, translation0=None):
    dimension_order = ''.join(sim.dims)
    sdims = ''.join(si_utils.get_spatial_dims_from_sim(sim))
    sdims = sdims.replace('zyx', 'xyz').replace('yx', 'xy')   # order xy(z)
    pixel_size = []
    for dim in sdims:
        pixel_size1 = si_utils.get_spacing_from_sim(sim)[dim]
        if pixel_size1 == 0:
            pixel_size1 = 1
        pixel_size.append(pixel_size1)
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

    if 'zar' in output_format:
        #save_ome_zarr(str(filename) + zarr_extension, sim.data, dimension_order, pixel_size,
        #              channels, position, rotation, compression=compression, scaler=scaler,
        #              zarr_version=3, ome_version='0.5')
        save_ome_ngff(str(filename) + zarr_extension, sim, channels, position, rotation,
                      pyramid_downsample=pyramid_downsample)
    if 'tif' in output_format:
        save_ome_tiff(str(filename) + tiff_extension, sim.data, dimension_order, pixel_size,
                      channels, positions, rotation, tile_size=tile_size, compression=compression, scaler=scaler)


def exists_output_image(path, output_format):
    exists = True
    if 'zar' in output_format:
        exists = exists and os.path.exists(path + zarr_extension)
    if 'tif' in output_format:
        exists = exists and os.path.exists(path + tiff_extension)
    return exists
