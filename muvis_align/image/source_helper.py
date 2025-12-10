import dask
import dask.array as da
import imageio
import numpy as np
import os
import tifffile
from tifffile import TiffFile
import zarr

from muvis_align.image.TiffDaskSource import TiffDaskSource
from muvis_align.image.ZarrDaskSource import ZarrDaskSource
from muvis_align.util import get_filetitle, get_orthogonal_pairs, dict_to_xyz


def create_dask_data(filename, level=0):
    ext = os.path.splitext(filename)[1]
    if 'zar' in ext:
        group = zarr.open_group(filename, mode='r')
        # using group.attrs to get multiscales is recommended by cgohlke
        paths = group.attrs['multiscales'][0]['datasets']
        path0 = paths[level]['path']
        dask_data = da.from_zarr(os.path.join(filename, path0))
    elif 'tif' in ext:
        with TiffFile(filename) as tif:
            series0 = tif.series[0]
            shape = series0.shape
            dtype = series0.dtype
        lazy_array = dask.delayed(tifffile.imread)(filename, level=level)
        dask_data = da.from_delayed(lazy_array, shape=shape, dtype=dtype)
    else:
        lazy_array = dask.delayed(imageio.v3.imread)(filename)
        # TODO get metadata from metadata = PIL.Image.info
        dask_data = da.from_delayed(lazy_array, shape=shape, dtype=dtype)
    return dask_data


def create_dask_source(filename, source_metadata=None):
    ext = os.path.splitext(filename)[1].lstrip('.').lower()
    if ext.startswith('tif'):
        dask_source = TiffDaskSource(filename, source_metadata)
    elif '.zar' in filename.lower():
        dask_source = ZarrDaskSource(filename, source_metadata)
    else:
        raise ValueError(f'Unsupported file type: {ext}')
    return dask_source


def get_images_metadata(filenames, source_metadata=None):
    summary = 'Filename\tPixel size\tSize\tPosition\tRotation\n'
    sizes = []
    centers = []
    rotations = []
    positions = []
    max_positions = []
    pixel_sizes = []
    for filename in filenames:
        source = create_dask_source(filename, source_metadata)
        pixel_size = dict_to_xyz(source.get_pixel_size())
        size = dict_to_xyz(source.get_physical_size())
        sizes.append(size)
        position = dict_to_xyz(source.get_position())
        rotation = source.get_rotation()
        rotations.append(rotation)

        summary += (f'{get_filetitle(filename)}'
                    f'\t{tuple(pixel_size)}'
                    f'\t{tuple(size)}'
                    f'\t{tuple(position)}')
        if rotation is not None:
            summary += f'\t{rotation}'
        summary += '\n'

        if len(size) < len(position):
            size = list(size) + [0]
        center = np.array(position) + np.array(size) / 2
        pixel_sizes.append(pixel_size)
        centers.append(center)
        positions.append(position)
        max_positions.append(np.array(position) + np.array(size))
    pixel_size = np.mean(pixel_sizes, 0)
    center = np.mean(centers, 0)
    area = np.max(max_positions, 0) - np.min(positions, 0)
    summary += f'Area: {tuple(area)} Center: {tuple(center)}\n'

    rotations2 = []
    for rotation, size in zip(rotations, sizes):
        if rotation is None:
            _, angles = get_orthogonal_pairs(centers, size)
            if len(angles) > 0:
                rotation = -np.mean(angles)
                rotations2.append(rotation)
    if len(rotations2) > 0:
        rotation = np.mean(rotations2)
    else:
        rotation = None
    return {'pixel_size': pixel_size,
            'center': center,
            'area': area,
            'rotation': rotation,
            'summary': summary}
