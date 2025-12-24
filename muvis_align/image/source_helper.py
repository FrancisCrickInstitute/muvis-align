import numpy as np
import os

from muvis_align.image.TiffDaskSource import TiffDaskSource
from muvis_align.image.ZarrDaskSource import ZarrDaskSource
from muvis_align.util import get_filetitle, get_orthogonal_pairs


def create_dask_source(filename, source_metadata=None, index=None):
    ext = os.path.splitext(filename)[1].lstrip('.').lower()
    if ext.startswith('tif'):
        dask_source = TiffDaskSource(filename, source_metadata, index=index)
    elif '.zar' in filename.lower():
        dask_source = ZarrDaskSource(filename, source_metadata, index=index)
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
        pixel_size = source.get_pixel_size()
        size = source.get_physical_size()
        sizes.append(size)
        position = source.get_position()
        rotation = source.get_rotation()
        rotations.append(rotation)

        summary += (f'{get_filetitle(filename)}'
                    f'\t{tuple(pixel_size)}'
                    f'\t{tuple(size)}'
                    f'\t{tuple(position)}')
        if rotation is not None:
            summary += f'\t{rotation}'
        summary += '\n'

        center = {dim: position[dim] + size.get(dim, 0)/2 for dim in position}
        pixel_sizes.append(pixel_size)
        centers.append(center)
        positions.append(position)
        max_positions.append({dim: position[dim] + size.get(dim, 0) for dim in position})
    pixel_size = {dim: float(np.mean([pixel_size[dim] for pixel_size in pixel_sizes])) for dim in pixel_sizes[0]}
    center = {dim: float(np.mean([center[dim] for center in centers])) for dim in centers[0]}
    min_position = {dim: min([position[dim] for position in positions]) for dim in positions[0]}
    max_position = {dim: max([position[dim] for position in positions]) for dim in positions[0]}
    area = {dim: max_position[dim] - min_position[dim] for dim in max_position}
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
