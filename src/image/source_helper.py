import numpy as np
import os

from src.OmeZarrSource import OmeZarrSource
from src.TiffSource import TiffSource
from src.util import get_filetitle, get_orthogonal_pairs


def create_source(filename):
    ext = os.path.splitext(filename)[1].lstrip('.').lower()
    if ext.startswith('tif'):
        source = TiffSource(filename)
    elif ext.startswith('zar'):
        source = OmeZarrSource(filename)
    else:
        raise ValueError(f'Unsupported file type: {ext}')
    return source


def get_images_metadata(filenames):
    summary = 'Filename\tSize\tPosition\tRotation\n'
    sizes = []
    centers = []
    rotations = []
    positions = []
    max_positions = []
    for filename in filenames:
        source = create_source(filename)
        size = source.get_physical_size_micrometer()
        sizes.append(size)
        position = source.get_position_micrometer()
        rotation = source.get_rotation()
        rotations.append(rotation)

        summary += f'{get_filetitle(filename)}\t{tuple(size)}\t{tuple(position)}'
        if rotation is not None:
            summary += f'\t{rotation}'
        summary += '\n'

        if len(size) < len(position):
            size = list(size) + [0]
        center = np.array(position) + np.array(size) / 2
        centers.append(center)
        positions.append(position)
        max_positions.append(np.array(position) + np.array(size))
    center = np.mean(centers, 0)
    area = np.max(max_positions, 0) - np.min(positions, 0)
    summary += f'Center: {tuple(center)} Area: {tuple(area)}\n'

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
    return {'center': center,
            'area': area,
            'rotation': rotation,
            'summary': summary}
