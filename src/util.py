import ast
import cv2 as cv
import os
import re
import numpy as np

from src.image.color_conversion import rgba_to_int


def get_default(x, default):
    return default if x is None else x


def ensure_list(x) -> list:
    if x is None:
        return []
    elif isinstance(x, list):
        return x
    else:
        return [x]


def reorder(items: list, old_order: str, new_order: str, default_value: int = 0) -> list:
    new_items = []
    for label in new_order:
        if label in old_order:
            item = items[old_order.index(label)]
        else:
            item = default_value
        new_items.append(item)
    return new_items


def filter_dict(dict0: dict) -> dict:
    new_dict = {}
    for key, value0 in dict0.items():
        if value0 is not None:
            values = []
            for value in ensure_list(value0):
                if isinstance(value, dict):
                    value = filter_dict(value)
                values.append(value)
            if len(values) == 1:
                values = values[0]
            new_dict[key] = values
    return new_dict


def desc_to_dict(desc: str) -> dict:
    desc_dict = {}
    if desc.startswith('{'):
        try:
            metadata = ast.literal_eval(desc)
            return metadata
        except:
            pass
    for item in re.split(r'[\r\n\t|]', desc):
        item_sep = '='
        if ':' in item:
            item_sep = ':'
        if item_sep in item:
            items = item.split(item_sep)
            key = items[0].strip()
            value = items[1].strip()
            for dtype in (int, float, bool):
                try:
                    value = dtype(value)
                    break
                except:
                    pass
            desc_dict[key] = value
    return desc_dict


def print_dict(dct: dict, indent: int = 0) -> str:
    s = ''
    if isinstance(dct, dict):
        for key, value in dct.items():
            s += '\n'
            if not isinstance(value, list):
                s += '\t' * indent + str(key) + ': '
            if isinstance(value, dict):
                s += print_dict(value, indent=indent + 1)
            elif isinstance(value, list):
                for v in value:
                    s += print_dict(v)
            else:
                s += str(value)
    else:
        s += str(dct)
    return s


def print_hbytes(nbytes: int) -> str:
    exps = ['', 'K', 'M', 'G', 'T']
    div = 1024
    exp = 0

    while nbytes > div:
        nbytes /= div
        exp += 1
    return f'{nbytes:.1f}{exps[exp]}B'


def check_round_significants(a: float, significant_digits: int) -> float:
    rounded = round_significants(a, significant_digits)
    if a != 0:
        dif = 1 - rounded / a
    else:
        dif = rounded - a
    if abs(dif) < 10 ** -significant_digits:
        return rounded
    return a


def round_significants(a: float, significant_digits: int) -> float:
    if a != 0:
        round_decimals = significant_digits - int(np.floor(np.log10(abs(a)))) - 1
        return round(a, round_decimals)
    return a


def get_filetitle(filename: str) -> str:
    filebase = os.path.basename(filename)
    title = os.path.splitext(filebase)[0].rstrip('.ome')
    return title


def find_all_numbers(text: str) -> list:
    return list(map(int, re.findall(r'\d+', text)))


def split_num_text(text: str) -> list:
    num_texts = []
    block = ''
    is_num0 = None
    if text is None:
        return None

    for c in text:
        is_num = (c.isnumeric() or c == '.')
        if is_num0 is not None and is_num != is_num0:
            num_texts.append(block)
            block = ''
        block += c
        is_num0 = is_num
    if block != '':
        num_texts.append(block)

    num_texts2 = []
    for block in num_texts:
        block = block.strip()
        try:
            block = float(block)
        except:
            pass
        if block not in [' ', ',', '|']:
            num_texts2.append(block)
    return num_texts2


def split_value_unit_list(text: str) -> list:
    value_units = []
    if text is None:
        return None

    items = split_num_text(text)
    if isinstance(items[-1], str):
        def_unit = items[-1]
    else:
        def_unit = ''

    i = 0
    while i < len(items):
        value = items[i]
        if i + 1 < len(items):
            unit = items[i + 1]
        else:
            unit = ''
        if not isinstance(value, str):
            if isinstance(unit, str):
                i += 1
            else:
                unit = def_unit
            value_units.append((value, unit))
        i += 1
    return value_units


def get_value_units_micrometer(value_units0: list) -> list:
    conversions = {'nm': 1e-3, 'µm': 1, 'um': 1, 'micrometer': 1, 'mm': 1e3, 'cm': 1e4, 'm': 1e6}
    if value_units0 is None:
        return None

    values_um = []
    for value_unit in value_units0:
        if not (isinstance(value_unit, int) or isinstance(value_unit, float)):
            value_um = value_unit[0] * conversions.get(value_unit[1], 1)
        else:
            value_um = value_unit
        values_um.append(value_um)
    return values_um


def convert_rational_value(value) -> float:
    if value is not None and isinstance(value, tuple):
        value = value[0] / value[1]
    return value


def create_transform(center=(0, 0), angle=0, scale=1, translate=(0, 0)):
    transform = cv.getRotationMatrix2D(center, angle, scale)
    transform[:, 2] += translate
    if len(transform) == 2:
        transform = np.vstack([transform, [0, 0, 1]])   # create 3x3 matrix
    return transform


def apply_transform(points, transform):
    new_points = []
    point_len = 0
    for point in points:
        point_len = len(point)
        if point_len == 2:
            point = list(point) + [1]
        new_point = np.dot(point, np.transpose(transform))
        new_points.append(new_point)
    if point_len == 2:
        new_points = np.array(new_points)[:, :2].tolist()
    return new_points


def create_tiff_metadata(metadata, is_ome=False):
    ome_metadata = None
    resolution = None
    resolution_unit = None
    pixel_size_um = None

    pixel_size = metadata.get('pixel_size')
    position = metadata.get('position')
    if pixel_size is not None:
        pixel_size_um = get_value_units_micrometer(pixel_size)
        resolution_unit = 'CENTIMETER'
        resolution = [1e4 / size for size in pixel_size_um]
    channels = metadata.get('channels', [])

    if is_ome:
        ome_metadata = {}
        ome_channels = []
        if pixel_size_um is not None:
            ome_metadata['PhysicalSizeX'] = pixel_size_um[0]
            ome_metadata['PhysicalSizeXUnit'] = 'µm'
            ome_metadata['PhysicalSizeY'] = pixel_size_um[1]
            ome_metadata['PhysicalSizeYUnit'] = 'µm'
            if len(pixel_size_um) > 2:
                ome_metadata['PhysicalSizeZ'] = pixel_size_um[2]
                ome_metadata['PhysicalSizeZUnit'] = 'µm'
        if position is not None:
            plane_metadata = {}
            plane_metadata['PositionX'] = position[0]
            plane_metadata['PositionXUnit'] = 'µm'
            plane_metadata['PositionY'] = position[1]
            plane_metadata['PositionYUnit'] = 'µm'
            if len(position) > 2:
                ome_metadata['PositionZ'] = position[2]
                ome_metadata['PositionZUnit'] = 'µm'
            ome_metadata['Plane'] = plane_metadata
        for channel in channels:
            ome_channel = {'Name': channel.get('label', '')}
            if 'color' in channel:
                ome_channel['Color'] = rgba_to_int(channel['color'])
            ome_channels.append(ome_channel)
        if ome_channels:
            ome_metadata['Channel'] = ome_channels
    return ome_metadata, resolution, resolution_unit
