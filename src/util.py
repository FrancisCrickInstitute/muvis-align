import ast
import csv
import json
import cv2 as cv
import glob
import math
import numpy as np
import os
import re
from scipy.spatial.transform import Rotation
from sklearn.neighbors import KDTree


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
    exps = ['', 'K', 'M', 'G', 'T', 'P', 'E']
    div = 1024
    exp = 0

    while nbytes > div:
        nbytes /= div
        exp += 1
    if exp < len(exps):
        e = exps[exp]
    else:
        e = f'e{exp * 3}'
    return f'{nbytes:.1f}{e}B'


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


def split_path(path: str) -> list:
    return os.path.normpath(path).split(os.path.sep)


def get_filetitle(filename: str) -> str:
    filebase = os.path.basename(filename)
    title = os.path.splitext(filebase)[0].rstrip('.ome')
    return title


def dir_regex(pattern):
    files = []
    for pattern_item in ensure_list(pattern):
        files.extend(glob.glob(pattern_item, recursive=True))
    files_sorted = sorted(files, key=lambda file: find_all_numbers(get_filetitle(file)))
    return files_sorted


def find_all_numbers(text: str) -> list:
    return list(map(int, re.findall(r'\d+', text)))


def split_numeric(text: str) -> list:
    num_parts = []
    parts = re.split(r'[_/\\.]', text)
    for part in parts:
        num_span = re.search(r'\d+', part)
        if num_span:
            num_parts.append(part)
    return num_parts


def split_numeric_dict(text: str) -> dict:
    num_parts = {}
    parts = re.split(r'[_/\\.]', text)
    parti = 0
    for part in parts:
        num_span = re.search(r'\d+', part)
        if num_span:
            index = num_span.start()
            label = part[:index]
            if label == '':
                label = parti
            num_parts[label] = num_span.group()
            parti += 1
    return num_parts


def get_unique_file_labels(filenames: list) -> list:
    file_labels = []
    file_parts = []
    label_indices = set()
    last_parts = None
    for filename in filenames:
        parts = split_numeric(filename)
        if len(parts) == 0:
            parts = split_numeric(filename)
            if len(parts) == 0:
                parts = filename
        file_parts.append(parts)
        if last_parts is not None:
            for parti, (part1, part2) in enumerate(zip(last_parts, parts)):
                if part1 != part2:
                    label_indices.add(parti)
        last_parts = parts
    label_indices = sorted(list(label_indices))

    for file_part in file_parts:
        file_label = '_'.join([file_part[i] for i in label_indices])
        file_labels.append(file_label)

    if len(set(file_labels)) < len(file_labels):
        # fallback for duplicate labels
        file_labels = [get_filetitle(filename) for filename in filenames]

    return file_labels


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


def eval_context(data, key, default_value, context):
    value = data.get(key, default_value)
    if isinstance(value, str):
        try:
            value = value.format_map(context)
        except:
            pass
        try:
            value = eval(value, context)
        except:
            pass
    return value


def get_value_units_micrometer(value_units0: list|dict) -> list|dict|None:
    conversions = {
        'nm': 1e-3,
        'µm': 1, 'um': 1, 'micrometer': 1,
        'mm': 1e3, 'millimeter': 1e3,
        'cm': 1e4, 'centimeter': 1e4,
        'm': 1e6, 'meter': 1e6
    }
    if value_units0 is None:
        return None

    if isinstance(value_units0, dict):
        values_um = {}
        for dim, value_unit in value_units0.items():
            if isinstance(value_unit, (list, tuple)):
                value_um = value_unit[0] * conversions.get(value_unit[1], 1)
            else:
                value_um = value_unit
            values_um[dim] = value_um
    else:
        values_um = []
        for value_unit in value_units0:
            if isinstance(value_unit, (list, tuple)):
                value_um = value_unit[0] * conversions.get(value_unit[1], 1)
            else:
                value_um = value_unit
            values_um.append(value_um)
    return values_um


def convert_rational_value(value) -> float:
    if value is not None and isinstance(value, tuple):
        if value[0] == value[1]:
            value = value[0]
        else:
            value = value[0] / value[1]
    return value


def get_moments(data, offset=(0, 0)):
    moments = cv.moments((np.array(data) + offset).astype(np.float32))    # doesn't work for float64!
    return moments


def get_moments_center(moments, offset=(0, 0)):
    return np.array([moments['m10'], moments['m01']]) / moments['m00'] + np.array(offset)


def get_center(data, offset=(0, 0)):
    moments = get_moments(data, offset=offset)
    if moments['m00'] != 0:
        center = get_moments_center(moments)
    else:
        center = np.mean(data, 0).flatten()  # close approximation
    return center.astype(np.float32)


def create_transform0(center=(0, 0), angle=0, scale=1, translate=(0, 0)):
    transform = cv.getRotationMatrix2D(center[:2], angle, scale)
    transform[:, 2] += translate
    if len(transform) == 2:
        transform = np.vstack([transform, [0, 0, 1]])   # create 3x3 matrix
    return transform


def create_transform(center, angle, matrix_size=3):
    if isinstance(center, dict):
        center = dict_to_xyz(center)
    if len(center) == 2:
        center = np.array(list(center) + [0])
    if angle is None:
        angle = 0
    r = Rotation.from_euler('z', angle, degrees=True)
    t = center - r.apply(center, inverse=True)
    transform = np.eye(matrix_size)
    transform[:3, :3] = np.transpose(r.as_matrix())
    transform[:3, -1] += t
    return transform


def apply_transform(points, transform):
    new_points = []
    for point in points:
        point_len = len(point)
        while len(point) < len(transform):
            point = list(point) + [1]
        new_point = np.dot(point, np.transpose(transform))
        new_points.append(new_point[:point_len])
    return new_points


def validate_transform(transform, max_rotation=None):
    if transform is None:
        return False
    transform = np.array(transform)
    if np.any(np.isnan(transform)):
        return False
    if np.any(np.isinf(transform)):
        return False
    if np.linalg.det(transform) == 0:
        return False
    if  max_rotation is not None and abs(normalise_rotation(get_rotation_from_transform(transform))) > max_rotation:
        return False
    return True


def get_scale_from_transform(transform):
    scale = np.mean(np.linalg.norm(transform, axis=0)[:-1])
    return scale


def get_translation_from_transform(transform):
    ndim = len(transform) - 1
    #translation = transform[:ndim, ndim]
    translation = apply_transform([[0] * ndim], transform)[0]
    return translation


def get_center_from_transform(transform):
    # from opencv:
    # t0 = (1-alpha) * cx - beta * cy
    # t1 = beta * cx + (1-alpha) * cy
    # where
    # alpha = cos(angle) * scale
    # beta = sin(angle) * scale
    # isolate cx and cy:
    t0, t1 = transform[:2, 2]
    scale = 1
    angle = np.arctan2(transform[0][1], transform[0][0])
    alpha = np.cos(angle) * scale
    beta = np.sin(angle) * scale
    cx = (t1 + t0 * (1 - alpha) / beta) / (beta + (1 - alpha) ** 2 / beta)
    cy = ((1 - alpha) * cx - t0) / beta
    return cx, cy


def get_rotation_from_transform(transform):
    rotation = np.rad2deg(np.arctan2(transform[0][1], transform[0][0]))
    return rotation


def normalise_rotation(rotation):
    """
    Normalise rotation to be in the range [-180, 180].
    """
    while rotation < -180:
        rotation += 360
    while rotation > 180:
        rotation -= 360
    return rotation


def points_to_3d(points):
    return [list(point) + [0] for point in points]


def xyz_to_dict(xyz, axes='xyz'):
    dct = {dim: value for dim, value in zip(axes, xyz)}
    return dct


def dict_to_xyz(dct, keys='xyz'):
    return [dct[key] for key in keys if key in dct]


def normalise_rotated_positions(positions0, rotations0, size, center):
    # in [xy(z)]
    positions = []
    rotations = []
    _, angles = get_orthogonal_pairs(positions0, size)
    for position0, rotation in zip(positions0, rotations0):
        if rotation is None and len(angles) > 0:
            rotation = -np.mean(angles)
        angle = -rotation if rotation is not None else None
        transform = create_transform(center=center, angle=angle, matrix_size=4)
        position = apply_transform([position0], transform)[0]
        positions.append(position)
        rotations.append(rotation)
    return positions, rotations


def get_nn_distance(points0):
    points = list(set(map(tuple, points0)))     # get unique points
    if len(points) >= 2:
        tree = KDTree(points, leaf_size=2)
        dist, ind = tree.query(points, k=2)
        nn_distance = np.median(dist[:, 1])
    else:
        nn_distance = 1
    return nn_distance


def get_mean_nn_distance(points1, points2):
    return np.mean([get_nn_distance(points1), get_nn_distance(points2)])


def filter_edge_points(points, bounds, filter_factor=0.1, threshold=0.5):
    center = np.array(bounds) / 2
    dist_center = np.abs(points / center - 1)
    position_weights = np.clip((1 - np.max(dist_center, axis=-1)) / filter_factor, 0, 1)
    order_weights = 1 - np.array(range(len(points))) / len(points) / 2
    weights = position_weights * order_weights
    return weights > threshold


def draw_edge_filter(bounds):
    out_image = np.zeros(np.flip(bounds))
    y, x = np.where(out_image == 0)
    points = np.transpose([x, y])

    center = np.array(bounds) / 2
    dist_center = np.abs(points / center - 1)
    position_weights = np.clip((1 - np.max(dist_center, axis=-1)) * 10, 0, 1)
    return position_weights.reshape(np.flip(bounds))


def get_orthogonal_pairs(origins, image_size_um):
    """
    Get pairs of orthogonal neighbors from a list of tiles.
    Tiles don't have to be placed on a regular grid.
    """
    pairs = []
    angles = []
    z_positions = [pos[0] for pos in origins]
    is_mixed_3dstack = len(set(z_positions)) < len(z_positions)
    for i, j in np.transpose(np.triu_indices(len(origins), 1)):
        origini = np.array(origins[i])
        originj = np.array(origins[j])
        if is_mixed_3dstack:
            # ignore z value for distance
            distance = math.dist(origini[-2:], originj[-2:])
            min_distance = max(image_size_um[-2:])
            is_same_z = (origini[0] == originj[0])
            if not is_same_z:
                # for tiles in different z stack, require greater overlap
                min_distance *= 0.8
        else:
            distance = math.dist(origini, originj)
            min_distance = max(image_size_um)
        if distance < min_distance:
            pairs.append((i, j))
            vector = origini - originj
            angle = math.degrees(math.atan2(vector[1], vector[0]))
            if distance < min(image_size_um):
                angle += 90
            while angle < -90:
                angle += 180
            while angle > 90:
                angle -= 180
            angles.append(angle)
    return pairs, angles


def retuple(chunks, shape):
    # from ome-zarr-py
    """
    Expand chunks to match shape.

    E.g. if chunks is (64, 64) and shape is (3, 4, 5, 1028, 1028)
    return (3, 4, 5, 64, 64)

    If chunks is an integer, it is applied to all dimensions, to match
    the behaviour of zarr-python.
    """

    if isinstance(chunks, int):
        return tuple([chunks] * len(shape))

    dims_to_add = len(shape) - len(chunks)
    return *shape[:dims_to_add], *chunks


def import_metadata(content, fields=None, input_path=None):
    # return dict[id] = {values}
    if isinstance(content, str):
        ext = os.path.splitext(content)[1].lower()
        if input_path:
            if isinstance(input_path, list):
                input_path = input_path[0]
            content = os.path.join(os.path.dirname(input_path), content)
        if ext == '.csv':
            content = import_csv(content)
        elif ext in ['.json', '.ome.json']:
            content = import_json(content)
    if fields is not None:
        content = [[data[field] for field in fields] for data in content]
    return content


def import_json(filename):
    with open(filename, encoding='utf8') as file:
        data = json.load(file)
    return data


def export_json(filename, data):
    with open(filename, 'w', encoding='utf8') as file:
        json.dump(data, file, indent=4)


def import_csv(filename):
    with open(filename, encoding='utf8') as file:
        data = csv.reader(file)
    return data


def export_csv(filename, data, header=None):
    with open(filename, 'w', encoding='utf8', newline='') as file:
        csvwriter = csv.writer(file)
        if header is not None:
            csvwriter.writerow(header)
        for row in data:
            csvwriter.writerow(row)
