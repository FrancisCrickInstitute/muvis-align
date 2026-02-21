import zarr
import ome_zarr.format
from ome_zarr.writer import write_image

from src.muvis_align.image.util import create_compression_filter
from src.muvis_align.image.ome_zarr_util import create_axes_metadata, create_transformation_metadata, create_channel_ome_metadata


def save_ome_zarr(filename, datas, dim_order, pixel_size, channels, translations, rotations,
                  compression=None, scaler=None, ome_version='0.4'):

    is_series = isinstance(datas, list)
    if not is_series:
        datas = [datas]

    zarr_format, ome_zarr_format = get_ome_zarr_format(ome_version)

    root = zarr.create_group(store=filename, zarr_format=zarr_format, overwrite=True)
    multi_metadata = []
    omero_metadata = None
    for index, data in enumerate(datas):
        translation = translations[index] if translations is not None else None
        rotation = rotations[index] if rotations is not None else None
        if is_series:
            group_path = str(index)
            group = root.create_group(name=group_path, overwrite=True)
        else:
            group_path = ''
            group = root
        save_ome_image(data, group=group, dim_order=dim_order, pixel_size=pixel_size, channels=channels,
                       translation=translation, rotation=rotation,
                       scaler=scaler, compression=compression, ome_version=ome_version)

        if is_series:
            is_ome_root = 'ome' in group.attrs
            if is_ome_root:
                meta = group.attrs['ome']['multiscales'][0].copy()
            else:
                meta = group.attrs['multiscales'][0].copy()
            for dataset_meta in meta['datasets']:
                dataset_meta['path'] = f'{group_path}/{dataset_meta["path"]}'
            multi_metadata.append(meta)
            omero_metadata = group.attrs['omero']

    if is_series:
        if is_ome_root:
            root.attrs['multiscales'] = multi_metadata
        else:
            root.attrs['multiscales'] = multi_metadata
        root.attrs['omero'] = omero_metadata


def save_ome_image(data, group, dim_order, pixel_size, channels, translation, rotation,
               scaler=None, compression=None, ome_version='0.4'):

    zarr_format, ome_zarr_format = get_ome_zarr_format(ome_version)

    storage_options = {}
    compressor, compression_filters = create_compression_filter(compression)
    if compressor is not None:
        storage_options['compressor'] = compressor
    if compression_filters is not None:
        storage_options['filters'] = compression_filters

    axes = create_axes_metadata(dim_order)

    if scaler is not None:
        npyramid_add = scaler.max_layer
        pyramid_downsample = scaler.downscale
    else:
        npyramid_add = 0
        pyramid_downsample = 1

    coordinate_transformations = []
    factor = 1
    for i in range(npyramid_add + 1):
        transform = create_transformation_metadata(dim_order, pixel_size, factor, translation, rotation)
        coordinate_transformations.append(transform)
        if pyramid_downsample:
            factor *= pyramid_downsample

    write_image(image=data, group=group, axes=axes, coordinate_transformations=coordinate_transformations,
                scaler=scaler, storage_options=storage_options, fmt=ome_zarr_format)

    # get smallest size image for (window) metadata
    keys = list(group.array_keys())
    data_smallest = group.get(keys[-1])
    group.attrs['omero'] = create_channel_ome_metadata(data_smallest, axes, channels, ome_version)


def get_ome_zarr_format(ome_version):
    if str(ome_version) == '0.4':
        ome_zarr_format = ome_zarr.format.FormatV04()
    elif str(ome_version) == '0.5':
        ome_zarr_format = ome_zarr.format.FormatV05()   # future support anticipated
    else:
        ome_zarr_format = ome_zarr.format.CurrentFormat()
    zarr_format = 3 if float(ome_zarr_format.version) >= 0.5 else 2
    return zarr_format, ome_zarr_format
