import dask.array as da
import numpy as np
import zarr
from dask.distributed import Client
from skimage.transform import resize
from tifffile import TiffFile, TiffWriter


def save_ome_tiff(filename, data, npyramid_add=0):
    size = np.array(data.shape[:2])
    with TiffWriter(filename, ome=True) as writer:
        for i in range(npyramid_add + 1):
            if i == 0:
                subifds = npyramid_add
                subfiletype = None
            else:
                subifds = None
                subfiletype = 1
                size //= 2
                data = resize(data, size)
            writer.write(data, subifds=subifds, subfiletype=subfiletype)


if __name__ == "__main__":
    data = np.ones(shape=(512, 1024)).astype(np.float32)
    filename = 'test.ome.tiff'
    save_ome_tiff(filename, data, npyramid_add=4)

    store = TiffFile(filename).aszarr(multiscales=True)
    group = zarr.open_group(store=store, mode='r')
    # using group.attrs to get multiscales is recommended by cgohlke
    paths = group.attrs['multiscales'][0]['datasets']
    data = [group[path_dict['path']] for path_dict in group.attrs['multiscales'][0]['datasets']]
    dask_data = da.from_zarr(data[0])

    with Client(processes=False) as client:
        value = np.mean(dask_data)
        result = value.compute()
        print(result)
