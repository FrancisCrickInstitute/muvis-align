# https://blog.dask.org/2019/06/20/load-image-data
# https://docs.dask.org/en/stable/best-practices.html#load-data-with-dask
# https://dask.discourse.group/t/could-not-serialize-object-of-type-highlevelgraph-with-client-ome-tiff/3959

import dask.array as da
from dask.distributed import Client
from distributed import performance_report
import glob
from multiview_stitcher import spatial_image_utils as si_utils
import numpy as np
from skimage.transform import resize
from tifffile import TiffFile, TiffWriter

from src.Timer import Timer
from src.image.util import normalise
from src.util import convert_xyz_to_dict


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


def dask_ome_tiff(filename):
    store = TiffFile(filename).aszarr(level=0)
    dask_data = da.from_zarr(store)
    return dask_data


def dask_ome_zarr(filename):
    return da.from_zarr(filename + '/0')


if __name__ == "__main__":
    #data = np.ones(shape=(512, 1024)).astype(np.float32)
    #filename = 'test.ome.tiff'
    #save_ome_tiff(filename, data, npyramid_add=4)

    chunk_size = [1024, 1024]

    with Client(processes=False) as client:
        print(client)

        with performance_report(filename="report.html"):

            #dask_data = dask_ome_tiff('D:/slides/12193/stitched/S???/registered.ome.tiff')

            sims = []
            transform_key = 'transform_key'
            for filename in glob.glob('D:/slides/12193/stitched/S???/registered.ome.zarr'):
                print(f'reading {filename}')
                dask_data = dask_ome_zarr(filename)
                sim = si_utils.get_sim_from_array(dask_data, transform_key=transform_key)
                sim = sim.chunk(convert_xyz_to_dict(chunk_size))
                sims.append(sim)

            print('normalising')
            #value = np.mean(dask_data).compute()
            new_sims = normalise(sims, transform_key=transform_key)
            print('computing')
            for new_sim in new_sims:
                new_sim.compute()
            print('done')
