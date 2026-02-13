from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import zarr
from ome_zarr.writer import write_image

# Example URL of remote data
#url = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr"
url = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0062A/6001240_labels.zarr"
url='D:/slides/6001240.zarr'


# read the image data
reader = Reader(parse_url(url))
# nodes may include images, labels etc
nodes = list(reader())
# first node will be the image pixel data
image_node = nodes[0]

# list of dask arrays at different pyramid size levels
data = image_node.data
# dictionary of OME-Zarr metadata
metadata = image_node.metadata
axes = ''.join([axis['name'] for axis in metadata['axes']])

full_size_image_data = data[0]  # access the image data array at full size


path = "path/to/output_image.zarr"

root = zarr.open_group(store=path)
# supports dask data, by default written out at various pyramid sizes
write_image(image=full_size_image_data, group=root, axes=axes)
