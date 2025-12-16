
from utilities import (
    load_nrrd_mask,
    ensure_continuous_body,
    preprocess_binary_mask,
    sort_labelled_bodies_by_size,
    resample_to_isotropic,
)
from visualizations import create_projection_view, visualize_3d_graph, visualize_binary_mask
from pipelines import process_single_artery
import numpy as np

# I'm loading the data here for a single file, it gets loaded into 3d numpy array, and
# a seperate header dictionary with information about the data
path = "data/batch_2/Normal_15.nrrd"
binary_mask, header = load_nrrd_mask(path, verbose=True)

# Here I'm extracting the spacing/direction information from the header data
# So that we can translate eucledian distances from pure voxel coordinate deltas
# to actual units
spacing_info = tuple(np.diag(header['space directions']))

binary_mask, spacing_info = resample_to_isotropic(binary_mask, spacing_info)
# So far the preprocessings can be an optional upsample and a gaussian filter for smoothing.
binary_mask = preprocess_binary_mask(binary_mask, upsample_factor=1)

# This function will check the mask to see if it is one continous body, a boolean flag
# indicates its continous while the labelled_bodies array is an array of the same shape
# as binary_mask, but each unique continous body has a unique label, ie 0 = background, 1 = body 1, 2 = body 2 etc..
is_continous, labelled_bodies = ensure_continuous_body(binary_mask, debug=True)

#CONNECTED COMPONENTS 3D
if is_continous:
    print("is continous")
    original_one_sided_mask = (labelled_bodies == 1).astype(np.uint8)
else:
    print("it ain't continous")
    sorted_bodies = sort_labelled_bodies_by_size(labelled_bodies)
    original_one_sided_mask = sorted_bodies[1]
    # original_one_sided_mask = (labelled_bodies == 1).astype(np.uint8)

create_projection_view(binary_mask)

# Run the complete artery analysis pipeline
result = process_single_artery(
    binary_mask=original_one_sided_mask,
    spacing_info=spacing_info,
    config_path='process_config.yaml'
)

# Extract the final graph from the result
final_graph = result['final_graph']
sparse_graph = result['sparse_graph']

# Visualize the results
visualize_3d_graph(sparse_graph, original_one_sided_mask, dark_mode=True, hide_background=True)

# Visualize just the binary mask
# visualize_binary_mask(original_one_sided_mask, dark_mode=True, hide_background=True)
