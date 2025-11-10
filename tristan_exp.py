from utilities import (
    load_nrrd_mask,
    ensure_continous_body,
    extract_centerline_skimage,
    extract_endpoint_and_bifurcation_coordinates,
    skeleton_to_sparse_graph,
    skeleton_to_dense_graph,
    dense_graph_to_skeleton,
    preprocess_binary_mask,
    remove_redundant_bifurcation_clusters,
    remove_sharp_bend_bifurcations,
    sort_labelled_bodies_by_size,
    create_distance_transform_from_mask,
    compute_branch_diameters_of_graph,
    compute_branch_lengths_of_graph,
    merge_branch_metrics,

)
from visualizations import create_projection_view, visualize_3d_graph
import numpy as np

# I'm loading the data here for a single file, it gets loaded into 3d numpy array, and
# a seperate header dictionary with information about the data
path = "data/batch_1/Normal_2.nrrd"
binary_mask, header = load_nrrd_mask(path, verbose=True)

# Here I'm extracting the spacing/direction information from the header data
# So that we can translate eucledian distances from pure voxel coordinate deltas
# to actual units
spacing_info = tuple(np.diag(header['space directions']))


# So far the preprocessings can be an optional upsample and a gaussian filter for smoothing.
binary_mask = preprocess_binary_mask(binary_mask, upsample_factor=1)

# This function will check the mask to see if it is one continous body, a boolean flag
# indicates its continous while the labelled_bodies array is an array of the same shape
# as binary_mask, but each unique continous body has a unique label, ie 0 = background, 1 = body 1, 2 = body 2 etc..
is_continous, labelled_bodies = ensure_continous_body(binary_mask)

if is_continous:
    print("is continous")
    original_one_sided_mask = (labelled_bodies == 1).astype(np.uint8)
else:
    print("it ain't continous")
    sorted_bodies = sort_labelled_bodies_by_size(labelled_bodies)
    original_one_sided_mask = sorted_bodies[0]
    # original_one_sided_mask = (labelled_bodies == 1).astype(np.uint8)

# create_projection_view(binary_mask)

# This calculates each voxels euclidean distance away from the background (0s) which I will use later on
# to approximate vessel diameters, although I think we need a more precise final approach.

distance_array = create_distance_transform_from_mask(binary_mask, spacing_info)

skeleton_binary_mask = extract_centerline_skimage(original_one_sided_mask)

skeleton_binary_mask_no_processing = skeleton_binary_mask

# is_continous, labelled_bodies = ensure_continous_body(centerline_binary_mask)
#
# if is_continous:
#     print("is continous")
# else:
#     print("it ain't continous")
#     print(np.unique(labelled_bodies))

# create_projection_view(centerline_binary_mask)

endpoints, bifurcation_points = extract_endpoint_and_bifurcation_coordinates(skeleton_binary_mask_no_processing)
unprocessed_sparse_skeleton_graph = skeleton_to_sparse_graph(skeleton_binary_mask_no_processing, bifurcation_points, endpoints)

endpoints, bifurcation_points = extract_endpoint_and_bifurcation_coordinates(skeleton_binary_mask)

bifurcation_points = remove_redundant_bifurcation_clusters(bifurcation_points)
bifurcation_points = remove_sharp_bend_bifurcations(bifurcation_points, skeleton_binary_mask)

# print('Endpoint Coordinates:')
# print(endpoints)
# print('Bifurcation Coordinates:')
# print(bifurcation_points)
sparse_skeleton_graph = skeleton_to_sparse_graph(skeleton_binary_mask, bifurcation_points, endpoints)

print(f'Unprocessed Skeleton has {unprocessed_sparse_skeleton_graph.number_of_nodes()} nodes and {unprocessed_sparse_skeleton_graph.number_of_edges()} edges')
print(f'Processed Skeleton has {sparse_skeleton_graph.number_of_nodes()} nodes and {sparse_skeleton_graph.number_of_edges()} edges')

diameters = compute_branch_diameters_of_graph(sparse_skeleton_graph, distance_array)
lengths = compute_branch_lengths_of_graph(sparse_skeleton_graph, spacing_info )
branch_metrics = merge_branch_metrics(diameters, lengths)
print('Branch Metrics')
for branch_name, branch_value in branch_metrics.items():
    print(branch_name)
    for key, value in branch_value.items():
        print(f'{key}: {value}')


visualize_3d_graph(sparse_skeleton_graph, original_one_sided_mask)
