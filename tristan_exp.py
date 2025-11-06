from utilities import load_nrrd_mask, ensure_continous_body, extract_centerline_skimage, extract_endpoint_and_bifurcation_coordinates, build_graph
from visualizations import create_projection_view, visualize_3d_graph
import numpy as np

path = "data/batch_1/Normal_1.nrrd"
binary_mask = load_nrrd_mask(path, verbose=True)

is_continous, labelled_bodies = ensure_continous_body(binary_mask)

if is_continous:
    print("is continous")
else:
    print("it ain't continous")
    print(np.unique(labelled_bodies))

original_one_sided_mask = (labelled_bodies == 2).astype(np.uint8)
# create_projection_view(binary_mask)

centerline_binary_mask = extract_centerline_skimage(binary_mask)

is_continous, labelled_bodies = ensure_continous_body(centerline_binary_mask)

if is_continous:
    print("is continous")
else:
    print("it ain't continous")
    print(np.unique(labelled_bodies))

# create_projection_view(centerline_binary_mask)

one_sided_mask = (labelled_bodies == 2).astype(np.uint8)

endpoints, bifurcation_points = extract_endpoint_and_bifurcation_coordinates(one_sided_mask)

# print('Endpoint Coordinates:')
# print(endpoints)
# print('Bifurcation Coordinates:')
# print(bifurcation_points)

one_sided_graph = build_graph(one_sided_mask, bifurcation_points, endpoints)

print(one_sided_graph.number_of_nodes())
print(one_sided_graph.number_of_edges())


visualize_3d_graph(one_sided_graph, original_one_sided_mask)
