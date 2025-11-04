from utilities import load_nrrd_mask, ensure_continous_body, extract_centerline_skimage, extract_endpoint_and_bifurcation_coordinates
from visualizations import create_projection_view
import numpy as np

path = "data/batch_1/Normal_1.nrrd"
binary_mask = load_nrrd_mask(path, verbose=True)

is_continous, labelled_bodies = ensure_continous_body(binary_mask)

if is_continous:
    print("is continous")
else:
    print("it ain't continous")
    print(np.unique(labelled_bodies))

create_projection_view(binary_mask)

reduced_binary_mask = extract_centerline_skimage(binary_mask)

create_projection_view(reduced_binary_mask)

endpoints, bifurcation_points = extract_endpoint_and_bifurcation_coordinates(reduced_binary_mask)

print('Endpoint Coordinates:')
print(endpoints)
print('Bifurcation Coordinates:')
print(bifurcation_points)



