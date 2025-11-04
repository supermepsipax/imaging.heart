from math import comb
import numpy as np
from scipy import ndimage
# import networkx as nx


def extract_endpoint_and_bifurcation_coordinates(skeletonized_binary_mask):
    """
    Takes the single voxel representation of the original binary mask and finds the
    bifurcation coordinates.

    The skeletonized mask is first convolved with a 3x3 array of 1's, this returns a convolved
    array of the same shape where each element corresponds to how many non-zero voxels were in this 3x3 region around each
    original voxel. It can be assumed that for every non-zero voxel in the original skeletonized mask its
    corresponding voxel in the convolved mask can be labelled as an endpoint, bifurcation, or a point inbetween.

    Or to put in more logical terms:

    ENDPOINT => skeletonized_binary_mask[x,y,z] == 1 & convolved_mask[x,y,z] == 2
    BIFURCATION => skeletonized_binary_mask[x,y,z] == 1 && convolved_mask[x,y,z] >= 4

    Args:
        skeletonized_binary_mask (array): A single voxel width binary mask 

    Returns:
        endpoint_coordinates (list): A list of coordinates corresponding to endpoint locations
        bifurcation_coordinates (list): A list of coordinates corresponding to bifurcation locations
    """

    weight_array = np.ones((3,3,3), dtype=int)

    convolved_binary_mask = ndimage.convolve(skeletonized_binary_mask.astype(int), weight_array, mode='constant', cval=0)
    combined_mask = skeletonized_binary_mask * convolved_binary_mask

    endpoint_coordinates = np.argwhere(combined_mask == 2)
    bifurcation_coordinates = np.argwhere(combined_mask >= 4)

    return endpoint_coordinates, bifurcation_coordinates 
