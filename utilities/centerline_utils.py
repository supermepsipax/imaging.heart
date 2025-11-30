import numpy as np
from skimage.morphology import skeletonize
from scipy import ndimage

def ensure_continous_body(binary_mask, debug=False):
    """
    Ensures the mask passed in is a single continous body.

    First the structure of the mask will be copied to define what counts as 
    a connecting item, currently this assumes any pixel/voxel is "touching"
    if they are connected in any way, ie 26-point connectivity for 3d. Then
    the mask is split into seperate arrays where each array will be a single
    continous body along with a count of how many continous bodies were present
    in the mask.

    Args:
        binary_mask (array): A numpy binary mask 

    Returns:
        bool: True if a continous body, false otherwise

    """

    # connectivity_structure = np.ones((3,) * binary_mask.ndim, dtype=int)
    connectivity_structure = np.ones((3,3,3), dtype=int)

    labelled_bodies, num_of_bodies = ndimage.label(binary_mask, structure=connectivity_structure)

    if debug:
        print(f'\nNumber of Continous Bodies: {num_of_bodies}')

    return num_of_bodies == 1, labelled_bodies

def extract_centerline_skimage(binary_mask):
    """
    Reduces the binary_mask into a single voxel representation

    Args:
        binary_mask (array): A numpy binary mask 

    Returns:
        reduced_mask (array): A reduced version numpy binary mask 
    """
    reduced_mask = skeletonize(binary_mask)
    
    
    return reduced_mask
