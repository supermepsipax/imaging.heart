import numpy as np
from scipy import ndimage


def sort_labelled_bodies_by_size(labelled_bodies):
    """
    Takes a numpy 3d array where each non-zero integer represents a single continous body 
    within the array, and seperates out each body, sorts and returns the continous bodies
    in descending order based on total size (volume) of each body.


    Args:
        labelled_bodies (np.ndarray): A 3D binary mask.

    Returns:
        sorted_masks (List(np.ndarray)): Ordered list of continous bodies
    """

    masks = []
    unique_bodies = np.unique(labelled_bodies)
    for unique_body in unique_bodies:
        if not unique_body:
            continue
        mask = (labelled_bodies == unique_body).astype(np.uint8)
        masks.append(mask)

    sorted_masks = sorted(masks, key=lambda mask: mask.sum(), reverse=True)

    return sorted_masks

def resample_to_isotropic(binary_mask, original_spacing):
    """
    Resamples the 3D mask to an isotropic voxel spacing.

    The function takes a binary mask and its voxel spacing, and resamples the mask so that
    all axes get the same spacing. The target spacing is chosen as the smallest spacing in
    the original mask.

    Args:
        binary_mask (np.ndarray): A 3D binary mask.
        original_spacing (tuple): Voxel spacing for each axis of the input mask in mm.

    Returns:
        resampled_mask: The resampled, isotropic mask.
        new_spacing: The new voxel spacing in mm.
    """

    target_spacing = min(original_spacing)

    original_spacing = np.array(original_spacing)

    scaling_factor = original_spacing / target_spacing

    resampled_mask = ndimage.zoom(binary_mask, zoom = scaling_factor, order = 0) # Order 0 because we have binary masks

    new_spacing = tuple([target_spacing] * binary_mask.ndim)

    return resampled_mask, new_spacing


def preprocess_binary_mask(binary_mask, upsample_factor=2, sigma=1, threshold=0.5):
    """
    General preprocessing for a binary mask, first performs an upscaling based on the upscaling
    factor, then next uses a gaussian filter to smooth the resultant upsampled mask based
    on the sigma factor, then converts the mask back to binary based on the threshold value.

    Args:
        binary_mask (array): A numpy binary mask
        upsample_factor (float): The factor to upsample the mask by
        sigma (float): The smoothing factor for the gaussian filter
        threshold (float): The value for voxel binary binning, x>threshold==1 x<threshold==0

    Returns:
        smoothed_binary_mask: The mask with the applied upsampling and filtering techniques

    """

    binary_mask_upsampled = ndimage.zoom(
        binary_mask.astype(float), zoom=upsample_factor, order=1
    )
    smoothed = ndimage.gaussian_filter(binary_mask_upsampled, sigma=sigma)
    smoothed_binary_mask = (smoothed > threshold).astype(np.uint8)

    return smoothed_binary_mask
