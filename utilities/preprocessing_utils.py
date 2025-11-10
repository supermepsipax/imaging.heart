import numpy as np
from scipy import ndimage


def sort_labelled_bodies_by_size(labelled_bodies):

    masks = []
    unique_bodies = np.unique(labelled_bodies)
    for unique_body in unique_bodies:
        if not unique_body:
            continue
        mask = (labelled_bodies == unique_body).astype(np.uint8)
        masks.append(mask)

    sorted_masks = sorted(masks, key=lambda mask: mask.sum(), reverse=True)

    return sorted_masks


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
