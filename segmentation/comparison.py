import numpy as np
from utilities.centerline_utils import extract_centerline_skimage


def cl_score(volume, skeleton):
    """Compute the fraction of skeleton voxels that overlap with the volume."""
    skeleton_sum = np.sum(skeleton)
    if skeleton_sum == 0:
        return 0.0
    return np.sum(volume * skeleton) / skeleton_sum


def cl_dice_score(mask_1, mask_2):
    """
    Compute the centerline Dice (clDice) between two binary masks.

    Skeletonizes each mask and computes the harmonic mean of the
    skeleton-volume overlap in both directions.

    Args:
        mask_1 (array-like): First segmentation mask.
        mask_2 (array-like): Second segmentation mask.

    Returns:
        float: clDice score in [0, 1].
    """
    mask_1 = np.asarray(mask_1) != 0
    mask_2 = np.asarray(mask_2) != 0

    tprec = cl_score(mask_1, extract_centerline_skimage(mask_2))
    tsens = cl_score(mask_2, extract_centerline_skimage(mask_1))

    if (tprec + tsens) == 0:
        return 0.0
    return 2 * tprec * tsens / (tprec + tsens)


def compare_masks(mask_1, mask_2):
    """
    Compute the Dice similarity coefficient between two binary masks.

    Masks are treated as binary via non-zero values. Returns 1.0 when
    both masks are empty, and raises a ValueError if shapes differ.

    Args:
        mask_1 (array-like): First segmentation mask.
        mask_2 (array-like): Second segmentation mask.

    Returns:
        float: Dice coefficient in [0, 1].
    """
    mask_1 = np.asarray(mask_1)
    mask_2 = np.asarray(mask_2)

    if mask_1.shape != mask_2.shape:
        raise ValueError(
            f"Masks must have the same shape. Got {mask_1.shape} and {mask_2.shape}."
        )

    mask_1 = mask_1 != 0
    mask_2 = mask_2 != 0

    intersection = np.logical_and(mask_1, mask_2).sum()
    total = mask_1.sum() + mask_2.sum()

    if total == 0:
        return 1.0

    return (2.0 * intersection) / total
