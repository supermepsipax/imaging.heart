import numpy as np
import matplotlib.pyplot as plt

def create_projection_view(binary_mask):
    """
    Creates 3 different 2D projections of a given 3D mask,
    a top (xy) projection, front (xz) projection, and
    a side (yz) projection

    Args:
        binary_mask (array): A numpy binary mask 

    """

    projection_xy = np.max(binary_mask, axis=2)
    projection_xz = np.max(binary_mask, axis=1)
    projection_yz = np.max(binary_mask, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(projection_xy, cmap='gray', origin='lower')
    axes[0].set_title('Top-down (XY view)')

    axes[1].imshow(projection_xz, cmap='gray', origin='lower')
    axes[1].set_title('Front (XZ view)')

    axes[2].imshow(projection_yz, cmap='gray', origin='lower')
    axes[2].set_title('Side (YZ view)')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

