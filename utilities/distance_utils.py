import numpy as np
from scipy import ndimage


def compute_branch_path_length(edge, voxel_path, spacing_information):
    """
    Computes the length along a branch path along with the direct path between nodes

    Args:
        edge (tuple(tuple)): The coordinates of the two ends of a single edge
        voxel_path (list): List of coordinate tuples representing the voxel path of the branch
                          Coordinates are in array index order (axis0, axis1, axis2)
        spacing_information (tuple): The spacing information for distances in (axis0, axis1, axis2) order

    Returns:
        path_length: The absolute path length in the units used in spacing_information
        direct_length: The direct (endpoint to endpoint) length in units used in spacing_information
    """

    voxel_path_array = np.array(voxel_path)
    spacing = np.array(spacing_information)

    max_voxel_distance = np.sqrt(3) + 0.01  # small epsilon for floating point errors

    for i in range(len(voxel_path_array) - 1):
        voxel_distance = np.linalg.norm(voxel_path_array[i+1] - voxel_path_array[i])
        if voxel_distance > max_voxel_distance:
            raise ValueError(
                f"Voxels at indices {i} and {i+1} are not adjacent in 26-connectivity. "
                f"Distance: {voxel_distance:.3f} voxels (max allowed: {max_voxel_distance:.3f}). "
                f"Voxel path may not be properly ordered."
            )

    path_length = 0.0
    for i in range(len(voxel_path_array) - 1):
        delta_voxels = voxel_path_array[i+1] - voxel_path_array[i]
        physical_delta = delta_voxels * spacing
        step_length = np.linalg.norm(physical_delta)
        path_length += step_length

    edge_start, edge_end = edge
    edge_start_array = np.array(edge_start)
    edge_end_array = np.array(edge_end)

    physical_displacement = (edge_end_array - edge_start_array) * spacing
    direct_length = np.linalg.norm(physical_displacement)

    return path_length, direct_length

def compute_branch_lengths_of_graph(graph, spacing_information):
    """
    Computes branch lengths for all branches in a vessel graph.

    Iterates through all edges in the graph and calculates the actual and direct length
    for each branch using the voxel path stored in the edge attributes. Each edge
    in the graph is expected to have a 'voxels' attribute containing the coordinate
    path of the branch.

    Args:
        graph (networkx.Graph): Graph representation of the vessel network where edges contain 'voxels' attribute
        spacing_information (tuple): The spacing information for distances in the x,y,z direction

    Returns:
        branch_lengths: Dictionary mapping branch identifiers (e.g., 'branch_0', 'branch_1') to their path lengths and direct lengths
    """
    branch_lengths = {}

    for index, edge in enumerate(list(graph.edges())):
        edge_info = {'edge': edge}
        voxel_path = graph.edges[edge]['voxels']
        path_length, direct_length = compute_branch_path_length(edge, voxel_path, spacing_information)
        edge_info['path_length'] = path_length
        edge_info['direct_length'] = direct_length
        branch_lengths[f'branch_{index}'] = edge_info

    return branch_lengths




