import numpy as np
import networkx as nx
from scipy import ndimage


def create_distance_transform_from_mask(binary_mask, space_information=None):
    """
    Computes the Euclidean distance transform of a binary mask.

    For each voxel in the binary mask, this function calculates the Euclidean distance
    to the nearest background (zero-valued) voxel. This is useful for measuring vessel
    radii where the distance from the centerline to the vessel wall represents the radius.

    Args:
        binary_mask (array): Binary 3D array where 1 represents foreground and 0 represents background
        space_information (tuple, optional): Voxel spacing in physical units (e.g., mm) as (z, y, x).
                                            If provided, distances are computed in physical units.

    Returns:
        distance_array: Array of the same shape as binary_mask containing distance values
    """

    if space_information:
        return ndimage.distance_transform_edt(binary_mask, sampling=space_information)
    else:
        return ndimage.distance_transform_edt(binary_mask)


def compute_average_diameter_of_branch(distance_array, branch_coordinates):
    """
    Computes the average diameter along a branch path using distance transform values.

    This function extracts distance values (radii) along a specified path of voxel coordinates
    and calculates the average diameter. The distance_array should be a distance transform
    where each voxel value represents the distance (radius) from the centerline to the vessel wall.

    The function creates a binary mask of the branch path, multiplies it with the distance array
    to isolate only the radii along the branch, then computes the average diameter as twice
    the average radius.

    Args:
        distance_array (array): 3D distance transform array where values represent radii in mm (or voxel units)
        branch_coordinates (list): List of (x, y, z) coordinate tuples representing the voxel path of the branch

    Returns:
        average_branch_diameter: The average diameter of the branch in the same units as distance_array
    """

    coordinate_array = np.array(branch_coordinates)
    branch_array = np.zeros_like(distance_array)
    branch_array[tuple(coordinate_array.T)] = 1

    branch_radius_array = branch_array * distance_array

    mean_diameter = branch_radius_array[branch_radius_array != 0].mean() * 2
    median_diameter = np.median(branch_radius_array[branch_radius_array != 0]) * 2

    return mean_diameter, median_diameter


def compute_branch_diameters_of_graph(graph, distance_array):
    """
    Computes average diameters for all branches in a vessel graph.

    Iterates through all edges in the graph and calculates the average diameter
    for each branch using the voxel path stored in the edge attributes. Each edge
    in the graph is expected to have a 'voxels' attribute containing the coordinate
    path of the branch.

    Args:
        graph (networkx.Graph): Graph representation of the vessel network where edges contain 'voxels' attribute
        distance_array (array): 3D distance transform array where values represent radii

    Returns:
        updated_graph (networkx.Graph): A new graph object with diameter information encoded into edge data
                                        edge_data['average_diameter_mm_edt'] => avg diameter using edt distance transform
                                        edge_data['median_diameter_mm_edt'] => median diameter using edt distance transform
    """
    if graph.is_directed():
        updated_graph = nx.DiGraph(graph)
    else:
        updated_graph = nx.Graph(graph)

    for edge in list(updated_graph.edges()):
        voxel_path = updated_graph.edges[edge]["voxels"]
        average_diameter, median_diameter = compute_average_diameter_of_branch(
            distance_array, voxel_path
        )
        updated_graph.edges[edge]["average_diameter_mm_edt"] = average_diameter
        updated_graph.edges[edge]["median_diameter_mm_edt"] = median_diameter

    return updated_graph


def determine_origin_node_from_diameter(graph, distance_array = None):
    largest_diameter = 0
    largest_edge = None

    for edge in list(graph.edges()):
        if 'average_diameter_mm_edt' in graph.edges[edge]:
            average_diameter = graph.edges[edge]['average_diameter_mm_edt']
        elif distance_array is not None:
            voxel_path = graph.edges[edge]["voxels"]
            average_diameter, median_diameter = compute_average_diameter_of_branch(distance_array, voxel_path)
        else:
            raise ValueError(
                "Unable to determine branch diameter without existing diameter information or distance array"
            )

        if average_diameter > largest_diameter:
            largest_diameter = average_diameter
            largest_edge = edge

    if largest_edge is not None:
        if graph.degree[largest_edge[0]] == 1:
            origin = largest_edge[0]
        elif graph.degree[largest_edge[1]] == 1:
            origin = largest_edge[1]
        else:
            origin = None
        return origin
    else:
        raise ValueError(
            "Unable to determine origin node from graph"
        )
