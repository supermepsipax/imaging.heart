import numpy as np
from scipy import ndimage
import networkx as nx
from .graph_utils import skeleton_to_dense_graph


def extract_endpoint_and_bifurcation_coordinates(skeletonized_binary_mask, remove_redundant_clusters=False):
    """
    Takes the single voxel representation of the original binary mask and finds the
    bifurcation coordinates.

    The skeletonized mask is first convolved with a 3x3x3 array of 1's, this returns a convolved
    array of the same shape where each element corresponds to how many non-zero voxels were in this 3x3x3 region around each
    original voxel. It can be assumed that for every non-zero voxel in the original skeletonized mask its
    corresponding voxel in the convolved mask can be labelled as an endpoint, bifurcation, or a point inbetween.

    Or to put in more logical terms:

    ENDPOINT => skeletonized_binary_mask[x,y,z] == 1 & convolved_mask[x,y,z] == 2
    BIFURCATION => skeletonized_binary_mask[x,y,z] == 1 && convolved_mask[x,y,z] >= 4

    Args:
        skeletonized_binary_mask (array): A single voxel width binary mask
        remove_redundant_clusters (bool): If True, removes redundant bifurcation clusters
                                         by selecting the most connected point in each cluster

    Returns:
        endpoint_coordinates (list): A list of coordinates corresponding to endpoint locations
        bifurcation_coordinates (list): A list of coordinates corresponding to bifurcation locations
    """

    weight_array = np.ones((3,3,3), dtype=int)

    convolved_binary_mask = ndimage.convolve(skeletonized_binary_mask.astype(int), weight_array, mode='constant', cval=0)
    combined_mask = skeletonized_binary_mask * convolved_binary_mask

    endpoint_coordinates = np.argwhere(combined_mask == 2)
    bifurcation_coordinates = np.argwhere(combined_mask >= 4)

    if remove_redundant_clusters:
        bifurcation_coordinates = remove_redundant_bifurcation_clusters(
            bifurcation_coordinates, combined_mask
        )

    return endpoint_coordinates, bifurcation_coordinates


def remove_redundant_bifurcation_clusters(bifurcation_coordinates, combined_mask=None):
    """
    Removes redundant bifurcation points from clusters of 3 or more mutually touching points.

    When thick vessels create multiple adjacent bifurcation points that are all touching
    each other (within sqrt(3) voxels for 26-connectivity), this indicates a single
    junction that has been over-detected. This function identifies such clusters of 3+
    mutually touching bifurcations and keeps only one representative point from each cluster.

    If a combined_mask is provided (skeleton * convolved), the most connected bifurcation
    point (highest connectivity value) is chosen as the representative. This helps prevent
    bypass paths in robust graph construction.

    Two legitimate bifurcation points that are 1 voxel apart are preserved (e.g., branches
    on opposite sides of a vessel).

    Args:
        bifurcation_coordinates (array): Array of 3D bifurcation point coordinates
        combined_mask (array, optional): Combined mask with connectivity values for selecting
                                        the most connected representative

    Returns:
        filtered_bifurcations: Array of bifurcation coordinates with redundant clusters removed

    """
    if len(bifurcation_coordinates) < 3:
        return bifurcation_coordinates

    bifurcations = np.array(bifurcation_coordinates)

    touch_graph = nx.Graph()
    touch_graph.add_nodes_from(range(len(bifurcations)))

    max_touch_distance = np.sqrt(3) + 0.01  # small epsilon for floating point

    for i in range(len(bifurcations)):
        for j in range(i + 1, len(bifurcations)):
            distance = np.linalg.norm(bifurcations[i] - bifurcations[j])
            if distance <= max_touch_distance:
                touch_graph.add_edge(i, j)

    components = list(nx.connected_components(touch_graph))

    # Get connectivity values if mask provided
    connectivity_values = None
    if combined_mask is not None:
        connectivity_values = np.array([combined_mask[tuple(coord)] for coord in bifurcations])

    indices_to_keep = []
    for component in components:
        if len(component) >= 3:
            if connectivity_values is not None:
                component_list = list(component)
                component_connectivity = [connectivity_values[i] for i in component_list]
                representative = component_list[np.argmax(component_connectivity)]
            else:
                representative = min(component)
            indices_to_keep.append(representative)
        else:
            indices_to_keep.extend(component)

    filtered_bifurcations = bifurcations[sorted(indices_to_keep)]

    return filtered_bifurcations


def remove_sharp_bend_bifurcations(bifurcation_coordinates, binary_mask, neighborhood_size=5):
    """
    Removes false bifurcation points that arise from sharp bends in the skeleton.

    At 90-degree bends in a skeleton, the voxels adjacent to the corner can be
    falsely detected as bifurcations. This function uses topological analysis
    to distinguish sharp bends from legitimate nearby bifurcations.

    For each pair of touching bifurcations, it extracts a local neighborhood and
    builds a graph representation. By counting the number of endpoints (degree-1 nodes)
    in this local region:
    - Exactly 2 endpoints → sharp bend (path in, path out)
    - 3+ endpoints → real bifurcations with multiple branches

    Args:
        bifurcation_coordinates (array): Array of 3D bifurcation point coordinates
        binary_mask (array): The skeletonized binary mask
        neighborhood_size (int): Size of local cubic neighborhood to analyze (default: 5)

    Returns:
        filtered_bifurcations: Array with sharp-bend false bifurcations removed

    """
    if len(bifurcation_coordinates) < 2:
        return bifurcation_coordinates

    bifurcations = np.array(bifurcation_coordinates)
    max_touch_distance = np.sqrt(3) + 0.01

    def analyze_local_topology(bifurc_i, bifurc_j):
        """Extract local neighborhood and count endpoints using dense graph."""
        min_coords = np.minimum(bifurc_i, bifurc_j) - neighborhood_size // 2
        max_coords = np.maximum(bifurc_i, bifurc_j) + neighborhood_size // 2 + 1

        min_coords = np.maximum(min_coords, 0)
        max_coords = np.minimum(max_coords, binary_mask.shape)

        local_mask = binary_mask[
            min_coords[0]:max_coords[0],
            min_coords[1]:max_coords[1],
            min_coords[2]:max_coords[2]
        ].copy()

        local_graph = skeleton_to_dense_graph(local_mask)

        endpoint_count = sum(1 for node in local_graph.nodes() if local_graph.degree(node) == 1)

        return endpoint_count

    indices_to_remove = set()

    for i in range(len(bifurcations)):
        for j in range(i + 1, len(bifurcations)):
            distance = np.linalg.norm(bifurcations[i] - bifurcations[j])

            if distance <= max_touch_distance:
                endpoint_count = analyze_local_topology(bifurcations[i], bifurcations[j])

                if endpoint_count == 2:
                    indices_to_remove.add(i)
                    indices_to_remove.add(j)

    indices_to_keep = [i for i in range(len(bifurcations)) if i not in indices_to_remove]
    filtered_bifurcations = bifurcations[indices_to_keep]

    return filtered_bifurcations 
