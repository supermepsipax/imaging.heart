import numpy as np
import networkx as nx
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, map_coordinates

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
        if 'mean_diameter' in graph.edges[edge]:
            average_diameter = graph.edges[edge]['mean_diameter']
            median_diameter = graph.edges[edge]['median_diameter']
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
        # Check if either end of the largest edge is an endpoint
        if graph.degree[largest_edge[0]] == 1:
            origin = largest_edge[0]
        elif graph.degree[largest_edge[1]] == 1:
            origin = largest_edge[1]
        else:
            # Largest edge is not connected to an endpoint
            # Find all endpoints and pick the one with the largest average diameter
            endpoints = [node for node in graph.nodes() if graph.degree[node] == 1]

            if len(endpoints) == 0:
                raise ValueError("No endpoints found in graph - cannot determine origin")

            # Find endpoint with largest diameter on its connected edge
            best_endpoint = None
            best_endpoint_diameter = 0

            for endpoint in endpoints:
                # Get the single edge connected to this endpoint
                endpoint_edges = list(graph.edges(endpoint))
                if len(endpoint_edges) > 0:
                    edge = endpoint_edges[0]
                    if 'mean_diameter' in graph.edges[edge]:
                        diameter = graph.edges[edge]['mean_diameter']
                    elif distance_array is not None:
                        voxel_path = graph.edges[edge]["voxels"]
                        diameter, _ = compute_average_diameter_of_branch(distance_array, voxel_path)
                    else:
                        continue

                    if diameter > best_endpoint_diameter:
                        best_endpoint_diameter = diameter
                        best_endpoint = endpoint

            if best_endpoint is None:
                # Fallback: just use the first endpoint
                origin = endpoints[0]
            else:
                origin = best_endpoint

        return origin
    else:
        raise ValueError(
            "Unable to determine origin node from graph"
        )
    
def local_diameter(mask, center, tangent):
    slice_mask = extract_plane(mask, center, tangent)  # pass both center + normal
    dist_map = distance_transform_edt(slice_mask)
    return 2 * np.max(dist_map)  # diameter = 2 * max radius


def diameter_profile(mask, voxels):
    profile = []
    for i, v in enumerate(voxels):
        tangent = tangent_vector(voxels, i)
        d = local_diameter(mask, v, tangent)
        profile.append(d)
    return profile

def summarize_profile(profile):
    return {
        'mean_diameter': np.mean(profile),
        'median_diameter': np.median(profile),
        'min_diameter': np.min(profile),
        'max_diameter': np.max(profile),
        'std_diameter': np.std(profile),
        'slope': (profile[-1] - profile[0]) / len(profile) if len(profile) > 1 else 0
    }


def tangent_vector(voxels, i):
    if i == 0:
        return np.array(voxels[1]) - np.array(voxels[0])
    elif i == len(voxels)-1:
        return np.array(voxels[-1]) - np.array(voxels[-2])
    else:
        return np.array(voxels[i+1]) - np.array(voxels[i-1])
    
def plane_basis(normal):
    normal = normal / np.linalg.norm(normal)
    # pick arbitrary vector not parallel to normal
    if abs(normal[0]) < 0.9:
        ref = np.array([1,0,0])
    else:
        ref = np.array([0,1,0])
    u = np.cross(normal, ref)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v



def extract_plane(mask, center, normal, size=20, resolution=1.0):
    center = np.array(center)
    u, v = plane_basis(normal)
    # grid coordinates in plane
    grid_range = np.arange(-size, size, resolution)
    X, Y = np.meshgrid(grid_range, grid_range)
    coords = center[:,None,None] + u[:,None,None]*X + v[:,None,None]*Y
    # map to mask coordinates
    slice_mask = map_coordinates(mask, [coords[0], coords[1], coords[2]], order=0, mode='nearest')
    return slice_mask