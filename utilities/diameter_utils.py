import numpy as np
import networkx as nx
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, map_coordinates
from skimage import measure
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

    The function extracts radius values at each coordinate in the branch path and computes
    statistics (mean, median) as well as returning the full diameter profile.

    Args:
        distance_array (array): 3D distance transform array where values represent radii in mm (or voxel units)
        branch_coordinates (list): List of (x, y, z) coordinate tuples representing the voxel path of the branch

    Returns:
        mean_diameter (float): Mean diameter along the branch
        median_diameter (float): Median diameter along the branch
        diameter_profile (list): List of diameter values at each voxel in branch_coordinates
    """

    coordinate_array = np.array(branch_coordinates)

    # Extract radius values at each coordinate along the branch path
    radius_values = distance_array[tuple(coordinate_array.T)]

    # Diameter profile: diameter at each voxel (2 * radius)
    diameter_profile = (radius_values * 2).tolist()

    mean_diameter = np.mean(diameter_profile)
    median_diameter = np.median(diameter_profile)

    return mean_diameter, median_diameter, diameter_profile


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
                                        edge_data['mean_diameter_edt'] => avg diameter using edt distance transform
                                        edge_data['median_diameter_edt'] => median diameter using edt distance transform
                                        edge_data['diameter_profile_edt'] => list of diameter values along the branch
    """
    if graph.is_directed():
        updated_graph = nx.DiGraph(graph)
    else:
        updated_graph = nx.Graph(graph)

    for edge in list(updated_graph.edges()):
        voxel_path = updated_graph.edges[edge]["voxels"]
        average_diameter, median_diameter, diameter_profile = compute_average_diameter_of_branch(
            distance_array, voxel_path
        )
        updated_graph.edges[edge]["mean_diameter_edt"] = average_diameter
        updated_graph.edges[edge]["median_diameter_edt"] = median_diameter
        updated_graph.edges[edge]["diameter_profile_edt"] = diameter_profile

    return updated_graph


def determine_origin_node_from_diameter(graph, distance_array = None):
    """
    Determines the origin node of a vessel tree based on branch diameter analysis.

    Identifies the root/origin of a vascular tree by finding the endpoint with the
    largest diameter branch. This is based on the principle that vessel diameter
    decreases from proximal to distal, so the largest diameter branch should be
    connected to the origin point (e.g., ostium of coronary artery).

    Algorithm:
    1. Finds all edges connected to endpoint nodes (degree = 1)
    2. Special case: For single-branch graphs (2 endpoints, 1 edge), uses z-coordinate
       (higher z = origin) instead of diameter
    3. Retrieves diameter information from edge attributes or computes from distance array
    4. Identifies the edge with the largest mean diameter
    5. Returns the endpoint node of that edge as the origin

    Args:
        graph (networkx.Graph): Vessel graph with diameter information in edge attributes
        distance_array (numpy.ndarray, optional): 3D distance transform array for computing
                                                  diameters if not already in graph attributes

    Returns:
        origin (tuple): Coordinates of the origin node (x, y, z) in voxel space

    Raises:
        ValueError: If no diameter information is available or no valid origin can be determined

    Notes:
        - Checks for diameter attributes in order: 'mean_diameter_slicing', 'mean_diameter_edt'
        - If neither exists, computes from distance_array if provided
        - Assumes the largest diameter branch is connected to the vessel origin
        - For single-branch cases, the node with higher z-coordinate is selected as origin
    """
    # Get all endpoint nodes (degree = 1)
    endpoint_nodes = [node for node in graph.nodes() if graph.degree(node) == 1]

    # Special case: single branch with two endpoints
    # Use z-coordinate comparison instead of diameter (higher z = origin)
    # This commonly occurs in RCA where a single branch might be mislabeled
    if len(endpoint_nodes) == 2 and graph.number_of_edges() == 1:
        node1, node2 = endpoint_nodes
        # Compare z-coordinates (dim_2, index 2)
        if node1[2] > node2[2]:
            return node1
        else:
            return node2

    # Normal case: use diameter-based approach
    largest_diameter = 0
    largest_edge = None

    endpoint_edges = [edge for node in endpoint_nodes
              for edge in graph.edges(node)]


    for edge in endpoint_edges:
        if 'mean_diameter_slicing' in graph.edges[edge]:
            average_diameter = graph.edges[edge]['mean_diameter_slicing']
        elif 'mean_diameter_edt' in graph.edges[edge]:
            average_diameter = graph.edges[edge]['mean_diameter_edt']
        elif distance_array is not None:
            voxel_path = graph.edges[edge]["voxels"]
            average_diameter, median_diameter, diameter_profile = compute_average_diameter_of_branch(distance_array, voxel_path)
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
            raise ValueError(
                "Unable to determine origin node from graph"
            )
        return origin
    else:
        raise ValueError(
            "Unable to determine origin node from graph"
        )


def compute_normal_vector_at_voxel(voxels, index):
    """
    Computes a smoothed normal/tangent vector at a specific voxel along a path.

    Uses averaging of multiple adjacent vectors to create a smoother estimate of the
    vessel direction at each point. This helps reduce noise in the tangent estimation
    which is crucial for accurate orthogonal slice extraction.

    Algorithm:
    - For paths with 5+ voxels: averages 4 segment vectors centered around the index
      (segments: i-2→i-1, i-1→i, i→i+1, i+1→i+2)
    - Near boundaries: takes extra vectors from the opposite direction
    - For paths with <5 voxels: averages all available segment vectors

    Args:
        voxels (list): List of (x, y, z) coordinate tuples representing the voxel path
        index (int): Index of the voxel where normal vector should be computed

    Returns:
        normal_vector (numpy.ndarray): Normalized 3D vector representing the tangent direction
    """
    voxels_array = np.array(voxels)
    n = len(voxels)

    if n < 2:
        raise ValueError("Need at least 2 voxels to compute a normal vector")

    # Special case: very short paths (< 5 voxels)
    # Use average of all available segment vectors
    if n < 5:
        vectors = []
        for i in range(n - 1):
            vec = voxels_array[i + 1] - voxels_array[i]
            vectors.append(vec)
        avg_vector = np.mean(vectors, axis=0)
        return avg_vector / np.linalg.norm(avg_vector)

    # Normal case: 5+ voxels
    # Collect 4 segment vectors around the current index
    vectors = []

    if index < 2:
        indices = [0, 1, 2, 3]
    elif index >= n - 2:
        indices = [n - 4, n - 3, n - 2, n - 1]
    else:
        indices = [index - 2, index - 1, index, index + 1]

    for i in range(len(indices) - 1):
        vec = voxels_array[indices[i + 1]] - voxels_array[indices[i]]
        vectors.append(vec)

    avg_vector = np.mean(vectors, axis=0)

    return avg_vector / np.linalg.norm(avg_vector)


def compute_diameter_at_voxel(binary_mask, voxel, normal_vector, spacing_info, slice_size=20, resolution=1.0):
    """
    Computes vessel diameter at a single voxel by extracting an orthogonal slice.

    Extracts a 2D plane perpendicular to the vessel direction (normal_vector) centered
    at the given voxel position, then computes diameter using distance transform.
    The radius is measured at the center of the slice, which corresponds to the
    centerline voxel position.

    Args:
        binary_mask (numpy.ndarray): 3D binary mask of the vessel
        voxel (tuple or array): (x, y, z) coordinates of the voxel center
        normal_vector (numpy.ndarray): 3D vector perpendicular to desired slice plane
        spacing_info (tuple): Voxel spacing in physical units (e.g., mm) as (z, y, x).
                             Should be isotropic after resampling.
        slice_size (int): Half-size of the extracted slice in voxels (default: 20)
        resolution (float): Sampling resolution for the slice in voxels (default: 1.0)

    Returns:
        diameter (float): Vessel diameter at this location in mm (2 * radius at center)
    """
    voxel = np.array(voxel)

    slice_mask = extract_plane(binary_mask, voxel, normal_vector, size=slice_size, resolution=resolution)
    voxel_spacing = spacing_info[0]  # Assuming isotropic
    slice_spacing = resolution * voxel_spacing
    distance_map = distance_transform_edt(slice_mask, sampling=(slice_spacing, slice_spacing))
    # diameter=compute_diameter_circle_fitting(slice_mask, spacing=slice_spacing)
    # Sample radius at the center of the slice (centerline position)
    # The center corresponds to the original voxel position
    centerpoint_index = slice_mask.shape[0] // 2
    radius = distance_map[centerpoint_index, centerpoint_index]

    diameter = 2 * radius

    return diameter


def compute_diameter_profile_of_branch(binary_mask, branch_coordinates, spacing_info, slice_size=20, resolution=1.0):
    """
    Computes diameter profile along an entire branch using orthogonal slice method.

    For each voxel along the branch path:
    1. Computes a smoothed normal/tangent vector
    2. Extracts a 2D slice perpendicular to the vessel direction
    3. Calculates diameter using distance transform

    Returns both the complete diameter profile and summary statistics.

    Args:
        binary_mask (numpy.ndarray): 3D binary mask of the vessel
        branch_coordinates (list): List of (x, y, z) coordinate tuples representing the voxel path
        spacing_info (tuple): Voxel spacing in physical units (e.g., mm) as (z, y, x).
                             Should be isotropic after resampling.
        slice_size (int): Half-size of extracted slices in voxels (default: 20)
        resolution (float): Sampling resolution for slices (default: 1.0)

    Returns:
        diameter_profile (list): List of diameter values along the branch path in mm
        mean_diameter (float): Mean diameter of the branch in mm
        median_diameter (float): Median diameter of the branch in mm
    """
    if len(branch_coordinates) < 2:
        raise ValueError("Branch must have at least 2 voxels")

    diameter_profile = []

    for i in range(len(branch_coordinates)):
        normal = compute_normal_vector_at_voxel(branch_coordinates, i)

        diameter = compute_diameter_at_voxel(
            binary_mask,
            branch_coordinates[i],
            normal,
            spacing_info,
            slice_size=slice_size,
            resolution=resolution
        )

        diameter_profile.append(diameter)

    mean_diameter = np.mean(diameter_profile)
    median_diameter = np.median(diameter_profile)

    return diameter_profile, mean_diameter, median_diameter


def compute_branch_diameters_of_graph_slicing(graph, binary_mask, spacing_info, slice_size=20, resolution=1.0):
    """
    Computes diameter profiles for all branches in a vessel graph using orthogonal slicing method.

    Iterates through all edges in the graph and calculates diameter profiles using the
    orthogonal slice extraction method. Each edge is expected to have a 'voxels' attribute
    containing the coordinate path of the branch.

    For each edge, stores:
    - Full diameter profile as a list
    - Mean diameter
    - Median diameter

    Args:
        graph (networkx.Graph): Graph representation of vessel network with 'voxels' in edge attributes
        binary_mask (numpy.ndarray): 3D binary mask of the vessel
        spacing_info (tuple): Voxel spacing in physical units (e.g., mm) as (z, y, x).
                             Should be isotropic after resampling.
        slice_size (int): Half-size of extracted slices in voxels (default: 20)
        resolution (float): Sampling resolution for slices (default: 1.0)

    Returns:
        updated_graph (networkx.Graph): Graph with diameter information in edge attributes:
            - edge_data['diameter_profile_slicing'] => list of diameters along branch in mm
            - edge_data['mean_diameter_slicing'] => mean diameter in mm
            - edge_data['median_diameter_slicing'] => median diameter in mm
    """
    if graph.is_directed():
        updated_graph = nx.DiGraph(graph)
    else:
        updated_graph = nx.Graph(graph)

    for edge in list(updated_graph.edges()):
        voxel_path = updated_graph.edges[edge]["voxels"]

        diameter_profile, mean_diameter, median_diameter = compute_diameter_profile_of_branch(
            binary_mask,
            voxel_path,
            spacing_info,
            slice_size=slice_size,
            resolution=resolution
        )

        updated_graph.edges[edge]["mean_diameter_slicing"] = mean_diameter
        updated_graph.edges[edge]["median_diameter_slicing"] = median_diameter
        updated_graph.edges[edge]["diameter_profile_slicing"] = diameter_profile

    return updated_graph
    
def local_diameter(mask, center, tangent):
    slice_mask = extract_plane(mask, center, tangent)  # pass both center + normal
    dist_map = distance_transform_edt(slice_mask)
    return 2 * np.max(dist_map) 


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
    """
    Computes an orthonormal basis for a plane perpendicular to the given normal vector.

    Creates two orthogonal unit vectors (u, v) that span the plane perpendicular to
    the input normal vector. This is useful for defining a coordinate system in the
    plane for extracting cross-sectional slices.

    Algorithm:
    1. Normalizes the input normal vector
    2. Selects a reference vector not parallel to the normal
    3. Computes first basis vector u = normal × reference, then normalizes
    4. Computes second basis vector v = normal × u (automatically normalized)

    Args:
        normal (numpy.ndarray): 3D vector defining the plane normal direction

    Returns:
        u (numpy.ndarray): First orthonormal basis vector in the plane
        v (numpy.ndarray): Second orthonormal basis vector in the plane
    """
    normal = normal / np.linalg.norm(normal)

    # Pick arbitrary reference vector not parallel to normal
    if abs(normal[0]) < 0.9:
        ref = np.array([1, 0, 0])
    else:
        ref = np.array([0, 1, 0])

    # Compute orthonormal basis using cross products
    u = np.cross(normal, ref)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    return u, v



def extract_plane(mask, center, normal, size=40, resolution=1.0):
    """
    Extracts a 2D planar slice from a 3D binary mask perpendicular to a normal vector.

    Samples a 2D grid of points in a plane perpendicular to the given normal vector,
    centered at the specified point. Uses interpolation to extract values from the 3D
    mask at these positions. This is used for computing vessel diameter at cross-sections.

    The extracted plane is defined by:
    - Center point: the position around which the plane is centered
    - Normal vector: defines the plane orientation (perpendicular to the plane)
    - Size: half-width of the extracted region in voxels
    - Resolution: sampling density (1.0 = sample every voxel, 0.5 = subsample)

    Args:
        mask (numpy.ndarray): 3D binary mask to extract the slice from
        center (array-like): (x, y, z) coordinates of the plane center in voxel space
        normal (numpy.ndarray): 3D vector perpendicular to the desired plane
        size (int): Half-size of the extracted region in voxels (default: 20)
                   Creates a grid from -size to +size in each dimension
        resolution (float): Step size for sampling in voxels (default: 1.0)
                           Lower values create finer sampling

    Returns:
        slice_mask (numpy.ndarray): 2D binary array representing the extracted planar slice
                                   Shape is approximately (2*size/resolution, 2*size/resolution)
    """
    center = np.array(center)

    # Compute orthonormal basis vectors in the plane
    u, v = plane_basis(normal)

    # Create 2D grid in plane coordinates
    grid_range = np.arange(-size, size, resolution)
    X, Y = np.meshgrid(grid_range, grid_range)

    # Convert plane coordinates to 3D voxel coordinates
    # Each point: center + x*u + y*v
    coords = center[:, None, None] + u[:, None, None] * X + v[:, None, None] * Y

    # Sample the mask at these 3D coordinates using nearest-neighbor interpolation
    slice_mask = map_coordinates(
        mask,
        [coords[0], coords[1], coords[2]],
        order=0,  # Nearest-neighbor (appropriate for binary masks)
        mode='nearest'  # Extend boundaries by repeating edge values
    )

    return slice_mask


def fit_circle(points):
    """
    Least-squares circle fitting (Kasa method).
    points: Nx2 array of (x, y) coordinates
    Returns: (xc, yc, r)
    """
    x = points[:, 0]
    y = points[:, 1]
    A = np.c_[2*x, 2*y, np.ones_like(x)]
    b = x**2 + y**2
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    xc, yc = c[0], c[1]
    r = np.sqrt(c[2] + xc**2 + yc**2)
    return xc, yc, r

def compute_diameter_circle_fitting(slice_mask, spacing=None):

    contours = measure.find_contours(slice_mask,level=0.5)
    if not contours:
        return None
    contour = max(contours, key=len)
    points = np.array(contour)
    xc, yc, r = fit_circle(points)
    diameter = 2 * r * spacing
    return diameter

