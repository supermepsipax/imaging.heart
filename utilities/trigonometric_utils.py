import numpy as np
import networkx as nx
from scipy.linalg import svd


def move_along_centerline(voxel_path, distance_mm, spacing_info):
    """
    Traverses a voxel path and returns all points up to a specified distance.

    Given a path of voxel coordinates and a target distance in millimeters,
    this function accumulates points along the path until the cumulative
    distance reaches or exceeds the target distance.

    Args:
        voxel_path (list): List of 3D coordinates in array index order (axis0, axis1, axis2)
        distance_mm (float): Target distance to travel along the path in millimeters
        spacing_info (tuple): Voxel spacing in mm for each dimension (d_dim0, d_dim1, d_dim2)

    Returns:
        points (list): List of voxel coordinates traversed up to the target distance
        actual_distance (float): The actual distance traveled in mm
    """
    if len(voxel_path) < 2:
        return voxel_path.copy(), 0.0

    points = [voxel_path[0]]
    cumulative_distance = 0.0

    for i in range(1, len(voxel_path)):
        prev_voxel = np.array(voxel_path[i-1])
        curr_voxel = np.array(voxel_path[i])

        diff = (curr_voxel - prev_voxel) * np.array(spacing_info)
        segment_distance = np.linalg.norm(diff)

        cumulative_distance += segment_distance
        points.append(tuple(curr_voxel))

        if cumulative_distance >= distance_mm:
            break

    return points, cumulative_distance


def fit_bifurcation_plane(points, spacing_info):
    """
    Fits a least-squares plane to a set of 3D points.

    Uses singular value decomposition (SVD) to find the best-fit plane through
    the given points. The plane is defined by a normal vector and a point on the plane.

    Args:
        points (list): List of 3D coordinates in array index order (axis0, axis1, axis2)
        spacing_info (tuple): Voxel spacing in mm for each dimension (d_dim0, d_dim1, d_dim2)

    Returns:
        plane_normal (array): Unit normal vector to the plane (3D)
        plane_point (array): Centroid of the points (on the plane)
    """
    if len(points) < 3:
        raise ValueError("Need at least 3 points to fit a plane")

    points_array = np.array(points) * np.array(spacing_info)

    centroid = np.mean(points_array, axis=0)

    centered_points = points_array - centroid

    # The plane normal is the right singular vector corresponding to the smallest singular value
    _, _, Vh = svd(centered_points)
    plane_normal = Vh[-1]  # Last row of Vh (corresponds to smallest singular value)

    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    return plane_normal, centroid


def compute_inflow_angle(proximal_points, plane_normal, spacing_info):
    """
    Computes the inflow angle: the angle at which the proximal vessel enters the bifurcation plane.

    The inflow angle is calculated as the angle between the proximal vessel direction
    vector and the bifurcation plane normal.

    Args:
        proximal_points (list): Points along the proximal vessel (from bifurcation moving backwards)
        plane_normal (array): Normal vector to the bifurcation plane
        spacing_info (tuple): Voxel spacing in mm for each dimension (d_dim0, d_dim1, d_dim2)

    Returns:
        inflow_angle (float): Inflow angle in degrees
    """
    if len(proximal_points) < 2:
        return None

    points_array = np.array(proximal_points) * np.array(spacing_info)

    direction = points_array[-1] - points_array[0]
    direction = direction / np.linalg.norm(direction)

    cos_angle = np.abs(np.dot(direction, plane_normal))
    angle_rad = np.arcsin(np.clip(cos_angle, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def compute_bifurcation_angles(proximal_points, distal_main_points, side_branch_points,
                              plane_normal, spacing_info):
    """
    Computes the three bifurcation angles A, B, and C in the bifurcation plane.

    Following the coronary atlas paper's definitions, all angles are calculated in 2D by projecting
    the vessel directions onto the bifurcation plane. The three angles should sum to 360 degrees.

    - Angle A: between proximal main vessel and side branch
    - Angle B: bifurcation angle between distal main vessel and side branch
    - Angle C: between proximal and distal main vessel

    Args:
        proximal_points (list): Points along the proximal vessel
        distal_main_points (list): Points along the distal main vessel
        side_branch_points (list): Points along the side branch
        plane_normal (array): Normal vector to the bifurcation plane
        spacing_info (tuple): Voxel spacing in mm for each dimension

    Returns:
        dict: Dictionary containing angles A, B, and C in degrees
    """
    results = {'angle_A': None, 'angle_B': None, 'angle_C': None}

    proximal_array = np.array(proximal_points) * np.array(spacing_info)
    distal_main_array = np.array(distal_main_points) * np.array(spacing_info)
    side_branch_array = np.array(side_branch_points) * np.array(spacing_info)

    if len(proximal_points) >= 2:
        proximal_dir = proximal_array[-1] - proximal_array[0]
        proximal_dir = proximal_dir / np.linalg.norm(proximal_dir)
    else:
        return results

    if len(distal_main_points) >= 2:
        distal_main_dir = distal_main_array[-1] - distal_main_array[0]
        distal_main_dir = distal_main_dir / np.linalg.norm(distal_main_dir)
    else:
        distal_main_dir = None

    if len(side_branch_points) >= 2:
        side_branch_dir = side_branch_array[-1] - side_branch_array[0]
        side_branch_dir = side_branch_dir / np.linalg.norm(side_branch_dir)
    else:
        side_branch_dir = None

    # Project direction vectors onto the bifurcation plane
    # Projection formula: v_projected = v - (v · n) * n
    proximal_proj = proximal_dir - np.dot(proximal_dir, plane_normal) * plane_normal
    proximal_proj = proximal_proj / np.linalg.norm(proximal_proj)

    if distal_main_dir is not None:
        distal_main_proj = distal_main_dir - np.dot(distal_main_dir, plane_normal) * plane_normal
        distal_main_proj = distal_main_proj / np.linalg.norm(distal_main_proj)
    else:
        distal_main_proj = None

    if side_branch_dir is not None:
        side_branch_proj = side_branch_dir - np.dot(side_branch_dir, plane_normal) * plane_normal
        side_branch_proj = side_branch_proj / np.linalg.norm(side_branch_proj)
    else:
        side_branch_proj = None

    # Calculate angles in the plane using the projected vectors
    # Angle A: between proximal main vessel and side branch
    if side_branch_proj is not None:
        cos_A = np.dot(proximal_proj, side_branch_proj)
        angle_A_rad = np.arccos(np.clip(cos_A, -1.0, 1.0))
        results['angle_A'] = np.degrees(angle_A_rad)

    # Angle B: bifurcation angle between distal main vessel and side branch
    if distal_main_proj is not None and side_branch_proj is not None:
        cos_B = np.dot(distal_main_proj, side_branch_proj)
        angle_B_rad = np.arccos(np.clip(cos_B, -1.0, 1.0))
        results['angle_B'] = np.degrees(angle_B_rad)

    # Angle C: between proximal and distal main vessel
    if distal_main_proj is not None:
        cos_C = np.dot(proximal_proj, distal_main_proj)
        angle_C_rad = np.arccos(np.clip(cos_C, -1.0, 1.0))
        results['angle_C'] = np.degrees(angle_C_rad)

    return results

def compute_bifurcation_angles_test(points_list, plane_normal, spacing_info):
    """
    Computes the three bifurcation angles between parent/child branches in the bifurcation plane
    without assuming which child branch is distal or a side branch.

    Following the coronary atlas paper's definitions, all angles are calculated in 2D by projecting
    the vessel directions onto the bifurcation plane. The three angles should sum to 360 degrees.

    - Angle A: between proximal main vessel and side branch
    - Angle B: bifurcation angle between distal main vessel and side branch
    - Angle C: between proximal and distal main vessel

    Args:
        points_list (list): A list of lists in the format [parent_pts[], child_1_pts[], child_2_pts[]]
        plane_normal (array): Normal vector to the bifurcation plane
        spacing_info (tuple): Voxel spacing in mm for each dimension

    Returns:
        angles (list): List of angles in degrees in the order [parent-child_1, child_1-child_2, parent-child_2]
    """

    angles = []
    projections = []

    for point_list in points_list:
        point_array = np.array(point_list) * np.array(spacing_info)
        if len(point_list) >= 2:
            direction = point_array[-1] - point_array[0]
            direction = direction / np.linalg.norm(direction)
            projection = direction - np.dot(direction, plane_normal) * plane_normal
            projection = projection / np.linalg.norm(projection)
        else:
            return angles 
    
    angle_indexes = [[0,1], [1,2], [0,2]]

    for angle_index in angle_indexes:
        cosine = np.dot(projections[angle_index[0]], projections[angle_index[1]])
        angle_radians = np.arccos(np.clip(cosine, -1.0, 1.0))
        angles.append(np.degrees(angle_radians))

    return angles

def determine_child_branch_angle_designations(edges, angles, angle_weight=0.75):

    diameter_weight = 1 - angle_weight
    
    labelled_angles = {'angle_A': None, 'angle_B': angles[1], 'angle_C': None}
    child_1_parent_angle = angles[0]
    child_2_parent_angle = angles[1]
    child_1_diameter =edges[1][2]['average_diameter_mm_edt']
    child_2_diameter =edges[2][2]['average_diameter_mm_edt']


    aA = max(0.0, min(angle_max, angle_with_parent_A))
    aB = max(0.0, min(angle_max, angle_with_parent_B))
    angle_score_A = 1.0 - (aA / angle_max)
    angle_score_B = 1.0 - (aB / angle_max)

    # Diameter normalization across the two children
    min_d = min(diam_A, diam_B)
    max_d = max(diam_A, diam_B)
    if abs(max_d - min_d) < eps:
        diam_score_A = diam_score_B = 0.5
    else:
        diam_score_A = (diam_A - min_d) / (max_d - min_d)
        diam_score_B = (diam_B - min_d) / (max_d - min_d)


def compute_angles_at_bifurcation_test(bifurcation_node, directed_graph, spacing_info,
                                   min_depth_mm=5.0, max_depth_mm=10.0, step_mm=0.5):
    """
    Computes all bifurcation angles at a single bifurcation node following the paper's methodology.

    At each depth from min_depth_mm to max_depth_mm (in step_mm increments), this function:
    1. Collects points along each vessel up to that depth
    2. Fits a bifurcation plane to all collected points
    3. Calculates inflow angle and bifurcation angles A, B, C
    4. Averages the angles over the depth range

    Args:
        bifurcation_node (tuple): The coordinate of the bifurcation point
        directed_graph (nx.DiGraph): The directed graph with 'voxels' edge attributes
        spacing_info (tuple): Voxel spacing in mm for each dimension
        min_depth_mm (float): Minimum depth for averaging (default 5.0 mm)
        max_depth_mm (float): Maximum depth for averaging (default 10.0 mm)
        step_mm (float): Step size for depth increments (default 0.5 mm)

    Returns:
        dict: Dictionary containing averaged angles and angle measurements at each depth
    """
    in_edges = list(directed_graph.in_edges(bifurcation_node, data=True))
    out_edges = list(directed_graph.out_edges(bifurcation_node, data=True))

    if len(in_edges) != 1 or len(out_edges) != 2:
        return None
    
    edges = in_edges + out_edges
    edge_voxel_paths = []

    for edge in edges:
        voxels = list(edge[2]['voxels'])

        if voxels[-1] == bifurcation_node:
            voxels = list(reversed(voxels))
        elif voxels[0] != bifurcation_node:
            # Neither end matches - this shouldn't happen but handle gracefully
            return None
        edge_voxel_paths.append(voxels)

    depth_measurements = []
    depths = np.arange(min_depth_mm, max_depth_mm + step_mm, step_mm)

    for depth in depths:
        try:
            total_points = []
            edge_point_lists = []
            invalid_length = False
            for edge, edge_voxel_path in zip(edges, edge_voxel_paths):
                points, distance = move_along_centerline(edge_voxel_path, depth, spacing_info)
                if len(points) < 2:
                    invalid_length = True
                total_points += points
                edge_point_lists.append(points)

            if invalid_length:
                continue

            plane_normal, plane_point = fit_bifurcation_plane(total_points, spacing_info)

            inflow_angle = compute_inflow_angle(edge_point_lists[0], plane_normal, spacing_info)

            bifurcation_angles = compute_bifurcation_angles_test(edge_point_lists, plane_normal, spacing_info)
            
            if len(bifurcation_angles) == 0:
                continue

            depth_measurements.append({
                'depth': depth,
                'inflow_angle': inflow_angle,
                'angle_A': bifurcation_angles['angle_0_1'],
                'angle_B': bifurcation_angles['angle_1_2'],
                'angle_C': bifurcation_angles['angle_0_2']
            })

        except Exception as e:
            print(e)
            continue

    if len(depth_measurements) == 0:
        return None

    inflow_angles = [m['inflow_angle'] for m in depth_measurements if m['inflow_angle'] is not None]
    angles_A = [m['angle_A'] for m in depth_measurements if m['angle_A'] is not None]
    angles_B = [m['angle_B'] for m in depth_measurements if m['angle_B'] is not None]
    angles_C = [m['angle_C'] for m in depth_measurements if m['angle_C'] is not None]

    result = {
        'bifurcation_node': bifurcation_node,
        'averaged_inflow_angle': np.mean(inflow_angles) if inflow_angles else None,
        'averaged_angle_A': np.mean(angles_A) if angles_A else None,
        'averaged_angle_B': np.mean(angles_B) if angles_B else None,
        'averaged_angle_C': np.mean(angles_C) if angles_C else None,
        'std_inflow_angle': np.std(inflow_angles) if inflow_angles else None,
        'std_angle_A': np.std(angles_A) if angles_A else None,
        'std_angle_B': np.std(angles_B) if angles_B else None,
        'std_angle_C': np.std(angles_C) if angles_C else None,
        'depth_measurements': depth_measurements,
        'num_measurements': len(depth_measurements)
    }

    return result

def compute_angles_at_bifurcation(bifurcation_node, directed_graph, spacing_info,
                                   min_depth_mm=5.0, max_depth_mm=10.0, step_mm=0.5):
    """
    Computes all bifurcation angles at a single bifurcation node following the paper's methodology.

    At each depth from min_depth_mm to max_depth_mm (in step_mm increments), this function:
    1. Collects points along each vessel up to that depth
    2. Fits a bifurcation plane to all collected points
    3. Calculates inflow angle and bifurcation angles A, B, C
    4. Averages the angles over the depth range

    Args:
        bifurcation_node (tuple): The coordinate of the bifurcation point
        directed_graph (nx.DiGraph): The directed graph with 'voxels' edge attributes
        spacing_info (tuple): Voxel spacing in mm for each dimension
        min_depth_mm (float): Minimum depth for averaging (default 5.0 mm)
        max_depth_mm (float): Maximum depth for averaging (default 10.0 mm)
        step_mm (float): Step size for depth increments (default 0.5 mm)

    Returns:
        dict: Dictionary containing averaged angles and angle measurements at each depth
    """
    in_edges = list(directed_graph.in_edges(bifurcation_node, data=True))
    out_edges = list(directed_graph.out_edges(bifurcation_node, data=True))

    if len(in_edges) != 1 or len(out_edges) != 2:
        return None

    proximal_edge = in_edges[0]
    proximal_voxels = list(proximal_edge[2]['voxels'])

    if proximal_voxels[0] == bifurcation_node:
        proximal_voxels = list(reversed(proximal_voxels))
    elif proximal_voxels[-1] != bifurcation_node:
        # Neither end matches - this shouldn't happen but handle gracefully
        return None

    proximal_voxels_reversed = list(reversed(proximal_voxels))

    distal_edge_1 = out_edges[0]
    distal_voxels_1 = list(distal_edge_1[2]['voxels'])

    if distal_voxels_1[-1] == bifurcation_node:
        distal_voxels_1 = list(reversed(distal_voxels_1))
    elif distal_voxels_1[0] != bifurcation_node:
        return None

    distal_edge_2 = out_edges[1]
    distal_voxels_2 = list(distal_edge_2[2]['voxels'])

    if distal_voxels_2[-1] == bifurcation_node:
        distal_voxels_2 = list(reversed(distal_voxels_2))
    elif distal_voxels_2[0] != bifurcation_node:
        return None

    # Determine which is main vessel and which is side branch
    # 
    def path_length(voxels, spacing):
        if len(voxels) < 2:
            return 0.0
        length = 0.0
        for i in range(1, len(voxels)):
            diff = (np.array(voxels[i]) - np.array(voxels[i-1])) * np.array(spacing)
            length += np.linalg.norm(diff)
        return length

    length_1 = path_length(distal_voxels_1, spacing_info)
    length_2 = path_length(distal_voxels_2, spacing_info)

    if length_1 >= length_2:
        distal_main_voxels = distal_voxels_1
        side_branch_voxels = distal_voxels_2
    else:
        distal_main_voxels = distal_voxels_2
        side_branch_voxels = distal_voxels_1

    depth_measurements = []
    depths = np.arange(min_depth_mm, max_depth_mm + step_mm, step_mm)

    for depth in depths:
        try:
            proximal_pts, prox_dist = move_along_centerline(proximal_voxels_reversed, depth, spacing_info)
            distal_main_pts, distal_dist = move_along_centerline(distal_main_voxels, depth, spacing_info)
            side_branch_pts, side_dist = move_along_centerline(side_branch_voxels, depth, spacing_info)

            if len(proximal_pts) < 2 or len(distal_main_pts) < 2 or len(side_branch_pts) < 2:
                continue

            all_points = proximal_pts + distal_main_pts + side_branch_pts
            plane_normal, plane_point = fit_bifurcation_plane(all_points, spacing_info)

            inflow_angle = compute_inflow_angle(proximal_pts, plane_normal, spacing_info)

            bifurcation_angles = compute_bifurcation_angles(
                proximal_pts, distal_main_pts, side_branch_pts, plane_normal, spacing_info
            )

            depth_measurements.append({
                'depth': depth,
                'inflow_angle': inflow_angle,
                'angle_A': bifurcation_angles['angle_A'],
                'angle_B': bifurcation_angles['angle_B'],
                'angle_C': bifurcation_angles['angle_C']
            })

        except Exception as e:
            print(e)
            continue

    if len(depth_measurements) == 0:
        return None

    inflow_angles = [m['inflow_angle'] for m in depth_measurements if m['inflow_angle'] is not None]
    angles_A = [m['angle_A'] for m in depth_measurements if m['angle_A'] is not None]
    angles_B = [m['angle_B'] for m in depth_measurements if m['angle_B'] is not None]
    angles_C = [m['angle_C'] for m in depth_measurements if m['angle_C'] is not None]

    result = {
        'bifurcation_node': bifurcation_node,
        'averaged_inflow_angle': np.mean(inflow_angles) if inflow_angles else None,
        'averaged_angle_A': np.mean(angles_A) if angles_A else None,
        'averaged_angle_B': np.mean(angles_B) if angles_B else None,
        'averaged_angle_C': np.mean(angles_C) if angles_C else None,
        'std_inflow_angle': np.std(inflow_angles) if inflow_angles else None,
        'std_angle_A': np.std(angles_A) if angles_A else None,
        'std_angle_B': np.std(angles_B) if angles_B else None,
        'std_angle_C': np.std(angles_C) if angles_C else None,
        'depth_measurements': depth_measurements,
        'num_measurements': len(depth_measurements)
    }

    return result


def traverse_graph_and_compute_angles(directed_graph, spacing_info,
                                      min_depth_mm=5.0, max_depth_mm=10.0, step_mm=0.5):
    """
    Traverses the entire directed graph and computes bifurcation angles at all bifurcation points.

    This is the main function that processes a complete vascular network graph,
    identifying bifurcation nodes (nodes with 1 in-edge and 2 out-edges) and
    computing all relevant angles following the paper's methodology.

    Args:
        directed_graph (nx.DiGraph): The directed graph with 'voxels' edge attributes
        spacing_info (tuple): Voxel spacing in mm for each dimension (d_dim0, d_dim1, d_dim2)
        min_depth_mm (float): Minimum depth for averaging (default 5.0 mm)
        max_depth_mm (float): Maximum depth for averaging (default 10.0 mm)
        step_mm (float): Step size for depth increments (default 0.5 mm)

    Returns:
        list: List of dictionaries containing angle measurements for each bifurcation
    """
    results = []
    updated_graph = nx.DiGraph(directed_graph)
    

    for node in updated_graph.nodes():

        in_degree = updated_graph.in_degree(node)
        out_degree = updated_graph.out_degree(node)

        if in_degree == 1 and out_degree == 2:
            angle_data = compute_angles_at_bifurcation(
                node, updated_graph, spacing_info,
                min_depth_mm, max_depth_mm, step_mm
            )

            if angle_data is not None:
                for key, value in angle_data.items():
                    updated_graph.nodes[node][key] = value
                results.append(angle_data)

    return updated_graph