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


def compute_bifurcation_angles_legacy(proximal_points, distal_main_points, side_branch_points,
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

def compute_bifurcation_angles(points_list, plane_normal, spacing_info):
    """
    Computes the three bifurcation angles between parent/child branches in the bifurcation plane
    without assuming which child branch is distal or a side branch.

    Following the coronary atlas paper's definitions, all angles are calculated in 2D by projecting
    the vessel directions onto the bifurcation plane. The three angles should sum to 360 degrees.

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
            projections.append(projection)
        else:
            return angles

    angle_indexes = [[0,1], [1,2], [0,2]]

    for angle_index in angle_indexes:
        cosine = np.dot(projections[angle_index[0]], projections[angle_index[1]])
        angle_radians = np.arccos(np.clip(cosine, -1.0, 1.0))
        angles.append(np.degrees(angle_radians))

    return angles

def determine_child_branch_angle_designations(edges, angles, angle_weight=0.75):
    """
    Determines which child branch is the distal main vessel vs side branch using a scoring system.

    Scores each child branch based on:
    - Angle with parent: angles closer to 180° (straight) score higher, 90° (perpendicular) scores lowest
    - Diameter: larger diameters score higher

    Args:
        edges (list): List of edges in format [(node1, node2, edge_data), ...]
                      where edges[0] is parent, edges[1] is child_1, edges[2] is child_2
        angles (list): List of angles [parent-child_1, child_1-child_2, parent-child_2]
        angle_weight (float): Weight for angle scoring (0-1), remainder is diameter weight

    Returns:
        dict: Labeled angles {'angle_A': parent-side, 'angle_B': child_1-child_2, 'angle_C': parent-distal}
    """
    diameter_weight = 1 - angle_weight
    eps = 1e-6  # Small epsilon for numerical stability

    # Validate input
    if len(angles) != 3:
        raise ValueError(f"Expected 3 angles, got {len(angles)}")
    if len(edges) < 3:
        raise ValueError(f"Expected at least 3 edges, got {len(edges)}")
    if any(a is None for a in angles):
        raise ValueError(f"Angles contain None values: {angles}")

    child_1_parent_angle = angles[0]
    child_2_parent_angle = angles[2]
    bifurcation_angle = angles[1]

    # Extract diameters (try multiple possible keys)
    edge1_data = edges[1][2]
    edge2_data = edges[2][2]

    # Try different diameter keys (from different processing methods)
    if 'mean_diameter' in edge1_data:
        child_1_diameter = edge1_data['mean_diameter']
        child_2_diameter = edge2_data['mean_diameter']
    elif 'average_diameter_mm_edt' in edge1_data:
        child_1_diameter = edge1_data['average_diameter_mm_edt']
        child_2_diameter = edge2_data['average_diameter_mm_edt']
    elif 'median_diameter' in edge1_data:
        child_1_diameter = edge1_data['median_diameter']
        child_2_diameter = edge2_data['median_diameter']
    else:
        raise KeyError(
            f"No diameter information found in edge data. "
            f"Available keys: {list(edge1_data.keys())}"
        )

    # Angle scoring: distance from 90° (perpendicular)
    # 180° (straight) → score = 1.0 (best)
    # 90° (perpendicular) → score = 0.0 (worst)

    angle_score_1 = abs(child_1_parent_angle - 90.0) / 90.0
    angle_score_2 = abs(child_2_parent_angle - 90.0) / 90.0

    # Diameter scoring: normalize between the two children
    # Larger diameter gets higher score [0, 1]
    min_d = min(child_1_diameter, child_2_diameter)
    max_d = max(child_1_diameter, child_2_diameter)
    if abs(max_d - min_d) < eps:
        # Diameters are essentially equal
        diam_score_1 = diam_score_2 = 0.5
    else:
        diam_score_1 = (child_1_diameter - min_d) / (max_d - min_d)
        diam_score_2 = (child_2_diameter - min_d) / (max_d - min_d)

    total_score_1 = angle_weight * angle_score_1 + diameter_weight * diam_score_1
    total_score_2 = angle_weight * angle_score_2 + diameter_weight * diam_score_2

    if total_score_1 >= total_score_2:
        labelled_angles = {
            'angle_A': child_2_parent_angle, 
            'angle_B': bifurcation_angle,    
            'angle_C': child_1_parent_angle  
        }
    else:
        labelled_angles = {
            'angle_A': child_1_parent_angle, 
            'angle_B': bifurcation_angle,    
            'angle_C': child_2_parent_angle  
        }

    return labelled_angles


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

            bifurcation_angles = compute_bifurcation_angles(edge_point_lists, plane_normal, spacing_info)

            if len(bifurcation_angles) == 0:
                continue

            depth_measurements.append({
                'depth': depth,
                'inflow_angle': inflow_angle,
                'angle_0_1': bifurcation_angles[0],
                'angle_1_2': bifurcation_angles[1],
                'angle_0_2': bifurcation_angles[2]
            })

        except Exception as e:
            print(e)
            continue

    if len(depth_measurements) == 0:
        return None

    # Compute averages for unlabeled angles
    inflow_angles = [m['inflow_angle'] for m in depth_measurements if m['inflow_angle'] is not None]
    angles_0_1 = [m['angle_0_1'] for m in depth_measurements if m['angle_0_1'] is not None]
    angles_1_2 = [m['angle_1_2'] for m in depth_measurements if m['angle_1_2'] is not None]
    angles_0_2 = [m['angle_0_2'] for m in depth_measurements if m['angle_0_2'] is not None]

    # Create averaged angle array for labeling
    averaged_unlabeled_angles = [
        np.mean(angles_0_1) if angles_0_1 else None,
        np.mean(angles_1_2) if angles_1_2 else None,
        np.mean(angles_0_2) if angles_0_2 else None
    ]

    # Determine A, B, C labels based on averaged angles (only done once!)
    labeled_angles = determine_child_branch_angle_designations(edges, averaged_unlabeled_angles)

    # Create mapping from unlabeled indices to labels
    # e.g., {0: 'angle_A', 1: 'angle_B', 2: 'angle_C'}
    index_to_label = {}
    std_mapping = {}

    for label, value in labeled_angles.items():
        if value == averaged_unlabeled_angles[0]:
            index_to_label[0] = label
            std_mapping[label] = np.std(angles_0_1) if angles_0_1 else None
        elif value == averaged_unlabeled_angles[1]:
            index_to_label[1] = label
            std_mapping[label] = np.std(angles_1_2) if angles_1_2 else None
        elif value == averaged_unlabeled_angles[2]:
            index_to_label[2] = label
            std_mapping[label] = np.std(angles_0_2) if angles_0_2 else None

    # Relabel depth measurements to use A, B, C
    relabeled_depth_measurements = []
    for measurement in depth_measurements:
        relabeled_measurement = {
            'depth': measurement['depth'],
            'inflow_angle': measurement['inflow_angle'],
            index_to_label[0]: measurement['angle_0_1'],
            index_to_label[1]: measurement['angle_1_2'],
            index_to_label[2]: measurement['angle_0_2']
        }
        relabeled_depth_measurements.append(relabeled_measurement)

    result = {
        'bifurcation_node': bifurcation_node,
        'averaged_inflow_angle': np.mean(inflow_angles) if inflow_angles else None,
        'averaged_angle_A': labeled_angles['angle_A'],
        'averaged_angle_B': labeled_angles['angle_B'],
        'averaged_angle_C': labeled_angles['angle_C'],
        'std_inflow_angle': np.std(inflow_angles) if inflow_angles else None,
        'std_angle_A': std_mapping['angle_A'],
        'std_angle_B': std_mapping['angle_B'],
        'std_angle_C': std_mapping['angle_C'],
        'depth_measurements': relabeled_depth_measurements,
        'num_measurements': len(depth_measurements)
    }

    return result

def compute_angles_at_bifurcation_legacy(bifurcation_node, directed_graph, spacing_info,
                                   min_depth_mm=5.0, max_depth_mm=10.0, step_mm=0.5):
    """
    LEGACY VERSION - uses manual main/side branch designation based on path length.

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

            bifurcation_angles = compute_bifurcation_angles_legacy(
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
