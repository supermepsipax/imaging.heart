import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional


def _get_branch_label_attr(artery_type):
    """
    Get the correct branch label attribute name based on artery type.

    Args:
        artery_type (str): 'LCA' or 'RCA'

    Returns:
        str: Attribute name ('lca_branch' or 'rca_branch')
    """
    if artery_type.upper() == 'LCA':
        return 'lca_branch'
    elif artery_type.upper() == 'RCA':
        return 'rca_branch'
    else:
        # Fallback to generic 'branch_label' for unknown types
        return 'branch_label'


def extract_bifurcation_statistics(graph, spacing_info, diameter_method='slicing'):
    """
    Extract angle and diameter statistics for all bifurcations in the graph.

    Identifies bifurcation nodes and extracts:
    - Averaged angles (averaged_angle_A, averaged_angle_B, averaged_angle_C, averaged_inflow_angle)
    - Diameters of the three branches involved (PMV, DMV, side branch)

    Args:
        graph (networkx.DiGraph): Directed graph with branch labels and angle measurements
        spacing_info (tuple): Voxel spacing in mm (not currently used, but for consistency)
        diameter_method (str): 'slicing' or 'edt' - which diameter measurement to use

    Returns:
        dict: Dictionary keyed by bifurcation identifier (e.g., 'LAD_D1', 'LCx_OM2'):
            {
                'LAD_D1': {
                    'bifurcation_node': (dim_0, dim_1, dim_2),
                    'main_branch_label': 'LAD',
                    'side_branch_label': 'D1',
                    'angles': {
                        'averaged_angle_A': 45.2,       # Angle A (degrees)
                        'averaged_angle_B': 68.5,       # Angle B (degrees)
                        'averaged_angle_C': 113.7,      # Angle C (degrees)
                        'averaged_inflow_angle': 155.3  # Inflow angle (degrees)
                    },
                    'diameters': {
                        'PMV': 3.5,    # Proximal main vessel diameter (mm)
                        'DMV': 3.2,    # Distal main vessel diameter (mm)
                        'side_branch': 2.1  # Side branch diameter (mm)
                    },
                    'parent_edge': (node1, bifurc_node),
                    'main_child_edge': (bifurc_node, node2),
                    'side_child_edge': (bifurc_node, node3)
                },
                'LCx_OM1': { ... },
                ...
            }

    Note:
        - Trifurcations are skipped (detected by presence of 'Ramus' branch labels)
        - Trifurcations appear as two close bifurcations in the graph structure, not single nodes
        - Main vessel is identified as the child with the same label as parent
        - If labels don't match, larger diameter branch is considered main vessel
    """
    # Determine diameter attribute based on method
    if diameter_method == 'slicing':
        diameter_attr = 'mean_diameter_slicing'
    elif diameter_method == 'edt':
        diameter_attr = 'mean_diameter_edt'
    else:
        raise ValueError(f"Unknown diameter_method: {diameter_method}")

    bifurcations = {}

    # Iterate through all nodes looking for bifurcations
    for node in graph.nodes():
        in_degree = graph.in_degree(node)
        out_degree = graph.out_degree(node)

        # Skip if not a bifurcation (should have 1 parent, 2 children)
        if in_degree != 1 or out_degree != 2:
            continue

        # Skip if no angle data available
        node_data = graph.nodes[node]
        if 'averaged_angle_A' not in node_data or 'averaged_angle_B' not in node_data:
            continue

        # Get parent edge and children edges
        parent_edges = list(graph.in_edges(node))
        child_edges = list(graph.out_edges(node))

        if len(parent_edges) != 1 or len(child_edges) != 2:
            continue

        parent_edge = parent_edges[0]
        child1_edge = child_edges[0]
        child2_edge = child_edges[1]

        # Skip trifurcations: check if any edge involves Ramus
        # Trifurcations are represented as two close bifurcations, not a single node
        # If we see 'Ramus' or 'R1', 'R2', etc. in branch labels, skip this bifurcation
        def get_branch_label_temp(u, v):
            edge_data = graph[u][v]
            # Try LCA attribute first, then RCA, then generic branch_label
            if 'lca_branch' in edge_data:
                return edge_data['lca_branch']
            elif 'rca_branch' in edge_data:
                return edge_data['rca_branch']
            else:
                return edge_data.get('branch_label', '')

        parent_label = get_branch_label_temp(parent_edge[0], parent_edge[1])
        child1_label = get_branch_label_temp(child1_edge[0], child1_edge[1])
        child2_label = get_branch_label_temp(child2_edge[0], child2_edge[1])

        # Check if any edge is part of trifurcation (has Ramus label)
        is_trifurcation_node = (
            parent_label == 'Ramus' or parent_label.startswith('R') and len(parent_label) > 1 and parent_label[1:].isdigit() or
            child1_label == 'Ramus' or child1_label.startswith('R') and len(child1_label) > 1 and child1_label[1:].isdigit() or
            child2_label == 'Ramus' or child2_label.startswith('R') and len(child2_label) > 1 and child2_label[1:].isdigit()
        )

        if is_trifurcation_node:
            # TODO: In the future, don't skip trifurcations. Instead:
            #       1. Detect the two sequential bifurcations that make up the trifurcation
            #       2. Combine/aggregate angles from both bifurcation nodes
            #       3. Return a different structure for the main LCA bifurcation
            #          (LAD-LCx-Ramus trifurcation) with combined angle data
            #       4. May need to compute additional geometric measurements for
            #          the 3-way split configuration
            continue  # Skip trifurcation-related bifurcations for now

        # Get branch labels (check both lca_branch and rca_branch attributes)
        def get_branch_label(u, v):
            edge_data = graph[u][v]
            # Try LCA attribute first, then RCA, then generic branch_label
            if 'lca_branch' in edge_data:
                return edge_data['lca_branch']
            elif 'rca_branch' in edge_data:
                return edge_data['rca_branch']
            else:
                return edge_data.get('branch_label', 'unknown')

        parent_label = get_branch_label(parent_edge[0], parent_edge[1])
        child1_label = get_branch_label(child1_edge[0], child1_edge[1])
        child2_label = get_branch_label(child2_edge[0], child2_edge[1])

        # Determine which child is main vessel vs side branch
        # Strategy 1: Check if label matches parent (main vessel continues with same label)
        # Strategy 2: Use diameter (larger = main vessel)
        child1_is_main = False
        child2_is_main = False

        if child1_label == parent_label:
            child1_is_main = True
        elif child2_label == parent_label:
            child2_is_main = True
        else:
            # Neither matches parent label, use diameter to decide
            child1_diam = graph[child1_edge[0]][child1_edge[1]].get(diameter_attr, 0)
            child2_diam = graph[child2_edge[0]][child2_edge[1]].get(diameter_attr, 0)
            if child1_diam >= child2_diam:
                child1_is_main = True
            else:
                child2_is_main = True

        # Assign main and side branches
        if child1_is_main:
            main_child_edge = child1_edge
            side_child_edge = child2_edge
            main_label = child1_label
            side_label = child2_label
        else:
            main_child_edge = child2_edge
            side_child_edge = child1_edge
            main_label = child2_label
            side_label = child1_label

        # Create bifurcation identifier key
        # Format: MainBranch_SideBranch (e.g., 'LAD_D1', 'LCx_OM2')
        bifurcation_key = f"{main_label}_{side_label}"

        # Extract angles (keep names exactly as stored in node data)
        angle_A = node_data.get('averaged_angle_A', None)
        angle_B = node_data.get('averaged_angle_B', None)
        angle_C = node_data.get('averaged_angle_C', None)
        inflow_angle = node_data.get('averaged_inflow_angle', None)

        # Extract diameters
        pmv_diameter = graph[parent_edge[0]][parent_edge[1]].get(diameter_attr, None)
        dmv_diameter = graph[main_child_edge[0]][main_child_edge[1]].get(diameter_attr, None)
        side_diameter = graph[side_child_edge[0]][side_child_edge[1]].get(diameter_attr, None)

        # Store bifurcation data
        bifurcations[bifurcation_key] = {
            'bifurcation_node': node,
            'main_branch_label': main_label,
            'side_branch_label': side_label,
            'angles': {
                'averaged_angle_A': angle_A,
                'averaged_angle_B': angle_B,
                'averaged_angle_C': angle_C,
                'averaged_inflow_angle': inflow_angle
            },
            'diameters': {
                'PMV': pmv_diameter,  # Proximal main vessel
                'DMV': dmv_diameter,  # Distal main vessel
                'side_branch': side_diameter
            },
            'parent_edge': parent_edge,
            'main_child_edge': main_child_edge,
            'side_child_edge': side_child_edge
        }

    return bifurcations


def extract_main_branch_statistics(graph, spacing_info, artery_type='LCA', diameter_method='slicing'):
    """
    Extract comprehensive statistics for main coronary artery branches.

    For LCA: Analyzes LAD, LCx, and Ramus (if present)
    For RCA: Analyzes RCA main branch

    Args:
        graph (networkx.DiGraph): Directed graph with branch labels and diameter profiles
        spacing_info (tuple): Voxel spacing in mm as (dim_0, dim_1, dim_2) - assumed isotropic
        artery_type (str): 'LCA' or 'RCA' to determine which branches to analyze
        diameter_method (str): 'slicing' or 'edt' - which diameter profile to use

    Returns:
        dict: Dictionary with statistics for each main branch:
            {
                'LAD': {
                    'total_path_length': float,      # Sum of all edge lengths (mm)
                    'direct_path_length': float,     # Euclidean distance start->end (mm)
                    'tortuosity': float,             # total_length / direct_length
                    'mean_diameter': float,          # Mean diameter along entire branch (mm)
                    'diameter_profile': list,        # Stitched diameter profile (mm)
                    'start_coord': tuple,            # (dim_0, dim_1, dim_2) start coordinate
                    'end_coord': tuple,              # (dim_0, dim_1, dim_2) end coordinate
                    'num_edges': int,                # Number of edges in branch
                    'edge_path': list                # List of (u, v) edge tuples
                },
                'LCx': { ... },
                'Ramus': { ... }  # Only if present
            }
    """
    # Determine which main branches to look for
    if artery_type.upper() == 'LCA':
        main_branches = ['LAD', 'LCx', 'Ramus']
    elif artery_type.upper() == 'RCA':
        main_branches = ['RCA']
    else:
        raise ValueError(f"Unknown artery_type: {artery_type}. Expected 'LCA' or 'RCA'")

    # Get correct branch label attribute for this artery type
    branch_attr = _get_branch_label_attr(artery_type)

    # Determine diameter attribute names based on method
    if diameter_method == 'slicing':
        diameter_attr = 'mean_diameter_slicing'
        profile_attr = 'diameter_profile_slicing'
    elif diameter_method == 'edt':
        diameter_attr = 'mean_diameter_edt'
        profile_attr = 'diameter_profile_edt'
    else:
        raise ValueError(f"Unknown diameter_method: {diameter_method}. Expected 'slicing' or 'edt'")

    results = {}

    for branch_name in main_branches:
        branch_edges = [
            (u, v) for u, v in graph.edges()
            if graph[u][v].get(branch_attr, '') == branch_name
        ]

        if not branch_edges:
            # Branch not found (e.g., Ramus may not exist)
            continue

        ordered_edges, ordered_nodes = _order_branch_edges(graph, branch_edges)

        if not ordered_edges:
            print(f"Warning: Could not order edges for {branch_name}")
            continue

        start_coord = ordered_nodes[0]
        end_coord = ordered_nodes[-1]

        total_path_length = sum(
            graph[u][v].get('path_length_mm', 0.0)
            for u, v in ordered_edges
        )

        direct_path_length = _compute_euclidean_distance(
            start_coord, end_coord, spacing_info
        )

        tortuosity = total_path_length / direct_path_length if direct_path_length > 0 else 1.0

        diameter_profile = _stitch_diameter_profiles(
            graph, ordered_edges, profile_attr
        )

        mean_diameter = np.mean(diameter_profile) if diameter_profile else 0.0

        results[branch_name] = {
            'total_path_length': total_path_length,
            'direct_path_length': direct_path_length,
            'tortuosity': tortuosity,
            'mean_diameter': mean_diameter,
            'diameter_profile': diameter_profile,
            'start_coord': start_coord,
            'end_coord': end_coord,
            'num_edges': len(ordered_edges),
            'edge_path': ordered_edges
        }

    return results


def _order_branch_edges(graph, branch_edges):
    """
    Order branch edges from proximal to distal by traversing the directed graph.

    Args:
        graph (networkx.DiGraph): Directed graph
        branch_edges (list): List of (u, v) tuples belonging to the branch

    Returns:
        tuple: (ordered_edges, ordered_nodes)
            - ordered_edges: List of (u, v) tuples in proximal->distal order
            - ordered_nodes: List of nodes in proximal->distal order
    """
    if not branch_edges:
        return [], []

    # Create a subgraph containing only these edges
    edge_set = set(branch_edges)

    # Build adjacency map for quick lookup
    adjacency = {}
    in_degree = {}
    for u, v in branch_edges:
        adjacency.setdefault(u, []).append(v)
        in_degree[v] = in_degree.get(v, 0) + 1
        if u not in in_degree:
            in_degree[u] = 0

    # Find the start node (in-degree = 0 or minimum in-degree)
    # In a directed graph from root, the proximal node has the lowest in-degree
    start_candidates = [node for node, deg in in_degree.items() if deg == 0]

    if not start_candidates:
        # If no node has in-degree 0, pick the one with minimum in-degree
        start_node = min(in_degree.keys(), key=lambda n: in_degree[n])
    else:
        # If multiple candidates, pick one (ideally there should be only one)
        start_node = start_candidates[0]

    # Traverse from start node following edges
    ordered_edges = []
    ordered_nodes = [start_node]
    current = start_node

    visited_edges = set()
    while current in adjacency:
        # Find next node
        next_nodes = adjacency[current]

        # Filter out already visited edges
        valid_next = [n for n in next_nodes if (current, n) not in visited_edges]

        if not valid_next:
            break

        # If multiple branches, follow the main path (could be improved with more logic)
        # For now, just take the first one
        next_node = valid_next[0]

        ordered_edges.append((current, next_node))
        visited_edges.add((current, next_node))
        ordered_nodes.append(next_node)
        current = next_node

    return ordered_edges, ordered_nodes


def _compute_euclidean_distance(coord1, coord2, spacing_info):
    """
    Compute Euclidean distance between two coordinates in physical space.

    Args:
        coord1 (tuple): (dim_0, dim_1, dim_2) first coordinate in voxel space
        coord2 (tuple): (dim_0, dim_1, dim_2) second coordinate in voxel space
        spacing_info (tuple): Voxel spacing in mm as (dim_0, dim_1, dim_2)

    Returns:
        float: Euclidean distance in mm
    """
    coord1 = np.array(coord1)
    coord2 = np.array(coord2)
    spacing = np.array(spacing_info)

    # Compute distance in physical space
    # Coordinates and spacing are in the same dimensional order
    diff = (coord2 - coord1) * spacing
    distance = np.linalg.norm(diff)

    return distance


def _stitch_diameter_profiles(graph, ordered_edges, profile_attr):
    """
    Stitch together diameter profiles from multiple edges in correct order.

    Checks if profiles need to be reversed based on diameter tapering
    (diameter should generally decrease from proximal to distal).

    Args:
        graph (networkx.DiGraph): Directed graph
        ordered_edges (list): List of (u, v) tuples in proximal->distal order
        profile_attr (str): Name of the diameter profile attribute

    Returns:
        list: Combined diameter profile in mm
    """
    stitched_profile = []

    for i, (u, v) in enumerate(ordered_edges):
        edge_data = graph[u][v]

        if profile_attr not in edge_data:
            continue

        profile = edge_data[profile_attr]

        if len(profile) == 0:
            continue

        # Check if profile needs to be reversed
        # Strategy: diameter should taper (decrease) from proximal to distal
        # If this is not the first edge, compare with the last value of the previous profile
        should_reverse = False

        if i > 0 and len(stitched_profile) > 0:
            # Compare start of current profile with end of previous profile
            # If current profile starts much smaller than previous ended, it might be reversed
            prev_end = stitched_profile[-1]
            curr_start = profile[0]
            curr_end = profile[-1]

            # If end of current is closer to previous end than start, profile is likely reversed
            if abs(curr_end - prev_end) < abs(curr_start - prev_end):
                should_reverse = True
        else:
            # For first edge, check if diameter increases along the profile
            # (it should decrease, so if it increases, it's reversed)
            if len(profile) > 1:
                # Use linear regression to check overall trend
                slope = (profile[-1] - profile[0]) / len(profile)
                # If slope is significantly positive, profile is likely reversed
                if slope > 0.1:  # Threshold for considering it reversed
                    should_reverse = True

        # Add profile (reversed if needed)
        if should_reverse:
            stitched_profile.extend(reversed(profile))
        else:
            stitched_profile.extend(profile)

    return stitched_profile


def extract_all_branch_statistics(graph, spacing_info, diameter_method='slicing'):
    """
    Extract statistics for ALL branches in the graph (not just main branches).

    Args:
        graph (networkx.DiGraph): Directed graph with branch labels and diameter profiles
        spacing_info (tuple): Voxel spacing in mm as (dim_0, dim_1, dim_2)
        diameter_method (str): 'slicing' or 'edt' - which diameter profile to use

    Returns:
        list: List of dictionaries, one per branch, containing:
            - 'branch_label': Branch name
            - 'length': Edge length (mm)
            - 'mean_diameter': Mean diameter (mm)
            - 'diameter_profile': Diameter profile (list)
            - 'start_coord': Start coordinate
            - 'end_coord': End coordinate
            - 'edge_info': Branch generation/level (if available)
    """
    if diameter_method == 'slicing':
        diameter_attr = 'mean_diameter_slicing'
        profile_attr = 'diameter_profile_slicing'
    elif diameter_method == 'edt':
        diameter_attr = 'mean_diameter_edt'
        profile_attr = 'diameter_profile_edt'
    else:
        raise ValueError(f"Unknown diameter_method: {diameter_method}")

    branches = []

    for u, v in graph.edges():
        edge_data = graph[u][v]

        # Get branch label (check lca_branch, rca_branch, then branch_label)
        if 'lca_branch' in edge_data:
            branch_label = edge_data['lca_branch']
        elif 'rca_branch' in edge_data:
            branch_label = edge_data['rca_branch']
        else:
            branch_label = edge_data.get('branch_label', 'unknown')

        branch_info = {
            'branch_label': branch_label,
            'length': edge_data.get('path_length_mm', 0.0),
            'mean_diameter': edge_data.get(diameter_attr, 0.0),
            'diameter_profile': edge_data.get(profile_attr, []),
            'start_coord': u,
            'end_coord': v,
            'edge_info': edge_data.get('edge_info', None)
        }

        branches.append(branch_info)

    return branches


def compute_branch_tapering(diameter_profile):
    """
    Compute tapering rate of a diameter profile.

    Args:
        diameter_profile (list): List of diameter values along a branch

    Returns:
        dict: Dictionary containing:
            - 'absolute_change': Change in diameter from start to end (mm)
            - 'relative_change': Relative change as percentage
            - 'tapering_rate': Slope of linear fit (mm per voxel)
            - 'r_squared': R² of linear fit
    """
    if not diameter_profile or len(diameter_profile) < 2:
        return {
            'absolute_change': 0.0,
            'relative_change': 0.0,
            'tapering_rate': 0.0,
            'r_squared': 0.0
        }

    profile = np.array(diameter_profile)
    x = np.arange(len(profile))

    # Compute absolute and relative change
    absolute_change = profile[0] - profile[-1]
    relative_change = (absolute_change / profile[0] * 100) if profile[0] > 0 else 0.0

    # Fit linear regression to get tapering rate
    coeffs = np.polyfit(x, profile, 1)
    tapering_rate = coeffs[0]  # Slope

    # Compute R²
    fit = np.polyval(coeffs, x)
    ss_res = np.sum((profile - fit) ** 2)
    ss_tot = np.sum((profile - np.mean(profile)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        'absolute_change': absolute_change,
        'relative_change': relative_change,
        'tapering_rate': tapering_rate,
        'r_squared': r_squared
    }
