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
    if diameter_method == 'slicing':
        diameter_attr = 'mean_diameter_slicing'
    elif diameter_method == 'edt':
        diameter_attr = 'mean_diameter_edt'
    else:
        raise ValueError(f"Unknown diameter_method: {diameter_method}")

    bifurcations = {}

    for node in graph.nodes():

        parent_edges = list(graph.in_edges(node))
        child_edges = list(graph.out_edges(node))

        if len(parent_edges) != 1 or len(child_edges) != 2:
            continue

        node_data = graph.nodes[node]
        if 'averaged_angle_A' not in node_data or 'averaged_angle_B' not in node_data:
            continue

        parent_edge = parent_edges[0]
        child1_edge = child_edges[0]
        child2_edge = child_edges[1]

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
            # NOTE: This is done in a seperate function to extract trifurcation
            continue

        def get_branch_label(u, v):
            edge_data = graph[u][v]
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

        if child1_label == parent_label:
            child1_is_main = True
        else:
            child1_diam = graph[child1_edge[0]][child1_edge[1]].get(diameter_attr, 0)
            child2_diam = graph[child2_edge[0]][child2_edge[1]].get(diameter_attr, 0)
            if child1_diam >= child2_diam:
                child1_is_main = True

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
        
        if bifurcation_key == "LCx_LAD":
            bifurcation_key = "LAD_LCx"

        angle_A = node_data.get('averaged_angle_A', None)
        angle_B = node_data.get('averaged_angle_B', None)
        angle_C = node_data.get('averaged_angle_C', None)
        inflow_angle = node_data.get('averaged_inflow_angle', None)

        pmv_diameter = graph[parent_edge[0]][parent_edge[1]].get(diameter_attr, None)
        dmv_diameter = graph[main_child_edge[0]][main_child_edge[1]].get(diameter_attr, None)
        side_diameter = graph[side_child_edge[0]][side_child_edge[1]].get(diameter_attr, None)

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
        main_branches = ['LAD', 'LCx', 'Ramus', 'D1', 'OM1']
    elif artery_type.upper() == 'RCA':
        main_branches = ['RCA', 'AM1']
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
            'tortuosity': edge_data.get('path_length_mm', 0.0)/edge_data.get('direct_path_length_mm', 0.0),
            'start_coord': u,
            'end_coord': v,
            'edge_info': edge_data.get('edge_position', None)
        }

        branches.append(branch_info)

    return branches


def _detect_lca_trifurcation(graph):
    """
    Detect LCA trifurcation (true or pseudo) by finding LAD, LCx, and Ramus branches.

    A trifurcation can be:
    - True: Single node with 3 children (LAD, LCx, Ramus)
    - Pseudo: Two close bifurcation nodes that together create the 3-way split

    Args:
        graph (networkx.DiGraph): Directed graph with 'lca_branch' labels

    Returns:
        dict or None: Dictionary with trifurcation info:
            {
                'type': 'true' or 'pseudo',
                'node': single node (for true trifurcation),
                'nodes': [node1, node2] (for pseudo-trifurcation),
                'parent_edge': (u, v, data),
                'lad_edge': (u, v, data),
                'lcx_edge': (u, v, data),
                'ramus_edge': (u, v, data)
            }
        Returns None if no trifurcation found
    """
    # Find all edges for each branch
    lad_edges = [(u, v, data) for u, v, data in graph.edges(data=True)
                 if data.get('lca_branch') == 'LAD']
    lcx_edges = [(u, v, data) for u, v, data in graph.edges(data=True)
                 if data.get('lca_branch') in ['LCx', 'LCX']]
    ramus_edges = [(u, v, data) for u, v, data in graph.edges(data=True)
                   if data.get('lca_branch') == 'Ramus' or
                   (data.get('lca_branch', '').startswith('R') and
                    len(data.get('lca_branch', '')) > 1 and
                    data.get('lca_branch', '')[1:].isdigit())]

    # If any branch is missing, no trifurcation
    if not (lad_edges and lcx_edges and ramus_edges):
        return None

    # Get the first edge of each branch (edge closest to the root)
    # Strategy: Use edge_position if available, otherwise use in-degree
    def get_first_edge(edges):
        # Check if edges have edge_position attribute
        has_position = all('edge_position' in e[2] for e in edges)

        if has_position:
            # Sort by edge_position (e.g., "11" comes before "111", "12")
            # Edge position is hierarchical: shorter = closer to root
            sorted_edges = sorted(edges, key=lambda e: (len(e[2]['edge_position']), e[2]['edge_position']))
        else:
            # Fall back to in-degree sorting
            sorted_edges = sorted(edges, key=lambda e: graph.in_degree(e[0]))

        return sorted_edges[0]

    lad_first = get_first_edge(lad_edges)
    lcx_first = get_first_edge(lcx_edges)
    ramus_first = get_first_edge(ramus_edges)

    # Get start nodes of each branch
    lad_start = lad_first[0]
    lcx_start = lcx_first[0]
    ramus_start = ramus_first[0]

    start_nodes = {lad_start, lcx_start, ramus_start}

    # Case 1: True trifurcation (all three branches start from same node)
    if len(start_nodes) == 1:
        trifurc_node = start_nodes.pop()
        parent_edges = list(graph.in_edges(trifurc_node, data=True))

        if len(parent_edges) != 1:
            return None

        return {
            'type': 'true',
            'node': trifurc_node,
            'parent_edge': parent_edges[0],
            'lad_edge': lad_first,
            'lcx_edge': lcx_first,
            'ramus_edge': ramus_first
        }

    # Case 2: Pseudo-trifurcation (two nodes)
    elif len(start_nodes) == 2:
        nodes = list(start_nodes)
        node1, node2 = nodes[0], nodes[1]

        # Check if node1 and node2 are connected (parent-child relationship)
        # One of them should be the parent of the other
        is_connected = False
        parent_node = None
        child_node = None

        if graph.has_edge(node1, node2):
            # node1 is parent of node2
            parent_node = node1
            child_node = node2
            is_connected = True
        elif graph.has_edge(node2, node1):
            # node2 is parent of node1
            parent_node = node2
            child_node = node1
            is_connected = True

        if not is_connected:
            # Nodes are not directly connected - check if they're siblings (common parent)
            # This might happen if the trifurcation is split differently
            parent1_edges = list(graph.in_edges(node1))
            parent2_edges = list(graph.in_edges(node2))

            if len(parent1_edges) == 1 and len(parent2_edges) == 1:
                parent1 = parent1_edges[0][0]
                parent2 = parent2_edges[0][0]

                if parent1 == parent2:
                    # They share a common parent - this is the trifurcation structure
                    # Use the common parent as the first node
                    parent_node = parent1
                    # The two nodes are both children
                    nodes = [node1, node2]

                    # Find the parent edge
                    parent_of_parent_edges = list(graph.in_edges(parent_node, data=True))
                    if len(parent_of_parent_edges) != 1:
                        return None

                    return {
                        'type': 'pseudo',
                        'nodes': nodes,
                        'parent_edge': parent_of_parent_edges[0],
                        'lad_edge': lad_first,
                        'lcx_edge': lcx_first,
                        'ramus_edge': ramus_first
                    }

            return None

        # Get the parent edge (edge coming into the parent_node)
        parent_edges = list(graph.in_edges(parent_node, data=True))
        if len(parent_edges) != 1:
            return None

        return {
            'type': 'pseudo',
            'nodes': [parent_node, child_node],
            'parent_edge': parent_edges[0],
            'lad_edge': lad_first,
            'lcx_edge': lcx_first,
            'ramus_edge': ramus_first
        }

    else:
        # All three branches start from different nodes - not a valid trifurcation
        return None


def extract_trifurcation_statistics(graph, spacing_info, min_depth_mm=5.0, max_depth_mm=10.0,
                                      step_mm=0.5, diameter_method='slicing'):
    """
    Extract angle statistics for LCA trifurcations (true or pseudo).

    Detects trifurcations by finding LAD, LCx, and Ramus branches, then computes:
    - Main plane angles (parent-LAD-LCx): A, B, C, inflow
    - LAD-Ramus angle: B1
    - LCx-Ramus angle: B2

    Args:
        graph (networkx.DiGraph): Directed graph with LCA branch labels
        spacing_info (tuple): Voxel spacing in mm (dim_0, dim_1, dim_2)
        min_depth_mm (float): Minimum depth for angle measurements (default 5.0 mm)
        max_depth_mm (float): Maximum depth for angle measurements (default 10.0 mm)
        step_mm (float): Step size for depth increments (default 0.5 mm)
        diameter_method (str): 'slicing' or 'edt' - which diameter measurement to use

    Returns:
        dict: Dictionary with trifurcation statistics or empty dict if no trifurcation found:
            {
                'LCA_TRIFURCATION': {
                    'type': 'true' or 'pseudo',
                    'trifurcation_node': (x, y, z) or None for pseudo,
                    'trifurcation_nodes': [(x1,y1,z1), (x2,y2,z2)] for pseudo,
                    'branches': ['LAD', 'LCx', 'Ramus'],
                    'main_plane_angles': {
                        'averaged_angle_A_main': float,  # parent-LCx angle
                        'averaged_angle_B_main': float,  # LAD-LCx angle
                        'averaged_angle_C_main': float,  # parent-LAD angle
                        'averaged_inflow_angle': float
                    },
                    'additional_angles': {
                        'averaged_angle_B1': float,  # LCx-Ramus
                        'averaged_angle_B2': float   # LAD-Ramus
                    },
                    'diameters': {
                        'parent': float,
                        'LAD': float,
                        'LCx': float,
                        'Ramus': float
                    },
                    'std_angles': { ... },
                    'num_measurements': int
                }
            }
    """
    from utilities.trigonometric_utils import compute_angles_at_trifurcation

    # Determine diameter attribute
    if diameter_method == 'slicing':
        diameter_attr = 'mean_diameter_slicing'
    elif diameter_method == 'edt':
        diameter_attr = 'mean_diameter_edt'
    else:
        raise ValueError(f"Unknown diameter_method: {diameter_method}")

    # Detect trifurcation
    trifurc_info = _detect_lca_trifurcation(graph)

    if trifurc_info is None:
        return {}

    # Extract edges
    parent_edge = trifurc_info['parent_edge']
    lad_edge = trifurc_info['lad_edge']
    lcx_edge = trifurc_info['lcx_edge']
    ramus_edge = trifurc_info['ramus_edge']

    # For true trifurcation, use the single node
    # For pseudo-trifurcation, compute angles using a virtual central point
    if trifurc_info['type'] == 'true':
        trifurc_node = trifurc_info['node']

        # Use existing compute_angles_at_trifurcation function
        angle_data = compute_angles_at_trifurcation(
            trifurc_node, graph, spacing_info,
            min_depth_mm=min_depth_mm,
            max_depth_mm=max_depth_mm,
            step_mm=step_mm
        )

        if angle_data is None:
            return {}

        # Extract diameters
        parent_diameter = graph[parent_edge[0]][parent_edge[1]].get(diameter_attr)
        lad_diameter = graph[lad_edge[0]][lad_edge[1]].get(diameter_attr)
        lcx_diameter = graph[lcx_edge[0]][lcx_edge[1]].get(diameter_attr)
        ramus_diameter = graph[ramus_edge[0]][ramus_edge[1]].get(diameter_attr)

        result = {
            'LCA_TRIFURCATION': {
                'type': 'true',
                'trifurcation_node': trifurc_node,
                'branches': ['LAD', 'LCx', 'Ramus'],
                'main_plane_angles': {
                    'averaged_angle_A_main': angle_data.get('averaged_angle_A_main'),
                    'averaged_angle_B_main': angle_data.get('averaged_angle_B_main'),
                    'averaged_angle_C_main': angle_data.get('averaged_angle_C_main'),
                    'averaged_inflow_angle': angle_data.get('averaged_inflow_angle')
                },
                'additional_angles': {
                    'averaged_angle_B1': angle_data.get('averaged_angle_B1'),
                    'averaged_angle_B2': angle_data.get('averaged_angle_B2')
                },
                'diameters': {
                    'parent': parent_diameter,
                    'LAD': lad_diameter,
                    'LCx': lcx_diameter,
                    'Ramus': ramus_diameter
                },
                'std_angles': {
                    'std_inflow_angle': angle_data.get('std_inflow_angle'),
                    'std_angle_A_main': angle_data.get('std_angle_A_main'),
                    'std_angle_B_main': angle_data.get('std_angle_B_main'),
                    'std_angle_C_main': angle_data.get('std_angle_C_main'),
                    'std_angle_B1': angle_data.get('std_angle_B1'),
                    'std_angle_B2': angle_data.get('std_angle_B2')
                },
                'num_measurements': angle_data.get('num_measurements', 0)
            }
        }

    else:  # pseudo-trifurcation
        # For pseudo-trifurcation, we need to handle it differently
        # Create a modified graph structure or compute angles manually
        # For now, we'll compute the midpoint and treat it as a virtual trifurcation
        nodes = trifurc_info['nodes']
        node1, node2 = nodes[0], nodes[1]

        # Compute midpoint in physical space
        node1_arr = np.array(node1) * np.array(spacing_info)
        node2_arr = np.array(node2) * np.array(spacing_info)
        midpoint_physical = (node1_arr + node2_arr) / 2
        midpoint_voxel = tuple((midpoint_physical / np.array(spacing_info)).astype(int))

        # We cannot directly use compute_angles_at_trifurcation for pseudo
        # Instead, we'll compute angles manually using the utility functions
        from utilities.trigonometric_utils import (
            move_along_centerline, fit_bifurcation_plane,
            compute_inflow_angle, compute_bifurcation_angles
        )

        # Get voxel paths for each branch
        parent_voxels = list(parent_edge[2]['voxels'])
        lad_voxels = list(lad_edge[2]['voxels'])
        lcx_voxels = list(lcx_edge[2]['voxels'])
        ramus_voxels = list(ramus_edge[2]['voxels'])

        # Orient voxels (they should start from their respective nodes)
        # For parent, reverse if needed to end at the trifurcation area
        # For children, they should start from their respective start nodes

        # Compute angles at multiple depths
        depth_measurements = []
        depths = np.arange(min_depth_mm, max_depth_mm + step_mm, step_mm)

        for depth in depths:
            try:
                parent_points, _ = move_along_centerline(parent_voxels, depth, spacing_info)
                lad_points, _ = move_along_centerline(lad_voxels, depth, spacing_info)
                lcx_points, _ = move_along_centerline(lcx_voxels, depth, spacing_info)
                ramus_points, _ = move_along_centerline(ramus_voxels, depth, spacing_info)

                if len(parent_points) < 2 or len(lad_points) < 2 or len(lcx_points) < 2 or len(ramus_points) < 2:
                    continue

                # Main plane
                main_points = parent_points + lad_points + lcx_points
                plane_normal_main, _ = fit_bifurcation_plane(main_points, spacing_info)
                inflow_angle = compute_inflow_angle(parent_points, plane_normal_main, spacing_info)
                main_angles = compute_bifurcation_angles(
                    [parent_points, lad_points, lcx_points],
                    plane_normal_main, spacing_info
                )

                # Angle labels (swapped A and C):
                # main_angles[0] = parent-LAD, main_angles[1] = LAD-LCx, main_angles[2] = parent-LCx
                angle_C_main = main_angles[0]  # parent-LAD
                angle_B_main = main_angles[1]  # LAD-LCx
                angle_A_main = main_angles[2]  # parent-LCx

                # LCx-Ramus plane (for B1)
                lcx_ramus_points = lcx_points + ramus_points
                plane_normal_cr, _ = fit_bifurcation_plane(lcx_ramus_points, spacing_info)

                # Calculate B1 angle (LCx-Ramus) directly
                lcx_array = np.array(lcx_points) * np.array(spacing_info)
                ramus_array = np.array(ramus_points) * np.array(spacing_info)

                lcx_dir = lcx_array[-1] - lcx_array[0]
                lcx_dir = lcx_dir / np.linalg.norm(lcx_dir)

                ramus_dir = ramus_array[-1] - ramus_array[0]
                ramus_dir = ramus_dir / np.linalg.norm(ramus_dir)

                # Project onto LCx-Ramus plane
                lcx_proj = lcx_dir - np.dot(lcx_dir, plane_normal_cr) * plane_normal_cr
                lcx_proj = lcx_proj / np.linalg.norm(lcx_proj)

                ramus_proj = ramus_dir - np.dot(ramus_dir, plane_normal_cr) * plane_normal_cr
                ramus_proj = ramus_proj / np.linalg.norm(ramus_proj)

                cos_B1 = np.dot(lcx_proj, ramus_proj)
                angle_B1 = np.degrees(np.arccos(np.clip(cos_B1, -1.0, 1.0)))

                # LAD-Ramus plane (for B2)
                lad_ramus_points = lad_points + ramus_points
                plane_normal_lr, _ = fit_bifurcation_plane(lad_ramus_points, spacing_info)

                # Calculate B2 angle (LAD-Ramus) directly
                lad_array = np.array(lad_points) * np.array(spacing_info)

                lad_dir = lad_array[-1] - lad_array[0]
                lad_dir = lad_dir / np.linalg.norm(lad_dir)

                # Project onto LAD-Ramus plane
                lad_proj = lad_dir - np.dot(lad_dir, plane_normal_lr) * plane_normal_lr
                lad_proj = lad_proj / np.linalg.norm(lad_proj)

                ramus_proj_lr = ramus_dir - np.dot(ramus_dir, plane_normal_lr) * plane_normal_lr
                ramus_proj_lr = ramus_proj_lr / np.linalg.norm(ramus_proj_lr)

                cos_B2 = np.dot(lad_proj, ramus_proj_lr)
                angle_B2 = np.degrees(np.arccos(np.clip(cos_B2, -1.0, 1.0)))

                depth_measurements.append({
                    'depth': depth,
                    'inflow_angle': inflow_angle,
                    'angle_A_main': angle_A_main,
                    'angle_B_main': angle_B_main,
                    'angle_C_main': angle_C_main,
                    'angle_B1': angle_B1,
                    'angle_B2': angle_B2
                })

            except Exception as e:
                continue

        if not depth_measurements:
            return {}

        # Average angles
        angles_A = [m['angle_A_main'] for m in depth_measurements]
        angles_B = [m['angle_B_main'] for m in depth_measurements]
        angles_C = [m['angle_C_main'] for m in depth_measurements]
        inflow_angles = [m['inflow_angle'] for m in depth_measurements]
        angles_B1 = [m['angle_B1'] for m in depth_measurements]
        angles_B2 = [m['angle_B2'] for m in depth_measurements]

        # Extract diameters
        parent_diameter = graph[parent_edge[0]][parent_edge[1]].get(diameter_attr)
        lad_diameter = graph[lad_edge[0]][lad_edge[1]].get(diameter_attr)
        lcx_diameter = graph[lcx_edge[0]][lcx_edge[1]].get(diameter_attr)
        ramus_diameter = graph[ramus_edge[0]][ramus_edge[1]].get(diameter_attr)

        result = {
            'LCA_TRIFURCATION': {
                'type': 'pseudo',
                'trifurcation_nodes': nodes,
                'midpoint': midpoint_voxel,
                'branches': ['LAD', 'LCx', 'Ramus'],
                'main_plane_angles': {
                    'averaged_angle_A_main': np.mean(angles_A),
                    'averaged_angle_B_main': np.mean(angles_B),
                    'averaged_angle_C_main': np.mean(angles_C),
                    'averaged_inflow_angle': np.mean(inflow_angles)
                },
                'additional_angles': {
                    'averaged_angle_B1': np.mean(angles_B1),
                    'averaged_angle_B2': np.mean(angles_B2)
                },
                'diameters': {
                    'parent': parent_diameter,
                    'LAD': lad_diameter,
                    'LCx': lcx_diameter,
                    'Ramus': ramus_diameter
                },
                'std_angles': {
                    'std_inflow_angle': np.std(inflow_angles),
                    'std_angle_A_main': np.std(angles_A),
                    'std_angle_B_main': np.std(angles_B),
                    'std_angle_C_main': np.std(angles_C),
                    'std_angle_B1': np.std(angles_B1),
                    'std_angle_B2': np.std(angles_B2)
                },
                'num_measurements': len(depth_measurements)
            }
        }

    return result


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
