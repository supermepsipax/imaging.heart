import numpy as np
import networkx as nx
import re


def compute_direction_vector(start_node, end_node, spacing_info):
    """
    Computes a direction vector from start to end node in physical coordinates.

    Args:
        start_node (tuple): Starting node (x, y, z) in voxel coordinates
        end_node (tuple): Ending node (x, y, z) in voxel coordinates
        spacing_info (tuple): Voxel spacing (z, y, x) in mm

    Returns:
        numpy.ndarray: Normalized direction vector in physical space
    """
    start = np.array(start_node) * np.array(spacing_info)
    end = np.array(end_node) * np.array(spacing_info)

    direction = end - start
    norm = np.linalg.norm(direction)

    if norm < 1e-6:
        return np.array([0, 0, 0])

    return direction / norm


def find_main_trunk_endpoint(graph, trunk_label):
    """
    Finds the endpoint of the main trunk for a given branch.

    Finds all edges starting with trunk_label and returns the endpoint
    of the longest one (furthest from origin).

    Args:
        graph (nx.DiGraph): Graph with edge_position labels
        trunk_label (str): Edge position label prefix (e.g., "11", "111", "122")

    Returns:
        tuple: Endpoint node coordinates, or None if not found
    """
    candidate_edges = []

    non_repeating = trunk_label[:-1]
    repeating = trunk_label[-1]
    regex = re.compile(rf"^{non_repeating}{repeating}*$")

    for edge in graph.edges():
        edge_position = graph.edges[edge].get('edge_position', '')
        if regex.match(edge_position):
            candidate_edges.append((edge, edge_position))

        # if edge_pos and edge_pos.startswith(trunk_label):
        #     # Check if it's a main continuation (not a side branch)
        #     # Main continuation: all digits after trunk_label are the same as the last digit
        #     if len(edge_pos) > len(trunk_label):
        #         # Get the last digit of trunk_label
        #         last_digit = trunk_label[-1]
        #         # Check if remaining digits are all the same as last_digit
        #         remaining = edge_pos[len(trunk_label):]
        #         if all(c == last_digit for c in remaining):
        #             candidate_edges.append((edge, len(edge_pos)))
        #     else:
        #         # This is exactly the trunk_label (no continuation)
        #         candidate_edges.append((edge, len(edge_pos)))

    if not candidate_edges:
        return None

    longest_edge = max(candidate_edges, key=lambda x: len(x[1]))[0]
    return longest_edge[1]


def detect_lca_trifurcation(graph, trifurcation_threshold_mm=5.0):
    """
    Detects if LCA has an anatomical trifurcation by checking for either:
    1. True trifurcation: Single node branches into 3 edges ("11", "12", "13")
    2. Pseudo-trifurcation: Edge "11" or "12" is very short and quickly bifurcates
       (indicating Left Main splits into 3 branches nearly simultaneously)

    In the graph structure, a trifurcation can appear as:
    - True: Edge "1" → edges "11", "12", "13" (all from same node)
    - Pseudo: Edge "1" → short "11" → "111", "112" + edge "12" (if "11" is short)
    - Pseudo: Edge "1" → edge "11" + short "12" → "122", "123" (if "12" is short)

    Args:
        graph (nx.DiGraph): LCA graph with edge_position labels
        trifurcation_threshold_mm (float): Max length of edge to be considered pseudo-trifurcation

    Returns:
        dict: {
            'is_trifurcation': bool,
            'trifurcation_type': 'true' | 'pseudo' | None,
            'left_main_edge': edge with label "1",
            'short_segment': the short edge ("11" or "12") for pseudo-trifurcation,
            'short_segment_label': "11" or "12" for pseudo-trifurcation,
            'primary_branches': list of 3 edge_position labels representing main branches
        }
    """
    left_main_edge = None
    for edge in graph.edges():
        if graph.edges[edge].get('edge_position') == '1':
            left_main_edge = edge
            break

    if left_main_edge is None:
        return {'is_trifurcation': False, 'trifurcation_type': None, 'left_main_edge': None, 'primary_branches': []}

    # Find edges "11", "12", and "13"
    edge_11 = None
    edge_11_length = None
    edge_12 = None
    edge_12_length = None
    edge_13 = None

    for edge in graph.edges():
        edge_pos = graph.edges[edge].get('edge_position')
        if edge_pos == '11':
            edge_11 = edge
            edge_11_length = graph.edges[edge].get('path_length_mm', 0)
        elif edge_pos == '12':
            edge_12 = edge
            edge_12_length = graph.edges[edge].get('path_length_mm', 0)
        elif edge_pos == '13':
            edge_13 = edge

    # Check for TRUE trifurcation: all three edges "11", "12", "13" exist
    if edge_11 is not None and edge_12 is not None and edge_13 is not None:
        return {
            'is_trifurcation': True,
            'trifurcation_type': 'true',
            'left_main_edge': left_main_edge,
            'short_segment': None,
            'short_segment_label': None,
            'short_segment_length_mm': None,
            'primary_branches': ['11', '12', '13']
        }

    # Check for PSEUDO-trifurcation: "11" is short and bifurcates → pattern "111", "112", "12"
    if edge_11 is not None and edge_11_length < trifurcation_threshold_mm:
        has_111 = any(graph.edges[e].get('edge_position') == '111' for e in graph.edges())
        has_112 = any(graph.edges[e].get('edge_position') == '112' for e in graph.edges())
        has_12 = edge_12 is not None

        if has_111 and has_112 and has_12:
            return {
                'is_trifurcation': True,
                'trifurcation_type': 'pseudo',
                'left_main_edge': left_main_edge,
                'short_segment': edge_11,
                'short_segment_label': '11',
                'short_segment_length_mm': edge_11_length,
                'primary_branches': ['111', '112', '12']
            }

    # Check for PSEUDO-trifurcation: "12" is short and bifurcates → pattern "11", "122", "123"
    if edge_12 is not None and edge_12_length < trifurcation_threshold_mm:
        has_11 = edge_11 is not None
        has_122 = any(graph.edges[e].get('edge_position') == '122' for e in graph.edges())
        has_123 = any(graph.edges[e].get('edge_position') == '123' for e in graph.edges())

        if has_11 and has_122 and has_123:
            return {
                'is_trifurcation': True,
                'trifurcation_type': 'pseudo',
                'left_main_edge': left_main_edge,
                'short_segment': edge_12,
                'short_segment_label': '12',
                'short_segment_length_mm': edge_12_length,
                'primary_branches': ['11', '122', '123']
            }

    primary_branches = []
    if edge_11 is not None:
        primary_branches.append('11')
    if edge_12 is not None:
        primary_branches.append('12')

    return {
        'is_trifurcation': False,
        'trifurcation_type': None,
        'left_main_edge': left_main_edge,
        'primary_branches': primary_branches
    }


def find_middle_branch_in_plane_projection(branch_directions, branch_labels, plane_normal_axis):
    """
    Projects branches onto a plane and finds which is angularly in the middle.

    Args:
        branch_directions (dict): Maps branch label to 3D direction vector
        branch_labels (list): List of 3 branch labels
        plane_normal_axis (int): Axis perpendicular to projection plane (0=X, 1=Y, 2=Z)

    Returns:
        str: Label of the middle branch in this projection
    """
    # Project onto plane by zeroing out the normal axis component
    projected = {}
    for label, direction in branch_directions.items():
        proj = direction.copy()
        proj[plane_normal_axis] = 0  # Zero out the component perpendicular to plane

        norm = np.linalg.norm(proj)
        if norm > 1e-6:
            proj = proj / norm
        else:
            proj = np.array([0, 0, 0])

        projected[label] = proj

    # Compute 2D angles in the plane
    # Use first non-zero axis as reference
    axes_in_plane = [i for i in range(3) if i != plane_normal_axis]
    x_axis_idx = axes_in_plane[0]
    y_axis_idx = axes_in_plane[1]

    angles = {}
    for label, proj in projected.items():
        if np.linalg.norm(proj) < 1e-6:
            angles[label] = 0.0
        else:
            x_comp = proj[x_axis_idx]
            y_comp = proj[y_axis_idx]
            angle_rad = np.arctan2(y_comp, x_comp)
            angle_deg = np.degrees(angle_rad)
            if angle_deg < 0:
                angle_deg += 360
            angles[label] = angle_deg

    # Sort by angle and return middle one
    sorted_branches = sorted(angles.items(), key=lambda x: x[1])
    return sorted_branches[1][0]  # Middle branch


def identify_central_branch_for_ramus(graph, branch_labels, bifurcation_node, spacing_info):
    """
    Identifies which of three branches is the most geometrically central (Ramus).

    Uses a multi-plane voting approach:
    1. Projects branches onto 3 orthogonal planes (XY, XZ, YZ)
    2. Finds the middle branch in each projection
    3. Branch with most votes (2+) is the Ramus
    4. If no consensus, falls back to dot product with input vessel

    Args:
        graph (nx.DiGraph): Graph with edge_position labels
        branch_labels (list): List of 3 edge_position labels (e.g., ['111', '112', '12'])
        bifurcation_node (tuple): The node where these branches originate
        spacing_info (tuple): Voxel spacing (z, y, x) in mm

    Returns:
        str: edge_position label of the central branch (Ramus)
    """
    if len(branch_labels) != 3:
        return None

    origin_node = next(n for n, d in graph.in_degree() if d == 0)

    # Get immediate endpoints for each branch (initial segment only)
    # This captures the branching angle at the trifurcation point
    endpoints = {}
    for label in branch_labels:
        # Use immediate endpoint of the first edge with this label
        # This represents the initial segment direction, not the entire trunk
        endpoint = None
        for edge in graph.edges():
            if graph.edges[edge].get('edge_position') == label:
                endpoint = edge[1]
                break
        if endpoint is not None:
            endpoints[label] = endpoint

    if len(endpoints) != 3:
        print(f"      [WARNING] Could not find all 3 branch endpoints for Ramus detection")
        return branch_labels[0]

    # Compute direction vectors
    input_direction = compute_direction_vector(origin_node, bifurcation_node, spacing_info)
    branch_directions = {}
    for label, endpoint in endpoints.items():
        direction = compute_direction_vector(bifurcation_node, endpoint, spacing_info)
        branch_directions[label] = direction

    # Vote across 3 orthogonal plane projections
    plane_names = ['YZ (perp to X)', 'XZ (perp to Y)', 'XY (perp to Z)']
    votes = {label: 0 for label in branch_labels}

    print(f"               Ramus detection (multi-plane voting):")

    for plane_axis in range(3):  # 0=X, 1=Y, 2=Z
        middle_branch = find_middle_branch_in_plane_projection(
            branch_directions, branch_labels, plane_axis
        )
        votes[middle_branch] += 1
        print(f"                 Plane {plane_names[plane_axis]}: '{middle_branch}' is middle")

    # Find branch with most votes
    sorted_by_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
    winner = sorted_by_votes[0]

    print(f"                 Votes: {dict(votes)}")

    # If clear winner (2+ votes), use it
    if winner[1] >= 2:
        central_branch = winner[0]
        print(f"                 → RAMUS: '{central_branch}' (won {winner[1]}/3 votes)")
        return central_branch

    # No consensus - fall back to complexity-based method
    # Ramus is typically the shortest/least complex intermediate branch
    print(f"                 No consensus (tie votes) - using fallback method")
    print(f"                 Fallback: shortest/least complex branch is likely Ramus")

    # Calculate total branch length for each branch (including all descendants)
    def get_longest_path_length(branch_label):
        """Find the longest path through all edges starting with this label."""
        # Find all edges that start with this label
        descendant_edges = []
        for edge in graph.edges():
            edge_pos = graph.edges[edge].get('edge_position', '')
            if edge_pos.startswith(branch_label):
                descendant_edges.append((edge, edge_pos))

        if not descendant_edges:
            return 0.0

        # Build subgraph of this branch
        branch_graph = nx.DiGraph()
        for edge, edge_pos in descendant_edges:
            length = graph.edges[edge].get('path_length_mm', 0)
            branch_graph.add_edge(edge[0], edge[1], length=length)

        # Find all leaf nodes (endpoints)
        leaf_nodes = [n for n in branch_graph.nodes() if branch_graph.out_degree(n) == 0]

        if not leaf_nodes:
            # No leaves found, just sum all edges
            return sum(graph.edges[e].get('path_length_mm', 0) for e, _ in descendant_edges)

        # Find the root node of this branch (node with in-degree 0)
        root_nodes = [n for n in branch_graph.nodes() if branch_graph.in_degree(n) == 0]

        if not root_nodes:
            # No root found, just sum all edges
            return sum(graph.edges[e].get('path_length_mm', 0) for e, _ in descendant_edges)

        # Use the first root node (should only be one in a tree structure)
        root_node = root_nodes[0]

        # Find longest path from root to any leaf
        max_path_length = 0
        for leaf in leaf_nodes:
            try:
                path = nx.shortest_path(branch_graph, root_node, leaf)
                path_length = sum(
                    branch_graph.edges[path[i], path[i+1]]['length']
                    for i in range(len(path) - 1)
                )
                max_path_length = max(max_path_length, path_length)
            except nx.NetworkXNoPath:
                continue

        return max_path_length

    # Get branch metrics
    branch_metrics = {}
    for label in branch_labels:
        total_length = get_longest_path_length(label)
        dot_product = np.dot(input_direction, branch_directions[label])

        # Complexity score: longer branches with higher alignment are more complex
        complexity = total_length * (1.0 + dot_product)

        branch_metrics[label] = {
            'total_length': total_length,
            'dot_product': dot_product,
            'complexity': complexity
        }
        print(f"                   '{label}': length={total_length:.1f}mm, dot={dot_product:.3f}, score={complexity:.2f}")

    # Ramus = least complex (shortest total path, least aligned)
    central_branch = min(branch_metrics.items(), key=lambda x: x[1]['complexity'])[0]

    print(f"                 → RAMUS: '{central_branch}' (fallback: least complex)")
    return central_branch

    # ============================================================================
    # ORIGINAL IMPLEMENTATION (commented out)
    # ============================================================================
    # This approach used dot product with input vessel direction, but failed when
    # the input vessel was highly aligned with one of the branches (e.g., LAD)
    # ============================================================================
    # if len(branch_labels) != 3:
    #     return None
    #
    # origin_node = next(n for n, d in graph.in_degree() if d == 0)
    #
    # endpoints = {}
    # for label in branch_labels:
    #     endpoint = find_main_trunk_endpoint(graph, label)
    #     if endpoint is None:
    #         for edge in graph.edges():
    #             if graph.edges[edge].get('edge_position') == label:
    #                 endpoint = edge[1]
    #                 break
    #     if endpoint is not None:
    #         endpoints[label] = endpoint
    #
    # if len(endpoints) != 3:
    #     return branch_labels[0]
    #
    # directions = {}
    # directions['origin'] = compute_direction_vector(origin_node, bifurcation_node, spacing_info)
    # for label, endpoint in endpoints.items():
    #     direction = compute_direction_vector(bifurcation_node, endpoint, spacing_info)
    #     directions[label] = direction
    #
    #
    # highest_dot_product = 0
    # central_branch = None
    #
    # for label in branch_labels:
    #     dot_product = np.dot(directions['origin'], directions[label])
    #     if dot_product > highest_dot_product:
    #         highest_dot_product = dot_product
    #         central_branch = label
    #
    # return central_branch if central_branch is not None else branch_labels[0]


# def compute_branch_complexity_from_label(graph, edge_position_label):
#     """
#     Computes complexity metrics for all edges descending from a given edge position label.
#
#     Args:
#         graph (nx.DiGraph): Graph with edge_position labels
#         edge_position_label (str): Edge position label (e.g., "11", "12", "111")
#
#     Returns:
#         dict: Complexity metrics (num_endpoints, total_length, num_edges, etc.)
#     """
#     # Find all edges that start with this label (descendants)
#     descendant_edges = []
#
#     for edge in graph.edges():
#         edge_label = graph.edges[edge].get('edge_position', '')
#         if edge_label.startswith(edge_position_label):
#             descendant_edges.append(edge)
#
#     # Compute metrics
#     num_edges = len(descendant_edges)
#     total_length = sum(graph.edges[e].get('path_length_mm', 0) for e in descendant_edges)
#
#     # Count endpoints (edges with no descendants)
#     num_endpoints = 0
#     for edge in descendant_edges:
#         edge_label = graph.edges[edge].get('edge_position', '')
#         # Check if any other edge starts with this label + one more digit
#         has_children = any(
#             graph.edges[e].get('edge_position', '').startswith(edge_label) and
#             len(graph.edges[e].get('edge_position', '')) > len(edge_label)
#             for e in graph.edges()
#         )
#         if not has_children:
#             num_endpoints += 1
#
#     # Count bifurcations (nodes where this branch splits)
#     bifurcation_count = 0
#     visited_nodes = set()
#
#     for edge in descendant_edges:
#         end_node = edge[1]
#         if end_node in visited_nodes:
#             continue
#         visited_nodes.add(end_node)
#
#         # Check if this node has 2+ outgoing edges in our branch
#         out_edges_in_branch = [
#             e for e in graph.out_edges(end_node)
#             if graph.edges[e].get('edge_position', '').startswith(edge_position_label)
#         ]
#         if len(out_edges_in_branch) >= 2:
#             bifurcation_count += 1
#
#     complexity_score = (
#         bifurcation_count * 3.0 +
#         num_endpoints * 2.0 +
#         num_edges * 1.0 +
#         (total_length / 100) * 1.0
#     )
#
#     return {
#         'num_edges': num_edges,
#         'num_endpoints': num_endpoints,
#         'num_bifurcations': bifurcation_count,
#         'total_path_length_mm': total_length,
#         'complexity_score': complexity_score
#     }
#
#
# def label_lca_branches(graph, trifurcation_threshold_mm=5.0):
#     """
#     Labels LCA branches as LAD, LCx, and optionally Ramus based on existing edge_position labels.
#
#     Strategy:
#     - Bifurcation (normal):
#       - LAD: Edge "11" (distal/main continuation, typically more complex)
#       - LCx: Edge "12" (side branch)
#
#     - Trifurcation (if edge "11" is very short and splits quickly):
#       - Identifies "111", "112", "12" as the three main branches
#       - LAD: Most complex of the three
#       - LCx: Least complex / sharpest angle
#       - Ramus: Intermediate
#
#     Args:
#         graph (nx.DiGraph): LCA graph with edge_position labels
#         trifurcation_threshold_mm (float): Max length of "11" to consider trifurcation
#
#     Returns:
#         dict: {
#             'type': 'bifurcation' | 'trifurcation',
#             'labels': {'LAD': edge_position, 'LCx': edge_position, 'Ramus': edge_position (if trifurcation)},
#             'metrics': dict of complexity metrics for each branch
#         }
#     """
#     trifurcation_info = detect_lca_trifurcation(graph, trifurcation_threshold_mm)
#
#     if not trifurcation_info['primary_branches']:
#         return {'type': 'unknown', 'labels': {}, 'metrics': {}}
#
#     primary_branches = trifurcation_info['primary_branches']
#
#     branch_metrics = {}
#     for branch_label in primary_branches:
#         metrics = compute_branch_complexity_from_label(graph, branch_label)
#         branch_metrics[branch_label] = metrics
#
#     labels = {}
#
#     if trifurcation_info['is_trifurcation']:
#         # Trifurcation: classify "111", "112", "12"
#         # LAD = most complex
#         # LCx = least complex (typically wraps around, shorter)
#         # Ramus = intermediate
#
#         sorted_by_complexity = sorted(
#             branch_metrics.items(),
#             key=lambda x: x[1]['complexity_score'],
#             reverse=True
#         )
#
#         labels['LAD'] = sorted_by_complexity[0][0]     # Most complex
#         labels['Ramus'] = sorted_by_complexity[1][0]   # Intermediate
#         labels['LCx'] = sorted_by_complexity[2][0]     # Least complex
#
#         return {
#             'type': 'trifurcation',
#             'short_segment_length_mm': trifurcation_info['short_segment_length_mm'],
#             'labels': labels,
#             'metrics': branch_metrics
#         }
#     else:
#         # Bifurcation: "11" is LAD (distal/main), "12" is LCx (side branch)
#         # This is already determined by the edge labeling algorithm
#         # But we verify using complexity
#
#         if '11' in branch_metrics and '12' in branch_metrics:
#             # Typically "11" (distal) is LAD and "12" (side) is LCx
#             # But verify with complexity - LAD should be more complex
#             if branch_metrics['11']['complexity_score'] >= branch_metrics['12']['complexity_score']:
#                 labels['LAD'] = '11'
#                 labels['LCx'] = '12'
#             else:
#                 # Unusual case: side branch is more complex
#                 # This might indicate a dominant LCx system
#                 labels['LAD'] = '12'
#                 labels['LCx'] = '11'
#                 print("[WARNING] Unusual LCA pattern: side branch ('12') is more complex than distal ('11')")
#                 print("          This may indicate a left-dominant system or misclassification")
#
#         return {
#             'type': 'bifurcation',
#             'labels': labels,
#             'metrics': branch_metrics
#         }


def is_side_branch(edge_position, parent_edge_position):
    """
    Determines if an edge is a side branch based on edge position labels.

    Side branches have a digit increase at the end (e.g., 11→12, 111→112, 1111→1112).
    Distal continuations keep adding the same digit (e.g., 11→111, 12→122).

    Args:
        edge_position (str): Edge position label
        parent_edge_position (str): Parent edge position label

    Returns:
        bool: True if this is a side branch
    """
    if not edge_position.startswith(parent_edge_position):
        return False

    if len(edge_position) != len(parent_edge_position) + 1:
        return False

    last_digit_parent = parent_edge_position[-1] if parent_edge_position else '0'
    last_digit_child = edge_position[-1]

    # Side branch: last digit increases (e.g., 1→2, 11→12)
    # Distal: last digit stays same (e.g., 1→11, 11→111)
    return last_digit_child > last_digit_parent


def annotate_lca_graph_with_branch_labels(graph, spacing_info, trifurcation_threshold_mm=5.0):
    """
    Annotates an LCA graph by adding 'lca_branch' attribute to all edges.

    Uses spatial validation to ensure correct LAD/LCx labeling:
    - LAD should go more anteriorly (decreasing axis 1 direction)
    - LCx should go more posteriorly/laterally

    For trifurcation, identifies Ramus as the most geometrically central branch.

    Main branches: 'LAD', 'LCx', 'Ramus' (if trifurcation), 'Left_Main'
    Side branches off LAD: 'D1', 'D2', 'D3', ... (Diagonal branches)
    Side branches off LCx: 'OM1', 'OM2', 'OM3', ... (Obtuse Marginal branches)
    Side branches off Ramus: 'R1', 'R2', 'R3', ... (Ramus branches)

    Args:
        graph (nx.DiGraph): LCA directed graph with edge_position labels
        spacing_info (tuple): Voxel spacing (z, y, x) in mm
        trifurcation_threshold_mm (float): Threshold for detecting trifurcation

    Returns:
        nx.DiGraph: Updated graph with 'lca_branch' attributes
    """
    updated_graph = nx.DiGraph(graph)

    # Step 1: Detect trifurcation
    trifurcation_info = detect_lca_trifurcation(updated_graph, trifurcation_threshold_mm)

    if not trifurcation_info['primary_branches']:
        print("[WARNING] Unable to detect LCA branch pattern")
        return updated_graph

    # Step 2: Find bifurcation node (where "11" and "12" originate)
    bifurcation_node = None
    for edge in updated_graph.edges():
        edge_pos = updated_graph.edges[edge].get('edge_position', '')
        if edge_pos == '11' or edge_pos == '12':
            bifurcation_node = edge[0]
            break

    if bifurcation_node is None:
        print("[WARNING] Could not find bifurcation node")
        return updated_graph

    # Step 3: Handle trifurcation or bifurcation with spatial validation
    main_branch_labels = {}  # Will map anatomical name to edge_position

    if trifurcation_info['is_trifurcation']:
        trifurcation_type = trifurcation_info['trifurcation_type']

        if trifurcation_type == 'true':
            print(f"[LCA Labeling] TRUE Trifurcation detected")
            print(f"               Three branches originate from same node: {trifurcation_info['primary_branches']}")
        else:  # pseudo
            print(f"[LCA Labeling] PSEUDO-Trifurcation detected")
            print(f"               Short segment '{trifurcation_info['short_segment_label']}': {trifurcation_info['short_segment_length_mm']:.1f}mm")

        primary_branches = trifurcation_info['primary_branches']
        ramus_current_label = identify_central_branch_for_ramus(updated_graph, primary_branches, bifurcation_node, spacing_info)

        print(f"               Ramus identified: '{ramus_current_label}' (most central branch)")

        other_branches = [b for b in primary_branches if b != ramus_current_label]

        endpoint_1 = find_main_trunk_endpoint(updated_graph, other_branches[0])
        endpoint_2 = find_main_trunk_endpoint(updated_graph, other_branches[1])

        direction_1 = compute_direction_vector(bifurcation_node, endpoint_1, spacing_info)
        direction_2 = compute_direction_vector(bifurcation_node, endpoint_2, spacing_info)

        anterior_component_1 = -direction_1[1]
        anterior_component_2 = -direction_2[1]

        if anterior_component_1 > anterior_component_2:
            lad_current_label = other_branches[0]
            lcx_current_label = other_branches[1]
        else:
            lad_current_label = other_branches[1]
            lcx_current_label = other_branches[0]

        print(f"               LAD: '{lad_current_label}' (more anterior)")
        print(f"               LCx: '{lcx_current_label}' (more posterior)")
        print(f"               Ramus: '{ramus_current_label}' (most central)")

        # Set main branch labels based on spatial validation
        # Keep original edge_position labels, just assign anatomical names
        main_branch_labels = {
            'LAD': lad_current_label,
            'LCx': lcx_current_label,
            'Ramus': ramus_current_label
        }
        print(f"               [OK] Anatomical labels assigned based on spatial validation")

    else:
        print(f"[LCA Labeling] Bifurcation detected")

        endpoint_11 = find_main_trunk_endpoint(updated_graph, "11")
        endpoint_12 = find_main_trunk_endpoint(updated_graph, "12")

        if endpoint_11 is None or endpoint_12 is None:
            print("[WARNING] Could not find branch endpoints for spatial validation")
            main_branch_labels = {'LAD': '11', 'LCx': '12'}
        else:
            direction_11 = compute_direction_vector(bifurcation_node, endpoint_11, spacing_info)
            direction_12 = compute_direction_vector(bifurcation_node, endpoint_12, spacing_info)

            anterior_component_11 = -direction_11[1]
            anterior_component_12 = -direction_12[1]

            branch_11_is_more_anterior = anterior_component_11 > anterior_component_12

            if branch_11_is_more_anterior:
                main_branch_labels = {'LAD': '11', 'LCx': '12'}
                print(f"               '11' is LAD (axis1={direction_11[1]:.3f}, more anterior)")
                print(f"               '12' is LCx (axis1={direction_12[1]:.3f}, more posterior)")
            else:
                main_branch_labels = {'LAD': '12', 'LCx': '11'}
                print(f"               '12' is LAD (axis1={direction_12[1]:.3f}, more anterior)")
                print(f"               '11' is LCx (axis1={direction_11[1]:.3f}, more posterior)")
                print(f"               [NOTE] Unusual anatomy: '12' branch is LAD")


    labeling_result = {
        'type': 'trifurcation' if trifurcation_info['is_trifurcation'] else 'bifurcation',
        'labels': main_branch_labels,
        'spatial_validation': True  # Mark that we used spatial validation
    }

    if trifurcation_info['is_trifurcation']:
        labeling_result['trifurcation_type'] = trifurcation_info['trifurcation_type']
        if trifurcation_info['trifurcation_type'] == 'pseudo':
            labeling_result['short_segment_length_mm'] = trifurcation_info['short_segment_length_mm']
            labeling_result['short_segment_label'] = trifurcation_info['short_segment_label']

    for edge in updated_graph.edges():
        edge_pos = updated_graph.edges[edge].get('edge_position', '')
        if edge_pos == '1':
            updated_graph.edges[edge]['lca_branch'] = 'Left_Main'

    side_branch_counters = {'LAD': 0, 'LCx': 0, 'Ramus': 0}

    all_edges_with_pos = [
        (edge, updated_graph.edges[edge].get('edge_position', ''))
        for edge in updated_graph.edges()
    ]
    # Sort by edge position length and value
    all_edges_with_pos.sort(key=lambda x: (len(x[1]), x[1]))

    # First pass: label main branch trunks
    for edge, edge_pos in all_edges_with_pos:
        if edge_pos == '1':
            continue  # Already labeled as Left_Main

        # Check if this edge belongs to one of the main branches
        for anatomical_name, main_edge_pos in main_branch_labels.items():
            if edge_pos.startswith(main_edge_pos):
                # This edge is part of this main branch's territory
                # But we need to determine if it's a side branch or continuation

                # Find the immediate parent label
                if len(edge_pos) > len(main_edge_pos):
                    parent_pos = edge_pos[:-1]

                    # Check if this is a side branch from parent
                    if is_side_branch(edge_pos, parent_pos):
                        # This is a side branch!
                        # Determine which main branch it comes from by checking parent's label
                        parent_anatomical = updated_graph.edges.get(
                            next((e for e in updated_graph.edges()
                                  if updated_graph.edges[e].get('edge_position') == parent_pos), None),
                            {}
                        ).get('lca_branch')

                        if parent_anatomical:
                            # Determine side branch type
                            if parent_anatomical.startswith('LAD') or parent_anatomical.startswith('D'):
                                side_branch_counters['LAD'] += 1
                                label = f"D{side_branch_counters['LAD']}"
                            elif parent_anatomical.startswith('LCx') or parent_anatomical.startswith('OM'):
                                side_branch_counters['LCx'] += 1
                                label = f"OM{side_branch_counters['LCx']}"
                            elif parent_anatomical.startswith('Ramus') or parent_anatomical.startswith('R'):
                                side_branch_counters['Ramus'] += 1
                                label = f"R{side_branch_counters['Ramus']}"
                            else:
                                label = anatomical_name  # Fallback to main branch name

                            updated_graph.edges[edge]['lca_branch'] = label
                        else:
                            # Parent not labeled yet, use main branch name
                            updated_graph.edges[edge]['lca_branch'] = anatomical_name
                    else:
                        # Distal continuation of parent
                        parent_anatomical = updated_graph.edges.get(
                            next((e for e in updated_graph.edges()
                                  if updated_graph.edges[e].get('edge_position') == parent_pos), None),
                            {}
                        ).get('lca_branch', anatomical_name)

                        updated_graph.edges[edge]['lca_branch'] = parent_anatomical
                else:
                    # This is the main branch itself
                    updated_graph.edges[edge]['lca_branch'] = anatomical_name

                break  # Found the main branch this belongs to

    # Store labeling metadata
    updated_graph.graph['lca_labeling'] = labeling_result

    # Print summary
    print(f"\n[LCA Branch Labeling] Type: {labeling_result['type'].upper()}")
    if labeling_result['type'] == 'trifurcation':
        if labeling_result.get('trifurcation_type') == 'true':
            print(f"                      Trifurcation type: TRUE (3 branches from same node)")
        elif labeling_result.get('trifurcation_type') == 'pseudo':
            print(f"                      Trifurcation type: PSEUDO")
            print(f"                      Short segment '{labeling_result['short_segment_label']}': {labeling_result['short_segment_length_mm']:.1f}mm")

    print(f"                      Main branch assignments (spatially validated):")
    for label, edge_pos in labeling_result['labels'].items():
        print(f"                        {label}: edge '{edge_pos}'")

    # Count side branches
    if side_branch_counters['LAD'] > 0:
        print(f"                      Diagonal branches (D): {side_branch_counters['LAD']}")
    if side_branch_counters['LCx'] > 0:
        print(f"                      Obtuse Marginal branches (OM): {side_branch_counters['LCx']}")
    if side_branch_counters['Ramus'] > 0:
        print(f"                      Ramus branches (R): {side_branch_counters['Ramus']}")

    return updated_graph
