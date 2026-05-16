import numpy as np
import networkx as nx
import re


def get_anatomical_axis_info(anatomical_info, axis_name):
    """
    Gets the axis index and direction multiplier for a given anatomical axis.

    Args:
        anatomical_info (dict): Anatomical info from extract_anatomical_info()
        axis_name (str): Anatomical axis to find (e.g., 'anterior', 'superior', 'left')

    Returns:
        tuple: (axis_index, multiplier) where:
            - axis_index: Which array axis (0, 1, or 2) corresponds to this direction
            - multiplier: +1 if positive values mean this direction, -1 if negative

        Returns (None, None) if axis not found

    Example:
        For LPS coordinate system with axis_directions=['left', 'posterior', 'superior']:
        - get_anatomical_axis_info(info, 'anterior') → (1, -1)
          (axis 1, negative direction)
        - get_anatomical_axis_info(info, 'superior') → (2, +1)
          (axis 2, positive direction)
    """
    if 'axis_directions' not in anatomical_info:
        return None, None

    axis_directions = anatomical_info['axis_directions']
    axis_name_lower = axis_name.lower()

    opposites = {
        'left': 'right', 'right': 'left',
        'anterior': 'posterior', 'posterior': 'anterior',
        'superior': 'inferior', 'inferior': 'superior'
    }

    for idx, direction in enumerate(axis_directions):
        direction_lower = direction.lower()

        if direction_lower == axis_name_lower:
            multiplier = +1
        elif direction_lower == opposites.get(axis_name_lower):
            multiplier = -1
        else:
            continue

        # The space string declares the positive world direction per axis, but
        # space_directions[idx][idx] can be negative, meaning increasing voxel index
        # actually moves opposite to what the space string implies. Correct for this.
        if 'space_directions' in anatomical_info:
            sd = np.array(anatomical_info['space_directions'])
            if idx < sd.shape[0] and idx < sd.shape[1]:
                diag_sign = int(np.sign(sd[idx, idx]))
                if diag_sign != 0:
                    multiplier *= diag_sign

        return idx, multiplier

    return None, None


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


def identify_central_branch_for_ramus(graph, branch_labels, bifurcation_node, spacing_info):
    """
    Identifies which of three branches is the most geometrically central (Ramus).

    Uses centroid-based spatial analysis:
    1. Collects all voxels from each branch and its descendants
    2. Computes centroid for each branch territory in physical coordinates
    3. Finds the two branches with largest centroid distance (exterior branches = LAD & LCx)
    4. Remaining branch is the Ramus (middle branch)

    Args:
        graph (nx.DiGraph): Graph with edge_position labels
        branch_labels (list): List of 3 edge_position labels (e.g., ['111', '112', '12'])
        bifurcation_node (tuple): The node where these branches originate
        spacing_info (tuple): Voxel spacing (z, y, x) in mm

    Returns:
        str: edge_position label of the central branch (Ramus), or None if detection fails
    """
    if len(branch_labels) != 3:
        return None

    print(f"               Ramus detection (centroid-based spatial analysis):")

    def get_all_branch_voxels(branch_label):
        """Collect all voxels from this branch and all its descendants."""
        all_voxels = []

        for edge in graph.edges():
            edge_pos = graph.edges[edge].get('edge_position', '')
            if edge_pos.startswith(branch_label):
                edge_voxels = graph.edges[edge].get('voxels', [])
                all_voxels.extend(edge_voxels)

        return all_voxels

    branch_centroids = {}

    for label in branch_labels:
        voxels = get_all_branch_voxels(label)

        if len(voxels) == 0:
            print(f"                   [WARNING] No voxels found for branch '{label}'")
            print(f"                   [ERROR] Ramus detection failed - incomplete voxel data")
            return None

        voxels_array = np.array(voxels) * np.array(spacing_info)
        centroid = np.mean(voxels_array, axis=0)
        branch_centroids[label] = centroid

        print(f"                   '{label}': centroid={centroid}, {len(voxels)} voxels")

    if len(branch_centroids) != 3:
        print(f"                   [ERROR] Ramus detection failed - could not compute centroids for all branches")
        return None

    max_distance = 0
    exterior_pair = None

    labels_list = list(branch_centroids.keys())
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            label1, label2 = labels_list[i], labels_list[j]
            centroid1, centroid2 = branch_centroids[label1], branch_centroids[label2]

            distance = np.linalg.norm(centroid1 - centroid2)

            if distance > max_distance:
                max_distance = distance
                exterior_pair = (label1, label2)

    central_branch = None
    for label in branch_labels:
        if label not in exterior_pair:
            central_branch = label
            break

    if central_branch is not None:
        print(f"                   Exterior branches (furthest apart): {exterior_pair} (distance: {max_distance:.1f}mm)")
        print(f"                 → RAMUS: '{central_branch}'")
        return central_branch

    print(f"                   [ERROR] Ramus detection failed - could not identify central branch")
    return None



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


def annotate_lca_graph_with_branch_labels(graph, spacing_info, anatomical_info=None,
                                            trifurcation_threshold_mm=5.0):
    """
    Annotates an LCA graph by adding 'lca_branch' attribute to all edges.

    Uses spatial validation to ensure correct LAD/LCx labeling:
    - LAD should go more anteriorly
    - LCx should go more posteriorly/laterally

    For trifurcation, identifies Ramus as the most geometrically central branch.

    Main branches: 'LAD', 'LCx', 'Ramus' (if trifurcation), 'Left_Main'
    Side branches off LAD: 'D1', 'D2', 'D3', ... (Diagonal branches)
    Side branches off LCx: 'OM1', 'OM2', 'OM3', ... (Obtuse Marginal branches)
    Side branches off Ramus: 'R1', 'R2', 'R3', ... (Ramus branches)

    Args:
        graph (nx.DiGraph): LCA directed graph with edge_position labels
        spacing_info (tuple): Voxel spacing (z, y, x) in mm
        anatomical_info (dict, optional): Anatomical orientation info from extract_anatomical_info().
            If None, assumes LPS coordinate system (left-posterior-superior).
        trifurcation_threshold_mm (float): Threshold for detecting trifurcation

    Returns:
        nx.DiGraph: Updated graph with 'lca_branch' attributes
    """
    updated_graph = nx.DiGraph(graph)

    # Get anatomical axis information for LAD/LCx spatial validation
    # LAD runs anteriorly, LCx runs posteriorly
    if anatomical_info is not None:
        anterior_axis, anterior_sign = get_anatomical_axis_info(anatomical_info, 'anterior')
        if anterior_axis is None:
            print("[WARNING] Could not determine anterior direction from anatomical info")
            print("          Falling back to default assumption: axis 1, negative direction (LPS)")
            anterior_axis = 1
            anterior_sign = -1
        else:
            print(f"[LCA Labeling] Using anatomical info: anterior = axis {anterior_axis}, "
                  f"{'positive' if anterior_sign > 0 else 'negative'} direction")
            print(f"               Coordinate system: {anatomical_info.get('space', 'unknown')}")
    else:
        # Default assumption: LPS coordinate system
        # Axis 1 = posterior-anterior (posterior is positive, anterior is negative)
        print("[LCA Labeling] No anatomical info provided, assuming LPS coordinate system")
        print("               (axis 1 = posterior-anterior, anterior is negative)")
        anterior_axis = 1
        anterior_sign = -1

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

        # Use anatomical info to determine anterior component
        anterior_component_1 = anterior_sign * direction_1[anterior_axis]
        anterior_component_2 = anterior_sign * direction_2[anterior_axis]

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

            # Use anatomical info to determine anterior component
            anterior_component_11 = anterior_sign * direction_11[anterior_axis]
            anterior_component_12 = anterior_sign * direction_12[anterior_axis]

            branch_11_is_more_anterior = anterior_component_11 > anterior_component_12

            if branch_11_is_more_anterior:
                main_branch_labels = {'LAD': '11', 'LCx': '12'}
                print(f"               '11' is LAD (anterior_component={anterior_component_11:.3f}, more anterior)")
                print(f"               '12' is LCx (anterior_component={anterior_component_12:.3f}, more posterior)")
            else:
                main_branch_labels = {'LAD': '12', 'LCx': '11'}
                print(f"               '12' is LAD (anterior_component={anterior_component_12:.3f}, more anterior)")
                print(f"               '11' is LCx (anterior_component={anterior_component_11:.3f}, more posterior)")
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
