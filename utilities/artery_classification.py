import numpy as np


def classify_lca_rca_from_spatial_position(body_masks, anatomical_info, verbose=True):
    """
    Classifies two vessel masks as LCA or RCA based on spatial position (left vs right).

    Uses anatomical orientation information from NRRD header to determine which vessel
    is positioned more to the left (LCA) vs right (RCA).

    This method is more reliable than complexity-based classification when anatomical
    orientation is available.

    Args:
        body_masks (list): List of 2 binary masks (numpy arrays)
        anatomical_info (dict): Anatomical orientation info from extract_anatomical_info()
        verbose (bool): If True, print classification details

    Returns:
        dict: Classification results containing:
            - lca_index: Index of the mask classified as LCA (0 or 1)
            - rca_index: Index of the mask classified as RCA (0 or 1)
            - method: 'spatial_position'
            - confidence: Always 'high' for spatial classification
            - lca_centroid: Centroid coordinates of LCA
            - rca_centroid: Centroid coordinates of RCA

        Returns None if left-right axis cannot be determined from anatomical_info
    """
    if len(body_masks) != 2:
        raise ValueError(f"Expected 2 masks, got {len(body_masks)}")

    if anatomical_info is None or 'axis_directions' not in anatomical_info:
        return None

    # Find which axis corresponds to left-right
    axis_directions = anatomical_info['axis_directions']
    left_axis = None
    left_sign = None

    for idx, direction in enumerate(axis_directions):
        direction_lower = direction.lower()
        if direction_lower == 'left':
            left_axis = idx
            left_sign = +1  # Positive values mean left
            break
        elif direction_lower == 'right':
            left_axis = idx
            left_sign = -1  # Negative values mean left (positive is right)
            break

    if left_axis is None:
        if verbose:
            print(f"[Spatial Classification] Cannot determine left-right axis from anatomical info")
        return None

    # The space string tells us which world axis is left/right, but not whether increasing
    # voxel indices along that axis go in the positive or negative world direction.
    # space_directions[left_axis][left_axis] gives the actual sign: if negative, increasing
    # voxel index moves opposite to what the space string implies, so we must flip left_sign.
    if 'space_directions' in anatomical_info:
        sd = np.array(anatomical_info['space_directions'])
        if left_axis < sd.shape[0] and left_axis < sd.shape[1]:
            diag_sign = int(np.sign(sd[left_axis, left_axis]))
            if diag_sign != 0:
                left_sign *= diag_sign

    if verbose:
        print(f"\n[Spatial Classification] Using spatial position to classify LCA/RCA")
        print(f"                         Left-right axis: {left_axis} ({'left' if left_sign > 0 else 'right'} is positive)")
        print(f"                         Coordinate system: {anatomical_info.get('space', 'unknown')}")

    # Compute centroids for both masks
    centroids = []
    for i, mask in enumerate(body_masks):
        # Get coordinates of all voxels in this mask
        coords = np.argwhere(mask > 0)

        if len(coords) == 0:
            raise ValueError(f"Mask {i} is empty")

        # Compute centroid (mean position)
        centroid = np.mean(coords, axis=0)
        centroids.append(centroid)

        if verbose:
            print(f"  Vessel {i+1} centroid: {centroid}")

    # Compare positions along left-right axis
    # Higher "left component" = more to the left = LCA
    left_component_0 = left_sign * centroids[0][left_axis]
    left_component_1 = left_sign * centroids[1][left_axis]

    if verbose:
        print(f"  Vessel 1 left-component: {left_component_0:.2f}")
        print(f"  Vessel 2 left-component: {left_component_1:.2f}")

    # Vessel with higher left component is LCA
    if left_component_0 > left_component_1:
        lca_index = 0
        rca_index = 1
        lca_centroid = centroids[0]
        rca_centroid = centroids[1]
    else:
        lca_index = 1
        rca_index = 0
        lca_centroid = centroids[1]
        rca_centroid = centroids[0]

    if verbose:
        print(f"\n  Classification result:")
        print(f"    Vessel {lca_index+1} → LCA (more to the left)")
        print(f"    Vessel {rca_index+1} → RCA (more to the right)")
        print(f"    Confidence: HIGH (spatial positioning is definitive)")

    return {
        'lca_index': lca_index,
        'rca_index': rca_index,
        'method': 'spatial_position',
        'confidence': 'high',
        'lca_centroid': lca_centroid,
        'rca_centroid': rca_centroid,
        'left_axis': left_axis,
        'left_sign': left_sign
    }


def compute_graph_complexity_metrics(graph):
    """
    Computes complexity metrics from an already-constructed vessel graph.

    The LCA typically exhibits:
    - More bifurcations (early split into LAD and LCX)
    - More endpoints (more complex branching)
    - Higher total path length
    - More edges in the graph
    - Greater vessel volume

    Args:
        graph (nx.DiGraph): Directed graph with vessel topology and metrics

    Returns:
        dict: Dictionary containing complexity metrics:
            - num_bifurcations: Number of bifurcation nodes (out_degree == 2)
            - num_endpoints: Number of endpoint nodes (out_degree == 0)
            - num_edges: Total number of edges in the graph
            - num_nodes: Total number of nodes in the graph
            - total_path_length_mm: Sum of all edge path lengths
            - max_path_depth_mm: Maximum path length from origin to any endpoint
            - avg_branching_factor: Average number of children per bifurcation
            - complexity_score: Weighted complexity score
    """
    num_bifurcations = sum(1 for node in graph.nodes() if graph.out_degree(node) == 2)
    num_endpoints = sum(1 for node in graph.nodes() if graph.out_degree(node) == 0)
    num_edges = graph.number_of_edges()
    num_nodes = graph.number_of_nodes()

    total_path_length = 0
    for _, _, edge_data in graph.edges(data=True):
        total_path_length += edge_data.get('path_length_mm', 0)

    origin_nodes = [n for n in graph.nodes() if graph.in_degree(n) == 0]
    max_path_depth = 0

    if origin_nodes:
        origin = origin_nodes[0]

        def dfs_max_depth(node, current_depth=0):
            """DFS to find maximum depth from origin."""
            out_edges = list(graph.out_edges(node, data=True))

            if len(out_edges) == 0:
                return current_depth

            max_child_depth = current_depth
            for _, next_node, edge_data in out_edges:
                edge_length = edge_data.get('path_length_mm', 0)
                child_depth = dfs_max_depth(next_node, current_depth + edge_length)
                max_child_depth = max(max_child_depth, child_depth)

            return max_child_depth

        max_path_depth = dfs_max_depth(origin)

    if num_bifurcations > 0:
        avg_branching_factor = num_endpoints / num_bifurcations
    else:
        avg_branching_factor = 0

    # Compute overall complexity score
    # Normalize each metric and weight them
    complexity_score = (
        num_bifurcations * 3.0 +        # Bifurcations are strong indicators
        num_endpoints * 2.0 +            # More endpoints = more complex
        num_edges * 1.0 +                # More edges = more complex
        (total_path_length / 100) * 1.0  # Total path length
    )

    return {
        'num_bifurcations': num_bifurcations,
        'num_endpoints': num_endpoints,
        'num_edges': num_edges,
        'num_nodes': num_nodes,
        'total_path_length_mm': total_path_length,
        'max_path_depth_mm': max_path_depth,
        'avg_branching_factor': avg_branching_factor,
        'complexity_score': complexity_score
    }


def classify_lca_rca_from_graphs(graphs, verbose=True):
    """
    Classifies two vessel graphs as LCA or RCA based on complexity metrics.

    DEPRECATED: This complexity-based approach is less reliable than spatial classification.
    Use classify_lca_rca_from_spatial_position() when anatomical orientation is available.

    The LCA (Left Coronary Artery) typically has:
    - More complex branching pattern (bifurcates into LAD and LCX)
    - More bifurcations and endpoints
    - Larger total path length

    The RCA (Right Coronary Artery) typically has:
    - Simpler, more linear structure
    - Fewer branches
    - Shorter total path length

    Args:
        graphs (list): List of 2 NetworkX DiGraphs (already processed)
        verbose (bool): If True, print classification details

    Returns:
        dict: Classification results containing:
            - lca_index: Index of the graph classified as LCA (0 or 1)
            - rca_index: Index of the graph classified as RCA (0 or 1)
            - lca_metrics: Complexity metrics for LCA
            - rca_metrics: Complexity metrics for RCA
            - method: 'complexity'
            - confidence: Classification confidence ('high', 'medium', 'low')
    """
    if len(graphs) != 2:
        raise ValueError(f"Expected 2 graphs, got {len(graphs)}")

    if verbose:
        print(f"\n[Complexity Classification] Analyzing vessel complexity to differentiate LCA/RCA...")
        print(f"                            NOTE: Spatial classification is more reliable when available")

    # Compute metrics for both graphs
    metrics_list = []
    for i, graph in enumerate(graphs):
        if verbose:
            print(f"  Computing metrics for vessel {i+1}...")
        metrics = compute_graph_complexity_metrics(graph)
        metrics_list.append(metrics)

        if verbose:
            print(f"    Bifurcations: {metrics['num_bifurcations']}")
            print(f"    Endpoints: {metrics['num_endpoints']}")
            print(f"    Edges: {metrics['num_edges']}")
            print(f"    Total path length: {metrics['total_path_length_mm']:.1f} mm")
            print(f"    Max path depth: {metrics['max_path_depth_mm']:.1f} mm")
            print(f"    Complexity score: {metrics['complexity_score']:.2f}")

    # Compare complexity scores
    score_0 = metrics_list[0]['complexity_score']
    score_1 = metrics_list[1]['complexity_score']

    # Determine confidence based on score difference
    score_diff = abs(score_0 - score_1)
    avg_score = (score_0 + score_1) / 2
    relative_diff = score_diff / avg_score if avg_score > 0 else 0

    if relative_diff > 0.5:
        confidence = 'high'
    elif relative_diff > 0.25:
        confidence = 'medium'
    else:
        confidence = 'low'

    # The graph with higher complexity score is classified as LCA
    if score_0 > score_1:
        lca_index = 0
        rca_index = 1
        lca_metrics = metrics_list[0]
        rca_metrics = metrics_list[1]
    else:
        lca_index = 1
        rca_index = 0
        lca_metrics = metrics_list[1]
        rca_metrics = metrics_list[0]

    if verbose:
        print(f"\n  Classification result:")
        print(f"    Vessel {lca_index+1} → LCA (complexity: {lca_metrics['complexity_score']:.2f})")
        print(f"    Vessel {rca_index+1} → RCA (complexity: {rca_metrics['complexity_score']:.2f})")
        print(f"    Confidence: {confidence.upper()} (relative difference: {relative_diff:.1%})")

        if confidence == 'low':
            print(f"    [WARNING] Low confidence - vessels have similar complexity")
            print(f"              Consider manual verification of classification")

    return {
        'lca_index': lca_index,
        'rca_index': rca_index,
        'lca_metrics': lca_metrics,
        'rca_metrics': rca_metrics,
        'method': 'complexity',
        'confidence': confidence
    }
