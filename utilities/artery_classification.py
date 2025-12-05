import numpy as np
import networkx as nx


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
    # Count bifurcations and endpoints
    num_bifurcations = sum(1 for node in graph.nodes() if graph.out_degree(node) == 2)
    num_endpoints = sum(1 for node in graph.nodes() if graph.out_degree(node) == 0)
    num_edges = graph.number_of_edges()
    num_nodes = graph.number_of_nodes()

    # Calculate total path length
    total_path_length = 0
    for _, _, edge_data in graph.edges(data=True):
        total_path_length += edge_data.get('path_length_mm', 0)

    # Calculate maximum path depth (longest path from origin to any endpoint)
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

    # Calculate average branching factor
    if num_bifurcations > 0:
        avg_branching_factor = num_endpoints / num_bifurcations
    else:
        avg_branching_factor = 0

    # Compute overall complexity score
    # Normalize each metric and weight them
    # This is a heuristic that can be tuned based on your data
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
            - confidence: Classification confidence ('high', 'medium', 'low')
    """
    if len(graphs) != 2:
        raise ValueError(f"Expected 2 graphs, got {len(graphs)}")

    if verbose:
        print(f"\n[Classification] Analyzing vessel complexity to differentiate LCA/RCA...")

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

    if relative_diff > 0.3:
        confidence = 'high'
    elif relative_diff > 0.15:
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
        'confidence': confidence
    }
