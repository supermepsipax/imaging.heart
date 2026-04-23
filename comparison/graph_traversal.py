"""
Graph traversal-based comparison of artery graphs.

Compares a ground truth artery graph to a model graph by simultaneous BFS
traversal from the origin. At each junction, child nodes are matched to
nearest neighbors in the model graph with parent verification, producing
branch-level topology and geometry metrics.
"""

import numpy as np
from collections import deque
from scipy.spatial.distance import cdist


def _to_physical(node, spacing_info):
    """Convert a voxel coordinate node to physical mm coordinates."""
    return np.array(node, dtype=float) * np.array(spacing_info, dtype=float)


def _find_origin(graph):
    """Find the origin node (in_degree == 0) in a directed graph."""
    origins = [n for n in graph.nodes() if graph.in_degree(n) == 0]
    if len(origins) != 1:
        raise ValueError(f"Expected 1 origin node, found {len(origins)}")
    return origins[0]


def _match_children(gt_children_mm, model_children_mm, model_children_nodes,
                    max_distance_mm):
    """
    Match GT child nodes to model child nodes by greedy nearest neighbor.

    Each model child can be matched to at most one GT child. Matches exceeding
    max_distance_mm are rejected.

    Args:
        gt_children_mm: np.ndarray of shape (N, 3) — GT children in physical coords
        model_children_mm: np.ndarray of shape (M, 3) — model children in physical coords
        model_children_nodes: list of model child node tuples (length M)
        max_distance_mm: maximum distance to consider a valid match

    Returns:
        matches: list of (gt_child_index, model_child_node, distance_mm)
        unmatched_gt: list of gt_child_index
        unmatched_model: list of model_child_node
    """
    if len(gt_children_mm) == 0 or len(model_children_mm) == 0:
        return (
            [],
            list(range(len(gt_children_mm))),
            list(model_children_nodes),
        )

    dists = cdist(gt_children_mm, model_children_mm)

    matches = []
    used_gt = set()
    used_model = set()

    # Greedy: pick smallest distance pair repeatedly
    flat_order = np.argsort(dists, axis=None)
    for flat_idx in flat_order:
        i, j = divmod(int(flat_idx), len(model_children_mm))
        if i in used_gt or j in used_model:
            continue
        if dists[i, j] > max_distance_mm:
            break
        matches.append((i, model_children_nodes[j], float(dists[i, j])))
        used_gt.add(i)
        used_model.add(j)

    unmatched_gt = [i for i in range(len(gt_children_mm)) if i not in used_gt]
    unmatched_model = [model_children_nodes[j]
                       for j in range(len(model_children_mm)) if j not in used_model]

    return matches, unmatched_gt, unmatched_model


def _count_subtree_edges(graph, root):
    """Count total edges in the subtree rooted at root (inclusive)."""
    count = 0
    queue = deque([root])
    while queue:
        node = queue.popleft()
        children = list(graph.successors(node))
        count += len(children)
        queue.extend(children)
    return count


def compare_graphs_by_traversal(gt_graph, model_graph,
                                spacing_info_gt, spacing_info_model,
                                branch_match_distance_mm=10.0):
    """
    Compare two artery graphs by simultaneous BFS traversal from origin.

    Traverses the GT graph breadth-first. At each junction, matches child nodes
    to nearest neighbors among the matched model node's children, verifying
    structural correspondence. Produces branch-level match results and summary
    metrics (recall, precision, F1, junction error, length ratio).

    Args:
        gt_graph: NetworkX DiGraph — ground truth artery graph
        model_graph: NetworkX DiGraph — model prediction graph
        spacing_info_gt: tuple (z, y, x) voxel spacing in mm for GT
        spacing_info_model: tuple (z, y, x) voxel spacing in mm for model
        branch_match_distance_mm: max distance in mm to match a downstream
            junction node (default 10.0)

    Returns:
        dict with keys:
            origin_offset_mm: float — distance between GT and model origin
            branch_results: list of per-branch dicts
            summary: dict of aggregate metrics
    """
    gt_origin = _find_origin(gt_graph)
    model_origin = _find_origin(model_graph)

    gt_origin_mm = _to_physical(gt_origin, spacing_info_gt)
    model_origin_mm = _to_physical(model_origin, spacing_info_model)

    origin_offset_mm = float(np.linalg.norm(gt_origin_mm - model_origin_mm))

    # BFS state
    node_mapping = {gt_origin: model_origin}  # gt_node -> model_node (or None)
    node_depth = {gt_origin: 0}
    queue = deque([gt_origin])

    branch_results = []
    matched_model_edges = set()

    while queue:
        gt_node = queue.popleft()
        depth = node_depth[gt_node]
        matched_model_node = node_mapping.get(gt_node)

        gt_children = list(gt_graph.successors(gt_node))
        if not gt_children:
            continue

        # If parent wasn't matched, all downstream GT branches are missed
        if matched_model_node is None:
            for gt_child in gt_children:
                edge_data = gt_graph.edges[gt_node, gt_child]
                branch_results.append({
                    'gt_edge': (gt_node, gt_child),
                    'model_edge': None,
                    'matched': False,
                    'reason': 'parent_unmatched',
                    'gt_length_mm': edge_data.get('path_length_mm', 0),
                    'depth': depth,
                })
                node_mapping[gt_child] = None
                node_depth[gt_child] = depth + 1
                queue.append(gt_child)
            continue

        model_children = list(model_graph.successors(matched_model_node))
        gt_children_mm = np.array([_to_physical(c, spacing_info_gt) for c in gt_children])

        # Model node is an endpoint but GT continues
        if not model_children:
            for gt_child in gt_children:
                edge_data = gt_graph.edges[gt_node, gt_child]
                branch_results.append({
                    'gt_edge': (gt_node, gt_child),
                    'model_edge': None,
                    'matched': False,
                    'reason': 'model_is_endpoint',
                    'gt_length_mm': edge_data.get('path_length_mm', 0),
                    'depth': depth,
                })
                node_mapping[gt_child] = None
                node_depth[gt_child] = depth + 1
                queue.append(gt_child)
            continue

        model_children_mm = np.array(
            [_to_physical(c, spacing_info_model) for c in model_children]
        )

        matches, unmatched_gt_idx, unmatched_model_nodes = _match_children(
            gt_children_mm, model_children_mm, model_children,
            branch_match_distance_mm,
        )

        # Matched branches
        for gt_idx, model_child, dist in matches:
            gt_child = gt_children[gt_idx]
            gt_edge = gt_graph.edges[gt_node, gt_child]
            model_edge = model_graph.edges[matched_model_node, model_child]

            gt_len = gt_edge.get('path_length_mm', 0)
            model_len = model_edge.get('path_length_mm', 0)

            branch_results.append({
                'gt_edge': (gt_node, gt_child),
                'model_edge': (matched_model_node, model_child),
                'matched': True,
                'junction_error_mm': dist,
                'gt_length_mm': gt_len,
                'model_length_mm': model_len,
                'length_difference_mm': model_len - gt_len,
                'length_ratio': model_len / gt_len if gt_len > 0 else float('inf'),
                'gt_edge_position': gt_edge.get('edge_position', ''),
                'model_edge_position': model_edge.get('edge_position', ''),
                'depth': depth,
            })

            matched_model_edges.add((matched_model_node, model_child))
            node_mapping[gt_child] = model_child
            node_depth[gt_child] = depth + 1
            queue.append(gt_child)

        # Missed GT branches (no match within threshold)
        for gt_idx in unmatched_gt_idx:
            gt_child = gt_children[gt_idx]
            gt_edge = gt_graph.edges[gt_node, gt_child]
            branch_results.append({
                'gt_edge': (gt_node, gt_child),
                'model_edge': None,
                'matched': False,
                'reason': 'no_match_within_threshold',
                'gt_length_mm': gt_edge.get('path_length_mm', 0),
                'gt_edge_position': gt_edge.get('edge_position', ''),
                'depth': depth,
            })
            node_mapping[gt_child] = None
            node_depth[gt_child] = depth + 1
            queue.append(gt_child)

        # Extra model branches at this junction (not in GT)
        for model_child in unmatched_model_nodes:
            model_edge = model_graph.edges[matched_model_node, model_child]
            extra_subtree_edges = _count_subtree_edges(model_graph, model_child)
            branch_results.append({
                'gt_edge': None,
                'model_edge': (matched_model_node, model_child),
                'matched': False,
                'reason': 'extra_model_branch',
                'model_length_mm': model_edge.get('path_length_mm', 0),
                'model_edge_position': model_edge.get('edge_position', ''),
                'extra_subtree_edges': extra_subtree_edges,
                'depth': depth,
            })

    # -- Bifurcation-specific metrics --
    # GT bifurcations: nodes with out_degree >= 2 (excluding origin)
    gt_bifurcations = [n for n in gt_graph.nodes()
                       if gt_graph.out_degree(n) >= 2 and gt_graph.in_degree(n) > 0]
    model_bifurcations = set(n for n in model_graph.nodes()
                             if model_graph.out_degree(n) >= 2 and model_graph.in_degree(n) > 0)

    matched_model_nodes = set(v for v in node_mapping.values() if v is not None)

    bifurcation_results = []
    for gt_bif in gt_bifurcations:
        gt_bif_mm = _to_physical(gt_bif, spacing_info_gt)
        model_match = node_mapping.get(gt_bif)
        if model_match is not None:
            model_match_mm = _to_physical(model_match, spacing_info_model)
            error = float(np.linalg.norm(gt_bif_mm - model_match_mm))
            bifurcation_results.append({
                'gt_node': gt_bif,
                'model_node': model_match,
                'matched': True,
                'position_error_mm': error,
                'depth': node_depth.get(gt_bif, -1),
            })
        else:
            bifurcation_results.append({
                'gt_node': gt_bif,
                'model_node': None,
                'matched': False,
                'position_error_mm': None,
                'depth': node_depth.get(gt_bif, -1),
            })

    matched_bifs = [b for b in bifurcation_results if b['matched']]
    total_gt_bifs = len(gt_bifurcations)
    total_model_bifs = len(model_bifurcations)
    num_matched_bifs = len(matched_bifs)
    # How many model bifurcations were matched to a GT node
    model_bifs_matched = len(matched_model_nodes & model_bifurcations)

    bif_accuracy = num_matched_bifs / total_gt_bifs if total_gt_bifs > 0 else 0.0
    bif_recall = model_bifs_matched / total_model_bifs if total_model_bifs > 0 else 0.0
    bif_errors = [b['position_error_mm'] for b in matched_bifs]

    # First bifurcation: the first GT bifurcation reached in BFS (shallowest depth)
    first_bif_error_mm = None
    if bifurcation_results:
        by_depth = sorted(bifurcation_results, key=lambda b: b['depth'])
        first_bif = by_depth[0]
        if first_bif['matched']:
            first_bif_error_mm = first_bif['position_error_mm']

    summary = {
        'origin_offset_mm': origin_offset_mm,
        'first_bifurcation_error_mm': first_bif_error_mm,
        'bifurcation_accuracy': float(bif_accuracy),
        'bifurcation_recall': float(bif_recall),
        'num_matched_bifurcations': num_matched_bifs,
        'total_gt_bifurcations': total_gt_bifs,
        'total_model_bifurcations': total_model_bifs,
        'model_bifurcations_matched': model_bifs_matched,
        'mean_junction_error_mm': float(np.mean(bif_errors)) if bif_errors else 0.0,
        'std_junction_error_mm': float(np.std(bif_errors)) if bif_errors else 0.0,
        'median_junction_error_mm': float(np.median(bif_errors)) if bif_errors else 0.0,
        'junction_errors': bif_errors,
    }

    return {
        'origin_offset_mm': origin_offset_mm,
        'first_bifurcation_error_mm': first_bif_error_mm,
        'branch_results': branch_results,
        'bifurcation_results': bifurcation_results,
        'summary': summary,
    }
