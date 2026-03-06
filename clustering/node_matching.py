import numpy as np
from scipy.spatial.distance import cdist
from collections import defaultdict


def compute_origin_spread(origin_coords):
    """
    Compute spread statistics for origin nodes across models.

    Since each model has exactly one origin node, this computes the centroid
    and the radius of the smallest enclosing sphere approximation.

    Args:
        origin_coords: dict of {model_label: np.ndarray of shape (1, 3)} — physical coords in mm

    Returns:
        dict with:
        - centroid: np.ndarray of shape (3,)
        - max_spread_radius_mm: float — max distance from centroid
        - mean_spread_radius_mm: float — mean distance from centroid
        - per_model_distance: dict of {model_label: float} — distance from centroid
    """
    labels = []
    coords = []
    for label, pts in origin_coords.items():
        if len(pts) > 0:
            labels.append(label)
            coords.append(pts[0])

    if len(coords) < 2:
        return {
            'centroid': coords[0] if coords else np.zeros(3),
            'max_spread_radius_mm': 0.0,
            'mean_spread_radius_mm': 0.0,
            'per_model_distance': {labels[0]: 0.0} if labels else {},
        }

    coords = np.array(coords)
    centroid = coords.mean(axis=0)
    distances = np.linalg.norm(coords - centroid, axis=1)

    per_model = {label: float(dist) for label, dist in zip(labels, distances)}

    return {
        'centroid': centroid,
        'max_spread_radius_mm': float(distances.max()),
        'mean_spread_radius_mm': float(distances.mean()),
        'per_model_distance': per_model,
    }


def cluster_nodes(node_sets, dbscan_eps_mm, max_match_distance_mm):
    """
    Cluster same-type nodes across multiple models using greedy nearest-neighbor
    assignment with a one-node-per-model constraint.

    For each cross-model node pair within max_match_distance_mm (sorted by distance),
    greedily assigns them to the same cluster — but only if neither node is already
    in a cluster that contains a node from the other's model. This prevents two
    nearby bifurcations from being lumped together.

    Args:
        node_sets: dict of {model_label: np.ndarray of shape (N, 3)} — physical coords in mm
        dbscan_eps_mm: float — not used directly, kept for config compatibility
        max_match_distance_mm: max distance to consider a match

    Returns:
        dict with:
        - clusters: list of dicts, each with:
            - models: dict of {model_label: np.ndarray coord}
            - centroid: np.ndarray of shape (3,)
            - max_pairwise_distance: float
            - mean_pairwise_distance: float
            - matched: bool (contains nodes from >= 2 models)
        - unmatched: dict of {model_label: list of np.ndarray coords}
        - stats: dict with summary statistics
    """
    all_nodes = []  # list of (model_label, node_index, coord)
    for model_label, coords in node_sets.items():
        if len(coords) == 0:
            continue
        for i, coord in enumerate(coords):
            all_nodes.append((model_label, i, coord))

    if len(all_nodes) == 0:
        return _empty_clustering_result(node_sets)

    pairs = []
    for i in range(len(all_nodes)):
        for j in range(i + 1, len(all_nodes)):
            if all_nodes[i][0] == all_nodes[j][0]:
                continue  # skip same-model pairs
            dist = float(np.linalg.norm(all_nodes[i][2] - all_nodes[j][2]))
            if dist <= max_match_distance_mm:
                pairs.append((dist, i, j))

    pairs.sort(key=lambda x: x[0])

    # Greedy assignment: each node belongs to at most one cluster,
    # each cluster has at most one node per model
    node_to_cluster = {}  # node_global_idx -> cluster_id
    cluster_models = defaultdict(set)  # cluster_id -> set of model labels
    cluster_members = defaultdict(list)  # cluster_id -> list of (model, coord)
    next_cluster_id = 0

    for dist, i, j in pairs:
        model_i, _, coord_i = all_nodes[i]
        model_j, _, coord_j = all_nodes[j]

        cluster_i = node_to_cluster.get(i)
        cluster_j = node_to_cluster.get(j)

        if cluster_i is not None and cluster_j is not None:
            if cluster_i == cluster_j:
                continue
            if cluster_models[cluster_i].isdisjoint(cluster_models[cluster_j]):
                for node_idx, cid in list(node_to_cluster.items()):
                    if cid == cluster_j:
                        node_to_cluster[node_idx] = cluster_i
                cluster_models[cluster_i] |= cluster_models[cluster_j]
                cluster_members[cluster_i].extend(cluster_members[cluster_j])
                del cluster_models[cluster_j]
                del cluster_members[cluster_j]
            continue
        elif cluster_i is not None:
            if model_j not in cluster_models[cluster_i]:
                node_to_cluster[j] = cluster_i
                cluster_models[cluster_i].add(model_j)
                cluster_members[cluster_i].append((model_j, coord_j))
        elif cluster_j is not None:
            if model_i not in cluster_models[cluster_j]:
                node_to_cluster[i] = cluster_j
                cluster_models[cluster_j].add(model_i)
                cluster_members[cluster_j].append((model_i, coord_i))
        else:
            cid = next_cluster_id
            next_cluster_id += 1
            node_to_cluster[i] = cid
            node_to_cluster[j] = cid
            cluster_models[cid] = {model_i, model_j}
            cluster_members[cid] = [(model_i, coord_i), (model_j, coord_j)]

    clusters = []
    unmatched = defaultdict(list)

    for cid, members in cluster_members.items():
        models_in_cluster = defaultdict(list)
        coords_list = []
        for model, coord in members:
            models_in_cluster[model].append(coord)
            coords_list.append(coord)

        cluster_coords = np.array(coords_list)
        centroid = cluster_coords.mean(axis=0)

        if len(cluster_coords) > 1:
            pairwise = cdist(cluster_coords, cluster_coords)
            upper_tri = pairwise[np.triu_indices_from(pairwise, k=1)]
            max_pw = float(upper_tri.max())
            mean_pw = float(upper_tri.mean())
        else:
            max_pw = 0.0
            mean_pw = 0.0

        matched = len(models_in_cluster) >= 2

        clusters.append({
            'models': {m: np.array(c) for m, c in models_in_cluster.items()},
            'centroid': centroid,
            'max_pairwise_distance': max_pw,
            'mean_pairwise_distance': mean_pw,
            'matched': matched,
        })

    assigned = set(node_to_cluster.keys())
    for idx, (model, _, coord) in enumerate(all_nodes):
        if idx not in assigned:
            unmatched[model].append(coord)

    num_models = len([l for l, c in node_sets.items() if len(c) > 0])
    matched_clusters = [c for c in clusters if c['matched']]
    stats = _compute_cluster_stats(matched_clusters, unmatched, node_sets, num_models)

    return {
        'clusters': clusters,
        'unmatched': dict(unmatched),
        'stats': stats,
    }


def compute_ground_truth_distances(node_sets, ground_truth_label, max_match_distance_mm):
    """
    Compute nearest-neighbor distances from each model's nodes to ground truth nodes.

    For each non-GT model, finds the nearest GT node for each of the model's nodes.
    Nodes beyond max_match_distance_mm are considered unmatched.

    Args:
        node_sets: dict of {model_label: np.ndarray of shape (N, 3)} — physical coords in mm
        ground_truth_label: str — key in node_sets for the ground truth model
        max_match_distance_mm: float — max distance to consider a match

    Returns:
        dict of {model_label: dict} for each non-GT model, containing:
        - matched_distances: list of float — distances to nearest GT node for matched nodes
        - mean_distance: float
        - median_distance: float
        - max_distance: float
        - num_matched: int
        - num_unmatched: int
        - unmatched_coords: np.ndarray of unmatched node coordinates
    """
    gt_coords = node_sets.get(ground_truth_label)
    if gt_coords is None or len(gt_coords) == 0:
        empty = {label: _empty_gt_result() for label in node_sets if label != ground_truth_label}
        empty['_gt_coverage'] = {
            'total': 0, 'matched': 0, 'unmatched': 0,
        }
        return empty

    # Track which GT nodes are matched by at least one model
    gt_matched_by_any = np.zeros(len(gt_coords), dtype=bool)

    results = {}
    for model_label, coords in node_sets.items():
        if model_label == ground_truth_label:
            continue

        if len(coords) == 0:
            results[model_label] = _empty_gt_result()
            continue

        dist_matrix = cdist(coords, gt_coords)
        nearest_gt_dist = dist_matrix.min(axis=1)

        matched_mask = nearest_gt_dist <= max_match_distance_mm
        matched_distances = nearest_gt_dist[matched_mask].tolist()
        unmatched_coords = coords[~matched_mask]

        nearest_gt_from_model = dist_matrix.min(axis=0)
        gt_matched_by_this = nearest_gt_from_model <= max_match_distance_mm
        gt_matched_by_any |= gt_matched_by_this

        num_gt_recalled = int(gt_matched_by_this.sum())
        total_model = len(coords)
        total_gt = len(gt_coords)
        num_overlapping = int(matched_mask.sum())
        dice = (2 * num_overlapping) / (total_gt + total_model) if (total_gt + total_model) > 0 else 0.0

        results[model_label] = {
            'matched_distances': matched_distances,
            'mean_distance': float(np.mean(matched_distances)) if matched_distances else 0.0,
            'median_distance': float(np.median(matched_distances)) if matched_distances else 0.0,
            'max_distance': float(np.max(matched_distances)) if matched_distances else 0.0,
            'num_matched': num_overlapping,
            'num_unmatched': int((~matched_mask).sum()),
            'num_gt_recalled': num_gt_recalled,
            'dice': float(dice),
            'unmatched_coords': unmatched_coords,
        }

    results['_gt_coverage'] = {
        'total': len(gt_coords),
        'matched': int(gt_matched_by_any.sum()),
        'unmatched': int((~gt_matched_by_any).sum()),
    }

    return results


def compute_pairwise_model_distances(node_sets):
    """
    Compute mean nearest-neighbor distance between each pair of models.

    For each ordered pair (A, B), computes the mean distance from each node in A
    to its nearest neighbor in B, then symmetrizes.

    Args:
        node_sets: dict of {model_label: np.ndarray of shape (N, 3)}

    Returns:
        dict of {(model_a, model_b): float} — symmetric mean NN distance for each pair
    """
    labels = [l for l, c in node_sets.items() if len(c) > 0]
    results = {}

    for i, label_a in enumerate(labels):
        for j, label_b in enumerate(labels):
            if j <= i:
                continue
            coords_a = node_sets[label_a]
            coords_b = node_sets[label_b]

            dist_ab = cdist(coords_a, coords_b)
            mean_a_to_b = float(dist_ab.min(axis=1).mean())
            mean_b_to_a = float(dist_ab.min(axis=0).mean())
            symmetric = (mean_a_to_b + mean_b_to_a) / 2.0

            results[(label_a, label_b)] = symmetric

    return results


def _compute_cluster_stats(matched_clusters, unmatched, node_sets, num_models):
    """Compute summary statistics from clustering results."""
    if matched_clusters:
        mean_matched_dist = float(np.mean([c['mean_pairwise_distance'] for c in matched_clusters]))
        max_matched_dist = float(np.max([c['max_pairwise_distance'] for c in matched_clusters]))
    else:
        mean_matched_dist = 0.0
        max_matched_dist = 0.0

    num_matched_clusters = len(matched_clusters)
    total_per_model = {label: len(coords) for label, coords in node_sets.items()}
    unmatched_per_model = {label: len(unmatched.get(label, [])) for label in node_sets}
    matched_per_model = {label: total_per_model[label] - unmatched_per_model[label]
                         for label in node_sets}

    # Break down matched nodes by agreement level:
    # full = cluster contains all models that had data, partial = 2+ but not all
    full_match_per_model = defaultdict(int)
    partial_match_per_model = defaultdict(int)
    full_clusters = 0
    partial_clusters = 0

    for cluster in matched_clusters:
        n_models_in_cluster = len(cluster['models'])
        is_full = n_models_in_cluster == num_models

        if is_full:
            full_clusters += 1
        else:
            partial_clusters += 1

        for model in cluster['models']:
            if is_full:
                full_match_per_model[model] += 1
            else:
                partial_match_per_model[model] += 1

    return {
        'num_matched_clusters': num_matched_clusters,
        'num_full_clusters': full_clusters,
        'num_partial_clusters': partial_clusters,
        'mean_matched_distance_mm': mean_matched_dist,
        'max_matched_distance_mm': max_matched_dist,
        'total_unmatched': sum(unmatched_per_model.values()),
        'total_per_model': total_per_model,
        'matched_per_model': matched_per_model,
        'full_match_per_model': dict(full_match_per_model),
        'partial_match_per_model': dict(partial_match_per_model),
        'unmatched_per_model': unmatched_per_model,
    }


def _empty_clustering_result(node_sets):
    """Return empty clustering result when no nodes to cluster."""
    return {
        'clusters': [],
        'unmatched': {},
        'stats': {
            'num_matched_clusters': 0,
            'num_full_clusters': 0,
            'num_partial_clusters': 0,
            'mean_matched_distance_mm': 0.0,
            'max_matched_distance_mm': 0.0,
            'total_unmatched': 0,
            'total_per_model': {label: 0 for label in node_sets},
            'matched_per_model': {label: 0 for label in node_sets},
            'full_match_per_model': {label: 0 for label in node_sets},
            'partial_match_per_model': {label: 0 for label in node_sets},
            'unmatched_per_model': {label: 0 for label in node_sets},
        },
    }


def _empty_gt_result():
    """Return empty ground truth result."""
    return {
        'matched_distances': [],
        'mean_distance': 0.0,
        'median_distance': 0.0,
        'max_distance': 0.0,
        'num_matched': 0,
        'num_unmatched': 0,
        'num_gt_recalled': 0,
        'dice': 0.0,
        'unmatched_coords': np.empty((0, 3)),
    }
