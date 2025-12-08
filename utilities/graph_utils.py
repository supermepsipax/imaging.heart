import numpy as np
import networkx as nx


def find_connected_voxels(coordinate, binary_mask):
    """
    Given a single coordinate within a 3D space, will find and return
    all "touching" voxel coordinates. This assumes a voxel is touching
    if it is non-zero in any of the 26 surrounding positions.

    Args:
        coordinate (tuple): The coordinate of the voxel in array index order (axis0, axis1, axis2)
        binary_mask (array): A numpy binary mask

    Returns:
        neighbor_voxels: A list of coordinates (if any) of touching voxels in array index order

    """

    i0, i1, i2 = coordinate
    neighbor_voxels = []
    for d0 in (-1, 0, 1):
        for d1 in (-1, 0, 1):
            for d2 in (-1, 0, 1):
                if d0 == d1 == d2 == 0:
                    continue
                ni0, ni1, ni2 = i0 + d0, i1 + d1, i2 + d2
                if (
                    0 <= ni0 < binary_mask.shape[0]
                    and 0 <= ni1 < binary_mask.shape[1]
                    and 0 <= ni2 < binary_mask.shape[2]
                    and binary_mask[ni0, ni1, ni2] == 1
                ):
                    neighbor_voxels.append((ni0, ni1, ni2))
    return neighbor_voxels


def skeleton_to_sparse_graph(binary_mask, bifurcation_points, endpoints):
    """
    Builds and returns a sparse graph representation of the arteries using networkx.

    Combines the list of bifurcation and endpoint coordinates together and adds
    them as nodes in the created graph. Then paths are traced outwardly from nodes
    via connected voxels to find and create the edges between nodes. The exact voxel
    path for each edge is logged and stored as edge attribute 'voxels'. Ideally this
    should be done on a mask that is known to be a single continous body.

    Args:
        binary_mask (array): A numpy binary mask
        bifurcation_points (list): List of 3D coordinates in array index order (axis0, axis1, axis2)
        endpoints (list): List of 3D coordinates in array index order (axis0, axis1, axis2)

    Returns:
        graph: A networkx graph object where edges have 'voxels' attribute containing
               coordinate paths in array index order

    """
    graph = nx.Graph() 
    nodes = set(map(tuple, np.vstack((bifurcation_points, endpoints))))
    visited_voxels = set()

    graph.add_nodes_from(nodes)

    for node_coordinate in nodes:
        for connected_voxel in find_connected_voxels(node_coordinate, binary_mask):
            if connected_voxel in visited_voxels:
                continue
            path = [node_coordinate]
            current_voxel = connected_voxel
            previous_voxel = node_coordinate

            while current_voxel not in nodes:
                path.append(current_voxel)
                visited_voxels.add(current_voxel)
                connected_voxels = [n for n in find_connected_voxels(current_voxel, binary_mask) if n != previous_voxel]

                if len(connected_voxels) == 0:
                    # dead end
                    break
                elif len(connected_voxels) == 1:
                    # simple continuation
                    previous_voxel, current_voxel = current_voxel, connected_voxels[0]
                else:
                    # multiple connections - check if any are target nodes
                    node_candidates = [v for v in connected_voxels if v in nodes]

                    if len(node_candidates) == 1:
                        # found our target node, continue to it
                        previous_voxel, current_voxel = current_voxel, node_candidates[0]
                    else:
                        # check for unvisited non-node paths (filtered bifurcation voxels)
                        unvisited = [v for v in connected_voxels if v not in visited_voxels]
                        if len(unvisited) == 1:
                            # continue through the filtered bifurcation
                            previous_voxel, current_voxel = current_voxel, unvisited[0]
                        else:
                            # unexpected situation - multiple unvisited paths or multiple nodes
                            break

            if current_voxel in nodes:
                path.append(current_voxel)
                graph.add_edge(node_coordinate, current_voxel, voxels=path)

    return graph


def skeleton_to_dense_graph(binary_mask):
    """
    Converts a skeletonized binary mask into a dense networkx graph representation.

    Creates a graph where every non-zero voxel in the skeleton becomes a node,
    and edges connect voxels that are touching in 26-connectivity. This dense
    graph representation is useful for topology analysis and cycle detection
    in skeletonized structures.

    Args:
        binary_mask (array): A numpy binary mask representing a skeletonized structure

    Returns:
        graph: A networkx graph object with nodes at each skeleton voxel

    """
    graph = nx.Graph()

    skeleton_coordinates = set(map(tuple, np.argwhere(binary_mask == 1)))

    graph.add_nodes_from(skeleton_coordinates)

    for coordinate in skeleton_coordinates:
        neighbor_voxels = find_connected_voxels(coordinate, binary_mask)

        for neighbor_coordinate in neighbor_voxels:
            if graph.has_node(neighbor_coordinate):
                graph.add_edge(coordinate, neighbor_coordinate)

    return graph


def skeleton_to_sparse_graph_robust(binary_mask, bifurcation_points, endpoints,
                                     min_branch_length_voxels=None, min_branch_length_mm=None,
                                     spacing=None, max_recursion_depth=5, _current_depth=0):
    """
    Builds a sparse graph using dense graph shortest paths for robust pathfinding.

    This approach is more robust to close centerlines and complex topology compared
    to the greedy pathfinding approach. It builds a dense graph first, then finds
    shortest paths between all node pairs, keeping only direct connections (paths
    that don't pass through other nodes).

    Optionally filters out short dead-end branches by recursively rebuilding the graph
    with updated node lists. When short branches are detected, their endpoints are removed
    and bifurcations that become degree-2 are demoted, then the graph is rebuilt.

    Args:
        binary_mask (array): A numpy binary mask representing a skeletonized structure
        bifurcation_points (list): List of 3D coordinates in array index order (axis0, axis1, axis2)
        endpoints (list): List of 3D coordinates in array index order (axis0, axis1, axis2)
        min_branch_length_voxels (int, optional): Minimum path length in voxels. Shorter branches
                                                   will trigger recursive graph rebuilding.
        min_branch_length_mm (float, optional): Minimum path length in mm. Requires spacing parameter.
        spacing (tuple, optional): Voxel spacing (z, y, x) in mm for converting voxel distances to mm.
                                   Required if using min_branch_length_mm.
        max_recursion_depth (int, optional): Maximum number of recursive rebuilds (default: 5)
        _current_depth (int, optional): Internal parameter tracking recursion depth

    Returns:
        graph: A networkx graph object where edges have 'voxels' attribute containing
               coordinate paths, with only direct connections between nodes
    """
    dense_graph = skeleton_to_dense_graph(binary_mask)

    sparse_graph = nx.Graph()
    nodes = set(map(tuple, np.vstack((bifurcation_points, endpoints))))
    sparse_graph.add_nodes_from(nodes)

    # Find direct connections (paths that don't pass through other nodes)
    nodes_list = list(nodes)
    edges_added = 0

    for i, node1 in enumerate(nodes_list):
        for node2 in nodes_list[i+1:]:
            try:
                path = nx.shortest_path(dense_graph, node1, node2)

                # Check if path passes through any other nodes (besides endpoints)
                intermediate_nodes = [v for v in path[1:-1] if v in nodes]

                if len(intermediate_nodes) == 0:
                    sparse_graph.add_edge(node1, node2, voxels=path)
                    edges_added += 1

            except nx.NetworkXNoPath:
                continue

    # If no filtering requested, return the graph as-is
    if min_branch_length_voxels is None and min_branch_length_mm is None:
        return sparse_graph

    # Determine minimum path length threshold
    min_voxels_threshold = None
    if min_branch_length_voxels is not None:
        min_voxels_threshold = min_branch_length_voxels
    elif min_branch_length_mm is not None and spacing is not None:
        # Convert mm to approximate voxel count using average spacing
        avg_spacing = np.mean(spacing)
        min_voxels_threshold = int(min_branch_length_mm / avg_spacing)

    # Identify short dead-end branches and nodes to remove
    endpoints_to_remove = set()
    bifurcations_to_demote = set()

    endpoint_set = set(map(tuple, endpoints))
    bifurcation_set = set(map(tuple, bifurcation_points))

    for node in sparse_graph.nodes():
        if sparse_graph.degree(node) == 1:  # Dead-end endpoint
            neighbor = list(sparse_graph.neighbors(node))[0]
            edge_data = sparse_graph.edges[node, neighbor]
            path_length = len(edge_data['voxels'])

            if path_length < min_voxels_threshold:
                # Mark endpoint for removal
                if node in endpoint_set:
                    endpoints_to_remove.add(node)

                # Check if neighbor will become degree-2 (should be demoted from bifurcation)
                if neighbor in bifurcation_set and sparse_graph.degree(neighbor) == 3:
                    bifurcations_to_demote.add(neighbor)

    # If nothing to remove, we're done
    if len(endpoints_to_remove) == 0 and len(bifurcations_to_demote) == 0:
        return sparse_graph

    # Check recursion depth
    if _current_depth >= max_recursion_depth:
        print(f"      --> Reached max recursion depth ({max_recursion_depth}), stopping branch removal")
        return sparse_graph

    # Update node lists
    new_endpoints = [ep for ep in endpoints if tuple(ep) not in endpoints_to_remove]
    new_bifurcations = [bf for bf in bifurcation_points if tuple(bf) not in bifurcations_to_demote]

    if _current_depth == 0:
        print(f"      --> Removing {len(endpoints_to_remove)} short dead-end branches (< {min_voxels_threshold} voxels)")
        if len(bifurcations_to_demote) > 0:
            print(f"      --> Demoting {len(bifurcations_to_demote)} bifurcations that became degree-2")

    # Recursively rebuild graph with cleaned node lists
    return skeleton_to_sparse_graph_robust(
        binary_mask, new_bifurcations, new_endpoints,
        min_branch_length_voxels, min_branch_length_mm, spacing,
        max_recursion_depth, _current_depth + 1
    )


def make_directed_graph(undirected_graph, origin_node):

    directed_graph = nx.DiGraph()

    for node, data in undirected_graph.nodes(data=True):
        directed_graph.add_node(node, **data)

    for u, v in nx.bfs_edges(undirected_graph, origin_node):
        edge_data = undirected_graph.get_edge_data(u, v, default={}).copy()
        
        directed_graph.add_edge(u, v, **edge_data)
        
    return directed_graph



def remove_bypass_edges(graph, distance_threshold=2.0, endpoints=None):
    """
    Removes bypass edges that pass near bifurcation nodes without going through them,
    and removes direct connections between endpoints that shouldn't exist.

    This function performs two types of bypass detection:
    1. Detects edges that pass within threshold distance of bifurcation nodes (degree >= 3)
    2. Detects endpoints with degree > 1 and removes false edges connecting endpoints

    Args:
        graph (networkx.Graph): Undirected sparse graph with 'voxels' edge attributes
        distance_threshold (float): Maximum distance (in voxels) for a path to be
                                   considered as passing near a node (default: 2.0)
        endpoints (list, optional): List of endpoint coordinates to check for false connections

    Returns:
        cleaned_graph: Graph with bypass edges removed
    """
    cleaned_graph = graph.copy()
    edges_to_remove = set()

    # Check 1: Remove direct connections between endpoints (endpoints should have degree 1)
    if endpoints is not None:
        endpoint_set = set(map(tuple, endpoints))

        for endpoint in endpoint_set:
            if endpoint not in cleaned_graph.nodes():
                continue

            if cleaned_graph.degree(endpoint) == 2:
                # Endpoint has degree 2 - one connection is false
                neighbors = list(cleaned_graph.neighbors(endpoint))

                # Check which neighbor is also an endpoint
                for neighbor in neighbors:
                    if neighbor in endpoint_set:
                        # Found direct connection between two endpoints - this is a bypass
                        edge = tuple(sorted([endpoint, neighbor]))
                        edges_to_remove.add(edge)
                        print(f"      --> Bypass edge detected: {endpoint} <-> {neighbor} (direct endpoint connection)")

    # Check 2: Remove edges that pass near bifurcation nodes
    high_degree_nodes = [node for node in cleaned_graph.nodes() if cleaned_graph.degree(node) >= 3]

    if len(high_degree_nodes) > 0:
        print(f"      [INFO] Found {len(high_degree_nodes)} nodes with degree >= 3")

    for node in high_degree_nodes:
        for edge in list(cleaned_graph.edges()):
            u, v = edge

            if u == node or v == node:
                continue

            voxel_path = cleaned_graph.edges[edge]['voxels']

            min_distance = float('inf')
            for voxel in voxel_path:
                distance = np.linalg.norm(np.array(voxel) - np.array(node))
                min_distance = min(min_distance, distance)

            if min_distance <= distance_threshold:
                edges_to_remove.add(edge)
                print(f"      --> Bypass edge detected: {edge} passes within {min_distance:.2f} voxels of node {node}")

    # Remove all detected bypass edges
    cleaned_graph.remove_edges_from(edges_to_remove)

    if len(edges_to_remove) > 0:
        print(f"      [OK] Removed {len(edges_to_remove)} bypass edges")
    else:
        print(f"      [INFO] No bypass edges found")

    return cleaned_graph


def dense_graph_to_skeleton(graph, reference_mask=None):
    """
    Converts a dense graph back into a binary mask representation.

    Takes a graph where nodes are coordinate tuples and creates a binary mask
    with 1's at all node positions. If a reference mask is provided, uses its
    shape to create the output array. Otherwise, determines the minimum bounding
    box needed to contain all coordinates.

    Args:
        graph (networkx.Graph): A networkx graph object with coordinate tuples as nodes
        reference_mask (array): Optional reference binary mask to determine output shape

    Returns:
        binary_mask: A numpy binary mask with 1's at all graph node positions

    """
    if reference_mask is not None:
        output_shape = reference_mask.shape
    else:
        node_coordinates = np.array(list(graph.nodes()))
        max_coordinates = np.max(node_coordinates, axis=0)
        output_shape = tuple(max_coordinates + 1)

    binary_mask = np.zeros(output_shape, dtype=np.uint8)

    for node_coordinate in graph.nodes():
        binary_mask[node_coordinate] = 1

    return binary_mask


def remove_small_y_branches(graph, max_branch_length_voxels=None, max_branch_length_mm=None):
    """
    Removes small Y-shaped spurious branches from the graph (single iteration).

    A Y-branch is detected when a bifurcation node (degree >= 3) has at least 2
    neighbors that are endpoints (degree == 1), and both endpoint branches are
    below the length threshold. This targets skeletonization artifacts that
    create small spurious branches.

    Args:
        graph (networkx.Graph): Undirected sparse graph with 'voxels' edge attribute
                                (and optionally 'path_length_mm' attribute)
        max_branch_length_voxels (int, optional): Maximum voxel path length for a
                                                   branch to be considered "small"
        max_branch_length_mm (float, optional): Maximum physical length (mm) for a
                                                branch to be considered "small"
                                                (requires 'path_length_mm' in edges)

    Returns:
        tuple: (cleaned_graph, num_y_branches_removed)
            - cleaned_graph: Graph with small Y-branches removed
            - num_y_branches_removed: Number of Y-branch structures removed

    Notes:
        - At least one threshold (voxels or mm) must be provided
        - If both are provided, a branch must satisfy BOTH criteria to be removed
        - Only removes Y-branches where BOTH endpoint branches are small (conservative)
    """
    if max_branch_length_voxels is None and max_branch_length_mm is None:
        raise ValueError("At least one threshold (max_branch_length_voxels or max_branch_length_mm) must be provided")

    cleaned_graph = graph.copy()
    edges_to_remove = set()
    nodes_to_remove = set()
    y_branches_removed = 0

    # Find bifurcation nodes (degree >= 3)
    bifurcation_nodes = [node for node in cleaned_graph.nodes() if cleaned_graph.degree(node) >= 3]

    for bifurcation_node in bifurcation_nodes:
        # Get all neighbors that are endpoints (degree == 1)
        neighbors = list(cleaned_graph.neighbors(bifurcation_node))
        endpoint_neighbors = [n for n in neighbors if cleaned_graph.degree(n) == 1]

        # Need at least 2 endpoint neighbors to form a Y-branch
        if len(endpoint_neighbors) < 2:
            continue

        # Check all pairs of endpoint neighbors
        # Find pairs where both branches are small
        small_endpoint_pairs = []

        for i in range(len(endpoint_neighbors)):
            for j in range(i + 1, len(endpoint_neighbors)):
                endpoint1 = endpoint_neighbors[i]
                endpoint2 = endpoint_neighbors[j]

                edge1 = (bifurcation_node, endpoint1) if cleaned_graph.has_edge(bifurcation_node, endpoint1) else (endpoint1, bifurcation_node)
                edge2 = (bifurcation_node, endpoint2) if cleaned_graph.has_edge(bifurcation_node, endpoint2) else (endpoint2, bifurcation_node)

                # Check if both branches are small
                edge1_is_small = False
                edge2_is_small = False

                # Check voxel length threshold
                if max_branch_length_voxels is not None:
                    voxels1 = cleaned_graph.edges[edge1].get('voxels', [])
                    voxels2 = cleaned_graph.edges[edge2].get('voxels', [])
                    edge1_voxel_length = len(voxels1)
                    edge2_voxel_length = len(voxels2)

                    edge1_is_small = edge1_voxel_length <= max_branch_length_voxels
                    edge2_is_small = edge2_voxel_length <= max_branch_length_voxels

                    # If mm threshold is also provided, check that too
                    if max_branch_length_mm is not None:
                        path_length1 = cleaned_graph.edges[edge1].get('path_length_mm')
                        path_length2 = cleaned_graph.edges[edge2].get('path_length_mm')

                        if path_length1 is not None and path_length2 is not None:
                            # Both thresholds must be satisfied
                            edge1_is_small = edge1_is_small and (path_length1 <= max_branch_length_mm)
                            edge2_is_small = edge2_is_small and (path_length2 <= max_branch_length_mm)
                        # else: path_length_mm not available, use only voxel threshold

                elif max_branch_length_mm is not None:
                    # Only mm threshold provided
                    path_length1 = cleaned_graph.edges[edge1].get('path_length_mm')
                    path_length2 = cleaned_graph.edges[edge2].get('path_length_mm')

                    if path_length1 is not None and path_length2 is not None:
                        edge1_is_small = path_length1 <= max_branch_length_mm
                        edge2_is_small = path_length2 <= max_branch_length_mm
                    else:
                        # path_length_mm not available, can't check
                        continue

                # If both branches are small, mark for removal
                if edge1_is_small and edge2_is_small:
                    small_endpoint_pairs.append((endpoint1, endpoint2, edge1, edge2))

        # Remove all small Y-branch pairs from this bifurcation
        for endpoint1, endpoint2, edge1, edge2 in small_endpoint_pairs:
            edges_to_remove.add(edge1)
            edges_to_remove.add(edge2)
            nodes_to_remove.add(endpoint1)
            nodes_to_remove.add(endpoint2)
            y_branches_removed += 1

    # Remove marked edges and nodes
    cleaned_graph.remove_edges_from(edges_to_remove)
    cleaned_graph.remove_nodes_from(nodes_to_remove)

    return cleaned_graph, y_branches_removed


def prune_small_y_branches_iterative(graph, max_branch_length_voxels=None,
                                      max_branch_length_mm=None, max_iterations=5):
    """
    Iteratively removes small Y-shaped spurious branches until none remain.

    Applies the Y-branch removal algorithm repeatedly, as removing one Y-branch
    may expose another (e.g., a node with degree 4 becomes degree 2 after removing
    two branches, potentially revealing a new Y-pattern with remaining branches).

    Args:
        graph (networkx.Graph): Undirected sparse graph with 'voxels' edge attribute
        max_branch_length_voxels (int, optional): Maximum voxel path length threshold
        max_branch_length_mm (float, optional): Maximum physical length threshold (mm)
        max_iterations (int): Maximum number of pruning iterations (default: 5)

    Returns:
        tuple: (cleaned_graph, total_y_branches_removed)
            - cleaned_graph: Graph with all small Y-branches removed
            - total_y_branches_removed: Total number of Y-branch structures removed

    Notes:
        - Stops early if no Y-branches are found in an iteration
        - Prevents infinite loops by capping iterations at max_iterations
    """
    cleaned_graph = graph.copy()
    total_removed = 0

    for iteration in range(max_iterations):
        cleaned_graph, count = remove_small_y_branches(
            cleaned_graph,
            max_branch_length_voxels=max_branch_length_voxels,
            max_branch_length_mm=max_branch_length_mm
        )
        total_removed += count

        if count == 0:
            # No more Y-branches found, stop iterating
            break

    return cleaned_graph, total_removed
