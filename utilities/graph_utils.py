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

    # Handle empty bifurcation_points or endpoints to avoid vstack shape mismatch
    if len(bifurcation_points) == 0 and len(endpoints) == 0:
        return graph  # Empty graph
    elif len(bifurcation_points) == 0:
        nodes = set(map(tuple, endpoints))
    elif len(endpoints) == 0:
        nodes = set(map(tuple, bifurcation_points))
    else:
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



def compute_branch_complexity(graph, start_node, exclude_node):
    """
    Computes complexity metrics for branches reachable from start_node,
    excluding paths through exclude_node.

    Uses BFS to explore the graph starting from start_node, but doesn't
    traverse back through exclude_node. Useful for analyzing the complexity
    of "other branches" at a bifurcation when deciding whether to remove
    a short endpoint.

    Args:
        graph (networkx.Graph): Undirected graph with 'voxels' edge attributes
        start_node (tuple): Node to start exploration from
        exclude_node (tuple): Node to exclude from traversal (typically the short endpoint)

    Returns:
        dict: Complexity metrics with keys:
            - num_nodes: Number of reachable nodes
            - total_voxel_length: Sum of all edge voxel path lengths
            - num_bifurcations: Number of degree >= 3 nodes found
            - num_endpoints: Number of degree == 1 nodes found
    """
    if start_node not in graph.nodes():
        return {'num_nodes': 0, 'total_voxel_length': 0, 'num_bifurcations': 0, 'num_endpoints': 0}

    visited = set([exclude_node])  # Don't explore back through exclude_node
    queue = [start_node]
    visited.add(start_node)

    num_nodes = 0
    total_voxel_length = 0
    num_bifurcations = 0
    num_endpoints = 0

    while queue:
        node = queue.pop(0)
        num_nodes += 1

        degree = graph.degree(node)
        if degree == 1:
            num_endpoints += 1
        elif degree >= 3:
            num_bifurcations += 1

        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                # Add edge voxel length
                edge_voxels = graph[node][neighbor].get('voxels', [])
                total_voxel_length += len(edge_voxels)

    return {
        'num_nodes': num_nodes,
        'total_voxel_length': total_voxel_length,
        'num_bifurcations': num_bifurcations,
        'num_endpoints': num_endpoints
    }


def skeleton_to_sparse_graph_robust(binary_mask, bifurcation_points, endpoints, distance_threshold=2.0,
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
        tuple: (graph, updated_endpoints, updated_bifurcations)
            - graph: A networkx graph object where edges have 'voxels' attribute containing
                    coordinate paths, with only direct connections between nodes
            - updated_endpoints: List of endpoint coordinates after pruning
            - updated_bifurcations: List of bifurcation coordinates after pruning
    """
    dense_graph = skeleton_to_dense_graph(binary_mask)

    sparse_graph = nx.Graph()

    # Handle empty bifurcation_points or endpoints to avoid vstack shape mismatch
    if len(bifurcation_points) == 0 and len(endpoints) == 0:
        return sparse_graph, [], []  # Empty graph
    elif len(bifurcation_points) == 0:
        nodes = set(map(tuple, endpoints))
    elif len(endpoints) == 0:
        nodes = set(map(tuple, bifurcation_points))
    else:
        nodes = set(map(tuple, np.vstack((bifurcation_points, endpoints))))

    sparse_graph.add_nodes_from(nodes)

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

    sparse_graph = remove_bypass_edges(sparse_graph, distance_threshold, endpoints)

    # Detect and break cycles using vector-based analysis
    if spacing is not None:
        sparse_graph = detect_and_break_cycle(sparse_graph, spacing)

    # Check for degree-2 nodes (continuations that shouldn't exist in vessel tree)
    # These can occur after bypass/cycle edge removal - demote them and rebuild
    bifurcation_set = set(map(tuple, bifurcation_points))
    degree_2_bifurcations = [
        node for node in sparse_graph.nodes()
        if sparse_graph.degree(node) == 2 and node in bifurcation_set
    ]

    if len(degree_2_bifurcations) > 0:
        if _current_depth == 0:
            print(f"      --> Found {len(degree_2_bifurcations)} degree-2 nodes in bifurcation list")
            print(f"      --> Demoting to continuations and rebuilding graph")

        # Remove degree-2 nodes from bifurcation list and rebuild
        new_bifurcations = [
            bf for bf in bifurcation_points
            if tuple(bf) not in set(degree_2_bifurcations)
        ]

        return skeleton_to_sparse_graph_robust(
            binary_mask, new_bifurcations, endpoints, distance_threshold,
            min_branch_length_voxels, min_branch_length_mm, spacing,
            max_recursion_depth, _current_depth + 1
        )

    # Short branch removal using neighbor branch complexity analysis
    if min_branch_length_voxels is None and min_branch_length_mm is None:
        return sparse_graph, endpoints, bifurcation_points

    if _current_depth >= max_recursion_depth:
        print(f"      --> Reached max recursion depth ({max_recursion_depth}), stopping branch removal")
        return sparse_graph, endpoints, bifurcation_points

    # Calculate threshold
    min_voxels_threshold = None
    if min_branch_length_voxels is not None:
        min_voxels_threshold = min_branch_length_voxels
    elif min_branch_length_mm is not None and spacing is not None:
        avg_spacing = np.mean(spacing)
        min_voxels_threshold = int(min_branch_length_mm / avg_spacing)

    # Find all short endpoint branches
    endpoint_set = set(map(tuple, endpoints))
    bifurcation_set = set(map(tuple, bifurcation_points))

    short_branches = []  # (endpoint, bifurcation_neighbor, voxel_length)
    for node in sparse_graph.nodes():
        if sparse_graph.degree(node) == 1 and node in endpoint_set:
            neighbor = list(sparse_graph.neighbors(node))[0]
            edge_voxels = sparse_graph[node][neighbor].get('voxels', [])
            voxel_length = len(edge_voxels)

            if voxel_length < min_voxels_threshold:
                short_branches.append((node, neighbor, voxel_length))

    if len(short_branches) == 0:
        return sparse_graph, endpoints, bifurcation_points

    if _current_depth == 0:
        print(f"      --> Found {len(short_branches)} short branches (< {min_voxels_threshold} voxels)")
        print(f"      --> Analyzing neighbor branch complexity to determine safe removals...")

    # Analyze each short branch
    endpoints_to_remove = set()

    for endpoint, bifurcation, voxel_length in short_branches:
        # Get all OTHER neighbors of the bifurcation (not the short endpoint)
        other_neighbors = [n for n in sparse_graph.neighbors(bifurcation) if n != endpoint]

        if len(other_neighbors) == 0:
            # Shouldn't happen - bifurcation only connects to this endpoint
            continue

        # Compute complexity for EACH other branch separately
        branch_complexities = []
        for other_neighbor in other_neighbors:
            branch_complexity = compute_branch_complexity(sparse_graph, other_neighbor, bifurcation)

            # Compute complexity score for this branch
            complexity_score = (
                branch_complexity['num_nodes'] * 1.0 +
                branch_complexity['total_voxel_length'] / 10.0 +  # Normalize voxel length
                branch_complexity['num_bifurcations'] * 5.0 +
                branch_complexity['num_endpoints'] * 2.0
            )
            branch_complexities.append(complexity_score)

        # Decision threshold for considering a branch "complex"
        # Lower threshold = more aggressive removal
        # Higher threshold = more conservative (protect more branches)
        COMPLEXITY_THRESHOLD = 20.0

        # Check if ALL other branches are complex
        # If all sides are complex → this short branch connects main vessels (like Left Main) → KEEP IT
        # If any side is not complex → this is a spurious artifact → REMOVE IT
        all_branches_complex = all(score > COMPLEXITY_THRESHOLD for score in branch_complexities)

        if all_branches_complex:
            # Both/all sides are complex - could be origin node connecting main branches, DON'T remove
            continue
        else:
            # At least one side is not complex - this is likely a spurious artifact, safe to remove
            endpoints_to_remove.add(endpoint)

    if len(endpoints_to_remove) == 0:
        if _current_depth == 0:
            print(f"      --> All short branches connect complex branches on both sides - keeping to preserve structure")
        return sparse_graph, endpoints, bifurcation_points

    # Now determine which bifurcations should be demoted or converted to endpoints
    # We need to count how many neighbors of each bifurcation are being removed
    bifurcations_to_demote = set()
    bifurcations_to_convert_to_endpoints = set()

    for bifurcation in bifurcation_set:
        # Count how many neighbors are being removed
        neighbors = list(sparse_graph.neighbors(bifurcation))
        num_neighbors_being_removed = sum(1 for n in neighbors if tuple(n) in endpoints_to_remove)

        if num_neighbors_being_removed > 0:
            # Calculate degree after ALL removals
            degree_after_removal = sparse_graph.degree(bifurcation) - num_neighbors_being_removed

            if degree_after_removal == 1:
                # This bifurcation will become a leaf - convert to endpoint
                bifurcations_to_convert_to_endpoints.add(bifurcation)
            elif degree_after_removal == 2:
                # This bifurcation will become a continuation - demote
                bifurcations_to_demote.add(bifurcation)
            # If degree_after_removal >= 3, keep as bifurcation (no action needed)

    # Remove endpoints and rebuild
    new_endpoints = [ep for ep in endpoints if tuple(ep) not in endpoints_to_remove]
    # Add bifurcations that are becoming endpoints
    for bf_node in bifurcations_to_convert_to_endpoints:
        # Convert tuple to list if needed
        new_endpoints.append(list(bf_node) if isinstance(bf_node, tuple) else bf_node)

    new_bifurcations = [bf for bf in bifurcation_points
                       if tuple(bf) not in bifurcations_to_demote
                       and tuple(bf) not in bifurcations_to_convert_to_endpoints]

    if _current_depth == 0:
        print(f"      --> Removing {len(endpoints_to_remove)}/{len(short_branches)} short branches (at least one non-complex side)")
        if len(bifurcations_to_convert_to_endpoints) > 0:
            print(f"      --> Converting {len(bifurcations_to_convert_to_endpoints)} bifurcations to endpoints (became degree-1)")
        if len(bifurcations_to_demote) > 0:
            print(f"      --> Demoting {len(bifurcations_to_demote)} bifurcations to continuations (became degree-2)")

    # Recursively rebuild to handle cascading removals
    return skeleton_to_sparse_graph_robust(
        binary_mask, new_bifurcations, new_endpoints, distance_threshold,
        min_branch_length_voxels, min_branch_length_mm, spacing,
        max_recursion_depth, _current_depth + 1
    )


def merge_edges(edge1_data, edge2_data, spacing):
    """
    Merges two consecutive edges by combining their attributes.

    Combines voxel paths, diameter profiles, and recalculates statistics.
    Used when removing intermediate nodes during leaf pruning.

    Args:
        edge1_data (dict): First edge attributes (parent -> intermediate)
        edge2_data (dict): Second edge attributes (intermediate -> child)
        spacing (tuple): Voxel spacing (z, y, x) in mm

    Returns:
        dict: Merged edge attributes
    """
    merged = {}

    voxels1 = edge1_data.get('voxels', [])
    voxels2 = edge2_data.get('voxels', [])

    if len(voxels1) == 0 or len(voxels2) == 0:
        merged_voxels = voxels1 + voxels2
    else:
        # Check which ends connect and orient voxels correctly
        # Case 1: edge1 ends where edge2 starts (correct orientation)
        if voxels1[-1] == voxels2[0]:
            merged_voxels = voxels1 + voxels2[1:]  # Skip duplicate
        # Case 2: edge1 ends where edge2 ends (reverse edge2)
        elif voxels1[-1] == voxels2[-1]:
            merged_voxels = voxels1 + list(reversed(voxels2))[1:]  # Reverse edge2, skip duplicate
        # Case 3: edge1 starts where edge2 starts (reverse edge1)
        elif voxels1[0] == voxels2[0]:
            merged_voxels = list(reversed(voxels1))[:-1] + voxels2  # Reverse edge1, skip duplicate
        # Case 4: edge1 starts where edge2 ends (reverse both - edge1 should come after edge2)
        elif voxels1[0] == voxels2[-1]:
            merged_voxels = voxels2 + list(reversed(voxels1))[1:]  # edge2 first, then reversed edge1, skip duplicate
        else:
            # No matching ends - just concatenate (shouldn't happen in well-formed graphs)
            merged_voxels = voxels1 + voxels2

    merged['voxels'] = merged_voxels

    # Merge diameter profiles using start/end coordinates to determine orientation
    profile1_start = edge1_data.get('diameter_profile_start_coord')
    profile1_end = edge1_data.get('diameter_profile_end_coord')
    profile2_start = edge2_data.get('diameter_profile_start_coord')
    profile2_end = edge2_data.get('diameter_profile_end_coord')

    # Determine diameter profile orientation
    if profile1_end is not None and profile2_start is not None:
        if profile1_end == profile2_start:
            # Case 1: profile1 ends where profile2 starts (correct orientation)
            reverse_profile1 = False
            reverse_profile2 = False
            profile1_first = True
        elif profile1_end == profile2_end:
            # Case 2: profile1 ends where profile2 ends (reverse profile2)
            reverse_profile1 = False
            reverse_profile2 = True
            profile1_first = True
        elif profile1_start == profile2_start:
            # Case 3: profile1 starts where profile2 starts (reverse profile1)
            reverse_profile1 = True
            reverse_profile2 = False
            profile1_first = True
        elif profile1_start == profile2_end:
            # Case 4: profile1 starts where profile2 ends (profile2 first, then reversed profile1)
            reverse_profile1 = True
            reverse_profile2 = False
            profile1_first = False
        else:
            # No match - use default orientation
            reverse_profile1 = False
            reverse_profile2 = False
            profile1_first = True
    else:
        # No coordinate info - use default orientation
        reverse_profile1 = False
        reverse_profile2 = False
        profile1_first = True

    # Merge diameter profiles - slicing method
    if 'diameter_profile_slicing' in edge1_data and 'diameter_profile_slicing' in edge2_data:
        profile1 = edge1_data['diameter_profile_slicing']
        profile2 = edge2_data['diameter_profile_slicing']

        if reverse_profile1:
            profile1 = list(reversed(profile1))
        else:
            profile1 = list(profile1)

        if reverse_profile2:
            profile2 = list(reversed(profile2))
        else:
            profile2 = list(profile2)

        if profile1_first:
            merged_profile = profile1 + profile2
        else:
            merged_profile = profile2 + profile1

        merged['diameter_profile_slicing'] = merged_profile
        merged['mean_diameter_slicing'] = np.mean(merged_profile)
        merged['median_diameter_slicing'] = np.median(merged_profile)

    # Merge diameter profiles - EDT method
    if 'diameter_profile_edt' in edge1_data and 'diameter_profile_edt' in edge2_data:
        profile1 = edge1_data['diameter_profile_edt']
        profile2 = edge2_data['diameter_profile_edt']

        if reverse_profile1:
            profile1 = list(reversed(profile1))
        else:
            profile1 = list(profile1)

        if reverse_profile2:
            profile2 = list(reversed(profile2))
        else:
            profile2 = list(profile2)

        if profile1_first:
            merged_profile = profile1 + profile2
        else:
            merged_profile = profile2 + profile1

        merged['diameter_profile_edt'] = merged_profile
        merged['mean_diameter_edt'] = np.mean(merged_profile)
        merged['median_diameter_edt'] = np.median(merged_profile)

    # Store merged diameter profile coordinates based on orientation
    if profile1_first:
        if reverse_profile1:
            merged['diameter_profile_start_coord'] = profile1_end
        else:
            merged['diameter_profile_start_coord'] = profile1_start

        if reverse_profile2:
            merged['diameter_profile_end_coord'] = profile2_start
        else:
            merged['diameter_profile_end_coord'] = profile2_end
    else:
        # profile2 comes first
        if reverse_profile2:
            merged['diameter_profile_start_coord'] = profile2_end
        else:
            merged['diameter_profile_start_coord'] = profile2_start

        if reverse_profile1:
            merged['diameter_profile_end_coord'] = profile1_start
        else:
            merged['diameter_profile_end_coord'] = profile1_end

    # Fallback to voxel path coordinates if diameter profile coordinates not available
    if merged.get('diameter_profile_start_coord') is None and len(merged_voxels) > 0:
        merged['diameter_profile_start_coord'] = merged_voxels[0]
    if merged.get('diameter_profile_end_coord') is None and len(merged_voxels) > 0:
        merged['diameter_profile_end_coord'] = merged_voxels[-1]

    # Add path lengths together (they're linearly additive)
    if 'path_length_mm' in edge1_data and 'path_length_mm' in edge2_data:
        merged['path_length_mm'] = edge1_data['path_length_mm'] + edge2_data['path_length_mm']

    # Recalculate end-to-end distance
    if spacing is not None and len(merged_voxels) >= 2:
        start = np.array(merged_voxels[0]) * np.array(spacing)
        end = np.array(merged_voxels[-1]) * np.array(spacing)
        merged['end_to_end_distance_mm'] = np.linalg.norm(end - start)

    # Copy other attributes from edge1 (like edge_position, etc.)
    for key in edge1_data:
        if key not in merged:
            merged[key] = edge1_data[key]

    return merged


def remove_short_leaves_directed(directed_graph, origin_node, min_branch_length_voxels=None,
                                  min_branch_length_mm=None, spacing=None):
    """
    Removes short dead-end leaves from a directed graph and merges edges.

    This is a second-pass pruning that happens after the graph is directed.
    It removes short leaf branches and merges edges when intermediate nodes
    are removed, properly combining voxel paths and diameter profiles.

    Protects the origin node from removal.

    Args:
        directed_graph (nx.DiGraph): Directed graph with edge attributes
        origin_node (tuple): Origin node coordinate (protected from removal)
        min_branch_length_voxels (int, optional): Minimum branch length threshold in voxels
        min_branch_length_mm (float, optional): Minimum branch length threshold in mm
        spacing (tuple, optional): Voxel spacing (z, y, x) in mm

    Returns:
        nx.DiGraph: Cleaned directed graph with short leaves removed
    """
    # Calculate threshold
    min_voxels_threshold = None
    if min_branch_length_voxels is not None:
        min_voxels_threshold = min_branch_length_voxels
    elif min_branch_length_mm is not None and spacing is not None:
        avg_spacing = np.mean(spacing)
        min_voxels_threshold = int(min_branch_length_mm / avg_spacing)

    if min_voxels_threshold is None:
        return directed_graph

    # Find all leaf nodes (out_degree = 0, endpoints)
    leaf_nodes = [node for node in directed_graph.nodes() if directed_graph.out_degree(node) == 0]

    # Find short leaves (excluding those from origin)
    short_leaves = []
    for leaf in leaf_nodes:
        in_edges = list(directed_graph.in_edges(leaf))
        if len(in_edges) == 0:
            continue  # No parent (shouldn't happen)

        parent, _ = in_edges[0]
        edge_data = directed_graph[parent][leaf]
        voxel_length = len(edge_data.get('voxels', []))

        # Check if short and not directly from origin
        if voxel_length < min_voxels_threshold and parent != origin_node:
            short_leaves.append((leaf, parent, voxel_length))

    if len(short_leaves) == 0:
        return directed_graph

    print(f"      [Directed Graph] Found {len(short_leaves)} short leaves (< {min_voxels_threshold} voxels)")

    modified_graph = directed_graph.copy()
    edges_merged = 0
    leaves_removed = 0

    for leaf_node, parent_node, voxel_length in short_leaves:
        if leaf_node not in modified_graph.nodes():
            continue  # Already removed

        if not modified_graph.has_edge(parent_node, leaf_node):
            continue

        # Check if parent is a trifurcation (out_degree >= 3)
        parent_out_degree = modified_graph.out_degree(parent_node)

        if parent_out_degree >= 3:
            # Just remove edge and leaf, keep parent (trifurcation)
            modified_graph.remove_edge(parent_node, leaf_node)
            modified_graph.remove_node(leaf_node)
            leaves_removed += 1
        else:
            # Try to merge edges
            grandparent_edges = list(modified_graph.in_edges(parent_node))

            if len(grandparent_edges) == 0:
                # Parent is origin, just remove leaf
                modified_graph.remove_edge(parent_node, leaf_node)
                modified_graph.remove_node(leaf_node)
                leaves_removed += 1
                continue

            grandparent, _ = grandparent_edges[0]
            grandparent_edge_data = modified_graph[grandparent][parent_node]

            # Get other children of parent (not the leaf we're removing)
            other_children = [child for child in modified_graph.successors(parent_node)
                            if child != leaf_node]

            if len(other_children) != 1:
                # Parent has multiple children or no other children, just remove leaf
                modified_graph.remove_edge(parent_node, leaf_node)
                modified_graph.remove_node(leaf_node)
                leaves_removed += 1
                continue

            child = other_children[0]
            child_edge_data = modified_graph[parent_node][child]

            # Merge grandparent->parent and parent->child edges
            merged_edge_data = merge_edges(grandparent_edge_data, child_edge_data, spacing)

            # Remove old edges and nodes
            modified_graph.remove_edge(grandparent, parent_node)
            modified_graph.remove_edge(parent_node, child)
            modified_graph.remove_edge(parent_node, leaf_node)
            modified_graph.remove_node(parent_node)
            modified_graph.remove_node(leaf_node)

            # Add merged edge
            modified_graph.add_edge(grandparent, child, **merged_edge_data)
            edges_merged += 1
            leaves_removed += 1

    print(f"      [Directed Graph] Removed {leaves_removed} short leaves, merged {edges_merged} edge pairs")

    return modified_graph


def make_directed_graph(undirected_graph, origin_node, min_branch_length_voxels=None,
                       min_branch_length_mm=None, spacing=None):
    """
    Converts an undirected graph to a directed graph rooted at origin_node.

    Uses BFS traversal from the origin node to establish edge directions.
    Optionally performs a second pass of short leaf removal after direction is established.

    Args:
        undirected_graph (nx.Graph): Undirected graph to convert
        origin_node (tuple): Root node for the directed graph
        min_branch_length_voxels (int, optional): Threshold for second-pass leaf removal
        min_branch_length_mm (float, optional): Threshold for second-pass leaf removal (mm)
        spacing (tuple, optional): Voxel spacing (z, y, x) in mm

    Returns:
        nx.DiGraph: Directed graph rooted at origin_node
    """
    directed_graph = nx.DiGraph()

    for node, data in undirected_graph.nodes(data=True):
        directed_graph.add_node(node, **data)

    for u, v in nx.bfs_edges(undirected_graph, origin_node):
        edge_data = undirected_graph.get_edge_data(u, v, default={}).copy()
        directed_graph.add_edge(u, v, **edge_data)

    # Second pass: remove short leaves in directed graph
    if min_branch_length_voxels is not None or min_branch_length_mm is not None:
        directed_graph = remove_short_leaves_directed(
            directed_graph, origin_node,
            min_branch_length_voxels, min_branch_length_mm, spacing
        )

    return directed_graph



def detect_and_break_cycle(graph, spacing_info):
    """
    Detects small cycles (triangles) in the graph and removes the most perpendicular edge.

    Uses vector-based analysis to determine which edge in a cycle is most perpendicular
    to the vessel flow and should be removed.

    Args:
        graph (networkx.Graph): Graph that may contain cycles
        spacing_info (tuple): Voxel spacing (z, y, x) for computing direction vectors

    Returns:
        networkx.Graph: Graph with cycle edges removed
    """
    N = graph.number_of_nodes()
    E = graph.number_of_edges()

    if E <= N - 1:
        return graph  # No cycles

    # Find a cycle using DFS
    try:
        cycle = nx.find_cycle(graph, orientation='ignore')
    except nx.NetworkXNoCycle:
        return graph  # No cycles found

    # Extract cycle nodes and edges
    cycle_edges = [(u, v) for u, v, _ in cycle]
    cycle_nodes = set()
    for u, v in cycle_edges:
        cycle_nodes.add(u)
        cycle_nodes.add(v)

    print(f"      [Cycle Detection] Found cycle with {len(cycle_nodes)} nodes, {len(cycle_edges)} edges")

    # Compute direction vectors for cycle edges
    cycle_edge_vectors = {}
    for u, v in cycle_edges:
        voxels = graph[u][v]['voxels']
        if len(voxels) < 2:
            cycle_edge_vectors[(u, v)] = np.array([0, 0, 0])
            continue

        # Direction vector from first to last voxel
        start = np.array(voxels[0]) * np.array(spacing_info)
        end = np.array(voxels[-1]) * np.array(spacing_info)
        direction = end - start
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction = direction / norm
        cycle_edge_vectors[(u, v)] = direction

    # Compute average cycle edge voxel length
    avg_cycle_length = int(np.mean([len(graph[u][v]['voxels']) for u, v in cycle_edges]))

    # Compute vectors for non-cycle edges (sampled at avg_cycle_length)
    non_cycle_edge_vectors = {}
    for node in cycle_nodes:
        for neighbor in graph.neighbors(node):
            edge = (node, neighbor) if (node, neighbor) in cycle_edges else (neighbor, node)
            if edge in cycle_edges:
                continue  # Skip cycle edges

            # Get voxels, oriented away from cycle node
            voxels = graph[node][neighbor]['voxels']
            if voxels[0] != node:
                voxels = list(reversed(voxels))

            # Sample up to avg_cycle_length voxels
            sample_length = min(len(voxels), avg_cycle_length)
            sampled_voxels = voxels[:sample_length]

            if len(sampled_voxels) < 2:
                non_cycle_edge_vectors[(node, neighbor)] = np.array([0, 0, 0])
                continue

            # Direction vector
            start = np.array(sampled_voxels[0]) * np.array(spacing_info)
            end = np.array(sampled_voxels[-1]) * np.array(spacing_info)
            direction = end - start
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction = direction / norm
            non_cycle_edge_vectors[(node, neighbor)] = direction

    # For each cycle edge, compute alignment with connected non-cycle edges
    alignment_scores = {}
    for u, v in cycle_edges:
        cycle_vec = cycle_edge_vectors[(u, v)]

        # Find non-cycle edges connected to node u
        non_cycle_u = [
            (u, n) if (u, n) in non_cycle_edge_vectors else (n, u)
            for n in graph.neighbors(u)
            if (u, n) not in cycle_edges and (n, u) not in cycle_edges
        ]

        # Find non-cycle edges connected to node v
        non_cycle_v = [
            (v, n) if (v, n) in non_cycle_edge_vectors else (n, v)
            for n in graph.neighbors(v)
            if (v, n) not in cycle_edges and (n, v) not in cycle_edges
        ]

        # Compute dot products
        dot_products = []
        for edge in non_cycle_u:
            if edge in non_cycle_edge_vectors:
                dot = abs(np.dot(cycle_vec, non_cycle_edge_vectors[edge]))
                dot_products.append(dot)

        for edge in non_cycle_v:
            if edge in non_cycle_edge_vectors:
                dot = abs(np.dot(cycle_vec, non_cycle_edge_vectors[edge]))
                dot_products.append(dot)

        alignment_scores[(u, v)] = sum(dot_products)
        print(f"      [Cycle] Edge {u}→{v}: alignment score = {sum(dot_products):.3f} ({len(dot_products)} dot products)")

    # Remove edge with lowest alignment (most perpendicular)
    if alignment_scores:
        edge_to_remove = min(alignment_scores, key=alignment_scores.get)
        graph_copy = graph.copy()
        graph_copy.remove_edge(*edge_to_remove)
        print(f"      [Cycle] Removing most perpendicular edge: {edge_to_remove[0]}→{edge_to_remove[1]}")
        return graph_copy

    return graph


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
