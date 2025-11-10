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
