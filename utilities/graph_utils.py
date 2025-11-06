import numpy as np
import networkx as nx


def find_connected_voxels(coordinate, binary_mask):
    """
    Given a single coordinate within a 3D space, will find and return
    all "touching" voxel coordinates. This assumes a voxel is touching
    if it is non-zero in any of the 26 surrounding positions.

    Args:
        coordinate (tuple): The coordinate of the voxel in x,y,z format
        binary_mask (array): A numpy binary mask

    Returns:
        neighbor_voxels: A list of coordinates (if any) of touching voxels

    """

    x, y, z = coordinate
    neighbor_voxels = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                nx_, ny_, nz_ = x + dx, y + dy, z + dz
                if (
                    0 <= nx_ < binary_mask.shape[0]
                    and 0 <= ny_ < binary_mask.shape[1]
                    and 0 <= nz_ < binary_mask.shape[2]
                    and binary_mask[nx_, ny_, nz_] == 1
                ):
                    neighbor_voxels.append((nx_, ny_, nz_))
    return neighbor_voxels


def build_graph(binary_mask, bifurcation_points, endpoints):
    """
    Builds and returns a graph representation of the arteries using networkx.

    Combines the list of bifurcation and endpoint coordinates together and adds
    them as nodes in the created graph. Then paths are traced outwardly from nodes
    via connected voxels to find and create the edges between nodes. The exact voxel
    path for each edge is logged and stored as well. Ideally this should be done on a mask
    that is known to be a single continous body.

    Args:
        binary_mask (array): A numpy binary mask
        bifuraction_points (list): A list of the 3D coordinates of all bifurcations in the binary mask
        endpoints (list): A list of the 3D coordinates of all endpoints in the binary mask

    Returns:
        graph: A networkx graph object 

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
                    # end point
                    break
                elif len(connected_voxels) > 1:
                    # unexpected bifurcation not marked
                    break
                previous_voxel, current_voxel = current_voxel, connected_voxels[0]

            if current_voxel in nodes:
                path.append(current_voxel)
                graph.add_edge(node_coordinate, current_voxel, voxels=path)

    return graph
