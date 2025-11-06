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
                nx_, ny_, nz_ = x+dx, y+dy, z+dz
                if (
                    0 <= nx_ < binary_mask.shape[0]
                    and 0 <= ny_ < binary_mask.shape[1]
                    and 0 <= nz_ < binary_mask.shape[2]
                    and binary_mask[nx_, ny_, nz_] == 1
                ):
                    neighbor_voxels.append((nx_, ny_, nz_))
    return neighbor_voxels

def build_graph(binary_mask, bifurcation_points, endpoints):

    graph = nx.Graph()
    nodes = set (bifurcation_points + endpoints)
    visited_voxels = set()

    graph.add_nodes_from(nodes)

