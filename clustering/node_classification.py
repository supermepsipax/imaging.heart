import numpy as np


def classify_and_extract_nodes(graph, spacing_info):
    """
    Classify nodes in a directed artery graph by type and convert to physical coordinates.

    Node types are determined by degree in the DiGraph:
    - origin: in_degree == 0 (root of the tree, always exactly 1)
    - bifurcation: out_degree >= 2 (branching points)
    - endpoint: out_degree == 0 (terminal nodes)

    Args:
        graph (networkx.DiGraph): Directed artery graph with nodes as (dim0, dim1, dim2) voxel tuples
        spacing_info (tuple): Voxel spacing in mm as (z, y, x)

    Returns:
        dict: Keys 'origin', 'bifurcation', 'endpoint', each mapping to an np.ndarray
              of shape (N, 3) with physical coordinates in mm. Empty arrays have shape (0, 3).
    """
    spacing = np.array(spacing_info)

    origins = []
    bifurcations = []
    endpoints = []

    for node in graph.nodes():
        in_deg = graph.in_degree(node)
        out_deg = graph.out_degree(node)

        if in_deg == 0:
            origins.append(np.array(node) * spacing)
        elif out_deg >= 2:
            bifurcations.append(np.array(node) * spacing)
        elif out_deg == 0:
            endpoints.append(np.array(node) * spacing)

    return {
        'origin': np.array(origins) if origins else np.empty((0, 3)),
        'bifurcation': np.array(bifurcations) if bifurcations else np.empty((0, 3)),
        'endpoint': np.array(endpoints) if endpoints else np.empty((0, 3)),
    }
