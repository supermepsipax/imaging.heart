"""
Utilities package for IMG-Heart project.

This package provides utility functions for:
- Input/Output operations (NRRD file handling)
- Centerline utilities
- Graph construction and analysis
- Pre and post-processing for skeleton topology
- Bifurcation angle computation and vessel geometry analysis
"""

from .input_output import load_nrrd_mask
from .centerline_utils import ensure_continuous_body, extract_centerline_skimage
from .bifurcation_utils import extract_endpoint_and_bifurcation_coordinates, remove_redundant_bifurcation_clusters, remove_sharp_bend_bifurcations
from .graph_utils import skeleton_to_sparse_graph, skeleton_to_sparse_graph_robust, find_connected_voxels, skeleton_to_dense_graph, dense_graph_to_skeleton, make_directed_graph 
from .preprocessing_utils import preprocess_binary_mask, sort_labelled_bodies_by_size,resample_to_isotropic
from .diameter_utils import create_distance_transform_from_mask, compute_average_diameter_of_branch, compute_branch_diameters_of_graph, determine_origin_node_from_diameter,diameter_profile,summarize_profile
from .distance_utils import compute_branch_path_length, compute_branch_lengths_of_graph
from .trigonometric_utils import move_along_centerline, fit_bifurcation_plane, compute_inflow_angle, compute_bifurcation_angles, compute_angles_at_bifurcation, traverse_graph_and_compute_angles

__all__ = [
    'load_nrrd_mask',
    'ensure_continuous_body',
    'extract_centerline_skimage',
    'extract_endpoint_and_bifurcation_coordinates',
    'remove_redundant_bifurcation_clusters',
    'remove_sharp_bend_bifurcations',
    'skeleton_to_sparse_graph',
    'skeleton_to_sparse_graph_robust',
    'find_connected_voxels',
    'skeleton_to_dense_graph',
    'dense_graph_to_skeleton',
    'make_directed_graph',
    'preprocess_binary_mask',
    'sort_labelled_bodies_by_size',
    'resample_to_isotropic',
    'create_distance_transform_from_mask',
    'compute_average_diameter_of_branch',
    'compute_branch_diameters_of_graph',
    'compute_branch_path_length',
    'compute_branch_lengths_of_graph',
    'determine_origin_node_from_diameter',
    'diameter_profile',
    'summarize_profile',
    'move_along_centerline',
    'fit_bifurcation_plane',
    'compute_inflow_angle',
    'compute_bifurcation_angles',
    'compute_angles_at_bifurcation',
    'traverse_graph_and_compute_angles',
]