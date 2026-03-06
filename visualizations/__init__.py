"""
Visualizations package for IMG-Heart project.

This package provides visualization functions for:
- Creating 2D projections of 3D masks
- Visualizing 3D graphs with vessel structures
- Troubleshooting bifurcation angle calculations
"""

from .visualize_2d import create_projection_view
from .visualize_3d import visualize_3d_graph, visualize_binary_mask, visualize_mask_overlap, get_edge_hierarchy_color, get_anatomical_branch_color
from .visualize_clusters import (
    plot_cluster_3d,
    plot_cluster_projections_2d,
    plot_summary_bar_charts,
    plot_distance_box_plots,
    plot_pairwise_heatmaps,
    plot_ground_truth_comparison,
)

__all__ = [
    'create_projection_view',
    'visualize_3d_graph',
    'visualize_binary_mask',
    'visualize_mask_overlap',
    'get_edge_hierarchy_color',
    'get_anatomical_branch_color',
    'plot_cluster_3d',
    'plot_cluster_projections_2d',
    'plot_summary_bar_charts',
    'plot_distance_box_plots',
    'plot_pairwise_heatmaps',
    'plot_ground_truth_comparison',
]
