"""
Clustering package for node-level sensitivity analysis across segmentation models.

Provides utilities for classifying graph nodes by type, clustering corresponding
nodes across models using DBSCAN, and computing ground truth reference distances.
"""

from .node_classification import classify_and_extract_nodes
from .node_matching import (
    compute_origin_spread,
    cluster_nodes,
    compute_ground_truth_distances,
    compute_pairwise_model_distances,
)

__all__ = [
    'classify_and_extract_nodes',
    'compute_origin_spread',
    'cluster_nodes',
    'compute_ground_truth_distances',
    'compute_pairwise_model_distances',
]
