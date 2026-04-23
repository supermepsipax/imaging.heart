"""
Comparison package for graph-level structural comparison of artery models.

Provides graph traversal-based comparison that matches branching topology
between a ground truth and model graph, reporting branch-level metrics.
"""

from .graph_traversal import compare_graphs_by_traversal

__all__ = [
    'compare_graphs_by_traversal',
]
