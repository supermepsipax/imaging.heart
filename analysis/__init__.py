"""
Analysis package for IMG-Heart project.

This package provides analysis functions for:
- Graph to DataFrame conversion for data analysis and export
"""

from .dataframe_conversion import convert_graph_to_dataframes

__all__ = [
    'convert_graph_to_dataframes',
]
