import nrrd
import json
import yaml
import pickle
import networkx as nx
import ast
from pathlib import Path

def load_nrrd_mask(path, verbose=False):
    data, header = nrrd.read(path)

    if verbose:
        print("Shape:", data.shape)
        print("Available Header Info:")
        for heading, value in header.items():
            print(f'{heading}: {value}')

    return data, header


def load_config(config_path):
    """
    Load a configuration file for pipeline parameters.

    Supports both JSON (.json) and YAML (.yaml, .yml) formats.
    File format is auto-detected based on extension.

    Args:
        config_path (str): Path to the config file (.json, .yaml, or .yml)

    Returns:
        dict: Dictionary containing configuration parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If file extension is not supported
        json.JSONDecodeError: If JSON file is not valid
        yaml.YAMLError: If YAML file is not valid
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    ext = path.suffix.lower()

    with open(config_path, 'r') as f:
        if ext == '.json':
            config = json.load(f)
        elif ext in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        else:
            raise ValueError(
                f"Unsupported config file extension: {ext}. "
                f"Use .json, .yaml, or .yml"
            )

    return config


def save_graph(graph, path):
    """
    Save a NetworkX graph to GraphML format.

    Lists and dictionaries in node/edge attributes will be automatically
    converted to strings by NetworkX. Use load_graph() to restore them.

    Args:
        graph (networkx.Graph): The graph to save
        path (str): Output path for .graphml file
    """
    nx.write_graphml(graph, path)


def load_graph(path, convert_numeric_strings=True):
    """
    Load a NetworkX graph from GraphML format.

    GraphML stores complex types (lists, dicts) as strings. This function
    attempts to convert string representations back to their original types.

    Args:
        path (str): Path to the .graphml file
        convert_numeric_strings (bool): If True, attempts to convert string
                                       representations of lists/dicts back to
                                       Python objects using ast.literal_eval

    Returns:
        networkx.Graph: The loaded graph with attributes restored
    """
    graph = nx.read_graphml(path)

    if convert_numeric_strings:
        # Convert string representations of lists/dicts back to Python objects
        # for both node and edge attributes
        for node in graph.nodes():
            for attr_name, attr_value in graph.nodes[node].items():
                if isinstance(attr_value, str):
                    graph.nodes[node][attr_name] = _try_convert_string(attr_value)

        for u, v in graph.edges():
            for attr_name, attr_value in graph[u][v].items():
                if isinstance(attr_value, str):
                    graph[u][v][attr_name] = _try_convert_string(attr_value)

    return graph


def _try_convert_string(value):
    """
    Helper function to safely convert string representations of Python objects.

    Attempts to use ast.literal_eval to convert strings that look like
    lists, dicts, tuples, numbers, etc. Falls back to original string if conversion fails.

    Args:
        value (str): String value to convert

    Returns:
        Original Python object if conversion succeeds, otherwise the original string
    """
    if not isinstance(value, str):
        return value

    # Skip conversion if it's just a regular string (doesn't start with [, {, (, or numeric)
    if not (value.startswith('[') or value.startswith('{') or value.startswith('(')
            or value.replace('.', '').replace('-', '').replace('e', '').replace('+', '').isdigit()):
        return value

    try:
        # ast.literal_eval safely evaluates strings containing Python literals
        # (lists, dicts, tuples, numbers, strings, booleans, None)
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # If conversion fails, return original string
        return value


def save_artery_analysis(path, final_graph, sparse_graph=None, binary_mask=None,
                         distance_array=None, spacing_info=None, nrrd_header=None,
                         processing_times=None, total_time=None, metadata=None):
    """
    Save complete artery analysis results to a pickle file.

    Creates a comprehensive dictionary containing all analysis outputs, masks, and metadata.
    This allows complete reconstruction of the analysis for later statistical work.

    Args:
        path (str): Output path for the .pkl file
        final_graph (networkx.DiGraph): Final directed graph with all computed metrics
        sparse_graph (networkx.Graph, optional): Undirected graph before orientation
        binary_mask (numpy.ndarray, optional): Binary 3D mask of the vessel
        distance_array (numpy.ndarray, optional): Distance transform array
        spacing_info (tuple, optional): Voxel spacing in mm (z, y, x)
        nrrd_header (dict, optional): Original NRRD header information
        processing_times (dict, optional): Dictionary of processing times for each step
        total_time (float, optional): Total processing time in seconds
        metadata (dict, optional): Any additional metadata to store

    Example:
        save_artery_analysis(
            'results/patient1_LCA.pkl',
            final_graph=result['final_graph'],
            sparse_graph=result['sparse_graph'],
            binary_mask=mask,
            distance_array=dist_array,
            spacing_info=(0.5, 0.5, 0.5),
            nrrd_header=header,
            processing_times=result['processing_times'],
            total_time=result['total_time'],
            metadata={'patient_id': 'patient1', 'artery': 'LCA'}
        )
    """
    analysis_data = {
        'final_graph': final_graph,
        'sparse_graph': sparse_graph,
        'binary_mask': binary_mask,
        'distance_array': distance_array,
        'spacing_info': spacing_info,
        'nrrd_header': nrrd_header,
        'processing_times': processing_times,
        'total_time': total_time,
        'metadata': metadata if metadata is not None else {}
    }

    with open(path, 'wb') as f:
        pickle.dump(analysis_data, f, pickle.HIGHEST_PROTOCOL)


def load_artery_analysis(path):
    """
    Load complete artery analysis results from a pickle file.

    Reconstructs the full analysis state including graphs, masks, spacing info, and metadata.

    Args:
        path (str): Path to the .pkl file

    Returns:
        dict: Dictionary containing all saved analysis data:
            - 'final_graph': NetworkX DiGraph with all computed metrics
            - 'sparse_graph': Undirected graph before orientation (if saved)
            - 'binary_mask': Binary 3D mask of the vessel (if saved)
            - 'distance_array': Distance transform array (if saved)
            - 'spacing_info': Voxel spacing in mm (if saved)
            - 'nrrd_header': Original NRRD header information (if saved)
            - 'processing_times': Dict of processing times (if saved)
            - 'total_time': Total processing time (if saved)
            - 'metadata': Additional metadata (if saved)

    Example:
        data = load_artery_analysis('results/patient1_LCA.pkl')
        graph = data['final_graph']
        spacing = data['spacing_info']

        # Perform statistical analysis
        for u, v in graph.edges():
            diameter_profile = graph[u][v]['diameter_profile_slicing']
            length = graph[u][v]['length']
            # ... analyze ...
    """
    with open(path, 'rb') as f:
        analysis_data = pickle.load(f)

    return analysis_data
