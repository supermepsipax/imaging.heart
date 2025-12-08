import nrrd
import json
import yaml
import pickle
import gzip
import bz2
import lzma
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


def save_artery_analysis_compressed(path, final_graph, sparse_graph=None, binary_mask=None,
                                    distance_array=None, spacing_info=None, nrrd_header=None,
                                    processing_times=None, total_time=None, metadata=None,
                                    compression='gzip'):
    """
    Save complete artery analysis results to a compressed pickle file.

    More space-efficient than save_artery_analysis(). Supports gzip, bz2, or lzma compression.
    Typical compression ratios: 5-10x smaller for graph data, 10-50x for masks/arrays.

    Args:
        path (str): Output path for the compressed .pkl file (e.g., 'result.pkl.gz')
        final_graph (networkx.DiGraph): Final directed graph with all computed metrics
        sparse_graph (networkx.Graph, optional): Undirected graph before orientation
        binary_mask (numpy.ndarray, optional): Binary 3D mask of the vessel
        distance_array (numpy.ndarray, optional): Distance transform array
        spacing_info (tuple, optional): Voxel spacing in mm (z, y, x)
        nrrd_header (dict, optional): Original NRRD header information
        processing_times (dict, optional): Dictionary of processing times for each step
        total_time (float, optional): Total processing time in seconds
        metadata (dict, optional): Any additional metadata to store
        compression (str): Compression method - 'gzip' (fast, good ratio), 'bz2' (slower, better ratio),
                          or 'lzma' (slowest, best ratio). Default: 'gzip'

    Example:
        save_artery_analysis_compressed(
            'results/patient1_LCA.pkl.gz',
            final_graph=result['final_graph'],
            binary_mask=mask,
            spacing_info=(0.5, 0.5, 0.5),
            compression='gzip'  # ~5-10x size reduction
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

    # Select compression module
    if compression == 'gzip':
        compressor = gzip
    elif compression == 'bz2':
        compressor = bz2
    elif compression == 'lzma':
        compressor = lzma
    else:
        raise ValueError(f"Unknown compression: {compression}. Use 'gzip', 'bz2', or 'lzma'")

    with compressor.open(path, 'wb') as f:
        pickle.dump(analysis_data, f, pickle.HIGHEST_PROTOCOL)


def load_artery_analysis_compressed(path):
    """
    Load complete artery analysis results from a compressed pickle file.

    Auto-detects compression format from file extension (.gz, .bz2, .xz).

    Args:
        path (str): Path to the compressed .pkl file

    Returns:
        dict: Dictionary containing all saved analysis data

    Example:
        data = load_artery_analysis_compressed('results/patient1_LCA.pkl.gz')
        graph = data['final_graph']
    """
    path = Path(path)
    suffix = path.suffix.lower()

    # Auto-detect compression from extension
    if suffix == '.gz':
        compressor = gzip
    elif suffix == '.bz2':
        compressor = bz2
    elif suffix in ['.xz', '.lzma']:
        compressor = lzma
    else:
        # Try as uncompressed pickle
        return load_artery_analysis(str(path))

    with compressor.open(path, 'rb') as f:
        analysis_data = pickle.load(f)

    return analysis_data


def merge_artery_analyses(input_folder, output_file, pattern='*.pkl', compression='gzip'):
    """
    Merge multiple individual artery analysis pickle files into one compressed file.

    Preserves ALL data from original files - merged files are identical to originals
    when extracted. Compression reduces storage space without losing information.

    Args:
        input_folder (str): Folder containing individual .pkl files
        output_file (str): Path for merged output file (e.g., 'all_results.pkl.gz')
        pattern (str): Glob pattern to match files (default: '*.pkl')
        compression (str): Compression method - 'gzip', 'bz2', or 'lzma' (default: 'gzip')

    Returns:
        dict: Statistics about the merge operation:
            - 'files_merged': Number of files merged
            - 'total_vessels': Number of vessels (LCA + RCA)
            - 'output_size_mb': Size of output file in MB

    Structure of merged file:
        {
            'filename1': {
                'LCA': {analysis_data...},  # Complete original data
                'RCA': {analysis_data...}   # Complete original data
            },
            'filename2': {
                'LCA': {analysis_data...},
                'RCA': {analysis_data...}
            },
            ...
        }

    Example:
        # Merge all results with gzip compression
        stats = merge_artery_analyses(
            'results/batch_1',
            'results/batch_1_merged.pkl.gz',
            pattern='*_analysis.pkl',
            compression='gzip'
        )
        print(f"Merged {stats['files_merged']} files, {stats['total_vessels']} vessels")
        print(f"Output size: {stats['output_size_mb']:.1f} MB")

        # Later: load and analyze
        merged_data = load_artery_analysis_compressed('results/batch_1_merged.pkl.gz')
        for filename, vessels in merged_data.items():
            lca_graph = vessels['LCA']['final_graph']
            # Data is identical to original individual files
    """
    input_path = Path(input_folder)
    pkl_files = sorted(input_path.glob(pattern))

    if not pkl_files:
        raise ValueError(f"No files found matching pattern '{pattern}' in {input_folder}")

    merged_data = {}
    files_merged = 0
    total_vessels = 0

    print(f"Merging {len(pkl_files)} files from {input_folder}...")

    for pkl_file in pkl_files:
        # Parse filename to extract case name and vessel type
        # Expected format: casename_VesselType_*.pkl (e.g., Normal_8_LCA_analysis.pkl)
        stem = pkl_file.stem
        parts = stem.split('_')

        # Try to extract vessel type (LCA or RCA)
        vessel_type = None
        case_name = None

        if 'LCA' in parts:
            vessel_type = 'LCA'
            case_name = '_'.join(parts[:parts.index('LCA')])
        elif 'RCA' in parts:
            vessel_type = 'RCA'
            case_name = '_'.join(parts[:parts.index('RCA')])
        else:
            # Fallback: use full stem as case name
            case_name = stem
            vessel_type = 'unknown'

        # Load analysis data (preserve ALL data exactly as is)
        try:
            analysis_data = load_artery_analysis(str(pkl_file))

            # Add to merged structure
            if case_name not in merged_data:
                merged_data[case_name] = {}

            merged_data[case_name][vessel_type] = analysis_data

            files_merged += 1
            total_vessels += 1

            print(f"  ✓ {pkl_file.name} -> {case_name}/{vessel_type}")

        except Exception as e:
            print(f"  ✗ Failed to load {pkl_file.name}: {e}")

    # Save merged data with compression
    print(f"\nSaving merged data to {output_file}...")

    if compression == 'gzip':
        compressor = gzip
    elif compression == 'bz2':
        compressor = bz2
    elif compression == 'lzma':
        compressor = lzma
    else:
        raise ValueError(f"Unknown compression: {compression}")

    with compressor.open(output_file, 'wb') as f:
        pickle.dump(merged_data, f, pickle.HIGHEST_PROTOCOL)

    # Get output file size
    output_size_mb = Path(output_file).stat().st_size / (1024 * 1024)

    print(f"✓ Merge complete!")
    print(f"  Files merged: {files_merged}")
    print(f"  Total vessels: {total_vessels}")
    print(f"  Output size: {output_size_mb:.1f} MB")

    return {
        'files_merged': files_merged,
        'total_vessels': total_vessels,
        'output_size_mb': output_size_mb
    }
