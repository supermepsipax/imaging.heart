import nrrd
import json
import yaml
import pickle
import tarfile
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


def archive_artery_analyses(input_folder, output_file, pattern='*.pkl', compression='gz'):
    """
    Create a compressed tar archive of artery analysis pickle files.

    Streams files into archive without loading all into memory - handles any dataset size.
    Works cross-platform (Windows/Mac/Linux) via Python's tarfile module.

    Args:
        input_folder (str): Folder containing individual .pkl files
        output_file (str): Path for tar archive (e.g., 'results.tar.gz')
        pattern (str): Glob pattern to match files (default: '*.pkl')
        compression (str): Compression mode - 'gz' (gzip), 'bz2', or 'xz' (default: 'gz')

    Returns:
        dict: Statistics about the archive operation:
            - 'files_archived': Number of files added
            - 'output_size_mb': Size of archive in MB

    Example:
        # Create gzip compressed archive (recommended)
        stats = archive_artery_analyses(
            'results/batch_1',
            'results/batch_1.tar.gz',
            pattern='*_analysis.pkl',
            compression='gz'
        )
        print(f"Archived {stats['files_archived']} files, {stats['output_size_mb']:.1f} MB")

        # Later: stream analysis without unpacking (see analyze_artery_batch)
        results = analyze_artery_batch(input_tar_file='results/batch_1.tar.gz')
    """
    input_path = Path(input_folder)
    pkl_files = sorted(input_path.glob(pattern))

    if not pkl_files:
        raise ValueError(f"No files found matching pattern '{pattern}' in {input_folder}")

    print(f"Creating archive from {len(pkl_files)} files in {input_folder}...")

    # Map compression to tarfile mode
    mode = f'w:{compression}'

    files_archived = 0

    with tarfile.open(output_file, mode) as tar:
        for pkl_file in pkl_files:
            print(f"  Adding {pkl_file.name}...")
            tar.add(pkl_file, arcname=pkl_file.name)
            files_archived += 1

    # Get output file size
    output_size_mb = Path(output_file).stat().st_size / (1024 * 1024)

    print(f"✓ Archive complete!")
    print(f"  Files archived: {files_archived}")
    print(f"  Output size: {output_size_mb:.1f} MB")

    return {
        'files_archived': files_archived,
        'output_size_mb': output_size_mb
    }
