import nrrd
import json
import yaml
import pickle
import tarfile
import numpy as np
import re
import SimpleITK as sitk
from pathlib import Path


SUPPORTED_EXTENSIONS = ('.nrrd', '.nii', '.nii.gz')


def _is_nifti(path):
    name = Path(path).name.lower()
    return name.endswith('.nii.gz') or name.endswith('.nii')


def load_nrrd_mask(path, verbose=False):
    data, header = nrrd.read(path)

    if verbose:
        print("Shape:", data.shape)
        print("Available Header Info:")
        for heading, value in header.items():
            print(f'{heading}: {value}')

    return data, header


def load_nifti_mask(path, verbose=False):
    img = sitk.ReadImage(str(path))
    data = sitk.GetArrayFromImage(img)

    if verbose:
        print("Shape:", data.shape)
        print("Spacing:", img.GetSpacing())
        print("Origin:", img.GetOrigin())
        print("Direction:", img.GetDirection())

    return data, img


def load_mask(path, verbose=False):
    if _is_nifti(path):
        data, sitk_img = load_nifti_mask(path, verbose=verbose)
        header = sitk_header_to_nrrd(sitk_img)
        return data, header
    return load_nrrd_mask(path, verbose=verbose)


def sitk_header_to_nrrd(sitk_img):
    spacing = sitk_img.GetSpacing()
    origin = sitk_img.GetOrigin()
    direction = sitk_img.GetDirection()
    dim = sitk_img.GetDimension()
    dir_matrix = np.array(direction).reshape(dim, dim)
    header = {
        'space': 'left-posterior-superior',
        'space directions': (dir_matrix * np.array(spacing)).T.tolist(),
        'space origin': list(origin),
    }
    return header


def glob_masks(folder):
    folder = Path(folder)
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(folder.glob(f'*{ext}'))
    return sorted(set(files))

def strip_filenames(paths, verbose=False):
    """
    Takes a list of nrrd filenames and strips them of the
    extra trailing words, ideally will result in a final output
    where filepaths look like Normal_1.nrrd or Diseased_2.nrrd
    """
    stripped_paths = []

    pattern = re.compile(r"(Normal|Diseased)_\d+", re.IGNORECASE)

    for path in paths:
        path_obj = Path(path)
        suffixes = path_obj.suffixes
        suffix = "".join(suffixes) if suffixes else ""
        base_name = path_obj.name[:-len(suffix)] if suffix else path_obj.name

        match = pattern.search(base_name)
        if match:
            stripped_name = match.group(0)
        else:
            parts = base_name.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                stripped_name = f"{parts[0]}_{parts[1]}"
            else:
                stripped_name = base_name

        new_name = f"{stripped_name}{suffix}"
        new_path = path_obj.with_name(new_name)
        stripped_paths.append(str(new_path))

        if verbose:
            print(f"{path_obj} -> {new_path}")

    return stripped_paths 


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


def extract_anatomical_info(nrrd_header):
    """
    Extract and parse anatomical orientation information from NRRD header.

    Parses the coordinate system (space) and determines which anatomical
    direction each array axis corresponds to. Handles common medical imaging
    coordinate systems.

    Args:
        nrrd_header (dict): NRRD header dictionary from nrrd.read()

    Returns:
        dict: Dictionary containing:
            - 'space': Coordinate system name (e.g., 'left-posterior-superior')
            - 'space_directions': 3x3 matrix of direction vectors (if available)
            - 'space_origin': Origin point in world coordinates (if available)
            - 'axis_labels': List of anatomical labels for each axis (e.g., ['left-right', ...])
            - 'axis_directions': List of positive direction for each axis (e.g., ['left', 'posterior', 'superior'])
            - 'spacings': Voxel spacing in mm for each axis
            - 'is_axis_aligned': True if space_directions is diagonal (axes align with anatomy)

    Common coordinate systems:
        - 'left-posterior-superior' (LPS): Used in DICOM, ITK
        - 'right-anterior-superior' (RAS): Used in NIfTI, FreeSurfer
        - 'left-anterior-superior' (LAS): Less common variant

    Example:
        >>> data, header = nrrd.read('scan.nrrd')
        >>> info = extract_anatomical_info(header)
        >>> print(info['axis_labels'])
        ['left-right', 'posterior-anterior', 'superior-inferior']
        >>> print(info['axis_directions'])
        ['left', 'posterior', 'superior']
    """
    anatomical_info = {}

    # Extract space coordinate system
    if 'space' in nrrd_header:
        space = nrrd_header['space']
        anatomical_info['space'] = space

        # Parse space string to determine axis labels
        # Format: "direction0-direction1-direction2"
        space_lower = space.lower()

        # Map of coordinate system to axis information
        coordinate_systems = {
            'left-posterior-superior': {
                'axis_directions': ['left', 'posterior', 'superior'],
                'axis_labels': ['left-right', 'posterior-anterior', 'superior-inferior']
            },
            'right-anterior-superior': {
                'axis_directions': ['right', 'anterior', 'superior'],
                'axis_labels': ['right-left', 'anterior-posterior', 'superior-inferior']
            },
            'left-anterior-superior': {
                'axis_directions': ['left', 'anterior', 'superior'],
                'axis_labels': ['left-right', 'anterior-posterior', 'superior-inferior']
            },
            'right-posterior-superior': {
                'axis_directions': ['right', 'posterior', 'superior'],
                'axis_labels': ['right-left', 'posterior-anterior', 'superior-inferior']
            },
            'scanner-xyz': {
                'axis_directions': ['x', 'y', 'z'],
                'axis_labels': ['x', 'y', 'z']
            }
        }

        if space_lower in coordinate_systems:
            anatomical_info['axis_directions'] = coordinate_systems[space_lower]['axis_directions']
            anatomical_info['axis_labels'] = coordinate_systems[space_lower]['axis_labels']
        else:
            # For unknown coordinate systems, try to parse from the space string
            parts = space_lower.split('-')
            if len(parts) == 3:
                anatomical_info['axis_directions'] = parts
                anatomical_info['axis_labels'] = [f"{parts[i]}-{_get_opposite_direction(parts[i])}"
                                                   for i in range(3)]
            else:
                anatomical_info['axis_directions'] = ['unknown', 'unknown', 'unknown']
                anatomical_info['axis_labels'] = ['unknown', 'unknown', 'unknown']

    # Extract space directions matrix
    if 'space directions' in nrrd_header:
        space_directions = np.array(nrrd_header['space directions'])
        anatomical_info['space_directions'] = space_directions

        # Check if axis-aligned (diagonal matrix)
        # An axis-aligned matrix has non-zero values only on the diagonal
        is_diagonal = True
        spacings = []

        for i in range(min(3, space_directions.shape[0])):
            for j in range(min(3, space_directions.shape[1])):
                if i == j:
                    # Diagonal element - should be non-zero
                    spacings.append(np.abs(space_directions[i, j]))
                else:
                    # Off-diagonal element - should be near zero for axis-aligned
                    if np.abs(space_directions[i, j]) > 1e-6:
                        is_diagonal = False

        anatomical_info['is_axis_aligned'] = is_diagonal
        anatomical_info['spacings'] = spacings

    # Extract space origin
    if 'space origin' in nrrd_header:
        anatomical_info['space_origin'] = np.array(nrrd_header['space origin'])

    # If space directions not available, try to get spacing from 'spacings' field
    if 'spacings' not in anatomical_info and 'spacings' in nrrd_header:
        anatomical_info['spacings'] = list(nrrd_header['spacings'])

    return anatomical_info


def _get_opposite_direction(direction):
    """
    Helper function to get the opposite anatomical direction.

    Args:
        direction (str): Anatomical direction (e.g., 'left', 'anterior')

    Returns:
        str: Opposite direction (e.g., 'right', 'posterior')
    """
    opposites = {
        'left': 'right',
        'right': 'left',
        'anterior': 'posterior',
        'posterior': 'anterior',
        'superior': 'inferior',
        'inferior': 'superior',
        'x': 'x',
        'y': 'y',
        'z': 'z'
    }
    return opposites.get(direction.lower(), 'unknown')


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
