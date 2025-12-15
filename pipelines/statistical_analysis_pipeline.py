import os
import io
import pickle
import tarfile
import numpy as np
from pathlib import Path
from utilities import load_artery_analysis, load_config
from analysis import (
    extract_main_branch_statistics,
    extract_all_branch_statistics,
    extract_bifurcation_statistics,
    extract_trifurcation_statistics,
    compute_branch_tapering
)
from collections import defaultdict
from scipy.stats import ttest_ind
import pprint
import csv


def analyze_artery_batch(input_folder=None, input_tar_file=None,
                         output_folder=None, config=None, config_path=None,
                         diameter_method=None, verbose=None):
    """
    Perform statistical analysis on a batch of artery analysis pickle files.

    This pipeline loads pre-processed artery analysis results and computes
    statistical metrics across multiple patients/samples.

    Supports two input modes:
    1. Individual files: Load from folder containing individual .pkl files
    2. Tar archive: Stream from tar.gz archive WITHOUT unpacking (memory-efficient)

    Args:
        input_folder (str, optional): Path to folder containing .pkl analysis files
        input_tar_file (str, optional): Path to tar archive (e.g., results.tar.gz) - streams without unpacking
        output_folder (str, optional): Path to folder where analysis results will be saved
        config (dict, optional): Configuration dictionary with analysis parameters
        config_path (str, optional): Path to config file (if config not provided directly)
        diameter_method (str, optional): 'slicing' or 'edt' - which diameter measurements to use
        verbose (bool, optional): Whether to print detailed statistics for each artery

    Returns:
        dict: Summary statistics and results from the batch analysis

    Note:
        Only one of input_folder or input_tar_file should be provided.
    """

    if config is None and config_path is not None:
        config = load_config(config_path)

    if config is None:
        config = {}

    # Load input source from config if not provided
    if input_folder is None:
        input_folder = config.get('input_folder')
    if input_tar_file is None:
        input_tar_file = config.get('input_tar_file')
    if output_folder is None:
        output_folder = config.get('output_folder')
    if diameter_method is None:
        diameter_method = config.get('diameter_method', 'slicing')
    if verbose is None:
        verbose = config.get('verbose', True)

    # Validate input source
    if input_folder is None and input_tar_file is None:
        raise ValueError("Must provide either input_folder or input_tar_file")

    if input_folder is not None and input_tar_file is not None:
        raise ValueError("Cannot specify both input_folder and input_tar_file - use only one")

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)

    # Determine input mode and load data
    if input_tar_file is not None:
        # Mode 1: Stream from tar archive (memory-efficient)
        print("=" * 80)
        print("STATISTICAL ANALYSIS PIPELINE (TAR ARCHIVE MODE)")
        print("=" * 80)
        print(f"Input tar file: {input_tar_file}")
        if output_folder:
            print(f"Output folder: {output_folder}")
        print(f"\nConfiguration:")
        print(f"  Diameter method: {diameter_method}")
        print(f"  Verbose output: {verbose}")
        print("=" * 80)
        print("\nStreaming from archive...")

        try:
            # Open tar file and get list of pickle files
            with tarfile.open(input_tar_file, 'r:*') as tar:
                # Filter for .pkl files
                pkl_members = [m for m in tar.getmembers() if m.name.endswith('.pkl') and m.isfile()]
                pkl_members.sort(key=lambda m: m.name)

                print(f"✓ Found {len(pkl_members)} pickle files in archive")

                # Create list of (identifier, member) tuples for streaming
                analysis_items = [(Path(m.name).stem, m) for m in pkl_members]

        except Exception as e:
            print(f"✗ Failed to open tar file: {e}")
            import traceback
            traceback.print_exc()
            return {'processed_count': 0, 'failed_count': 0, 'total_files': 0}

    else:
        # Mode 2: Load from individual files in folder
        input_path = Path(input_folder)
        pkl_files = sorted(list(input_path.glob('*_analysis.pkl')))

        if len(pkl_files) == 0:
            print(f"No analysis pickle files found in {input_folder}")
            return {'processed_count': 0, 'total_files': 0}

        print("=" * 80)
        print("STATISTICAL ANALYSIS PIPELINE (FOLDER MODE)")
        print("=" * 80)
        print(f"Input folder: {input_folder}")
        if output_folder:
            print(f"Output folder: {output_folder}")
        print(f"Found {len(pkl_files)} analysis files to process")
        print(f"\nConfiguration:")
        print(f"  Diameter method: {diameter_method}")
        print(f"  Verbose output: {verbose}")
        print("=" * 80)

        # Convert pkl_files to list of (identifier, filepath) tuples
        analysis_items = [(pkl_file.stem, str(pkl_file)) for pkl_file in pkl_files]

    # Unified processing for both modes
    results_summary = {
        'processed_count': 0,
        'failed_count': 0,
        'total_files': len(analysis_items),
        'failed_files': []
    }

    main_branch_stats = {
        "total_path_length": [],
        "tortuosity": [],
        "mean_diameter": []
    }

    bifurcation_angle_stats = {
        "averaged_angle_A": [],
        "averaged_angle_B": [],
        "averaged_angle_C": [],
        "averaged_inflow_angle": []
    }

    bifurcation_diameter_stats = {
        "PMV": [],
        "DMV": [],
        "side_branch": []
    }

    trifurcation_main_angle_stats = {"averaged_angle_A_main":[], "averaged_angle_B_main":[], "averaged_angle_C_main":[], "averaged_inflow_angle":[],"additional_angles":[]}
    trifurcation_add_angle_stats={"averaged_angle_B1":{},"averaged_angle_B2":{}}
    trifurcation_diameter_stats = {"parent":[], "LAD":[], "LCx":[], "Ramus":[]}
    branches = ["LAD", "LCx", "RCA", "Ramus", "D1", "OM1", "AM1"]
    bifurcations = ["LAD_LCx", "LAD_D1", "RCA_AM1", "LCx_OM1"]
    conditions = ["Normal", "Diseased"]
    trifurcations = ["LCA_TRIFURCATION"]

    # Create dictionary for all statistics
    all_stats = {}
    for condition in conditions:
        all_stats[condition] = {}

        for branch in branches:
            all_stats[condition][branch] = {
                k: [] for k in main_branch_stats.keys()
            }

        for bifurc in bifurcations:
            all_stats[condition][bifurc] = {
                "Angles": {key: [] for key in bifurcation_angle_stats.keys()},
                "Diameters": {key: [] for key in bifurcation_diameter_stats.keys()}
            }
        for trifurc in trifurcations:
            all_stats[condition][trifurc] = {
                "main_plane_angles": {key:[] for key in trifurcation_main_angle_stats.keys()},
                "additional_angles": {key:[] for key in trifurcation_add_angle_stats.keys()},
                "Diameters": {key: [] for key in trifurcation_diameter_stats.keys()}
            }

    if input_tar_file is not None:
        processed_tarfile = tarfile.open(input_tar_file, 'r:*')
        
    for file_idx, (identifier, item) in enumerate(analysis_items, 1):
        print(f"\n{'=' * 80}")
        print(f"Processing {file_idx}/{len(analysis_items)}: {identifier}")
        print(f"{'=' * 80}")

        try:
            # Load data based on mode
            if input_tar_file is not None:
                # Item is a tarfile.TarInfo member - extract and load on-the-fly
                file_obj = processed_tarfile.extractfile(item)
                if file_obj is None:
                    raise ValueError(f"Could not extract {item.name} from archive")
                data = pickle.load(file_obj)
            else:
                # Item is a filepath, load it
                data = load_artery_analysis(item)

            # ========================================================================
            # AVAILABLE DATA IN THE LOADED DICTIONARY
            # ========================================================================
            #
            # data['final_graph']          - NetworkX DiGraph with all computed metrics
            # data['sparse_graph']         - Undirected graph before orientation
            # data['binary_mask']          - Binary 3D vessel mask (numpy array)
            # data['distance_array']       - Distance transform array (numpy array)
            # data['spacing_info']         - Voxel spacing in mm as tuple (z, y, x)
            # data['nrrd_header']          - Original NRRD header dictionary
            # data['processing_times']     - Dict of processing times for each step
            # data['total_time']           - Total processing time in seconds
            # data['metadata']             - Additional metadata dictionary containing:
            #     - 'filename': Original NRRD filename
            #     - 'file_basename': Filename without extension
            #     - 'artery': 'LCA' or 'RCA'
            #     - 'original_shape': Shape before preprocessing
            #     - 'config': Pipeline configuration parameters
            #
            # ========================================================================

            # Extract basic information
            graph = data['final_graph']
            metadata = data['metadata']
            spacing = data['spacing_info']

            artery_type = metadata.get('artery', 'unknown')
            patient_id = metadata.get('file_basename', 'unknown')

            if verbose:
                print(f"  Patient: {patient_id}")
                print(f"  Artery: {artery_type}")
                print(f"  Spacing: {spacing}")
                print(f"  Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            else:
                print(f"  Patient: {patient_id}, Artery: {artery_type}")

            # Example 1: Main branch statistics
            #   main_branches = extract_main_branch_statistics(
            #       graph, spacing, artery_type='LCA', diameter_method='slicing'
            #   )
            #
            # Returns dict with structure:
            #   {
            #       'LAD': {
            #           'total_path_length': 120.5,      # Sum of edge lengths (mm)
            #           'direct_path_length': 95.2,      # Euclidean start->end (mm)
            #           'tortuosity': 1.266,             # total / direct
            #           'mean_diameter': 3.42,           # Mean diameter (mm)
            #           'diameter_profile': [3.8, 3.7, ...],  # Stitched profile
            #           'start_coord': (100, 50, 25),    # Start voxel coords
            #           'end_coord': (200, 80, 30),      # End voxel coords
            #           'num_edges': 15,                 # Number of segments
            #           'edge_path': [(u1,v1), (u2,v2), ...]  # Ordered edges
            #       },
            #       'LCx': { ... },
            #       'Ramus': { ... }  # Only if present
            #   }
            #
            # Example 2: Bifurcation statistics
            #   bifurcations = extract_bifurcation_statistics(
            #       graph, spacing, diameter_method='slicing'
            #   )
            #
            # Returns dict with structure:
            #   {
            #       'LAD_D1': {
            #           'bifurcation_node': (x, y, z),
            #           'main_branch_label': 'LAD',
            #           'side_branch_label': 'D1',
            #           'angles': {
            #               'averaged_angle_A': 45.2,       # Angle A (degrees)
            #               'averaged_angle_B': 68.5,       # Angle B (degrees)
            #               'averaged_angle_C': 113.7,      # Angle C (degrees)
            #               'averaged_inflow_angle': 155.3  # Inflow angle (degrees)
            #           },
            #           'diameters': {
            #               'PMV': 3.5,           # Proximal main vessel (mm)
            #               'DMV': 3.2,           # Distal main vessel (mm)
            #               'side_branch': 2.1    # Side branch (mm)
            #           }
            #       },
            #       'LCx_OM1': { ... },
            #       ...
            #   }
            #
            # ========================================================================

            # ========================================================================
            # STATISTICAL ANALYSIS
            # ========================================================================

            if verbose:
                print(f"\n  --- Extracting Statistics ---")

            if verbose:
                print(f"\n  [Main Branch Statistics]")
            try:
                main_branches = extract_main_branch_statistics(
                    graph, spacing, artery_type=artery_type, diameter_method=diameter_method
                )

                if "Normal" in patient_id:
                    condition = "Normal"
                else:
                    condition = "Diseased"

                for branch_name, stats in main_branches.items():
                    all_stats[condition][branch_name]['total_path_length'].append(stats['total_path_length'])
                    all_stats[condition][branch_name]['tortuosity'].append(stats['tortuosity'])
                    all_stats[condition][branch_name]['mean_diameter'].append(stats['mean_diameter'])

                if verbose:
                    for branch_name, stats in main_branches.items():
                        print(f"\n    {branch_name}:")
                        print(f"      Total path length:    {stats['total_path_length']:.2f} mm")
                        print(f"      Direct path length:   {stats['direct_path_length']:.2f} mm")
                        print(f"      Tortuosity:           {stats['tortuosity']:.3f}")
                        print(f"      Mean diameter:        {stats['mean_diameter']:.2f} mm")
                        print(f"      Number of segments:   {stats['num_edges']}")

                    if not main_branches:
                        print(f"    No main branches found")

            except Exception as e:
                if verbose:
                    print(f"    [ERROR] Failed to extract main branch statistics: {str(e)}")

            if verbose:
                print(f"\n  [Bifurcation Statistics]")
            try:
                bifurcations = extract_bifurcation_statistics(
                    graph, spacing, diameter_method=diameter_method
                )

                if bifurcations:
                    for bifurc_name, bifurc_data in bifurcations.items():

                        if bifurc_name not in all_stats[condition]:
                            continue

                        angles = bifurc_data['angles']
                        diameters = bifurc_data['diameters']

                        

                        for angle_key, angle_value in angles.items():
                            all_stats[condition][bifurc_name]["Angles"][angle_key].append(angle_value)
                        
                        for diam_key, diam_value in diameters.items():
                            all_stats[condition][bifurc_name]["Diameters"][diam_key].append(diam_value)
                        

                if verbose:
                    
                    if bifurcations:
                        for bifurc_name, bifurc_data in bifurcations.items():
                            print(f"\n    {bifurc_name}:")

                            angles = bifurc_data['angles']
                            print(f"      Angles:")
                            if angles['averaged_angle_A'] is not None:
                                print(f"        Angle A (parent-side):     {angles['averaged_angle_A']:.1f}°")
                            if angles['averaged_angle_B'] is not None:
                                print(f"        Angle B (bifurcation):     {angles['averaged_angle_B']:.1f}°")
                            if angles['averaged_angle_C'] is not None:
                                print(f"        Angle C (parent-distal):   {angles['averaged_angle_C']:.1f}°")
                            if angles['averaged_inflow_angle'] is not None:
                                print(f"        Inflow angle:              {angles['averaged_inflow_angle']:.1f}°")

                            diameters = bifurc_data['diameters']
                            print(f"      Diameters:")
                            if diameters['PMV'] is not None:
                                print(f"        Proximal main vessel:  {diameters['PMV']:.2f} mm")
                            if diameters['DMV'] is not None:
                                print(f"        Distal main vessel:    {diameters['DMV']:.2f} mm")
                            if diameters['side_branch'] is not None:
                                print(f"        Side branch:           {diameters['side_branch']:.2f} mm")

                    else:
                        print(f"    No bifurcations found")

            except Exception as e:
                if verbose:
                    print(f"    [ERROR] Failed to extract bifurcation statistics: {str(e)}")

            # Trifurcation statistics (LCA only)
            if verbose and artery_type.upper() == 'LCA':
                print(f"\n  [Trifurcation Statistics]")
            try:
                if artery_type.upper() == 'LCA':
                    trifurcations = extract_trifurcation_statistics(
                        graph, spacing, min_depth_mm=2.0, max_depth_mm=7.0, step_mm=0.5,
                        diameter_method=diameter_method
                    )

                    if verbose:
                        if trifurcations:
                            for trifurc_name, trifurc_data in trifurcations.items():
                                if trifurc_name not in all_stats[condition]:
                                  continue

                            main_angles = trifurc_data['main_plane_angles']
                            additional_angles=trifurc_data['additional_angles']
                            diameters = trifurc_data['diameters']

                            for main_angle_key, main_angle_value in main_angles.items():
                                all_stats[condition][trifurc_name]["main_plane_angles"][main_angle_key].append(main_angle_value)

                            for additional_angle_key, additional_angle_value in additional_angles.items():
                                all_stats[condition][trifurc_name]['additional_angles'][additional_angle_key].append(additional_angle_value)
                            
                            for diam_key, diam_value in diameters.items():
                                all_stats[condition][trifurc_name]["Diameters"][diam_key].append(diam_value)

                            
                            for trifurc_name, trifurc_data in trifurcations.items():
                                print(f"\n    {trifurc_name}:")
                                print(f"      Type: {trifurc_data['type']}")
                                print(f"      Branches: {', '.join(trifurc_data['branches'])}")

                                main_angles = trifurc_data['main_plane_angles']
                                print(f"      Main plane angles (parent-LAD-LCx):")
                                if main_angles['averaged_angle_A_main'] is not None:
                                    print(f"        Angle A (parent-LCx):     {main_angles['averaged_angle_A_main']:.1f}°")
                                if main_angles['averaged_angle_B_main'] is not None:
                                    print(f"        Angle B (LAD-LCx):        {main_angles['averaged_angle_B_main']:.1f}°")
                                if main_angles['averaged_angle_C_main'] is not None:
                                    print(f"        Angle C (parent-LAD):     {main_angles['averaged_angle_C_main']:.1f}°")
                                if main_angles['averaged_inflow_angle'] is not None:
                                    print(f"        Inflow angle:             {main_angles['averaged_inflow_angle']:.1f}°")

                                add_angles = trifurc_data['additional_angles']
                                print(f"      Additional angles:")
                                if add_angles['averaged_angle_B1'] is not None:
                                    print(f"        B1 (LCx-Ramus):           {add_angles['averaged_angle_B1']:.1f}°")
                                if add_angles['averaged_angle_B2'] is not None:
                                    print(f"        B2 (LAD-Ramus):           {add_angles['averaged_angle_B2']:.1f}°")

                                diameters = trifurc_data['diameters']
                                print(f"      Diameters:")
                                if diameters['parent'] is not None:
                                    print(f"        Parent:  {diameters['parent']:.2f} mm")
                                if diameters['LAD'] is not None:
                                    print(f"        LAD:     {diameters['LAD']:.2f} mm")
                                if diameters['LCx'] is not None:
                                    print(f"        LCx:     {diameters['LCx']:.2f} mm")
                                if diameters['Ramus'] is not None:
                                    print(f"        Ramus:   {diameters['Ramus']:.2f} mm")

                                print(f"      Number of measurements: {trifurc_data['num_measurements']}")
                        else:
                            print(f"    No trifurcation found (LAD/LCx/Ramus not all present)")

            except Exception as e:
                if verbose:
                    print(f"    [ERROR] Failed to extract trifurcation statistics: {str(e)}")

            if verbose:
                print(f"\n  [All Branch Statistics]")
            try:
                all_branches = extract_all_branch_statistics(
                    graph, spacing, diameter_method=diameter_method
                )

                if verbose:
                    if all_branches:
                        branches_by_label = defaultdict(list)
                        for branch in all_branches:
                            branches_by_label[branch['branch_label']].append(branch)

                        for branch_label in sorted(branches_by_label.keys()):
                            branches = branches_by_label[branch_label]
                            print(f"\n    {branch_label}: ({len(branches)} segment{'s' if len(branches) > 1 else ''})")

                            for i, branch in enumerate(branches, 1):
                                if len(branches) > 1:
                                    print(f"      Segment {i}:")
                                    indent = "  "
                                else:
                                    indent = ""

                                print(f"      {indent}Length:         {branch['length']:.2f} mm")
                                print(f"      {indent}Mean diameter:  {branch['mean_diameter']:.2f} mm")

                    else:
                        print(f"    No branches found")

            except Exception as e:
                if verbose:
                    print(f"    [ERROR] Failed to extract all branch statistics: {str(e)}")

            if verbose:
                print(f"\n  [Summary]")
                total_branches = len([
                    e for e in graph.edges()
                    if 'lca_branch' in graph.edges[e] or 'rca_branch' in graph.edges[e] or 'branch_label' in graph.edges[e]
                ])
                total_bifurcations = len([n for n in graph.nodes()
                                         if 'averaged_angle_A' in graph.nodes[n]])
                print(f"    Total labeled branches:    {total_branches}")
                print(f"    Total bifurcations:        {total_bifurcations}")


            results_summary['processed_count'] += 1
            print(f"\n  [OK] Analysis processed successfully")

        except Exception as e:
            print(f"\n  [ERROR] Failed to process {identifier}")
            print(f"          Error: {str(e)}")
            results_summary['failed_files'].append({
                'identifier': identifier,
                'error': str(e)
            })
            results_summary['failed_count'] += 1

    # Compute averages and standard deviations for each list in the dictionary all_stats
    all_stats_avg = {}
    for condition, branches_data in all_stats.items():
        all_stats_avg[condition] = {}

        for branch_name, metrics in branches_data.items():
            all_stats_avg[condition][branch_name] = {}

            # Check if branch has bifurcation metrics
            if "Angles" in metrics and "Diameters" in metrics:
                all_stats_avg[condition][branch_name]["Angles"] = {
                    key: {
                        "mean": np.mean(value),
                        "std": np.std(value)
                    } for key, value in metrics["Angles"].items()
                }
                all_stats_avg[condition][branch_name]["Diameters"] = {
                    key: {
                        "mean": np.mean(value),
                        "std": np.std(value)
                    } for key, value in metrics["Diameters"].items()
                }

            elif "main_plane_angles" in metrics and "Diameters" in metrics:
                all_stats_avg[condition][branch_name]["main_plane_angles"] = {
                    key: {
                        "mean": np.mean(value),
                        "std": np.std(value)
                    } for key, value in metrics["main_plane_angles"].items()
                }

                all_stats_avg[condition][branch_name]["additional_angles"] = {
                    key: {
                        "mean": np.mean(value),
                        "std": np.std(value)
                    } for key, value in metrics["additional_angles"].items()}
                
                all_stats_avg[condition][branch_name]["Diameters"] = {
                    key: {
                        "mean": np.mean(value),
                        "std": np.std(value)
                    } for key, value in metrics["Diameters"].items()
                }

            else:
                # Main branch
                # print(f"Metrics:",metrics)
                all_stats_avg[condition][branch_name] = {
                    key: {
                        "mean": np.mean(value),
                        "std": np.std(value)
                    } for key, value in metrics.items()
                }

    # Compute std for each list in the dictionary all_stats
    all_stats_std = {}
    for condition, branches_data in all_stats.items():
        all_stats_std[condition] = {}

        for branch_name, metrics in branches_data.items():
            all_stats_std[condition][branch_name] = {}

            # Check if branch has bifurcation metrics
            if "Angles" in metrics and "Diameters" in metrics:
                all_stats_std[condition][branch_name]["Angles"] = {
                    key: np.std(value) for key, value in metrics["Angles"].items()
                }
                all_stats_std[condition][branch_name]["Diameters"] = {
                    key: np.std(value) for key, value in metrics["Diameters"].items()
                }

            elif "main_plane_angles" in metrics and "Diameters" in metrics:
                all_stats_std[condition][branch_name]["main_plane_angles"] = {
                    key: np.std(value) for key, value in metrics["main_plane_angles"].items()
                }

                all_stats_std[condition][branch_name]["additional_angles"] = {
                    key: np.std(value) for key, value in metrics["additional_angles"].items()}
                
                all_stats_std[condition][branch_name]["Diameters"] = {
                    key: np.std(value) for key, value in metrics["Diameters"].items()
                }

            else:
                # Main branch
                # print(f"Metrics:",metrics)
                all_stats_std[condition][branch_name] = {
                    key: np.std(value) for key, value in metrics.items()
                }
    # print(all_stats_avg)
    print(all_stats_std)
            
    # Compute combined averages for both diseased and normal subjects
    combined_stats = {}

    all_branches = set(all_stats.get("Normal").keys()) | set(all_stats.get("Diseased").keys())

    for branch_name in all_branches:
        combined_metrics = {}

        normal_branch = all_stats.get("Normal", {}).get(branch_name, {})
        diseased_branch = all_stats.get("Diseased", {}).get(branch_name, {})

        # Bifurcation
        if "Angles" in normal_branch or "Angles" in diseased_branch:
            combined_metrics["Angles"] = {}
            combined_metrics["Diameters"] = {}

            angle_metrics = set(normal_branch.get("Angles", {}).keys()) | set(diseased_branch.get("Angles", {}).keys())

            for metric in angle_metrics:
                values = normal_branch.get("Angles", {}).get(metric, []) + diseased_branch.get("Angles", {}).get(metric, [])

                if values:
                    combined_metrics["Angles"][metric] = {
                        "mean": np.mean(values),
                        "std": np.std(values)
                    }

            diameter_metrics = set(normal_branch.get("Diameters", {}).keys()) | set(diseased_branch.get("Diameters", {}).keys())

            for metric in diameter_metrics:
                values = normal_branch.get("Diameters", {}).get(metric, []) + diseased_branch.get("Diameters", {}).get(metric, [])

                if values:
                    combined_metrics["Diameters"][metric] = {
                        "mean": np.mean(values),
                        "std": np.std(values)
                    }

        # Trifurcation
        elif ("main_plane_angles" in normal_branch or "main_plane_angles" in diseased_branch):
            for angle_type in ["main_plane_angles", "additional_angles"]:
                combined_metrics[angle_type] = {}

                angle_metrics = set(normal_branch.get(angle_type, {}).keys()) | set(diseased_branch.get(angle_type, {}).keys())

                for metric in angle_metrics:
                    values = normal_branch.get(angle_type, {}).get(metric, []) + diseased_branch.get(angle_type, {}).get(metric, [])

                    if values:
                        combined_metrics[angle_type][metric] = {
                            "mean": np.mean(values),
                            "std": np.std(values)
                        }

            combined_metrics["Diameters"] = {}
            diameter_metrics = set(normal_branch.get("Diameters", {}).keys()) | set(diseased_branch.get("Diameters", {}).keys())

            for metric in diameter_metrics:
                values = normal_branch.get("Diameters", {}).get(metric, []) + diseased_branch.get("Diameters", {}).get(metric, [])

                if values:
                    combined_metrics["Diameters"][metric] = {
                        "mean": np.mean(values),
                        "std": np.std(values)
                    }

        # Main branch
        else:
            metrics = set(normal_branch.keys()) | set(diseased_branch.keys())

            for metric in metrics:
                values = normal_branch.get(metric, []) + diseased_branch.get(metric, [])

                if values:
                    combined_metrics[metric] = {
                        "mean": np.mean(values),
                        "std": np.std(values)
                    }
        
        combined_stats[branch_name] = combined_metrics

    ttest_results = compute_ttests(all_stats)

    print("\n" + "=" * 80)
    print("BATCH ANALYSIS COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"Files processed: {results_summary['processed_count']}/{results_summary['total_files']}")
    print(f"Files failed: {results_summary['failed_count']}/{results_summary['total_files']}")

    if results_summary['failed_files']:
        print("\nFailed analyses:")
        for failed in results_summary['failed_files']:
            print(f"  - {failed['identifier']}: {failed['error']}")

    print("=" * 80)

    if input_tar_file is not None:
        processed_tarfile.close()
    
    pp = pprint.PrettyPrinter(depth = 5, width = 120, compact = False)
    #pp.pprint(all_stats)

    if output_folder:
        save_avg_stats_to_csv(all_stats_avg, os.path.join(output_folder, "all_stats_avg.csv"))
        save_combined_stats_to_csv(combined_stats, os.path.join(output_folder, "combined_stats.csv"))
        save_ttest_results_to_csv(ttest_results, os.path.join(output_folder, "ttest_results.csv"))


    return results_summary


if __name__ == "__main__":
    results = analyze_artery_batch(
        config_path='analysis_config.yaml'
    )

    print(f"\n{'=' * 80}")
    print(f"ANALYSIS COMPLETE - SUMMARY")
    print(f"{'=' * 80}")
    print(f"Files processed: {results['processed_count']}/{results['total_files']}")
    print(f"Files failed: {results['failed_count']}/{results['total_files']}")
    if results['failed_files']:
        print(f"\nFailed analyses:")
        for failed in results['failed_files']:
            print(f"  - {failed['identifier']}: {failed['error']}")
    print(f"{'=' * 80}")

def save_avg_stats_to_csv(all_stats_avg, output_path):
    rows = []

    for condition, branches in all_stats_avg.items():
        for branch, metrics in branches.items():

            # Bifurcation with angles + diameters
            if "Angles" in metrics and "Diameters" in metrics:
                for metric, value in metrics["Angles"].items():
                    rows.append([condition, branch, "B_Angle", metric, value["mean"], value["std"]])

                for metric, value in metrics["Diameters"].items():
                    rows.append([condition, branch, "B_Diameter", metric, value["mean"], value["std"]])

            # Trifurcation with main-plane angles + additional angles + diameters 
            elif ("main_plane_angles" in metrics and "additional_angles" in metrics and "Diameters" in metrics):
                for metric, value in metrics["main_plane_angles"].items():
                    rows.append([condition, branch, "T_MainPlaneAngle", metric, value["mean"], value["std"]])

                for metric, value in metrics["additional_angles"].items():
                    rows.append([condition, branch, "T_AdditionalAngle", metric, value["mean"], value["std"]])
                
                for metric, value in metrics["Diameters"].items():
                    rows.append([condition, branch, "T_Diameter", metric, value["mean"], value["std"]])

            # Main branch
            else:
                for metric, value in metrics.items():
                    rows.append([condition, branch, "Main", metric, value["mean"], value["std"]])

    with open(output_path, "w", newline = "") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "branch", "metric_type", "metric", "mean", "std"])
        writer.writerows(rows)

def save_combined_stats_to_csv(combined_stats, output_path):
    rows = []

    for branch, metrics in combined_stats.items():

        # Bifurcation
        if "Angles" in metrics:
            for metric, value in metrics.get("Angles", {}).items():
                rows.append([branch, "B_Angle", metric, value["mean"], value["std"]])
        
        if "Diameters" in metrics and "Angles" in metrics:
            for metric, value in metrics.get("Diameters", {}).items():
                rows.append([branch, "B_Diameters", metric, value["mean"], value["std"]])

        # Trifurcation 
        if "main_plane_angles" in metrics: 
            for metric, value in metrics.get("main_plane_angles", {}).items():
                rows.append([branch, "T_MainPlaneAngle", metric, value["mean"], value["std"]])

        if "additional_angles" in metrics:
            for metric, value in metrics.get("additional_angles", {}).items():
                rows.append([branch, "T_AdditionalAngle", metric, value["mean"], value["std"]])
        
        if "Diameters" in metrics and ("main_plane_angles" in metrics or "additional_angles" in metrics):
            for metric, value in metrics.get("Diameters", {}).items():
                rows.append([branch, "T_Diameter", metric, value["mean"], value["std"]])

        # Main branch
        if not any(key in metrics for key in ["Angles", "main_plane_angles", "additional_angles"]):
                for metric, value in metrics.items():
                    rows.append([branch, "Main", metric, value["mean"], value["std"]])
        # else:
        #     for metric, value in metrics.items():
        #         rows.append([branch, "Main", metric, value["mean"], value["std"]])

    with open(output_path, "w", newline = "") as f:
        writer = csv.writer(f)
        writer.writerow(["branch", "metric_type", "metric", "mean", "std"])
        writer.writerows(rows)

def save_ttest_results_to_csv(ttest_results, output_path):
    rows = []

    for branch, metrics in ttest_results.items():

        # Bifurcation
        if "Angles" in metrics:
            for metric, result in metrics.get("Angles", {}).items():
                rows.append([
                    branch, "B_Angle", metric,
                    result["t-statistic"], result["p-value"],
                    result["n_normal"], result["n_diseased"]])

        if "Diameters" in metrics and "Angles" in metrics:
            for metric, result in metrics.get("Diameters", {}).items():
                rows.append([
                    branch, "B_Diameter", metric,
                    result["t-statistic"], result["p-value"],
                    result["n_normal"], result["n_diseased"]])

        # Trifurcation
        if "main_plane_angles" in metrics:
            for metric, results in metrics.get("main_plane_angles", {}).items():
                rows.append([
                    branch, "MainPlaneAngle", metric,
                    result["t-statistic"], result["p-value"],
                    result["n_normal"], result["n_diseased"]])

        if "additional_angles" in metrics:
            for metric, results in metrics.get("additional_angles", {}).items():
                rows.append([
                    branch, "AdditionalAngle", metric,
                    result["t-statistic"], result["p-value"],
                    result["n_normal"], result["n_diseased"]])

        if "Diameters" in metrics and ("main_plane_angles" in metrics or "additional_angles" in metrics):
            for metric, result in metrics.get("Diameters", {}).items():
                rows.append([
                    branch, "T_Diameter", metric,
                    result["t-statistic"], result["p-value"],
                    result["n_normal"], result["n_diseased"]])

        # Main branch
        if not any(key in metrics for key in ["Angles", "main_plane_angles", "additional_angles"]):
            for metric, result in metrics.items():
                rows.append([
                    branch, "Main", metric,
                    result["t-statistic"], result["p-value"],
                    result["n_normal"], result["n_diseased"]])

    with open(output_path, "w", newline = "") as f:
        writer = csv.writer(f)
        writer.writerow(["branch", "metric_type", "metric", "t_statistic", "p_value", "n_normal", "n_diseased"])
        writer.writerows(rows)


def compute_ttests(all_stats):
    ttest_results = {}

    normal = all_stats.get("Normal", {})
    diseased = all_stats.get("Diseased", {})

    all_branches = set(normal.keys()) | set(diseased.keys())
    
    for branch in all_branches:
        ttest_results[branch] = {}
        
        normal_branch = normal.get(branch, {})
        diseased_branch = diseased.get(branch, {})

        # Bifurcation
        if "Angles" in normal_branch:
            ttest_results[branch]["Angles"] = {}
            ttest_results[branch]["Diameters"] = {}

            for metric in normal_branch.get("Angles", {}):
                n = normal_branch["Angles"].get(metric, [])
                d = diseased_branch.get("Angles", {}).get(metric, [])

                if len(n) > 1 and len(d) > 1:
                    tstat, pvalue = ttest_ind(n, d, equal_var = False)
                    ttest_results[branch]["Angles"][metric] = {
                        "t-statistic": tstat, 
                        "p-value": pvalue, 
                        "n_normal": len(n), 
                        "n_diseased": len(d)
                        }
            
            for metric in normal_branch.get("Diameters", {}):
                n = normal_branch["Diameters"].get(metric, [])
                d = diseased_branch.get("Diameters", {}).get(metric, [])

                if len(n) > 1 and len(d) > 1:
                    tstat, pvalue = ttest_ind(n, d, equal_var = False)
                    ttest_results[branch]["Diameters"][metric] = {
                        "t-statistic": tstat, 
                        "p-value": pvalue, 
                        "n_normal": len(n), 
                        "n_diseased": len(d)
                        }

        # Trifurcation
        if "main_plane_angles" in normal_branch:
            ttest_results[branch]["main_plane_angles"] = {}
            ttest_results[branch]["additional_angles"] = {}
            ttest_results[branch]["Diameters"] = {}

            for metric in normal_branch.get("main_plane_angles", {}):
                n = normal_branch["main_plane_angles"].get(metric, [])
                d = diseased_branch.get("main_plane_angles", {}).get(metric, [])

                if len(n) > 1 and len(d) > 1:
                    tstat, pvalue = ttest_ind(n, d, equal_var=False)
                    ttest_results[branch]["main_plane_angles"][metric] = {
                        "t-statistic": tstat, 
                        "p-value": pvalue, 
                        "n_normal": len(n), 
                        "n_diseased": len(d)
                        }

            for metric in normal_branch.get("additional_angles", {}):
                n = normal_branch["additional_angles"].get(metric, [])
                d = diseased_branch.get("additional_angles", {}).get(metric, [])

                if len(n) > 1 and len(d) > 1:
                    tstat, pvalue = ttest_ind(n, d, equal_var=False)
                    ttest_results[branch]["additional_angles"][metric] = {
                        "t-statistic": tstat, 
                        "p-value": pvalue, 
                        "n_normal": len(n), 
                        "n_diseased": len(d)
                        }

            for metric in normal_branch.get("Diameters", {}):
                n = normal_branch["Diameters"].get(metric, [])
                d = diseased_branch.get("Diameters", {}).get(metric, [])

                if len(n) > 1 and len(d) > 1:
                    tstat, pvalue = ttest_ind(n, d, equal_var=False)
                    ttest_results[branch]["Diameters"][metric] = {
                        "t-statistic": tstat, 
                        "p-value": pvalue, 
                        "n_normal": len(n), 
                        "n_diseased": len(d)
                        }
        # Main branch
        if not any(key in normal_branch for key in ["Angles", "main_plane_angles", "additional_angles"]):
            # Main branch metrics
            for metric in normal_branch:
                n = normal_branch.get(metric, [])
                d = diseased_branch.get(metric, [])

                if len(n) > 1 and len(d) > 1:
                    tstat, pvalue = ttest_ind(n, d, equal_var = False)
                    ttest_results[branch][metric] = {
                        "t-statistic": tstat, 
                        "p-value": pvalue, 
                        "n_normal": len(n), 
                        "n_diseased": len(d)
                        }

    return ttest_results