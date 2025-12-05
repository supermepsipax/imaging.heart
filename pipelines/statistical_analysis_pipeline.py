import os
import numpy as np
from pathlib import Path
from utilities import load_artery_analysis, load_config
from analysis import (
    extract_main_branch_statistics,
    extract_all_branch_statistics,
    extract_bifurcation_statistics,
    compute_branch_tapering
)
from collections import defaultdict


def analyze_artery_batch(input_folder=None, output_folder=None, config=None, config_path=None,
                         diameter_method=None, verbose=None):
    """
    Perform statistical analysis on a batch of artery analysis pickle files.

    This pipeline loads pre-processed artery analysis results and computes
    statistical metrics across multiple patients/samples.

    Args:
        input_folder (str, optional): Path to folder containing .pkl analysis files
        output_folder (str, optional): Path to folder where analysis results will be saved
        config (dict, optional): Configuration dictionary with analysis parameters
        config_path (str, optional): Path to config file (if config not provided directly)
        diameter_method (str, optional): 'slicing' or 'edt' - which diameter measurements to use
        verbose (bool, optional): Whether to print detailed statistics for each artery

    Returns:
        dict: Summary statistics and results from the batch analysis
    """

    if config is None and config_path is not None:
        config = load_config(config_path)

    if config is None:
        config = {}

    if input_folder is None:
        input_folder = config.get('input_folder')
    if output_folder is None:
        output_folder = config.get('output_folder')
    if diameter_method is None:
        diameter_method = config.get('diameter_method', 'slicing')
    if verbose is None:
        verbose = config.get('verbose', True)

    if input_folder is None:
        raise ValueError("input_folder must be provided either as a parameter or in the config file")

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)

    input_path = Path(input_folder)
    pkl_files = sorted(list(input_path.glob('*_analysis.pkl')))

    if len(pkl_files) == 0:
        print(f"No analysis pickle files found in {input_folder}")
        return {'processed_count': 0, 'total_files': 0}

    print("=" * 80)
    print("STATISTICAL ANALYSIS PIPELINE")
    print("=" * 80)
    print(f"Input folder: {input_folder}")
    if output_folder:
        print(f"Output folder: {output_folder}")
    print(f"Found {len(pkl_files)} analysis files to process")
    print(f"\nConfiguration:")
    print(f"  Diameter method: {diameter_method}")
    print(f"  Verbose output: {verbose}")
    print("=" * 80)

    results_summary = {
        'processed_count': 0,
        'failed_count': 0,
        'total_files': len(pkl_files),
        'failed_files': []
    }

    for file_idx, pkl_file in enumerate(pkl_files, 1):
        print(f"\n{'=' * 80}")
        print(f"Processing file {file_idx}/{len(pkl_files)}: {pkl_file.name}")
        print(f"{'=' * 80}")

        try:
            data = load_artery_analysis(str(pkl_file))

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
            print(f"\n  [OK] File processed successfully")

        except Exception as e:
            print(f"\n  [ERROR] Failed to process {pkl_file.name}")
            print(f"          Error: {str(e)}")
            results_summary['failed_files'].append({
                'filename': pkl_file.name,
                'error': str(e)
            })
            results_summary['failed_count'] += 1

    print("\n" + "=" * 80)
    print("BATCH ANALYSIS COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"Files processed: {results_summary['processed_count']}/{results_summary['total_files']}")
    print(f"Files failed: {results_summary['failed_count']}/{results_summary['total_files']}")

    if results_summary['failed_files']:
        print("\nFailed files:")
        for failed in results_summary['failed_files']:
            print(f"  - {failed['filename']}: {failed['error']}")

    print("=" * 80)

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
        print(f"\nFailed files:")
        for failed in results['failed_files']:
            print(f"  - {failed['filename']}: {failed['error']}")
    print(f"{'=' * 80}")
