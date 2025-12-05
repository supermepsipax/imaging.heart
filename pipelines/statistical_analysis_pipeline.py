import os
import numpy as np
from pathlib import Path
from utilities import load_artery_analysis, load_config


def analyze_artery_batch(input_folder=None, output_folder=None, config=None, config_path=None):
    """
    Perform statistical analysis on a batch of artery analysis pickle files.

    This pipeline loads pre-processed artery analysis results and computes
    statistical metrics across multiple patients/samples.

    Args:
        input_folder (str, optional): Path to folder containing .pkl analysis files
        output_folder (str, optional): Path to folder where analysis results will be saved
        config (dict, optional): Configuration dictionary with analysis parameters
        config_path (str, optional): Path to JSON config file (if config not provided directly)

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
            # GRAPH NODE ATTRIBUTES (accessed via graph.nodes[node_id])
            # ========================================================================
            #
            # Bifurcation nodes (degree > 2) may have:
            #   - 'averaged_angle_A': Averaged bifurcation angle A (degrees)
            #   - 'averaged_angle_B': Averaged bifurcation angle B (degrees)
            #   - 'averaged_inflow_angle': Averaged inflow angle (degrees)
            #   - 'branch_label': Anatomical label (e.g., 'LAD', 'LCx', 'RCA', etc.)
            #
            # All nodes have:
            #   - Position coordinates as the node ID itself (tuple of x, y, z)
            #
            # ========================================================================
            # GRAPH EDGE ATTRIBUTES (accessed via graph[u][v] or graph.edges[u, v])
            # ========================================================================
            #
            # Geometric properties:
            #   - 'length': Branch length in mm
            #   - 'voxels': List of (x,y,z) coordinates along the branch path
            #
            # Diameter measurements (EDT method):
            #   - 'mean_diameter_edt': Mean diameter using distance transform (mm)
            #   - 'median_diameter_edt': Median diameter using distance transform (mm)
            #   - 'diameter_profile_edt': List of diameter values along branch (mm)
            #
            # Diameter measurements (Slicing method):
            #   - 'mean_diameter_slicing': Mean diameter using orthogonal slicing (mm)
            #   - 'median_diameter_slicing': Median diameter using orthogonal slicing (mm)
            #   - 'diameter_profile_slicing': List of diameter values along branch (mm)
            #
            # Anatomical labeling:
            #   - 'branch_label': Anatomical branch name (e.g., 'LAD', 'LCx', 'D1', 'RCA', 'PDA')
            #   - 'generation': Branch generation/level in hierarchy (0=main, 1=primary, etc.)
            #
            # ========================================================================

            # Extract basic information
            graph = data['final_graph']
            metadata = data['metadata']
            spacing = data['spacing_info']

            artery_type = metadata.get('artery', 'unknown')
            patient_id = metadata.get('file_basename', 'unknown')

            print(f"  Patient: {patient_id}")
            print(f"  Artery: {artery_type}")
            print(f"  Spacing: {spacing}")
            print(f"  Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

            # ========================================================================
            # EXAMPLE: Accessing graph data
            # ========================================================================

            print(f"\n  Example data access:")

            # Example 1: Iterate over all edges and access diameter profiles
            print(f"    Branches with diameter profiles:")
            for edge_idx, (u, v) in enumerate(graph.edges()):
                if edge_idx >= 3:  # Only show first 3 as examples
                    print(f"    ... ({graph.number_of_edges() - 3} more edges)")
                    break

                edge_data = graph[u][v]
                branch_label = edge_data.get('branch_label', 'unlabeled')
                length = edge_data.get('length', 0.0)
                mean_diam_edt = edge_data.get('mean_diameter_edt', 0.0)
                mean_diam_slice = edge_data.get('mean_diameter_slicing', 0.0)

                print(f"      Edge {u}->{v} ({branch_label}): "
                      f"length={length:.2f}mm, "
                      f"diam_edt={mean_diam_edt:.2f}mm, "
                      f"diam_slice={mean_diam_slice:.2f}mm")

            # Example 2: Find bifurcations and access angle data
            print(f"\n    Bifurcations with angle measurements:")
            bifurcation_count = 0
            for node in graph.nodes():
                if graph.out_degree(node) > 1:  # Bifurcation point
                    node_data = graph.nodes[node]
                    if 'averaged_angle_A' in node_data:
                        bifurcation_count += 1
                        if bifurcation_count <= 3:  # Only show first 3
                            angle_A = node_data['averaged_angle_A']
                            angle_B = node_data['averaged_angle_B']
                            inflow = node_data.get('averaged_inflow_angle', 'N/A')
                            print(f"      Node {node}: angle_A={angle_A:.1f}°, "
                                  f"angle_B={angle_B:.1f}°, inflow={inflow}")

            if bifurcation_count > 3:
                print(f"      ... ({bifurcation_count - 3} more bifurcations)")

            # ========================================================================
            # YOUR STATISTICAL ANALYSIS CODE GOES HERE
            # ========================================================================
            #
            # This is where you would add your own analysis code. Examples:
            #
            # 1. Compute statistics by branch type:
            #    lad_branches = [graph[u][v] for u, v in graph.edges()
            #                    if graph[u][v].get('branch_label') == 'LAD']
            #    lad_diameters = [b['mean_diameter_slicing'] for b in lad_branches]
            #    print(f"LAD mean diameter: {np.mean(lad_diameters):.2f} mm")
            #
            # 2. Analyze diameter tapering along branches:
            #    for u, v in graph.edges():
            #        profile = graph[u][v]['diameter_profile_slicing']
            #        tapering = (profile[0] - profile[-1]) / len(profile)
            #        print(f"Branch {u}->{v} tapering: {tapering:.4f} mm/voxel")
            #
            # 3. Compare EDT vs Slicing diameter methods:
            #    for u, v in graph.edges():
            #        edt = graph[u][v]['mean_diameter_edt']
            #        slicing = graph[u][v]['mean_diameter_slicing']
            #        diff = abs(edt - slicing)
            #        print(f"Diameter difference: {diff:.2f} mm")
            #
            # 4. Analyze bifurcation angle distributions:
            #    angles = [graph.nodes[n]['averaged_angle_A']
            #              for n in graph.nodes()
            #              if 'averaged_angle_A' in graph.nodes[n]]
            #    print(f"Mean bifurcation angle: {np.mean(angles):.1f}°")
            #
            # 5. Export data to pandas DataFrame for further analysis:
            #    import pandas as pd
            #    edge_data = []
            #    for u, v in graph.edges():
            #        edge_data.append({
            #            'patient': patient_id,
            #            'artery': artery_type,
            #            'branch': graph[u][v].get('branch_label', 'unknown'),
            #            'length': graph[u][v]['length'],
            #            'diameter': graph[u][v]['mean_diameter_slicing']
            #        })
            #    df = pd.DataFrame(edge_data)
            #    df.to_csv(f"{output_folder}/{patient_id}_{artery_type}_analysis.csv")
            #
            # ========================================================================


            # TODO: Add your statistical analysis here


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

    # Print summary
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
    # Example usage
    results = analyze_artery_batch(
        input_folder='results/test_batch',
        output_folder='results/statistics',
        config_path='config.json'  # Optional
    )
