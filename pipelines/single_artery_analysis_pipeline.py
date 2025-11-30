import time
import numpy as np
import networkx as nx
from utilities import (
    extract_centerline_skimage,
    extract_endpoint_and_bifurcation_coordinates,
    skeleton_to_sparse_graph,
    skeleton_to_sparse_graph_robust,
    remove_redundant_bifurcation_clusters,
    remove_sharp_bend_bifurcations,
    create_distance_transform_from_mask,
    compute_branch_diameters_of_graph,
    compute_branch_lengths_of_graph,
    make_directed_graph,
    determine_origin_node_from_diameter,
    diameter_profile,
    summarize_profile,
    traverse_graph_and_compute_angles,
)
from analysis import convert_graph_to_dataframes


def process_single_artery(binary_mask, spacing_info, min_depth_mm=1.0, max_depth_mm=5.0,
                          step_mm=0.5, output_csv=True, nodes_csv="nodes.csv", edges_csv="edges.csv"):
    """
    Process a single continuous artery binary mask through the complete analysis pipeline.

    This function performs centerline extraction, graph construction, diameter computation,
    angle analysis at bifurcations, and exports results to CSV files.

    Args:
        binary_mask (ndarray): Binary 3D mask of a single continuous artery (1 = vessel, 0 = background)
        spacing_info (tuple): Voxel spacing in mm for each dimension (z, y, x)
        min_depth_mm (float): Minimum depth for bifurcation angle averaging (default 1.0 mm)
        max_depth_mm (float): Maximum depth for bifurcation angle averaging (default 5.0 mm)
        step_mm (float): Step size for depth increments in angle computation (default 0.5 mm)
        output_csv (bool): Whether to output CSV files (default True)
        nodes_csv (str): Output filename for nodes CSV (default "nodes.csv")
        edges_csv (str): Output filename for edges CSV (default "edges.csv")

    Returns:
        dict: Dictionary containing:
            - 'final_graph': NetworkX DiGraph with all computed metrics
            - 'sparse_graph': Undirected graph before orientation
            - 'processing_times': Dict of processing times for each step
            - 'total_time': Total processing time in seconds
    """

    start_time = time.time()
    processing_times = {}

    print("=" * 80)
    print("ARTERY ANALYSIS PIPELINE - STARTED")
    print("=" * 80)

    # Step 1: Create distance transform
    print("\n[1/9] Creating distance transform from binary mask...")
    step_start = time.time()
    distance_array = create_distance_transform_from_mask(binary_mask, spacing_info)
    processing_times['distance_transform'] = time.time() - step_start
    print(f"      [OK] Distance transform computed in {processing_times['distance_transform']:.3f}s")

    # Step 2: Extract centerline skeleton
    print("\n[2/9] Extracting centerline skeleton using skeletonization...")
    step_start = time.time()
    skeleton_binary_mask = extract_centerline_skimage(binary_mask)
    processing_times['centerline_extraction'] = time.time() - step_start
    print(f"      [OK] Centerline extracted in {processing_times['centerline_extraction']:.3f}s")

    # Step 3: Extract endpoints and bifurcation points (before processing)
    print("\n[3/9] Extracting endpoints and bifurcation points...")
    step_start = time.time()
    endpoints, bifurcation_points = extract_endpoint_and_bifurcation_coordinates(skeleton_binary_mask)
    print(f"      --> Found {len(endpoints)} endpoints and {len(bifurcation_points)} bifurcation points")
    processing_times['endpoint_bifurcation_extraction'] = time.time() - step_start
    print(f"      [OK] Extracted in {processing_times['endpoint_bifurcation_extraction']:.3f}s")

    # Step 4: Remove redundant bifurcation clusters
    print("\n[4/9] Removing redundant bifurcation clusters...")
    step_start = time.time()
    initial_bifurcation_count = len(bifurcation_points)
    bifurcation_points = remove_redundant_bifurcation_clusters(bifurcation_points)
    removed_redundant = initial_bifurcation_count - len(bifurcation_points)
    print(f"      --> Removed {removed_redundant} redundant bifurcations")
    print(f"      --> {len(bifurcation_points)} bifurcations remaining")
    processing_times['remove_redundant_bifurcations'] = time.time() - step_start
    print(f"      [OK] Processed in {processing_times['remove_redundant_bifurcations']:.3f}s")

    # Step 5: Remove sharp bend bifurcations
    print("\n[5/9] Removing sharp bend (false) bifurcations...")
    step_start = time.time()
    initial_bifurcation_count = len(bifurcation_points)
    bifurcation_points = remove_sharp_bend_bifurcations(bifurcation_points, skeleton_binary_mask)
    removed_sharp_bends = initial_bifurcation_count - len(bifurcation_points)
    print(f"      --> Removed {removed_sharp_bends} sharp bend bifurcations")
    print(f"      --> {len(bifurcation_points)} bifurcations remaining")
    processing_times['remove_sharp_bend_bifurcations'] = time.time() - step_start
    print(f"      [OK] Processed in {processing_times['remove_sharp_bend_bifurcations']:.3f}s")

    # Step 6: Create sparse skeleton graph
    print("\n[6/9] Constructing sparse skeleton graph from centerline...")
    step_start = time.time()
    sparse_skeleton_graph = skeleton_to_sparse_graph_robust(skeleton_binary_mask, bifurcation_points, endpoints)
    print(f"      --> Graph has {sparse_skeleton_graph.number_of_nodes()} nodes")
    print(f"      --> Graph has {sparse_skeleton_graph.number_of_edges()} edges")

    num_components = nx.number_connected_components(sparse_skeleton_graph)
    if num_components > 1:
        print(f"      [WARNING] Graph has {num_components} disconnected components!")
        components = list(nx.connected_components(sparse_skeleton_graph))
        for i, component in enumerate(components):
            print(f"                Component {i+1}: {len(component)} nodes")

    processing_times['graph_construction'] = time.time() - step_start
    print(f"      [OK] Graph constructed in {processing_times['graph_construction']:.3f}s")

    # Step 7: Compute branch lengths and diameter profiles
    print("\n[7/9] Computing branch metrics (lengths and diameter profiles)...")
    step_start = time.time()
    sparse_skeleton_graph = compute_branch_lengths_of_graph(sparse_skeleton_graph, spacing_info)

    # Method 1: EDT-based diameter (original method using distance transform)
    print("      --> Computing EDT-based diameters...")
    sparse_skeleton_graph = compute_branch_diameters_of_graph(sparse_skeleton_graph, distance_array)

    # Method 2: Plane-based diameter (new method using perpendicular slices)
    print("      --> Computing plane-based diameter profiles...")
    edge_count = 0
    for edge in sparse_skeleton_graph.edges:
        voxels = sparse_skeleton_graph.edges[edge]['voxels']
        profile = diameter_profile(binary_mask, voxels)
        stats = summarize_profile(profile)

        sparse_skeleton_graph.edges[edge]['diameter_profile'] = profile
        sparse_skeleton_graph.edges[edge].update(stats)
        edge_count += 1

    print(f"      --> Computed diameters using both methods for {edge_count} branches")
    print(f"          - EDT method: average_diameter_mm_edt, median_diameter_mm_edt")
    print(f"          - Plane method: mean_diameter, median_diameter, min/max/std_diameter")
    processing_times['branch_metrics'] = time.time() - step_start
    print(f"      [OK] Branch metrics computed in {processing_times['branch_metrics']:.3f}s")

    # Step 8: Determine origin node and create directed graph
    print("\n[8/9] Determining origin node and creating directed graph...")
    step_start = time.time()
    origin_node = determine_origin_node_from_diameter(sparse_skeleton_graph)
    print(f"      --> Origin node identified at: {origin_node}")
    directed_skeleton_graph = make_directed_graph(sparse_skeleton_graph, origin_node)
    processing_times['graph_orientation'] = time.time() - step_start
    print(f"      [OK] Directed graph created in {processing_times['graph_orientation']:.3f}s")

    # Step 9: Compute bifurcation angles
    print("\n[9/9] Computing bifurcation angles across the vascular tree...")
    print(f"      --> Depth range: {min_depth_mm}mm to {max_depth_mm}mm (step: {step_mm}mm)")
    step_start = time.time()
    final_graph = traverse_graph_and_compute_angles(
        directed_skeleton_graph, spacing_info,
        min_depth_mm=min_depth_mm,
        max_depth_mm=max_depth_mm,
        step_mm=step_mm
    )

    # Count bifurcations with computed angles
    bifurcations_with_angles = sum(1 for node in final_graph.nodes()
                                   if 'averaged_angle_A' in final_graph.nodes[node])
    print(f"      --> Computed angles for {bifurcations_with_angles} bifurcations")
    processing_times['angle_computation'] = time.time() - step_start
    print(f"      [OK] Angles computed in {processing_times['angle_computation']:.3f}s")

    # Output to CSV if requested
    if output_csv:
        print(f"\n[CSV Export] Writing results to {nodes_csv} and {edges_csv}...")
        step_start = time.time()
        convert_graph_to_dataframes(final_graph, nodes_csv=nodes_csv, edges_csv=edges_csv)
        processing_times['csv_export'] = time.time() - step_start
        print(f"             [OK] CSV files written in {processing_times['csv_export']:.3f}s")

    # Calculate total time
    total_time = time.time() - start_time

    # Print summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"\nTotal processing time: {total_time:.3f}s ({total_time/60:.2f} minutes)")
    print("\nBreakdown by stage:")
    for stage, duration in processing_times.items():
        percentage = (duration / total_time) * 100
        print(f"  {stage:.<40} {duration:>7.3f}s ({percentage:>5.1f}%)")

    print("\nFinal graph statistics:")
    print(f"  Total nodes: {final_graph.number_of_nodes()}")
    print(f"  Total edges: {final_graph.number_of_edges()}")
    print(f"  Bifurcations analyzed: {bifurcations_with_angles}")
    print("=" * 80)

    return {
        'final_graph': final_graph,
        'sparse_graph': sparse_skeleton_graph,
        'processing_times': processing_times,
        'total_time': total_time
    }
