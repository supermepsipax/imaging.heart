import os
import time
import psutil
import numpy as np
from pathlib import Path
from utilities import (
    load_nrrd_mask,
    ensure_continuous_body,
    preprocess_binary_mask,
    sort_labelled_bodies_by_size,
    resample_to_isotropic,
    load_config,
    classify_lca_rca_from_graphs,
    create_distance_transform_from_mask,
    annotate_lca_graph_with_branch_labels,
    annotate_rca_graph_with_branch_labels,
    save_artery_analysis,
    VesselErrorTracker,
    BatchErrorLogger,
    validate_anatomical_labels,
    validate_lca_branch_length_ratio,
)
from pipelines import process_single_artery
from analysis import convert_graph_to_dataframes
from visualizations import visualize_3d_graph


def process_batch_arteries(input_folder=None, output_folder=None, config=None, config_path=None, visualize=None):
    """
    Process multiple artery masks from an input folder and save results to an output folder.

    Each mask is expected to contain at least two continuous bodies (LCA and RCA).
    The two largest bodies will be extracted and processed separately, then classified
    as LCA/RCA based on vessel complexity analysis.

    Args:
        input_folder (str, optional): Path to folder containing .nrrd mask files (can be set in config)
        output_folder (str, optional): Path to folder where CSV results will be saved (can be set in config)
        config (dict, optional): Configuration dictionary with pipeline parameters
        config_path (str, optional): Path to JSON config file (if config not provided directly)
        visualize (bool, optional): Whether to create 3D visualizations for each artery (can be set in config)

    Returns:
        dict: Summary of batch processing including success/failure counts and processing times

    Raises:
        ValueError: If input_folder or output_folder are not provided either as parameters or in config
    """
    if config is None and config_path is not None:
        config = load_config(config_path)

    if config is None:
        config = {}

    if input_folder is None:
        input_folder = config.get('input_folder')
    if output_folder is None:
        output_folder = config.get('output_folder')
    if visualize is None:
        visualize = config.get('visualize', False)

    if input_folder is None:
        raise ValueError("input_folder must be provided either as a parameter or in the config file")
    if output_folder is None:
        raise ValueError("output_folder must be provided either as a parameter or in the config file")

    upsample_factor = config.get('upsample_factor', 1)
    min_depth_mm = config.get('min_depth_mm', 2.0)
    max_depth_mm = config.get('max_depth_mm', 7.0)
    step_mm = config.get('step_mm', 0.5)
    remove_bypass = config.get('remove_bypass', True)
    bypass_threshold = config.get('bypass_threshold', 2.0)
    prune_small_y_branches = config.get('prune_small_y_branches', True)
    max_y_branch_length_voxels = config.get('max_y_branch_length_voxels', 3)
    max_y_branch_length_mm = config.get('max_y_branch_length_mm')
    min_branch_length_voxels = config.get('min_branch_length_voxels')
    min_branch_length_mm = config.get('min_branch_length_mm')
    max_recursion_depth = config.get('max_recursion_depth', 5)
    lca_trifurcation_threshold_mm = config.get('lca_trifurcation_threshold_mm', 5.0)
    max_unlabeled_edges = config.get('max_unlabeled_edges', 1)
    min_lca_branch_length_ratio = config.get('min_lca_branch_length_ratio', 0.1)
    angle_weight = config.get('angle_weight', 0.15)
    diameter_weight = config.get('diameter_weight', 0.25)
    path_length_weight = config.get('path_length_weight', 0.6)
    diameter_method = config.get('diameter_method', 'both')

    os.makedirs(output_folder, exist_ok=True)

    input_path = Path(input_folder)
    output_path = Path(output_folder)
    nrrd_files = sorted(list(input_path.glob('*.nrrd')))

    if len(nrrd_files) == 0:
        print(f"No .nrrd files found in {input_folder}")
        return {'success_count': 0, 'failure_count': 0, 'total_files': 0}

    already_processed = []
    for nrrd_file in nrrd_files:
        file_basename = nrrd_file.stem
        lca_pkl = output_path / f"{file_basename}_LCA_analysis.pkl"
        rca_pkl = output_path / f"{file_basename}_RCA_analysis.pkl"

        if lca_pkl.exists() and rca_pkl.exists():
            already_processed.append(nrrd_file.name)

    if already_processed:
        print("=" * 80)
        print(f"FOUND {len(already_processed)} ALREADY-PROCESSED FILES")
        print("=" * 80)
        print("\nThe following files have both LCA and RCA analysis results in the output folder:")
        for filename in already_processed:
            print(f"  • {filename}")
        print()

        while True:
            response = input("Do you want to [S]kip these files or [R]eprocess them? (S/R): ").strip().upper()
            if response in ['S', 'SKIP']:
                nrrd_files = [f for f in nrrd_files if f.name not in already_processed]
                print(f"\n✓ Skipping {len(already_processed)} already-processed files")
                print(f"  Will process {len(nrrd_files)} remaining files\n")
                break
            elif response in ['R', 'REPROCESS']:
                print(f"\n✓ Reprocessing all {len(already_processed)} files\n")
                break
            else:
                print("Invalid input. Please enter 'S' to skip or 'R' to reprocess.")

        if len(nrrd_files) == 0:
            print("No files left to process.")
            return {'success_count': 0, 'failure_count': 0, 'total_files': 0, 'skipped_count': len(already_processed)}

    print("=" * 80)
    print(f"BATCH ARTERY ANALYSIS PIPELINE")
    print("=" * 80)
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Found {len(nrrd_files)} .nrrd files to process")
    print(f"\nConfiguration:")
    print(f"  Upsample factor: {upsample_factor}")
    print(f"  Depth range: {min_depth_mm}mm - {max_depth_mm}mm (step: {step_mm}mm)")
    print(f"  Diameter computation method: {diameter_method}")
    print(f"  Remove bypass edges: {remove_bypass}")
    if remove_bypass:
        print(f"  Bypass threshold: {bypass_threshold} voxels")
    print(f"  Prune small Y-branches: {prune_small_y_branches}")
    if prune_small_y_branches:
        if max_y_branch_length_voxels is not None:
            print(f"  Y-branch voxel threshold: {max_y_branch_length_voxels} voxels")
        if max_y_branch_length_mm is not None:
            print(f"  Y-branch mm threshold: {max_y_branch_length_mm} mm")
    if min_branch_length_voxels is not None or min_branch_length_mm is not None:
        print(f"  Filter short branches during graph construction:")
        if min_branch_length_voxels is not None:
            print(f"    Min branch length: {min_branch_length_voxels} voxels")
        if min_branch_length_mm is not None:
            print(f"    Min branch length: {min_branch_length_mm} mm")
        print(f"    Max recursion depth: {max_recursion_depth}")
    print(f"  LCA trifurcation threshold: {lca_trifurcation_threshold_mm} mm")
    print(f"  Max unlabeled edges (validation): {max_unlabeled_edges}")
    print(f"  Min LCA branch length ratio (validation): {min_lca_branch_length_ratio}")
    print(f"  Branch designation weights: angle={angle_weight}, diameter={diameter_weight}, path_length={path_length_weight}")
    print(f"  Visualize results: {visualize}")
    print("=" * 80)

    batch_start_time = time.time()
    results_summary = {
        'success_count': 0,
        'failure_count': 0,
        'total_files': len(nrrd_files),
        'skipped_count': len(already_processed) if already_processed else 0,
        'processed_files': [],
        'failed_files': [],
        'total_arteries_processed': 0,
    }

    error_logger = BatchErrorLogger(output_folder, config)

    for file_idx, nrrd_file in enumerate(nrrd_files, 1):
        file_basename = nrrd_file.stem  # filename without extension

        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024**3  # GB

        print(f"\n{'=' * 80}")
        print(f"Processing file {file_idx}/{len(nrrd_files)}: {nrrd_file.name}")
        print(f"Memory usage: {mem_before:.2f} GB")
        print(f"{'=' * 80}")

        try:
            file_size_mb = nrrd_file.stat().st_size / 1024**2
            print(f"\n[Loading] Reading {nrrd_file.name}... ({file_size_mb:.1f} MB)")
            binary_mask, header = load_nrrd_mask(str(nrrd_file), verbose=False)

            spacing_info = tuple(np.diag(header['space directions']))
            print(f"          Original shape: {binary_mask.shape}")
            print(f"          Original spacing: {spacing_info}")

            print(f"[Resampling] Converting to isotropic spacing...")
            binary_mask, spacing_info = resample_to_isotropic(binary_mask, spacing_info)
            print(f"             New shape: {binary_mask.shape}")
            print(f"             New spacing: {spacing_info}")

            print(f"[Preprocessing] Applying preprocessing (upsample factor: {upsample_factor})...")
            binary_mask = preprocess_binary_mask(binary_mask, upsample_factor=upsample_factor)

            print(f"[Body Detection] Identifying continuous bodies...")
            is_continuous, labelled_bodies = ensure_continuous_body(binary_mask, debug=False)

            if is_continuous:
                print(f"                 Found 1 continuous body - skipping (expected at least 2)")
                reason = 'Only 1 continuous body found (expected at least 2 for LCA/RCA)'
                results_summary['failed_files'].append({
                    'filename': nrrd_file.name,
                    'reason': reason
                })
                results_summary['failure_count'] += 1

                error_logger.add_file_result(nrrd_file.name, 'failed', reason=reason)
                continue

            sorted_bodies = sort_labelled_bodies_by_size(labelled_bodies)
            num_bodies = len(sorted_bodies)
            print(f"                 Found {num_bodies} continuous bodies")

            if num_bodies < 2:
                print(f"                 Insufficient bodies found - skipping")
                reason = f'Only {num_bodies} bodies found (expected at least 2)'
                results_summary['failed_files'].append({
                    'filename': nrrd_file.name,
                    'reason': reason
                })
                results_summary['failure_count'] += 1

                # Log to error logger
                error_logger.add_file_result(nrrd_file.name, 'failed', reason=reason)
                continue

            # Step 1: Compute distance transform ONCE on full mask (optimization) - only if needed
            compute_distance_transform = diameter_method in ['edt', 'both']
            if compute_distance_transform:
                print(f"\n[Distance Transform] Computing on full combined mask (both vessels)...")
                print(f"                     Mask shape: {binary_mask.shape}, dtype: {binary_mask.dtype}")
                mem_before_dist = process.memory_info().rss / 1024**3
                print(f"                     Memory before: {mem_before_dist:.2f} GB")

                dist_start = time.time()
                distance_array_full = create_distance_transform_from_mask(binary_mask, spacing_info)
                dist_time = time.time() - dist_start

                mem_after_dist = process.memory_info().rss / 1024**3
                print(f"                     [OK] Computed in {dist_time:.3f}s")
                print(f"                     Memory after: {mem_after_dist:.2f} GB (Δ {mem_after_dist - mem_before_dist:.2f} GB)")
            else:
                print(f"\n[Distance Transform] Skipping (diameter_method='slicing' - not needed)...")
                distance_array_full = None

            # Step 2: Process both bodies without saving CSVs
            print(f"\n  --- Processing both vessels for classification ---")
            graphs = []
            body_masks = []
            distance_arrays = []
            processing_results = []
            for body_idx in range(2):
                body_mask = sorted_bodies[body_idx]
                body_masks.append(body_mask)

                # Extract distance transform values only where this body exists
                # This is much faster than computing distance transform separately for each body
                if distance_array_full is not None:
                    distance_array_body = distance_array_full * (body_mask > 0)
                else:
                    distance_array_body = None
                distance_arrays.append(distance_array_body)

                print(f"\n  Processing vessel {body_idx + 1}/2...")

                result = process_single_artery(
                    binary_mask=body_mask,
                    spacing_info=spacing_info,
                    min_depth_mm=min_depth_mm,
                    max_depth_mm=max_depth_mm,
                    step_mm=step_mm,
                    remove_bypass=remove_bypass,
                    bypass_threshold=bypass_threshold,
                    prune_small_y_branches=prune_small_y_branches,
                    max_y_branch_length_voxels=max_y_branch_length_voxels,
                    max_y_branch_length_mm=max_y_branch_length_mm,
                    min_branch_length_voxels=min_branch_length_voxels,
                    min_branch_length_mm=min_branch_length_mm,
                    max_recursion_depth=max_recursion_depth,
                    angle_weight=angle_weight,
                    diameter_weight=diameter_weight,
                    path_length_weight=path_length_weight,
                    diameter_method=diameter_method,
                    output_csv=False,  # Don't save yet - we need to classify first
                    distance_array=distance_array_body  # Pass pre-computed distance transform
                )

                graphs.append(result['final_graph'])
                processing_results.append(result)
                print(f"  [OK] Vessel {body_idx + 1} processed successfully")

            classification = classify_lca_rca_from_graphs(graphs, verbose=True)
            lca_index = classification['lca_index']
            rca_index = classification['rca_index']
            original_lca_index = lca_index  # Store original for potential revert
            original_rca_index = rca_index
            classification_confidence = classification.get('confidence', 'UNKNOWN')  # Store confidence

            lca_tracker = VesselErrorTracker(nrrd_file.name, 'LCA')
            rca_tracker = VesselErrorTracker(nrrd_file.name, 'RCA')
            vessel_trackers = [lca_tracker, rca_tracker]

            print(f"\n  --- Applying anatomical branch labels ---")
            graphs[lca_index] = annotate_lca_graph_with_branch_labels(
                graphs[lca_index],
                spacing_info,
                trifurcation_threshold_mm=lca_trifurcation_threshold_mm
            )

            graphs[rca_index] = annotate_rca_graph_with_branch_labels(graphs[rca_index])

            print(f"\n  --- Validating anatomical labels ---")

            lca_valid, lca_unlabeled = validate_anatomical_labels(
                graphs[lca_index], 'LCA', lca_tracker, max_unlabeled=max_unlabeled_edges
            )
            if lca_valid:
                print(f"  [LCA] ✓ Validation passed ({lca_tracker.metrics.get('unlabeled_edges', 0)} unlabeled edges)")
            else:
                print(f"  [LCA] ✗ Validation failed ({lca_tracker.metrics.get('unlabeled_edges', 0)} unlabeled edges, max: {max_unlabeled_edges})")

            # Validate RCA
            rca_valid, rca_unlabeled = validate_anatomical_labels(
                graphs[rca_index], 'RCA', rca_tracker, max_unlabeled=max_unlabeled_edges
            )
            if rca_valid:
                print(f"  [RCA] ✓ Validation passed ({rca_tracker.metrics.get('unlabeled_edges', 0)} unlabeled edges)")
            else:
                print(f"  [RCA] ✗ Validation failed ({rca_tracker.metrics.get('unlabeled_edges', 0)} unlabeled edges, max: {max_unlabeled_edges})")

            print(f"\n  --- Validating LCA branch length ratios ---")

            ratio_valid, lad_len, lcx_len, ratio = validate_lca_branch_length_ratio(
                graphs[lca_index], spacing_info, lca_tracker,
                min_ratio=min_lca_branch_length_ratio,
                log_critical=False
            )

            if ratio_valid:
                print(f"  [LCA] ✓ Branch length ratio passed (LAD={lad_len:.1f}mm, LCx={lcx_len:.1f}mm, ratio={ratio:.3f})")
            else:
                print(f"  [LCA] ✗ Branch length ratio failed (LAD={lad_len:.1f}mm, LCx={lcx_len:.1f}mm, ratio={ratio:.3f} < {min_lca_branch_length_ratio:.3f})")

                # Check if original classification had HIGH confidence - be conservative about swapping
                if classification_confidence and classification_confidence.upper() == 'HIGH':
                    print(f"  [!] Original classification had HIGH confidence ({classification_confidence})")
                    print(f"  [!] Branch length ratio is close to threshold - keeping original classification")
                    print(f"  [NOTE] This may indicate anatomical variation (e.g., short LCx, dominant LAD)")
                else:
                    print(f"  [!] Attempting to swap LCA/RCA classification and re-label...")

                    # Swap indices
                    lca_index, rca_index = rca_index, lca_index

                    # Get fresh unlabeled graphs from processing_results (before any labeling)
                    # Use .copy() to avoid modifying the original
                    graphs[0] = processing_results[0]['final_graph'].copy()
                    graphs[1] = processing_results[1]['final_graph'].copy()

                    print(f"  [!] Re-labeling vessel 1 as {'LCA' if lca_index == 0 else 'RCA'}...")
                    print(f"  [!] Re-labeling vessel 2 as {'LCA' if lca_index == 1 else 'RCA'}...")

                    # Try to label the swapped LCA
                    try:
                        swapped_lca_graph = annotate_lca_graph_with_branch_labels(
                            graphs[lca_index],
                            spacing_info,
                            trifurcation_threshold_mm=lca_trifurcation_threshold_mm
                        )

                        # Check if LCA labeling succeeded by checking if branch pattern was detected
                        lca_labeling_info = swapped_lca_graph.graph.get('lca_labeling', {})
                        lca_labels = lca_labeling_info.get('labels', {})

                        if not lca_labels or len(lca_labels) == 0:
                            # LCA labeling failed - no branch pattern detected
                            print(f"  [!] Failed to detect LCA branch pattern after swap")
                            print(f"  [!] Reverting to original LCA/RCA classification")
                            print(f"  [NOTE] Both classifications problematic - skipping save for both vessels")

                            # Revert the swap
                            lca_index = original_lca_index
                            rca_index = original_rca_index

                            # Restore original labeled graphs (already done above before swap attempt)
                            graphs[lca_index] = annotate_lca_graph_with_branch_labels(
                                graphs[lca_index],
                                spacing_info,
                                trifurcation_threshold_mm=lca_trifurcation_threshold_mm
                            )
                            graphs[rca_index] = annotate_rca_graph_with_branch_labels(graphs[rca_index])

                            # Mark both vessels as failed validation - neither should be saved
                            lca_tracker.log_critical(
                                "LCA/RCA Classification",
                                "Original classification failed ratio validation, swap attempt failed to detect branch pattern"
                            )
                            rca_tracker.log_critical(
                                "LCA/RCA Classification",
                                "Classification uncertain - both LCA and RCA assignments problematic"
                            )
                        else:
                            # LCA labeling succeeded, continue with swap
                            graphs[lca_index] = swapped_lca_graph
                            graphs[rca_index] = annotate_rca_graph_with_branch_labels(graphs[rca_index])

                            print(f"  [!] Re-validating anatomical labels after swap...")

                            # Re-validate anatomical labels
                            lca_valid, lca_unlabeled = validate_anatomical_labels(
                                graphs[lca_index], 'LCA', lca_tracker, max_unlabeled=max_unlabeled_edges
                            )

                            rca_valid, rca_unlabeled = validate_anatomical_labels(
                                graphs[rca_index], 'RCA', rca_tracker, max_unlabeled=max_unlabeled_edges
                            )

                            # Re-validate branch length ratio
                            ratio_valid_retry, lad_len_retry, lcx_len_retry, ratio_retry = validate_lca_branch_length_ratio(
                                graphs[lca_index], spacing_info, lca_tracker,
                                min_ratio=min_lca_branch_length_ratio,
                                log_critical=False  # Don't log critical yet
                            )

                            # Handle potential None values from failed validation
                            lad_str = f"{lad_len_retry:.1f}" if lad_len_retry is not None else "N/A"
                            lcx_str = f"{lcx_len_retry:.1f}" if lcx_len_retry is not None else "N/A"
                            ratio_str = f"{ratio_retry:.3f}" if ratio_retry is not None else "N/A"

                            if ratio_valid_retry:
                                print(f"  [LCA] ✓ Branch length ratio passed after swap (LAD={lad_str}mm, LCx={lcx_str}mm, ratio={ratio_str})")
                                print(f"  [✓] Swap successful - using corrected classification")
                            else:
                                print(f"  [LCA] ✗ Branch length ratio still failed after swap (LAD={lad_str}mm, LCx={lcx_str}mm, ratio={ratio_str})")
                                print(f"  [!] Reverting to original LCA/RCA classification")
                                print(f"  [NOTE] Both classifications problematic - skipping save for both vessels")

                                # Revert the swap
                                lca_index = original_lca_index
                                rca_index = original_rca_index

                                # Restore original labeled graphs
                                graphs[0] = processing_results[0]['final_graph'].copy()
                                graphs[1] = processing_results[1]['final_graph'].copy()
                                graphs[lca_index] = annotate_lca_graph_with_branch_labels(
                                    graphs[lca_index],
                                    spacing_info,
                                    trifurcation_threshold_mm=lca_trifurcation_threshold_mm
                                )
                                graphs[rca_index] = annotate_rca_graph_with_branch_labels(graphs[rca_index])

                                # Mark both vessels as failed validation - neither should be saved
                                lca_tracker.log_critical(
                                    "LCA/RCA Classification",
                                    "Both original and swapped classifications failed branch length ratio validation"
                                )
                                rca_tracker.log_critical(
                                    "LCA/RCA Classification",
                                    "Classification uncertain - both LCA and RCA assignments problematic"
                                )

                    except Exception as e:
                        print(f"  [ERROR] Exception during swap attempt: {str(e)}")
                        print(f"  [!] Reverting to original LCA/RCA classification")
                        print(f"  [NOTE] Both classifications problematic - skipping save for both vessels")

                        # Revert the swap
                        lca_index = original_lca_index
                        rca_index = original_rca_index

                        # Restore original labeled graphs
                        graphs[0] = processing_results[0]['final_graph'].copy()
                        graphs[1] = processing_results[1]['final_graph'].copy()
                        graphs[lca_index] = annotate_lca_graph_with_branch_labels(
                            graphs[lca_index],
                            spacing_info,
                            trifurcation_threshold_mm=lca_trifurcation_threshold_mm
                        )
                        graphs[rca_index] = annotate_rca_graph_with_branch_labels(graphs[rca_index])

                        # Mark both vessels as failed validation - neither should be saved
                        lca_tracker.log_critical(
                            "LCA/RCA Classification",
                            f"Original classification failed ratio validation, swap attempt raised exception: {str(e)}"
                        )
                        rca_tracker.log_critical(
                            "LCA/RCA Classification",
                            "Classification uncertain - both LCA and RCA assignments problematic"
                        )

            print(f"\n  --- Saving results (validation-based) ---")
            vessel_info = [
                {
                    'index': lca_index,
                    'label': 'LCA',
                    'tracker': lca_tracker,
                    'graph': graphs[lca_index],
                    'mask': body_masks[lca_index],
                    'distance_array': distance_arrays[lca_index],
                    'result': processing_results[lca_index]
                },
                {
                    'index': rca_index,
                    'label': 'RCA',
                    'tracker': rca_tracker,
                    'graph': graphs[rca_index],
                    'mask': body_masks[rca_index],
                    'distance_array': distance_arrays[rca_index],
                    'result': processing_results[rca_index]
                }
            ]

            vessels_saved = 0
            for vessel in vessel_info:
                label = vessel['label']
                tracker = vessel['tracker']
                graph = vessel['graph']
                mask = vessel['mask']
                distance_array = vessel['distance_array']
                result = vessel['result']

                if not tracker.should_save():
                    print(f"  [{label}] ⚠️  Skipping save - failed validation")
                    continue

                nodes_csv = os.path.join(output_folder, f"{file_basename}_{label}_nodes.csv")
                edges_csv = os.path.join(output_folder, f"{file_basename}_{label}_edges.csv")
                analysis_pkl = os.path.join(output_folder, f"{file_basename}_{label}_analysis.pkl")

                # # Save as CSV (for human-readable inspection)
                # convert_graph_to_dataframes(graph, nodes_csv=nodes_csv, edges_csv=edges_csv)

                # Save complete analysis as pickle (for Python statistical analysis)
                save_artery_analysis(
                    analysis_pkl,
                    final_graph=graph,
                    sparse_graph=result['sparse_graph'],
                    binary_mask=mask,
                    distance_array=distance_array,
                    spacing_info=spacing_info,
                    nrrd_header=header,
                    processing_times=result['processing_times'],
                    total_time=result['total_time'],
                    metadata={
                        'filename': nrrd_file.name,
                        'file_basename': file_basename,
                        'artery': label,
                        'original_shape': binary_mask.shape,
                        'config': {
                            'upsample_factor': upsample_factor,
                            'min_depth_mm': min_depth_mm,
                            'max_depth_mm': max_depth_mm,
                            'step_mm': step_mm,
                            'remove_bypass': remove_bypass,
                            'bypass_threshold': bypass_threshold
                        }
                    }
                )

                print(f"  [OK] {label} results saved")
                print(f"       Nodes CSV: {nodes_csv}")
                print(f"       Edges CSV: {edges_csv}")
                print(f"       Analysis PKL: {analysis_pkl}")

                if visualize:
                    print(f"  [Visualization] Opening 3D visualization for {label}...")
                    viz_title = f"{file_basename} - {label}"
                    visualize_3d_graph(graph, binary_mask=mask, title=viz_title, dark_mode=True, hide_background=True)
                    print(f"                  Visualization displayed")

                vessels_saved += 1
                results_summary['total_arteries_processed'] += 1

            if vessels_saved == 2:
                file_status = 'success'
                results_summary['success_count'] += 1
            elif vessels_saved == 1:
                file_status = 'partial'
                results_summary['success_count'] += 1
            else:
                file_status = 'failed'
                results_summary['failure_count'] += 1

            error_logger.add_file_result(
                nrrd_file.name,
                status=file_status,
                trackers=vessel_trackers
            )

            results_summary['processed_files'].append({
                'filename': nrrd_file.name,
                'arteries_processed': vessels_saved
            })

            mem_after = process.memory_info().rss / 1024**3
            print(f"\n[Memory] File processing complete")
            print(f"         Memory after: {mem_after:.2f} GB (Δ {mem_after - mem_before:.2f} GB from start)")

            # Reminder to close visualization tabs to prevent memory issues
            if visualize and file_idx % 2 == 0 and file_idx < len(nrrd_files):
                print(f"\n{'!' * 80}")
                print(f"  REMINDER: {file_idx * 2} visualization tabs have been opened.")
                print(f"  Please close browser tabs to free memory before continuing.")
                print(f"  Files remaining: {len(nrrd_files) - file_idx}")
                print(f"{'!' * 80}")
                input("Press Enter to continue to next file... ")

        except Exception as e:
            print(f"\n[ERROR] Failed to process {nrrd_file.name}")
            print(f"        Error: {str(e)}")
            reason = f"Exception during processing: {str(e)}"
            results_summary['failed_files'].append({
                'filename': nrrd_file.name,
                'reason': reason
            })
            results_summary['failure_count'] += 1

            error_logger.add_file_result(nrrd_file.name, 'failed', reason=reason)

    batch_total_time = time.time() - batch_start_time

    print("\n" + "=" * 80)
    print("WRITING ERROR LOG")
    print("=" * 80)
    error_logger.write_log()

    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"\nTotal batch processing time: {batch_total_time:.3f}s ({batch_total_time/60:.2f} minutes)")
    print(f"\nFiles processed: {results_summary['success_count']}/{results_summary['total_files']}")
    print(f"Files failed: {results_summary['failure_count']}/{results_summary['total_files']}")
    if results_summary['skipped_count'] > 0:
        print(f"Files skipped (already processed): {results_summary['skipped_count']}")
    print(f"Total arteries processed: {results_summary['total_arteries_processed']}")

    if results_summary['success_count'] > 0:
        print(f"\nAverage time per file: {batch_total_time/results_summary['success_count']:.3f}s")

    if results_summary['failed_files']:
        print("\nFailed files:")
        for failed in results_summary['failed_files']:
            print(f"  - {failed['filename']}: {failed['reason']}")

    print("=" * 80)

    if results_summary['success_count'] > 0 or results_summary['skipped_count'] > 0:
        print("\n" + "=" * 80)
        print("ARCHIVE RESULTS")
        print("=" * 80)
        print("\nWould you like to archive all analysis files into a single compressed file?")
        print("This reduces storage space, and works cross-platform (Windows/Mac/Linux).")
        print()

        while True:
            response = input("Archive results? [Y]es / [N]o: ").strip().upper()
            if response in ['Y', 'YES']:
                # Ask for compression method
                print("\nSelect compression method:")
                print("  [1] gzip  - Fast, good compression (~5-10x smaller) [default]")
                print("  [2] bz2   - Slower, better compression (~8-15x smaller)")
                print("  [3] xz    - Slowest, best compression (~10-20x smaller)")
                print()

                compression_choice = input("Enter choice (1/2/3) or press Enter for default: ").strip()

                if compression_choice == '2':
                    compression = 'bz2'
                    ext = '.tar.bz2'
                elif compression_choice == '3':
                    compression = 'xz'
                    ext = '.tar.xz'
                else:
                    compression = 'gz'
                    ext = '.tar.gz'

                folder_name = Path(output_folder).name
                archive_filename = f"{folder_name}_archive{ext}"
                archive_path = Path(output_folder) / archive_filename

                print(f"\nCreating archive: {archive_path}")
                print()

                try:
                    from utilities import archive_artery_analyses

                    archive_stats = archive_artery_analyses(
                        input_folder=output_folder,
                        output_file=str(archive_path),
                        pattern='*_analysis.pkl',
                        compression=compression
                    )

                    print()
                    print("✓ Archive created successfully!")
                    print(f"  Archive file: {archive_path}")
                    print(f"  Size: {archive_stats['output_size_mb']:.2f} MB")
                    print()
                    print("  You can now analyze this archive without unpacking using:")
                    print(f"    analyze_artery_batch(input_tar_file='{archive_path}')")
                    print()

                except Exception as e:
                    print(f"\n✗ Archive creation failed: {e}")
                    import traceback
                    traceback.print_exc()

                break

            elif response in ['N', 'NO']:
                print("\nSkipping archive. You can create an archive later using:")
                print(f"  from utilities import archive_artery_analyses")
                print(f"  archive_artery_analyses('{output_folder}', 'output.tar.gz')")
                print()
                break
            else:
                print("Invalid input. Please enter 'Y' for Yes or 'N' for No.")

        print("=" * 80)

    results_summary['total_time'] = batch_total_time
    return results_summary
