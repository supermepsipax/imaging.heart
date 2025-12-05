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

    os.makedirs(output_folder, exist_ok=True)

    input_path = Path(input_folder)
    nrrd_files = sorted(list(input_path.glob('*.nrrd')))

    if len(nrrd_files) == 0:
        print(f"No .nrrd files found in {input_folder}")
        return {'success_count': 0, 'failure_count': 0, 'total_files': 0}

    print("=" * 80)
    print(f"BATCH ARTERY ANALYSIS PIPELINE")
    print("=" * 80)
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Found {len(nrrd_files)} .nrrd files to process")
    print(f"\nConfiguration:")
    print(f"  Upsample factor: {upsample_factor}")
    print(f"  Depth range: {min_depth_mm}mm - {max_depth_mm}mm (step: {step_mm}mm)")
    print(f"  Remove bypass edges: {remove_bypass}")
    if remove_bypass:
        print(f"  Bypass threshold: {bypass_threshold} voxels")
    print(f"  Visualize results: {visualize}")
    print("=" * 80)

    batch_start_time = time.time()
    results_summary = {
        'success_count': 0,
        'failure_count': 0,
        'total_files': len(nrrd_files),
        'processed_files': [],
        'failed_files': [],
        'total_arteries_processed': 0,
    }

    for file_idx, nrrd_file in enumerate(nrrd_files, 1):
        file_basename = nrrd_file.stem  # filename without extension

        # Memory monitoring
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024**3  # GB

        print(f"\n{'=' * 80}")
        print(f"Processing file {file_idx}/{len(nrrd_files)}: {nrrd_file.name}")
        print(f"Memory usage: {mem_before:.2f} GB")
        print(f"{'=' * 80}")

        try:
            # Check file size
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
                results_summary['failed_files'].append({
                    'filename': nrrd_file.name,
                    'reason': 'Only 1 continuous body found (expected at least 2 for LCA/RCA)'
                })
                results_summary['failure_count'] += 1
                continue

            sorted_bodies = sort_labelled_bodies_by_size(labelled_bodies)
            num_bodies = len(sorted_bodies)
            print(f"                 Found {num_bodies} continuous bodies")

            if num_bodies < 2:
                print(f"                 Insufficient bodies found - skipping")
                results_summary['failed_files'].append({
                    'filename': nrrd_file.name,
                    'reason': f'Only {num_bodies} bodies found (expected at least 2)'
                })
                results_summary['failure_count'] += 1
                continue

            # Step 1: Compute distance transform ONCE on full mask (optimization)
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
                distance_array_body = distance_array_full * (body_mask > 0)
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
                    output_csv=False,  # Don't save yet - we need to classify first
                    distance_array=distance_array_body  # Pass pre-computed distance transform
                )

                graphs.append(result['final_graph'])
                processing_results.append(result)
                print(f"  [OK] Vessel {body_idx + 1} processed successfully")

            # Step 3: Classify LCA vs RCA based on complexity
            classification = classify_lca_rca_from_graphs(graphs, verbose=True)
            lca_index = classification['lca_index']
            rca_index = classification['rca_index']

            # Step 4: Apply anatomical branch labeling (with integrated spatial validation)
            print(f"\n  --- Applying anatomical branch labels ---")
            graphs[lca_index] = annotate_lca_graph_with_branch_labels(
                graphs[lca_index],
                spacing_info,
                trifurcation_threshold_mm=5.0
            )

            graphs[rca_index] = annotate_rca_graph_with_branch_labels(graphs[rca_index])

            # Step 5: Save results with correct labels
            print(f"\n  --- Saving results with correct labels ---")
            vessel_info = [
                {
                    'index': lca_index,
                    'label': 'LCA',
                    'graph': graphs[lca_index],
                    'mask': body_masks[lca_index],
                    'distance_array': distance_arrays[lca_index],
                    'result': processing_results[lca_index]
                },
                {
                    'index': rca_index,
                    'label': 'RCA',
                    'graph': graphs[rca_index],
                    'mask': body_masks[rca_index],
                    'distance_array': distance_arrays[rca_index],
                    'result': processing_results[rca_index]
                }
            ]

            for vessel in vessel_info:
                label = vessel['label']
                graph = vessel['graph']
                mask = vessel['mask']
                distance_array = vessel['distance_array']
                result = vessel['result']

                nodes_csv = os.path.join(output_folder, f"{file_basename}_{label}_nodes.csv")
                edges_csv = os.path.join(output_folder, f"{file_basename}_{label}_edges.csv")
                analysis_pkl = os.path.join(output_folder, f"{file_basename}_{label}_analysis.pkl")

                # Save as CSV (for human-readable inspection)
                convert_graph_to_dataframes(graph, nodes_csv=nodes_csv, edges_csv=edges_csv)

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

                # Visualize if requested
                if visualize:
                    print(f"  [Visualization] Opening 3D visualization for {label}...")
                    viz_title = f"{file_basename} - {label}"
                    visualize_3d_graph(graph, binary_mask=mask, title=viz_title)
                    print(f"                  Visualization displayed")

                results_summary['total_arteries_processed'] += 1

            results_summary['processed_files'].append({
                'filename': nrrd_file.name,
                'arteries_processed': 2
            })
            results_summary['success_count'] += 1

            # Memory monitoring after processing
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
            results_summary['failed_files'].append({
                'filename': nrrd_file.name,
                'reason': str(e)
            })
            results_summary['failure_count'] += 1

    batch_total_time = time.time() - batch_start_time

    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"\nTotal batch processing time: {batch_total_time:.3f}s ({batch_total_time/60:.2f} minutes)")
    print(f"\nFiles processed: {results_summary['success_count']}/{results_summary['total_files']}")
    print(f"Files failed: {results_summary['failure_count']}/{results_summary['total_files']}")
    print(f"Total arteries processed: {results_summary['total_arteries_processed']}")

    if results_summary['success_count'] > 0:
        print(f"\nAverage time per file: {batch_total_time/results_summary['success_count']:.3f}s")

    if results_summary['failed_files']:
        print("\nFailed files:")
        for failed in results_summary['failed_files']:
            print(f"  - {failed['filename']}: {failed['reason']}")

    print("=" * 80)

    results_summary['total_time'] = batch_total_time
    return results_summary
