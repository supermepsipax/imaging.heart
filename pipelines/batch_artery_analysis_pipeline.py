import os
import time
import numpy as np
from pathlib import Path
from utilities import (
    load_nrrd_mask,
    ensure_continuous_body,
    preprocess_binary_mask,
    sort_labelled_bodies_by_size,
    resample_to_isotropic,
    load_config,
)
from pipelines import process_single_artery


def process_batch_arteries(input_folder=None, output_folder=None, config=None, config_path=None):
    """
    Process multiple artery masks from an input folder and save results to an output folder.

    Each mask is expected to contain at least two continuous bodies (LCA and RCA).
    The two largest bodies will be extracted and processed separately.

    Args:
        input_folder (str, optional): Path to folder containing .nrrd mask files (can be set in config)
        output_folder (str, optional): Path to folder where CSV results will be saved (can be set in config)
        config (dict, optional): Configuration dictionary with pipeline parameters
        config_path (str, optional): Path to JSON config file (if config not provided directly)

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

        print(f"\n{'=' * 80}")
        print(f"Processing file {file_idx}/{len(nrrd_files)}: {nrrd_file.name}")
        print(f"{'=' * 80}")

        try:
            print(f"\n[Loading] Reading {nrrd_file.name}...")
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

            body_labels = ['LCA', 'RCA']  # Assuming largest is LCA, second largest is RCA

            for body_idx in range(2):
                body_mask = sorted_bodies[body_idx]
                body_label = body_labels[body_idx]

                print(f"\n  --- Processing {body_label} (body {body_idx + 1}/2) ---")

                nodes_csv = os.path.join(output_folder, f"{file_basename}_{body_label}_nodes.csv")
                edges_csv = os.path.join(output_folder, f"{file_basename}_{body_label}_edges.csv")

                result = process_single_artery(
                    binary_mask=body_mask,
                    spacing_info=spacing_info,
                    min_depth_mm=min_depth_mm,
                    max_depth_mm=max_depth_mm,
                    step_mm=step_mm,
                    remove_bypass=remove_bypass,
                    bypass_threshold=bypass_threshold,
                    output_csv=True,
                    nodes_csv=nodes_csv,
                    edges_csv=edges_csv
                )

                print(f"  [OK] {body_label} processed successfully")
                print(f"       Nodes saved to: {nodes_csv}")
                print(f"       Edges saved to: {edges_csv}")

                results_summary['total_arteries_processed'] += 1

            results_summary['processed_files'].append({
                'filename': nrrd_file.name,
                'arteries_processed': 2
            })
            results_summary['success_count'] += 1

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
