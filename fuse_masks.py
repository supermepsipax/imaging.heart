"""
Mask Fusion via Majority Voting

Combines binary segmentation masks from multiple models into a single
consensus mask using configurable voting thresholds.

Usage:
    python fuse_masks.py --input_folders folder1 folder2 folder3 folder4 \
                         --output_folder results/fused_masks \
                         --threshold 0.5

Each input folder should contain .nrrd masks with matching filenames across
folders. The script validates that corresponding masks share the same
dimensions, spacing, and orientation before fusing.
"""

import argparse
import os
import sys
import numpy as np
import nrrd
from pathlib import Path
from collections import defaultdict


def validate_headers(headers, filename):
    """
    Validate that all headers for a given filename have matching geometry.

    Checks dimensions, space directions, space origin, and coordinate space.

    Args:
        headers: list of (folder_name, header) tuples
        filename: the mask filename being validated

    Raises:
        ValueError: if any geometric property differs across headers
    """
    ref_folder, ref_header = headers[0]

    for folder, header in headers[1:]:
        # Check dimensions via sizes
        ref_sizes = tuple(ref_header.get('sizes', []))
        cur_sizes = tuple(header.get('sizes', []))
        if ref_sizes != cur_sizes:
            raise ValueError(
                f"{filename}: dimension mismatch — "
                f"{ref_folder} has sizes {ref_sizes}, "
                f"{folder} has sizes {cur_sizes}"
            )

        # Check space directions (voxel spacing and orientation)
        if 'space directions' in ref_header and 'space directions' in header:
            ref_dirs = np.array(ref_header['space directions'])
            cur_dirs = np.array(header['space directions'])
            if not np.allclose(np.abs(ref_dirs), np.abs(cur_dirs), atol=1e-6):
                raise ValueError(
                    f"{filename}: space directions mismatch — "
                    f"{ref_folder} has {ref_dirs.tolist()}, "
                    f"{folder} has {cur_dirs.tolist()}"
                )

        # Check space origin
        if 'space origin' in ref_header and 'space origin' in header:
            ref_origin = np.array(ref_header['space origin'])
            cur_origin = np.array(header['space origin'])
            if not np.allclose(np.abs(ref_origin), np.abs(cur_origin), atol=1e-6):
                raise ValueError(
                    f"{filename}: space origin mismatch — "
                    f"{ref_folder} has {ref_origin.tolist()}, "
                    f"{folder} has {cur_origin.tolist()}"
                )

        # Check coordinate space
        ref_space = ref_header.get('space', '')
        cur_space = header.get('space', '')
        if ref_space != cur_space:
            raise ValueError(
                f"{filename}: coordinate space mismatch — "
                f"{ref_folder} has '{ref_space}', "
                f"{folder} has '{cur_space}'"
            )


def fuse_masks(input_folders, output_folder, threshold=0.5):
    """
    Fuse binary masks from multiple input folders using majority voting.

    For each .nrrd file found across the input folders, loads corresponding
    masks, validates geometric consistency, computes a vote map (each model
    contributes 1/N), and thresholds to produce a consensus binary mask.

    Args:
        input_folders: list of paths to folders containing .nrrd masks
        output_folder: path to write fused masks
        threshold: voting threshold (0.0-1.0). A voxel is included if its
                   vote fraction >= threshold. With 4 models:
                     0.25 = at least 1/4 agree (union)
                     0.50 = at least 2/4 agree (majority)
                     0.75 = at least 3/4 agree (strong consensus)
                     1.00 = all 4/4 agree (intersection)

    Returns:
        dict: summary statistics per fused mask
    """
    input_paths = [Path(f) for f in input_folders]
    output_path = Path(output_folder)
    os.makedirs(output_path, exist_ok=True)

    num_models = len(input_paths)
    vote_weight = 1.0 / num_models

    print("=" * 70)
    print("MASK FUSION VIA MAJORITY VOTING")
    print("=" * 70)
    print(f"Input folders ({num_models}):")
    for p in input_paths:
        print(f"  - {p}")
    print(f"Output folder: {output_path}")
    print(f"Threshold: {threshold} (need >= {threshold * num_models:.1f}/{num_models} models to agree)")
    print("=" * 70)

    # Collect all .nrrd filenames across folders
    folder_files = {}
    for folder in input_paths:
        if not folder.is_dir():
            print(f"WARNING: {folder} is not a directory, skipping")
            continue
        files = {f.name for f in folder.glob('*.nrrd')}
        folder_files[str(folder)] = files

    if not folder_files:
        print("No valid input folders found.")
        return {}

    # Find filenames present in all folders
    all_filenames = set()
    for files in folder_files.values():
        all_filenames |= files

    results = {}

    for filename in sorted(all_filenames):
        # Find which folders have this file
        available_folders = [f for f, files in folder_files.items() if filename in files]

        if len(available_folders) < 2:
            print(f"\n  SKIP {filename}: only found in {len(available_folders)} folder(s)")
            continue

        print(f"\n  Fusing {filename} ({len(available_folders)}/{num_models} models)...")

        # Load masks and headers
        masks = []
        headers = []
        for folder in available_folders:
            filepath = Path(folder) / filename
            data, header = nrrd.read(str(filepath))
            masks.append(data)
            headers.append((Path(folder).name, header))

        # Validate geometric consistency
        try:
            validate_headers(headers, filename)
        except ValueError as e:
            print(f"    ERROR: {e}")
            continue

        # Compute vote map
        actual_models = len(masks)
        actual_weight = 1.0 / actual_models
        vote_map = np.zeros_like(masks[0], dtype=np.float32)
        for mask in masks:
            vote_map += (mask > 0).astype(np.float32) * actual_weight

        # Apply threshold
        fused = (vote_map >= threshold).astype(masks[0].dtype)

        # Use the header from the first available folder
        ref_header = headers[0][1]

        # Write output
        out_path = output_path / filename
        nrrd.write(str(out_path), fused, ref_header)

        # Statistics
        voxels_fused = int(np.sum(fused > 0))
        voxels_per_model = [int(np.sum(m > 0)) for m in masks]
        mean_model_voxels = np.mean(voxels_per_model)

        results[filename] = {
            'num_models': actual_models,
            'voxels_fused': voxels_fused,
            'mean_model_voxels': float(mean_model_voxels),
            'ratio': voxels_fused / mean_model_voxels if mean_model_voxels > 0 else 0.0,
        }

        print(f"    Models: {actual_models}, "
              f"Fused voxels: {voxels_fused}, "
              f"Mean model voxels: {mean_model_voxels:.0f}, "
              f"Ratio: {results[filename]['ratio']:.2f}")

    print(f"\n{'=' * 70}")
    print(f"Fused {len(results)} masks -> {output_path}/")
    print(f"{'=' * 70}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Fuse segmentation masks from multiple models via majority voting."
    )
    parser.add_argument(
        '--input_folders', nargs='+', required=True,
        help="Paths to folders containing .nrrd masks (one folder per model)"
    )
    parser.add_argument(
        '--output_folder', required=True,
        help="Path to write fused output masks"
    )
    parser.add_argument(
        '--threshold', type=float, default=0.5,
        help="Voting threshold (0.0-1.0). Default: 0.5 (majority vote)"
    )

    args = parser.parse_args()

    if args.threshold < 0.0 or args.threshold > 1.0:
        print(f"ERROR: threshold must be between 0.0 and 1.0, got {args.threshold}")
        sys.exit(1)

    fuse_masks(args.input_folders, args.output_folder, args.threshold)


if __name__ == '__main__':
    main()
