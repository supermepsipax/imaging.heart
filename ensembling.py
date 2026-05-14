"""
Mask Fusion for Multi-Model Segmentation Ensembles

Combines binary segmentation masks from multiple models into a single
consensus mask. Supports three methods:

  simple  — majority voting with a configurable threshold.
  staple  — SimpleITK's STAPLE (Simultaneous Truth and Performance Level
            Estimation) probability map, thresholded.
  cldice  — iterative clDice-weighted voting: models are re-weighted each
            iteration by how well they cover / stay connected to the
            consensus skeleton. Designed for tubular structures.

Usage:
    python ensembling.py --input_folder path/to/models_parent \\
                         --output_folder results/fused_masks \\
                         --method simple --threshold 0.5

    python ensembling.py --input_folder ... --output_folder ... \\
                         --method staple --threshold 0.5

    python ensembling.py --input_folder ... --output_folder ... \\
                         --method cldice --threshold 0.5 --max-iter 5

The input folder is expected to contain one subdirectory per model, e.g.:

    input_folder/
        model_1/   case_a.nrrd  case_b.nrrd  ...
        model_2/   case_a.nrrd  case_b.nrrd  ...
        model_3/   case_a.nrrd  case_b.nrrd  ...

Each subdirectory should contain .nrrd masks with matching filenames across
subdirectories. The script validates that corresponding masks share the same
dimensions, spacing, and orientation before fusing.
"""

import argparse
import os
import sys
import numpy as np
import nrrd
from pathlib import Path

from scipy import ndimage

from segmentation.comparison import cl_score
from utilities.centerline_utils import extract_centerline_skimage


# 26-connectivity structuring element, matching utilities/centerline_utils.py
_CONN_3D = np.ones((3, 3, 3), dtype=int)


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
        ref_sizes = tuple(ref_header.get('sizes', []))
        cur_sizes = tuple(header.get('sizes', []))
        if ref_sizes != cur_sizes:
            raise ValueError(
                f"{filename}: dimension mismatch — "
                f"{ref_folder} has sizes {ref_sizes}, "
                f"{folder} has sizes {cur_sizes}"
            )

        if 'space directions' in ref_header and 'space directions' in header:
            ref_dirs = np.array(ref_header['space directions'])
            cur_dirs = np.array(header['space directions'])
            if not np.allclose(np.abs(ref_dirs), np.abs(cur_dirs), atol=1e-6):
                raise ValueError(
                    f"{filename}: space directions mismatch — "
                    f"{ref_folder} has {ref_dirs.tolist()}, "
                    f"{folder} has {cur_dirs.tolist()}"
                )

        if 'space origin' in ref_header and 'space origin' in header:
            ref_origin = np.array(ref_header['space origin'])
            cur_origin = np.array(header['space origin'])
            if not np.allclose(np.abs(ref_origin), np.abs(cur_origin), atol=1e-6):
                raise ValueError(
                    f"{filename}: space origin mismatch — "
                    f"{ref_folder} has {ref_origin.tolist()}, "
                    f"{folder} has {cur_origin.tolist()}"
                )

        ref_space = ref_header.get('space', '')
        cur_space = header.get('space', '')
        if ref_space != cur_space:
            raise ValueError(
                f"{filename}: coordinate space mismatch — "
                f"{ref_folder} has '{ref_space}', "
                f"{folder} has '{cur_space}'"
            )


def discover_model_subfolders(input_folder):
    """Return the sorted list of model subdirectories inside `input_folder`.

    Hidden directories (dot-prefixed) are skipped. Raises if the parent isn't
    a directory or contains no subdirectories.
    """
    parent = Path(input_folder)
    if not parent.is_dir():
        raise ValueError(f"--input_folder {parent} is not a directory")
    subfolders = sorted(
        p for p in parent.iterdir()
        if p.is_dir() and not p.name.startswith('.') and p.name != 'ensembled'
    )
    if not subfolders:
        raise ValueError(f"No model subdirectories found inside {parent}")
    return [str(p) for p in subfolders]


def discover_masks(input_folders):
    """Walk input folders and return {folder_path: set_of_filenames}."""
    input_paths = [Path(f) for f in input_folders]
    folder_files = {}
    for folder in input_paths:
        if not folder.is_dir():
            print(f"WARNING: {folder} is not a directory, skipping")
            continue
        files = {f.name for f in folder.glob('*.nrrd')}
        folder_files[str(folder)] = files
    return folder_files


def load_and_validate(folders, filename):
    """Read nrrd masks from each folder, validate geometric consistency.

    Returns (masks, ref_header). Raises ValueError on mismatch.
    """
    masks = []
    headers = []
    for folder in folders:
        filepath = Path(folder) / filename
        data, header = nrrd.read(str(filepath))
        masks.append(data)
        headers.append((Path(folder).name, header))
    validate_headers(headers, filename)
    return masks, headers[0][1]


def fuse_simple(masks, threshold):
    """Majority-vote fusion. Each model contributes an equal 1/N weight."""
    n = len(masks)
    vote_map = np.zeros_like(masks[0], dtype=np.float32)
    for m in masks:
        vote_map += (m > 0).astype(np.float32) / n
    return (vote_map >= threshold).astype(masks[0].dtype)


def fuse_staple(masks, threshold, model_names=None, foreground_value=1):
    """STAPLE fusion via SimpleITK.

    STAPLE produces a per-voxel probability map from multiple rater masks;
    we threshold it to get the consensus. The operation is voxel-wise so
    we don't need to set physical spacing on the sitk images.

    Args:
        masks: list of numpy mask arrays, one per model.
        threshold: probability cutoff for the final binary mask.
        model_names: optional list of names aligned with `masks` (e.g. the
            model subdirectory names). Used to label per-rater stats in the
            returned `info` dict so the sensitivity/specificity values can
            be traced back to specific models unambiguously.
        foreground_value: label value treated as foreground by STAPLE.

    Returns:
        (fused_array, info) where info contains a 'per_model' list of
        {name, sensitivity, specificity} dicts in input order, plus the
        number of EM iterations the filter ran.
    """
    try:
        import SimpleITK as sitk
    except ImportError as e:
        raise ImportError(
            "SimpleITK is required for --method staple. "
            "Install it with: uv add SimpleITK"
        ) from e

    if model_names is None:
        model_names = [f"model_{i}" for i in range(len(masks))]
    if len(model_names) != len(masks):
        raise ValueError(
            f"model_names length {len(model_names)} does not match "
            f"masks length {len(masks)}"
        )

    images = [sitk.GetImageFromArray((m > 0).astype(np.uint16)) for m in masks]
    staple_filter = sitk.STAPLEImageFilter()
    staple_filter.SetForegroundValue(foreground_value)
    prob_map = staple_filter.Execute(images)
    prob_array = sitk.GetArrayFromImage(prob_map)
    fused = (prob_array >= threshold).astype(masks[0].dtype)

    sensitivity = [float(x) for x in staple_filter.GetSensitivity()]
    specificity = [float(x) for x in staple_filter.GetSpecificity()]
    per_model = [
        {'name': name, 'sensitivity': s, 'specificity': q}
        for name, s, q in zip(model_names, sensitivity, specificity)
    ]

    info = {
        'per_model': per_model,
        'elapsed_iterations': int(staple_filter.GetElapsedIterations()),
    }
    return fused, info


def _connected_fraction(mask_bool, skeleton_bool):
    """Fraction of `mask_bool` voxels inside connected components that touch
    `skeleton_bool`.

    Models contributing components disjoint from the skeleton (stray blobs)
    will score low here; models whose voxels all hang off the consensus tree
    will score close to 1.
    """
    total = int(mask_bool.sum())
    if total == 0:
        return 0.0
    labeled, n_components = ndimage.label(mask_bool, structure=_CONN_3D)
    if n_components == 0:
        return 0.0
    touching_ids = np.unique(labeled[skeleton_bool & (labeled > 0)])
    touching_ids = touching_ids[touching_ids > 0]
    if touching_ids.size == 0:
        return 0.0
    connected = int(np.isin(labeled, touching_ids).sum())
    return connected / total


def fuse_cldice(masks, threshold, max_iter=5, tol=1e-3, verbose=True):
    """Iterative clDice-weighted fusion.

    Starts from equal weights, then each iteration:
      1. Compute weighted vote map and threshold to consensus.
      2. Skeletonize the consensus.
      3. Score each model by (skeleton coverage) × (connected fraction).
         cl_score(model, skel) rewards models whose voxels lie along
         the consensus centerline; _connected_fraction penalises
         disconnected blobs.
      4. Renormalize scores into new weights.
    Stops when weights stop moving (by `tol`) or after `max_iter`.
    """
    n = len(masks)
    mask_bins = [m > 0 for m in masks]
    weights = np.full(n, 1.0 / n, dtype=np.float64)

    consensus = None
    for iteration in range(max_iter):
        vote_map = np.zeros_like(masks[0], dtype=np.float32)
        for w, mb in zip(weights, mask_bins):
            vote_map += w * mb.astype(np.float32)

        consensus = vote_map >= threshold
        if consensus.sum() == 0:
            if verbose:
                print(f"    iter {iteration + 1}: empty consensus, stopping")
            break

        skeleton = extract_centerline_skimage(consensus) > 0

        scores = np.zeros(n, dtype=np.float64)
        for i, mb in enumerate(mask_bins):
            sens = cl_score(mb, skeleton)
            conn = _connected_fraction(mb, skeleton)
            scores[i] = sens * conn

        if scores.sum() == 0:
            if verbose:
                print(f"    iter {iteration + 1}: all scores zero, stopping")
            break
        new_weights = scores / scores.sum()

        weight_change = float(np.abs(new_weights - weights).max())
        if verbose:
            print(
                f"    iter {iteration + 1}: weights = "
                f"{np.round(new_weights, 3).tolist()}, "
                f"Δmax = {weight_change:.4f}"
            )
        weights = new_weights

        if weight_change < tol:
            break

    if consensus is None:
        consensus = np.zeros_like(masks[0], dtype=bool)
    return consensus.astype(masks[0].dtype)


def _write_staple_ranks(results, output_path):
    """Write a text summary of STAPLE per-model sensitivity / specificity.

    Aggregates by model *name* (not index) so a model that's missing from
    some cases still aggregates correctly. Ranks by mean sensitivity.
    """
    from collections import defaultdict

    sens_by_model = defaultdict(list)
    spec_by_model = defaultdict(list)

    lines = ["STAPLE per-rater performance", "=" * 60, ""]
    lines.append("Per-case sensitivity / specificity (J = sens + spec - 1):")
    lines.append("")

    had_any = False
    for filename in sorted(results.keys()):
        info = results[filename].get('method_info') or {}
        per_model = info.get('per_model') or []
        if not per_model:
            continue
        had_any = True
        iters = info.get('elapsed_iterations', '?')
        lines.append(f"  {filename}  (EM iterations: {iters})")
        for entry in per_model:
            name = entry['name']
            s = entry['sensitivity']
            q = entry['specificity']
            j = s + q - 1
            lines.append(f"    {name:<30s}  sens={s:.4f}  spec={q:.4f}  J={j:.4f}")
            sens_by_model[name].append(s)
            spec_by_model[name].append(q)
        lines.append("")

    if not had_any:
        return

    means = []
    for name in sens_by_model:
        mean_s = float(np.mean(sens_by_model[name]))
        mean_q = float(np.mean(spec_by_model[name]))
        means.append((name, mean_s, mean_q, mean_s + mean_q - 1, len(sens_by_model[name])))

    lines.append("=" * 60)
    lines.append("Mean across cases (per model):")
    lines.append("")
    for name, s, q, j, n in sorted(means, key=lambda x: x[0]):
        lines.append(f"  {name:<30s}  sens={s:.4f}  spec={q:.4f}  J={j:.4f}  (n={n})")
    lines.append("")

    lines.append("Ranking by mean sensitivity (descending):")
    lines.append("")
    for rank, (name, s, q, j, n) in enumerate(sorted(means, key=lambda x: -x[1]), 1):
        lines.append(f"  {rank}. {name:<30s}  sens={s:.4f}  spec={q:.4f}  J={j:.4f}")
    lines.append("")

    lines.append("Ranking by mean Youden's J (descending):")
    lines.append("")
    for rank, (name, s, q, j, n) in enumerate(sorted(means, key=lambda x: -x[3]), 1):
        lines.append(f"  {rank}. {name:<30s}  J={j:.4f}  (sens={s:.4f}, spec={q:.4f})")

    out_file = output_path / 'staple_ranks.txt'
    out_file.write_text('\n'.join(lines) + '\n')
    print(f"\nWrote STAPLE ranking: {out_file}")


def run_ensembling(
    input_folders,
    output_folder,
    method='simple',
    threshold=0.5,
    max_iter=5,
    tol=1e-3,
):
    """
    Fuse binary masks from multiple input folders using the given method.

    Args:
        input_folders: list of paths to folders containing .nrrd masks
        output_folder: path to write fused masks
        method: 'simple', 'staple', or 'cldice'
        threshold: voting / probability threshold in [0, 1]
        max_iter: (cldice) maximum refinement iterations
        tol: (cldice) max-weight-change convergence threshold

    Returns:
        dict: summary statistics per fused mask
    """
    output_path = Path(output_folder)
    os.makedirs(output_path, exist_ok=True)

    folder_files = discover_masks(input_folders)

    print("=" * 70)
    print(f"MASK FUSION — method: {method.upper()}")
    print("=" * 70)
    print(f"Input folders ({len(folder_files)}):")
    for p in folder_files:
        print(f"  - {p}")
    print(f"Output folder: {output_path}")
    print(f"Threshold: {threshold}")
    if method == 'cldice':
        print(f"Max iterations: {max_iter}, tolerance: {tol}")
    if method == 'simple':
        n = len(folder_files)
        print(f"(need >= {threshold * n:.1f}/{n} models to agree)")
    print("=" * 70)

    if not folder_files:
        print("No valid input folders found.")
        return {}

    all_filenames = set()
    for files in folder_files.values():
        all_filenames |= files

    results = {}
    num_models = len(folder_files)

    for filename in sorted(all_filenames):
        available_folders = [f for f, files in folder_files.items() if filename in files]

        if len(available_folders) < 2:
            print(f"\n  SKIP {filename}: only found in {len(available_folders)} folder(s)")
            continue

        print(
            f"\n  Fusing {filename} "
            f"({len(available_folders)}/{num_models} models) via {method}..."
        )

        try:
            masks, ref_header = load_and_validate(available_folders, filename)
        except ValueError as e:
            print(f"    ERROR: {e}")
            continue

        model_names = [Path(f).name for f in available_folders]
        method_info = {}

        if method == 'simple':
            fused = fuse_simple(masks, threshold)
        elif method == 'staple':
            fused, method_info = fuse_staple(masks, threshold, model_names=model_names)
        elif method == 'cldice':
            fused = fuse_cldice(masks, threshold, max_iter=max_iter, tol=tol)
        else:
            raise ValueError(f"Unknown method: {method!r}")

        out_path = output_path / filename
        nrrd.write(str(out_path), fused, ref_header)

        voxels_fused = int(np.sum(fused > 0))
        voxels_per_model = [int(np.sum(m > 0)) for m in masks]
        mean_model_voxels = float(np.mean(voxels_per_model))

        results[filename] = {
            'method': method,
            'num_models': len(masks),
            'model_names': model_names,
            'method_info': method_info,
            'voxels_fused': voxels_fused,
            'mean_model_voxels': mean_model_voxels,
            'ratio': voxels_fused / mean_model_voxels if mean_model_voxels > 0 else 0.0,
        }

        print(
            f"    Models: {len(masks)}, "
            f"Fused voxels: {voxels_fused}, "
            f"Mean model voxels: {mean_model_voxels:.0f}, "
            f"Ratio: {results[filename]['ratio']:.2f}"
        )

    print(f"\n{'=' * 70}")
    print(f"Fused {len(results)} masks -> {output_path}/")
    print(f"{'=' * 70}")

    return results


def simple_ensembling(input_folders, output_folder, threshold=0.5):
    """Backwards-compatible wrapper that dispatches to the simple method."""
    return run_ensembling(input_folders, output_folder, method='simple', threshold=threshold)


def main():
    parser = argparse.ArgumentParser(
        description="Fuse segmentation masks from multiple models "
                    "(simple majority vote, STAPLE, or iterative clDice-weighted).",
    )
    parser.add_argument(
        '--input_folder', required=True,
        help="Parent folder containing one subdirectory per model. Each "
             "subdirectory should hold .nrrd masks with matching filenames.",
    )
    parser.add_argument(
        '--output_folder', required=True,
        help="Path to write fused output masks",
    )
    parser.add_argument(
        '--method', choices=['simple', 'staple', 'cldice'], default='simple',
        help="Fusion method (default: simple)",
    )
    parser.add_argument(
        '--threshold', type=float, default=0.5,
        help="Voting / probability threshold in [0, 1]. Default: 0.5",
    )
    parser.add_argument(
        '--max-iter', dest='max_iter', type=int, default=5,
        help="(cldice only) max refinement iterations. Default: 5",
    )
    parser.add_argument(
        '--tol', type=float, default=1e-3,
        help="(cldice only) weight-change convergence tolerance. Default: 1e-3",
    )

    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        print(f"ERROR: threshold must be between 0.0 and 1.0, got {args.threshold}")
        sys.exit(1)

    try:
        model_folders = discover_model_subfolders(args.input_folder)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"Discovered {len(model_folders)} model subdirectories in {args.input_folder}:")
    for p in model_folders:
        print(f"  - {Path(p).name}")

    run_ensembling(
        model_folders,
        args.output_folder,
        method=args.method,
        threshold=args.threshold,
        max_iter=args.max_iter,
        tol=args.tol,
    )


if __name__ == '__main__':
    main()
