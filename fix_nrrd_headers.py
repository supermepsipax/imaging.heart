"""
Corrects `space directions` headers in NRRD files that were saved by the
reconnection pipeline when it still had the axis-reversal bug in sitk_header_to_nrrd.

For each .nrrd in OUTPUT_FOLDER, finds the matching source NIfTI in INPUT_FOLDER,
recomputes the correct space_directions (accounting for SimpleITK's reversed NumPy
axes), and patches the header in-place only when it differs from what is stored.

The voxel data is never modified.

Usage:
    uv run python fix_nrrd_headers.py <input_folder> <output_folder> [--dry-run]
"""
import argparse
import numpy as np
import nrrd
import SimpleITK as sitk
from pathlib import Path


def _correct_space_directions(sitk_img):
    """Compute space_directions matching the NumPy array returned by GetArrayFromImage."""
    spacing = sitk_img.GetSpacing()
    dim = sitk_img.GetDimension()
    dir_matrix = np.array(sitk_img.GetDirection()).reshape(dim, dim)
    # GetArrayFromImage reverses axes: NumPy axis i = ITK axis (dim-1-i)
    return [
        (dir_matrix[:, dim - 1 - i] * spacing[dim - 1 - i]).tolist()
        for i in range(dim)
    ]


def _find_source_nifti(input_folder, stem):
    for ext in ('.nii.gz', '.nii'):
        matches = list(Path(input_folder).glob(f"{stem}{ext}"))
        if matches:
            return matches[0]
    return None


def main():
    parser = argparse.ArgumentParser(description='Fix NRRD space_directions headers')
    parser.add_argument('input_folder', help='Folder containing the source NIfTI files')
    parser.add_argument('output_folder', help='Folder containing the NRRD files to check/fix')
    parser.add_argument('--dry-run', action='store_true',
                        help='Report what would change without writing anything')
    args = parser.parse_args()

    nrrd_files = sorted(Path(args.output_folder).glob('*.nrrd'))
    if not nrrd_files:
        print(f"No .nrrd files found in {args.output_folder}")
        return

    n_ok = n_fixed = n_skipped = 0

    for nrrd_path in nrrd_files:
        source = _find_source_nifti(args.input_folder, nrrd_path.stem)
        if source is None:
            print(f"[SKIP]  {nrrd_path.name} — no matching NIfTI in {args.input_folder}")
            n_skipped += 1
            continue

        img = sitk.ReadImage(str(source))
        correct_dirs = _correct_space_directions(img)

        data, header = nrrd.read(str(nrrd_path))
        current_dirs = header.get('space directions')

        if current_dirs is None:
            print(f"[SKIP]  {nrrd_path.name} — no space directions field in header")
            n_skipped += 1
            continue

        if np.allclose(np.array(current_dirs), np.array(correct_dirs), atol=1e-6):
            print(f"[OK]    {nrrd_path.name}")
            n_ok += 1
            continue

        print(f"[FIX]   {nrrd_path.name}")
        print(f"        was:    {np.array(current_dirs).tolist()}")
        print(f"        fixed:  {correct_dirs}")

        if not args.dry_run:
            header['space directions'] = correct_dirs
            nrrd.write(str(nrrd_path), data, header)

        n_fixed += 1

    action = "would fix" if args.dry_run else "fixed"
    print(f"\nDone.  OK: {n_ok}  |  {action.capitalize()}: {n_fixed}  |  Skipped: {n_skipped}")


if __name__ == '__main__':
    main()
